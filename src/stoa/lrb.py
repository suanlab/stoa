"""LRB -- Learning Relaxed Belady (Song et al., NSDI'20, pp. 529-544).

The one learned-cache baseline our reactive numbers are actually measured against. LRB
matters here because it is the strongest counter-argument to this paper's finding: it
learns from a far richer feature set than the four trace-local statistics our
predictability result is built on, and it won on production CDN traces. If a learned
policy is going to beat reaction on KV blocks, this is the design that should do it.

The idea, in the paper's terms: exact Belady is unlearnable because it needs a total order
over next-access times, but the *relaxed* objective -- evict anything whose next access
falls beyond a "Belady boundary" -- is learnable. So LRB predicts time-to-next-request with
a gradient-boosted regressor over each object's access history, samples a small set of
resident objects on each eviction, and drops the one with the largest predicted next
access. It retrains online from labels that the trace itself supplies as it replays.

What is faithful here:
  * GBDT regression on log time-to-next-request (`stoa.gbdt`, flattened for fast scoring)
  * the LRB feature family: the most recent inter-access deltas, exponentially decayed
    counters at several time scales, age and frequency
  * sampled eviction (default 64 candidates), argmax predicted next access
  * online retraining every `train_interval` requests from a sliding memory window
  * right-censored labels: objects not re-requested inside the window are labelled at the
    window edge rather than dropped, which is what stops the model from only ever seeing
    the popular objects

What differs, and why:
  * **Uniform object size.** LRB optimizes byte miss ratio over variable-size CDN objects
    and carries size as a feature. KV blocks in both our sources are fixed-size, so size is
    a constant column and byte miss ratio collapses to object miss ratio. This removes one
    of LRB's advantages over LRU; we state it rather than quietly benefiting from it.
  * **Request-index time.** We measure time in requests, not seconds, so that LRB is billed
    on the same clock as Belady/LRU/LFU/ARC. Both traces do carry wall-clock timestamps.

Anything reported from this module should be called LRB-style, not LRB: it is a
reimplementation of the published mechanism, not the authors' artifact.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .environment import Tier
from .eval.online import HOT, COLD, OnlineResult
from .gbdt import GBDT
from .simulator import TieringSimulator

# Half-lives (in requests) for the exponentially decayed counters. Spread over three orders
# of magnitude so the model can distinguish "hot right now" from "steadily warm".
_EDC_HALFLIVES = (64.0, 256.0, 1024.0, 4096.0)
_N_DELTAS = 8            # most recent inter-access gaps kept per block
_N_FEATURES = _N_DELTAS + len(_EDC_HALFLIVES) + 3


@dataclass
class _Meta:
    """Per-block access history, updated in place as the trace replays."""
    last: int = -1
    first: int = -1
    count: int = 0
    deltas: deque = field(default_factory=lambda: deque(maxlen=_N_DELTAS))
    edc: list = field(default_factory=lambda: [0.0] * len(_EDC_HALFLIVES))


def _features(m: _Meta, now: int) -> list[float]:
    """Feature vector for one block as of `now`, using only its observed past.

    Deltas are padded with the sentinel below rather than with zero: a zero gap means "used
    again immediately", the opposite of "never seen again", and padding with it would tell
    the model that a block with no history is the hottest thing in the cache.
    """
    pad = math.log1p(1e6)
    ds = [math.log1p(d) for d in m.deltas]
    ds = ds + [pad] * (_N_DELTAS - len(ds))
    return [
        math.log1p(now - m.last if m.last >= 0 else 0),      # recency
        math.log1p(now - m.first if m.first >= 0 else 0),    # age
        math.log1p(m.count),                                 # frequency
        *ds,
        *[math.log1p(e) for e in m.edc],
    ]


def _touch(m: _Meta, now: int) -> None:
    if m.last >= 0:
        m.deltas.append(now - m.last)
        for i, hl in enumerate(_EDC_HALFLIVES):
            m.edc[i] = 1.0 + m.edc[i] * (2.0 ** (-(now - m.last) / hl))
    else:
        m.first = now
        m.edc = [1.0] * len(_EDC_HALFLIVES)
    m.last = now
    m.count += 1


def simulate_lrb(
    workload,
    cache_slots: int,
    sim: TieringSimulator | None = None,
    *,
    sample_size: int = 64,
    train_interval: int = 20_000,
    memory_window: int = 50_000,
    n_trees: int = 30,
    max_depth: int = 5,
    max_train_rows: int = 60_000,
    seed: int = 0,
) -> OnlineResult:
    """Replay `workload` through a `cache_slots` hot tier under an LRB-style policy.

    Until the first model is trained the policy falls back to LRU over the sampled
    candidates, so the warm-up period is a well-defined heuristic rather than random
    eviction -- otherwise the first `train_interval` requests would inject noise that has
    nothing to do with what is being measured.
    """
    sim = sim or TieringSimulator()
    hot_lat, cold_lat = sim.read_latency(HOT), sim.read_latency(COLD)
    rng = random.Random(seed)
    # A SEPARATE stream for subsampling the training buffer. Drawing those samples from
    # `rng` would consume draws that drive eviction candidate selection, so the training
    # fix would silently perturb the policy itself and a before/after comparison would
    # measure two changes at once.
    train_rng = random.Random(seed ^ 0x5EED)

    meta: dict[str, _Meta] = defaultdict(_Meta)
    cache: list[str] = []                 # resident blocks, sampled from by index
    resident: dict[str, int] = {}         # block -> index into `cache`
    # Training rows awaiting a label, keyed by a monotone id. `by_item` finds the rows a
    # re-access labels; `order` is the same ids in time order, for censoring the tail.
    # Keying by id rather than by list position is deliberate: positions shift whenever the
    # structure is compacted, and a stale position silently labels the wrong row.
    pend: dict[int, tuple[str, list[float], int]] = {}
    by_item: dict[str, list[int]] = defaultdict(list)
    order: deque = deque()
    next_pid = 0
    train_X: list[list[float]] = []
    train_y: list[float] = []
    train_t: list[int] = []          # when each row entered the buffer, for the sliding window
    model = None
    hits = misses = 0
    total = 0.0
    n_censored = n_labelled = n_fits = 0
    log_window = math.log1p(memory_window)

    def _evict(now: int) -> None:
        """Drop one resident block: the sampled candidate with the farthest predicted use.

        Every sampled candidate also becomes a training row, with its features frozen HERE,
        at sampling time. This is the paper's design and it matters: computing the features
        at the moment of a request instead would give every training row recency zero, while
        at inference time recency is whatever has elapsed since the block was last used. The
        model would then be fitted on a distribution it never sees in deployment -- the same
        reference-frame mismatch this project catalogs elsewhere. An early version of this
        file had exactly that bug and LRB scored 15 points below LRU because of it.
        """
        nonlocal next_pid
        k = min(sample_size, len(cache))
        idxs = rng.sample(range(len(cache)), k)
        feats = [_features(meta[cache[i]], now) for i in idxs]
        if model is None:
            victim_i = max(idxs, key=lambda i: now - meta[cache[i]].last)   # LRU warm-up
        else:
            scores = model.predict(feats)
            victim_i = idxs[int(max(range(k), key=lambda j: scores[j]))]
        for i, f in zip(idxs, feats):
            blk = cache[i]
            pend[next_pid] = (blk, f, now)
            by_item[blk].append(next_pid)
            order.append(next_pid)
            next_pid += 1
        victim = cache[victim_i]
        last = cache.pop()
        if victim_i < len(cache):
            cache[victim_i] = last
            resident[last] = victim_i
        del resident[victim]

    for now, (_step, item) in enumerate(workload.accesses):
        m = meta[item]

        # --- close out any pending training rows this access labels ---
        for pid in by_item.pop(item, ()):
            row = pend.pop(pid, None)
            if row is not None:
                train_X.append(row[1])
                train_y.append(math.log1p(now - row[2]))
                train_t.append(now)
                n_labelled += 1

        if item in resident:
            hits += 1
            total += hot_lat
        else:
            misses += 1
            total += cold_lat
            if len(cache) >= cache_slots:
                _evict(now)
            resident[item] = len(cache)
            cache.append(item)

        _touch(m, now)

        if (now + 1) % train_interval == 0:
            # Harvest censored rows: still unlabelled and older than the memory window.
            # Without these the model would only ever see blocks that WERE re-accessed, and
            # would learn the distribution of reuse gaps rather than whether reuse happens.
            cutoff = now - memory_window
            while order:
                pid = order[0]
                row = pend.get(pid)
                if row is None:                 # already labelled by a re-access
                    order.popleft()
                    continue
                if row[2] >= cutoff:
                    break
                order.popleft()
                del pend[pid]
                train_X.append(row[1])
                train_y.append(log_window)
                train_t.append(now)
                n_censored += 1
            # A SLIDING memory window, then a uniform subsample -- in that order.
            #
            # The published design keeps a sliding window of recent requests and trains from
            # what is inside it. Two things went wrong here in turn. First the buffer was
            # truncated to its TAIL: labelled rows arrive one at a time as blocks are
            # re-accessed, censored rows arrive in one contiguous burst per checkpoint, so
            # the tail held only the newest burst and from the third retrain onward every
            # label was identical -- the regressor returned a constant and eviction
            # degenerated to a uniform draw over the sampled candidates. Replacing that with
            # a uniform sample over ALL accumulated rows fixed the labels but made the buffer
            # unbounded in age, which is not a sliding window either, and moved the reported
            # numbers by up to 19 points relative to a window-bounded reading.
            #
            # Dropping by age first is what "sliding memory window" means; the subsample only
            # caps memory once the window itself is larger than the budget.
            cutoff_t = now - memory_window
            # `min`, not `train_t[0]`: the uniform subsample below reorders the buffer, so
            # after the first time it fires the timestamps are no longer sorted and a
            # first-element guard silently stops firing. That is how this filter came to be
            # a no-op -- the run produced numbers identical to the unfiltered version, which
            # is the only reason it was noticed.
            if train_t and min(train_t) < cutoff_t:
                fresh = [i for i, t in enumerate(train_t) if t >= cutoff_t]
                train_X = [train_X[i] for i in fresh]
                train_y = [train_y[i] for i in fresh]
                train_t = [train_t[i] for i in fresh]
            if len(train_X) > max_train_rows:
                keep = train_rng.sample(range(len(train_X)), max_train_rows)
                train_X = [train_X[i] for i in keep]
                train_y = [train_y[i] for i in keep]
                train_t = [train_t[i] for i in keep]
            if len(train_X) >= 1000:
                # A constant target silently turns this policy into random replacement, so
                # refuse to fit rather than emit a number that looks like a learned result.
                lo_y, hi_y = min(train_y), max(train_y)
                if hi_y - lo_y < 1e-12:
                    raise RuntimeError(
                        f"LRB training labels are constant ({len(train_y)} rows, all "
                        f"{lo_y:.4f}); the model would be a no-op and eviction would be "
                        "uniformly random. Check the memory-window sampling.")
                model = GBDT(n_trees=n_trees, lr=0.1, max_depth=max_depth).fit(
                    train_X, train_y).flatten()
                n_fits += 1

    # Diagnostics. A learned policy that loses is only reportable if we can show it was
    # actually trained; `censored_frac` is the quantity that decides whether it COULD learn,
    # since a right-censored row carries no information about WHEN a block returns.
    r = OnlineResult("lrb", total, hits, misses)
    r.ghost_hits = n_labelled          # reuse the diagnostic slots rather than widen the type
    r.p_moves = n_fits
    r.final_p = n_censored
    return r
