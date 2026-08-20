"""Expert-based adaptive replacement: LeCaR and CACHEUS.

These complete the adaptive-replacement family the paper positions against. They matter
here for one reason: \\Cref{sec:reaction} predicts they will inherit ARC's limitation,
because they learn from the same evidence channel. ARC shifts its recency/frequency split
on a *ghost hit* -- a miss on something it recently evicted -- and on these traces only
3.5--7.3% of misses carry that signal. LeCaR and CACHEUS replace ARC's arithmetic update
with regret-based multiplicative weights, but the regret is still observed only on ghost
hits. If the prediction is right they should land near ARC rather than near Belady.

Running them is how that prediction gets tested rather than asserted.

**LeCaR** (Vietri et al., HotStorage'18) keeps two experts, LRU and LFU, and samples which
one picks the victim from a weight vector. When an evicted item is later requested again,
the expert that chose it is charged a regret discounted by how long ago the eviction
happened, and its weight is multiplied down.

**CACHEUS** (Rodriguez et al., FAST'21) keeps LeCaR's regret machinery and changes two
things: the learning rate adapts to the observed hit rate instead of being fixed, and the
two experts become SR-LRU (scan-resistant) and CR-LFU (churn-resistant). We implement the
adaptive learning rate and SR-LRU; CR-LFU is approximated by LFU with an LRU tie-break,
which is noted below because it is a real simplification.

Both are reimplementations from the papers, not the authors' artifacts. Time is measured in
request index, as for every other policy here, so all of them are billed on one clock.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict, defaultdict

from .eval.online import COLD, HOT, OnlineResult
from .simulator import TieringSimulator


class _VictimHeaps:
    """Lazily-invalidated min-heaps over the resident set, one per expert.

    A direct `min()` over the cache is O(cache) per eviction, which does not finish on
    traces with tens of thousands of slots and hundreds of thousands of evictions -- the
    reason this class exists rather than the two-line version. Entries are never deleted;
    a popped entry is accepted only if the item is still resident AND its stored key still
    matches its current key, so stale entries are skipped instead of removed.

    `scan` is the SR-LRU region R (items not yet reused). CACHEUS drains it first; LeCaR
    ignores it.
    """

    def __init__(self) -> None:
        self.lru: list = []
        self.lfu: list = []
        self.scan: list = []
        self.seq = 0

    def push(self, item: str, last: int, freq: int, is_scan: bool) -> None:
        heapq.heappush(self.lru, (last, self.seq, item))
        heapq.heappush(self.lfu, (freq, last, self.seq, item))
        if is_scan:
            heapq.heappush(self.scan, (last, self.seq, item))
        self.seq += 1

    def pop_lru(self, cache: dict[str, int]) -> str | None:
        while self.lru:
            last, _s, item = heapq.heappop(self.lru)
            if cache.get(item) == last:
                return item
        return None

    def pop_lfu(self, cache: dict[str, int], freq: dict[str, int]) -> str | None:
        while self.lfu:
            f, last, _s, item = heapq.heappop(self.lfu)
            if cache.get(item) == last and freq[item] == f:
                return item
        return None

    def pop_scan(self, cache: dict[str, int], reused: set) -> str | None:
        while self.scan:
            last, _s, item = heapq.heappop(self.scan)
            if cache.get(item) == last and item not in reused:
                return item
        return None


def simulate_lecar(
    workload,
    cache_slots: int,
    sim: TieringSimulator | None = None,
    *,
    learning_rate: float = 0.45,
    discount: float | None = None,
    history_mult: float = 1.0,
    seed: int = 0,
) -> OnlineResult:
    """LeCaR: regret-minimizing selection between an LRU and an LFU expert.

    `discount` defaults to the paper's $d = 0.005^{1/N}$ with $N$ the cache size, which makes
    the regret for an eviction decay to near zero over roughly one cache-full of requests.
    """
    sim = sim or TieringSimulator()
    hot_lat, cold_lat = sim.read_latency(HOT), sim.read_latency(COLD)
    n = max(1, cache_slots)
    d = discount if discount is not None else 0.005 ** (1.0 / n)
    rng = random.Random(seed)

    w = [0.5, 0.5]                                   # [LRU expert, LFU expert]
    heaps = _VictimHeaps()
    cache: dict[str, int] = {}                       # item -> last use
    freq: dict[str, int] = defaultdict(int)
    hist: OrderedDict[str, tuple[int, int]] = OrderedDict()   # evicted -> (time, expert)
    hist_cap = max(1, int(history_mult * n))
    hits = misses = 0
    total = 0.0

    for t, (_step, item) in enumerate(workload.accesses):
        freq[item] += 1
        if item in cache:
            hits += 1
            total += hot_lat
            cache[item] = t
            heaps.push(item, t, freq[item], False)
            continue

        misses += 1
        total += cold_lat
        # Regret: this item was evicted by one of the experts and has now been requested.
        # Charge that expert, discounted by how stale the decision is, and renormalize.
        prev = hist.pop(item, None)
        if prev is not None:
            t0, which = prev
            r = d ** (t - t0)
            w[which] *= math.exp(-learning_rate * r)
            s = w[0] + w[1]
            w = [w[0] / s, w[1] / s]

        if len(cache) >= n:
            which = 0 if rng.random() < w[0] else 1
            victim = (heaps.pop_lru(cache) if which == 0
                      else heaps.pop_lfu(cache, freq))
            if victim is None:                       # both heaps exhausted of live entries
                victim = min(cache, key=cache.get)
            del cache[victim]
            hist[victim] = (t, which)
            if len(hist) > hist_cap:
                hist.popitem(last=False)
        cache[item] = t
        heaps.push(item, t, freq[item], False)

    return OnlineResult("lecar", total, hits, misses)


def simulate_cacheus(
    workload,
    cache_slots: int,
    sim: TieringSimulator | None = None,
    *,
    seed: int = 0,
) -> OnlineResult:
    """CACHEUS: LeCaR's regret machinery with an adaptive learning rate and SR-LRU.

    SR-LRU splits the resident set into a scan region R (blocks seen once since admission)
    and a reused region S, with an adaptive target size for R. On a miss the block enters R;
    on a hit it moves to the MRU end of S. Eviction takes R's LRU when R is at or above its
    target, and otherwise gives S's LRU a second chance by demoting it to the MRU end of R
    before evicting R's LRU. The target itself moves like ARC's split: a history hit on
    something evicted from R says R was drained too eagerly and grows it; a history hit from
    S shrinks it.

    The demotion and the adaptive target are both load-bearing. An earlier version here
    simply always evicted from R, which lets S grow without bound until the policy is a
    fill-once cache -- it scored six points BELOW plain LRU on a singleton-heavy workload,
    which would have confirmed this paper's prediction about expert policies for entirely
    the wrong reason.

    One documented deviation: CR-LFU is approximated by LFU with an LRU tie-break, since
    CR-LFU's churn resistance comes from a demotion rule we do not implement.

    Both this and `simulate_lecar` are reimplementations from the papers, not the authors'
    artifacts.
    """
    sim = sim or TieringSimulator()
    hot_lat, cold_lat = sim.read_latency(HOT), sim.read_latency(COLD)
    n = max(1, cache_slots)
    d = 0.005 ** (1.0 / n)
    rng = random.Random(seed)

    lr = 0.45
    lr_prev, hr_prev = 0.45, -1.0
    w = [0.5, 0.5]                                  # [SR-LRU expert, CR-LFU expert]
    R: OrderedDict[str, None] = OrderedDict()       # scan region: seen once since admission
    S: OrderedDict[str, None] = OrderedDict()       # reused region
    freq: dict[str, int] = defaultdict(int)
    last: dict[str, int] = {}
    lfu_heap: list = []
    seq = 0
    target_r = n // 2
    hist: OrderedDict[str, tuple[int, int, str]] = OrderedDict()   # -> (time, expert, region)
    hits = misses = 0
    total = 0.0
    win_hits = win_n = 0

    def resident(k: str) -> bool:
        return k in R or k in S

    def drop(k: str) -> str:
        R.pop(k, None)
        S.pop(k, None)
        return k

    def evict_lfu() -> str | None:
        while lfu_heap:
            f, lu, _s, item = heapq.heappop(lfu_heap)
            if resident(item) and freq[item] == f and last.get(item) == lu:
                return drop(item)
        return None

    def evict_srlru() -> str:
        if len(R) >= max(1, target_r) and R:
            return drop(next(iter(R)))
        if S:                                        # second chance: demote S's LRU into R
            k = next(iter(S))
            del S[k]
            R[k] = None
        return drop(next(iter(R))) if R else drop(next(iter(S)))

    for t, (_step, item) in enumerate(workload.accesses):
        freq[item] += 1
        win_n += 1
        if resident(item):
            hits += 1
            win_hits += 1
            total += hot_lat
            R.pop(item, None)
            S.pop(item, None)
            S[item] = None                           # promote / refresh in the reused region
        else:
            misses += 1
            total += cold_lat
            prev = hist.pop(item, None)
            if prev is not None:
                t0, which, region = prev
                r = d ** (t - t0)
                w[which] *= math.exp(-lr * r)
                s_ = w[0] + w[1]
                w = [w[0] / s_, w[1] / s_]
                # ARC-style: a returning block we evicted from R means R was too small.
                target_r = (min(n - 1, target_r + 1) if region == "R"
                            else max(1, target_r - 1))

            if len(R) + len(S) >= n:
                which = 0 if rng.random() < w[0] else 1
                if which == 0:
                    victim, region = evict_srlru(), "R"
                else:
                    v = evict_lfu()
                    victim, region = (v, "S") if v is not None else (evict_srlru(), "R")
                hist[victim] = (t, which, region)
                if len(hist) > n:
                    hist.popitem(last=False)
            R[item] = None                           # admissions land in the scan region

        last[item] = t
        heapq.heappush(lfu_heap, (freq[item], t, seq, item))
        seq += 1

        # --- adaptive learning rate, evaluated once per cache-size window ---
        if win_n >= n:
            hr = win_hits / win_n
            if hr_prev >= 0.0:
                delta = abs(lr - lr_prev) or 0.01
                if hr > hr_prev:                     # keep moving the same direction
                    lr_new = lr + delta if lr >= lr_prev else lr - delta
                else:                                # reverse
                    lr_new = lr - delta if lr >= lr_prev else lr + delta
                lr_prev, lr = lr, min(1.0, max(0.001, lr_new))
            hr_prev = hr
            win_hits = win_n = 0

    return OnlineResult("cacheus", total, hits, misses)
