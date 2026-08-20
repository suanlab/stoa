"""Online tier-axis eviction ablation (docs/research_plan.md §11.3 E1; M3-6).

The tier axis is a classic caching problem: a scarce hot tier (GPU) over a cold
backing store, filled *online* as the access stream arrives. Belady's rule (evict
the item whose next use is farthest) is the offline OPTIMUM and the oracle ceiling
(PARROT, arXiv:2006.16239); LRU and H2O/LFU (arXiv:2306.14048) are the online
heuristics a learned tier controller must beat. NOTE: the policy named `h2o` here is
plain LFU with an LRU tie-break, NOT H2O (arXiv:2306.14048), which evicts tokens within
one sequence's KV cache by accumulated attention score -- a signal a block-hash trace
cannot express. The alias is kept for artifact compatibility; papers must say LFU.
On a static placement these
collapse into one another (total access count is a sufficient statistic), so the
controller-vs-heuristic question is only meaningful online, where past != future.

This harness replays a synthetic trace (traces.py) under each policy and reports
serving latency + hit rate. It isolates the TIER axis: representation is fixed to
plaintext, so per-access token cost is invariant and the only signal is tier read
latency. The representation / joint axes are covered by the static oracle in
eval/oracle.py — together they give the per-axis (축별) ablation for RQ1.

The `belady` policy doubles as the M3-6 single-axis controller's warm-start target
(it replays hindsight-optimal evictions). TODO(M6-9): replace the oracle next-use
lookup with a learned predictor over ItemState features (recency, access_freq,
predicted_next_access) so the controller works without future knowledge.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from ..environment import Tier
from ..learn import ReusePredictor, features
from ..simulator import TieringSimulator
from ..traces import Workload

HOT: Tier = Tier.GPU
COLD: Tier = Tier.REMOTE_RDMA

# Policies expressed as a VICTIM RULE over a flat cache set -- `simulate`/`simulate_fast`
# both implement them by choosing which resident item to drop. ARC is deliberately not in
# this tuple: it is not a victim rule but a four-list structure (T1/T2 + ghosts B1/B2) with
# its own state, so it has a dedicated entry point (`simulate_arc`) and appears only in
# ALL_POLICIES. Adding it here silently breaks every caller that dispatches on POLICIES.
POLICIES = ("belady", "learned", "lru", "h2o", "static")
ALL_POLICIES = POLICIES + ("arc",)


@dataclass
class OnlineResult:
    policy: str
    total_latency_ms: float
    hits: int
    misses: int
    # ARC-only diagnostics. ARC adapts its T1/T2 target `p` only on a GHOST HIT (a miss on
    # a block it recently evicted), so `ghost_hits / misses` is the rate at which it gets
    # any adaptation signal at all. On KV traces, where most blocks are never reused, this
    # is the quantity that decides whether adaptation can happen.
    ghost_hits: int = 0
    final_p: int = 0
    p_moves: int = 0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            **({"ghost_hits": self.ghost_hits, "ghost_hit_rate_of_misses":
                round(self.ghost_hits / self.misses, 4) if self.misses else 0.0,
                "final_p": self.final_p, "p_moves": self.p_moves}
               if self.policy == "arc" else {}),
        }


def _global_indices(workload: Workload) -> dict[str, list[int]]:
    """Per-item list of global access positions (for Belady next-use lookup)."""
    idxs: dict[str, list[int]] = defaultdict(list)
    for gidx, (_step, item) in enumerate(workload.accesses):
        idxs[item].append(gidx)
    return idxs


def _victim(
    policy: str,
    cache: set[str],
    now: int,
    last_used: dict[str, int],
    first_seen: dict[str, int],
    freq: dict[str, int],
    item_idxs: dict[str, list[int]],
    ptr: dict[str, int],
    predictor: ReusePredictor | None,
) -> str:
    if policy == "lru":
        return min(cache, key=lambda c: last_used.get(c, -1))
    if policy == "h2o":
        # Fewest hits so far; ties broken by least-recently-used. The tie rule is stated
        # explicitly because plain `min` over a set would break ties by iteration order,
        # leaving the policy under-specified and two correct implementations disagreeing.
        return min(cache, key=lambda c: (freq[c], last_used.get(c, -1)))
    if policy == "belady":
        def next_use(c: str) -> float:
            p = ptr[c]
            lst = item_idxs[c]
            return lst[p] if p < len(lst) else float("inf")
        return max(cache, key=next_use)                   # farthest true next use
    if policy == "learned":
        assert predictor is not None, "learned policy needs a fitted predictor"
        return max(cache, key=lambda c: predictor.score(
            features(now, last_used.get(c), freq[c], first_seen.get(c))))  # farthest predicted use
    raise ValueError(f"no eviction for policy {policy!r}")


def simulate_arc(workload: Workload, cache_slots: int,
                 sim: TieringSimulator | None = None) -> OnlineResult:
    """ARC (Megiddo & Modha, FAST'03) -- the adaptive-replacement baseline.

    Our two sources disagree about which pure rule wins: recency dominates on the agentic
    traces, frequency below the working-set knee on the serving traces. ARC exists exactly
    because neither dominates in general -- it keeps a recency list (T1) and a frequency
    list (T2) plus ghost lists (B1, B2) of what each recently evicted, and shifts the
    target split `p` toward whichever ghost list is being hit. Reporting reactive numbers
    against LRU and LFU alone therefore understates what a deployed cache would do, which
    is why this is here.

    Implements the published algorithm directly: cache contents are T1 + T2; B1 and B2 hold
    keys only. Complexity is O(1) amortized per access.
    """
    sim = sim or TieringSimulator()
    hot_lat, cold_lat = sim.read_latency(HOT), sim.read_latency(COLD)
    c = max(1, cache_slots)
    t1: OrderedDict[str, None] = OrderedDict()   # recency: seen once
    t2: OrderedDict[str, None] = OrderedDict()   # frequency: seen twice or more
    b1: OrderedDict[str, None] = OrderedDict()   # ghosts evicted from T1
    b2: OrderedDict[str, None] = OrderedDict()   # ghosts evicted from T2
    p = 0
    hits = misses = 0
    ghost_hits = 0          # the ONLY signal ARC adapts on
    p_trace: list[int] = []
    total = 0.0

    def replace(key_in_b2: bool) -> None:
        if t1 and ((key_in_b2 and len(t1) == p) or len(t1) > p):
            k, _ = t1.popitem(last=False)         # LRU of T1 -> ghost B1
            b1[k] = None
        elif t2:
            k, _ = t2.popitem(last=False)         # LRU of T2 -> ghost B2
            b2[k] = None

    for _step, x in workload.accesses:
        if x in t1:                                # hit: promote to frequency list
            hits += 1
            total += hot_lat
            del t1[x]
            t2[x] = None
            continue
        if x in t2:
            hits += 1
            total += hot_lat
            t2.move_to_end(x)
            continue

        misses += 1
        total += cold_lat
        if x in b1:                                # recency ghost -> favour T1
            ghost_hits += 1
            p = min(c, p + max(len(b2) // max(len(b1), 1), 1))
            replace(False)
            del b1[x]
            t2[x] = None
        elif x in b2:                              # frequency ghost -> favour T2
            ghost_hits += 1
            p = max(0, p - max(len(b1) // max(len(b2), 1), 1))
            replace(True)
            del b2[x]
            t2[x] = None
        else:                                      # genuine miss
            if len(t1) + len(b1) == c:
                if len(t1) < c:
                    b1.popitem(last=False)
                    replace(False)
                else:
                    t1.popitem(last=False)
            elif len(t1) + len(b1) < c and len(t1) + len(t2) + len(b1) + len(b2) >= c:
                if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * c and b2:
                    b2.popitem(last=False)
                replace(False)
            t1[x] = None
        p_trace.append(p)
    return OnlineResult("arc", total, hits, misses,
                       ghost_hits=ghost_hits, final_p=p,
                       p_moves=sum(1 for i in range(1, len(p_trace))
                                   if p_trace[i] != p_trace[i - 1]))


def simulate_fast(
    workload: Workload,
    policy: str,
    cache_slots: int,
    sim: TieringSimulator | None = None,
) -> OnlineResult:
    """Same semantics as `simulate` for belady/lru/h2o/static, in O(log n) per eviction.

    The reference implementation scans the whole cache to pick a victim, which is O(cache)
    per eviction and does not finish on production traces (millions of accesses over
    thousands of slots). Here each policy keeps a lazily-invalidated heap:

      * belady: max-heap on next use  -> push (-next_use, seq, item) whenever an item's
        next use changes; on eviction pop until the popped key matches the item's current
        next use and the item is still resident.
      * lru:    min-heap on last use.
      * h2o:    min-heap on frequency.

    Stale entries are skipped rather than deleted, so the heap can hold several entries
    per item; correctness comes from re-checking the current key at pop time.
    `tests/test_online_fast.py` asserts this returns exactly what `simulate` returns.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    sim = sim or TieringSimulator()
    hot_lat, cold_lat = sim.read_latency(HOT), sim.read_latency(COLD)
    item_idxs = _global_indices(workload)
    ptr = {i: 0 for i in item_idxs}
    horizon = len(workload.accesses)

    def next_use(item: str) -> float:
        p, lst = ptr[item], item_idxs[item]
        return lst[p] if p < len(lst) else float("inf")

    cache: set[str] = set()
    freq: dict[str, int] = defaultdict(int)
    last_used: dict[str, int] = {}
    heap: list[tuple] = []
    seq = 0
    hits = misses = 0
    total = 0.0

    def key_of(item: str):
        if policy == "belady":
            return (-next_use(item), 0)            # max-heap via negation
        if policy == "lru":
            return (last_used.get(item, -1), 0)
        return (float(freq[item]), last_used.get(item, -1))   # h2o/LFU, LRU tie-break

    for gidx, (_step, item) in enumerate(workload.accesses):
        ptr[item] += 1
        freq[item] += 1
        if item in cache:
            hits += 1
            total += hot_lat
        else:
            misses += 1
            total += cold_lat
            if policy == "static" and len(cache) >= cache_slots:
                last_used[item] = gidx
                continue
            if len(cache) >= cache_slots:
                while heap:                        # lazy deletion
                    k, _s, cand = heapq.heappop(heap)
                    if cand in cache and k == key_of(cand):
                        cache.discard(cand)
                        break
            cache.add(item)
        last_used[item] = gidx
        if policy != "static":
            heapq.heappush(heap, (key_of(item), seq, item))
            seq += 1
    _ = horizon
    return OnlineResult(policy, total, hits, misses)


def simulate(
    workload: Workload,
    policy: str,
    cache_slots: int,
    sim: TieringSimulator | None = None,
    predictor: ReusePredictor | None = None,
) -> OnlineResult:
    """Replay the access stream through a `cache_slots`-slot hot tier under `policy`."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    sim = sim or TieringSimulator()
    hot_lat = sim.read_latency(HOT)
    cold_lat = sim.read_latency(COLD)
    item_idxs = _global_indices(workload)
    ptr = {i: 0 for i in item_idxs}                       # next-future-occurrence pointer

    cache: set[str] = set()
    last_used: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    freq: dict[str, int] = defaultdict(int)
    hits = misses = 0
    total = 0.0

    for gidx, (_step, item) in enumerate(workload.accesses):
        ptr[item] += 1                                    # advance past the current occurrence
        freq[item] += 1
        if item in cache:
            hits += 1
            total += hot_lat
        else:
            misses += 1
            total += cold_lat                             # served from cold this time
            if not (policy == "static" and len(cache) >= cache_slots):
                if len(cache) >= cache_slots:
                    cache.discard(_victim(policy, cache, gidx, last_used, first_seen,
                                          freq, item_idxs, ptr, predictor))
                cache.add(item)
        first_seen.setdefault(item, gidx)
        last_used[item] = gidx

    return OnlineResult(policy, total, hits, misses)


def run_online(
    workload: Workload,
    cache_frac: float = 0.1,
    sim: TieringSimulator | None = None,
    predictor: ReusePredictor | None = None,
) -> dict[str, OnlineResult]:
    """Run every policy on one workload; Belady is the oracle lower bound on misses.

    The `learned` policy needs a fitted `predictor` (learn.fit_reuse_predictor);
    if none is given it is skipped so the harness still runs oracle + heuristics.
    """
    cache_slots = max(1, int(cache_frac * len(workload.items)))
    policies = POLICIES if predictor is not None else tuple(p for p in POLICIES if p != "learned")
    out = {p: simulate(workload, p, cache_slots, sim, predictor) for p in policies}
    out["arc"] = simulate_arc(workload, cache_slots, sim)
    return out
