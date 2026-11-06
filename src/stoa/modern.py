"""Modern single-tier eviction and admission baselines: S3-FIFO, SIEVE, W-TinyLFU (notebook §BB).

An independent review (2026-10-25, E2) noted that the reactive comparison stops at ARC, LeCaR, CACHEUS
and LRB, while the policies built for workloads dominated by one-hit wonders -- which is what KV traces
are -- are newer. They are reimplemented here from their papers, on the same access sequence and with
the same hit-rate definition as `stoa.eval.online` (every access counts; a block's first access is a
compulsory miss). Uniform block sizes, so capacity is in blocks.

  S3-FIFO   Yang et al., SOSP'23, Algorithm 1 as printed: small FIFO S (10% of the cache), main FIFO M
            with reinsertion, ghost FIFO G holding as many keys as M; frequency capped at 3; an object
            leaves S for M only if its frequency is > 1.
  SIEVE     Zhang et al., NSDI'24: one FIFO queue, a visited bit, and a hand that moves from the tail
            toward the head, clearing visited bits and evicting the first unvisited object; new objects
            enter at the head and the hand keeps its position.
  W-TinyLFU Einziger, Friedman & Manes, ACM TOS 2017 (the Caffeine design): a window LRU (1%), a main
            segmented LRU (probation 20% / protected 80% of the main region), and TinyLFU admission --
            a block evicted from the window enters main only if its estimated frequency exceeds that of
            main's victim. Frequencies here are EXACT counts halved every 10 x cache-size accesses
            (the aging the paper specifies); the paper's count-min sketch and doorkeeper approximate
            exactly this, so this is the idealized TinyLFU, not a sketch-error-faithful one.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict

from .eval.online import COLD, HOT, OnlineResult
from .simulator import TieringSimulator
from .traces import Workload


def _result(name, hits, misses, sim):
    sim = sim or TieringSimulator()
    lat = hits * sim.read_latency(HOT) + misses * sim.read_latency(COLD)
    return OnlineResult(name, lat, hits, misses)


def simulate_s3fifo(workload: Workload, cache_slots: int, small_frac: float = 0.1,
                    sim: TieringSimulator | None = None) -> OnlineResult:
    c = max(1, cache_slots)
    s_cap = max(1, int(small_frac * c))
    m_cap = max(1, c - s_cap)
    S: OrderedDict[str, None] = OrderedDict()      # head = last inserted (end), tail = first
    M: OrderedDict[str, None] = OrderedDict()
    G: OrderedDict[str, None] = OrderedDict()
    freq: dict[str, int] = {}
    hits = misses = 0

    def ghost_add(k):
        G[k] = None
        while len(G) > m_cap:
            G.popitem(last=False)

    def evict_m():
        while M:
            t = next(iter(M))
            if freq[t] > 0:
                M.move_to_end(t)                   # reinsert at head
                freq[t] -= 1
            else:
                del M[t]
                del freq[t]
                return

    def evict_s():
        while S:
            t = next(iter(S))
            del S[t]
            if freq[t] > 1:
                M[t] = None
                if len(M) > m_cap:
                    evict_m()
            else:
                ghost_add(t)
                del freq[t]
                return

    for _step, x in workload.accesses:
        if x in S or x in M:
            hits += 1
            freq[x] = min(freq[x] + 1, 3)
            continue
        misses += 1
        while len(S) + len(M) >= c:
            if len(S) >= s_cap:
                evict_s()
            else:
                evict_m()
        if x in G:
            del G[x]
            M[x] = None
        else:
            S[x] = None
        freq[x] = 0
    return _result("s3fifo", hits, misses, sim)


def simulate_sieve(workload: Workload, cache_slots: int,
                   sim: TieringSimulator | None = None) -> OnlineResult:
    c = max(1, cache_slots)
    # Doubly linked list by dicts; head = newest. `nxt` points toward the head, `prv` toward the tail.
    nxt: dict[str, str | None] = {}
    prv: dict[str, str | None] = {}
    visited: dict[str, bool] = {}
    head = tail = hand = None
    hits = misses = 0

    def unlink(k):
        nonlocal head, tail
        p, n = prv[k], nxt[k]
        if p is not None:
            nxt[p] = n
        else:
            tail = n
        if n is not None:
            prv[n] = p
        else:
            head = p
        del nxt[k], prv[k], visited[k]

    for _step, x in workload.accesses:
        if x in visited:
            hits += 1
            visited[x] = True
            continue
        misses += 1
        if len(visited) >= c:
            o = hand if hand is not None else tail
            while visited[o]:
                visited[o] = False
                o = nxt[o] if nxt[o] is not None else tail      # toward the head, wrap to tail
            hand = nxt[o]                                       # None = restart from the tail
            unlink(o)
        prv[x], nxt[x] = head, None
        if head is not None:
            nxt[head] = x
        head = x
        if tail is None:
            tail = x
        visited[x] = False
    return _result("sieve", hits, misses, sim)


def simulate_wtinylfu(workload: Workload, cache_slots: int, window_frac: float = 0.01,
                      protected_frac: float = 0.8, sample_mult: int = 10,
                      sim: TieringSimulator | None = None) -> OnlineResult:
    c = max(2, cache_slots)
    w_cap = max(1, int(round(window_frac * c)))
    main_cap = c - w_cap
    p_cap = max(1, int(protected_frac * main_cap))
    window: OrderedDict[str, None] = OrderedDict()      # LRU first
    prob: OrderedDict[str, None] = OrderedDict()
    prot: OrderedDict[str, None] = OrderedDict()
    cnt: dict[str, int] = defaultdict(int)
    sample, seen = sample_mult * c, 0
    hits = misses = 0

    def admit_to_main(cand):
        if len(prob) + len(prot) < main_cap:
            prob[cand] = None
            return
        victim = next(iter(prob)) if prob else next(iter(prot))
        if cnt[cand] > cnt[victim]:
            (prob if victim in prob else prot).pop(victim)
            prob[cand] = None

    for _step, x in workload.accesses:
        cnt[x] += 1
        seen += 1
        if seen >= sample:                               # aging: halve every count
            for k in list(cnt):
                cnt[k] //= 2
                if cnt[k] == 0:
                    del cnt[k]
            seen = 0
        if x in window:
            hits += 1
            window.move_to_end(x)
            continue
        if x in prot:
            hits += 1
            prot.move_to_end(x)
            continue
        if x in prob:
            hits += 1
            del prob[x]
            prot[x] = None
            if len(prot) > p_cap:
                d, _ = prot.popitem(last=False)
                prob[d] = None
            continue
        misses += 1
        window[x] = None
        if len(window) > w_cap:
            cand, _ = window.popitem(last=False)
            admit_to_main(cand)
    return _result("wtinylfu", hits, misses, sim)
