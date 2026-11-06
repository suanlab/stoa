"""S3-FIFO, SIEVE and W-TinyLFU (stoa.modern): each against a hand-computed case, the property it was
designed for (keeping a small hot set through a stream of one-hit wonders, where LRU keeps nothing),
and Belady's bound on random traces."""
import random

from stoa.environment import ItemState
from stoa.eval.online import simulate_fast
from stoa.modern import simulate_s3fifo, simulate_sieve, simulate_wtinylfu
from stoa.traces import AccessTrace, Workload

POLICIES = (simulate_s3fifo, simulate_sieve, simulate_wtinylfu)


def _wl(seq):
    steps = {}
    for t, k in enumerate(seq):
        steps.setdefault(k, []).append(t)
    return Workload(items=[ItemState(item_id=k, size_bytes=1) for k in steps],
                    accesses=[(t, k) for t, k in enumerate(seq)],
                    traces={k: AccessTrace(item_id=k, access_steps=v) for k, v in steps.items()})


def test_sieve_by_hand():
    # cache 3: a b c fill; a hits (visited). d: hand starts at the tail a, clears it, moves to b
    # (unvisited) and evicts it. b misses: the hand is at c (unvisited), evicts c. a still resident.
    r = simulate_sieve(_wl(list("abcadba")), 3)
    assert (r.hits, r.misses) == (2, 5)


def test_s3fifo_by_hand():
    # cache 10 -> S holds 1, M 9. x is read three times while in S (freq 3 > 1), so when the next
    # miss evicts S's tail it moves to M instead of the ghost queue, and the later x hits.
    seq = ["x", "x", "x", "y", "z", "x"]
    r = simulate_s3fifo(_wl(seq), 10)
    assert (r.hits, r.misses) == (3, 3)


def _hot_and_scan(n_rounds=200, hot=5, scan_per_hot=2):
    seq, s = [], 0
    for r in range(n_rounds):
        seq.append(f"h{r % hot}")
        for _ in range(scan_per_hot):
            seq.append(f"s{s}")
            s += 1
    return _wl(seq)


def test_scan_resistance_where_lru_keeps_nothing():
    wl = _hot_and_scan()                       # reuse distance of a hot block: 14 distinct > 10
    assert simulate_fast(wl, "lru", 10).hits == 0
    # S3-FIFO (ghost queue) and W-TinyLFU (counts of non-resident blocks) remember a block they
    # evicted, so its next miss admits it for good.
    for f in (simulate_s3fifo, simulate_wtinylfu):
        assert f(wl, 10).hits > 0.8 * 195, f.__name__
    # SIEVE cannot: a block earns its visited bit only by a hit while resident, and here every hot
    # block is evicted before its first reuse. By design, not a defect.
    assert simulate_sieve(wl, 10).hits == 0


def test_sieve_keeps_a_hot_set_once_it_has_been_hit():
    seq = ["h0", "h1", "h2", "h3"] * 2         # each hot block hit once while resident
    s = 0
    for r in range(200):
        seq.append(f"h{r % 4}")
        seq += [f"s{s + i}" for i in range(3)]
        s += 3
    wl = _wl(seq)
    assert simulate_fast(wl, "lru", 8).hits < 10
    assert simulate_sieve(wl, 8).hits == simulate_fast(wl, "belady", 8).hits


def test_never_above_belady():
    rng = random.Random(1)
    for _ in range(30):
        seq = [f"k{int(rng.paretovariate(1.2)) % 40}" for _ in range(400)]
        wl = _wl(seq)
        for slots in (3, 8, 15):
            b = simulate_fast(wl, "belady", slots).hits
            for f in POLICIES:
                assert f(wl, slots).hits <= b, (f.__name__, slots)
