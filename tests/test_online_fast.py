"""The O(log n) eviction simulator must match the reference scan exactly.

`simulate_fast` replaces a per-eviction linear scan with lazily-invalidated heaps so the
production traces finish. Since it is what produces the paper's numbers, it is pinned to
the reference implementation rather than trusted.
"""
from stoa.eval.online import simulate, simulate_fast
from stoa.traces import WorkloadConfig, generate_workload


def _wl(seed=0, beta=0.0):
    return generate_workload(WorkloadConfig(n_items=64, horizon=2_000, zipf_s=1.1,
                                            locality_beta=beta, seed=seed))


def test_fast_matches_reference_all_policies():
    for seed in (0, 1):
        for beta in (0.0, 0.5):
            wl = _wl(seed, beta)
            for frac in (0.05, 0.1, 0.2):
                slots = max(1, int(frac * 64))
                for pol in ("belady", "lru", "h2o", "static"):
                    a = simulate(wl, pol, slots)
                    b = simulate_fast(wl, pol, slots)
                    assert (a.hits, a.misses) == (b.hits, b.misses), f"{pol} f={frac} s={seed}"
                    assert abs(a.total_latency_ms - b.total_latency_ms) < 1e-6


def test_h2o_tie_break_is_specified():
    """LFU ties must break by least-recently-used in BOTH implementations; leaving the
    rule implicit made two correct implementations disagree by ~0.5% of hits."""
    wl = _wl(2, 0.5)
    a = simulate(wl, "h2o", 12)
    b = simulate_fast(wl, "h2o", 12)
    assert (a.hits, a.misses) == (b.hits, b.misses)
