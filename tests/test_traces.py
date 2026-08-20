"""M0-3: synthetic trace generation — reproducibility (the M0-3 go/no-go gate)."""
from stoa.traces import WorkloadConfig, generate_workload


def test_workload_is_reproducible_by_seed():
    a = generate_workload(WorkloadConfig(n_items=32, horizon=500, seed=7))
    b = generate_workload(WorkloadConfig(n_items=32, horizon=500, seed=7))
    assert a.accesses == b.accesses
    assert {i: a.n_accesses(i) for i in a.traces} == {i: b.n_accesses(i) for i in b.traces}


def test_different_seeds_differ():
    a = generate_workload(WorkloadConfig(seed=1))
    b = generate_workload(WorkloadConfig(seed=2))
    assert a.accesses != b.accesses


def test_access_counts_sum_to_horizon_and_are_skewed():
    wl = generate_workload(WorkloadConfig(n_items=50, horizon=2_000, zipf_s=1.2, seed=0))
    assert sum(wl.n_accesses(i) for i in wl.traces) == 2_000
    counts = sorted((wl.n_accesses(i) for i in wl.traces), reverse=True)
    # Zipf skew: the hottest item is accessed far more than the median item.
    assert counts[0] > counts[len(counts) // 2]


def test_access_steps_are_within_horizon_and_ordered():
    wl = generate_workload(WorkloadConfig(n_items=16, horizon=300, seed=3))
    for tr in wl.traces.values():
        assert tr.access_steps == sorted(tr.access_steps)
        assert all(0 <= s < 300 for s in tr.access_steps)
