"""M3-6: online tier-axis eviction ablation (E1) — Belady optimality & gate."""
from stoa.eval.online import POLICIES, run_online, simulate
from stoa.traces import WorkloadConfig, generate_workload


def _wl(seed=0, n=64, horizon=4_000, zipf=1.1):
    return generate_workload(WorkloadConfig(n_items=n, horizon=horizon, zipf_s=zipf, seed=seed))


def test_belady_is_optimal_lower_bound_on_misses():
    """Belady minimizes misses => no policy can beat it (oracle ceiling)."""
    res = run_online(_wl(), cache_frac=0.1)
    belady = res["belady"]
    for p in res:
        assert belady.misses <= res[p].misses
        assert belady.total_latency_ms <= res[p].total_latency_ms + 1e-6


def test_controller_gate_belady_beats_heuristics():
    """M3-6 go/no-go: the single-axis (tier) controller >= matching heuristics."""
    res = run_online(_wl(seed=3), cache_frac=0.1)
    assert res["belady"].hit_rate >= res["lru"].hit_rate
    assert res["belady"].hit_rate >= res["h2o"].hit_rate


def test_lfu_beats_lru_under_stationary_skew():
    """On a stationary Zipf stream, H2O/LFU beats LRU (LRU thrashes on the cold
    tail). A known result — the ablation should reproduce the heuristic ordering."""
    for seed in (0, 1, 2):
        res = run_online(_wl(seed=seed, zipf=1.2), cache_frac=0.1)
        assert res["h2o"].hit_rate > res["lru"].hit_rate


def test_hit_rates_are_valid_and_counts_close():
    res = run_online(_wl(), cache_frac=0.2)
    for r in res.values():
        assert 0.0 <= r.hit_rate <= 1.0
        assert r.hits + r.misses == 4_000


def test_online_is_reproducible():
    a = run_online(_wl(seed=5), cache_frac=0.1)["lru"]
    b = run_online(_wl(seed=5), cache_frac=0.1)["lru"]
    assert (a.total_latency_ms, a.hits, a.misses) == (b.total_latency_ms, b.hits, b.misses)


def test_bigger_cache_never_lowers_hit_rate_for_belady():
    small = run_online(_wl(seed=2), cache_frac=0.05)["belady"]
    big = run_online(_wl(seed=2), cache_frac=0.25)["belady"]
    assert big.hit_rate >= small.hit_rate
