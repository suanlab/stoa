"""M6-9: learned Belady-imitation tier controller (RQ2 basis)."""
from stoa.eval.online import run_online
from stoa.learn import N_FEATURES, fit_reuse_predictor
from stoa.traces import WorkloadConfig, generate_workload


def _locality_wl(seed):
    return generate_workload(WorkloadConfig(n_items=64, horizon=8_000, zipf_s=1.1,
                                            locality_beta=0.5, seed=seed))


def test_predictor_fits_finite_weights_reproducibly():
    a = fit_reuse_predictor(_locality_wl(100))
    b = fit_reuse_predictor(_locality_wl(100))
    assert len(a.weights) == N_FEATURES
    assert all(w == w and abs(w) < 1e6 for w in a.weights)   # finite, not NaN
    assert a.weights == b.weights


def test_learned_never_beats_oracle():
    pred = fit_reuse_predictor(_locality_wl(100))
    res = run_online(_locality_wl(0), cache_frac=0.1, predictor=pred)
    assert res["learned"].hit_rate <= res["belady"].hit_rate + 1e-9


def test_learned_beats_both_heuristics_under_locality():
    """The whole point of RQ2: a learned freq+recency controller beats LRU and H2O
    when the workload has both popularity and recency structure."""
    pred = fit_reuse_predictor(_locality_wl(100))
    wins_lru = wins_h2o = 0
    for seed in (0, 1, 2, 3, 4):
        res = run_online(_locality_wl(seed), cache_frac=0.1, predictor=pred)
        wins_lru += res["learned"].hit_rate >= res["lru"].hit_rate
        wins_h2o += res["learned"].hit_rate >= res["h2o"].hit_rate
    assert wins_lru >= 4 and wins_h2o >= 4   # beats each heuristic on >=4/5 held-out traces


def test_run_online_skips_learned_without_predictor():
    res = run_online(_locality_wl(0), cache_frac=0.1)      # no predictor
    assert "learned" not in res and "belady" in res


def test_locality_beta_zero_is_backward_compatible():
    """Default (beta=0) must reproduce the pre-locality trace exactly."""
    a = generate_workload(WorkloadConfig(n_items=32, horizon=500, seed=7))
    b = generate_workload(WorkloadConfig(n_items=32, horizon=500, seed=7, locality_beta=0.0))
    assert a.accesses == b.accesses
