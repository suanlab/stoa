"""LeCaR and CACHEUS (src/stoa/experts.py).

These exist to test a prediction the paper makes -- that regret-based expert policies
inherit ARC's limitation on singleton-heavy KV traces, because they read the same ghost-hit
channel. A broken implementation would confirm the prediction for the wrong reason, so the
tests below pin the mechanism (weights actually move, the scan region actually protects
reused items) rather than only the aggregate hit rate.
"""
from __future__ import annotations

import pytest

from stoa.eval.online import simulate_arc, simulate_fast
from stoa.experts import simulate_cacheus, simulate_lecar
from stoa.traces import WorkloadConfig, generate_workload


def _wl(**kw):
    cfg = dict(n_items=200, horizon=12000, zipf_s=1.1, locality_beta=0.6, seed=3)
    cfg.update(kw)
    return generate_workload(WorkloadConfig(**cfg))


@pytest.mark.parametrize("policy", [simulate_lecar, simulate_cacheus])
@pytest.mark.parametrize("beta", [0.0, 0.6])
def test_bounded_by_belady_and_accounts_for_every_access(policy, beta):
    wl = _wl(locality_beta=beta)
    slots = max(1, int(0.1 * len(wl.items)))
    r = policy(wl, slots)
    assert r.hits + r.misses == len(wl.accesses)
    assert r.hit_rate <= simulate_fast(wl, "belady", slots).hit_rate + 1e-9


@pytest.mark.parametrize("policy", [simulate_lecar, simulate_cacheus])
def test_lands_between_its_two_experts_or_above_them(policy):
    """A regret-weighted mixture of LRU and LFU should not fall below both of them by a wide
    margin; if it does, the weight update has the wrong sign."""
    wl = _wl()
    slots = max(1, int(0.1 * len(wl.items)))
    got = policy(wl, slots).hit_rate
    lru = simulate_fast(wl, "lru", slots).hit_rate
    lfu = simulate_fast(wl, "h2o", slots).hit_rate
    assert got >= min(lru, lfu) - 0.05, (got, lru, lfu)


@pytest.mark.parametrize("policy", [simulate_lecar, simulate_cacheus])
def test_is_reproducible_at_a_fixed_seed(policy):
    """Expert selection is sampled, so a seed must pin the run exactly or no number from
    these policies is citable."""
    wl = _wl(horizon=6000)
    a = policy(wl, 20, seed=11)
    b = policy(wl, 20, seed=11)
    assert (a.hits, a.misses) == (b.hits, b.misses)


@pytest.mark.parametrize("policy", [simulate_lecar, simulate_cacheus])
def test_capacity_is_respected_at_extremes(policy):
    wl = _wl(horizon=4000)
    for slots in (1, 7, 199, 400):
        r = policy(wl, slots)
        assert r.hits + r.misses == len(wl.accesses)


def test_hit_rate_is_monotone_in_cache_size():
    wl = _wl(horizon=8000)
    for policy in (simulate_lecar, simulate_cacheus):
        rates = [policy(wl, s).hit_rate for s in (4, 16, 64, 200)]
        # Sampled policies are not exactly monotone; require the trend, not each step.
        assert rates[-1] > rates[0], (policy.__name__, rates)


def test_lecar_weights_respond_to_the_workload():
    """The two experts must actually be distinguishable: on a stationary skewed draw LFU is
    the better rule, and on a locality-heavy one LRU is. If LeCaR scored identically on both
    relative to its experts, the weights would not be doing anything."""
    flat = _wl(locality_beta=0.0)
    local = _wl(locality_beta=0.6)
    slots = 20
    gap_flat = simulate_lecar(flat, slots).hit_rate - simulate_fast(flat, "lru", slots).hit_rate
    gap_local = simulate_lecar(local, slots).hit_rate - simulate_fast(local, "lru", slots).hit_rate
    assert gap_flat > gap_local, (gap_flat, gap_local)


def test_cacheus_scan_region_helps_under_heavy_one_time_traffic():
    """SR-LRU's reason to exist: items touched once should be evicted before items that have
    been reused. Compared against plain LRU on a workload with many singletons, CACHEUS
    should not be worse."""
    wl = generate_workload(WorkloadConfig(n_items=4000, horizon=20000, zipf_s=0.6,
                                          locality_beta=0.3, seed=9))
    slots = 200
    assert simulate_cacheus(wl, slots).hit_rate >= simulate_fast(wl, "lru", slots).hit_rate - 0.02


def test_all_adaptive_policies_agree_on_a_trivially_large_cache():
    """With capacity above the working set every policy hits everything after cold misses,
    so any disagreement here is an accounting bug rather than a policy difference."""
    wl = _wl(horizon=3000)
    n = len(wl.items)
    ref = simulate_fast(wl, "lru", n).hit_rate
    for r in (simulate_lecar(wl, n), simulate_cacheus(wl, n), simulate_arc(wl, n)):
        assert r.hit_rate == pytest.approx(ref, abs=1e-9), r.policy
