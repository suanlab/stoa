"""M9-12: budget-conditioned control frontier (RQ3 controllability)."""
from stoa.eval.budget import budget_conditioned_placement, frontier, static_point
from stoa.traces import WorkloadConfig, generate_workload


def _wl(seed=0):
    return generate_workload(WorkloadConfig(n_items=64, horizon=8_000, zipf_s=1.1, seed=seed))


def _t0(wl):
    return sum(wl.n_accesses(i.item_id) for i in wl.items) * 800


def test_controller_meets_any_budget():
    wl = _wl()
    t0 = _t0(wl)
    for frac in (1.0, 0.5, 0.1, 0.0):
        _, pt = budget_conditioned_placement(wl, int(frac * t0))
        assert pt.feasible and pt.tokens_used <= int(frac * t0)


def test_frontier_is_monotone():
    """Tighter budget => fewer tokens used, more dollars, more promotions."""
    wl = _wl()
    t0 = _t0(wl)
    pts = frontier(wl, [int(f * t0) for f in (1.0, 0.5, 0.25, 0.1, 0.0)])
    for a, b in zip(pts, pts[1:]):
        assert b.tokens_used <= a.tokens_used
        assert b.dollar_cost >= a.dollar_cost - 1e-9
        assert b.promotions >= a.promotions


def test_static_plaintext_violates_tight_budget_but_controller_adapts():
    """RQ3: a single static policy can't cover the budget range; the knob does."""
    wl = _wl()
    t0 = _t0(wl)
    tight = int(0.1 * t0)
    assert static_point(wl, 0).tokens_used > tight        # all-plaintext violates
    _, pt = budget_conditioned_placement(wl, tight)
    assert pt.feasible                                     # controller meets it


def test_loose_budget_spends_nothing():
    """At a loose budget the controller keeps everything plaintext (free)."""
    wl = _wl()
    _, pt = budget_conditioned_placement(wl, _t0(wl))
    assert pt.dollar_cost == 0.0 and pt.promotions == 0


def test_frontier_reproducible():
    b = [int(f * _t0(_wl())) for f in (0.5, 0.1)]
    p1 = [(x.tokens_used, x.dollar_cost) for x in frontier(_wl(), b)]
    p2 = [(x.tokens_used, x.dollar_cost) for x in frontier(_wl(), b)]
    assert p1 == p2
