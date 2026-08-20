"""Guards for the four metric defects the second review round found.

Each of these moved a reported quantity by more than the effect the paper was measuring, and
each was invisible in every aggregate we looked at. They are pinned here because the paper's
numbers are only meaningful while these properties hold.
"""
from __future__ import annotations

import pytest

from stoa.mooncake import load_mooncake
from stoa.sequential import (attainable_ceiling, place_with_beliefs, prefix_greedy_cost,
                             quantile_match)
from stoa.traces import WorkloadConfig, generate_workload


def _wl(**kw):
    cfg = dict(n_items=300, horizon=4000, zipf_s=1.1, locality_beta=0.4, seed=5)
    cfg.update(kw)
    return generate_workload(WorkloadConfig(**cfg))


def test_an_infinitesimal_belief_shift_cannot_move_the_cost():
    """Adding 1e-9 to every belief changes no ordering and no real decision. It used to move
    the frequency baseline by 36%, because ties in the per-item argmin were broken by the
    Tier enum's declaration order and so parked every zero-belief block on DISK while the
    faster remote tier sat empty."""
    wl = _wl()
    st = max(1, wl.time_split(0.5))
    stats = wl.prefix_stats(st)
    b0 = {i.item_id: float(stats[i.item_id]["count"]) for i in wl.items}
    b1 = {k: v + 1e-9 for k, v in b0.items()}
    assert place_with_beliefs(wl, b0) == pytest.approx(place_with_beliefs(wl, b1), rel=1e-9)


def test_ties_never_become_a_ranking():
    """A constant score vector must map to a constant belief. With a plain stable sort it
    became strictly increasing in input order, which gave a zero-information control a
    systematic ranking -- and the control then beat a real model."""
    out = quantile_match([0.5] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert len(set(out)) == 1
    assert out[0] == pytest.approx(3.5)


def test_quantile_match_preserves_order_and_the_magnitude_multiset():
    scores = [0.4, 0.1, 0.9, 0.2]
    ref = [10.0, 40.0, 20.0, 30.0]
    out = quantile_match(scores, ref)
    assert sorted(out) == sorted(ref)
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] < scores[j]:
                assert out[i] < out[j]


def test_quantile_match_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        quantile_match([1.0, 2.0], [1.0])


def test_the_attainable_ceiling_is_below_the_hindsight_oracle():
    """The oracle ranks blocks that do not exist yet at the split; a causal policy cannot.
    The ceiling must therefore be a strictly weaker reference whenever any block is invisible."""
    wl = _wl()
    st = max(1, wl.time_split(0.5))
    stats = wl.prefix_stats(st)
    visible = [i.item_id for i in wl.items if stats[i.item_id]["count"] > 0]
    assert len(visible) < len(wl.items), "test needs some blocks to be invisible at the split"
    fut = wl.future_counts(st)
    c_orac = place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
    c_ceil = attainable_ceiling(wl, visible)
    c_pref = prefix_greedy_cost(wl)
    assert c_ceil >= c_orac, "the ceiling cannot beat full hindsight"
    assert c_ceil <= c_pref, "the ceiling must at least match the no-learning baseline"


def test_the_ceiling_equals_the_oracle_when_everything_is_visible():
    wl = _wl()
    all_ids = [i.item_id for i in wl.items]
    st = max(1, wl.time_split(0.5))
    fut = wl.future_counts(st)
    c_orac = place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
    assert attainable_ceiling(wl, all_ids) == pytest.approx(c_orac)


def test_on_a_real_trace_most_of_the_hindsight_gap_is_unreachable():
    """The finding that retracted C2. Recorded as a test so the paper cannot quietly go back
    to normalizing by the full gap."""
    path = "data/mooncake_toolagent_trace.jsonl"
    pytest.importorskip("json")
    import os
    if not os.path.exists(path):
        pytest.skip("Mooncake trace not present")
    wl = load_mooncake(path, max_requests=1500)
    st = max(1, wl.time_split(0.5))
    stats = wl.prefix_stats(st)
    visible = [i.item_id for i in wl.items if stats[i.item_id]["count"] > 0]
    fut = wl.future_counts(st)
    c_pref = prefix_greedy_cost(wl)
    c_orac = place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
    c_ceil = attainable_ceiling(wl, visible)
    reachable = (c_pref - c_ceil) / (c_pref - c_orac)
    assert reachable < 0.5, f"reachable share is {reachable:.1%}; the paper's framing assumes it is small"


def test_the_two_baseline_routines_agree():
    """`prefix_greedy_cost` and `place_with_beliefs(prefix counts)` compute the same thing by
    two different code paths, and every reported fraction divides by one of them. They must
    agree. They did not: fixing the tie-break in one and not the other left them disagreeing
    by a wide margin, and two of our own scripts silently used different denominators."""
    wl = _wl()
    st = max(1, wl.time_split(0.5))
    stats = wl.prefix_stats(st)
    a = prefix_greedy_cost(wl)
    b = place_with_beliefs(wl, {i.item_id: float(stats[i.item_id]["count"]) for i in wl.items})
    assert a == pytest.approx(b, rel=1e-6), f"baselines disagree: {a} vs {b}"


def test_all_three_placement_routines_use_the_same_tie_rule():
    """`place_with_beliefs`, `prefix_greedy_cost` and `_oracle_cost` each choose among
    equal-cost actions. A block with no future accesses makes every action cost the same, so
    the choice is pure tie-breaking -- and the numerator and denominator of every reported
    fraction come from different routines. If they disagree, the fraction is meaningless.

    The check: a clairvoyant belief run through `place_with_beliefs` must equal the oracle
    that `reference_costs` computes by the other path."""
    from stoa.sequential import reference_costs
    wl = _wl()
    st = max(1, wl.time_split(0.5))
    fut = wl.future_counts(st)
    via_beliefs = place_with_beliefs(
        wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
    via_reference = reference_costs(wl)["oracle_future"]
    assert via_beliefs == pytest.approx(via_reference, rel=1e-6), (
        f"oracle disagrees across routines: {via_beliefs} vs {via_reference}")
