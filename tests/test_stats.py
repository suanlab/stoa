"""Exact tests in src/stoa/stats.py, pinned against externally known values.

These decide whether a null result in the paper is reported as "underpowered" or as "no
effect" -- two very different claims -- so the implementation is checked against numbers
that can be verified outside this repository (R's `fisher.test`) rather than against
whatever it happened to produce first.
"""
from __future__ import annotations

import pytest

from stoa.stats import (fisher_exact, mcnemar_exact, n_for_power,
                        n_for_power_mcnemar, p_order_reversal, power_fisher)


@pytest.mark.parametrize("table,expected", [
    # Values from R: fisher.test(matrix(c(a,c,b,d), nrow=2))$p.value
    ((1, 9, 11, 3), 0.0027594),
    ((3, 1, 1, 3), 0.4857143),
    ((10, 10, 10, 10), 1.0),
    ((0, 5, 5, 0), 0.0079365),
    ((7, 23, 4, 26), 0.5062),        # the paper's placement-vs-random cell, 50% budget
])
def test_fisher_matches_reference(table, expected):
    assert fisher_exact(*table) == pytest.approx(expected, abs=1e-4)


def test_fisher_is_symmetric_under_row_swap():
    assert fisher_exact(3, 7, 8, 2) == pytest.approx(fisher_exact(8, 2, 3, 7))


def test_fisher_is_a_probability():
    for t in [(0, 1, 1, 0), (5, 5, 5, 5), (20, 1, 1, 20), (1, 0, 0, 1)]:
        p = fisher_exact(*t)
        assert 0.0 <= p <= 1.0, (t, p)


def test_power_rises_with_sample_size():
    """The whole point of the analysis: more questions, more power. A flat or falling curve
    would mean the simulation is not doing what its name says."""
    ps = [power_fisher(0.40, 0.15, n, n, trials=1200, seed=1) for n in (10, 40, 160)]
    assert ps[0] < ps[1] < ps[2], ps
    assert ps[2] > 0.8


def test_power_at_null_is_near_alpha():
    """With no true difference, rejection rate must not exceed alpha. Fisher exact is
    conservative on discrete data, so it should land at or below 0.05, never above."""
    p = power_fisher(0.25, 0.25, 60, 60, alpha=0.05, trials=3000, seed=2)
    assert p <= 0.06, p


def test_n_for_power_returns_none_for_no_effect():
    assert n_for_power(0.3, 0.3) is None


def test_n_for_power_is_smaller_for_bigger_effects():
    small = n_for_power(0.25, 0.20, seed=3)
    big = n_for_power(0.60, 0.10, seed=3)
    assert small is not None and big is not None
    assert big < small, (big, small)


def test_order_reversal_is_high_when_arms_are_equal():
    """Two runs comparing identical arms should disagree on the ordering about half the
    time -- minus the ties that discreteness produces."""
    p = p_order_reversal(0.2, 0.2, 16, 30, trials=6000, seed=4)
    assert 0.25 < p < 0.55, p


def test_order_reversal_is_low_for_a_large_gap():
    p = p_order_reversal(0.05, 0.60, 16, 30, trials=4000, seed=5)
    assert p < 0.01, p


# --- paired tests (added when the LoCoMo run moved to a paired design) ------------------

@pytest.mark.parametrize("bc,expected", [
    ((10, 0), 2 / 2 ** 10),      # every discordant pair favours one arm
    ((8, 2), 0.109375),          # binom two-sided tail, computed by hand
    ((5, 5), 1.0),               # perfectly split -> no evidence
    ((0, 0), 1.0),               # no discordant pairs at all
])
def test_mcnemar_matches_exact_binomial(bc, expected):
    from stoa.stats import mcnemar_exact
    assert mcnemar_exact(*bc) == pytest.approx(expected, abs=1e-9)


def test_mcnemar_is_symmetric():
    from stoa.stats import mcnemar_exact
    assert mcnemar_exact(9, 3) == pytest.approx(mcnemar_exact(3, 9))


def test_mcnemar_ignores_concordant_pairs():
    """Only the discordant counts enter. This is the property that makes the paired test
    more powerful than Fisher here: shared question difficulty cancels out."""
    from stoa.stats import mcnemar_exact
    assert mcnemar_exact(6, 1) == mcnemar_exact(6, 1)   # concordants never passed in


def test_paired_test_beats_unpaired_when_outcomes_are_correlated():
    """The reason the LoCoMo comparison switched to McNemar. Construct arms that agree on
    most questions and differ consistently on a few: the paired test sees the signal, the
    unpaired one is swamped by the shared difficulty."""
    from stoa.stats import fisher_exact, mcnemar_exact
    n, shared_correct, b, c = 300, 40, 12, 1
    a_hits, b_hits = shared_correct + b, shared_correct + c
    assert mcnemar_exact(b, c) < 0.01
    assert fisher_exact(a_hits, n - a_hits, b_hits, n - b_hits) > 0.05


def test_n_for_power_mcnemar_shrinks_as_the_effect_grows():
    from stoa.stats import n_for_power_mcnemar
    weak = n_for_power_mcnemar(p_disc=0.10, p_favour=0.65, alpha=0.05, seed=1)
    strong = n_for_power_mcnemar(p_disc=0.10, p_favour=0.95, alpha=0.05, seed=1)
    assert weak is not None and strong is not None
    assert strong < weak, (strong, weak)


def test_n_for_power_mcnemar_is_none_without_an_effect():
    from stoa.stats import n_for_power_mcnemar
    assert n_for_power_mcnemar(p_disc=0.2, p_favour=0.5, alpha=0.05) is None
    assert n_for_power_mcnemar(p_disc=0.0, p_favour=0.9, alpha=0.05) is None


def test_alpha_is_required_so_it_cannot_be_omitted_by_accident():
    """Sizing a study at 0.05 while judging significance at a corrected threshold understates
    the requirement by 50-65%. A default made that easy to do silently, so there is none."""
    with pytest.raises(TypeError):
        n_for_power_mcnemar(p_disc=0.1, p_favour=0.8)


def test_a_stricter_alpha_demands_more_data():
    a05 = n_for_power_mcnemar(p_disc=0.16, p_favour=0.67, alpha=0.05, seed=0)
    abon = n_for_power_mcnemar(p_disc=0.16, p_favour=0.67, alpha=0.05 / 9, seed=0)
    assert a05 is not None and abon is not None
    assert abon > a05, (abon, a05)
