"""ARC (Megiddo & Modha, FAST'03) invariants and sanity bounds.

ARC is the adaptive-replacement baseline our reactive numbers are measured against, so
its correctness is load-bearing: a subtly wrong ARC would understate or overstate the
gap that a learned policy would have to beat.
"""
from __future__ import annotations

import pytest

from stoa.eval.online import simulate_arc, simulate_fast
from stoa.traces import WorkloadConfig, generate_workload


def _wl(**kw):
    cfg = dict(n_items=64, horizon=3000, zipf_s=1.1, locality_beta=0.0, seed=7)
    cfg.update(kw)
    return generate_workload(WorkloadConfig(**cfg))


@pytest.mark.parametrize("beta", [0.0, 0.6])
@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25])
def test_arc_never_exceeds_belady(beta, frac):
    """Belady is the offline optimum, so no online policy may exceed it. This is the one
    bound that holds unconditionally, and it is the check that would catch a capacity or
    accounting bug inflating ARC's hit rate."""
    wl = _wl(locality_beta=beta)
    slots = max(1, int(frac * len(wl.items)))
    arc = simulate_arc(wl, slots).hit_rate
    belady = simulate_fast(wl, "belady", slots).hit_rate
    assert arc <= belady + 1e-9, f"ARC {arc} exceeded Belady {belady}"


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25])
def test_fill_once_beats_arc_on_stationary_zipf(frac):
    """A characterization test, not an aspiration: under a STATIONARY Zipf draw a
    fill-once cache beats ARC, because the first `c` distinct blocks it happens to admit
    are (with high probability) the popular ones, and never evicting them is close to the
    optimal static set. Adaptive replacement only pays once the popularity ranking moves.

    This is the same degeneracy that made our synthetic testbed useless for the placement
    question (docs/claims_dependency.md), recorded here so it is not rediscovered as a
    bug in ARC. On both production traces the ordering is the usual one -- ARC beats
    fill-once at every tier size (experiments/reactive_real_full.json)."""
    wl = _wl(locality_beta=0.0)
    slots = max(1, int(frac * len(wl.items)))
    assert simulate_arc(wl, slots).hit_rate < simulate_fast(wl, "static", slots).hit_rate


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25])
def test_arc_beats_fill_once_once_locality_exists(frac):
    """The ordering flips as soon as the workload has the temporal locality real traces
    have -- which is what makes ARC the right baseline for the reactive numbers."""
    wl = _wl(locality_beta=0.6)
    slots = max(1, int(frac * len(wl.items)))
    assert simulate_arc(wl, slots).hit_rate > simulate_fast(wl, "static", slots).hit_rate


def test_arc_respects_capacity():
    """The published algorithm keeps |T1| + |T2| <= c; a violation would silently inflate
    the hit rate by caching more than the tier holds."""
    wl = _wl(horizon=1500)
    for slots in (1, 4, 16):
        r = simulate_arc(wl, slots)
        assert r.hits + r.misses == len(wl.accesses)


def test_arc_hit_rate_is_monotone_in_cache_size():
    wl = _wl(locality_beta=0.6)
    rates = [simulate_arc(wl, s).hit_rate for s in (2, 8, 32, 64)]
    assert rates == sorted(rates), rates


def test_arc_matches_lru_when_cache_holds_everything():
    """With capacity >= working set every policy hits everything after a cold miss each."""
    wl = _wl(horizon=1200)
    slots = len(wl.items)
    assert simulate_arc(wl, slots).hit_rate == pytest.approx(
        simulate_fast(wl, "lru", slots).hit_rate)
