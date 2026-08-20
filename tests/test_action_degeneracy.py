"""Pin the action-space degeneracy the paper now reports, so it cannot silently change."""
import pytest
from stoa.credit import weighted_cost
from stoa.environment import Representation
from stoa.sequential import DEFAULT_WEIGHTS, LEGAL_ACTIONS
from stoa.simulator import TieringSimulator


def _argmin_action(n_hat, sim=None):
    sim = sim or TieringSimulator()
    best, bc = None, float("inf")
    for a in LEGAL_ACTIONS:
        c = weighted_cost(sim.cost_of_placement(a, n_hat, Representation.PLAINTEXT),
                          DEFAULT_WEIGHTS)
        if c < bc:
            best, bc = a, c
    return (best.representation.value, best.tier.value, best.timing.value)


def test_action_choice_saturates_in_the_belief():
    """The paper reports that under our cost tables the per-item argmin is nearly constant in
    the belief: two distinct actions over the whole range, saturating at 0.1. This is a stated
    LIMITATION of the decoupling result, so it must fail loudly if the cost model changes and
    the paper is not updated with it."""
    grid = [0.02 * i for i in range(1, 51)] + [1.5, 2, 5, 10, 50, 100, 1000]
    acts = [_argmin_action(n) for n in grid]
    assert len(set(acts)) == 2, f"action variety changed: {sorted(set(acts))}"
    above = [a for n, a in zip(grid, acts) if n >= 0.1]
    assert len(set(above)) == 1, f"no longer saturated above 0.1: {sorted(set(above))}"


def test_saturation_survives_every_derived_cost_table():
    """Varying tier latency cannot fix it -- the token term dominates. Checked across all
    hardware-derived tables so the paper's claim is not specific to the hand-written one."""
    import stoa.simulator as sim_mod
    from stoa.calibration import DEVICES, MODELS, derive_tier_read_ms
    base = dict(sim_mod._TIER_READ_MS)
    try:
        for m in MODELS.values():
            for d in DEVICES.values():
                sim_mod._TIER_READ_MS.update(derive_tier_read_ms(m, d, 256))
                acts = {_argmin_action(n) for n in (0.2, 1, 10, 100)}
                assert len(acts) == 1, (m.name, d.name, acts)
    finally:
        sim_mod._TIER_READ_MS.update(base)
