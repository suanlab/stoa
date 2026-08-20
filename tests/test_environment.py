"""Smoke tests for the STOA scaffold — confirm the pipeline is wired end-to-end."""
from stoa import (
    Action,
    ItemState,
    Orchestrator,
    Representation,
    Tier,
    Timing,
)
from stoa.credit import AccessTrace, belady_optimal_tier


def test_action_space_is_masked():
    space = Action.space()
    assert len(space) > 0
    # PARAMETER promotion must never be ONLINE.
    assert all(
        not (a.representation is Representation.PARAMETER and a.timing is Timing.ONLINE)
        for a in space
    )
    # LATENT memory only on GPU/CPU.
    assert all(
        not (a.representation is Representation.LATENT and a.tier in (Tier.DISK, Tier.REMOTE_RDMA))
        for a in space
    )


def test_orchestrator_step_returns_metrics():
    items = [ItemState(item_id=f"m{i}", tier=Tier.CPU) for i in range(4)]
    out = Orchestrator().step(items)
    for key in ("latency_ms", "cost_usd", "tokens", "reward", "within_budget"):
        assert key in out
    assert out["tokens"] >= 0


def test_belady_prefers_hot_tier_for_near_reuse():
    near = AccessTrace(item_id="x", access_steps=[5])
    far = AccessTrace(item_id="y", access_steps=[9999])
    assert belady_optimal_tier(near, current_step=0, horizon=1000) is Tier.GPU
    assert belady_optimal_tier(far, current_step=0, horizon=1000) is Tier.REMOTE_RDMA
