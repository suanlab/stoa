"""Credit assignment for memory-tier placement (docs/stoa_design.md §3).

The payoff of a placement arrives many steps later (at query time), and memory
edits fracture the trajectory prefix. STOA combines three ingredients:

  1. Belady-style oracle imitation (PARROT, arXiv:2006.16239) — hindsight-optimal
     placement from full future access traces, used to warm-start the policy.
  2. Process supervision (RAG-Gym, arXiv:2502.13957) — dense step-level rewards on
     migration/consolidation acts via a learned critic.
  3. Segment-level advantages (Memory-as-Action / DCPO, arXiv:2510.12635) across
     edit-induced trajectory segments.

Scaffold: the Belady labeler (M0-3) is the first concrete deliverable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environment import Action, Representation, RewardWeights, Tier, Timing


@dataclass
class AccessTrace:
    """A logged sequence of future accesses for one item (offline labeling)."""
    item_id: str
    access_steps: list[int]   # timesteps at which the item is next accessed


def belady_optimal_tier(trace: AccessTrace, current_step: int, horizon: int) -> Tier:
    """Hindsight-optimal tier by Belady's principle (reuse-distance).

    Near-future reuse → hot tier (GPU); far/none → cold tier (disk/remote).
    This label warm-starts the policy before RL fine-tuning (RQ2).
    """
    future = [s for s in trace.access_steps if s > current_step]
    if not future:
        return Tier.REMOTE_RDMA
    reuse_distance = min(future) - current_step
    if reuse_distance <= horizon // 4:
        return Tier.GPU
    if reuse_distance <= horizon // 2:
        return Tier.CPU
    if reuse_distance <= horizon:
        return Tier.DISK
    return Tier.REMOTE_RDMA


def process_reward(action: Action, hindsight_tier: Tier) -> float:
    """Step-level process reward: agreement with the Belady/oracle label.

    Placeholder scalar; replace with a learned critic (RAG-Gym style) in M6-9.
    """
    return 1.0 if action.tier == hindsight_tier else 0.0


def weighted_cost(outcome, weights: RewardWeights) -> float:
    """Scalarize a StepOutcome the way the MDP reward penalizes it (§9.3):

        cost = lambda_L*latency + lambda_C*dollars + lambda_B*tokens

    Task utility is constant across legal placements (the item is served either
    way), so minimizing this equals maximizing reward. `outcome` is duck-typed
    (has .latency_ms/.cost_usd/.tokens) to avoid a simulator import cycle.
    """
    return (
        weights.lambda_latency * outcome.latency_ms
        + weights.lambda_cost * outcome.cost_usd
        + weights.lambda_tokens * outcome.tokens
    )


def belady_optimal_action(
    n_future_accesses: int,
    sim,
    weights: RewardWeights,
    current_repr: Representation = Representation.PLAINTEXT,
) -> tuple[Action, float]:
    """Hindsight-optimal placement for one item, generalizing Belady from tier to
    the full (representation x tier x timing) action (docs/research_plan.md §9.2).

    Given the item's full future access count, pick the legal placement minimizing
    `weighted_cost` across those accesses. Because capacity is *not* considered
    here, the summed per-item optima form a valid UPPER BOUND on achievable reward
    (a lower bound on cost) — the oracle ceiling used to warm-start the policy and
    to bound the gap in RQ2. `sim` supplies the cost model (TieringSimulator).
    """
    if n_future_accesses <= 0:
        # Never reused again -> coldest, cheapest placement (no promotion).
        return Action(current_repr, Tier.REMOTE_RDMA, Timing.NOOP), 0.0
    best: Action | None = None
    best_cost = float("inf")
    for action in Action.space():
        cost = weighted_cost(sim.cost_of_placement(action, n_future_accesses, current_repr), weights)
        if cost < best_cost:
            best, best_cost = action, cost
    assert best is not None
    return best, best_cost
