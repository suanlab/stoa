"""Budget-conditioned control frontier (docs/research_plan.md §8 RQ1/RQ3; M9-12).

RQ3's promise is a single serve-time knob (L, C, B): the operator sets a budget and
one policy adapts, tracing the accuracy/latency/cost frontier without retraining.
This module demonstrates that torch-free on the token budget B: starting from the
cheapest-to-write placement (all plaintext), it promotes items off plaintext in
order of token-savings-per-dollar until total context tokens fall under B — a
budget-conditioned controller that, swept over B, traces the achievable frontier.

Because promotion cost rises as tokens fall (plaintext $0 -> vector -> latent), the
controller spends the least dollars needed to meet each budget, and hot items
(more accesses) are promoted first (more token savings per dollar). Static baselines
(all-plaintext, all-vector) are single points the frontier Pareto-dominates.

TODO(M9-12): replace the greedy sweep with a single (L,C,B)-conditioned learned
policy (online GRPO + segment advantage) that reproduces this frontier in one shot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..environment import Action, Representation, RewardWeights, Tier, Timing
from ..simulator import TieringSimulator
from ..traces import Workload

# Representation ladder (token-decreasing), all on CPU. Plaintext is written for
# free (noop); promotions are consolidated offline (sleep).
_LADDER = [
    Action(Representation.PLAINTEXT, Tier.CPU, Timing.NOOP),
    Action(Representation.VECTOR, Tier.CPU, Timing.SLEEP),
    Action(Representation.LATENT, Tier.CPU, Timing.SLEEP),
]


@dataclass
class FrontierPoint:
    budget_tokens: int
    tokens_used: int
    dollar_cost: float
    latency_ms: float
    promotions: int
    feasible: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dollar_cost"] = round(self.dollar_cost, 6)
        d["latency_ms"] = round(self.latency_ms, 3)
        return d


def _ladder_costs(workload: Workload, sim: TieringSimulator) -> dict[str, list]:
    """Per-item (tokens, dollars, latency) at each ladder level."""
    out: dict[str, list] = {}
    for item in workload.items:
        n = workload.n_accesses(item.item_id)
        out[item.item_id] = [sim.cost_of_placement(a, n, Representation.PLAINTEXT) for a in _LADDER]
    return out


def budget_conditioned_placement(
    workload: Workload,
    budget_tokens: int,
    sim: TieringSimulator | None = None,
) -> tuple[dict[str, Action], FrontierPoint]:
    """Min-dollar placement whose total context tokens fit `budget_tokens` (RQ3)."""
    sim = sim or TieringSimulator()
    costs = _ladder_costs(workload, sim)
    level = {i: 0 for i in costs}                      # start all-plaintext
    tokens = sum(costs[i][0].tokens for i in costs)
    dollars = sum(costs[i][0].cost_usd for i in costs)

    while tokens > budget_tokens:
        best_item, best_ratio, best = None, -1.0, None
        for i, lv in level.items():
            if lv >= len(_LADDER) - 1:
                continue
            cur, nxt = costs[i][lv], costs[i][lv + 1]
            d_tok = cur.tokens - nxt.tokens
            d_cost = nxt.cost_usd - cur.cost_usd
            ratio = d_tok / d_cost if d_cost > 0 else float("inf")
            if ratio > best_ratio:
                best_item, best_ratio, best = i, ratio, (d_tok, d_cost)
        if best_item is None:
            break                                      # fully promoted; cannot reduce further
        level[best_item] += 1
        tokens -= best[0]
        dollars += best[1]

    actions = {i: _LADDER[level[i]] for i in level}
    latency = sum(costs[i][level[i]].latency_ms for i in level)
    promotions = sum(1 for lv in level.values() if lv > 0)
    point = FrontierPoint(budget_tokens, tokens, dollars, latency, promotions,
                          feasible=tokens <= budget_tokens)
    return actions, point


def frontier(workload: Workload, budgets: list[int], sim: TieringSimulator | None = None) -> list[FrontierPoint]:
    """Sweep the token budget -> the achievable (tokens, $, latency) frontier."""
    sim = sim or TieringSimulator()
    return [budget_conditioned_placement(workload, b, sim)[1] for b in budgets]


def static_point(workload: Workload, level: int, sim: TieringSimulator | None = None) -> FrontierPoint:
    """A budget-agnostic baseline that pins every item to one ladder level."""
    sim = sim or TieringSimulator()
    costs = _ladder_costs(workload, sim)
    tokens = sum(costs[i][level].tokens for i in costs)
    dollars = sum(costs[i][level].cost_usd for i in costs)
    latency = sum(costs[i][level].latency_ms for i in costs)
    promotions = len(costs) if level > 0 else 0
    return FrontierPoint(tokens, tokens, dollars, latency, promotions, feasible=True)
