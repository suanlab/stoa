"""STOA control plane (docs/stoa_design.md §3).

Sits above a MemOS-style store, LMCache-style tiering, and an M+-style latent
path. Online actions migrate/serve; offline (sleep-time) actions consolidate
during idle windows.

Scaffold: wires policy + simulator + reward into a single-step loop so the
pipeline is runnable end-to-end before real backends are attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .environment import Budget, ItemState, RewardWeights, reward
from .policy import OrchestratorPolicy
from .simulator import TieringSimulator


@dataclass
class Orchestrator:
    policy: OrchestratorPolicy = field(default_factory=OrchestratorPolicy)
    simulator: TieringSimulator = field(default_factory=TieringSimulator)
    budget: Budget = field(default_factory=Budget)
    weights: RewardWeights = field(default_factory=RewardWeights)

    def step(self, items: list[ItemState], telemetry: dict | None = None) -> dict:
        """One orchestration step over a batch of candidate items.

        Returns aggregate realized metrics + reward. This is the unit the RL
        training loop (M6-12) will optimize.
        """
        telemetry = telemetry or {}
        actions = self.policy.act(items, telemetry)

        tot_lat = tot_cost = tot_tok = 0.0
        for action in actions:
            out = self.simulator.step(action)
            tot_lat += out.latency_ms
            tot_cost += out.cost_usd
            tot_tok += out.tokens

        r = reward(
            task_utility=1.0,       # TODO: from downstream benchmark (eval/benchmarks.py)
            latency_ms=tot_lat,
            cost_usd=tot_cost,
            tokens=int(tot_tok),
            freshness=1.0,
            w=self.weights,
        )
        within_budget = (
            tot_lat <= self.budget.latency_ms
            and tot_cost <= self.budget.cost_usd
            and tot_tok <= self.budget.tokens
        )
        return {
            "actions": actions,
            "latency_ms": tot_lat,
            "cost_usd": tot_cost,
            "tokens": int(tot_tok),
            "reward": r,
            "within_budget": within_budget,
        }
