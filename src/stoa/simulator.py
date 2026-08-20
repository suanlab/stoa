"""Tiering simulator (docs/stoa_design.md §7, M0-3 deliverable).

Calibrated from LMCache-style traces, the simulator provides the MDP transition
dynamics: given an Action, it returns the realized (latency, cost, tokens,
freshness) so policies can be trained offline before real vLLM+LMCache validation.

Scaffold: cost/latency tables are illustrative placeholders. Replace with values
fit from real GPU/CPU/disk/RDMA traces during M0-3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .environment import Action, Representation, Tier

# Illustrative per-tier read latency (ms) — REPLACE with trace-fit values.
_TIER_READ_MS: dict[Tier, float] = {
    Tier.GPU: 0.2,
    Tier.CPU: 2.0,
    Tier.DISK: 12.0,
    Tier.REMOTE_RDMA: 6.0,
}

# Illustrative token cost of serving a representation into context.
_REPR_TOKENS: dict[Representation, int] = {
    Representation.PLAINTEXT: 800,
    Representation.VECTOR: 120,
    Representation.GRAPH: 200,
    Representation.LATENT: 0,       # served as activations, not tokens
    Representation.PARAMETER: 0,
}

# Illustrative one-off write/consolidation cost (USD) of promoting to a representation.
_PROMOTE_COST: dict[Representation, float] = {
    Representation.PLAINTEXT: 0.0,
    Representation.VECTOR: 0.001,
    Representation.GRAPH: 0.004,
    Representation.LATENT: 0.02,
    Representation.PARAMETER: 0.10,
}

# Illustrative per-tier capacity as a FRACTION of the total working-set bytes.
# Scarcity of hot tiers is what makes reuse-distance-aware placement pay off
# (Belady). REPLACE with real device capacities during M0-3 calibration.
_TIER_CAPACITY_FRACTION: dict[Tier, float] = {
    Tier.GPU: 0.10,
    Tier.CPU: 0.30,
    Tier.REMOTE_RDMA: 0.60,           # remote DRAM: fast but finite and expensive
    Tier.DISK: float("inf"),          # local storage: slow but effectively unbounded
}
# Ordering note: an earlier table gave DISK 12 ms / 1.0x and REMOTE_RDMA 6 ms / inf, which
# made remote both faster AND larger, so no cost-minimizing policy would ever pick DISK and
# the "four-tier" hierarchy was really three. Real hierarchies trade latency against
# capacity monotonically; this table now does too.


@dataclass
class StepOutcome:
    latency_ms: float
    cost_usd: float
    tokens: int
    freshness: float


@dataclass
class TieringSimulator:
    """Minimal transition model over placement actions."""
    horizon: int = 1_000
    _t: int = field(default=0, init=False)

    def reset(self) -> None:
        self._t = 0

    def step(self, action: Action, task_utility: float = 1.0) -> StepOutcome:
        """Return realized costs of executing `action` at the current step."""
        self._t += 1
        latency = _TIER_READ_MS[action.tier]
        tokens = _REPR_TOKENS[action.representation]
        # Sleep-time consolidation defers cost off the serving path (amortized).
        cost = _PROMOTE_COST[action.representation]
        if action.timing.value == "sleep":
            cost *= 0.25          # amortized offline
            latency = 0.0         # not on serving path
        elif action.timing.value == "noop":
            cost = 0.0
        freshness = 1.0           # TODO: decay with staleness under knowledge updates
        return StepOutcome(latency_ms=latency, cost_usd=cost, tokens=tokens, freshness=freshness)

    def cost_of_placement(
        self,
        action: Action,
        n_accesses: int,
        current_repr: Representation = Representation.PLAINTEXT,
    ) -> StepOutcome:
        """Aggregate cost of *holding* an item under `action` and serving it
        `n_accesses` times (docs/research_plan.md §9.3).

        One-off promotion (write) cost + per-access read latency and token cost:
          - promotion is charged only when the representation actually changes;
          - sleep amortizes it off the serving path (0.25x); online pays full;
          - noop cannot change representation, so noop+change is infeasible (inf).
        This is the per-item cost the Belady oracle minimizes (credit.py).
        """
        read_lat = _TIER_READ_MS[action.tier]
        tokens_per = _REPR_TOKENS[action.representation]
        changed = action.representation != current_repr
        promote = _PROMOTE_COST[action.representation] if changed else 0.0
        if action.timing.value == "sleep":
            promote *= 0.25
        elif action.timing.value == "noop":
            promote = float("inf") if changed else 0.0
        return StepOutcome(
            latency_ms=read_lat * n_accesses,
            cost_usd=promote,
            tokens=int(tokens_per * n_accesses),
            freshness=1.0,
        )

    @staticmethod
    def tier_capacities(total_bytes: int) -> dict[Tier, float]:
        """Absolute per-tier byte capacities for a workload of `total_bytes`."""
        return {t: frac * total_bytes for t, frac in _TIER_CAPACITY_FRACTION.items()}

    @staticmethod
    def read_latency(tier: Tier) -> float:
        """Per-access read latency (ms) of a tier — the online tier-axis signal."""
        return _TIER_READ_MS[tier]
