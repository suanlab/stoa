"""Factored orchestrator policy (docs/stoa_design.md §3).

π_θ = π_rep · π_tier · π_time on a shared GNN trunk over the memory graph.
Acting on batches of items via a Decima-style GNN avoids per-item combinatorics.

Scaffold only: replace `_encode` / heads with a real GNN + actor-critic once the
simulator (M0–3) and single-axis controllers (M3–6) land.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environment import Action, ItemState, Representation, Tier, Timing


@dataclass
class PolicyConfig:
    hidden_dim: int = 256
    gnn_layers: int = 3
    factored: bool = True   # π_rep · π_tier · π_time vs. flat joint head


class OrchestratorPolicy:
    """Perception (GNN) + factored actor + constrained critic.

    TODO(M6-9): implement GNN trunk, three softmax heads with action masking,
    and a critic estimating constrained value under (L, C, B).
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def _encode(self, items: list[ItemState], telemetry: dict) -> list[list[float]]:
        """GNN over (nodes=items, edges=co-access/entity links) + systems telemetry.

        Placeholder: returns zero embeddings sized to hidden_dim.
        """
        return [[0.0] * self.config.hidden_dim for _ in items]

    def act(self, items: list[ItemState], telemetry: dict) -> list[Action]:
        """Emit one placement Action per item.

        Placeholder policy: keep current tier, no-op. Real policy samples from the
        three factored heads with legality masking (see Action.is_legal).
        """
        _ = self._encode(items, telemetry)
        return [
            Action(item.representation, item.tier, Timing.NOOP)
            if Action(item.representation, item.tier, Timing.NOOP).is_legal()
            else Action(Representation.PLAINTEXT, Tier.CPU, Timing.NOOP)
            for item in items
        ]

    def value(self, items: list[ItemState], telemetry: dict) -> float:
        """Constrained value estimate. Placeholder returns 0.0."""
        return 0.0
