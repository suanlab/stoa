"""STOA MDP environment.

Formalizes the budget-constrained MDP from docs/stoa_design.md §2:
each memory item's placement decision is an action over
    Representation x Tier x Timing
optimized for downstream task utility under latency + cost + token budgets.

This module defines the *interfaces* (state, action space, reward). The
transition dynamics live in `simulator.py`; the policy in `policy.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Representation(str, Enum):
    """How a memory item is stored (semantic form)."""
    PLAINTEXT = "plaintext"
    VECTOR = "vector"
    GRAPH = "graph"
    LATENT = "latent"          # activation / KV-space memory (cf. M+, LMCache)
    PARAMETER = "parameter"    # baked into weights (cf. MemoryLLM, Larimar)


class Tier(str, Enum):
    """Where a memory item physically lives (storage tier)."""
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"
    REMOTE_RDMA = "remote_rdma"


class Timing(str, Enum):
    """When the placement action is executed."""
    ONLINE = "online"          # act now, in the serving path
    SLEEP = "sleep"            # defer to offline consolidation window
    NOOP = "noop"


@dataclass(frozen=True)
class Action:
    """A per-item placement decision. Illegal combos are filtered by `is_legal`."""
    representation: Representation
    tier: Tier
    timing: Timing

    @staticmethod
    def space() -> list["Action"]:
        """Enumerate the full (masked) action space."""
        return [
            Action(r, t, ts)
            for r in Representation
            for t in Tier
            for ts in Timing
            if Action(r, t, ts).is_legal()
        ]

    def is_legal(self) -> bool:
        """Mask physically/semantically impossible combinations.

        Examples encoded here:
          - PARAMETER promotion is expensive → SLEEP only (never inline ONLINE).
          - LATENT (KV) memory is only meaningful on GPU/CPU tiers.
        """
        if self.representation is Representation.PARAMETER and self.timing is Timing.ONLINE:
            return False
        if self.representation is Representation.LATENT and self.tier in (Tier.DISK, Tier.REMOTE_RDMA):
            return False
        return True

    @staticmethod
    def sample() -> "Action":
        """Deterministic first legal action (placeholder; real sampling in policy)."""
        return Action.space()[0]


@dataclass
class ItemState:
    """Per-candidate memory item state (docs/stoa_design.md §2, state s_t)."""
    item_id: str
    # (a) semantic
    embedding: Sequence[float] = field(default_factory=list)
    recency: float = 0.0
    access_freq: float = 0.0
    version: int = 0                       # MemCube provenance/version
    predicted_next_access: float = 0.0     # PARROT/Belady-style forecast
    # (b) systems
    representation: Representation = Representation.PLAINTEXT
    tier: Tier = Tier.CPU
    size_bytes: int = 0


@dataclass
class Budget:
    """Serve-time budget knob (L, C, B). Lagrangian duals learned in policy."""
    latency_ms: float = 500.0     # L
    cost_usd: float = 1.0         # C
    tokens: int = 8_000           # B


@dataclass
class RewardWeights:
    """Learned Lagrangian multipliers λ_L, λ_C, λ_B + freshness bonus weight."""
    lambda_latency: float = 1.0
    lambda_cost: float = 1.0
    lambda_tokens: float = 1.0
    freshness_bonus: float = 0.1


def reward(
    task_utility: float,
    latency_ms: float,
    cost_usd: float,
    tokens: int,
    freshness: float,
    w: RewardWeights,
) -> float:
    """Lagrangian-relaxed reward r_t (docs/stoa_design.md §2).

    r_t = U - λ_L·lat - λ_C·cost - λ_B·tok + freshness_bonus
    """
    return (
        task_utility
        - w.lambda_latency * latency_ms
        - w.lambda_cost * cost_usd
        - w.lambda_tokens * tokens
        + w.freshness_bonus * freshness
    )
