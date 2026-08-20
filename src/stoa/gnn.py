"""GNN factored policy (docs/research_plan.md §10, M6-9).

The factored policy pi_theta = pi_rep * pi_tier * pi_time on a shared GNN trunk
over the memory graph (nodes = items, edges = co-access / entity links). Acting on
the whole batch via a Decima-style GNN (arXiv:1810.01963) avoids per-item
combinatorics. The three factor heads emit logits that are *summed* over the 50
legal joint actions (environment.Action.space) and softmaxed only there, so the
distribution respects Action.is_legal masking exactly while staying factored.

Torch-only, CPU-friendly (the policy trains over the tiering *simulator*, not an
LLM — no GPU needed). Deliberately NOT imported from stoa/__init__ so that
`import stoa` stays dependency-free for the stdlib M0-3 core. torch-geometric is
not required: message passing is a plain normalized-adjacency matmul.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .environment import Action, Representation, Tier, Timing

_REPRS = list(Representation)
_TIERS = list(Tier)
_TIMINGS = list(Timing)
_REPR_IX = {r: i for i, r in enumerate(_REPRS)}
_TIER_IX = {t: i for i, t in enumerate(_TIERS)}
_TIME_IX = {t: i for i, t in enumerate(_TIMINGS)}

# The fixed 50-action legal set and its factor indices (built once).
LEGAL_ACTIONS = Action.space()
_LEGAL_REP = torch.tensor([_REPR_IX[a.representation] for a in LEGAL_ACTIONS])
_LEGAL_TIER = torch.tensor([_TIER_IX[a.tier] for a in LEGAL_ACTIONS])
_LEGAL_TIME = torch.tensor([_TIME_IX[a.timing] for a in LEGAL_ACTIONS])


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    n = adj.shape[0]
    a = adj + torch.eye(n, dtype=adj.dtype)
    deg = a.sum(1)
    dinv = torch.diag(deg.clamp(min=1e-8).pow(-0.5))
    return dinv @ a @ dinv


class GNNPolicy(nn.Module):
    """Shared GCN trunk + factored (rep, tier, time) heads + a value head."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.gcs = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.rep_head = nn.Linear(hidden, len(_REPRS))
        self.tier_head = nn.Linear(hidden, len(_TIERS))
        self.time_head = nn.Linear(hidden, len(_TIMINGS))
        self.value = nn.Linear(hidden, 1)

    def trunk(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for gc in self.gcs:
            h = F.relu(adj_norm @ gc(h)) + h              # residual GCN layer
        return h

    def action_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Per-node logits over the 50 legal joint actions (factored, masked)."""
        rl, tl, ml = self.rep_head(h), self.tier_head(h), self.time_head(h)
        return (rl[:, _LEGAL_REP] + tl[:, _LEGAL_TIER] + ml[:, _LEGAL_TIME])  # (N, 50)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x, adj_norm)
        return self.action_logits(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor, adj_norm: torch.Tensor) -> list[Action]:
        """Greedy legal action per node (argmax over the 50-action distribution)."""
        logits, _ = self.forward(x, adj_norm)
        return [LEGAL_ACTIONS[i] for i in logits.argmax(1).tolist()]
