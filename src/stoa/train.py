"""Behavioral-cloning warm-start for the GNN factored policy (RQ2; M6-9).

The Belady oracle is expensive (needs the future) but its *labels* are a supervised
target. Following PARROT (arXiv:2006.16239), we warm-start the GNN policy by
cloning the capacity-feasible Belady placement (eval.oracle.greedy_belady_actions),
then measure how close the cloned policy gets to that target on held-out traces.
This is the imitation stage that offline-RL (Cold-RL style) fine-tunes next.

Torch-only; not imported from stoa/__init__ (keeps the stdlib core dependency-free).
CPU is sufficient — the graph is tiny and the target is precomputed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .credit import weighted_cost
from .environment import Representation, RewardWeights, Tier
from .eval.oracle import DEFAULT_WEIGHTS, greedy_belady_actions
from .gnn import LEGAL_ACTIONS, GNNPolicy, normalize_adj
from .simulator import TieringSimulator
from .traces import Workload

_ACTION_IX = {a: i for i, a in enumerate(LEGAL_ACTIONS)}
_TIERS = list(Tier)
_REPRS = list(Representation)
# [log prefix-count, count-frac, recency, mean-interarrival, size-frac] + tier/repr onehots
FEATURE_DIM = 5 + len(_TIERS) + len(_REPRS)
DEFAULT_SPLIT_FRAC = 0.5      # observe the first half, predict placement for the rest


def split_step(workload: Workload, split_frac: float = DEFAULT_SPLIT_FRAC) -> int:
    """Split by TIME (see Workload.time_split) -- not by index into `accesses`."""
    return max(1, workload.time_split(split_frac))


def build_features(workload: Workload, split_frac: float = DEFAULT_SPLIT_FRAC) -> torch.Tensor:
    """Per-item node features from the OBSERVABLE PREFIX only (no future knowledge).

    Uses accesses before `split_frac` of the trace: prefix count, recency, mean
    inter-arrival, plus size and current tier/representation. The Belady oracle
    scores the *future* (see `belady_labels`), so the learning problem is the real
    one -- predict future reuse from what has been observed (cf. PARROT).
    """
    st = split_step(workload, split_frac)
    stats = workload.prefix_stats(st)
    counts = [stats[it.item_id]["count"] for it in workload.items]
    max_c = max(counts) or 1.0
    max_sz = max((it.size_bytes for it in workload.items), default=1) or 1
    rows: list[list[float]] = []
    for it in workload.items:
        s = stats[it.item_id]
        row = [
            torch.log1p(torch.tensor(s["count"])).item(),
            s["count"] / max_c,
            s["recency"] / max(st, 1),
            s["mean_interarrival"] / max(st, 1),
            it.size_bytes / max_sz,
        ]
        tier_oh = [1.0 if it.tier is t else 0.0 for t in _TIERS]
        repr_oh = [1.0 if it.representation is r else 0.0 for r in _REPRS]
        rows.append(row + tier_oh + repr_oh)
    return torch.tensor(rows, dtype=torch.float32)


def build_adjacency(workload: Workload, window: int = 5) -> torch.Tensor:
    """Co-access graph: items seen within `window` steps of each other get an edge
    (weighted by co-occurrence). This is the structure the GNN propagates over."""
    idx = {it.item_id: i for i, it in enumerate(workload.items)}
    n = len(workload.items)
    adj = torch.zeros(n, n)
    recent: list[str] = []
    for _step, item in workload.accesses:
        i = idx[item]
        for prev in recent:
            j = idx[prev]
            if i != j:
                adj[i, j] += 1.0
                adj[j, i] += 1.0
        recent.append(item)
        if len(recent) > window:
            recent.pop(0)
    return adj


def belady_labels(workload: Workload, sim: TieringSimulator, weights: RewardWeights,
                  split_frac: float = DEFAULT_SPLIT_FRAC) -> torch.Tensor:
    """Per-item target action index: capacity-feasible Belady placement scored on the
    FUTURE portion of the trace (hindsight is legitimate in the label, not the input)."""
    counts = workload.future_counts(split_step(workload, split_frac))
    actions = greedy_belady_actions(workload, sim, weights, counts=counts)
    return torch.tensor([_ACTION_IX[actions[it.item_id]] for it in workload.items], dtype=torch.long)


@dataclass
class TrainResult:
    epochs: int
    final_loss: float
    train_acc: float


def train_bc(
    workloads: list[Workload],
    sim: TieringSimulator | None = None,
    weights: RewardWeights | None = None,
    hidden: int = 64,
    layers: int = 3,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    split_frac: float = DEFAULT_SPLIT_FRAC,
) -> tuple[GNNPolicy, TrainResult]:
    """Clone the Belady placement across `workloads` (full-batch per graph).

    Features come from the observable prefix; labels from the future (see above).
    """
    torch.manual_seed(seed)
    sim = sim or TieringSimulator()
    w = weights or DEFAULT_WEIGHTS
    graphs = [(build_features(wl, split_frac), normalize_adj(build_adjacency(wl)),
               belady_labels(wl, sim, w, split_frac)) for wl in workloads]

    policy = GNNPolicy(FEATURE_DIM, hidden=hidden, layers=layers)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for x, adj, y in graphs:
            logits, _ = policy(x, adj)
            loss = loss + F.cross_entropy(logits, y)
        loss = loss / len(graphs)
        loss.backward()
        opt.step()

    with torch.no_grad():
        correct = total = 0
        for x, adj, y in graphs:
            logits, _ = policy(x, adj)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return policy, TrainResult(epochs, loss.item(), correct / total)


@dataclass
class EvalResult:
    imitation_acc: float          # fraction of items matching the Belady target
    bc_cost: float                # weighted cost of the cloned policy's placement
    target_cost: float            # weighted cost of the Belady target (achievable)
    naive_cost: float             # all-plaintext/CPU
    tier_hist: dict[str, int]

    @property
    def gap_closed_pct(self) -> float:
        span = self.naive_cost - self.target_cost
        return 100.0 * (self.naive_cost - self.bc_cost) / span if span > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "imitation_acc": round(self.imitation_acc, 4),
            "bc_cost": round(self.bc_cost, 4),
            "target_cost": round(self.target_cost, 4),
            "naive_cost": round(self.naive_cost, 4),
            "gap_closed_pct": round(self.gap_closed_pct, 2),
            "tier_hist": self.tier_hist,
        }


def evaluate(policy: GNNPolicy, workload: Workload, sim: TieringSimulator | None = None,
             weights: RewardWeights | None = None,
             split_frac: float = DEFAULT_SPLIT_FRAC) -> EvalResult:
    """Apply the cloned policy on a held-out trace and compare to the Belady target.

    Policy sees only prefix features; all costs are scored on the future portion.
    """
    sim = sim or TieringSimulator()
    w = weights or DEFAULT_WEIGHTS
    x = build_features(workload, split_frac)
    adj = normalize_adj(build_adjacency(workload))
    actions = policy.act(x, adj)
    counts = workload.future_counts(split_step(workload, split_frac))
    target = greedy_belady_actions(workload, sim, w, counts=counts)

    from .eval.oracle import NAIVE_ACTION
    match = bc_cost = target_cost = naive_cost = 0.0
    for it, action in zip(workload.items, actions):
        n = counts.get(it.item_id, 0)
        match += action == target[it.item_id]
        bc_cost += weighted_cost(sim.cost_of_placement(action, n, it.representation), w)
        target_cost += weighted_cost(sim.cost_of_placement(target[it.item_id], n, it.representation), w)
        naive_cost += weighted_cost(sim.cost_of_placement(NAIVE_ACTION, n, it.representation), w)
    hist = dict(Counter(a.tier.value for a in actions))
    return EvalResult(match / len(actions), bc_cost, target_cost, naive_cost, hist)
