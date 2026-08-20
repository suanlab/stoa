"""Offline policy-gradient fine-tuning of the GNN policy (docs/research_plan.md §3, §10; M6-9/M9-12).

Behavioral cloning (train.py) imitates the capacity-*greedy* Belady heuristic, but
the factored per-node argmax has no view of the *global* tier-capacity constraint,
so it can over-subscribe hot tiers. This module fine-tunes the BC-warm-started
policy with REINFORCE on the true constrained objective —

    reward = -(weighted resource cost / naive cost) - beta * (tier overflow / total bytes)

— using the GNN value head as a variance-reducing baseline (the critic the plan
calls for). Because the reward includes the capacity term, RL can do something BC
structurally cannot: trade a little cost to remove infeasible over-subscription.

Torch-only, CPU (tiny graph; reward is a lookup, no LLM). Not imported by __init__.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .credit import weighted_cost
from .environment import RewardWeights, Tier
from .eval.oracle import DEFAULT_WEIGHTS, NAIVE_ACTION
from .gnn import LEGAL_ACTIONS, GNNPolicy, normalize_adj
from .simulator import TieringSimulator
from .train import build_adjacency, build_features
from .traces import Workload

_TIERS = list(Tier)
_TIER_IX = {t: i for i, t in enumerate(_TIERS)}
_ACTION_TIER = torch.tensor([_TIER_IX[a.tier] for a in LEGAL_ACTIONS])


@dataclass
class _Graph:
    x: torch.Tensor            # (N, F) node features
    adj: torch.Tensor          # (N, N) normalized adjacency
    cost: torch.Tensor         # (N, 50) weighted cost of each legal action
    sizes: torch.Tensor        # (N,) item bytes
    cap: torch.Tensor          # (4,) tier byte capacities (inf where unbounded)
    naive_cost: float
    total_bytes: float


def _prepare(wl: Workload, sim: TieringSimulator, w: RewardWeights,
             split_frac: float = 0.5) -> _Graph:
    """Features come from build_features (observable prefix); costs are scored on the
    FUTURE accesses — hindsight belongs in the reward, never in the input."""
    n = len(wl.items)
    st = max(1, wl.time_split(split_frac))
    future = wl.future_counts(st)
    cost = torch.zeros(n, len(LEGAL_ACTIONS))
    for i, item in enumerate(wl.items):
        k = future.get(item.item_id, 0)
        for j, a in enumerate(LEGAL_ACTIONS):
            cost[i, j] = weighted_cost(sim.cost_of_placement(a, k, item.representation), w)
    sizes = torch.tensor([float(it.size_bytes) for it in wl.items])
    caps = sim.tier_capacities(wl.total_bytes)
    cap = torch.tensor([caps[t] for t in _TIERS])            # inf for REMOTE_RDMA
    naive = sum(weighted_cost(sim.cost_of_placement(NAIVE_ACTION, future.get(it.item_id, 0),
                                                    it.representation), w) for it in wl.items)
    return _Graph(build_features(wl), normalize_adj(build_adjacency(wl)), cost,
                  sizes, cap, float(naive), float(wl.total_bytes))


def placement_metrics(actions: torch.Tensor, g: _Graph) -> tuple[torch.Tensor, torch.Tensor]:
    """Total weighted cost and total tier-overflow (bytes) for a per-node action vector."""
    total_cost = g.cost[torch.arange(g.cost.shape[0]), actions].sum()
    tier_of = _ACTION_TIER[actions]
    used = torch.zeros(len(_TIERS)).index_add_(0, tier_of, g.sizes)
    finite = torch.isfinite(g.cap)
    overflow = torch.clamp(used[finite] - g.cap[finite], min=0.0).sum()
    return total_cost, overflow


def _per_item_reward(actions: torch.Tensor, g: _Graph, beta: float) -> torch.Tensor:
    """Per-item reward: own cost + its *share* of the overflow it sits in.

    Attributing overflow to the items causing it (proportional to size on the
    over-subscribed tier) gives the per-node credit REINFORCE needs to learn WHICH
    items to demote — a global scalar reward cannot. Returned tensor is detached
    (it indexes sampled actions); gradient flows only through log-prob / value.
    """
    n = g.cost.shape[0]
    cost_i = g.cost[torch.arange(n), actions]
    tier_of = _ACTION_TIER[actions]
    used = torch.zeros(len(_TIERS)).index_add_(0, tier_of, g.sizes)
    overflow_t = torch.clamp(used - g.cap, min=0.0)                 # inf cap -> 0
    used_by_tier = used[tier_of].clamp(min=1.0)
    penalty_i = overflow_t[tier_of] * (g.sizes / used_by_tier)      # bytes attributed to item
    return (-(cost_i / g.naive_cost) - beta * (penalty_i / g.total_bytes)).detach()


@dataclass
class FinetuneResult:
    steps: int
    beta: float
    final_reward: float


def finetune(
    policy: GNNPolicy,
    workloads: list[Workload],
    sim: TieringSimulator | None = None,
    weights: RewardWeights | None = None,
    steps: int = 300,
    lr: float = 3e-3,
    beta: float = 1.0,
    seed: int = 0,
) -> tuple[GNNPolicy, FinetuneResult]:
    """REINFORCE fine-tuning with the value head as baseline (starts from `policy`)."""
    torch.manual_seed(seed)
    sim = sim or TieringSimulator()
    w = weights or DEFAULT_WEIGHTS
    graphs = [_prepare(wl, sim, w) for wl in workloads]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        rewards = []
        for g in graphs:
            logits, value = policy(g.x, g.adj)                     # value: (N,) per-node baseline
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)                          # (N,)
            r_i = _per_item_reward(actions, g, beta)               # (N,)
            adv = r_i - value.detach()
            loss = loss - (adv * logp).sum() + 0.5 * ((value - r_i) ** 2).sum()
            rewards.append(r_i.mean().item())
        (loss / len(graphs)).backward()
        opt.step()
        last = sum(rewards) / len(rewards)
    return policy, FinetuneResult(steps, beta, last)


@dataclass
class PlacementEval:
    cost: float
    overflow_bytes: float
    feasible: bool

    def to_dict(self) -> dict:
        return {"cost": round(self.cost, 4), "overflow_bytes": round(self.overflow_bytes, 1),
                "feasible": self.feasible}


def evaluate_placement(policy: GNNPolicy, wl: Workload, sim: TieringSimulator | None = None,
                       weights: RewardWeights | None = None) -> PlacementEval:
    """Greedy (argmax) placement metrics for a policy on one workload."""
    sim = sim or TieringSimulator()
    g = _prepare(wl, sim, weights or DEFAULT_WEIGHTS)
    with torch.no_grad():
        logits, _ = policy(g.x, g.adj)
        actions = logits.argmax(1)
    cost, overflow = placement_metrics(actions, g)
    return PlacementEval(cost.item(), overflow.item(), overflow.item() <= 0.0)
