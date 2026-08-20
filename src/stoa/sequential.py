"""Sequential, occupancy-aware placement MDP (docs/research_plan.md §6.1; the fix
the static-RL negative result (rl.py) points to).

The static joint policy can't satisfy a *global* capacity constraint because each
node decides simultaneously, blind to what others take. Here we place items one at a
time in a heat order; the policy observes the *running tier occupancy* and chooses
among the actions that still FIT (unaffordable tiers are masked, and REMOTE_RDMA is
unbounded, so every rollout is feasible by construction). Reward is the negative
placement cost; REINFORCE with return-to-go + a per-step value baseline then learns
a low-cost feasible policy — approaching the sequential greedy-Belady oracle and
strictly dominating the static BC policy, which overflows.

Torch-only, CPU, tiny. Not imported by stoa/__init__.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .credit import weighted_cost
from .environment import RewardWeights, Tier
from .eval.oracle import DEFAULT_WEIGHTS, NAIVE_ACTION
from .gnn import LEGAL_ACTIONS
from .simulator import TieringSimulator
from .traces import Workload

_TIERS = list(Tier)
_TIER_IX = {t: i for i, t in enumerate(_TIERS)}
_ACTION_TIER = torch.tensor([_TIER_IX[a.tier] for a in LEGAL_ACTIONS])
_BASE_FEATS = 4                                  # [log-acc, acc-frac, size-frac]
FEATURE_DIM = _BASE_FEATS + len(_TIERS)          # prefix feats + per-tier occupancy


@dataclass
class _Graph:
    feat: torch.Tensor        # (N, 3) per-item base features, in heat order
    cost: torch.Tensor        # (N, 50) weighted cost of each legal action
    sizes: torch.Tensor       # (N,)
    cap: torch.Tensor         # (4,) tier byte capacities (inf on REMOTE)
    cap_norm: torch.Tensor    # (4,) finite normalizer for occupancy features
    naive_cost: float


def _prepare(wl: Workload, sim: TieringSimulator, w: RewardWeights,
             split_frac: float = 0.5, order_by: str = "prefix") -> _Graph:
    """Build the sequential decision problem.

    HONEST SPLIT: item ordering and features come from the OBSERVABLE PREFIX
    (accesses before the split); placement cost is scored on the FUTURE accesses.
    Feeding whole-trace counts as features would leak hindsight into the policy.
    """
    items = list(wl.items)
    st = max(1, wl.time_split(split_frac))
    stats = wl.prefix_stats(st)
    future = wl.future_counts(st)
    obs = [stats[it.item_id]["count"] for it in items]
    # A policy must order by what it can observe; only the ORACLE reference may order by
    # the future. Ordering the oracle by the prefix (the earlier bug) hides all of its
    # advantage and makes headroom look like zero.
    rank = obs if order_by == "prefix" else [future.get(it.item_id, 0) for it in items]
    order = sorted(range(len(items)), key=lambda i: rank[i], reverse=True)
    items = [items[i] for i in order]
    obs = [obs[i] for i in order]
    accs = [future.get(it.item_id, 0) for it in items]          # oracle quantity: cost only
    max_obs = max(obs) or 1.0
    max_sz = max((it.size_bytes for it in items), default=1) or 1

    feat = torch.tensor([[torch.log1p(torch.tensor(float(o))).item(),
                          o / max_obs,
                          stats[it.item_id]["recency"] / st,
                          it.size_bytes / max_sz]
                         for it, o in zip(items, obs)], dtype=torch.float32)
    cost = torch.zeros(len(items), len(LEGAL_ACTIONS))
    for i, (it, a) in enumerate(zip(items, accs)):
        for j, act in enumerate(LEGAL_ACTIONS):
            cost[i, j] = weighted_cost(sim.cost_of_placement(act, a, it.representation), w)
    sizes = torch.tensor([float(it.size_bytes) for it in items])
    caps = sim.tier_capacities(wl.total_bytes)
    cap = torch.tensor([caps[t] for t in _TIERS])
    finite = torch.isfinite(cap)
    cap_norm = torch.where(finite, cap, torch.full_like(cap, float(wl.total_bytes)))
    naive = sum(weighted_cost(sim.cost_of_placement(NAIVE_ACTION, a, it.representation), w)
                for it, a in zip(items, accs))
    return _Graph(feat, cost, sizes, cap, cap_norm.clamp(min=1.0), float(naive) or 1.0)


class SeqPolicy(nn.Module):
    """MLP actor over the 50 actions + a scalar critic, on [item feat | occupancy]."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, len(LEGAL_ACTIONS))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def _step_input(g: _Graph, i: int, remaining: torch.Tensor) -> torch.Tensor:
    occ = (remaining / g.cap_norm).clamp(0.0, 1.0)
    return torch.cat([g.feat[i], occ])


def _affordable(g: _Graph, i: int, remaining: torch.Tensor) -> torch.Tensor:
    """Actions that fit the tier AND are cost-feasible (finite cost). A noop that
    would change representation is infinite-cost and thus never selectable; a
    finite affordable action always exists (plaintext/REMOTE/noop)."""
    return (remaining[_ACTION_TIER] >= g.sizes[i]) & torch.isfinite(g.cost[i])


def rollout(policy: SeqPolicy, g: _Graph, sample: bool = True):
    """One feasible placement pass. Returns (total_cost, logps, values, rewards)."""
    remaining = g.cap.clone()
    total = 0.0
    logps, values, rewards = [], [], []
    for i in range(g.feat.shape[0]):
        x = _step_input(g, i, remaining)
        logits, value = policy(x)
        logits = logits.masked_fill(~_affordable(g, i, remaining), float("-inf"))
        dist = Categorical(logits=logits)
        a = dist.sample() if sample else logits.argmax()
        logps.append(dist.log_prob(a))
        values.append(value)
        c = g.cost[i, a]
        total += c.item()
        rewards.append(-(c.item() / g.naive_cost))
        remaining = remaining.clone()
        remaining[_ACTION_TIER[a]] -= g.sizes[i]
    return total, logps, values, rewards


@dataclass
class SeqResult:
    steps: int
    final_cost: float


def train_seq(workloads: list[Workload], sim: TieringSimulator | None = None,
              weights: RewardWeights | None = None, steps: int = 400, lr: float = 3e-3,
              seed: int = 0) -> tuple[SeqPolicy, SeqResult]:
    torch.manual_seed(seed)
    sim = sim or TieringSimulator()
    graphs = [_prepare(wl, sim, weights or DEFAULT_WEIGHTS) for wl in workloads]
    policy = SeqPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        costs = []
        for g in graphs:
            total, logps, values, rewards = rollout(policy, g, sample=True)
            returns, acc = [], 0.0
            for r in reversed(rewards):                  # return-to-go, gamma=1
                acc = r + acc
                returns.insert(0, acc)
            ret = torch.tensor(returns)
            val = torch.stack(values)
            adv = ret - val.detach()
            loss = loss - (adv * torch.stack(logps)).sum() + 0.5 * F.mse_loss(val, ret, reduction="sum")
            costs.append(total)
        (loss / len(graphs)).backward()
        opt.step()
        last = sum(costs) / len(costs)
    return policy, SeqResult(steps, last)


def evaluate_seq(policy: SeqPolicy, wl: Workload, sim: TieringSimulator | None = None,
                 weights: RewardWeights | None = None) -> float:
    """Greedy (argmax) feasible placement cost."""
    g = _prepare(wl, sim or TieringSimulator(), weights or DEFAULT_WEIGHTS)
    with torch.no_grad():
        total, *_ = rollout(policy, g, sample=False)
    return total


def _oracle_cost(g: _Graph, mode: str, seed: int = 0) -> float:
    """Reference feasible passes: 'greedy' = min-cost affordable; 'random' = random affordable.

    Ties are broken by tier speed, matching `place_with_beliefs` and `prefix_greedy_cost`.
    All three routines choose among equal-cost actions, and a block with no future accesses
    makes every action cost the same -- so an `argmin` that returns the first index parks
    those blocks on whichever tier the enum happens to list first. The three must agree or
    the numerator and denominator of every reported fraction are computed under different
    rules.
    """
    rng = torch.Generator().manual_seed(seed)
    tier_lat = torch.tensor(
        [TieringSimulator().read_latency(act.tier) for act in LEGAL_ACTIONS],
        dtype=torch.float32)
    remaining = g.cap.clone()
    total = 0.0
    for i in range(g.feat.shape[0]):
        afford = _affordable(g, i, remaining)
        idx = afford.nonzero(as_tuple=True)[0]
        if mode == "greedy":
            vals = g.cost[i, idx]
            tied = idx[(vals <= vals.min() + 1e-12).nonzero(as_tuple=True)[0]]
            a = tied[tier_lat[tied].argmin()]
        else:
            a = idx[torch.randint(len(idx), (1,), generator=rng)]
        total += g.cost[i, a].item()
        remaining = remaining.clone()
        remaining[_ACTION_TIER[a]] -= g.sizes[i]
    return total


def place_with_beliefs(wl: Workload, believed: dict[str, float],
                       sim: TieringSimulator | None = None,
                       weights: RewardWeights | None = None, split_frac: float = 0.5) -> float:
    """Greedy capacity-feasible placement driven by `believed` per-item access counts,
    but PAID at the true future cost.

    This is the single knob that separates every policy we compare: pass observed
    prefix counts for the no-learning baseline, a model's predictions for a learned
    policy, or the true future counts for the oracle. Ordering and action choice both
    follow the belief; only the bill is real.
    """
    sim = sim or TieringSimulator()
    w = weights or DEFAULT_WEIGHTS
    st = max(1, wl.time_split(split_frac))
    future = wl.future_counts(st)
    caps = sim.tier_capacities(wl.total_bytes)
    remaining = {t: caps[t] for t in _TIERS}
    total = 0.0
    for it in sorted(wl.items, key=lambda i: believed.get(i.item_id, 0.0), reverse=True):
        n_hat = believed.get(it.item_id, 0.0)
        n_true = future.get(it.item_id, 0)
        # Ties are broken by TIER SPEED, not by enumeration order. At n_hat = 0 every action
        # costs exactly 0, so `if c < best_c` kept whichever the enum happened to list first.
        # Tier enumerates GPU, CPU, DISK, REMOTE_RDMA -- so once the two fast tiers filled,
        # every zero-belief block landed on DISK (12 ms) while REMOTE_RDMA (6 ms) sat empty.
        # Adding 1e-9 to every belief -- an operation that cannot change any ordering or any
        # real decision -- moved the frequency baseline by 36%. That is larger than any effect
        # this file is used to measure.
        best, best_key = None, None
        for a in LEGAL_ACTIONS:
            if remaining[a.tier] < it.size_bytes:
                continue
            c = weighted_cost(sim.cost_of_placement(a, n_hat, it.representation), w)
            key = (c, sim.read_latency(a.tier))
            if best_key is None or key < best_key:
                best, best_key = a, key
        if best is None:
            continue
        remaining[best.tier] -= it.size_bytes
        total += weighted_cost(sim.cost_of_placement(best, n_true, it.representation), w)
    return total


def attainable_ceiling(wl: Workload, scored_ids, sim: TieringSimulator | None = None,
                       weights: RewardWeights | None = None, split_frac: float = 0.5) -> float:
    """Cost of the best placement a CAUSAL policy could reach, given what it can see.

    The hindsight oracle ranks every block in the trace, including blocks that do not exist
    yet at the split instant. A model cannot: feature construction can only describe blocks
    with at least one pre-split access. Scoring a policy against an oracle that ranks the
    unseeable makes the denominator mostly unreachable -- on our traces 97% of it -- and any
    policy then looks flat no matter how well it ranks what it CAN see.

    This is the clairvoyant belief restricted to `scored_ids`: perfect knowledge of the
    future for every block the policy could have had an opinion about, and nothing for the
    rest. It is the correct ceiling for `captured`.
    """
    sim = sim or TieringSimulator()
    st = max(1, wl.time_split(split_frac))
    fut = wl.future_counts(st)
    ids = set(scored_ids)
    belief = {i.item_id: (float(fut.get(i.item_id, 0)) if i.item_id in ids else 0.0)
              for i in wl.items}
    return place_with_beliefs(wl, belief, sim, weights, split_frac)


def captured_fractions(wl: Workload, belief: dict[str, float], scored_ids,
                       sim: TieringSimulator | None = None,
                       weights: RewardWeights | None = None,
                       split_frac: float = 0.5) -> dict[str, float]:
    """Both normalizations of `captured`, so neither can be quoted without the other.

    `of_full_gap` is what the paper reported: the fraction of the hindsight oracle's total
    advantage recovered. `of_attainable` divides instead by what a causal policy could
    reach. They differ by more than an order of magnitude here, and only the second is a
    statement about the policy rather than about the protocol.
    """
    sim = sim or TieringSimulator()
    c_pref = prefix_greedy_cost(wl, sim, weights)
    c_orac = place_with_beliefs(
        wl, {i.item_id: float(wl.future_counts(max(1, wl.time_split(split_frac)))
                              .get(i.item_id, 0)) for i in wl.items},
        sim, weights, split_frac)
    c_ceil = attainable_ceiling(wl, scored_ids, sim, weights, split_frac)
    c_pol = place_with_beliefs(wl, belief, sim, weights, split_frac)
    full = c_pref - c_orac
    reach = c_pref - c_ceil
    return {"cost_policy": c_pol, "cost_prefix": c_pref, "cost_oracle": c_orac,
            "cost_ceiling": c_ceil,
            "of_full_gap": 100.0 * (c_pref - c_pol) / full if full > 0 else 0.0,
            "of_attainable": 100.0 * (c_pref - c_pol) / reach if reach > 0 else 0.0,
            "ceiling_share_of_full_gap": 100.0 * reach / full if full > 0 else 0.0}


def quantile_match(scores: list[float], reference: list[float]) -> list[float]:
    """Re-express `scores` on `reference`'s magnitude scale, preserving their order exactly.

    Needed because `place_with_beliefs` is NOT ordering-invariant: the per-item argmin has
    thresholds in the belief (around 7.4e-4, 0.079 and 0.41 under our tables), so two
    policies with identical rankings but different scales choose different representations
    and score differently. Comparing model rungs therefore requires putting every belief on
    one common magnitude multiset; otherwise the ladder measures the min-max transform as
    much as the model.

    The i-th smallest score receives the i-th smallest reference value. TIED scores all
    receive the MEAN of the values their rank block would have taken, which matters more
    than it sounds: with a plain stable sort a constant score vector becomes a strictly
    increasing belief in input order, so a "no information" control silently acquires a
    systematic ranking and can beat a real model. That happened here -- the constant arm
    scored above the learned one -- and it is the same mid-rank tie handling this project
    already needed once, in the AUC computation.
    """
    if len(scores) != len(reference):
        raise ValueError(f"length mismatch: {len(scores)} scores, {len(reference)} reference")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ref = sorted(reference)
    out = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        share = sum(ref[i:j + 1]) / (j + 1 - i)      # mid-rank magnitude for the tie block
        for k in range(i, j + 1):
            out[order[k]] = share
        i = j + 1
    return out


def prefix_greedy_cost(wl: Workload, sim: TieringSimulator | None = None,
                       weights: RewardWeights | None = None, split_frac: float = 0.5) -> float:
    """THE ABLATION THAT MATTERS: greedy placement driven by the OBSERVED prefix count,
    scored against the true future cost.

    This is what a system can do with no learning at all — assume the future looks like
    the past and place greedily. If a learned policy cannot beat this, the learning adds
    nothing on this workload, and we should say so.
    """
    sim = sim or TieringSimulator()
    w = weights or DEFAULT_WEIGHTS
    g = _prepare(wl, sim, w, split_frac)                     # true-future costs, prefix order
    st = max(1, wl.time_split(split_frac))
    stats = wl.prefix_stats(st)
    items = sorted(wl.items, key=lambda it: stats[it.item_id]["count"], reverse=True)
    index = {it.item_id: i for i, it in enumerate(items)}    # _prepare uses the same order
    # Greedy under the *believed* counts (prefix), paying the *true* (future) cost.
    believed = torch.zeros(len(items), len(LEGAL_ACTIONS))
    for it in items:
        i = index[it.item_id]
        n_hat = int(stats[it.item_id]["count"])
        for j, act in enumerate(LEGAL_ACTIONS):
            believed[i, j] = weighted_cost(sim.cost_of_placement(act, n_hat, it.representation), w)
    # Tie-break by tier speed, exactly as `place_with_beliefs` does. `argmin` returns the
    # FIRST minimum, and LEGAL_ACTIONS inherits Tier's declaration order (GPU, CPU, DISK,
    # REMOTE_RDMA), so at a believed count of zero -- where every action costs the same --
    # this routine parked blocks on DISK while the faster remote tier sat empty. Fixing only
    # `place_with_beliefs` left the two baselines disagreeing by a wide margin, which is the
    # frame mismatch this project keeps finding: two code paths that must agree, silently
    # not agreeing.
    tier_lat = torch.tensor(
        [sim.read_latency(act.tier) for act in LEGAL_ACTIONS], dtype=torch.float32)
    remaining = g.cap.clone()
    total = 0.0
    for i in range(len(items)):
        afford = _affordable(g, i, remaining) & torch.isfinite(believed[i])
        idx = afford.nonzero(as_tuple=True)[0]
        vals = believed[i, idx]
        best = vals.min()
        tied = idx[(vals <= best + 1e-12).nonzero(as_tuple=True)[0]]
        a = tied[tier_lat[tied].argmin()]                    # cheapest, then fastest tier
        total += g.cost[i, a].item()                         # pay the truth
        remaining = remaining.clone()
        remaining[_ACTION_TIER[a]] -= g.sizes[i]
    return total


def reference_costs(wl: Workload, sim: TieringSimulator | None = None,
                    weights: RewardWeights | None = None, seed: int = 0) -> dict[str, float]:
    """Reference points, all scored on the FUTURE portion of the trace.

    `oracle_future` ranks and chooses with hindsight (the true upper bound);
    `prefix_greedy` is the no-learning baseline that assumes the future looks like the
    observed past. Their difference is the HEADROOM any predictor can compete for --
    ~0 on stationary workloads, large under non-stationarity (phase_shifts>0).
    """
    sim = sim or TieringSimulator()
    g_pref = _prepare(wl, sim, weights or DEFAULT_WEIGHTS)
    g_orac = _prepare(wl, sim, weights or DEFAULT_WEIGHTS, order_by="future")
    pg = prefix_greedy_cost(wl, sim, weights)
    orac = _oracle_cost(g_orac, "greedy")
    return {"oracle_future": orac,
            "greedy_feasible": orac,                     # back-compat alias
            "random_feasible": _oracle_cost(g_pref, "random", seed),
            "prefix_greedy": pg,
            "headroom_pct": 100.0 * (pg - orac) / pg if pg > 0 else 0.0}
