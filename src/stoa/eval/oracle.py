"""Oracle upper bound + reference baselines (docs/research_plan.md §11.2, §13; M0-3).

Establishes the offline reference points every learned policy is measured against:

  * oracle_upper_bound  — sum of per-item unconstrained Belady optima. Ignores
                          tier capacity, so it is a valid UPPER BOUND on reward
                          (lower bound on cost); generally infeasible.
  * greedy_belady       — capacity-feasible greedy allocation in reuse-benefit
                          order. A strong ACHIEVABLE reference (the tiering an
                          idealized-but-feasible controller would produce).
  * naive_baseline      — everything plaintext / CPU / noop (capacity-ignoring).

Invariants (asserted in tests): oracle_ub_cost <= greedy_cost and
oracle_ub_cost <= naive_cost, item-wise, because each item's greedy/naive cost is
>= its unconstrained minimum.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from ..credit import belady_optimal_action, weighted_cost
from ..environment import Action, Representation, RewardWeights, Tier, Timing
from ..simulator import TieringSimulator
from ..traces import Workload, WorkloadConfig, generate_workload

# Unit-normalized duals so the three cost terms are comparable (ms, USD, tokens
# live on very different scales). These are illustrative init values for the
# offline baseline; the learned policy treats (L, C, B) as its control knob.
DEFAULT_WEIGHTS = RewardWeights(
    lambda_latency=0.1,     # per ms
    lambda_cost=1.0,        # per USD
    lambda_tokens=0.0005,   # per token
    freshness_bonus=0.0,
)

NAIVE_ACTION = Action(Representation.PLAINTEXT, Tier.CPU, Timing.NOOP)


@dataclass
class BaselineResult:
    name: str
    total_cost: float
    tier_hist: dict[str, int] = field(default_factory=dict)
    repr_hist: dict[str, int] = field(default_factory=dict)


@dataclass
class OracleReport:
    config: dict
    n_items: int
    horizon: int
    total_accesses: int
    oracle_upper_bound: BaselineResult
    greedy_belady: BaselineResult
    naive: BaselineResult

    @property
    def greedy_gap_pct(self) -> float:
        """How far the feasible greedy sits above the (infeasible) oracle ceiling."""
        ub = self.oracle_upper_bound.total_cost
        return 100.0 * (self.greedy_belady.total_cost - ub) / ub if ub > 0 else 0.0

    @property
    def greedy_savings_vs_naive_pct(self) -> float:
        naive = self.naive.total_cost
        return 100.0 * (naive - self.greedy_belady.total_cost) / naive if naive > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["greedy_gap_pct"] = round(self.greedy_gap_pct, 3)
        d["greedy_savings_vs_naive_pct"] = round(self.greedy_savings_vs_naive_pct, 3)
        return d


def _hist(actions_by_item: dict[str, Action]) -> tuple[dict[str, int], dict[str, int]]:
    tiers = Counter(a.tier.value for a in actions_by_item.values())
    reprs = Counter(a.representation.value for a in actions_by_item.values())
    return dict(tiers), dict(reprs)


def _oracle_upper_bound(wl: Workload, sim: TieringSimulator, w: RewardWeights) -> BaselineResult:
    actions: dict[str, Action] = {}
    total = 0.0
    for item in wl.items:
        action, cost = belady_optimal_action(wl.n_accesses(item.item_id), sim, w, item.representation)
        actions[item.item_id] = action
        total += cost
    th, rh = _hist(actions)
    return BaselineResult("oracle_upper_bound", total, th, rh)


def _naive(wl: Workload, sim: TieringSimulator, w: RewardWeights) -> BaselineResult:
    actions: dict[str, Action] = {}
    total = 0.0
    for item in wl.items:
        n = wl.n_accesses(item.item_id)
        total += weighted_cost(sim.cost_of_placement(NAIVE_ACTION, n, item.representation), w)
        actions[item.item_id] = NAIVE_ACTION
    th, rh = _hist(actions)
    return BaselineResult("naive", total, th, rh)


def greedy_belady_actions(wl: Workload, sim: TieringSimulator, w: RewardWeights,
                          counts: dict[str, int] | None = None) -> dict[str, Action]:
    """Per-item capacity-feasible Belady placement — the non-degenerate imitation
    target for the GNN warm-start (train.py). Unlike the unconstrained per-item
    optimum (which collapses to all-latent), this yields a real tier/repr spread.

    `counts` supplies the per-item access counts the oracle scores against; pass
    `wl.future_counts(split_step)` to make the oracle score only the FUTURE (the
    honest label when features come from the observable prefix). Defaults to the
    whole trace.
    """
    counts = counts or {it.item_id: wl.n_accesses(it.item_id) for it in wl.items}
    remaining = sim.tier_capacities(wl.total_bytes)
    ranked: dict[str, list[tuple[float, Action]]] = {}
    benefit: dict[str, float] = {}
    for item in wl.items:
        n = counts.get(item.item_id, 0)
        opts = sorted(
            ((weighted_cost(sim.cost_of_placement(a, n, item.representation), w), a) for a in Action.space()),
            key=lambda x: x[0],
        )
        ranked[item.item_id] = opts
        naive_cost = weighted_cost(sim.cost_of_placement(NAIVE_ACTION, n, item.representation), w)
        benefit[item.item_id] = naive_cost - opts[0][0]
    order = sorted(wl.items, key=lambda it: benefit[it.item_id], reverse=True)
    size = {it.item_id: it.size_bytes for it in wl.items}
    actions: dict[str, Action] = {}
    for item in order:
        for _cost, action in ranked[item.item_id]:
            if remaining[action.tier] >= size[item.item_id]:
                remaining[action.tier] -= size[item.item_id]
                actions[item.item_id] = action
                break
        else:
            actions[item.item_id] = ranked[item.item_id][-1][1]
    return actions


def _greedy_belady(wl: Workload, sim: TieringSimulator, w: RewardWeights) -> BaselineResult:
    """Capacity-feasible allocation: place high-reuse-benefit items into scarce
    hot tiers first, overflow to colder tiers (Belady admission)."""
    actions = greedy_belady_actions(wl, sim, w)
    total = 0.0
    for item in wl.items:
        n = wl.n_accesses(item.item_id)
        total += weighted_cost(sim.cost_of_placement(actions[item.item_id], n, item.representation), w)
    th, rh = _hist(actions)
    return BaselineResult("greedy_belady", total, th, rh)


def run(config: WorkloadConfig | None = None, weights: RewardWeights | None = None) -> OracleReport:
    """Generate a workload and compute the three offline reference points."""
    cfg = config or WorkloadConfig()
    w = weights or DEFAULT_WEIGHTS
    wl = generate_workload(cfg)
    sim = TieringSimulator(horizon=cfg.horizon)
    return OracleReport(
        config={"n_items": cfg.n_items, "horizon": cfg.horizon, "zipf_s": cfg.zipf_s, "seed": cfg.seed},
        n_items=cfg.n_items,
        horizon=cfg.horizon,
        total_accesses=len(wl.accesses),
        oracle_upper_bound=_oracle_upper_bound(wl, sim, w),
        greedy_belady=_greedy_belady(wl, sim, w),
        naive=_naive(wl, sim, w),
    )
