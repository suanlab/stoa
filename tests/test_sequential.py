"""Sequential occupancy-aware placement (§6.1). torch-gated."""
import statistics

import pytest

torch = pytest.importorskip("torch")

from stoa.sequential import (  # noqa: E402
    _affordable,
    _prepare,
    evaluate_seq,
    reference_costs,
    train_seq,
)
from stoa.simulator import TieringSimulator  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402


def _wl(seed=0, n=48):
    return generate_workload(WorkloadConfig(n_items=n, horizon=4_000, zipf_s=1.1, seed=seed))


def test_a_finite_affordable_action_always_exists():
    """Feasibility by construction: every item has >=1 affordable finite-cost action."""
    g = _prepare(_wl(), TieringSimulator(), DEFAULT_WEIGHTS)
    remaining = torch.zeros_like(g.cap)              # worst case: all finite tiers full
    remaining[-1] = g.cap[-1]                        # REMOTE unbounded
    for i in range(g.feat.shape[0]):
        assert _affordable(g, i, remaining).any()


def test_greedy_beats_random_reference():
    r = reference_costs(_wl(0))
    assert r["greedy_feasible"] < r["random_feasible"]


def test_sequential_rl_learns_below_random():
    """After training, the greedy rollout costs far less than random-feasible."""
    train = [_wl(100), _wl(101)]
    test = [_wl(0), _wl(1)]
    policy, _ = train_seq(train, steps=200, seed=1)
    seq = statistics.mean(evaluate_seq(policy, w) for w in test)
    rand = statistics.mean(reference_costs(w)["random_feasible"] for w in test)
    greedy = statistics.mean(reference_costs(w)["greedy_feasible"] for w in test)
    assert seq < rand                                # learned policy beats random-feasible
    assert seq <= greedy * 1.5                       # and lands near the greedy oracle


def test_train_seq_reproducible():
    _, r1 = train_seq([_wl(100)], steps=40, seed=3)
    _, r2 = train_seq([_wl(100)], steps=40, seed=3)
    assert r1.final_cost == r2.final_cost
