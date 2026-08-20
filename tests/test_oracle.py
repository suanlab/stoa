"""M0-3: Belady oracle placement + offline reference baselines."""
from stoa.credit import belady_optimal_action
from stoa.environment import Representation, RewardWeights, Tier
from stoa.eval.oracle import DEFAULT_WEIGHTS, run
from stoa.simulator import TieringSimulator
from stoa.traces import WorkloadConfig


def test_belady_never_reused_goes_cold():
    sim = TieringSimulator()
    action, cost = belady_optimal_action(0, sim, DEFAULT_WEIGHTS)
    assert action.tier is Tier.REMOTE_RDMA
    assert cost == 0.0


def test_belady_hot_item_avoids_plaintext():
    """A heavily reused item should not be left as raw plaintext (token-expensive)."""
    sim = TieringSimulator()
    action, _ = belady_optimal_action(500, sim, DEFAULT_WEIGHTS)
    assert action.representation is not Representation.PLAINTEXT


def test_hotter_item_never_costs_more_than_colder():
    sim = TieringSimulator()
    _, cost_hot = belady_optimal_action(2, sim, DEFAULT_WEIGHTS)
    _, cost_hotter = belady_optimal_action(200, sim, DEFAULT_WEIGHTS)
    assert cost_hotter >= cost_hot  # more accesses -> at least as much serving cost


def test_oracle_is_a_valid_lower_bound_on_cost():
    """Unconstrained oracle <= any feasible/naive assignment (§11.2 invariant)."""
    report = run(WorkloadConfig(n_items=48, horizon=1_500, seed=0))
    ub = report.oracle_upper_bound.total_cost
    assert ub <= report.greedy_belady.total_cost + 1e-6
    assert ub <= report.naive.total_cost + 1e-6


def test_greedy_beats_naive_and_report_is_serializable():
    report = run(WorkloadConfig(n_items=64, horizon=2_000, seed=1))
    # Reuse-aware tiering should cost less than dumping everything on one tier.
    assert report.greedy_belady.total_cost < report.naive.total_cost
    d = report.to_dict()
    assert "greedy_gap_pct" in d and "oracle_upper_bound" in d


def test_dear_tokens_push_to_token_free_representation():
    """A high token dual should push the oracle toward token-free reprs."""
    sim = TieringSimulator()
    dear_tokens = RewardWeights(lambda_latency=0.1, lambda_cost=1.0, lambda_tokens=1.0)
    action, _ = belady_optimal_action(50, sim, dear_tokens)
    assert action.representation in (Representation.LATENT, Representation.PARAMETER)
