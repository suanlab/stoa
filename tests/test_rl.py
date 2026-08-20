"""M6-9/M9-12: offline RL fine-tuning machinery. torch-gated."""
import pytest

torch = pytest.importorskip("torch")

from stoa.gnn import LEGAL_ACTIONS  # noqa: E402
from stoa.environment import Tier  # noqa: E402
from stoa.rl import _prepare, evaluate_placement, finetune, placement_metrics  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402
from stoa.train import train_bc  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402


def _wl(seed=0):
    return generate_workload(WorkloadConfig(n_items=48, horizon=4_000, zipf_s=1.1, seed=seed))


def _first_action_on(tier):
    return next(i for i, a in enumerate(LEGAL_ACTIONS) if a.tier is tier)


def test_unbounded_tier_never_overflows():
    """DISK is the unbounded tier — placing everything there yields zero overflow.

    (This test previously asserted the same of REMOTE_RDMA. The capacity table gave remote
    both lower latency and unbounded size, which strictly dominated disk and made the
    four-tier hierarchy effectively three; the table now trades latency against capacity
    monotonically, so the unbounded tier is disk.)"""
    g = _prepare(_wl(), TieringSimulator(), DEFAULT_WEIGHTS)
    disk = torch.full((g.cost.shape[0],), _first_action_on(Tier.DISK))
    _cost, overflow = placement_metrics(disk, g)
    assert overflow.item() == 0.0


def test_tier_hierarchy_is_monotonic():
    """No tier may be dominated: faster must not also mean larger."""
    from stoa.simulator import _TIER_CAPACITY_FRACTION, _TIER_READ_MS
    order = sorted(_TIER_READ_MS, key=lambda t: _TIER_READ_MS[t])
    caps = [_TIER_CAPACITY_FRACTION[t] for t in order]
    assert caps == sorted(caps), f"latency order {[t.value for t in order]} must imply capacity order"


def test_scarce_gpu_tier_overflows_when_everything_lands_there():
    g = _prepare(_wl(), TieringSimulator(), DEFAULT_WEIGHTS)
    gpu = torch.full((g.cost.shape[0],), _first_action_on(Tier.GPU))
    _cost, overflow = placement_metrics(gpu, g)
    assert overflow.item() > 0.0                       # GPU capacity is a small fraction


def test_evaluate_placement_fields_consistent():
    policy, _ = train_bc([_wl(100)], epochs=50)
    ev = evaluate_placement(policy, _wl(0))
    assert ev.cost > 0.0 and ev.overflow_bytes >= 0.0
    assert ev.feasible == (ev.overflow_bytes <= 0.0)


def test_finetune_runs_and_is_reproducible():
    policy, _ = train_bc([_wl(100)], epochs=50)
    import copy
    _, r1 = finetune(copy.deepcopy(policy), [_wl(100)], steps=30, beta=2.0, seed=3)
    _, r2 = finetune(copy.deepcopy(policy), [_wl(100)], steps=30, beta=2.0, seed=3)
    assert r1.final_reward == r2.final_reward           # deterministic given seed
    assert r1.final_reward == r1.final_reward           # finite (not NaN)
