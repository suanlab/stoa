"""M6-9: GNN factored policy + Belady behavioral cloning (RQ2). torch-gated."""
import pytest

torch = pytest.importorskip("torch")   # skip cleanly when the [rl] extra isn't installed

from stoa.gnn import LEGAL_ACTIONS, GNNPolicy, normalize_adj  # noqa: E402
from stoa.train import FEATURE_DIM, build_adjacency, build_features, evaluate, train_bc  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402


def _wl(seed):
    return generate_workload(WorkloadConfig(n_items=48, horizon=4_000, zipf_s=1.1, seed=seed))


def test_legal_action_set_is_50():
    assert len(LEGAL_ACTIONS) == 50


def test_policy_forward_shapes_and_masking():
    wl = _wl(0)
    x, adj = build_features(wl), normalize_adj(build_adjacency(wl))
    policy = GNNPolicy(FEATURE_DIM, hidden=32, layers=2)
    logits, value = policy(x, adj)
    assert logits.shape == (len(wl.items), 50)      # distribution over legal actions only
    assert value.shape == (len(wl.items),)
    actions = policy.act(x, adj)
    assert len(actions) == len(wl.items)
    assert all(a.is_legal() for a in actions)       # masking holds by construction


def test_bc_training_reduces_loss_and_fits():
    policy, tr = train_bc([_wl(100), _wl(101)], epochs=150)
    assert tr.final_loss < 1.0
    assert tr.train_acc > 0.6                        # clones most of the Belady labels


def test_bc_generalizes_and_closes_gap():
    policy, _ = train_bc([_wl(100), _wl(101), _wl(102)], epochs=200)
    ev = evaluate(policy, _wl(0))                     # held-out trace
    assert ev.imitation_acc > 0.6
    assert ev.bc_cost < ev.naive_cost                # far better than all-plaintext
    assert ev.gap_closed_pct > 80.0                  # closes most of the naive->target gap


def test_bc_is_reproducible():
    p1, r1 = train_bc([_wl(100)], epochs=50, seed=7)
    p2, r2 = train_bc([_wl(100)], epochs=50, seed=7)
    assert abs(r1.final_loss - r2.final_loss) < 1e-6
