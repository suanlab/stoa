"""Correctness of the hand-rolled GBDT (src/stoa/gbdt.py).

This model exists to make one argument in the paper: that the failure to convert ranking
quality into placement benefit is not a symptom of using too weak a model. That argument
is worthless if the "stronger" model is quietly broken -- a GBDT that silently underfits
would manufacture exactly the result we want. So the tests below pin the property the
argument rests on: it must fit structure a linear model provably cannot.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from stoa.gbdt import GBDT  # noqa: E402


def _r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _linear_ref(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ w


@pytest.fixture
def xor_data():
    """A multiplicative interaction plus a sinusoid: zero linear signal by construction."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    y = (X[:, 0] * X[:, 1] > 0).astype(float) + 0.3 * np.sin(3 * X[:, 2])
    y += 0.05 * rng.normal(size=4000)
    return X[:3000], y[:3000], X[3000:], y[3000:]


def test_beats_linear_on_structure_linear_cannot_represent(xor_data):
    Xtr, ytr, Xte, yte = xor_data
    gb = GBDT(n_trees=150, lr=0.1, max_depth=4).fit(Xtr, ytr)
    r2_gb = _r2(yte, gb.predict(Xte))
    r2_lin = _r2(yte, _linear_ref(Xtr, ytr, Xte))
    assert r2_gb > 0.5, f"GBDT underfit (R2={r2_gb:.3f}) — the capacity argument needs a real fit"
    assert r2_gb > r2_lin + 0.3, f"GBDT {r2_gb:.3f} did not clearly beat linear {r2_lin:.3f}"


def test_more_trees_never_hurts_training_fit(xor_data):
    """Boosting is monotone on TRAINING loss by construction; a violation means the residual
    update or the leaf values are wrong."""
    Xtr, ytr, _, _ = xor_data
    losses = []
    for n in (10, 50, 150):
        p = GBDT(n_trees=n, lr=0.1, max_depth=4).fit(Xtr, ytr).predict(Xtr)
        losses.append(float(((ytr - p) ** 2).mean()))
    assert losses == sorted(losses, reverse=True), losses


def test_is_deterministic():
    """No subsampling and no random feature selection, so two fits must agree exactly.
    Reproducibility is a claim the paper makes about every artifact."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(800, 4)), rng.normal(size=800)
    a = GBDT(n_trees=30, max_depth=3).fit(X, y).predict(X)
    b = GBDT(n_trees=30, max_depth=3).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_predicts_the_mean_with_no_trees():
    X, y = np.zeros((50, 3)), np.arange(50, dtype=float)
    gb = GBDT(n_trees=0).fit(X, y)
    assert gb.predict(X) == pytest.approx(np.full(50, y.mean()))


def test_respects_min_samples_leaf():
    """A leaf below the floor would let the model memorize single rows, which is how a
    'higher-capacity' rung turns into pure overfitting rather than a fair comparison."""
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(500, 3)), rng.normal(size=500)
    gb = GBDT(n_trees=5, max_depth=8, min_samples_leaf=50).fit(X, y)

    def leaf_sizes(node, idx):
        if node.is_leaf:
            return [idx.size]
        mask = X[idx, node.feat] <= node.thresh
        return leaf_sizes(node.left, idx[mask]) + leaf_sizes(node.right, idx[~mask])

    for tree in gb.trees:
        sizes = leaf_sizes(tree, np.arange(len(X)))
        assert min(sizes) >= 50, f"leaf of size {min(sizes)} < min_samples_leaf"


def test_constant_target_gives_constant_prediction():
    X = np.random.default_rng(3).normal(size=(300, 4))
    y = np.full(300, 7.0)
    p = GBDT(n_trees=20, max_depth=3).fit(X, y).predict(X)
    assert p == pytest.approx(np.full(300, 7.0), abs=1e-9)
