"""Histogram gradient-boosted regression trees (numpy only).

Why this exists rather than an import: the project's runtime dependencies are deliberately
empty and sklearn/LightGBM are not available here, but the paper's central claim -- that
ranking quality does not convert into placement benefit -- is only worth stating if it
survives a change of model class. A linear model failing to convert AUC into placement is
consistent with "the model is too weak"; a gradient-boosted ensemble failing the same way,
at strictly higher AUC, is not. So the ladder needs a genuinely higher-capacity rung, and
this is it.

Implementation is the standard one (Friedman 2001), specialized to squared loss so the
leaf value is the mean residual and no second-order approximation is needed:

    F_0(x) = mean(y);  F_m(x) = F_{m-1}(x) + nu * h_m(x)

where h_m is a depth-limited regression tree fit to the residuals y - F_{m-1}(x). Splits
are chosen over per-feature quantile bins (`n_bins`) rather than over every distinct value,
which is what makes it linear in rows per level instead of quadratic. Deterministic: no
subsampling, no random feature selection, so a rerun reproduces exactly.

Not a general-purpose library -- squared loss only, no missing values, no categorical
handling. It is here to answer one question about model capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    """A binary split on `feat <= threshold`, or a leaf carrying `value`."""
    feat: int = -1
    thresh: float = 0.0
    value: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class GBDT:
    """Squared-loss gradient boosting over depth-limited histogram trees.

    Args mirror the usual knobs: `n_trees` boosting rounds at learning rate `lr`, each tree
    grown to `max_depth` with at least `min_samples_leaf` rows per leaf, splitting over
    `n_bins` quantile bins per feature.
    """
    n_trees: int = 200
    lr: float = 0.1
    max_depth: int = 4
    min_samples_leaf: int = 20
    n_bins: int = 64
    base: float = 0.0
    trees: list[_Node] = field(default_factory=list)
    _edges: np.ndarray | None = None

    # ---- fitting -------------------------------------------------------------

    def _bin_edges(self, X: np.ndarray) -> np.ndarray:
        """Per-feature quantile cut points, shape (n_features, n_bins - 1).

        Quantiles rather than a uniform grid because these features are heavy-tailed
        (log counts, recency in a trace with a long idle tail); a uniform grid would put
        almost every row in one bin and the tree would find no usable split.
        """
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        return np.quantile(X, qs, axis=0).T

    def _best_split(self, X: np.ndarray, g: np.ndarray, idx: np.ndarray):
        """Variance-reduction split over the precomputed bin edges. None if no split helps."""
        best = None
        parent_sum, n = g[idx].sum(), idx.size
        parent_sse = parent_sum * parent_sum / n
        for f in range(X.shape[1]):
            edges = self._edges[f]
            if edges.size == 0:
                continue
            col = X[idx, f]
            order = np.argsort(col, kind="stable")
            cs, cg = col[order], g[idx][order]
            csum = np.cumsum(cg)
            # candidate cut positions = last index whose value <= edge
            pos = np.searchsorted(cs, edges, side="right")
            pos = pos[(pos >= self.min_samples_leaf) & (n - pos >= self.min_samples_leaf)]
            if pos.size == 0:
                continue
            left_sum = csum[pos - 1]
            right_sum = parent_sum - left_sum
            gain = left_sum ** 2 / pos + right_sum ** 2 / (n - pos) - parent_sse
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (float(gain[k]), f, float(cs[pos[k] - 1]))
        return best

    def _grow(self, X: np.ndarray, g: np.ndarray, idx: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(g[idx].mean()))
        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(X, g, idx)
        if split is None or split[0] <= 0.0:
            return node
        _gain, f, thr = split
        mask = X[idx, f] <= thr
        li, ri = idx[mask], idx[~mask]
        if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
            return node
        node.feat, node.thresh = f, thr
        node.left = self._grow(X, g, li, depth + 1)
        node.right = self._grow(X, g, ri, depth + 1)
        return node

    def fit(self, X, y) -> "GBDT":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._edges = self._bin_edges(X)
        self.base = float(y.mean())
        pred = np.full(y.shape, self.base)
        all_idx = np.arange(X.shape[0])
        self.trees = []
        for _ in range(self.n_trees):
            residual = y - pred
            tree = self._grow(X, residual, all_idx, 0)
            self.trees.append(tree)
            pred += self.lr * self._apply(tree, X)
        return self

    # ---- prediction ----------------------------------------------------------

    @staticmethod
    def _apply(tree: _Node, X: np.ndarray) -> np.ndarray:
        """Route every row to its leaf iteratively (recursion would blow the stack on
        large batches, and a per-row Python loop is too slow at 180k blocks)."""
        out = np.empty(X.shape[0])
        stack = [(tree, np.arange(X.shape[0]))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf or idx.size == 0:
                out[idx] = node.value
                continue
            mask = X[idx, node.feat] <= node.thresh
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.base)
        for tree in self.trees:
            pred += self.lr * self._apply(tree, X)
        return pred

    def flatten(self) -> "FlatGBDT":
        """Pack the ensemble into rectangular arrays for fast small-batch prediction.

        `predict` walks one tree at a time, which costs O(trees x depth) Python-level numpy
        calls per invocation. That is fine for a single large batch but ruinous for LRB,
        which predicts on ~64 rows once per eviction and so calls the model hundreds of
        thousands of times. The flat form descends ALL trees simultaneously, so the cost is
        `max_depth` numpy operations per call regardless of ensemble size.
        """
        n_nodes = []
        packed = []
        for tree in self.trees:
            nodes: list[_Node] = []

            def walk(nd: _Node) -> int:
                i = len(nodes)
                nodes.append(nd)
                if not nd.is_leaf:
                    li = walk(nd.left)
                    ri = walk(nd.right)
                    idx[i] = (li, ri)
                return i

            idx: dict[int, tuple[int, int]] = {}
            walk(tree)
            packed.append((nodes, idx))
            n_nodes.append(len(nodes))

        width = max(n_nodes) if n_nodes else 1
        n_t = max(len(self.trees), 1)
        feat = np.full((n_t, width), -1, dtype=np.int64)
        thresh = np.zeros((n_t, width))
        left = np.zeros((n_t, width), dtype=np.int64)
        right = np.zeros((n_t, width), dtype=np.int64)
        value = np.zeros((n_t, width))
        for t, (nodes, idx) in enumerate(packed):
            for i, nd in enumerate(nodes):
                value[t, i] = nd.value
                if nd.is_leaf:
                    continue
                feat[t, i] = nd.feat
                thresh[t, i] = nd.thresh
                left[t, i], right[t, i] = idx[i]
        return FlatGBDT(self.base, self.lr, self.max_depth, feat, thresh, left, right, value)


@dataclass
class FlatGBDT:
    """Read-only, array-packed form of a fitted :class:`GBDT`. See :meth:`GBDT.flatten`."""
    base: float
    lr: float
    max_depth: int
    feat: np.ndarray            # (n_trees, n_nodes); -1 marks a leaf
    thresh: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n, n_t = X.shape[0], self.feat.shape[0]
        node = np.zeros((n, n_t), dtype=np.int64)
        tcol = np.arange(n_t)[None, :]
        for _ in range(self.max_depth + 1):
            f = self.feat[tcol, node]                    # (n, n_trees)
            internal = f >= 0
            if not internal.any():
                break
            # Leaves carry feat == -1; clamp so the gather is legal, then mask the result.
            fx = np.where(internal, f, 0)
            xv = np.take_along_axis(X, fx, axis=1)
            go_left = xv <= self.thresh[tcol, node]
            nxt = np.where(go_left, self.left[tcol, node], self.right[tcol, node])
            node = np.where(internal, nxt, node)
        return self.base + self.lr * self.value[tcol, node].sum(axis=1)
