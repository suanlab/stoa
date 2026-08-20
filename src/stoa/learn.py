"""Learned reuse-distance predictor (docs/research_plan.md §10, RQ2; M6-9).

PARROT (arXiv:2006.16239) showed cache replacement can be learned by *imitating*
the Belady oracle. This module does the torch-free version: fit a linear model
that predicts an item's next-use distance from online-observable features
(recency, frequency, age, inter-arrival), trained on a trace where the true
next-use is known offline. Online, the tier controller evicts the item with the
largest *predicted* next use — approximating Belady without future knowledge.

This isolates the RQ2 claim (imitation warm-start closes the gap to the oracle)
in a small, reproducible, dependency-free form. TODO(M6-9): swap the linear model
for the GNN actor-critic + offline-RL fine-tuning once torch/torch-geometric land;
the feature/label interface here is the warm-start target the RL policy inherits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .traces import Workload

N_FEATURES = 6   # bias + recency, freq, age, mean_interarrival, inv_recency


def features(t: int, last_used: int | None, freq: int, first_seen: int | None) -> list[float]:
    """Online feature vector for an item as of step `t` (no future knowledge)."""
    recency = float(t - last_used) if last_used is not None else float(t + 1)
    age = float(t - first_seen) if first_seen is not None else 0.0
    mean_ia = age / freq if freq > 0 else age
    return [
        1.0,                       # bias
        recency,
        float(freq),
        age,
        mean_ia,
        1.0 / (1.0 + recency),     # inverse recency (LRU-like signal)
    ]


def _solve(a: list[list[float]], b: list[float], ridge: float = 1e-3) -> list[float]:
    """Ridge-regularized Gaussian elimination for a small dense system (A w = b)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        m[i][i] += ridge
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


@dataclass
class ReusePredictor:
    """Linear model predicting log(1 + next-use distance). Higher score = evict."""
    weights: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    mean: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    std: list[float] = field(default_factory=lambda: [1.0] * N_FEATURES)

    def _standardize(self, x: list[float]) -> list[float]:
        return [1.0 if i == 0 else (x[i] - self.mean[i]) / self.std[i] for i in range(N_FEATURES)]

    def score(self, x: list[float]) -> float:
        z = self._standardize(x)
        return sum(w * zi for w, zi in zip(self.weights, z))

    def to_dict(self) -> dict:
        return {"weights": self.weights, "mean": self.mean, "std": self.std}


def _collect(workload: Workload) -> tuple[list[list[float]], list[float]]:
    """Replay the trace; for each access emit (features-as-of-t, log1p next-use)."""
    idxs: dict[str, list[int]] = {}
    for gidx, (_s, item) in enumerate(workload.accesses):
        idxs.setdefault(item, []).append(gidx)
    ptr: dict[str, int] = {i: 0 for i in idxs}
    last_used: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    freq: dict[str, int] = {i: 0 for i in idxs}
    horizon = len(workload.accesses)

    X: list[list[float]] = []
    y: list[float] = []
    for gidx, (_s, item) in enumerate(workload.accesses):
        x = features(gidx, last_used.get(item), freq[item], first_seen.get(item))
        ptr[item] += 1
        nxt = idxs[item][ptr[item]] if ptr[item] < len(idxs[item]) else horizon + gidx
        X.append(x)
        y.append(math.log1p(nxt - gidx))
        freq[item] += 1
        first_seen.setdefault(item, gidx)
        last_used[item] = gidx
    return X, y


def fit_reuse_predictor(workload: Workload) -> ReusePredictor:
    """Least-squares fit of the reuse predictor on a training trace (imitation)."""
    X, y = _collect(workload)
    n = len(X)
    mean = [0.0] * N_FEATURES
    std = [1.0] * N_FEATURES
    for j in range(1, N_FEATURES):
        col = [X[i][j] for i in range(n)]
        mu = sum(col) / n
        var = sum((v - mu) ** 2 for v in col) / n
        mean[j] = mu
        std[j] = math.sqrt(var) or 1.0

    def z(i: int) -> list[float]:
        return [1.0 if j == 0 else (X[i][j] - mean[j]) / std[j] for j in range(N_FEATURES)]

    ata = [[0.0] * N_FEATURES for _ in range(N_FEATURES)]
    aty = [0.0] * N_FEATURES
    for i in range(n):
        zi = z(i)
        yi = y[i]
        for r in range(N_FEATURES):
            aty[r] += zi[r] * yi
            for c in range(N_FEATURES):
                ata[r][c] += zi[r] * zi[c]
    weights = _solve(ata, aty)
    return ReusePredictor(weights=weights, mean=mean, std=std)
