"""Causal versions of the quantities the arrival-time and belief experiments need.

Why this module exists (docs/claims_dependency.md §AP). An audit of the arrival-time experiment
behind contribution C2 found three ways it used information a deployed policy would not have,
none of which any earlier check could see because each produced a plausible number:

1. **Training labels ran to the end of the trace.** A training block's label counted every later
   access, so blocks arriving just before the train/eval cut were labelled partly by accesses in
   the evaluation period -- 34.3% and 46.3% of all training-label mass on the two traces.
2. **The train/eval cut and the billing split were different instants** (median block arrival vs.
   `Workload.time_split(0.5)`), so the "evaluation" population and the instant costs were billed
   at did not coincide -- the reference-frame mismatch the paper's own protocol section warns of.
3. **Belief magnitudes came from the evaluation labels.** Every policy's scores, FCFS included,
   were quantile-matched onto the multiset of the evaluation blocks' *true* future access counts.
   No block's own label leaked, but the marginal -- how many blocks would be reused, and how
   much -- did.

The functions here replace each with the quantity observable at decision time. They are separate
from `sequential` so that adding them does not perturb the provenance of artifacts that never
used them.

Boundary convention, matching `Workload.prefix_stats` (s < split) and `future_counts` (s >= split):
an access at step `split` belongs to the evaluation period.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

__all__ = ["split_by_arrival", "windowed_training_labels", "resample_to",
           "analogue_magnitudes"]


def split_by_arrival(arrival: Sequence[int], split: int) -> tuple[list[int], list[int]]:
    """Indices of blocks first seen before `split` (train) and at or after it (eval).

    The cut is the *same* instant costs are billed at, so the evaluation population is exactly the
    set of blocks with no pre-split access -- the population §5.2 shows carries most of the
    oracle's advantage.
    """
    tr = [i for i, t in enumerate(arrival) if t < split]
    ev = [i for i, t in enumerate(arrival) if t >= split]
    return tr, ev


def windowed_training_labels(arrival: Sequence[int], ids: Sequence[str],
                             steps_of: Mapping[str, Sequence[int]], split: int,
                             window: int) -> tuple[list[int], list[float]]:
    """Training rows whose label is fully observed before `split`.

    A block arriving at `t` is labelled with its accesses in `(t, t + window]`, and is used only if
    `t + window < split`, so every access its label counts happened before the evaluation period
    began. A fixed window keeps the label's meaning the same for every training row; labelling each
    block "until the split" instead would give late arrivals systematically shorter windows.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    idx, labels = [], []
    for i, (t, b) in enumerate(zip(arrival, ids)):
        if t + window >= split:
            continue
        hi = t + window
        labels.append(float(sum(1 for s in steps_of[b] if t < s <= hi)))
        idx.append(i)
    return idx, labels


def resample_to(values: Iterable[float], n: int) -> list[float]:
    """`n` values whose empirical distribution matches `values` (by evenly spaced quantiles)."""
    v = sorted(float(x) for x in values)
    if n <= 0:
        return []
    if not v:
        raise ValueError("cannot resample an empty reference distribution")
    m = len(v)
    return [v[min(m - 1, int((k + 0.5) * m / n))] for k in range(n)]


def analogue_magnitudes(prefix_counts: Iterable[int], n: int) -> list[float]:
    """The magnitude multiset a causal policy can use for blocks that are new at the split.

    Evaluation blocks are those first seen at or after the split, and their true magnitude is their
    access count from first appearance to the end of the trace. The analogue observable at the
    split is the previous period's equivalent: blocks first seen before it, counted up to it --
    which is exactly their prefix count. `time_split(0.5)` makes the two periods equal in length,
    so no rescaling is applied.
    """
    seen = [c for c in prefix_counts if c > 0]
    return resample_to(seen, n)
