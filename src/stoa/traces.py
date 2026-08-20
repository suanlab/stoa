"""Synthetic workload / access-trace generation (docs/research_plan.md §13, M0-3).

Real LMCache-style traces are not yet available, so M0-3 bootstraps the offline
loop on *synthetic* traces with controllable reuse-distance structure. Full future
knowledge lets the Belady oracle (credit.py) compute hindsight-optimal placement,
so the whole pipeline runs end-to-end before real vLLM+LMCache logs arrive (M12-15).

Calibration hook (M0-3 gate): replace the popularity / size draws in
`generate_workload` with distributions fit from real traces; the rest of the
offline loop is unchanged.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .credit import AccessTrace
from .environment import ItemState, Representation, Tier


@dataclass
class WorkloadConfig:
    """Controls the synthetic access distribution.

    `zipf_s` sets popularity skew: 0 = uniform, higher = a few very hot items
    (the regime where reuse-distance-aware tiering pays off).

    `locality_beta` injects temporal locality: a fraction ~beta of accesses are
    re-accesses drawn from a recency window (`recency_window` most-recent items)
    instead of the stationary Zipf popularity. beta=0 is pure stationary Zipf
    (LFU-favorable); beta>0 mixes in a recency signal (LRU-favorable), the regime
    where a learned freq+recency controller beats either heuristic alone.
    """
    n_items: int = 64
    horizon: int = 1_000
    zipf_s: float = 1.1
    min_size_bytes: int = 512
    max_size_bytes: int = 65_536
    seed: int = 0
    locality_beta: float = 0.0
    recency_window: int = 8
    phase_shifts: int = 0
    """Abrupt non-stationarity: reshuffle the popularity ranking `phase_shifts` times.
    Creates large headroom but the reshuffles are random, so the future is genuinely
    UNPREDICTABLE from the past (corr ~0.1) -- the headroom is irreducible noise that
    no predictor can capture. Useful as a control, not as a learning benchmark."""
    drift_rate: float = 0.0
    """STRUCTURED non-stationarity: at each step, with this probability, swap two
    adjacent items in the popularity ranking. Popularity then performs a random walk,
    so *distant* past decays as a predictor while the *recent* past stays informative.
    This is the regime where learning can beat a frequency-only heuristic -- headroom
    and predictability coexist, as in real traces (the reason PARROT/Cold-RL are
    evaluated on real workloads rather than i.i.d. synthetic ones)."""


@dataclass
class Workload:
    """An offline trace: items + the global access sequence + per-item futures."""
    items: list[ItemState]
    accesses: list[tuple[int, str]]        # (step, item_id), ordered by step
    traces: dict[str, AccessTrace]         # item_id -> future access steps

    def n_accesses(self, item_id: str) -> int:
        """Total accesses over the WHOLE trace.

        This is hindsight (future) information. It is legitimate as an oracle
        *label* or upper bound, but must never be fed to a policy as an input
        feature — use `prefix_stats()` for that. See docs/research_plan.md.
        """
        return len(self.traces[item_id].access_steps)

    def time_split(self, frac: float = 0.5) -> int:
        """Timestep at which `frac` of the trace's TIME has elapsed.

        Split by timestep, never by index into `accesses`: one timestep can carry many
        accesses (a real request touches many KV blocks), so the two differ by orders of
        magnitude on real traces and an index-based split can fall past the end of the
        trace, making every future count zero.
        """
        if not self.accesses:
            return 0
        lo = self.accesses[0][0]
        hi = self.accesses[-1][0]
        return int(lo + frac * (hi - lo))

    def prefix_stats(self, split_step: int) -> dict[str, dict[str, float]]:
        """Per-item statistics OBSERVABLE at `split_step` (no future knowledge).

        Returns count/recency/age/mean inter-arrival computed only from accesses
        strictly before `split_step` — the honest inputs for a deployed policy.
        """
        out: dict[str, dict[str, float]] = {}
        for item_id, tr in self.traces.items():
            past = [s for s in tr.access_steps if s < split_step]
            count = len(past)
            last = past[-1] if past else None
            first = past[0] if past else None
            recency = float(split_step - last) if last is not None else float(split_step + 1)
            age = float(split_step - first) if first is not None else 0.0
            out[item_id] = {
                "count": float(count),
                "recency": recency,
                "age": age,
                "mean_interarrival": age / count if count else age,
            }
        return out

    def future_counts(self, split_step: int) -> dict[str, int]:
        """Per-item accesses AFTER `split_step` — the oracle quantity Belady needs."""
        return {i: sum(1 for s in tr.access_steps if s >= split_step)
                for i, tr in self.traces.items()}

    @property
    def total_bytes(self) -> int:
        return sum(it.size_bytes for it in self.items)


def generate_workload(config: WorkloadConfig | None = None) -> Workload:
    """Draw a reproducible synthetic workload.

    Popularity follows a Zipf-like law over a *shuffled* item order (so item
    index is uncorrelated with heat), then `horizon` accesses are sampled i.i.d.
    by popularity. Deterministic given `config.seed`.
    """
    cfg = config or WorkloadConfig()
    rng = random.Random(cfg.seed)

    ids = [f"m{i}" for i in range(cfg.n_items)]
    order = ids[:]
    rng.shuffle(order)                                   # decorrelate id from heat
    weights = [1.0 / ((rank + 1) ** cfg.zipf_s) for rank in range(cfg.n_items)]

    steps_by_item: dict[str, list[int]] = defaultdict(list)
    accesses: list[tuple[int, str]] = []
    recent: list[str] = []                                # rolling recency window
    phase_len = cfg.horizon // cfg.phase_shifts if cfg.phase_shifts > 0 else 0
    for step in range(cfg.horizon):
        if phase_len and step and step % phase_len == 0:
            rng.shuffle(order)                            # abrupt churn (unpredictable)
        if cfg.drift_rate > 0.0 and rng.random() < cfg.drift_rate:
            k = rng.randrange(len(order) - 1)             # gradual drift (predictable short-term)
            order[k], order[k + 1] = order[k + 1], order[k]
        if cfg.locality_beta > 0.0 and recent and rng.random() < cfg.locality_beta:
            item = rng.choice(recent)                     # temporal re-access (recency signal)
        else:
            item = rng.choices(order, weights=weights, k=1)[0]
        accesses.append((step, item))
        steps_by_item[item].append(step)
        if cfg.locality_beta > 0.0:
            recent.append(item)
            if len(recent) > cfg.recency_window:
                recent.pop(0)

    traces = {i: AccessTrace(item_id=i, access_steps=steps_by_item.get(i, [])) for i in ids}
    items = [
        ItemState(
            item_id=i,
            recency=0.0,
            access_freq=len(steps_by_item.get(i, [])) / cfg.horizon,
            size_bytes=rng.randint(cfg.min_size_bytes, cfg.max_size_bytes),
            tier=Tier.CPU,
            representation=Representation.PLAINTEXT,
        )
        for i in ids
    ]
    return Workload(items=items, accesses=accesses, traces=traces)
