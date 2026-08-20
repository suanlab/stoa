"""Mooncake production-trace adapter (Qin et al., FAST'25, arXiv:2407.00079).

The M0-3 calibration gate needs REAL access patterns; our synthetic generator turned
out to produce only degenerate regimes (see docs/research_plan.md). Mooncake publishes
replayed production traces from the Kimi serving stack as JSONL:

    {"timestamp": ..., "input_length": ..., "output_length": ..., "hash_ids": [...]}

`hash_ids` are KV-block identifiers, so a repeated hash is a genuine cache reuse --
exactly the memory-item access stream STOA schedules. The `toolagent` trace is agent
workload, i.e. STOA's target domain.

Structure differs sharply from our synthetic Zipf: ~3/4 of blocks are touched exactly
once while a few shared prefixes are touched thousands of times. Deciding "will this
block ever be reused?" is a real prediction problem -- the one PARROT-style learning
addresses and the one our generator lacked.

Download (no GPU, ~4 MB):
    curl -sL -o data/mooncake_toolagent_trace.jsonl \\
      https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .credit import AccessTrace
from .environment import ItemState, Representation, Tier
from .traces import Workload

BLOCK_TOKENS = 256          # Mooncake hashes one block per 256 tokens (paper §3)
BYTES_PER_TOKEN = 2 * 2 * 32 * 128 * 2 // 1024   # illustrative KV bytes/token; calibrate in M0-3


def load_mooncake(path: str, max_requests: int | None = None,
                  min_accesses: int = 1, with_wallclock: bool = False):
    """Load a Mooncake JSONL trace as a STOA `Workload`.

    Each request contributes one access per `hash_id`; the request index is the
    timestep. `min_accesses` filters ultra-cold blocks to bound problem size (report
    it when you use it -- dropping one-timers removes the hardest prediction cases).

    `with_wallclock=True` additionally returns `{"wall": {timestep: ms}}`.

    NOTE (2026-08-13): this loader previously ignored the `timestamp` field entirely,
    which led us to state in a draft that Mooncake "omits request timing". It does not:
    both released traces carry timestamps spanning ~59 minutes at ~3 s resolution
    (1,180 distinct values). Any claim about timing availability must use this field.
    """
    steps: dict[int, list[int]] = defaultdict(list)
    wall: dict[int, float] = {}
    t = 0
    with Path(path).open() as fh:
        for line in fh:
            if max_requests is not None and t >= max_requests:
                break
            rec = json.loads(line)
            wall[t] = float(rec.get("timestamp", t))      # real arrival time, was discarded
            for h in rec.get("hash_ids", []):
                steps[h].append(t)
            t += 1

    keep = {h: s for h, s in steps.items() if len(s) >= min_accesses}
    horizon = t
    items = [
        ItemState(
            item_id=f"b{h}",
            access_freq=len(s) / horizon if horizon else 0.0,
            size_bytes=BLOCK_TOKENS * BYTES_PER_TOKEN,
            tier=Tier.CPU,
            representation=Representation.PLAINTEXT,
        )
        for h, s in keep.items()
    ]
    accesses = sorted(((step, f"b{h}") for h, ss in keep.items() for step in ss),
                      key=lambda x: x[0])
    traces = {f"b{h}": AccessTrace(item_id=f"b{h}", access_steps=s) for h, s in keep.items()}
    wl = Workload(items=items, accesses=accesses, traces=traces)
    return (wl, {"wall": wall}) if with_wallclock else wl


def trace_summary(wl: Workload) -> dict:
    """Descriptive stats a calibration report should cite."""
    reuse = [len(t.access_steps) for t in wl.traces.values()]
    n = len(reuse) or 1
    return {
        "items": len(reuse),
        "accesses": sum(reuse),
        "one_time_frac": round(sum(1 for r in reuse if r == 1) / n, 4),
        "mean_reuse": round(sum(reuse) / n, 3),
        "max_reuse": max(reuse) if reuse else 0,
    }
