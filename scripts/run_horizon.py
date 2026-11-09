#!/usr/bin/env python3
"""H6: is "dead" an artifact of where the trace ends? (§BF; ADVISE H6)

A visible block is dead when it is not read again before the trace ends, so the dead share is
right-censored by the trace's length (Mooncake's traces are about an hour). This script fixes the split
at a quarter of the trace (the most future the traces allow) and cuts the evaluation window to a
wall-clock horizon of 7.5, 15, 30 and 45 minutes after it, then reports at each horizon the dead share of
visible blocks and the ladder (the clairvoyant reference under both tie rules, the seen-once rule),
tier-only. Capacities follow each truncated trace's distinct blocks, as a study of that shorter trace
would set them.

Control (§BJ; review 2026-11-09, R2/R4): setting capacities from each truncated trace changes the system
with the window, so the ladder's change with the window mixes time with tier size. `fixed_capacity_rows`
repeats every horizon with the tiers' absolute byte capacities held at the full trace's (fractions
rescaled by full bytes / truncated bytes), and records visible, live visible and new block counts and
the GPU tier's slots.

Pre-registered (ADVISE H6): censoring matters if the dead share at 15 minutes exceeds the share at the
longest horizon by 10 points or more; it does not if they differ by less than 3.
Usage: python3 scripts/run_horizon.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_evidence_ladder import TIER, _rows  # noqa: E402
from stoa import simulator as simmod  # noqa: E402
from stoa.environment import Tier  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.traces import AccessTrace, Workload  # noqa: E402

TRACES = {n: f"data/mooncake_{n}_trace.jsonl" for n in ("conversation", "toolagent", "synthetic")}
SPLIT = 0.25
HORIZONS_MIN = (7.5, 15.0, 30.0, 45.0)


def truncated(wl: Workload, last_step: int) -> Workload:
    acc = [(t, i) for t, i in wl.accesses if t <= last_step]
    keep = {i for _, i in acc}
    tr = {i: AccessTrace(item_id=i, access_steps=[t for t in wl.traces[i].access_steps if t <= last_step])
          for i in keep}
    return Workload(items=[it for it in wl.items if it.item_id in keep], accesses=acc, traces=tr)


def counts(w: Workload, st: int) -> dict:
    fut = w.future_counts(st)
    vis = [i.item_id for i in w.items if w.traces[i.item_id].access_steps[0] < st]
    return {"n_visible": len(vis), "n_live_visible": sum(1 for b in vis if fut.get(b, 0) > 0),
            "n_new": len(w.items) - len(vis)}


def main() -> int:
    rows, fixed_rows = [], []
    base = dict(simmod._TIER_CAPACITY_FRACTION)
    for name, path in TRACES.items():
        wl, meta = load_mooncake(str(ROOT / path), with_wallclock=True)
        wall = meta["wall"]
        st = max(1, wl.time_split(SPLIT))
        t0 = wall[st]
        for h in HORIZONS_MIN:
            last = max((s for s, ms in wall.items() if ms < t0 + h * 60_000), default=st)
            if last <= st:
                continue
            w = truncated(wl, last)
            lo, hi = w.accesses[0][0], w.accesses[-1][0]
            frac = (st - lo + 0.5) / (hi - lo)
            assert w.time_split(frac) == st, "the truncated trace moved the split"
            r = _rows(name, w, (frac,), {"tier": TIER})[0]
            minutes = (wall[last] - t0) / 60_000
            rows.append({"trace": name, "horizon_min": h, "horizon_min_actual": round(minutes, 2),
                         "post_split_requests": last - st, "dead_visible_pct": r["dead_visible_pct"],
                         "captured_pct": r["captured_pct"]})
            print(name, h, round(minutes, 1), r["dead_visible_pct"],
                  {k: r["captured_pct"][k] for k in ("seen-once-last", "visible-dead-first", "visible-dead-last")},
                  flush=True)
            scale = wl.total_bytes / w.total_bytes
            try:
                for t in (Tier.GPU, Tier.CPU, Tier.REMOTE_RDMA):
                    simmod._TIER_CAPACITY_FRACTION[t] = base[t] * scale
                rf = _rows(name, w, (frac,), {"tier": TIER})[0]
            finally:
                simmod._TIER_CAPACITY_FRACTION.clear()
                simmod._TIER_CAPACITY_FRACTION.update(base)
            size = w.items[0].size_bytes
            fixed_rows.append({"trace": name, "horizon_min": h, "horizon_min_actual": round(minutes, 2),
                               "dead_visible_pct": rf["dead_visible_pct"], **counts(w, st),
                               "gpu_slots_fixed": int(base[Tier.GPU] * wl.total_bytes // size),
                               "gpu_slots_per_window": int(base[Tier.GPU] * w.total_bytes // size),
                               "captured_pct": rf["captured_pct"]})
            print("  fixed capacity", {k: rf["captured_pct"][k] for k in ("seen-once-last", "visible-dead-first",
                  "visible-dead-last")}, flush=True)
            if minutes < h - 1:                      # the trace ended before this horizon
                break
    (ROOT / "experiments" / "eval_window.json").write_text(json.dumps(
        {"config": {"split": SPLIT, "horizons_min": list(HORIZONS_MIN), "actions": "tier only",
                    "registered": "censoring matters if dead share @15 min - @longest >= 10 points; "
                                  "it does not if < 3",
                    "fixed_capacity": "absolute tier bytes held at the full trace's"},
         "rows": rows, "fixed_capacity_rows": fixed_rows}, indent=2) + "\n")
    print("wrote experiments/eval_window.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
