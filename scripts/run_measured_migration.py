#!/usr/bin/env python3
"""What a migration costs on a real machine, against the price the online replay assumed (§AX).

The online replay (`run_online_admission.py`, `run_reactive_cost.py`) prices a move between tiers as
one read from the source plus one write to the destination, each at the destination's read latency
("read+write"), and reports the break-even multiple of that price below which a policy keeps 70% of its
static gain (7.3% / 7.2% for the hazard arm). Nothing had measured what a move costs. This script reads
`experiments/tier_costs_a100.json` (raw timings from `measure_tier_costs.py`, three runs on one A100
80GB PCIe Gen4 x16 with a RAID-backed rotational disk) and reports, per migration type:

  alpha_sync   a move on the serving path: its measured time / (measured read_src + read_dst)
  alpha_async  a move overlapped with serving: the demand-load time it displaces (link contention) /
               the same read+write price

Usage: python3 scripts/run_measured_migration.py [--collect run1.json run2.json ...]

Measured read latency per tier is for one 512-token Mooncake block of the 8B shape (64 MiB):
  gpu   half a device-to-device copy (one HBM read of the block; a copy reads and writes it)
  cpu   pinned host -> GPU
  disk  page-cache-evicted file read into pinned host memory and on to the GPU
The remote RDMA tier, to which the hazard arm's online demotions mostly go, was not measured: the
machine has no RDMA fabric. Reported as unmeasured, not imputed.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
BLOCK = 2 * 32 * 8 * 128 * 2 * 512          # llama3-8b-like, 512 tokens = 64 MiB


def med(xs):
    return round(statistics.median(xs), 4)


def collect(paths: list[str]) -> None:
    """Assemble `measure_tier_costs.py` outputs into experiments/tier_costs_a100.json. The raw runs
    are hardware measurements; this step only orders them and replaces the absolute home path in
    each run's --dir with "~"."""
    runs = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        d = r["machine"].get("dir", "")
        r["machine"]["dir"] = "~" + d[d.index("/", len("/home/")):] if d.startswith("/home/") else d
        runs.append(r)
    (EXP / "tier_costs_a100.json").write_text(json.dumps(
        {"producer": "scripts/measure_tier_costs.py (one run per entry, in run order)",
         "runs": runs}, indent=1) + "\n")
    print(f"wrote {EXP / 'tier_costs_a100.json'} ({len(runs)} runs)")


def main() -> int:
    import sys
    if "--collect" in sys.argv:
        collect(sys.argv[sys.argv.index("--collect") + 1:])
    raw = json.loads((EXP / "tier_costs_a100.json").read_text())
    runs = raw["runs"]
    key = str(BLOCK)
    p = lambda path: med([r["paths"][key][path]["median_ms"] for r in runs])          # noqa: E731
    c = lambda k, f: med([r["contention"][k][f] for r in runs])                     # noqa: E731
    read = {"gpu": p("gpu_copy") / 2, "cpu": p("h2d"), "disk": p("disk_read")}
    rw = lambda s, d: read[s] + read[d]                                             # noqa: E731
    sync = {"gpu->cpu": p("d2h"), "cpu->gpu": p("h2d"),
            # disk_write times GPU -> pinned host -> file + fsync; cpu->disk drops the first leg
            "gpu->disk": p("disk_write"), "cpu->disk": p("disk_write") - p("d2h")}
    # GPU<->CPU, from the event-timed runs: K loads and K moves enqueued on two streams from one
    # thread; the price of a move is the extra time the whole batch takes over the loads alone, i.e.
    # the link time it consumes whichever stream the copy engine serves first.
    ev_runs = [r for r in runs if "contention_events" in r]

    def makespan(k):
        return med([(max(r["contention_events"][k]["demand_ms_with_migration"],
                         r["contention_events"][k]["migration_stream_ms"])
                     - r["contention_events"][k]["demand_ms_alone"]) / r["contention_events"][k]["k"]
                    for r in ev_runs])
    asyn = {"gpu->cpu": makespan("h2d_load_vs_demotion"),
            "cpu->gpu": makespan("h2d_load_vs_promotion"),
            "cpu->disk": c("disk_load_vs_demotion", "price_per_migration_ms")}
    # cross-check: the thread-based design (every run), demand time displaced per move
    thread_async = {"gpu->cpu": c("h2d_load_vs_demotion", "price_per_migration_ms"),
                    "cpu->gpu": c("h2d_load_vs_promotion", "price_per_migration_ms")}
    rows = []
    for mt in ("gpu->cpu", "cpu->gpu", "gpu->disk", "cpu->disk"):
        s, d = mt.split("->")
        rows.append({"migration": mt, "read_plus_write_ms": round(rw(s, d), 4),
                     "sync_ms": round(sync[mt], 4), "alpha_sync": round(sync[mt] / rw(s, d), 4),
                     "async_ms": asyn.get(mt), "alpha_async": (round(asyn[mt] / rw(s, d), 4)
                                                              if mt in asyn else None),
                     "async_ms_thread_design": thread_async.get(mt)})
    # spread across runs, as a multiple of the median read+write price of the same move
    def per_run(k, design):
        if design == "events":
            return [(max(r["contention_events"][k]["demand_ms_with_migration"],
                         r["contention_events"][k]["migration_stream_ms"])
                     - r["contention_events"][k]["demand_ms_alone"]) / r["contention_events"][k]["k"]
                    for r in ev_runs]
        return [r["contention"][k]["price_per_migration_ms"] for r in runs]
    spread = {
        "gpu->cpu": per_run("h2d_load_vs_demotion", "events"),
        "cpu->gpu": per_run("h2d_load_vs_promotion", "events"),
        "cpu->disk": per_run("disk_load_vs_demotion", "threads")}
    sync_runs = {"gpu->cpu": [r["paths"][key]["d2h"]["median_ms"] for r in runs],
                 "cpu->gpu": [r["paths"][key]["h2d"]["median_ms"] for r in runs],
                 "cpu->disk": [r["paths"][key]["disk_write"]["median_ms"] - r["paths"][key]["d2h"]["median_ms"]
                               for r in runs]}
    for row in rows:
        s, d = row["migration"].split("->")
        if row["migration"] in spread:
            v = [x / rw(s, d) for x in spread[row["migration"]]]
            row["alpha_async_range"] = [round(min(v), 3), round(max(v), 3)]
        if row["migration"] in sync_runs:
            v = [x / rw(s, d) for x in sync_runs[row["migration"]]]
            row["alpha_sync_range"] = [round(min(v), 3), round(max(v), 3)]
    per_run_async = {k: [r["contention"][k]["price_per_migration_ms"] for r in runs]
                     for k in ("h2d_load_vs_demotion", "h2d_load_vs_promotion", "disk_load_vs_demotion")}
    oa = json.loads((EXP / "online_admission.json").read_text())
    be = {r["trace"]: r["alpha_at_70pct_of_static"] for r in oa["rows"]
          if r["trace"] in ("conversation", "toolagent") and r["arm"] == "hazard-new"
          and r["policy"] == "onestep" and r["migration_cost_model"] == "read+write"
          and r["tie"] == "new-behind"}
    mix = {r["trace"]: r["migrations_by_tier"] for r in oa["rows"]
           if r["trace"] in ("conversation", "toolagent") and r["arm"] == "hazard-new"
           and r["policy"] == "onestep" and r["migration_cost_model"] == "zero" and r["tie"] == "new-behind"}
    out = {"config": {"block_bytes": BLOCK, "block": "llama3-8b-like, 512 tokens (Mooncake)",
                      "machine": runs[0]["machine"].get("nvidia_smi", "").split("\n")[0],
                      "disk": "RAID-backed rotational (MegaRAID SAS-3 3108), not NVMe",
                      "n_runs": len(runs), "n_event_runs": len(ev_runs),
                      "statistic": "median over runs of each run's median",
                      "gpu_read": "half a device-to-device copy",
                      "unmeasured": "remote_rdma (no RDMA fabric on the machine)"},
           "read_ms": {k: round(v, 4) for k, v in read.items()}, "rows": rows,
           "async_price_per_run_ms": per_run_async,
           "hazard_break_even_alpha70": be, "hazard_onestep_migration_mix": mix}
    (EXP / "measured_migration.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"read_ms": out["read_ms"], "rows": rows, "break_even": be}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
