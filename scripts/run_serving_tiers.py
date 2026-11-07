#!/usr/bin/env python3
"""Per-tier KV costs in a real serving stack, assembled from measure_serving_tiers.py runs (§BE).

`--collect DIR` reads the runs a GPU host produced (serve_none.json, serve_cpu.json, serve_disk.json,
serve_disk_buf1.json, serve_disk_cold.json) and LMCache's own per-request log lines (run_cpu.log,
run_disk_cold.log: "Retrieved N ... cost X ms", "Stored N ... cost X ms") into
experiments/serving_tiers_a100.json. Without --collect it re-derives the summary from that file.

Derived per 512-token block (the Mooncake block; 64 MiB for this 8B shape), from the 8,192-token prefix:
  gpu_hit          time to first token with the prefix in vLLM's GPU prefix cache (fixed overhead included)
  recompute        (TTFT recompute - TTFT gpu hit) / 16 blocks
  cpu_load         LMCache's logged retrieval cost from host memory / 16
  disk_cold_load   LMCache's logged retrieval cost from disk after the page cache and the RAID
                   controller's cache were flushed / 16
  cpu_store        LMCache's logged offload cost GPU -> host / 16 (paid by the request that creates it)
The cache-assisted disk runs (serve_disk*.json) are kept and labelled: their loads were faster than the
array can read cold, so they are not the disk tier's price.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "serving_tiers_a100.json"
RUNS = ("serve_none.json", "serve_cpu.json", "serve_disk.json", "serve_disk_buf1.json", "serve_disk_cold.json")
RET = re.compile(r"Retrieved (\d+) out of \d+ required tokens.*?cost ([\d.]+) ms")
STO = re.compile(r"Stored (\d+) out of total \d+ tokens.*?cost ([\d.]+) ms")


def logged(path: Path, pattern, tokens: int) -> list[float]:
    return [float(m.group(2)) for m in pattern.finditer(path.read_text(errors="ignore")) if int(m.group(1)) == tokens]


def collect(d: Path) -> None:
    runs = {}
    for name in RUNS:
        r = json.loads((d / name).read_text())
        r.pop("lmcache_env", None)
        runs[name.removesuffix(".json")] = r
    logs = {"cpu_retrieve_ms_8192": logged(d / "run_cpu.log", RET, 8192),
            "cpu_store_ms_8192": logged(d / "run_cpu.log", STO, 8192),
            "disk_cold_retrieve_ms_8192": logged(d / "run_disk_cold.log", RET, 8192),
            "disk_store_ms_8192": logged(d / "run_disk_cold.log", STO, 8192)}
    OUT.write_text(json.dumps({"producer": "scripts/measure_serving_tiers.py on one A100 80GB PCIe Gen4 x16 "
                                           "(GPU idle, host shared); assembled by run_serving_tiers.py --collect",
                               "runs": runs, "lmcache_logged": logs}, indent=1) + "\n")
    print("wrote", OUT)


def per_tier_costs() -> dict:
    """Per-512-token-block cost (ms) of serving a block from each measured tier: the GPU tier's marginal
    time to first token, plus LMCache's load cost for host memory and cold disk. The one place these are
    derived; run_measured_table.py imports it."""
    s = json.loads(OUT.read_text())["summary"]
    t, pb = s["ttft_median_ms"], s["per_512_token_block"]
    gpu = (t["8192"]["gpu"] - t["512"]["gpu"]) / 15
    return {"gpu": gpu, "host": gpu + pb["cpu_load_ms"], "disk": gpu + pb["disk_cold_load_ms"]}


def main() -> int:
    if "--collect" in sys.argv:
        collect(Path(sys.argv[sys.argv.index("--collect") + 1]))
    d = json.loads(OUT.read_text())
    r, lg = d["runs"], d["lmcache_logged"]
    P, nb = "8192", 8192 // 512
    gpu = r["serve_none"]["results"][P]["gpu"]["median_ms"]
    rec = r["serve_none"]["results"][P]["recompute"]["median_ms"]
    med = lambda xs: statistics.median(xs)                                             # noqa: E731
    per_block = {"recompute_ms": round((rec - gpu) / nb, 3),
                 "cpu_load_ms": round(med(lg["cpu_retrieve_ms_8192"]) / nb, 3),
                 "disk_cold_load_ms": round(med(lg["disk_cold_retrieve_ms_8192"]) / nb, 3),
                 "cpu_store_ms": round(med(lg["cpu_store_ms_8192"]) / nb, 3)}
    ttft = {L: {"recompute": r["serve_none"]["results"][L]["recompute"]["median_ms"],
                "gpu": r["serve_none"]["results"][L]["gpu"]["median_ms"],
                "cpu": r["serve_cpu"]["results"][L]["cpu"]["median_ms"],
                "disk_cold": r["serve_disk_cold"]["results"][L]["disk_cold"]["median_ms"],
                "disk_cache_assisted": r["serve_disk"]["results"][L]["disk"]["median_ms"]}
            for L in ("512", "2048", "8192")}
    flush = r["serve_disk_cold"]["results"]["flush"]
    summary = {"ttft_median_ms": ttft, "per_512_token_block": per_block,
               "n_logged": {k: len(v) for k, v in lg.items()},
               "disk_cold_load_gbps": round(2**30 / (med(lg["disk_cold_retrieve_ms_8192"]) / 1e3) / 1e9, 3),
               "cpu_load_gbps": round(2**30 / (med(lg["cpu_retrieve_ms_8192"]) / 1e3) / 1e9, 3),
               "flush_read_gbps": flush["gbps"], "flush_gib": round(flush["bytes"] / 2**30, 1),
               "env": r["serve_none"]["env"]}
    d["summary"] = summary
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("ttft_median_ms", "per_512_token_block", "disk_cold_load_gbps",
                                              "cpu_load_gbps", "n_logged")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
