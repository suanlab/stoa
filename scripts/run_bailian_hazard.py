#!/usr/bin/env python3
"""The estimated prior for unseen blocks (run_hazard_prior.py) on Alibaba's four Bailian traces.

The paper's arrival-time claim rests on the hazard arm: ranking a never-seen block where the prefix's
reuse statistics put it. This replicates it on the external traces run_bailian.py prepares (16-token
blocks, corrected bill), in both action spaces, and records each trace's split-time ceiling beside it
(from experiments/bailian_reachable_birthx.json). Notebook section AU.

Each (trace, action space) job runs in a fresh process through run_bailian.py's worker, so the
block-size and action-space patches cannot leak. Finished parts are reused (resumable).

Usage: python3 scripts/run_bailian.py --birthx   (once, to download and convert)
       python3 scripts/run_bailian_hazard.py [--parts DIR] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ("traceA", "traceB", "coder", "thinking")
SPACES = ("full", "tier_only")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default=str(Path(tempfile.gettempdir()) / "stoa_bailian_hazard_parts"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    parts = Path(args.parts)
    parts.mkdir(parents=True, exist_ok=True)

    def run(job):
        t, sp = job
        out = parts / f"hz_{t}_{sp}.json"
        if out.exists():
            return 0
        data = ROOT / "data" / f"bailian_{t}_mooncake.jsonl"
        if not data.exists():
            sys.exit(f"missing {data}; run scripts/run_bailian.py --birthx first")
        with (parts / f"hz_{t}_{sp}.log").open("w") as fh:
            return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_bailian.py"), "--worker",
                                   t, sp, "run_hazard_prior.py", str(data), str(out), "--birthx"],
                                  cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode

    jobs = [(t, sp) for t in TRACES for sp in SPACES]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rcs = list(ex.map(run, jobs))
    if any(rcs):
        sys.exit(f"{sum(1 for r in rcs if r)} job(s) failed; see {parts}/*.log")

    reach = json.loads((ROOT / "experiments" / "bailian_reachable_birthx.json").read_text())
    rows = []
    for t, sp in jobs:
        d = json.loads((parts / f"hz_{t}_{sp}.json").read_text())
        ceil = [r["reachable_pct"] for r in reach[sp]["traces"][t]["rows"] if r["split_frac"] == 0.5][0]
        for r in d["rows"]:
            rows.append({**r, "actions": sp, "split_ceiling_pct": ceil})
        print(t, sp, "ceiling", ceil, {(r["arm"], r["tie"]): r["captured_pct"] for r in d["rows"]})
    est = {t: json.loads((parts / f"hz_{t}_full.json").read_text())["estimates"][t] for t in TRACES}
    dest = ROOT / "experiments" / "bailian_hazard_birthx.json"
    dest.write_text(json.dumps({"config": {"source": "experiments/bailian_reachable_birthx.json traces",
                                           "producer": "scripts/run_hazard_prior.py via run_bailian.py --worker",
                                           "bill": "creation accesses are compulsory misses"},
                                "estimates": est, "rows": rows}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
