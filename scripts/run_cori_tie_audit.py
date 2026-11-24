#!/usr/bin/env python3
"""A tie rule in a published evaluator outside KV caching: Cori's hybrid-memory page scheduler (notebook BO).

Cori (Doudali, Zahka, Gavrilovska, IPDPS'21) tunes how often a periodic page scheduler moves pages between fast
and slow memory. Its public simulator (github.com/GTkernel/cori-sim) ranks the pages touched in a period by
access count, rounded down to multiples of 20, and moves the top ones into a fast tier of 20% of the pages.
The rounding makes large groups of equal counts, and the ranking breaks their ties with numpy's default
(unstable) argsort; the paper states no tie rule. This script reruns the simulator as published (Python 2.7,
pinned commit, unmodified) and with the tie order changed, nothing else:

  default   the simulator as published (np.argsort, then reversed)
  first     lower page id first (ids are assigned in first-touch order: the item-order rule of section 5)
  last      higher page id first
  rand0-4   five random orders

over Cori's own candidate period lengths (its run_cori.py, steps 1-2) on the nine traces the repository ships,
for the history scheduler Cori tunes and the oracle scheduler it compares with.

  python3 scripts/run_cori_tie_audit.py setup DIR             clone the pinned commit, check trace digests
  python3 scripts/run_cori_tie_audit.py run DIR --python PY27 [--jobs N]
  python3 scripts/run_cori_tie_audit.py collect DIR           -> experiments/cori_ties.json
  python3 scripts/run_cori_tie_audit.py                       re-derive the summary from that file

PY27 is a Python 2.7 with numpy (1.16.5 here), e.g. `micromamba create -p ENV -c conda-forge python=2.7
numpy=1.16`. The worker is scripts/patches/cori_tie_worker.py.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "cori_ties.json"
WORKER = ROOT / "scripts" / "patches" / "cori_tie_worker.py"
REPO, COMMIT = "https://github.com/GTkernel/cori-sim", "ca2ef92"
MODES = ("default", "first", "last") + tuple(f"rand{i}" for i in range(5))
POLICIES = ("history", "oracle")


def traces(cori: Path) -> list[Path]:
    return sorted((cori / "traces" / "pin_traces").glob("trace_*.txt"))


def setup(d: Path) -> None:
    cori = d / "cori-sim"
    if not cori.exists():
        subprocess.run(["git", "clone", "-q", REPO, str(cori)], check=True)
    subprocess.run(["git", "-C", str(cori), "checkout", "-q", COMMIT], check=True)
    for t in traces(cori):
        print(hashlib.sha256(t.read_bytes()).hexdigest()[:16], t.name)


def run(d: Path, py: str, jobs: int) -> None:
    cori, raw = d / "cori-sim", d / "raw.jsonl"
    cand = {}
    for t in traces(cori):
        out = subprocess.run([py, str(WORKER), str(cori), "candidates", str(t)], capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()[-1]
        cand[t] = json.loads(out)
    (d / "candidates.json").write_text(json.dumps({t.name: c for t, c in cand.items()}, indent=1) + "\n")
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            done.add((r["trace"], r["policy"], r["reqs_per_period"], r["mode"]))
    todo = [(t, p, q, m) for t in cand for q in cand[t]["candidates"] for p in POLICIES for m in MODES
            if (t.name, p, q, m) not in done]
    print(f"{len(todo)} runs ({len(done)} already done)", flush=True)

    def one(job):
        t, p, q, m = job
        out = subprocess.run([py, str(WORKER), str(cori), "run", str(t), p, str(q), m], capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()[-1]
        return out

    with ThreadPoolExecutor(jobs) as ex, open(raw, "a") as f:
        for n, line in enumerate(ex.map(one, todo), 1):
            f.write(line + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def collect(d: Path) -> None:
    cori = d / "cori-sim"
    head = subprocess.run(["git", "-C", str(cori), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    rows = [json.loads(line) for line in (d / "raw.jsonl").read_text().splitlines()]
    OUT.write_text(json.dumps({
        "producer": f"scripts/run_cori_tie_audit.py over {REPO}@{head} (Python 2.7, numpy 1.16.5), "
                    "scripts/patches/cori_tie_worker.py",
        "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        "trace_sha256": {t.name: hashlib.sha256(t.read_bytes()).hexdigest() for t in traces(cori)},
        "candidates": json.loads((d / "candidates.json").read_text()),
        "modes": list(MODES), "policies": list(POLICIES), "fast_capacity": 0.2,
        "runs": sorted(rows, key=lambda r: (r["trace"], r["policy"], r["reqs_per_period"], r["mode"]))},
        indent=1) + "\n")
    print("wrote", OUT)


def summarize(d: dict) -> dict:
    by = {}
    for r in d["runs"]:
        by.setdefault((r["trace"], r["policy"]), {}).setdefault(r["reqs_per_period"], {})[r["mode"]] = r
    out = {"per_trace": [], "best_period_changes": 0, "traces": len({k[0] for k in by})}
    for (t, p), periods in sorted(by.items()):
        spread = {q: max(m["slowdown_pct"] for m in ms.values()) - min(m["slowdown_pct"] for m in ms.values())
                  for q, ms in periods.items()}
        best = {mode: min(periods, key=lambda q: (periods[q][mode]["slowdown_pct"], q)) for mode in d["modes"]}
        cut = sum(ms["default"]["periods_cut_in_tie"] for ms in periods.values())
        over = sum(ms["default"]["periods_over_capacity"] for ms in periods.values())
        row = {"trace": t, "policy": p, "max_spread_points": round(max(spread.values()), 2),
               "spread_at_max": {"reqs_per_period": max(spread, key=spread.get),
                                 "slowdown_pct": {m: periods[max(spread, key=spread.get)][m]["slowdown_pct"]
                                                  for m in d["modes"]}},
               "best_period_by_mode": best, "best_period_moves": len(set(best.values())) > 1,
               "best_slowdown_by_mode": {m: periods[best[m]][m]["slowdown_pct"] for m in d["modes"]},
               "periods_cut_in_tie": cut, "periods_over_capacity": over}
        out["per_trace"].append(row)
        out["best_period_changes"] += row["best_period_moves"]
    hist = [r for r in out["per_trace"] if r["policy"] == "history"]
    out["history"] = {
        # every over-capacity period of every candidate run, the simulator as published
        "periods_cut_in_tie": sum(r["periods_cut_in_tie"] for r in hist),
        "periods_over_capacity": sum(r["periods_over_capacity"] for r in hist),
        "best_period_moves_traces": sum(r["best_period_moves"] for r in hist),
        # how far the tie order alone moves the best period's runtime (slowdown is % over all-fast)
        "best_runtime_change_pct": round(max(100 * ((100 + max(r["best_slowdown_by_mode"].values()))
                                                    / (100 + min(r["best_slowdown_by_mode"].values())) - 1)
                                             for r in hist), 2)}
    out["max_spread_points"] = round(max(r["max_spread_points"] for r in out["per_trace"]), 2)
    return out


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] in ("setup", "run", "collect"):
        d = Path(sys.argv[2]).resolve()
        d.mkdir(parents=True, exist_ok=True)
        if sys.argv[1] == "setup":
            setup(d)
            return 0
        if sys.argv[1] == "run":
            py = sys.argv[sys.argv.index("--python") + 1]
            jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
            run(d, py, jobs)
            return 0
        collect(d)
    d = json.loads(OUT.read_text())
    d["summary"] = summarize(d)
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    for r in d["summary"]["per_trace"]:
        print(f"{r['trace']:<32} {r['policy']:<8} spread {r['max_spread_points']:6.2f}  best period moves "
              f"{r['best_period_moves']}  cut in tie {r['periods_cut_in_tie']}/{r['periods_over_capacity']}")
    print("traces x policies whose best period moves:", d["summary"]["best_period_changes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
