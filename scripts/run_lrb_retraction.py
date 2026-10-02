#!/usr/bin/env python3
"""Regenerate `experiments/lrb_retraction.json` from code: the LRB column, three ways.

Why this script exists. The retraction artifact behind §5.5's LRB band had no producer. Its two
repaired columns were computed during the session that fixed the bug, and its broken column
came from code that no longer exists -- the fix replaced it in place. A committee reproducing
the paper could read the band but not regenerate it, which ICDE's Experiment, Analysis and
Benchmark category forbids outright ("MUST provide all artifacts necessary to reproduce the
results. No exceptions").

`simulate_lrb(buffer_policy=...)` now keeps all three behaviours selectable, so each column
is one call:

  tail_truncation  -- the retracted bug: keep the newest rows, labels collapse to constant
  all_history      -- no age bound, uniform subsample over everything accumulated
  sliding_window   -- drop rows older than the memory window, then subsample (adopted)

Belady, ARC and the operating points are exactly those of `run_reactive_real.py`, through the
same functions, so this table and §5.5's main table share a frame by construction.

What this artifact no longer carries: a "~63%" reading attributed to an independent
reimplementation "produced during review". That figure came from an AI agent in an internal
review simulation we ran, its code is not part of this artifact, and nothing here can
regenerate it. See docs/claims_dependency.md §AO.

    python3 scripts/run_lrb_retraction.py          # full traces, 24 LRB runs in parallel
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
FRACS = (0.02, 0.05, 0.10, 0.20)          # identical to run_reactive_real.FRACS
POLICIES = ("tail_truncation", "all_history", "sliding_window")


def _lrb_job(job):
    trace, frac, policy, requests = job
    from stoa.lrb import simulate_lrb
    from stoa.mooncake import load_mooncake
    wl = load_mooncake(str(ROOT / TRACES[trace]), max_requests=requests or None)
    slots = max(1, int(frac * len(wl.items)))
    return trace, frac, policy, simulate_lrb(wl, slots, buffer_policy=policy).hit_rate


def _ref_job(job):
    trace, frac, requests = job
    from stoa.eval.online import simulate_arc, simulate_fast
    from stoa.mooncake import load_mooncake
    wl = load_mooncake(str(ROOT / TRACES[trace]), max_requests=requests or None)
    slots = max(1, int(frac * len(wl.items)))
    return trace, frac, simulate_fast(wl, "belady", slots).hit_rate, simulate_arc(wl, slots).hit_rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=str, default="experiments/lrb_retraction.json")
    args = ap.parse_args()

    for p in TRACES.values():
        if not (ROOT / p).exists():
            sys.exit(f"missing {p} — see data/README.md")

    lrb_jobs = [(t, f, p, args.requests) for t in TRACES for f in FRACS for p in POLICIES]
    ref_jobs = [(t, f, args.requests) for t in TRACES for f in FRACS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        refs = {(t, f): (b, a) for t, f, b, a in ex.map(_ref_job, ref_jobs)}
        lrb = {(t, f, p): h for t, f, p, h in ex.map(_lrb_job, lrb_jobs)}

    rows = []
    print(f"{'trace':<13} {'tier':>5} | {'broken':>7} {'all-hist':>9} {'sliding':>8} | {'ARC':>6} | below ARC?")
    for t in TRACES:
        for f in FRACS:
            b, a = refs[(t, f)]
            norm = (lambda h: round(h / b, 4) if b > 0 else 0.0)
            r = {"trace": t, "cache_frac": f, "arc": norm(a),
                 "lrb_broken_truncation": norm(lrb[(t, f, "tail_truncation")]),
                 "lrb_repair_all_history": norm(lrb[(t, f, "all_history")]),
                 "lrb_repair_sliding_window": norm(lrb[(t, f, "sliding_window")])}
            r["below_arc_under_all_repairs"] = (
                r["lrb_repair_all_history"] < r["arc"] and r["lrb_repair_sliding_window"] < r["arc"])
            rows.append(r)
            print(f"{t:<13} {f:>5.0%} | {100*r['lrb_broken_truncation']:>6.1f}% "
                  f"{100*r['lrb_repair_all_history']:>8.1f}% {100*r['lrb_repair_sliding_window']:>7.1f}% "
                  f"| {100*r['arc']:>5.1f}% | {r['below_arc_under_all_repairs']}")

    out = {
        "config": {"requests": args.requests or "full", "cache_fracs": list(FRACS),
                   "policies": list(POLICIES), "normalisation": "hit rate / Belady hit rate"},
        "status": "RESOLVED as a band of two reproducible repairs",
        "cause": ("the training buffer was truncated to its tail; censored rows arrive in one "
                  "contiguous burst per checkpoint, so from the third retrain the labels were "
                  "constant and eviction degenerated to a uniform draw"),
        "repairs": {
            "all_history": "uniform subsample over every accumulated row",
            "sliding_window": ("drop rows older than memory_window first, then subsample -- "
                               "what the published 'sliding memory window' means; adopted")},
        "not_reproduced": ("an AI agent in an internal review simulation reported ~63% at "
                           "conversation/10% for its own repair; its code is not part of this "
                           "artifact and the paper does not cite the figure"),
        "rows": rows,
    }
    dest = ROOT / args.out
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
