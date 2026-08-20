#!/usr/bin/env python3
"""Hyperparameter sweep for the LRB-style learned eviction policy.

`scripts/run_reactive_real.py --lrb` reports LRB at one setting, and it loses to ARC at
every operating point. A negative result about someone else's method is exactly where our
own free-parameter defect (docs/claims_dependency.md §I) would do the most damage, so this
script varies LRB's three knobs -- memory window, eviction sample size, ensemble size --
and reports the whole range rather than a point.

It also records the quantity that decides whether LRB *can* learn on these traces: the
fraction of its training rows that are RIGHT-CENSORED. A censored row is one whose block
was never re-requested inside the memory window; it says "not within the window" and
nothing about when, so it carries no information about the regression target. With 76-78%
of blocks touched exactly once, most rows are censored by construction.

CPU, roughly 25 minutes for the default grid. Deterministic given `--seed`.

Usage:  python3 scripts/run_lrb_sweep.py [--tier 0.10] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.online import simulate_arc, simulate_fast  # noqa: E402
from stoa.lrb import simulate_lrb  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}

# (memory_window, sample_size, n_trees). The first row is the default used elsewhere; the
# rest vary one knob at a time so a difference is attributable.
GRID = ((10_000, 64, 30), (50_000, 64, 30), (200_000, 64, 30),
        (50_000, 32, 30), (50_000, 128, 30), (50_000, 64, 100))


def main() -> None:
    ap = argparse.ArgumentParser(description="LRB hyperparameter sweep")
    ap.add_argument("--tier", type=float, default=0.10, help="hot-tier fraction of blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/lrb_sweep.json")
    args = ap.parse_args()

    report = {"config": {"tier_frac": args.tier, "seed": args.seed,
                         "grid": [list(g) for g in GRID]}, "traces": {}}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — see src/stoa/mooncake.py for the download command")
        wl = load_mooncake(path)
        slots = max(1, int(args.tier * len(wl.items)))
        belady = simulate_fast(wl, "belady", slots).hit_rate
        arc = simulate_arc(wl, slots).hit_rate
        print(f"\n=== {name} @ {args.tier:.0%} tier: Belady {belady:.1%}, "
              f"ARC {arc / belady:.1%} of Belady ===", flush=True)
        rows = []
        for mw, ss, nt in GRID:
            t0 = time.perf_counter()
            r = simulate_lrb(wl, slots, memory_window=mw, sample_size=ss, n_trees=nt,
                             train_interval=20_000, seed=args.seed)
            # simulate_lrb reuses the ARC diagnostic slots: final_p = censored rows,
            # ghost_hits = labelled rows, p_moves = number of model fits.
            n_rows = r.final_p + r.ghost_hits
            cens = r.final_p / n_rows if n_rows else 0.0
            frac = r.hit_rate / belady if belady else 0.0
            print(f"  window={mw:>7,} sample={ss:>3} trees={nt:>3} -> {frac:5.1%} of Belady"
                  f" | censored {cens:.1%} of {n_rows:,} rows | {r.p_moves} fits"
                  f" | {time.perf_counter() - t0:.0f}s", flush=True)
            rows.append({"memory_window": mw, "sample_size": ss, "n_trees": nt,
                         "hit_rate": round(r.hit_rate, 4),
                         "over_belady": round(frac, 4),
                         "censored_frac": round(cens, 4), "train_rows": n_rows,
                         "n_fits": r.p_moves})
        best = max(r["over_belady"] for r in rows)
        worst = min(r["over_belady"] for r in rows)
        print(f"  LRB spans {worst:.1%}-{best:.1%} of Belady; ARC is {arc / belady:.1%}",
              flush=True)
        report["traces"][name] = {"belady": round(belady, 4), "arc": round(arc, 4),
                                  "arc_over_belady": round(arc / belady, 4),
                                  "lrb_min_over_belady": round(worst, 4),
                                  "lrb_max_over_belady": round(best, 4), "rows": rows}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
