#!/usr/bin/env python3
"""React, don't predict: reactive tiering on production KV traces.

Our predictive experiments established that block reuse is barely forecastable from
trace-local features, so a learned *placement* policy captures ~0% of the ~80%
headroom (scripts/train_on_mooncake.py, train_session_features.py). The design
implication is to stop predicting and start reacting: an online policy that evicts
based on what it has just observed needs no forecast at all.

This script measures that on the same Mooncake traces: Belady (offline optimum) vs
LRU, H2O/LFU, and a fill-once static cache, across hot-tier sizes. The headline number
is each reactive policy's hit rate as a FRACTION of Belady's -- how much of the
achievable benefit reaction alone delivers.

Usage:  python3 scripts/run_reactive_real.py [--requests N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.online import simulate_arc, simulate_fast  # noqa: E402
from stoa.experts import simulate_cacheus, simulate_lecar  # noqa: E402
from stoa.lrb import simulate_lrb  # noqa: E402
from stoa.mooncake import load_mooncake, trace_summary  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
FRACS = (0.02, 0.05, 0.10, 0.20)


def main() -> None:
    p = argparse.ArgumentParser(description="Reactive tiering on real traces")
    p.add_argument("--requests", type=int, default=0, help="0 = full trace")
    # LRB retrains online and scores 64 candidates per eviction, so it costs ~4 min per
    # (trace, tier size) against seconds for the rule-based policies. Off by default so the
    # cheap comparison stays cheap; the paper's numbers come from a run with it on.
    p.add_argument("--lrb", action="store_true", help="also run the LRB-style learned policy")
    # The output name ENCODES the scale. A subsample therefore cannot overwrite the
    # full-trace artifact that the figures and the paper read -- an earlier revision had
    # `reactive_real.json` and `reactive_real_full.json` diverge silently, which is the
    # same reference-frame hazard we catalog in the paper.
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    out_path = Path(args.out) if args.out else Path(
        "experiments/reactive_real_full.json" if not args.requests
        else f"experiments/reactive_real_sub{args.requests}.json")
    if args.lrb and not args.out:
        out_path = out_path.with_name(out_path.stem + "_lrb.json")

    report = {"config": {"requests": args.requests, "cache_fracs": list(FRACS)}, "traces": {}}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — see src/stoa/mooncake.py for the download command")
        wl = load_mooncake(path, max_requests=args.requests or None)
        summary = trace_summary(wl)
        rows = []
        print(f"\n=== {name}: {summary['items']:,} blocks / {summary['accesses']:,} accesses "
              f"({summary['one_time_frac']:.0%} one-time) ===")
        print(f"{'cache':>6} | {'Belady':>8} | {'LRU':>7} | {'LFU':>7} | {'static':>7} | "
              f"{'ARC':>7} | {'LeCaR':>7} | {'CACHEUS':>8} | {'best/Belady':>11}")
        print("-" * 92)
        for frac in FRACS:
            slots = max(1, int(frac * len(wl.items)))
            r = {k: simulate_fast(wl, k, slots) for k in ("belady", "lru", "h2o", "static")}
            b, l, h, st = (r[k].hit_rate for k in ("belady", "lru", "h2o", "static"))
            arc_r = simulate_arc(wl, slots)
            arc = arc_r.hit_rate
            lecar = simulate_lecar(wl, slots).hit_rate
            cacheus = simulate_cacheus(wl, slots).hit_rate
            lrb = simulate_lrb(wl, slots).hit_rate if args.lrb else None
            # ARC is a SINGLE policy; `best` is an oracle over policies and is not deployable.
            # Reporting both separates "what one adaptive policy achieves" from the envelope.
            best = max([l, h, st, arc, lecar, cacheus] + ([lrb] if lrb is not None else []))
            ratio = best / b if b > 0 else 0.0
            arc_ratio = arc / b if b > 0 else 0.0
            print(f"{frac:6.0%} | {b:7.1%} | {l:6.1%} | {h:6.1%} | {st:6.1%} | "
                  f"{arc:6.1%} | {lecar:6.1%} | {cacheus:7.1%} | {ratio:10.1%}"
                  + (f" | LRB {lrb:6.1%} ({lrb / b:.1%} of Belady)" if lrb is not None else ""))
            rows.append({"cache_frac": frac, "belady": round(b, 4), "lru": round(l, 4),
                         "h2o": round(h, 4), "static": round(st, 4), "arc": round(arc, 4),
                         "arc_over_belady": round(arc_ratio, 4),
                         "lecar": round(lecar, 4), "cacheus": round(cacheus, 4),
                         "lecar_over_belady": round(lecar / b, 4) if b > 0 else 0.0,
                         "cacheus_over_belady": round(cacheus / b, 4) if b > 0 else 0.0,
                         "best_reactive_over_belady": round(ratio, 4),
                         # ARC only adapts on a ghost hit, so this rate bounds how much
                         # adaptation is even possible on a trace of mostly one-time blocks.
                         "arc_ghost_hit_rate_of_misses":
                             round(arc_r.ghost_hits / arc_r.misses, 4) if arc_r.misses else 0.0,
                         "arc_final_p_frac": round(arc_r.final_p / slots, 4),
                         "winner": max([(l, "lru"), (h, "lfu"), (st, "static"), (arc, "arc"),
                                        (lecar, "lecar"), (cacheus, "cacheus")]
                                       + ([(lrb, "lrb")] if lrb is not None else []))[1],
                         **({"lrb": round(lrb, 4),
                             "lrb_over_belady": round(lrb / b, 4) if b > 0 else 0.0}
                            if lrb is not None else {})})
        report["traces"][name] = {"summary": summary, "rows": rows}

    print("\nReaction captures most of the achievable benefit on the AGENT trace, far less on")
    print("open-ended conversation -- the gap a scheduler could still target is workload-dependent.")
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
