#!/usr/bin/env python3
"""RQ2 on REAL traces: can a learned reuse predictor capture the placement headroom?

Everything before this ran on a synthetic Zipf workload where prefix frequency
predicts future frequency almost perfectly, leaving ~3% headroom -- no policy could
distinguish itself. Real Mooncake traces leave ~80% headroom because ~78% of KV
blocks are never reused while a few shared prefixes are reused thousands of times.
So the learnable question is finally the real one: *will this block be reused?*

Protocol (leak-free by construction):
  features  <- statistics observable in the trace PREFIX (count, recency, age,
               inter-arrival, mean position within a request)
  label     <- future access count (oracle; used for the label only)
  placement <- greedy over predicted counts, billed at the TRUE future cost
  compare   <- prefix-greedy (no learning) | learned | oracle (knows the future)

Train on one trace, evaluate on the OTHER (cross-trace generalization). Torch-free,
CPU, seconds. Usage:  python3 scripts/train_on_mooncake.py [--requests N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.learn import _solve  # noqa: E402  (shared ridge solver)
from stoa.mooncake import load_mooncake, trace_summary  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
N_FEAT = 7


def block_positions(path: str, max_requests: int) -> dict[str, float]:
    """Mean position of a block within its requests — early blocks are shared prefixes,
    a genuinely deployable signal (you know where a block sits in the sequence)."""
    acc: dict[str, list[int]] = defaultdict(list)
    with Path(path).open() as fh:
        for t, line in enumerate(fh):
            if t >= max_requests:
                break
            for pos, h in enumerate(json.loads(line).get("hash_ids", [])):
                acc[f"b{h}"].append(pos)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def featurize(wl, positions, split_frac=0.5):
    """(X, y, ids): prefix-only features and the future-count label."""
    st = max(1, wl.time_split(split_frac))
    stats = wl.prefix_stats(st)
    future = wl.future_counts(st)
    X, y, ids = [], [], []
    for it in wl.items:
        s = stats[it.item_id]
        if s["count"] == 0:                       # unseen in the prefix -> nothing to predict from
            continue
        pos = positions.get(it.item_id, 0.0)
        X.append([
            1.0,
            math.log1p(s["count"]),
            s["recency"] / st,
            s["age"] / st,
            s["mean_interarrival"] / st,
            math.log1p(pos),
            1.0 / (1.0 + s["recency"]),
        ])
        y.append(math.log1p(future.get(it.item_id, 0)))
        ids.append(it.item_id)
    return X, y, ids


def fit(X, y, ridge=1e-3):
    n = len(X)
    ata = [[0.0] * N_FEAT for _ in range(N_FEAT)]
    aty = [0.0] * N_FEAT
    for i in range(n):
        xi, yi = X[i], y[i]
        for r in range(N_FEAT):
            aty[r] += xi[r] * yi
            for c in range(N_FEAT):
                ata[r][c] += xi[r] * xi[c]
    return _solve(ata, aty, ridge)


def main() -> None:
    p = argparse.ArgumentParser(description="RQ2 on real Mooncake traces")
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--out", type=str, default="experiments/mooncake_rq2.json")
    args = p.parse_args()

    data = {}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — download the Mooncake traces first (see stoa/mooncake.py)")
        wl = load_mooncake(path, max_requests=args.requests)
        pos = block_positions(path, args.requests)
        X, y, ids = featurize(wl, pos)
        data[name] = {"wl": wl, "X": X, "y": y, "ids": ids, "pos": pos,
                      "summary": trace_summary(wl)}

    rows = []
    print(f"\nRQ2 on real Mooncake traces (first {args.requests} requests each)")
    print(f"{'eval trace':<14} | {'prefix-greedy':>13} | {'learned':>10} | {'oracle':>9} | headroom captured")
    print("-" * 78)
    for eval_name in TRACES:
        train_name = next(n for n in TRACES if n != eval_name)     # cross-trace: train on the other
        w = fit(data[train_name]["X"], data[train_name]["y"])
        d = data[eval_name]
        wl = d["wl"]
        st = max(1, wl.time_split(0.5))
        stats = wl.prefix_stats(st)
        future = wl.future_counts(st)

        pred = {i: max(0.0, math.expm1(sum(wi * xi for wi, xi in zip(w, x))))
                for i, x in zip(d["ids"], d["X"])}
        prefix_belief = {it.item_id: stats[it.item_id]["count"] for it in wl.items}
        true_belief = {it.item_id: float(future.get(it.item_id, 0)) for it in wl.items}

        c_pref = place_with_beliefs(wl, prefix_belief)
        c_learn = place_with_beliefs(wl, pred)
        c_orac = place_with_beliefs(wl, true_belief)
        cap = 100.0 * (c_pref - c_learn) / (c_pref - c_orac) if c_pref > c_orac else 0.0
        print(f"{eval_name:<14} | {c_pref:13.1f} | {c_learn:10.1f} | {c_orac:9.1f} | {cap:+7.1f}%"
              f"   (train: {train_name})")
        rows.append({"eval_trace": eval_name, "train_trace": train_name,
                     "prefix_greedy": round(c_pref, 2), "learned": round(c_learn, 2),
                     "oracle": round(c_orac, 2), "headroom_captured_pct": round(cap, 2),
                     "trace_summary": d["summary"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"requests": args.requests, "protocol": "cross-trace, "
                                          "prefix features / future label, true-cost billing"},
                               "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
