#!/usr/bin/env python3
"""Does wall-clock think time make KV reuse predictable? (closing our own open problem)

On Mooncake we concluded that block reuse hinges on whether a session stays alive, and
that the trace could not express it -- the signal was missing, not the model. The
kv-cache-tester traces record real wall-clock timestamps per request, so the same
hypothesis is finally testable, on a second and independent production source.

We predict the binary event "is this block accessed again after the split" from three
nested feature sets and report AUC for each:

    block    : log prefix count, recency (in requests), age
    +session : requests so far in this conversation, requests since its last activity
               (conversations are explicit here -- blocks are namespaced per conversation,
                unlike Mooncake where we had to infer sessions from a shared prefix id)
    +think   : seconds since the conversation's last request, its mean think time, and
               the ratio between them -- i.e. "is this conversation going stale?"

If +think lifts AUC materially, the earlier negative result was about missing
information, and the design implication is to expose session liveness at the serving
interface. If it does not, reuse is unpredictable even with the signal we asked for.

CPU, numpy. Usage:  python3 scripts/train_thinktime.py [--convs N]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_session_features import auc, fit_logistic, predict  # noqa: E402

from stoa.kvct import _iter_conversations  # noqa: E402

FEATURE_SETS = ("block", "block+session", "block+session+think")


def build(conv_range: tuple[int, int], split_frac: float = 0.5, data_dir: str = "data/kvct"):
    """Per-block features at the split, plus the binary 'reused later' label.

    `conv_range` is a half-open [lo, hi) slice of conversations, so the caller can make
    the fit and evaluation sets DISJOINT. (An earlier version passed a limit and took
    the first N for both, which put every training conversation inside the evaluation
    set -- the same overlap bug this project has hit before.)
    """
    lo, hi = conv_range
    events = []                                        # (wall_t, conv_id, [block ids])
    for ci, conv in enumerate(_iter_conversations(data_dir, hi)):
        if ci < lo:
            continue
        cid = conv.get("id", "c")
        for req in conv.get("requests", []):
            hs = req.get("hash_ids") or []
            if hs:
                events.append((float(req.get("t", 0.0)), cid, [f"{cid}:b{h}" for h in hs]))
    events.sort(key=lambda e: e[0])
    if not events:
        return None
    split_idx = int(split_frac * len(events))

    blk_steps: dict[str, list[int]] = defaultdict(list)
    blk_conv: dict[str, str] = {}
    conv_steps: dict[str, list[int]] = defaultdict(list)
    conv_walls: dict[str, list[float]] = defaultdict(list)
    for i, (wall, cid, blocks) in enumerate(events):
        conv_steps[cid].append(i)
        conv_walls[cid].append(wall)
        for b in blocks:
            blk_steps[b].append(i)
            blk_conv.setdefault(b, cid)

    split_wall = events[split_idx][0]
    rows = {k: [] for k in FEATURE_SETS}
    y = []
    for b, steps in blk_steps.items():
        past = [s for s in steps if s < split_idx]
        if not past:
            continue
        cnt, recency, age = len(past), split_idx - past[-1], split_idx - past[0]
        block = [1.0, math.log1p(cnt), recency / split_idx, age / split_idx]

        cid = blk_conv[b]
        c_past = [s for s in conv_steps[cid] if s < split_idx]
        c_cnt = len(c_past)
        c_rec = (split_idx - c_past[-1]) if c_past else split_idx
        session = [math.log1p(c_cnt), c_rec / split_idx]

        w_past = [w for w, s in zip(conv_walls[cid], conv_steps[cid]) if s < split_idx]
        since = (split_wall - w_past[-1]) if w_past else 0.0
        gaps = [b2 - a2 for a2, b2 in zip(w_past, w_past[1:])] if len(w_past) > 1 else [0.0]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        staleness = since / mean_gap if mean_gap > 0 else 0.0     # >1 => overdue for a turn
        think = [math.log1p(max(since, 0.0)), math.log1p(max(mean_gap, 0.0)),
                 math.log1p(max(staleness, 0.0))]

        rows["block"].append(block)
        rows["block+session"].append(block + session)
        rows["block+session+think"].append(block + session + think)
        y.append(1.0 if any(s >= split_idx for s in steps) else 0.0)
    return rows, y


def main() -> None:
    p = argparse.ArgumentParser(description="Think-time features for reuse prediction")
    p.add_argument("--convs", type=int, default=30)
    p.add_argument("--train-convs", type=int, default=None,
                   help="fit on a DISJOINT set of conversations (default: half of --convs)")
    p.add_argument("--out", type=str, default="experiments/thinktime_rq2.json")
    args = p.parse_args()

    n_train = args.train_convs or max(2, args.convs // 2)
    train = build((0, n_train))                        # conversations [0, n_train)
    full = build((n_train, args.convs))                # DISJOINT: [n_train, convs)
    if not train or not full:
        sys.exit("no kv-cache-tester data — run: python3 -c 'from stoa.kvct import download; download(200)'")
    Xtr, ytr = train
    Xte, yte = full

    print(f"\nThink-time reuse prediction (kv-cache-tester; fit on convs [0,{n_train}), "
          f"evaluate on DISJOINT convs [{n_train},{args.convs}))")
    print(f"  train blocks {len(ytr):,} ({sum(ytr)/len(ytr):.1%} reused) | "
          f"eval blocks {len(yte):,} ({sum(yte)/len(yte):.1%} reused)")
    print(f"{'features':>24} | {'AUC':>7} | lift vs block-only")
    print("-" * 60)
    base = None
    rows = []
    for fs in FEATURE_SETS:
        m = fit_logistic(Xtr[fs], ytr, epochs=3000, lr=2.0)
        a = auc(predict(m, Xte[fs]), yte)
        base = a if base is None else base
        print(f"{fs:>24} | {a:7.3f} | {a - base:+.3f}")
        rows.append({"features": fs, "auc": round(a, 4), "lift_vs_block": round(a - base, 4)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"train_convs": n_train, "eval_convs": args.convs,
                   "source": "kv-cache-tester (Claude Code agentic traces)"},
        "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
