#!/usr/bin/env python3
"""Does predictable reuse convert into cheaper placement? (kv-cache-tester)

On Mooncake, reuse prediction reached AUC 0.43-0.64 and captured ~0% of the placement
headroom. On agentic kv-cache-tester traces the same features reach AUC 0.83-0.93. This
script asks whether that predictability actually buys cheaper placement.

CORRECTED PROTOCOL. An earlier version built features from a separate event stream
containing only the evaluation conversations, while billing placement on a workload
pooling every conversation -- two different split instants, so beliefs fitted at one
were applied to costs at another (docs/claims_dependency.md §D). Here there is exactly
ONE workload and ONE split instant:

    workload  = all conversations, pooled and time-ordered
    split     = wl.time_split(0.5)            <- the only split in the script
    features  = wl.prefix_stats(split) + session state at that same instant
    label     = wl.future_counts(split)
    fit       = blocks owned by TRAIN conversations
    evaluate  = blocks owned by EVAL conversations (disjoint), placed and billed on the
                true future

Reported against the frequency heuristic (no learning) and a hindsight oracle, using
both belief constructions: a classifier probability rescaled by mean reuse, and a direct
count regression. CPU. Usage: python3 scripts/train_kvct_placement.py [--convs N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_session_features import _standardizer, _z, auc, fit_logistic, predict  # noqa: E402

from stoa.kvct import load_kvct_with_meta  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402

FEATURE_SETS = ("block", "block+session", "block+session+think")


def featurize(wl, meta, split):
    """Features at `split` for every block seen before it. One instant, by construction."""
    stats = wl.prefix_stats(split)
    fut = wl.future_counts(split)
    wall, conv_of, conv_steps = meta["wall"], meta["conv_of"], meta["conv_steps"]
    split_wall = wall.get(split, max(wall.values()) if wall else 0.0)

    conv_past = {c: [s for s in steps if s < split] for c, steps in conv_steps.items()}
    rows = {k: [] for k in FEATURE_SETS}
    ids, y_bin, y_cnt, convs = [], [], [], []
    for it in wl.items:
        b = it.item_id
        s = stats[b]
        if s["count"] == 0:
            continue
        block = [1.0, math.log1p(s["count"]), s["recency"] / split, s["age"] / split]

        cid = conv_of.get(b, "?")
        cp = conv_past.get(cid, [])
        c_cnt = len(cp)
        c_rec = (split - cp[-1]) if cp else split
        session = [math.log1p(c_cnt), c_rec / split]

        w_past = [wall[t] for t in cp]
        since = (split_wall - w_past[-1]) if w_past else 0.0
        gaps = [q - p for p, q in zip(w_past, w_past[1:])] if len(w_past) > 1 else [0.0]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        stale = since / mean_gap if mean_gap > 0 else 0.0
        think = [math.log1p(max(since, 0.0)), math.log1p(max(mean_gap, 0.0)),
                 math.log1p(max(stale, 0.0))]

        rows["block"].append(block)
        rows["block+session"].append(block + session)
        rows["block+session+think"].append(block + session + think)
        ids.append(b)
        convs.append(cid)
        n = fut.get(b, 0)
        y_bin.append(1.0 if n > 0 else 0.0)
        y_cnt.append(float(n))
    return rows, ids, convs, y_bin, y_cnt


def fit_regression(X, y_counts, ridge=1e-3):
    """Least squares on log1p(count) — predicts MAGNITUDE, which is what placement needs."""
    import numpy as np
    mean, std = _standardizer(X)
    A = np.asarray(_z(X, mean, std), dtype=np.float64)
    b = np.asarray([math.log1p(v) for v in y_counts], dtype=np.float64)
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
    return {"w": w.tolist(), "mean": mean, "std": std}


def predict_counts(model, X):
    import numpy as np
    A = np.asarray(_z(X, model["mean"], model["std"]), dtype=np.float64)
    return np.maximum(0.0, np.expm1(A @ np.asarray(model["w"]))).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="kvct: prediction -> placement (corrected)")
    p.add_argument("--convs", type=int, default=40)
    p.add_argument("--out", type=str, default="experiments/kvct_placement_fixed.json")
    args = p.parse_args()

    wl, meta = load_kvct_with_meta(limit=args.convs)
    split = wl.time_split(0.5)                          # the ONE split instant
    rows, ids, convs, y_bin, y_cnt = featurize(wl, meta, split)
    if not ids:
        sys.exit("no kv-cache-tester data — see stoa.kvct.download")

    uniq = sorted(set(convs))
    train_convs = set(uniq[: len(uniq) // 2])           # conversation-disjoint split
    tr = [i for i, c in enumerate(convs) if c in train_convs]
    te = [i for i, c in enumerate(convs) if c not in train_convs]

    # Placement workload: evaluation conversations only. `split` stays fixed, and every
    # place_with_beliefs call below is given the same instant explicitly.
    keep = {ids[i] for i in te}
    wl.items = [i for i in wl.items if i.item_id in keep]
    wl.traces = {k: v for k, v in wl.traces.items() if k in keep}
    wl.accesses = [(t, b) for t, b in wl.accesses if b in keep]

    stats = wl.prefix_stats(split)
    fut = wl.future_counts(split)
    reused = [v for v in fut.values() if v > 0]
    mean_reuse = sum(reused) / max(1, len(reused))

    def place(belief):
        return place_with_beliefs_at(wl, belief, split)

    c_pref = place({i.item_id: stats[i.item_id]["count"] for i in wl.items})
    c_orac = place({i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
    span = c_pref - c_orac

    print("\nkv-cache-tester: prediction -> placement (corrected protocol)")
    print(f"  {len(uniq)} conversations, {len(train_convs)} train / "
          f"{len(uniq) - len(train_convs)} eval (disjoint) | eval blocks {len(te):,}")
    print(f"  headroom {100 * span / c_pref:.1f}%  "
          f"(prefix-greedy {c_pref:,.0f} vs oracle {c_orac:,.0f})")
    print(f"{'features':>22} | {'AUC':>6} | {'p x mean':>10} | {'count regr':>11}")
    print("-" * 62)
    out_rows = []
    for fs in FEATURE_SETS:
        Xtr = [rows[fs][i] for i in tr]
        Xte = [rows[fs][i] for i in te]
        m = fit_logistic(Xtr, [y_bin[i] for i in tr], epochs=3000, lr=2.0)
        pr = predict(m, Xte)
        a = auc(pr, [y_bin[i] for i in te])
        cls = place({ids[i]: p * mean_reuse for i, p in zip(te, pr)})
        rg = predict_counts(fit_regression(Xtr, [y_cnt[i] for i in tr]), Xte)
        reg = place({ids[i]: v for i, v in zip(te, rg)})
        cap = lambda c: 100.0 * (c_pref - c) / span if span > 0 else 0.0  # noqa: E731
        print(f"{fs:>22} | {a:6.3f} | {cap(cls):+9.1f}% | {cap(reg):+10.1f}%")
        out_rows.append({"features": fs, "auc": round(a, 4),
                         "classifier_headroom_pct": round(cap(cls), 2),
                         "regression_headroom_pct": round(cap(reg), 2)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"convs": args.convs, "source": "kv-cache-tester",
                   "protocol": "single workload, single split instant, "
                               "conversation-disjoint train/eval"},
        "prefix_greedy": round(c_pref, 2), "oracle": round(c_orac, 2),
        "headroom_pct": round(100 * span / c_pref, 2), "results": out_rows}, indent=2))
    print(f"\nwrote {out}")


def place_with_beliefs_at(wl, belief, split):
    """`place_with_beliefs` pinned to an explicit split instant.

    The library helper takes a fraction and recomputes the split from the workload it is
    given; after filtering the workload that fraction no longer names the same instant.
    Passing the instant through removes the only remaining way the two could drift apart.
    """
    from stoa.sequential import _TIERS
    from stoa.credit import weighted_cost
    from stoa.eval.oracle import DEFAULT_WEIGHTS
    from stoa.gnn import LEGAL_ACTIONS
    from stoa.simulator import TieringSimulator

    sim = TieringSimulator()
    w = DEFAULT_WEIGHTS
    future = wl.future_counts(split)
    caps = sim.tier_capacities(wl.total_bytes)
    remaining = {t: caps[t] for t in _TIERS}
    total = 0.0
    for it in sorted(wl.items, key=lambda i: belief.get(i.item_id, 0.0), reverse=True):
        n_hat = belief.get(it.item_id, 0.0)
        n_true = future.get(it.item_id, 0)
        best, best_c = None, float("inf")
        for a in LEGAL_ACTIONS:
            if remaining[a.tier] < it.size_bytes:
                continue
            c = weighted_cost(sim.cost_of_placement(a, n_hat, it.representation), w)
            if c < best_c:
                best, best_c = a, c
        if best is None:
            continue
        remaining[best.tier] -= it.size_bytes
        total += weighted_cost(sim.cost_of_placement(best, n_true, it.representation), w)
    return total


if __name__ == "__main__":
    main()
