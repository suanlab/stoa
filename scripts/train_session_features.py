#!/usr/bin/env python3
"""RQ2, corrected form: does SESSION-level information make reuse predictable?

The block-local experiment (train_on_mooncake.py) captured ~0.5% of an ~80% headroom,
and the diagnosis explained why: a binary oracle that knows only *whether* a block is
reused captures 99.7% of the headroom, and that binary event is dominated by something
block statistics cannot see -- whether the conversation the block belongs to continues.

So we lift the features to the session level. Mooncake requests in one conversation
share their prefix blocks, so a request's first hash id identifies its session. For
each block we then know how alive its session is: how recently it was touched, how
many requests it has accumulated, how fast they arrive.

We compare, on the same leak-free split (prefix features / future label):
    block-only     : count, recency, age, inter-arrival, position-in-request
    block+session  : the above + session recency / size / rate / age
reporting both ranking quality (AUC for "will be reused") and the share of placement
headroom each captures. Torch-free, CPU. Usage: python3 scripts/train_session_features.py
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

from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}


def parse_requests(path: str, max_requests: int):
    """[(t, session_id, [block_ids])] — session inferred from the shared prefix block."""
    out = []
    with Path(path).open() as fh:
        for t, line in enumerate(fh):
            if t >= max_requests:
                break
            rec = json.loads(line)
            hs = rec.get("hash_ids", [])
            if not hs:
                continue
            out.append((t, f"s{hs[0]}", [f"b{h}" for h in hs]))
    return out


def build_features(path: str, max_requests: int, split_frac: float = 0.5):
    """Per-block features observable in the prefix + binary 'reused later' label."""
    reqs = parse_requests(path, max_requests)
    if not reqs:
        return [], [], [], []
    t_max = reqs[-1][0]
    split = int(split_frac * t_max)

    blk_steps: dict[str, list[int]] = defaultdict(list)
    blk_pos: dict[str, list[int]] = defaultdict(list)
    blk_session: dict[str, str] = {}
    sess_steps: dict[str, list[int]] = defaultdict(list)
    for t, sid, blocks in reqs:
        sess_steps[sid].append(t)
        for pos, b in enumerate(blocks):
            blk_steps[b].append(t)
            blk_pos[b].append(pos)
            blk_session.setdefault(b, sid)

    X_blk, X_ses, y, ids = [], [], [], []
    for b, steps in blk_steps.items():
        past = [s for s in steps if s < split]
        if not past:
            continue
        cnt = len(past)
        recency = split - past[-1]
        age = split - past[0]
        pos = sum(blk_pos[b]) / len(blk_pos[b])
        blk = [1.0, math.log1p(cnt), recency / max(split, 1), age / max(split, 1),
               (age / cnt) / max(split, 1), math.log1p(pos)]
        # --- session state: is the conversation this block belongs to still alive? ---
        sid = blk_session[b]
        s_past = [s for s in sess_steps[sid] if s < split]
        s_cnt = len(s_past)
        s_rec = (split - s_past[-1]) if s_past else split
        s_age = (split - s_past[0]) if s_past else 0
        s_rate = s_cnt / max(s_age, 1)
        ses = [math.log1p(s_cnt), s_rec / max(split, 1), s_age / max(split, 1), s_rate]
        X_blk.append(blk)
        X_ses.append(blk + ses)
        y.append(1.0 if any(s >= split for s in steps) else 0.0)
        ids.append(b)
    return X_blk, X_ses, y, ids


def _standardizer(X):
    """Per-feature mean/std from the TRAINING set (index 0 is the bias column).

    Without this, gradient descent crawls along the low-variance directions and the
    fit lands far below what a single raw feature achieves -- which is exactly what
    happened before this was added (in-trace AUC 0.56 vs 0.94 for log(count) alone).
    """
    d = len(X[0])
    n = len(X)
    mean = [0.0] * d
    std = [1.0] * d
    for j in range(1, d):
        col = [x[j] for x in X]
        mu = sum(col) / n
        var = sum((v - mu) ** 2 for v in col) / n
        mean[j] = mu
        std[j] = math.sqrt(var) or 1.0
    return mean, std


def _z(X, mean, std):
    return [[x[j] if j == 0 else (x[j] - mean[j]) / std[j] for j in range(len(x))] for x in X]


def fit_logistic(X, y, epochs=4000, lr=1.0):
    """Standardized logistic regression, Newton/IRLS-style via numpy when available.

    Pure-Python gradient descent under-fit badly here (AUC 0.59 when a single raw
    feature reaches 0.94), so we solve properly: vectorized gradient steps with a
    generous budget, falling back to the slow loop only if numpy is missing.
    """
    mean, std = _standardizer(X)
    Xz = _z(X, mean, std)
    d = len(X[0])
    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        w = [0.0] * d
        n = len(Xz)
        for _ in range(epochs):
            g = [0.0] * d
            for xi, yi in zip(Xz, y):
                z = sum(wj * xj for wj, xj in zip(w, xi))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                for j in range(d):
                    g[j] += (p - yi) * xi[j]
            for j in range(d):
                w[j] -= lr * (g[j] / n + 1e-5 * w[j])
        return {"w": w, "mean": mean, "std": std}

    A = np.asarray(Xz, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    w = np.zeros(d)
    for _ in range(epochs):
        z = np.clip(A @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = A.T @ (p - b) / len(b) + 1e-5 * w
        w -= lr * grad
    return {"w": w.tolist(), "mean": mean, "std": std}


def predict(model, X):
    w, mean, std = model["w"], model["mean"], model["std"]
    out = []
    for xi in _z(X, mean, std):
        z = sum(wj * xj for wj, xj in zip(w, xi))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return out


def auc(scores, labels):
    """Mann-Whitney AUC with MID-RANKS for ties.

    Ties must get averaged ranks. Sorting `(score, label)` pairs instead lets the label
    break ties, which pushes positives to the top of every tie group and fabricates
    signal: a raw count feature with thousands of tied values scored 0.94 that way and
    0.55 once ties were handled correctly.
    """
    pos = sum(1 for l in labels if l > 0)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0                      # 1-indexed mid-rank
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    rank_sum = sum(r for r, l in zip(ranks, labels) if l > 0)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    p = argparse.ArgumentParser(description="Session-level reuse prediction (RQ2)")
    p.add_argument("--requests", type=int, default=1500,
                   help="applies to BOTH traces unless --train-requests is given")
    # One knob for two roles confounds the sweep: raising --requests lengthens the evaluation
    # trace (moving its split instant and changing which blocks exist) AND the cross-trace
    # training set at the same time, so a rising AUC cannot be attributed to either. The
    # conversation trace is exhausted at 12,031 requests, which made this visible: past that
    # point only the training set grows, and AUC goes DOWN.
    p.add_argument("--train-requests", type=int, default=0,
                   help="0 = same as --requests; set separately to attribute the effect")
    p.add_argument("--out", type=str, default="experiments/mooncake_session_rq2.json")
    args = p.parse_args()

    n_train = args.train_requests or args.requests
    feats = {}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path}")
        # Two feature sets per trace: one at the EVAL length (used when this trace is the
        # evaluation trace) and one at the TRAIN length (used when it is the training trace).
        # Identical unless --train-requests is given, so default behaviour is unchanged.
        Xb, Xs, y, ids = build_features(path, args.requests)
        Xb_tr, Xs_tr, y_tr, _ = ((Xb, Xs, y, ids) if n_train == args.requests
                                 else build_features(path, n_train))
        feats[name] = {"Xb": Xb, "Xs": Xs, "y": y, "ids": ids,
                       "Xb_tr": Xb_tr, "Xs_tr": Xs_tr, "y_tr": y_tr,
                       "wl": load_mooncake(path, max_requests=args.requests)}

    rows = []
    print(f"\nSession-level reuse prediction (cross-trace, first {args.requests} requests)")
    print(f"{'eval':<14} | {'AUC block':>9} | {'AUC +session':>12} | "
          f"{'headroom: block':>15} | {'+session':>9}")
    print("-" * 78)
    for ev in TRACES:
        tr = next(n for n in TRACES if n != ev)
        wb = fit_logistic(feats[tr]["Xb_tr"], feats[tr]["y_tr"])
        ws = fit_logistic(feats[tr]["Xs_tr"], feats[tr]["y_tr"])
        d = feats[ev]
        pb, ps = predict(wb, d["Xb"]), predict(ws, d["Xs"])
        a_b, a_s = auc(pb, d["y"]), auc(ps, d["y"])

        wl = d["wl"]
        st = max(1, wl.time_split(0.5))
        stats = wl.prefix_stats(st)
        fut = wl.future_counts(st)
        mean_reuse = (sum(v for v in fut.values() if v > 0)
                      / max(1, sum(1 for v in fut.values() if v > 0)))
        prefix_belief = {it.item_id: stats[it.item_id]["count"] for it in wl.items}
        true_belief = {it.item_id: float(fut.get(it.item_id, 0)) for it in wl.items}
        c_pref = place_with_beliefs(wl, prefix_belief)
        c_orac = place_with_beliefs(wl, true_belief)
        span = c_pref - c_orac

        caps = {}
        for tag, pred in (("block", pb), ("session", ps)):
            belief = {i: pr * mean_reuse for i, pr in zip(d["ids"], pred)}
            c = place_with_beliefs(wl, belief)
            caps[tag] = 100.0 * (c_pref - c) / span if span > 0 else 0.0
        print(f"{ev:<14} | {a_b:9.3f} | {a_s:12.3f} | {caps['block']:14.1f}% | {caps['session']:8.1f}%")
        rows.append({"eval": ev, "train": tr, "auc_block": round(a_b, 4),
                     "auc_session": round(a_s, 4),
                     "headroom_block_pct": round(caps["block"], 2),
                     "headroom_session_pct": round(caps["session"], 2),
                     "prefix_greedy": round(c_pref, 2), "oracle": round(c_orac, 2)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"requests": args.requests,
                                          "train_requests": n_train,
                                          "note": ("eval and train lengths are separate; equal "
                                                   "unless --train-requests is passed")},
                               "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
