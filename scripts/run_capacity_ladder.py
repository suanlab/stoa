#!/usr/bin/env python3
"""Model-capacity ladder: does a stronger predictor convert into a better placement?

The paper's central claim is a decoupling -- ranking quality rises from AUC 0.57 to 0.93
while the placement benefit stays inside [+0.8%, +2.9%] of the oracle gap. The obvious
objection is that we simply used too weak a model. This script answers it directly by
holding EVERYTHING fixed except model capacity:

  * same features (prefix count, recency, age, inter-arrival, mean position)
  * same split instant (Workload.time_split(0.5)), same cross-trace train/eval pairing
  * same placement routine, same billing at the true future

and sweeping the rungs

  constant  -> no learning at all; the floor
  count     -> a single raw feature, log(prefix count); what "no model" already achieves
  linear    -> standardized logistic regression (the paper's model)
  gbdt      -> 200 depth-4 boosted trees (stoa.gbdt; hand-rolled, numpy)
  mlp       -> 2-layer torch MLP, 64 hidden units (skipped if torch is absent)
  oracle    -> the TRUE future counts as the belief; the ceiling

If capacity were the limitation, `gbdt`/`mlp` would land between `linear` and `oracle`. If
the decoupling is real, they track `linear` on placement no matter what they do to AUC.

Full traces by default -- trace length is a free parameter and fixing it at a convenient
value is one of the defects the paper catalogs. CPU, a few minutes.

Usage:  python3 scripts/run_capacity_ladder.py [--requests N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random as _random  # noqa: E402

from stoa.gbdt import GBDT  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import (attainable_ceiling, captured_fractions,  # noqa: E402
                             place_with_beliefs, quantile_match)
from train_session_features import TRACES, auc, build_features, fit_logistic, predict  # noqa: E402


def _mlp_fit_predict(Xtr, ytr, Xte, seed=0, hidden=64, epochs=300):
    """2-layer MLP, standardized inputs. Returns None if torch is unavailable.

    Standardization uses TRAINING statistics only -- applying the evaluation trace's own
    mean/std would leak the evaluation distribution into the model, the same reference-frame
    error the paper catalogs.
    """
    try:
        import torch
    except ImportError:
        return None
    # Pin threads: 300 epochs of Adam amplify floating-point reduction-order differences, and
    # the MLP rung's cross-trace AUC moves by 0.006 between 1 and 112 threads. That rung
    # supplies a third of the "linear/GBDT/MLP agree" claim, so it must not depend on the
    # reproducer's core count.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True).clamp_min(1e-8)
    Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd
    y_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr_t.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr_t), y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(Xte_t)).squeeze(1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-capacity ladder on Mooncake")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/capacity_ladder.json")
    args = ap.parse_args()
    n_req = args.requests or 10 ** 9

    feats = {}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — see src/stoa/mooncake.py for the download command")
        Xb, _Xs, y, ids = build_features(path, n_req)
        feats[name] = {"X": Xb, "y": y, "ids": ids,
                       "wl": load_mooncake(path, max_requests=args.requests or None)}

    rows = []
    print(f"\nModel-capacity ladder ({'full traces' if not args.requests else f'{n_req} requests'})")
    print(f"{'eval':<14} | {'rung':<9} | {'AUC x':>6} | {'of gap':>9} | {'of reach':>10} | {'AUC in':>9}")
    print("  'of gap' = % of the full hindsight gap (what the paper reported)")
    print("  'of reach' = % of what a CAUSAL policy could attain -- the honest denominator")
    print("-" * 78)
    for ev in TRACES:
        tr = next(n for n in TRACES if n != ev)
        d, dtr = feats[ev], feats[tr]

        # --- references, computed once per eval trace, all billed on the true future ---
        wl = d["wl"]
        st = max(1, wl.time_split(0.5))
        stats, fut = wl.prefix_stats(st), wl.future_counts(st)
        c_pref = place_with_beliefs(wl, {i.item_id: stats[i.item_id]["count"] for i in wl.items})
        c_orac = place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
        c_ceil = attainable_ceiling(wl, d["ids"])
        span, reach = c_pref - c_orac, c_pref - c_ceil

        # Every rung is placed on ONE common magnitude scale -- the clairvoyant belief's own
        # multiset -- so only the ORDERING differs between rungs. Without this the min-max
        # transform moves items across the argmin's belief thresholds differently per model
        # and the ladder measures the transform as much as the model.
        ref_magnitudes = [float(fut.get(i, 0)) for i in d["ids"]]

        def captured(scores_by_rank) -> tuple[float, float]:
            """(% of the full hindsight gap, % of what a causal policy could attain)."""
            matched = quantile_match(list(scores_by_rank), ref_magnitudes)
            c = place_with_beliefs(wl, dict(zip(d["ids"], matched)))
            return (100.0 * (c_pref - c) / span if span > 0 else 0.0,
                    100.0 * (c_pref - c) / reach if reach > 0 else 0.0)

        # --- the rungs: each produces a per-block score on the EVAL trace ---
        scores: dict[str, list[float] | None] = {}
        scores["constant"] = [0.5] * len(d["y"])
        # index 1 of the feature vector is log1p(prefix count) -- see build_features
        scores["count"] = [x[1] for x in d["X"]]
        scores["linear"] = predict(fit_logistic(dtr["X"], dtr["y"]), d["X"])
        gb = GBDT(n_trees=200, lr=0.1, max_depth=4).fit(dtr["X"], dtr["y"])
        scores["gbdt"] = [float(v) for v in gb.predict(d["X"])]
        scores["mlp"] = _mlp_fit_predict(dtr["X"], dtr["y"], d["X"], seed=args.seed)
        # Zero-information control. A metric whose random arm scores like its learned arms is
        # not measuring ranking value, and this one previously did.
        _rr = _random.Random(args.seed + 991)
        scores["random"] = [_rr.random() for _ in d["y"]]

        # --- in-trace ranking control: is the extra capacity being USED at all? ---
        # Fit and score on a block-disjoint 70/30 split of the SAME trace. This is leak-free
        # for ranking (disjoint blocks; features from the prefix, labels from the future) but
        # NOT usable for placement: placement is scored over the whole workload, so the model
        # would have been fitted on the very blocks whose placement it is then credited for.
        # We therefore report in-trace AUC only, as a capacity diagnostic.
        order = list(range(len(d["y"])))
        _random.Random(args.seed).shuffle(order)
        k = int(0.7 * len(order))
        a_idx, b_idx = order[:k], order[k:]
        Xa = [d["X"][i] for i in a_idx]
        ya = [d["y"][i] for i in a_idx]
        Xb_ = [d["X"][i] for i in b_idx]
        yb = [d["y"][i] for i in b_idx]
        in_trace = {"linear": predict(fit_logistic(Xa, ya), Xb_),
                    "gbdt": [float(v) for v in
                             GBDT(n_trees=200, lr=0.1, max_depth=4).fit(Xa, ya).predict(Xb_)],
                    "mlp": _mlp_fit_predict(Xa, ya, Xb_, seed=args.seed)}
        in_auc = {k2: (auc(v, yb) if v is not None else None) for k2, v in in_trace.items()}

        for rung in ("constant", "random", "count", "linear", "gbdt", "mlp"):
            s = scores[rung]
            if s is None:
                print(f"{ev:<14} | {rung:<9} |   skipped (torch unavailable)")
                continue
            a = auc(s, d["y"])
            cap_full, cap_reach = captured(s)
            ia = in_auc.get(rung)
            print(f"{ev:<14} | {rung:<9} | {a:6.3f} | {cap_full:8.1f}% | {cap_reach:9.1f}% | "
                  f"{(f'{ia:.3f}' if ia is not None else '—'):>9}")
            rows.append({"eval": ev, "train": tr, "rung": rung,
                         "auc_cross_trace": round(a, 4),
                         "captured_pct": round(cap_full, 2),
                         "captured_of_attainable_pct": round(cap_reach, 2),
                         "auc_in_trace": round(ia, 4) if ia is not None else None})

        ceil_full = 100.0 * reach / span if span > 0 else 0.0
        print(f"{ev:<14} | {'ceiling':<9} | {1.0:6.3f} | {ceil_full:8.1f}% | {100.0:9.1f}% | {1.0:9.3f}")
        print(f"{ev:<14} | {'oracle':<9} | {1.0:6.3f} | {100.0:8.1f}% | "
              f"{'(unreachable)':>10} | {1.0:9.3f}")
        rows.append({"eval": ev, "train": "—", "rung": "ceiling", "auc_cross_trace": 1.0,
                     "captured_pct": round(ceil_full, 2),
                     "captured_of_attainable_pct": 100.0, "auc_in_trace": 1.0})
        rows.append({"eval": ev, "train": "—", "rung": "oracle", "auc_cross_trace": 1.0,
                     "captured_pct": 100.0, "captured_of_attainable_pct": None,
                     "auc_in_trace": 1.0})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"requests": args.requests or "full", "seed": args.seed,
                    "protocol": "cross-trace; prefix features; identical split, placement and billing"},
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
