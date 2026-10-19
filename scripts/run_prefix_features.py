#!/usr/bin/env python3
"""H4: does the prefix-chain structure make unseen blocks predictable?

Mooncake hash ids are prefix-chained: a block at depth k is reused only if the request re-sends its
prefix. At a block's arrival these features are observable: how often its parent (the block at depth
k-1 of the same request) was accessed BEFORE this request, whether the parent is itself new in this
request, and how many distinct children the parent has had so far. The base features are those of
run_arrival_admission.py; the model and labels are fitted exactly as there (windowed pre-split labels).
H4 predicts the prefix features lift arrival AUC on production from 0.56-0.61 to >= 0.70 and widen the
order arm (learned vs arrival order) to >= 5 points (falsified if AUC < 0.65).
Usage: python3 scripts/run_prefix_features.py [--tier-only]
"""
from __future__ import annotations

import sys

TIER_ONLY = "--tier-only" in sys.argv
if TIER_ONLY:
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))
    from stoa.environment import Action, Representation, Timing
    _T = [a for a in Action.space() if a.representation == Representation.PLAINTEXT
          and a.timing == Timing.ONLINE]
    Action.space = staticmethod(lambda: list(_T))

import json  # noqa: E402
import math  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES, arrival_features  # noqa: E402
from run_arrival_diagnostics import _auc  # noqa: E402
from stoa.causal import analogue_magnitudes, split_by_arrival, windowed_training_labels  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs, prefix_greedy_cost, quantile_match  # noqa: E402
from train_session_features import fit_logistic, parse_requests, predict  # noqa: E402

EPS = 1e-9


def prefix_features(path):
    """Per block, at its first appearance: parent's prior count, parent-new flag, parent's children."""
    count = defaultdict(int)
    children = defaultdict(set)
    feats, seen = {}, set()
    for _t, _sid, blocks in parse_requests(path, 10 ** 9):
        new_here = set()
        for k, b in enumerate(blocks):
            if b in seen:
                continue
            par = blocks[k - 1] if k > 0 else None
            pc = count[par] if par else 0
            feats[b] = [math.log1p(pc), 1.0 if (par in new_here) else 0.0,
                        math.log1p(len(children[par])) if par else 0.0, 1.0 if pc >= 2 else 0.0]
            new_here.add(b)
            seen.add(b)
        for k, b in enumerate(blocks):              # update AFTER the request: causal
            count[b] += 1
            if k > 0:
                children[blocks[k - 1]].add(b)
    return feats


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        full = str(ROOT / path)
        wl = load_mooncake(full)
        X, y, ids, arrival = arrival_features(full, 10 ** 9)
        pf = prefix_features(full)
        Xp = [x + pf[b] for x, b in zip(X, ids)]
        st = max(1, wl.time_split(0.5))
        _, ev = split_by_arrival(arrival, st)
        steps = {b: wl.traces[b].access_steps for b in ids}
        fit_idx, fit_lab = windowed_training_labels(arrival, ids, steps, st, st // 2)
        lab01 = [1.0 if v > 0 else 0.0 for v in fit_lab]
        truth = [y[i] > 0 for i in ev]
        pre = wl.prefix_stats(st)
        fut = wl.future_counts(st)
        ev_ids = [ids[i] for i in ev]
        ref = analogue_magnitudes((v["count"] for v in pre.values()), len(ev_ids))
        base = {i.item_id: float(pre[i.item_id]["count"]) for i in wl.items}
        c_pref = prefix_greedy_cost(wl)
        span = c_pref - place_with_beliefs(wl, {b: float(v) for b, v in fut.items()})
        out = {"trace": name, "actions": "tier" if TIER_ONLY else "all"}
        scores = {}
        for tag, XX in (("base", X), ("prefix", Xp)):
            w = fit_logistic([XX[i] for i in fit_idx], lab01)
            sc = predict(w, [XX[i] for i in ev])
            out[f"auc_{tag}"] = round(_auc(sc, truth), 3)
            scores[tag] = sc
        scores["arrival"] = [-float(k) for k in range(len(ev))]
        for tie in ("new-behind", "new-ahead"):
            for tag, sc in scores.items():
                order = sorted(range(len(sc)), key=lambda k: (sc[k], -k))
                rank = [0.0] * len(sc)
                for r, k in enumerate(order):
                    rank[k] = float(r)
                vals = quantile_match(rank, ref)
                bel = dict(base)
                e = EPS if tie == "new-ahead" else 0.0
                bel.update({b: v + e for b, v in zip(ev_ids, vals)})
                out[f"captured_{tag}_{tie}"] = round(100.0 * (c_pref - place_with_beliefs(wl, bel)) / span, 2)
        rows.append(out)
        print(out)
    dest = ROOT / "experiments" / f"prefix_features{'_tier' if TIER_ONLY else ''}.json"
    dest.write_text(json.dumps({"config": {"features": "base + parent prior count, parent-new, "
                                           "parent children, parent count>=2"}, "rows": rows}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
