#!/usr/bin/env python3
"""One physical ledger for static and online policies, and prices by path (§BJ).

A review (2026-11-09, R2/R3) found two asymmetries in how the paper bills static against online
placements, both stated in the text but never measured:

  * a block created after the split costs nothing wherever it lands. Its KV is computed on the GPU, so
    a static placement that puts it on the remote tier performs a GPU -> remote store the ledger does not
    charge, while LRU, which creates on the GPU, pays for every demotion that follows;
  * a static assignment is realized by one rearrangement at the split, which is free, while an online
    policy starting from the heuristic's placement pays for every move.

And one transfer of a measurement: the measured move band (0.75-1.87 of read+write) is for GPU <-> host
copies, but most of the arrival-time policies' moves and many of LRU's touch the remote tier, which was
not measured.

This script bills every policy on one ledger, from the heuristic's placement of the visible blocks at
the split, tier-only, split 0.5:

  access      every post-split read at the tier the block holds (creations unbilled, as everywhere)
  store       every block created after the split: GPU -> the tier it is first placed in (0 if the GPU)
  rearrange   static policies: every visible block whose tier differs from the heuristic's, once
  moves       online policies: every move of the replay (run_reactive_cost.replay), under each ledger of
              run_migration_ledger.py (reread, exclusive, retained, write-through)

Store and rearrangement are priced read(src) + write(dst) under every ledger. Every priced transfer is
put in one of two classes, M (between GPU and host memory: the measured path) and U (any path that
touches the remote tier or the disk), with its own price multiple alpha_M, alpha_U. Policies:
heuristic, oracle (both static, the references), seen-once rule (static), multi-tier LRU and cache on
second hit (online). The arrival-time policies are not here: their creation tiers are not recorded by
their replays.

Reported: absolute cost components per policy; captured share (heuristic - policy) / (the oracle's
access-only advantage) over a grid of (alpha_M, alpha_U) -- one fixed scale, because charging stores and
the rearrangement can make the static oracle dearer than the heuristic, so a share of each ledger's own
oracle gap is undefined there; the oracle's own share on that scale; for each online policy, the alpha_U
at which it breaks even with alpha_M at each end of the measured band. Checked: with stores and
rearrangement left out and alpha_M = alpha_U, LRU and second-hit reproduce run_migration_ledger's
summary to 0.01 points.
Usage: python3 scripts/run_physical_ledger.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reactive_cost as rc  # noqa: E402  (patches Action.space to tier-only on import)
import run_online_admission as oa  # noqa: E402
from run_arrival_admission import TRACES  # noqa: E402
from run_evidence_ladder import TIER, ordered  # noqa: E402
from run_migration_ledger import price, summarize  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

LEDGERS = ("reread", "exclusive", "retained", "write-through")
GRID = (0.0, 0.75, 1.0, 1.87, 3.0)
BAND = (0.75, 1.87)


def cls(s, d):
    """'M' for a transfer between the GPU (0) and host memory (1), the measured path; 'U' otherwise."""
    return "M" if {s, d} == {0, 1} else "U"


def rw(s, d, tiers, read_ms, lam):
    return lam * (read_ms[tiers[s]] + read_ms[tiers[d]]) if s != d else 0.0


def main() -> int:
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    lam = w.lambda_latency
    out = {"config": {"split": 0.5, "actions": "tier only", "ledgers": list(LEDGERS), "grid": list(GRID),
                      "band_M": list(BAND), "classes": {"M": "GPU <-> host memory (measured)",
                                                        "U": "any transfer touching remote or disk"},
                      "store": "GPU -> creation tier, read+write, every policy",
                      "rearrange": "static policies: heuristic tier -> own tier at the split, read+write"},
           "rows": []}
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        st = max(1, wl.time_split(0.5))
        pre, fut = wl.prefix_stats(st), wl.future_counts(st)
        ids = [i.item_id for i in wl.items]
        index = {b: j for j, b in enumerate(ids)}
        sizes = [float(i.size_bytes) for i in wl.items]
        cm = oa.cost_model(sim, w, rc.TIER_ONLY, wl.items[0].representation)
        tiers, read_ms = cm["tiers"], cm["read_ms"]
        tidx = {t: j for j, t in enumerate(tiers)}
        caps = sim.tier_capacities(wl.total_bytes)
        slots = [len(ids) if caps[t] == float("inf") else int(caps[t] // sizes[0]) for t in tiers]
        events = oa.post_split_events(str(ROOT / path), st, 10 ** 9, index)
        counts = [float(pre[b]["count"]) for b in ids]
        is_new = [c == 0 for c in counts]
        once = [counts[j] == 1 for j in range(len(ids))]
        f = [float(fut.get(b, 0)) for b in ids]

        # static tier assignments (each validated against place_with_beliefs)
        heur = oa.static_tiers(oa.rank_order(counts, is_new, False), sizes, caps, tiers)
        so_bel = [0.0 if once[j] else counts[j] for j in range(len(ids))]
        seen_once = oa.static_tiers(sorted(range(len(ids)), key=lambda j: (-so_bel[j], once[j], j)), sizes, caps, tiers)
        orac = oa.static_tiers(sorted(range(len(ids)), key=lambda j: (-f[j], j)), sizes, caps, tiers)
        all_actions = list(sq.LEGAL_ACTIONS)
        sq.LEGAL_ACTIONS = TIER
        try:
            once_ids = {ids[j] for j in range(len(ids)) if once[j]}
            ref = {"heuristic": sq.place_with_beliefs(wl, {b: counts[j] for j, b in enumerate(ids) if not is_new[j]}),
                   "seen-once": sq.place_with_beliefs(ordered(wl, once_ids), {b: so_bel[j] for j, b in enumerate(ids)
                                                                              if not is_new[j]}),
                   "oracle": sq.place_with_beliefs(wl, {b: f[j] for j, b in enumerate(ids)})}
        finally:
            sq.LEGAL_ACTIONS = all_actions
        statics = {"heuristic": heur, "seen-once": seen_once, "oracle": orac}
        comp = {}
        for pol, tier_of in statics.items():
            acc = oa.bill_static(events, tier_of, cm)
            assert abs(acc - ref[pol]) <= 1e-6 * acc, f"{name} {pol}: static tiers do not reproduce place_with_beliefs"
            store, rearr = Counter(), Counter()
            for j in range(len(ids)):
                t = tidx[tier_of[j]]
                if is_new[j]:
                    store[cls(0, t)] += rw(0, t, tiers, read_ms, lam)
                elif t != tidx[heur[j]]:
                    rearr[cls(tidx[heur[j]], t)] += rw(tidx[heur[j]], t, tiers, read_ms, lam)
            comp[pol] = {"access": acc, "store": dict(store), "rearrange": dict(rearr),
                         "moves": {led: {} for led in LEDGERS}}
        init = {j: tidx[heur[j]] for j in range(len(ids)) if not is_new[j]}
        moves_of, creations_of = {}, {}
        for pol in ("lru", "second-hit"):
            led: dict = {}
            c = rc.replay(pol, init, is_new, slots, cm, events, w, ledger=led)
            moves_of[pol], creations_of[pol] = led["moves"], led["creations"]
            store = Counter()
            for t, n in led["created_in"].items():
                store[cls(0, t)] += n * rw(0, t, tiers, read_ms, lam)
            mv = {}
            for L in LEDGERS:
                m = Counter()
                for (s, d, k, cl), n in led["moves"].items():
                    m[cls(s, d)] += n * price(s, d, k, cl, L, tiers, read_ms, lam)
                if L == "write-through":
                    m["M"] += led["creations"] * lam * read_ms[tiers[1]]
                mv[L] = dict(m)
            comp[pol] = {"access": c["zero"], "store": dict(store), "rearrange": {}, "moves": mv,
                         "created_in": {tiers[t].value: n for t, n in sorted(led["created_in"].items())}}

        def total(pol, L, aM, aU, physical=True):
            v = comp[pol]
            a = {"M": aM, "U": aU}
            parts = list(v["moves"][L].items())
            if physical:
                parts += list(v["store"].items()) + list(v["rearrange"].items())
            return v["access"] + sum(a[k] * x for k, x in parts)

        span0 = comp["heuristic"]["access"] - comp["oracle"]["access"]

        def share(pol, L, aM, aU, physical=True):
            """On one scale for every price: the oracle's access-only advantage, as the online shares
            are. A share-of-its-own-ledger denominator is not usable: charging stores and the
            split-time rearrangement can make the static oracle dearer than the heuristic."""
            return 100.0 * (total("heuristic", L, aM, aU, physical) - total(pol, L, aM, aU, physical)) / span0

        # the legacy convention (no stores, statics free) reproduces run_migration_ledger's summary
        for pol in ("lru", "second-hit"):
            sz = 100.0 * (comp["heuristic"]["access"] - comp[pol]["access"]) / span0
            want = summarize(sz, moves_of[pol], span0, tiers, read_ms, lam, creations_of[pol])
            for L in LEDGERS:
                got = share(pol, L, 1.0, 1.0, physical=False)
                assert abs(got - want[L]["captured_pct"]) < 0.01, f"{name} {pol} {L}: {got:.2f} != {want[L]}"
        pols = ("oracle", "seen-once", "lru", "second-hit")
        grid = {L: {pol: {f"{aM}/{aU}": round(share(pol, L, aM, aU), 2) for aM in GRID for aU in GRID}
                    for pol in pols} for L in LEDGERS}
        be = {}
        for L in LEDGERS:
            for pol in ("lru", "second-hit"):
                for aM in BAND:
                    # the share is linear in alpha_U: n0 + n1 * alpha_U (fixed denominator)
                    n0 = total("heuristic", L, aM, 0.0) - total(pol, L, aM, 0.0)
                    n1 = total("heuristic", L, aM, 1.0) - total(pol, L, aM, 1.0) - n0
                    if n1 == 0:
                        v = {"pays": "always" if n0 > 0 else "never"}
                    else:
                        r = -n0 / n1
                        if n1 < 0:
                            v = {"pays": "never" if r <= 0 else "below", "alpha_U": round(r, 4)}
                        else:
                            v = {"pays": "always" if r <= 0 else "above", "alpha_U": round(r, 4)}
                    be[f"{L}|{pol}|{aM}"] = v

        band = {L: {pol: [min(share(pol, L, a, a) for a in BAND), max(share(pol, L, a, a) for a in BAND)]
                    for pol in pols} for L in LEDGERS}
        row = {"trace": name,
               "components": {pol: {"access": round(v["access"], 3),
                                    "store": {k: round(x, 3) for k, x in v["store"].items()},
                                    "rearrange": {k: round(x, 3) for k, x in v["rearrange"].items()},
                                    "moves": {L: {k: round(x, 3) for k, x in m.items()} for L, m in v["moves"].items()},
                                    **({"created_in": v["created_in"]} if "created_in" in v else {})}
                              for pol, v in comp.items()},
               "span_access_only": round(span0, 3),
               "physical_share_in_band": {L: {p_: [round(x, 2) for x in r] for p_, r in v.items()} for L, v in band.items()},
               "alpha_U_break_even": be, "grid": grid}
        out["rows"].append(row)
        print(name, "band", row["physical_share_in_band"], flush=True)
        print("   break-even alpha_U", be, flush=True)
    (ROOT / "experiments" / "physical_ledger.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/physical_ledger.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
