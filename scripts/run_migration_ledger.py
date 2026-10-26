#!/usr/bin/env python3
"""Who pays for a move depends on the hierarchy: the online conclusion under four ledgers (§AY).

The online replays price every move as one read from the source plus one write to the destination.
An independent review (2026-10-25) asked whether that bills some transfers twice and whether it
assumes an exclusive hierarchy, where real KV stores may keep lower copies. Two things are true of KV
blocks that the original ledger ignored: a block is immutable once computed, and a promotion on an
access happens because the serving read has just brought the block up. So each move is recorded with
its kind and whether the block still has a clean copy at the destination (it resided there before),
and billed four ways (lambda_L x read latency, as `run_online_admission.migration_cost`):

  reread     every move: read(src) + write(dst). The original ledger.
  exclusive  a promotion costs only write(dst): the serving read was the read. Demotion as reread.
  retained   as exclusive, and a demotion to a tier holding a clean copy is a discard (free). The
             copy's capacity is not charged, so this is OPTIMISTIC for policies that move blocks.
  write-through  as exclusive, but every block is written to the second tier when it is created
             (LMCache-style), so a demotion from the fastest to the second tier is a discard, and the
             policy pays one write(second tier) per creation it makes. The copies' capacity is not
             charged (optimistic). The static heuristic writes nothing through and pays nothing.
  zero       every move free: the bound.

Policies: the reactive replays of run_reactive_cost.py (multi-tier LRU, cache on second hit) and the
arrival arms of run_online_admission.py (onestep). An arrival arm only demotes, and only blocks placed
once, so no demotion of theirs finds a clean copy and their four ledgers collapse to two (zero vs the
rest); they are read from online_admission.json, not re-run.

Reported per ledger: the captured share at the assumed price (alpha = 1), and alpha_0 -- the multiple
of that price at which the policy stops beating the static heuristic (captured share 0). Measured
multiples on one A100 PCIe host are 0.75-1.87 (run_measured_migration.py; remote RDMA unmeasured);
check_paper_numbers.py compares the two.
Usage: python3 scripts/run_migration_ledger.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reactive_cost as rc  # noqa: E402  (patches Action.space to tier-only on import)
import run_online_admission as oa  # noqa: E402
from run_arrival_admission import TRACES  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

LEDGERS = ("reread", "exclusive", "retained", "write-through", "zero")


def price(src, dst, kind, clean, ledger, tiers, read_ms, lam):
    r_s, r_d = read_ms[tiers[src]], read_ms[tiers[dst]]
    if ledger == "zero":
        return 0.0
    if ledger == "write-through" and kind == "demotion" and (src, dst) == (0, 1):
        return 0.0                                    # the creation already wrote it to tier 1
    if ledger == "reread" or kind == "demotion" and not (ledger == "retained" and clean):
        return lam * (r_s + r_d)
    if kind == "promotion":
        return lam * r_d
    return 0.0                                        # retained, demotion onto a clean copy


def summarize(share_zero, moves, span, tiers, read_ms, lam, creations):
    out = {}
    for led in LEDGERS:
        m = sum(c * price(s, d, k, cl, led, tiers, read_ms, lam) for (s, d, k, cl), c in moves.items())
        if led == "write-through":
            m += creations * lam * read_ms[tiers[1]]
        share = share_zero - 100.0 * m / span
        out[led] = {"migration_cost": round(m, 3), "captured_pct": round(share, 2),
                    "alpha_0": (round(share_zero * span / (100.0 * m), 4) if m > 0 else None)}
    return out


def main() -> int:
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    lam = w.lambda_latency
    online = json.loads((ROOT / "experiments" / "online_admission.json").read_text())
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        st = max(1, wl.time_split(0.5))
        pre, fut = wl.prefix_stats(st), wl.future_counts(st)
        ids = [i.item_id for i in wl.items]
        index = {b: j for j, b in enumerate(ids)}
        sizes = [float(i.size_bytes) for i in wl.items]
        cm = oa.cost_model(sim, w, rc.TIER_ONLY, wl.items[0].representation)
        tiers, read_ms = cm["tiers"], cm["read_ms"]
        caps = sim.tier_capacities(wl.total_bytes)
        slots = [len(ids) if caps[t] == float("inf") else int(caps[t] // sizes[0]) for t in tiers]
        events = oa.post_split_events(str(ROOT / path), st, 10 ** 9, index)
        counts = [float(pre[b]["count"]) for b in ids]
        is_new = [c == 0 for c in counts]
        heur = oa.static_tiers(oa.rank_order(counts, is_new, False), sizes, caps, tiers)
        c_heur = oa.bill_static(events, heur, cm)
        span = c_heur - place_with_beliefs(wl, {b: float(fut.get(b, 0)) for b in ids})
        tidx = {t: j for j, t in enumerate(tiers)}
        init = {j: tidx[heur[j]] for j in range(len(ids)) if not is_new[j]}
        pols = {}
        for pol in ("lru", "second-hit"):
            led: dict = {}
            c = rc.replay(pol, init, is_new, slots, cm, events, w, ledger=led)
            share_zero = 100.0 * (c_heur - c["zero"]) / span
            # the ledger must reproduce the replay's own read+write total exactly
            m_rw = sum(n * oa.migration_cost(tiers[s], tiers[d], read_ms, w, "read+write")
                       for (s, d, _, _), n in led["moves"].items())
            assert abs((c["read+write"] - c["zero"]) - m_rw) <= 1e-6 * max(1.0, m_rw), "ledger != replay"
            mv = led["moves"]
            pols[pol] = {"share_zero": round(share_zero, 2),
                         "n_moves": {f"{tiers[s].value}->{tiers[d].value} {k}{' clean' if cl else ''}": n
                                     for (s, d, k, cl), n in sorted(mv.items())},
                         "creations": led["creations"],
                         "ledgers": summarize(share_zero, mv, span, tiers, read_ms, lam, led["creations"])}
        for r in online["rows"]:
            if (r["trace"] == name and r["policy"] == "onestep" and r["migration_cost_model"] == "zero"
                    and r["arm"] in ("hazard-new", "order-arrival", "constant-mean")):
                mv = {}
                for key, n in r["migrations_by_tier"].items():
                    s, d = key.split("->")
                    mv[(tidx[next(t for t in tiers if t.value == s)],
                        tidx[next(t for t in tiers if t.value == d)], "demotion", False)] = n
                pols[f"{r['arm']} ({r['tie']})"] = {
                    "share_zero": r["online_captured_pct"],
                    "creations": sum(is_new),
                    "ledgers": summarize(r["online_captured_pct"], mv, span, tiers, read_ms, lam,
                                         sum(is_new))}
        rows.append({"trace": name, "policies": pols})
        for p, v in pols.items():
            print(f"{name:<13} {p:<28} " + "  ".join(
                f"{led} {v['ledgers'][led]['captured_pct']:8.2f} (a0 {v['ledgers'][led]['alpha_0']})"
                for led in LEDGERS), flush=True)
    (ROOT / "experiments" / "migration_ledger.json").write_text(json.dumps(
        {"config": {"ledgers": {
            "reread": "every move: read(src) + write(dst) (the original online ledger)",
            "exclusive": "promotion on an access: write(dst) only (the serving read was the read)",
            "retained": "as exclusive; a demotion onto a clean lower copy is free (copy capacity not "
                        "charged: optimistic for moving policies)",
            "write-through": "as exclusive; every creation is also written to the second tier (one "
                             "write each, billed to the policy), so fastest->second demotions are "
                             "discards (copy capacity not charged)",
            "zero": "all moves free"},
            "price_unit": "lambda_L x simulator read latency, as run_online_admission.migration_cost",
            "alpha_0": "multiple of the ledger's price at which captured share reaches 0",
            "arrival_arms": "onestep rows of online_admission.json; demotions only, never onto a copy"},
         "rows": rows}, indent=2) + "\n")
    print("wrote experiments/migration_ledger.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
