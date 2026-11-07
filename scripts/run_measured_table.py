#!/usr/bin/env python3
"""Do the paper's conclusions survive the measured serving costs? (§BE; ADVISE H7)

The placement results use a declared per-tier read-latency table (GPU 0.2, host 2, remote 6, disk 12 ms).
The serving runs (run_serving_tiers.py) measured, per 512-token block in vLLM + LMCache on one A100 host: the GPU tier's
marginal time to first token (attention over cached blocks), a load from host memory, and a load from a
cold rotational disk. This script replaces the table with those costs and reruns, at split 0.5 with
tier-only actions on the production traces,

  the ladder     heuristic, seen-once rule, the clairvoyant reference under both tie rules, oracle
  online         multi-tier LRU with free and read+write-priced moves, and its break-even multiple

The remote tier was not measured; it is swept from 2x the host cost to just under the disk cost (the cost
model needs tier costs strictly increasing). Per tier, cost in ms
per block (the context-token term is the same on every tier and cancels in every share):
  gpu   = (TTFT gpu hit @8192 - TTFT gpu hit @512) / 15
  host  = gpu + LMCache load from host per block
  disk  = gpu + LMCache cold-disk load per block
Usage: python3 scripts/run_measured_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reactive_cost as rc  # noqa: E402  (tier-only Action.space for the replay)
import run_online_admission as oa  # noqa: E402
from run_evidence_ladder import TIER, ordered  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa import simulator as simmod  # noqa: E402
from stoa.environment import Tier  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}


def tables():
    from run_serving_tiers import per_tier_costs
    m = per_tier_costs()
    gpu, host, disk = m["gpu"], m["host"], m["disk"]
    out = {}
    for label, remote in (("remote = 2x host", 2 * host), ("remote = 3x host", 3 * host), ("remote ≈ disk (0.99x)", 0.99 * disk)):
        out[label] = {Tier.GPU: gpu, Tier.CPU: host, Tier.REMOTE_RDMA: remote, Tier.DISK: disk}
    return out


def main() -> int:
    declared = dict(simmod._TIER_READ_MS)
    rows = []
    try:
        for label, table in [("declared", declared)] + list(tables().items()):
            simmod._TIER_READ_MS.clear()
            simmod._TIER_READ_MS.update(table)
            for name, path in TRACES.items():
                wl = load_mooncake(str(ROOT / path))
                st = max(1, wl.time_split(0.5))
                fut, pre = wl.future_counts(st), wl.prefix_stats(st)
                vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
                dead = {b for b in vis if fut.get(b, 0) == 0}
                once = {b for b in vis if int(pre[b]["count"]) == 1}
                cnt = {b: float(pre[b]["count"]) for b in vis}
                sq.LEGAL_ACTIONS = TIER
                arms = {"heuristic": (wl, cnt),
                        "seen-once-last": (ordered(wl, once), {b: (0.0 if b in once else c) for b, c in cnt.items()}),
                        "dead-first": (wl, {b: float(fut.get(b, 0)) for b in vis}),
                        "dead-last": (ordered(wl, dead), {b: float(fut.get(b, 0)) for b in vis}),
                        "oracle": (wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})}
                cost = {a: sq.place_with_beliefs(w_, b_) for a, (w_, b_) in arms.items()}
                span = cost["heuristic"] - cost["oracle"]
                ladder = {a: round(100 * (cost["heuristic"] - c) / span, 2) for a, c in cost.items()}
                # online replay on the same table (run_reactive_cost's setup)
                sim, w = simmod.TieringSimulator(), DEFAULT_WEIGHTS
                ids = [i.item_id for i in wl.items]
                index = {b: j for j, b in enumerate(ids)}
                sizes = [float(i.size_bytes) for i in wl.items]
                cm = oa.cost_model(sim, w, rc.TIER_ONLY, wl.items[0].representation)
                caps = sim.tier_capacities(wl.total_bytes)
                slots = [len(ids) if caps[t] == float("inf") else int(caps[t] // sizes[0]) for t in cm["tiers"]]
                events = oa.post_split_events(str(ROOT / path), st, 10 ** 9, index)
                counts = [float(pre[b]["count"]) for b in ids]
                is_new = [c == 0 for c in counts]
                heur = oa.static_tiers(oa.rank_order(counts, is_new, False), sizes, caps, cm["tiers"])
                c_heur = oa.bill_static(events, heur, cm)
                ospan = c_heur - sq.place_with_beliefs(wl, {b: float(fut.get(b, 0)) for b in ids})
                tidx = {t: j for j, t in enumerate(cm["tiers"])}
                init = {j: tidx[heur[j]] for j in range(len(ids)) if not is_new[j]}
                c = rc.replay("lru", init, is_new, slots, cm, events, w)
                lz = 100 * (c_heur - c["zero"]) / ospan
                lrw = 100 * (c_heur - c["read+write"]) / ospan
                rows.append({"table": label, "trace": name,
                             "read_ms": {t.value: round(v, 4) for t, v in table.items()},
                             "ladder_pct": ladder,
                             "lru_free_pct": round(lz, 2), "lru_read_write_pct": round(lrw, 2),
                             "lru_alpha_0": round(lz / (lz - lrw), 4) if lz != lrw else None})
                print(label, name, ladder, "LRU", rows[-1]["lru_free_pct"], rows[-1]["lru_read_write_pct"],
                      "a0", rows[-1]["lru_alpha_0"], flush=True)
    finally:
        simmod._TIER_READ_MS.clear()
        simmod._TIER_READ_MS.update(declared)
        sq.LEGAL_ACTIONS = list(sq.LEGAL_ACTIONS)
    (ROOT / "experiments" / "measured_table.json").write_text(json.dumps(
        {"config": {"split": 0.5, "actions": "tier only",
                    "tables": "declared, and three measured tables differing only in the unmeasured remote tier",
                    "source": "run_serving_tiers.per_tier_costs()"}, "rows": rows}, indent=2) + "\n")
    print("wrote experiments/measured_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
