#!/usr/bin/env python3
"""M6-9 deliverable: GNN factored-policy warm-start via Belady behavioral cloning (RQ2).

Trains the GNN factored policy (stoa/gnn.py) to clone the capacity-feasible Belady
placement on training traces, then evaluates imitation accuracy and achieved cost
on HELD-OUT traces. CPU-only (torch>=2, no GPU): the policy trains over the tiering
simulator, not an LLM.

Requires:  pip install -e ".[rl]"   (torch, numpy)
Usage:     python3 scripts/train_gnn_policy.py [--items N] [--horizon T] [--zipf S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402
from stoa.train import evaluate, train_bc  # noqa: E402

TRAIN_SEEDS = (100, 101, 102)
TEST_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    p = argparse.ArgumentParser(description="STOA M6-9 GNN policy warm-start (RQ2)")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=6_000)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", type=str, default="experiments/m6_gnn_policy.json")
    args = p.parse_args()

    def cfg(seed):
        return WorkloadConfig(n_items=args.items, horizon=args.horizon, zipf_s=args.zipf, seed=seed)

    train_wls = [generate_workload(cfg(s)) for s in TRAIN_SEEDS]
    policy, tr = train_bc(train_wls, epochs=args.epochs)

    evals = [evaluate(policy, generate_workload(cfg(s))) for s in TEST_SEEDS]
    imit = sum(e.imitation_acc for e in evals) / len(evals)
    gap = sum(e.gap_closed_pct for e in evals) / len(evals)
    bc = sum(e.bc_cost for e in evals) / len(evals)
    tgt = sum(e.target_cost for e in evals) / len(evals)

    print(f"\nM6-9 GNN factored policy — Belady behavioral cloning (RQ2)")
    print(f"items={args.items} horizon={args.horizon} zipf={args.zipf} epochs={args.epochs} (CPU)")
    print(f"train seeds={TRAIN_SEEDS}  test seeds={TEST_SEEDS} (held-out)")
    print("-" * 72)
    print(f"  train action-match accuracy : {tr.train_acc*100:6.2f}%  (final loss {tr.final_loss:.3f})")
    print(f"  held-out imitation accuracy : {imit*100:6.2f}%")
    print(f"  held-out cost (BC / target) : {bc:.1f} / {tgt:.1f}  "
          f"(BC within {(bc-tgt)/tgt*100:+.1f}% of Belady target)")
    print(f"  naive->target gap closed    : {gap:6.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"items": args.items, "horizon": args.horizon, "zipf": args.zipf,
                   "epochs": args.epochs, "train_seeds": list(TRAIN_SEEDS),
                   "test_seeds": list(TEST_SEEDS), "device": "cpu"},
        "train_acc": round(tr.train_acc, 4),
        "held_out_imitation_acc": round(imit, 4),
        "held_out_bc_cost": round(bc, 4),
        "held_out_target_cost": round(tgt, 4),
        "held_out_gap_closed_pct": round(gap, 2),
        "per_seed": [e.to_dict() for e in evals],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
