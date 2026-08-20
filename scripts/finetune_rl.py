#!/usr/bin/env python3
"""M6-9/M9-12: offline policy-gradient fine-tuning on top of the BC warm-start.

Trains the GNN by Belady behavioral cloning, then fine-tunes with REINFORCE on the
constrained objective (cost + tier-overflow penalty) at several penalty weights.
CPU-only (torch), no GPU.

HONEST FINDING (this placeholder cost model): BC already matches/beats the greedy
Belady target, so the fine-tune does not cleanly dominate it — RL can be steered to
enforce feasibility (drive tier overflow toward 0) but at a cost premium, and the
cost/feasibility trade-off is noisy. Diagnosis: a *simultaneous* per-node policy
lacks the tier-occupancy/pressure state (design doc §2) that a *sequential* placement
uses to decide which items to demote. The sequential occupancy-aware MDP is the
identified next formulation; see docs/research_plan.md.

Usage:  python3 scripts/finetune_rl.py [--items N] [--epochs E]
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.rl import evaluate_placement, finetune  # noqa: E402
from stoa.train import train_bc  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402

TRAIN_SEEDS = (100, 101, 102)
TEST_SEEDS = (0, 1, 2)
BETAS = (1.0, 2.0, 4.0)


def main() -> None:
    p = argparse.ArgumentParser(description="STOA offline RL fine-tune (BC -> REINFORCE)")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=6_000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", type=str, default="experiments/m6_rl_finetune.json")
    args = p.parse_args()

    def wl(seed):
        return generate_workload(WorkloadConfig(n_items=args.items, horizon=args.horizon,
                                                zipf_s=1.1, seed=seed))

    train = [wl(s) for s in TRAIN_SEEDS]
    test = [wl(s) for s in TEST_SEEDS]

    def summarize(policy):
        es = [evaluate_placement(policy, w) for w in test]
        return (statistics.mean(e.cost for e in es),
                sum(e.overflow_bytes for e in es),
                sum(e.feasible for e in es))

    bc, _ = train_bc(train, epochs=args.epochs)
    bc_cost, bc_over, bc_feas = summarize(bc)

    print(f"\nOffline RL fine-tune (BC -> REINFORCE), CPU  items={args.items}")
    print(f"{'policy':>12} | {'mean cost':>9} | {'overflow(B)':>12} | feasible/{len(test)}")
    print("-" * 56)
    print(f"{'BC':>12} | {bc_cost:9.0f} | {bc_over:12.0f} | {bc_feas}")
    rows = [{"policy": "BC", "mean_cost": round(bc_cost, 2), "overflow_bytes": round(bc_over, 1),
             "feasible": bc_feas}]
    for beta in BETAS:
        pol, fr = finetune(copy.deepcopy(bc), train, steps=args.steps, beta=beta, seed=1)
        c, o, f = summarize(pol)
        print(f"{'RL b=' + str(beta):>12} | {c:9.0f} | {o:12.0f} | {f}")
        rows.append({"policy": f"RL_beta_{beta}", "mean_cost": round(c, 2),
                     "overflow_bytes": round(o, 1), "feasible": f, "final_reward": round(fr.final_reward, 4)})

    print("\ninterpretation: RL enforces feasibility as beta grows but at a cost premium; it does")
    print("not cleanly dominate BC here (BC ~ greedy heuristic; placeholder costs). Next: sequential")
    print("occupancy-aware MDP so the policy can condition on tier pressure (design doc §2).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"items": args.items, "horizon": args.horizon, "epochs": args.epochs,
                   "steps": args.steps, "betas": list(BETAS), "device": "cpu"},
        "results": rows,
        "note": "BC already ~ greedy Belady; RL fine-tune trades cost for feasibility, no clean "
                "dominance on the static placeholder-cost formulation. See research_plan.md.",
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
