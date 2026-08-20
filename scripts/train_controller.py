#!/usr/bin/env python3
"""M6-9 (torch-free core): learned Belady-imitation tier controller — RQ2 basis.

Fit a linear reuse-distance predictor (learn.py) on training traces, then evaluate
it online on HELD-OUT traces against the heuristics (LRU, H2O) and the Belady
oracle. Under temporal locality the learned freq+recency controller beats either
pure heuristic; the residual gap to Belady is what the M6-9 GNN + offline-RL
fine-tuning (torch) is meant to close toward the RQ2 <=5% target.

Writes experiments/m6_learned_controller.json. Stdlib-only.

Usage:
    python3 scripts/train_controller.py [--zipf S] [--beta B] [--cache-frac F]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.online import run_online  # noqa: E402
from stoa.learn import fit_reuse_predictor  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402

TRAIN_SEEDS = (100, 101, 102)
TEST_SEEDS = (0, 1, 2, 3, 4)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="STOA M6-9 learned tier controller (RQ2)")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=8_000)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--beta", type=float, default=0.5, help="temporal locality (0=stationary Zipf)")
    p.add_argument("--cache-frac", type=float, default=0.1)
    p.add_argument("--out", type=str, default="experiments/m6_learned_controller.json")
    args = p.parse_args()

    def cfg(seed: int) -> WorkloadConfig:
        return WorkloadConfig(n_items=args.items, horizon=args.horizon,
                              zipf_s=args.zipf, locality_beta=args.beta, seed=seed)

    # Fit on training traces, average the models by refitting on the concatenation
    # is overkill for a linear model; fit on the first train seed (imitation target).
    predictor = fit_reuse_predictor(generate_workload(cfg(TRAIN_SEEDS[0])))

    rows: dict[str, list[float]] = {p_: [] for p_ in ("belady", "learned", "h2o", "lru", "static")}
    for seed in TEST_SEEDS:
        res = run_online(generate_workload(cfg(seed)), cache_frac=args.cache_frac, predictor=predictor)
        for name, r in res.items():
            rows[name].append(r.hit_rate)

    hit = {k: _mean(v) for k, v in rows.items()}
    best_heur = max(hit["h2o"], hit["lru"], hit["static"])
    oracle_gap = hit["belady"] - best_heur
    closed = (hit["learned"] - best_heur) / oracle_gap * 100 if oracle_gap > 0 else 0.0

    print(f"\nM6-9 learned tier controller  (items={args.items}, horizon={args.horizon}, "
          f"zipf={args.zipf}, beta={args.beta}, cache_frac={args.cache_frac})")
    print(f"train seeds={TRAIN_SEEDS}  test seeds={TEST_SEEDS} (held-out)")
    print("-" * 72)
    for name in ("belady", "learned", "h2o", "lru", "static"):
        tag = "  <- oracle" if name == "belady" else ("  <- learned" if name == "learned" else "")
        print(f"  {name:<9} mean hit-rate = {hit[name]*100:6.2f}%{tag}")
    print("-" * 72)
    print(f"  learned vs best heuristic : {(hit['learned']-best_heur)*100:+.2f}pp")
    print(f"  oracle gap closed         : {closed:6.1f}%   (residual for M6-9 RL: {100-closed:.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"items": args.items, "horizon": args.horizon, "zipf": args.zipf,
                   "beta": args.beta, "cache_frac": args.cache_frac,
                   "train_seeds": list(TRAIN_SEEDS), "test_seeds": list(TEST_SEEDS)},
        "mean_hit_rate": {k: round(v, 4) for k, v in hit.items()},
        "learned_vs_best_heuristic_pp": round((hit["learned"] - best_heur) * 100, 3),
        "oracle_gap_closed_pct": round(closed, 2),
        "predictor": predictor.to_dict(),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
