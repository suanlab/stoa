#!/usr/bin/env python3
"""Real-benchmark RQ1 check on LoCoMo (arXiv:2402.17753) with a real LLM.

Loads a LoCoMo sample, then measures QA accuracy vs context-token budget for
budget-aware (evidence-demand) turn selection vs random. NO GPU (LLM via API).

Answers scored by substring (rough vs LoCoMo's official F1/LLM-judge), so absolute
numbers are a lower bound — the STOA-vs-random gap is the signal.

Requires:  pip install -e ".[eval]"  and  export OPENAI_API_KEY=sk-...  (rotate leaked keys)
Usage:     python3 scripts/run_locomo.py [--sample I] [--max-questions N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.locomo import load_locomo  # noqa: E402
from stoa.eval.memqa import MemQATask, compare  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="STOA LoCoMo RQ1 check (real LLM)")
    p.add_argument("--sample", type=int, default=0, help="first sample index")
    p.add_argument("--samples", type=int, default=1, help="number of samples to average over")
    p.add_argument("--max-questions", type=int, default=30, help="subsample to bound API cost")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true", help="LLM-judge scoring (vs substring lower bound)")
    p.add_argument("--out", type=str, default="experiments/eval_locomo.json")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Export a (rotated) key: export OPENAI_API_KEY=sk-...")

    from stoa.llm import OpenAIClient
    client = OpenAIClient()
    judge = client.judge_fn() if args.judge else None
    answer = client.answer_fn()

    all_tasks = load_locomo()
    fracs = (0.1, 0.25, 0.5, 1.0)
    # Aggregate accuracy per budget *fraction* across samples (absolute tokens vary per sample).
    agg = {p: [[] for _ in fracs] for p in ("stoa", "random")}
    n_turns = n_q = 0
    for si in range(args.sample, args.sample + args.samples):
        task = all_tasks[si]
        if len(task.questions) > args.max_questions:
            rng = random.Random(args.seed + si)
            task = MemQATask(task.facts, rng.sample(task.questions, args.max_questions),
                             task.tokens_per_fact, task.fact_tokens)
        n_turns += len(task.facts); n_q += len(task.questions)
        total = sum(task.fact_tokens.values())
        budgets = [int(f * total) for f in fracs]
        fr = compare(task, answer, budgets, seed=args.seed, judge_fn=judge)
        for p in ("stoa", "random"):
            for k, r in enumerate(fr[p]):
                agg[p][k].append(r.accuracy)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nLoCoMo ({args.samples} sample(s), ~{n_turns//max(args.samples,1)} turns/sample, "
          f"model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}, "
          f"metric={'LLM-judge' if args.judge else 'substring'})")
    print(f"{'budget':>8} | {'STOA acc':>9} | {'random acc':>11} | gain")
    print("-" * 48)
    for k, f in enumerate(fracs):
        s, r = mean(agg["stoa"][k]) * 100, mean(agg["random"][k]) * 100
        print(f"{f*100:6.0f}% | {s:8.1f}% | {r:10.1f}% | {s-r:+.1f}pp")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"samples": args.samples, "first_sample": args.sample, "questions_total": n_q,
                   "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                   "metric": "LLM-judge" if args.judge else "substring"},
        "budget_fractions": list(fracs),
        "stoa": [round(mean(agg["stoa"][k]), 4) for k in range(len(fracs))],
        "random": [round(mean(agg["random"][k]), 4) for k in range(len(fracs))],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
