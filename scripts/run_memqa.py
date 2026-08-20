#!/usr/bin/env python3
"""LLM-backed memory-QA under budget — measures real task-utility U (RQ1).

Runs the accuracy-vs-budget frontier with a real LLM, comparing STOA-heat fact
selection to budget-agnostic random selection. NO GPU (LLM via HTTP API).

Requires:  pip install -e ".[eval]"   and   export OPENAI_API_KEY=sk-...  (rotate leaked keys!)
Optional:  export OPENAI_MODEL=gpt-4o-mini
Usage:     python3 scripts/run_memqa.py [--facts N] [--questions Q] [--seed K]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.memqa import MemQAConfig, compare, generate_multihop_task, generate_task  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="STOA LLM-backed memory-QA frontier (RQ1)")
    p.add_argument("--facts", type=int, default=40)
    p.add_argument("--questions", type=int, default=60)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--multihop", action="store_true", help="2-hop reasoning task (harder, U<1)")
    p.add_argument("--out", type=str, default="experiments/eval_memqa.json")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Export a (rotated) key: export OPENAI_API_KEY=sk-...")

    from stoa.llm import OpenAIClient
    answer_fn = OpenAIClient().answer_fn()

    gen = generate_multihop_task if args.multihop else generate_task
    task = gen(MemQAConfig(n_facts=args.facts, n_questions=args.questions,
                           zipf_s=args.zipf, seed=args.seed))
    budgets = [b * task.tokens_per_fact for b in (5, 10, 20, args.facts)]
    fr = compare(task, answer_fn, budgets, seed=args.seed)

    kind = "2-hop" if args.multihop else "1-hop"
    print(f"\nLLM memory-QA frontier ({kind}, model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}, "
          f"facts={args.facts}, questions={args.questions})")
    print(f"{'budget(tok)':>12} | {'STOA acc':>9} | {'random acc':>11} | {'gain':>8} | STOA reason-acc")
    print("-" * 72)
    for sp, rp in zip(fr["stoa"], fr["random"]):
        print(f"{sp.budget_tokens:>12} | {sp.accuracy*100:8.1f}% | {rp.accuracy*100:10.1f}% | "
              f"{(sp.accuracy-rp.accuracy)*100:+7.1f}pp | {sp.reasoning_acc*100:6.1f}% "
              f"(n={sp.n_answerable})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"facts": args.facts, "questions": args.questions, "zipf": args.zipf,
                   "seed": args.seed, "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")},
        "stoa": [r.to_dict() for r in fr["stoa"]],
        "random": [r.to_dict() for r in fr["random"]],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
