#!/usr/bin/env python3
"""LoCoMo with REAL retrieval baselines — the comparison reviewers will demand.

Compares four context-selection strategies under the same token budget:
  stoa        (global, query-agnostic)  rank turns by evidence demand
  random      (global, query-agnostic)  the previous straw-man baseline
  centrality  (global, query-agnostic)  embedding centrality -- fair same-regime RAG baseline
  retrieval   (per-query, query-AWARE)  embedding top-k per question -- deployed-RAG bar

The per-query retriever sees the question, so it is strictly better informed than any
global placement; we report it because it is the honest bar, not because it is fair.

Requires:  pip install -e ".[eval]"  and  export OPENAI_API_KEY=...  (embeddings + answers)
Usage:     python3 scripts/run_locomo_baselines.py [--samples N] [--max-questions Q] [--judge]
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
from stoa.eval.memqa import MemQATask, evaluate, split_history  # noqa: E402
from stoa.eval.retrieval import Retriever  # noqa: E402

STRATEGIES = ("stoa", "random", "centrality", "retrieval")


def main() -> None:
    p = argparse.ArgumentParser(description="LoCoMo: STOA vs random vs embedding retrieval")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-questions", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--out", type=str, default="experiments/eval_locomo_baselines.json")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")

    from stoa.llm import OpenAIClient
    client = OpenAIClient()
    answer = client.answer_fn()
    judge = client.judge_fn() if args.judge else None
    embed_fn = client.embed

    fracs = (0.1, 0.25, 0.5)
    agg = {s: [[] for _ in fracs] for s in STRATEGIES}
    tasks = load_locomo()

    for si in range(args.sample, args.sample + args.samples):
        task = tasks[si]
        # Subsample for API cost, then split into disjoint (history, eval) questions so the
        # demand signal is estimated from PAST questions — never from the eval answer key.
        if len(task.questions) > args.max_questions * 2:
            rng = random.Random(args.seed + si)
            task = MemQATask(task.facts, rng.sample(task.questions, args.max_questions * 2),
                             task.tokens_per_fact, task.fact_tokens)
        task = split_history(task, history_frac=0.5, seed=args.seed + si)
        assert task.leak_free, "demand must come from held-out history questions"
        total = sum(task.fact_tokens.values())
        retr = Retriever(task, embed_fn, tag=f"locomo{si}")
        retr.embed_questions(task.questions)
        cent_order = retr.centrality_order()

        for k, f in enumerate(fracs):
            b = int(f * total)
            agg["stoa"][k].append(evaluate(task, answer, b, priority="stoa", judge_fn=judge).accuracy)
            agg["random"][k].append(evaluate(task, answer, b, priority="random", seed=args.seed,
                                             judge_fn=judge).accuracy)
            agg["centrality"][k].append(evaluate(task, answer, b, order=cent_order,
                                                 judge_fn=judge).accuracy)
            agg["retrieval"][k].append(
                evaluate(task, answer, b, judge_fn=judge,
                         retriever=lambda q, bt: retr.top_k_within_budget(q, bt)).accuracy)
        print(f"  [sample {si} done]", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nLoCoMo with retrieval baselines ({args.samples} dialogues, "
          f"metric={'LLM-judge' if args.judge else 'substring'})")
    print(f"{'budget':>7} | " + " | ".join(f"{s:>10}" for s in STRATEGIES))
    print("-" * 60)
    for k, f in enumerate(fracs):
        row = " | ".join(f"{mean(agg[s][k])*100:9.1f}%" for s in STRATEGIES)
        print(f"{f*100:6.0f}% | {row}")

    print("\nregimes: stoa/random/centrality = global (query-agnostic) | retrieval = per-query (query-aware)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"samples": args.samples, "max_questions": args.max_questions,
                   "metric": "LLM-judge" if args.judge else "substring",
                   "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")},
        "budget_fractions": list(fracs),
        "results": {s: [round(mean(agg[s][k]), 4) for k in range(len(fracs))] for s in STRATEGIES},
        "regimes": {"stoa": "global", "random": "global", "centrality": "global",
                    "retrieval": "per-query (query-aware)"},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
