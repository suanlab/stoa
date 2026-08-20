#!/usr/bin/env python3
"""Does representation choice actually trade tokens against accuracy? Measured, not assumed.

The paper's formulation decides representation x tier x timing jointly and argues the axes
are coupled. Every experiment in it so far fixes representation at plaintext, so the
coupling claim -- the reason the joint decision exists at all -- has never been measured.
This script measures the utility half of it on real LoCoMo dialogues.

Same fact ids, same questions, same gold answers; only the text and its token cost change:

  plaintext  the turn verbatim
  summary    an LLM rewrite keeping names, numbers and dates, dropping phrasing
  extract    an aggressive 'speaker: key=value' reduction

The quantity of interest is accuracy at EQUAL token budget. Three outcomes are possible and
all are reportable:

  * one representation dominates at every budget -> representation need not be per item, and
    the coupling argument in Section 3 is weaker than stated;
  * the curves CROSS as the budget moves -> the coupling is real and the joint decision is
    justified, with the crossing point being the thing a policy would have to learn;
  * compression loses accuracy without saving enough tokens -> plaintext is simply right on
    this workload, which is also a finding.

Because every representation answers the SAME questions, the comparison is paired and is
tested with McNemar rather than Fisher (`stoa.stats`).

Requires OPENAI_API_KEY in the environment (never in a file inside the repository).
Compression is cached per fact, so cost is dominated by answering, not by rewriting.

Usage: python3 scripts/run_representation_axis.py [--dialogues 5] [--questions 20] [--judge]
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
from stoa.eval.representation import MODES, compression_ratio, rewrite_task  # noqa: E402
from stoa.stats import mcnemar_exact  # noqa: E402

FRACS = (0.05, 0.10, 0.25, 0.50)


def main() -> None:
    ap = argparse.ArgumentParser(description="Representation axis on LoCoMo")
    ap.add_argument("--dialogues", type=int, default=5)
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", type=str, default="experiments/representation_axis.json")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (keep it in the environment, never in the repo).")

    from stoa.llm import OpenAIClient
    client = OpenAIClient()
    answer = client.answer_fn()
    judge = client.judge_fn() if args.judge else None
    compress = client.answer_fn()          # the rewrite is just another completion

    tasks = load_locomo()
    n_dialogues = min(args.dialogues, len(tasks))
    outcomes = {m: {f: [] for f in FRACS} for m in MODES}
    ratios = {m: [] for m in MODES}

    # Compression is the expensive part -- thousands of calls, one per turn per mode -- and
    # it is deterministic given (mode, text). Persisting it means a crash or a rerun costs
    # nothing already paid for. An earlier run died on a network timeout two dialogues in
    # and threw away every rewrite it had bought.
    cache_path = Path("data/repr_cache.json")
    cache: dict[str, str] = (json.loads(cache_path.read_text())
                             if cache_path.exists() else {})
    print(f"compression cache: {len(cache)} entries at {cache_path}")

    print(f"Representation axis: {n_dialogues} dialogues x {args.questions} questions x "
          f"{len(FRACS)} budgets x {len(MODES)} representations")

    for si in range(n_dialogues):
        task = tasks[si]
        if len(task.questions) > args.questions * 2:
            rng = random.Random(args.seed + si)
            task = MemQATask(task.facts, rng.sample(task.questions, args.questions * 2),
                             task.tokens_per_fact, task.fact_tokens)
        task = split_history(task, history_frac=0.5, seed=args.seed + si)
        assert task.leak_free, "demand must come from held-out history questions"

        variants = {}
        for mode in MODES:
            v = rewrite_task(task, mode, compress_fn=compress, cache=cache)
            variants[mode] = v
            ratios[mode].append(compression_ratio(task, v))

        # Budgets are a fraction of the PLAINTEXT total, so every representation is given
        # the same absolute token allowance. Deriving each budget from its own total would
        # hand the compressed representations a larger effective budget and make the
        # comparison meaningless -- the reference-frame error this project keeps finding.
        plaintext_total = sum(task.cost(f) for f in task.facts)

        for frac in FRACS:
            b = int(frac * plaintext_total)
            for mode in MODES:
                r = evaluate(variants[mode], answer, b, priority="stoa", judge_fn=judge)
                assert len(r.per_question) == len(task.questions)
                outcomes[mode][frac].extend(r.per_question)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))     # checkpoint after every dialogue
        print(f"  [dialogue {si + 1}/{n_dialogues} done, cache {len(cache)}]", flush=True)

    def acc(m, f):
        o = outcomes[m][f]
        return sum(o) / len(o) if o else 0.0

    n_q = len(outcomes["plaintext"][FRACS[0]])
    print(f"\nAccuracy at equal token budget ({n_q} questions per cell, "
          f"metric={'LLM-judge' if args.judge else 'substring'})")
    print(f"{'budget':>7} | " + " | ".join(f"{m:>10}" for m in MODES) + " | winner")
    print("-" * 62)
    rows = []
    for f in FRACS:
        accs = {m: acc(m, f) for m in MODES}
        win = max(accs, key=accs.get)
        print(f"{f * 100:6.0f}% | " + " | ".join(f"{accs[m] * 100:9.1f}%" for m in MODES)
              + f" | {win}")
        rows.append({"budget_frac": f, **{f"acc_{m}": round(accs[m], 4) for m in MODES},
                     "winner": win})

    print(f"\nmean compression ratio: "
          + ", ".join(f"{m}={sum(ratios[m]) / len(ratios[m]):.2f}" for m in MODES))

    # Paired tests against plaintext -- same questions, so McNemar.
    print(f"\n{'budget':>7} | {'contrast':<26} | {'b':>4} {'c':>4} | {'McNemar p':>9}")
    print("-" * 60)
    contrasts = []
    for f in FRACS:
        for m in ("summary", "extract"):
            a_o, p_o = outcomes[m][f], outcomes["plaintext"][f]
            b_c = sum(1 for x, y in zip(a_o, p_o) if x and not y)
            c_c = sum(1 for x, y in zip(a_o, p_o) if y and not x)
            p = mcnemar_exact(b_c, c_c)
            print(f"{f * 100:6.0f}% | {m + ' vs plaintext':<26} | {b_c:>4} {c_c:>4} | {p:9.4f}")
            contrasts.append({"budget_frac": f, "mode": m, "vs": "plaintext",
                              "discordant_mode": b_c, "discordant_plaintext": c_c,
                              "mcnemar_p": round(p, 5)})

    winners = {r["winner"] for r in rows}
    crossed = len(winners) > 1
    print(f"\nwinner changes with budget: {'YES' if crossed else 'no'} ({sorted(winners)})")
    print("  A crossing means representation has to be decided per operating point, which is"
          if crossed else
          "  A single winner at every budget means representation need NOT be decided per")
    print("  what the joint formulation claims. No crossing weakens that claim."
          if crossed else
          "  budget on this workload -- a weaker coupling than the formulation assumes.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"dialogues": n_dialogues, "questions_per_dialogue": args.questions,
                   "n_scored_per_cell": n_q, "seed": args.seed,
                   "metric": "LLM-judge" if args.judge else "substring",
                   "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                   "note": "budgets are a fraction of the PLAINTEXT total, identical for all modes"},
        "budget_fractions": list(FRACS),
        "accuracy": {m: [round(acc(m, f), 4) for f in FRACS] for m in MODES},
        "compression_ratio": {m: round(sum(ratios[m]) / len(ratios[m]), 4) for m in MODES},
        "rows": rows, "contrasts": contrasts,
        "winner_changes_with_budget": crossed,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
