#!/usr/bin/env python3
"""LoCoMo at the sample size the power analysis asked for.

`scripts/run_locomo_power.py` established three things about our earlier LoCoMo runs: not
one of eighteen comparisons survived correction for multiplicity; the reversal between two
runs was not explainable by sampling noise, so it came from the runs drawing different
dialogues; and separating query-agnostic placement from random selection needs 246-710
scored questions per arm against the 30 we had. This script is the run that answers it.

Three things change from `run_locomo_baselines.py`:

  * **All ten dialogues, not two or three.** Between-dialogue heterogeneity is what produced
    the reversal; drawing every dialogue removes it as a variable rather than hoping it
    averages out.
  * **Per-question outcomes are recorded**, so the arms can be compared with the test the
    design actually calls for. Every arm answers the SAME questions, which makes this a
    PAIRED comparison: McNemar on the discordant pairs, not Fisher on independent samples.
  * **Both a pooled question-level test and a per-dialogue paired test** are reported. If
    they disagree, that disagreement is itself the finding, and it is exactly what
    heterogeneity looks like.

Requires: pip install -e ".[eval]" and OPENAI_API_KEY in the environment (never in a file
inside the repository).

Cost scales as dialogues x questions x budgets x arms; the default grid is about 3k answer
calls plus judging on gpt-4o-mini. Budget an hour and a few dollars.

Usage: python3 scripts/run_locomo_powered.py [--questions 30] [--judge]
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
from stoa.stats import mcnemar_exact, n_for_power_mcnemar  # noqa: E402

STRATEGIES = ("stoa", "random", "centrality", "retrieval")
# The contrasts the paper actually makes. Order is (better-expected, worse-expected); the
# test is two-sided regardless, so the ordering only labels the output.
CONTRASTS = (("stoa", "random"), ("retrieval", "stoa"), ("stoa", "centrality"))


def _sign_test(diffs: list[float]) -> float:
    """Two-sided exact sign test over per-dialogue differences; ties dropped.

    The per-dialogue view treats each dialogue as one observation, which is the conservative
    reading when dialogues differ from each other more than the arms differ within a dialogue.
    """
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return mcnemar_exact(pos, neg)


def main() -> None:
    ap = argparse.ArgumentParser(description="LoCoMo at adequate sample size")
    ap.add_argument("--dialogues", type=int, default=10, help="LoCoMo releases 10")
    ap.add_argument("--questions", type=int, default=30, help="EVAL questions per dialogue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", action="store_true", help="LLM judge instead of substring")
    ap.add_argument("--out", type=str, default="experiments/eval_locomo_powered.json")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (keep it in the environment, never in the repo).")

    from stoa.llm import OpenAIClient
    client = OpenAIClient()
    answer = client.answer_fn()
    judge = client.judge_fn() if args.judge else None

    fracs = (0.1, 0.25, 0.5)
    tasks = load_locomo()
    n_dialogues = min(args.dialogues, len(tasks))
    print(f"LoCoMo: {n_dialogues} dialogues x {args.questions} eval questions x "
          f"{len(fracs)} budgets x {len(STRATEGIES)} arms")

    # outcomes[strategy][budget_index] = flat list of per-question booleans, pooled over
    # dialogues but kept in a fixed order so the arms stay aligned for the paired test.
    outcomes = {s: [[] for _ in fracs] for s in STRATEGIES}
    per_dialogue = {s: [[] for _ in fracs] for s in STRATEGIES}

    for si in range(n_dialogues):
        task = tasks[si]
        # Subsample to control cost, then split into disjoint (history, eval) question sets
        # so the demand signal is estimated from PAST questions, never from the answer key.
        if len(task.questions) > args.questions * 2:
            rng = random.Random(args.seed + si)
            task = MemQATask(task.facts, rng.sample(task.questions, args.questions * 2),
                             task.tokens_per_fact, task.fact_tokens)
        task = split_history(task, history_frac=0.5, seed=args.seed + si)
        assert task.leak_free, "demand must come from held-out history questions"
        total = sum(task.fact_tokens.values())
        retr = Retriever(task, client.embed, tag=f"locomo{si}")
        retr.embed_questions(task.questions)
        cent_order = retr.centrality_order()

        for k, f in enumerate(fracs):
            b = int(f * total)
            runs = {
                "stoa": evaluate(task, answer, b, priority="stoa", judge_fn=judge),
                "random": evaluate(task, answer, b, priority="random", seed=args.seed,
                                   judge_fn=judge),
                "centrality": evaluate(task, answer, b, order=cent_order, judge_fn=judge),
                "retrieval": evaluate(task, answer, b, judge_fn=judge,
                                      retriever=lambda q, bt: retr.top_k_within_budget(q, bt)),
            }
            n_q = len(task.questions)
            for s, r in runs.items():
                assert len(r.per_question) == n_q, (s, len(r.per_question), n_q)
                outcomes[s][k].extend(r.per_question)
                per_dialogue[s][k].append(r.accuracy)
        print(f"  [dialogue {si + 1}/{n_dialogues} done, {len(task.questions)} questions]",
              flush=True)

    # --- report -----------------------------------------------------------------------
    def acc(s, k):
        o = outcomes[s][k]
        return sum(o) / len(o) if o else 0.0

    n_total = len(outcomes["stoa"][0])
    print(f"\nPooled over {n_dialogues} dialogues: {n_total} scored questions per arm "
          f"(metric={'LLM-judge' if args.judge else 'substring'})")
    print(f"{'budget':>7} | " + " | ".join(f"{s:>10}" for s in STRATEGIES))
    print("-" * 60)
    for k, f in enumerate(fracs):
        print(f"{f * 100:6.0f}% | " + " | ".join(f"{acc(s, k) * 100:9.1f}%" for s in STRATEGIES))

    rows = []
    print(f"\n{'budget':>7} | {'contrast':<24} | {'b':>4} {'c':>4} | {'McNemar p':>9} | "
          f"{'sign p':>7}")
    print("-" * 72)
    for k, f in enumerate(fracs):
        for hi, lo in CONTRASTS:
            a_out, b_out = outcomes[hi][k], outcomes[lo][k]
            b_cnt = sum(1 for x, y in zip(a_out, b_out) if x and not y)
            c_cnt = sum(1 for x, y in zip(a_out, b_out) if y and not x)
            p_mc = mcnemar_exact(b_cnt, c_cnt)
            diffs = [x - y for x, y in zip(per_dialogue[hi][k], per_dialogue[lo][k])]
            p_sign = _sign_test(diffs)
            print(f"{f * 100:6.0f}% | {hi + ' vs ' + lo:<24} | {b_cnt:>4} {c_cnt:>4} | "
                  f"{p_mc:9.4f} | {p_sign:7.3f}")
            rows.append({"budget_frac": f, "high": hi, "low": lo,
                         "acc_high": round(acc(hi, k), 4), "acc_low": round(acc(lo, k), 4),
                         "discordant_high": b_cnt, "discordant_low": c_cnt,
                         "n_questions": n_total,
                         "mcnemar_p": round(p_mc, 5), "sign_test_p": round(p_sign, 4),
                         "dialogues_favouring_high": sum(1 for d in diffs if d > 0),
                         "dialogues_favouring_low": sum(1 for d in diffs if d < 0)})

    thr = 0.05 / len(rows)
    surv = [r for r in rows if r["mcnemar_p"] < thr]
    print(f"\nBonferroni threshold for {len(rows)} comparisons: p < {thr:.4f}")
    print(f"surviving (paired test): {len(surv)}" +
          (": " + ", ".join(f"{r['high']}>{r['low']}@{r['budget_frac']:.0%}" for r in surv)
           if surv else " — none"))

    # What would still be needed, if anything is still unresolved. Computed BEFORE the
    # artifact is written so the numbers land in it, and sized against the SAME threshold
    # used to declare significance -- sizing at 0.05 while judging at the corrected
    # threshold understates the requirement by 50-65%.
    for r in rows:
        if r["mcnemar_p"] >= thr and (r["discordant_high"] + r["discordant_low"]) > 0:
            d = r["discordant_high"] + r["discordant_low"]
            p_disc = d / r["n_questions"]
            p_fav = r["discordant_high"] / d
            need = n_for_power_mcnemar(p_disc, p_fav, alpha=thr, seed=args.seed)
            need05 = n_for_power_mcnemar(p_disc, p_fav, alpha=0.05, seed=args.seed)
            r["n_for_80pct_power_corrected"] = need
            r["n_for_80pct_power_alpha05"] = need05
            if need:
                print(f"  {r['high']} vs {r['low']} @ {r['budget_frac']:.0%}: still "
                      f"unresolved; 80% power needs ~{need} questions at the corrected "
                      f"threshold ({need05} at an uncorrected 0.05); have {r['n_questions']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"dialogues": n_dialogues, "questions_per_dialogue": args.questions,
                   "n_scored_per_arm": n_total, "seed": args.seed,
                   "metric": "LLM-judge" if args.judge else "substring",
                   "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                   "test": "paired: McNemar exact (question-level), sign test (dialogue-level)",
                   "power_alpha": "corrected (Bonferroni) and uncorrected 0.05, both stored"},
        "budget_fractions": list(fracs),
        "accuracy": {s: [round(acc(s, k), 4) for k in range(len(fracs))] for s in STRATEGIES},
        "per_dialogue_accuracy": {s: [[round(v, 4) for v in per_dialogue[s][k]]
                                      for k in range(len(fracs))] for s in STRATEGIES},
        "contrasts": rows,
        "bonferroni_threshold": round(thr, 5),
        "n_surviving_correction": len(surv),
        "regimes": {"stoa": "global", "random": "global", "centrality": "global",
                    "retrieval": "per-query (query-aware)"},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
