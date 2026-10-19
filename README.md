# STOA — Storage & Tiered Orchestration for Agents

Reproducibility package for **"Decide on Arrival: Placement Timing Beats Placement Prediction on
Production KV Traces"** (prepared for ICDE 2027, Experiment, Analysis & Benchmark category).

The paper asks what a KV-block placement policy has to get right, and answers that it has to decide
*early* rather than *well*. The short version:

- **Split-instant evaluation understates placement.** On the two production traces we use, 45–46% of
  blocks have no access before the split, and they carry **94.7% and 95.9%** of a hindsight oracle's
  advantage, so the best possible split-time policy reaches **5.3% and 4.1%** of it. The share stays at
  92–99% for every split from 0.3 to 0.7 and under order-of-magnitude changes to the cost weights.
  Inside that protocol a better ranker helps monotonically, but the strongest model gains under 3% of
  the advantage. The diagnostic is one extra placement pass (`stoa.sequential.attainable_ceiling`); we
  recommend reporting the reachable share alongside any "fraction of headroom captured".
- **Deciding at each block's arrival puts most of the advantage within reach.** Every arrival-time rule we
  test captures **45–85%** of it on production, against 5.3% / 4.1% for the best split-time policy. Within
  that range, the order new blocks are ranked in barely matters: with their magnitudes held equal, arrival
  order, a learned ranker and a random order land within 1.1 / 0.9 points. What matters is the reuse a
  rule assumes for a block it has never seen: moving that constant from the median (1.0) to the mean
  (1.54 / 2.14) moves the result by **29.5 / 39.1 points**. On Mooncake's synthetic control the margin
  is not robust (33.6–52.6% against a 35.0% split-time ceiling). Two earlier versions of this experiment
  were wrong, one leaking future information and one comparing order arms that also differed in
  magnitude; the paper's §V-C and the notebook's §AP and §AQ say how.
- **Reaction is competitive.** Rules that forecast nothing reach 75–99% of Belady on toolagent and 40–94% on conversation, and the
  adaptive family (ARC, LeCaR, CACHEUS) beats the best of them by at most 4.6 points.

## Quick start

```bash
# poppler-utils (pdftotext, pdfinfo, pdffonts) is needed by the checker, which reads the built PDF:
#   apt-get install poppler-utils   |   brew install poppler
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact versions the paper's numbers came from, FIRST
pip install -e . --no-deps                # then the package itself, without re-resolving them

pytest -q                                  # 279 tests, ~3 min
python3 scripts/check_paper_numbers.py     # every paper number vs its artifact
```

`check_paper_numbers.py` re-derives each headline quantity from `experiments/*.json` and greps the
LaTeX source for it. It exits non-zero on drift, on a missing artifact, and on too few checks — a
checker that passes vacuously is worse than none.

Then fetch the data (`data/README.md`) and re-run whatever you want; `REPRODUCIBILITY.md` maps every
claim to a command and an artifact.

## Layout

| path | what |
|---|---|
| `src/stoa/` | the library: MDP interfaces, simulator, placement, cache policies, statistics |
| `scripts/` | one script per experiment, each writing a named artifact |
| `src/stoa/{rl,train,orchestrator,policy,llm}.py`, `src/stoa/eval/{benchmarks,budget,locomo,memqa,representation,retrieval}.py` | scaffolding for the §3–4 frame and the PVLDB-only LLM results; no ICDE number uses them |
| `experiments/` | the artifacts every number in the paper is checked against |
| `experiments/superseded/` | artifacts behind retracted claims, kept as record, read by nothing |
| `paper/` | LaTeX source (IEEE format for ICDE; the PVLDB version is `main_pvldb.tex`) |
| `docs/CONVENTIONS.md` | architecture, commands, and the conventions the codebase holds itself to |
| `docs/claims_dependency.md` | the lab notebook: every claim, what it depends on, and what was retracted |
| `tests/` | 279 tests, most of which pin a specific defect we hit |

## What is hand-rolled, and why it matters

scipy, scikit-learn and LightGBM are **absent by design, with no fallback path**. Fisher's exact test,
exact McNemar and the power simulations (`src/stoa/stats.py`), the gradient-boosted trees
(`src/stoa/gbdt.py`), and every cache policy (ARC, LeCaR, CACHEUS, LRB) are reimplementations. A
reproducer who has those libraries installed therefore gets identical numbers — nothing here silently
switches implementation based on what is present.

The cost is that our GBDT is not `HistGradientBoostingRegressor`, and we say so where it matters.

## Read this before trusting a number

The paper's §V-G catalogues **twenty-five measurement defects whose only symptom was a plausible
number**; `docs/claims_dependency.md` records each one, with its fix and the guard that now catches it.
Five claims were retracted (the paper counts three; two predate its framing). Two of the retractions
were found by adversarial review *after* the error catalogue had been written. Two further sub-claims
were falsified by a re-verification pass, and later withdrawn with two more readings when an audit
found that the arrival-time experiment had leaked future information into its labels and belief
magnitudes (§AP), after every number check had passed; re-verifying that fix found its order controls
had differed in magnitude too, and two more readings went (§AQ). Every one produced a reasonable-looking
aggregate. None was caught by a check that did not compare a reported quantity
against what it was arithmetically able to be — or against the timestamp of the program that wrote it.

That document is not an apology; it is the part of this work most likely to transfer. If you take one
thing from this repository, take the habit of asking what a number's denominator could have been.

## Status

- The LRB column is reported as a **band, not a point**: two defensible repairs of the same bug differ
  by up to 2.1 points. What survives is the
  ordering — LRB below ARC at all eight operating points under every repair.
- The belief an admission point assigns an unseen block is set by hand (the previous period's mean or
  median); learning it is open. Ranking rules against each other would also need an equal-exposure
  evaluation (a rolling window or a fixed post-arrival horizon), which this artifact does not contain.
- The timing axis of the formulation is not exercised at all; the representation axis only as an
  underpowered pilot.
- No live vLLM+LMCache deployment was measured. Cost tables are derived from stated hardware
  assumptions and swept, not calibrated.

## Licence

Code, experiment artifacts and documentation: **Apache-2.0** (see `LICENSE`). This matches the Mooncake
project whose traces we measure, and carries an explicit patent grant.

The paper text under `paper/` is the authors'. Third-party data is **not** redistributed — see
`data/README.md` for each source and its terms; we make no licence claim over any of it.
