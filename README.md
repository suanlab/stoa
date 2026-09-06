# STOA — Storage & Tiered Orchestration for Agents

Reproducibility package for **"Decide on Arrival: Placement Timing Beats Placement Prediction on
Production KV Traces"** (submitted to PVLDB, Experiments Analysis & Benchmarks track).

The paper asks what a KV-block placement policy has to get right, and answers that it has to decide
*early* rather than *well*. The short version:

- **Split-instant evaluation understates placement.** On the two production traces we use, 45–46% of
  blocks have no access before the split, and they carry **97%** of a hindsight oracle's advantage. A
  policy measured inside that protocol looks flat however well it ranks. The diagnostic is one extra
  placement pass (`stoa.sequential.attainable_ceiling`); we recommend reporting the reachable share
  alongside any "fraction of headroom captured".
- **Moving the decision to each block's arrival recovers the benefit** — reachable share 3% → **83.5%**
  on one trace, 2% → **19.0%** on the other, a factor of 27 and 8. There, first-come-first-served
  admission with a fixed action captures **87.4%** and **50.2%** of what is reachable, and a learned
  arrival-time ranker captures *less* on both. But the arrival **order** is not free: shuffling it costs
  3.3 points on one trace and **14.1** on the other. So the implication is narrower than "deciding at
  all is enough" — build an admission point, and do not assume the order it admits in is incidental.
- **Reaction is competitive.** Rules that forecast nothing reach 75–99% / 40–94% of Belady, and the
  adaptive family (ARC, LeCaR, CACHEUS) beats the best of them by at most 4.6 points.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,rl,eval]"
pip install -r requirements-lock.txt      # exact versions the paper's numbers came from

pytest -q                                  # 173 tests, ~2 min
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
| `experiments/` | the artifacts every number in the paper is checked against |
| `experiments/superseded/` | artifacts behind retracted claims, kept as record, read by nothing |
| `paper/` | LaTeX source, PVLDB template vendored |
| `docs/CONVENTIONS.md` | architecture, commands, and the conventions the codebase holds itself to |
| `docs/claims_dependency.md` | the lab notebook: every claim, what it depends on, and what was retracted |
| `tests/` | 173 tests, most of which pin a specific defect we hit |

## What is hand-rolled, and why it matters

scipy, scikit-learn and LightGBM are **absent by design, with no fallback path**. Fisher's exact test,
exact McNemar and the power simulations (`src/stoa/stats.py`), the gradient-boosted trees
(`src/stoa/gbdt.py`), and every cache policy (ARC, LeCaR, CACHEUS, LRB) are reimplementations. A
reproducer who has those libraries installed therefore gets identical numbers — nothing here silently
switches implementation based on what is present.

The cost is that our GBDT is not `HistGradientBoostingRegressor`, and we say so where it matters.

## Read this before trusting a number

`docs/claims_dependency.md` records **twenty-four defects whose only symptom was a plausible number**, five
claims we retracted, and two further sub-claims falsified after the paper had been rewritten around the
surviving result. Two of the retractions were found by adversarial review *after* the error catalogue in
the paper had been written; the two falsifications were found by a re-verification pass that noticed
three cited artifacts predated a fix to the code that produced them. Every one produced a
reasonable-looking aggregate. None was caught by a check that did not compare a reported quantity
against what it was arithmetically able to be — or against the timestamp of the program that wrote it.

That document is not an apology; it is the part of this work most likely to transfer. If you take one
thing from this repository, take the habit of asking what a number's denominator could have been.

## Status

- The LRB column is reported as a **band, not a point**: two defensible repairs of the same bug differ
  by up to 2.1 points, and an independent reimplementation differs from ours by 17. What survives is the
  ordering — LRB below ARC at all eight operating points under every repair.
- The timing axis of the formulation is not exercised at all; the representation axis only as an
  underpowered pilot.
- No live vLLM+LMCache deployment was measured. Cost tables are derived from stated hardware
  assumptions and swept, not calibrated.

## Licence

Code, experiment artifacts and documentation: **Apache-2.0** (see `LICENSE`). This matches the Mooncake
project whose traces we measure, and carries an explicit patent grant.

The paper text under `paper/` is the authors'. Third-party data is **not** redistributed — see
`data/README.md` for each source and its terms; we make no licence claim over any of it.
