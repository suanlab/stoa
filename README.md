# STOA — Storage & Tiered Orchestration for Agents

Reproducibility package for **"The Unseen Block: What Split-Instant Evaluation
Hides in KV-Cache Placement"** (prepared for PVLDB Volume 20 / VLDB 2027, Experiment, Analysis & Benchmark category).

The paper asks what an offline evaluation of KV-block placement has to fix, and finds that on production
traces the order it gives **blocks without evidence** decides what it reports. The short version:

- **Most blocks a policy can see are dead.** On the two production Mooncake traces, **92.8% and 95.0%** of
  the blocks seen before the midpoint split are never read again (88.8–96.0% across splits); the few live
  ones fit in the fastest tier, and the rest of the reuse falls on blocks not yet created.
- **A tie rule was the headline.** A clairvoyant split-time reference gives dead seen blocks and unseen
  blocks the same belief, zero. Broken by item order (dead blocks first), as earlier versions of this
  paper did, it reaches 20.4% and 15.3% of a hindsight oracle's advantage (tier-only actions); with dead
  blocks parked behind new ones, **77.8% and 76.2%**. Neither is an upper bound (an exhaustive check on
  small workloads beats both). The earlier "84.4% / 87.6% is out of a split-time policy's reach" and the
  "2.3–4.9x the split ceiling" ratio were this tie rule (notebook §AZ; `run_ceiling_ties.py`).
- **Deadness, not ranking, is what placement has to get right.** Knowing which seen blocks are dead is worth
  exactly as much as knowing every seen block's future; a parameter-free rule that parks seen-once blocks
  behind new ones captures 39.2% / 43.3%, an estimated prior 47.0% / 51.7%, a better ranker of seen blocks
  at most 8.2% / 7.4% (`run_evidence_ladder.py`, notebook §BA).
- **Online, the price of a move decides.** A multi-tier LRU captures 84.5% / 87.7% when moves are free;
  timed on an A100 PCIe host a move costs 0.75–1.87x one read plus one write, and under five ways of
  billing a move none of the replayed policies that move blocks beats the static frequency heuristic
  (`run_reactive_cost.py`, `measure_tier_costs.py`, `run_migration_ledger.py`; §AV, §AX, §AY).
- **Reaction stays near plain rules.** Rules that forecast nothing reach 75–99% of Belady on toolagent and
  40–94% on conversation; ARC, LeCaR and CACHEUS beat the best of them by at most 4.6 points, LRB stays
  below ARC, and S3-FIFO, SIEVE and W-TinyLFU stay below the best plain rule everywhere
  (`run_modern_baselines.py`, §BB).

A block's creation (its KV is computed) is a compulsory miss and is never billed; billing it as a cache
read, as earlier versions did, inflated the oracle's advantage about 3x (notebook §AS).

## Quick start

```bash
# poppler-utils (pdftotext, pdfinfo, pdffonts) is needed by the checker, which reads the built PDF:
#   apt-get install poppler-utils   |   brew install poppler
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact versions the paper's numbers came from, FIRST
pip install -e . --no-deps                # then the package itself, without re-resolving them

pytest -q                                  # 318 tests, ~3 min
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
| `src/stoa/{rl,train,orchestrator,policy,llm}.py`, `src/stoa/eval/{benchmarks,budget,locomo,memqa,representation,retrieval}.py` | scaffolding for the §3–4 frame and the LLM results of the earlier PVLDB version; no number in this paper uses them |
| `experiments/` | the artifacts every number in the paper is checked against |
| `experiments/superseded/` | artifacts behind retracted claims, kept as record, read by nothing |
| `paper/` | LaTeX source (official PVLDB template; the ICDE version is `main_icde.tex`, an earlier PVLDB version `main_pvldb.tex`) |
| `docs/CONVENTIONS.md` | architecture, commands, and the conventions the codebase holds itself to |
| `docs/claims_dependency.md` | the lab notebook: every claim, what it depends on, and what was retracted |
| `tests/` | 318 tests, most of which pin a specific defect we hit |

## What is hand-rolled, and why it matters

scipy, scikit-learn and LightGBM are **absent by design, with no fallback path**. Fisher's exact test,
exact McNemar and the power simulations (`src/stoa/stats.py`), the gradient-boosted trees
(`src/stoa/gbdt.py`), and every cache policy (ARC, LeCaR, CACHEUS, LRB) are reimplementations. A
reproducer who has those libraries installed therefore gets identical numbers — nothing here silently
switches implementation based on what is present.

The cost is that our GBDT is not `HistGradientBoostingRegressor`, and we say so where it matters.

## Read this before trusting a number

The paper's §V-G catalogues **twenty-seven measurement defects whose only symptom was a plausible
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
