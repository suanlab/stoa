# STOA paper source

**Current framing (2026-08-13):** *Ranking Is Not Placement: What Production KV Traces Say About Learned
Memory Tiering.* The paper gives the budget-constrained formulation, then reports what production traces
say about learning it — a decoupling result (ranking quality improves; the placement benefit it buys does
not) plus the measurement protocol that separated the two. Not a learned-policy win, and deliberately so:
the earlier headline claim was retracted (see `../docs/claims_dependency.md` §I). Full experimental record
in `../docs/research_plan.md`.

## Build (verified: 11 pages, 0 undefined citations/references, no `??` in the PDF)

Uses the **official VLDB template** (github.com/vldbproceedings/VLDB-Template), vendored here so the build
does not depend on the local TeX installation's acmart version:

| File | Provenance |
|---|---|
| `acmart.cls` | v2.19 — the version the template pins |
| `pvldb.sty` | v1.0, "VLDB 2027 / PVLDB Vol. 20" — matches our target volume |
| `ACM-Reference-Format.bst` | ACM reference style |

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"   # userspace TeX already installed here
pdflatex main && bibtex main && pdflatex main && pdflatex main   # -> main.pdf
python3 ../scripts/make_figures.py                     # regenerate figs/ from experiments/*.json
```

Three template requirements are easy to break and produce a silently wrong front page:

1. `\documentclass[sigconf, nonacm]{acmart}` followed by `\usepackage{pvldb}` inside the VLDB block.
2. `\vldbdoi` / `\vldbpages` / `\vldbavailabilityurl` set via `\renewcommand` (the availability URL is
   the **EA&B reproducibility-package link** and must point at the real repository before submission —
   it currently reads `ANONYMIZED`).
3. **`\vldbtopmatter` immediately after `\maketitle`.** `pvldb.sty` has no `AtBeginDocument` hook, so
   omitting this drops the PVLDB Reference Format and Artifact Availability blocks with no error. Check
   for them in the rendered page 1, not in the log.

`\let\Bbbk\relax` before `amssymb` avoids a clash with the `newtxmath` that acmart already loads.

`main_cidr.tex` preserves the superseded CIDR 2027 version (that deadline, 2026-08-04, has passed);
`main_simple.tex` is a plain-`article` fallback.

## Venue status — target: PVLDB Workload Characterization / EA&B

Verified from the official PVLDB Vol. 20 submission guidelines (2026-08-12):

| Item | Value |
|---|---|
| Deadline | **1st of each month, 5 PM PT**; abstract mandatory by the **25th of the prior month** |
| Next cycle | abstract **2026-08-25**, paper **2026-09-01** |
| Page limit | **12 pages excluding references** (this draft is 11) |
| EA&B requirement | all experimental data and software public; **reproducibility package linked at submission**; evaluated by the PVLDB Reproducibility Committee |

Category fit: the research track's *Workload Characterization* papers cover "real-world workload
characteristics, the working of existing methods and systems under such workloads … insights into
workloads and reusable artifacts such as benchmark suites or traces" — which is this paper.

Reproducibility package: `../REPRODUCIBILITY.md` (claim → command → artifact, plus the controls).

## Length

**11 pp** against a 12-page limit, so space is now the binding constraint. Remaining work is
measurement, not writing: the representation pilot (§5.7) needs 129–718 questions per cell against the
100 run, the LoCoMo demand-vs-random contrast needs 486–667 against 300, and the timing axis has not
been exercised at all. A calibration pass against a live vLLM+LMCache deployment is still the open gate
(`docs/claims_dependency.md` §B, §P).

## Figures

Generated into `figs/` from `experiments/*.json` by `../scripts/make_figures.py`. Two are used —
`fig_reactive.pdf` (reactive envelope, ARC and LRB) and `fig_capacity_ladder.pdf` (the decoupling across
model class). The rest are available for a longer version, notably `fig_locomo_baselines.pdf` (leak-free
LoCoMo: per-query retrieval dominates any query-agnostic placement).

`fig_reactive` prefers **`reactive_real_full_lrb.json`** (the `--lrb` run) and falls back to
`reactive_real_full.json`; it never reads a subsampled run. The run script derives its output name from
the scale (`_full` vs `_sub{N}`) precisely so a subsample cannot overwrite the full-trace artifact the
paper reads — an earlier revision had two such files diverge silently.

## Section status

| Section | State |
|---|---|
| Abstract, §1–§8 | complete prose, aligned to the decoupling framing |
| Table 1 (positioning) | complete |
| §5.2 length sweep | both traces run to exhaustion |
| §5.3 capacity ladder | linear / GBDT / MLP / oracle, full traces |
| §5.5 reaction | full traces; ARC, LeCaR, CACHEUS + LRB, with ghost-hit and censoring diagnostics |
| §5.6 placement vs retrieval | 10 dialogues, 300 questions/arm, paired McNemar + sign test |
| §5.7 representation axis | first exercise of the axis; underpowered pilot, direction reported |
| §5.8 protocol | ten defects + the controls that catch them |

Every number in the paper is asserted against its artifact by `../scripts/check_paper_numbers.py`; run it
after regenerating any experiment.
