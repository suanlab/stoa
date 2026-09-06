# STOA paper source

**Current framing (2026-09-05):** *Decide on Arrival: Placement Timing Beats Placement Prediction on
Production KV Traces.* The paper gives the budget-constrained formulation, then reports what production
traces say about it: split-instant evaluation makes 97% of the achievable benefit unreachable, moving the
decision to each block's arrival recovers it, and what captures the recovered benefit is deciding early
rather than deciding well — though not order-independently, which an earlier draft got wrong.

**Five claims were retracted on the way here** and two further sub-claims falsified, including the
previous framing ("Ranking Is Not Placement"), which was an artifact of normalizing against an oracle a
causal policy cannot reach. See `../docs/claims_dependency.md` §I, §T, §U (the retractions), §AC (the
two falsifications), and §AD–§AF (the guards added so each class fails loudly next time).

The experimental record is `../docs/claims_dependency.md` — the lab notebook, one section per finding,
in the order they were found. `../docs/research_plan.md` is the **pre-registration**, written 2026-07-15
before any of these experiments ran; read it for what we intended to measure, not for what we found.

## Build (verified: **12-page body** — the limit excludes references, so the file is 13 pages; 0 undefined citations/references, no `??`)

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
2. `\vldbdoi` / `\vldbpages` / `\vldbavailabilityurl` set via `\renewcommand`. The availability URL is
   the **EA&B reproducibility-package link** and it currently reads `ANONYMIZED`. Two things must happen
   before submission, in this order:

   **(a) The artifact repository is PRIVATE.** Flip it, then put the URL in `\vldbavailabilityurl`:

   ```bash
   gh repo edit <owner>/stoa --visibility public --accept-visibility-change-consequences
   ```

   **(b) Verify the link the way a reviewer will see it, not the way the tool reports it.** A successful
   `gh repo edit` is a report; a URL that opens is an observation. EA&B states the requirement as "there
   are no excuses", and a private URL renders as a 404 to the committee.

   ```bash
   curl -sSo /dev/null -w '%{http_code}\n' https://github.com/<owner>/stoa   # must be 200, not 404
   ```

   Run that from a shell with no GitHub credentials, or in a logged-out browser. Checking it while
   authenticated tests your access, not the reviewer's — which is the exact failure mode
   `../docs/claims_dependency.md` catalogues twenty-five times over.
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
| Next cycle | abstract **2026-09-25**, paper **2026-10-01** (we are deliberately not targeting 09-01; see `../docs/claims_dependency.md` §AB) |
| Page limit | **12 pages excluding references** (this draft: body ends on p. 12, references run to p. 13 — at the limit, ~3 lines spare) |
| EA&B requirement | all experimental data and software public; **reproducibility package linked at submission**; evaluated by the PVLDB Reproducibility Committee |

Category fit: the research track's *Workload Characterization* papers cover "real-world workload
characteristics, the working of existing methods and systems under such workloads … insights into
workloads and reusable artifacts such as benchmark suites or traces" — which is this paper.

Reproducibility package: `../REPRODUCIBILITY.md` (claim → command → artifact, plus the controls).

## Length

**12-page body** against a 12-page limit (the PDF is 13 pages; references do not count). The draft is
*at* the limit with about three lines to spare, so any addition now requires a deletion. Measure the
body, not the file: `stoa.verify.body_pages` returns the page the bibliography starts on, which is the
quantity the venue constrains — five re-verification passes checked the file and agreed with the truth
only by coincidence. Remaining work is measurement, not writing: the representation pilot (§5.7) needs 129–718
questions per cell against the 100 run, and the LoCoMo demand-vs-random contrast needs 758 / 1078 /
2589 questions (one per budget, at the Bonferroni-corrected alpha the paper actually judges at)
against the 300 run — an earlier revision of this file said "486–667", which dropped the third budget
and sized against an uncorrected alpha. A calibration pass against a live vLLM+LMCache deployment is
still the open gate (`docs/claims_dependency.md` §B, §P).

## Figures

Generated into `figs/` from `experiments/*.json` by `../scripts/make_figures.py`. Two are used —
`fig_reactive.pdf` (reactive envelope, ARC and LRB) and `fig_capacity_ladder.pdf` (ranking quality
against captured benefit across model class). The rest are available for a longer version, notably `fig_locomo_baselines.pdf` (leak-free
LoCoMo: per-query retrieval dominates any query-agnostic placement).

`fig_reactive` prefers **`reactive_real_full_lrb.json`** (the `--lrb` run) and falls back to
`reactive_real_full.json`; it never reads a subsampled run. The run script derives its output name from
the scale (`_full` vs `_sub{N}`) precisely so a subsample cannot overwrite the full-trace artifact the
paper reads — an earlier revision had two such files diverge silently.

## Section status

| Section | State |
|---|---|
| Abstract, §1–§8 | complete prose, aligned to the arrival-timing framing |
| Table 1 (positioning) | complete |
| §5.2 length sweep | both traces run to exhaustion |
| §5.3 capacity ladder | linear / GBDT / MLP / oracle, full traces |
| §5.5 reaction | full traces; ARC, LeCaR, CACHEUS + LRB, with ghost-hit and censoring diagnostics |
| §5.6 placement vs retrieval | 10 dialogues, 300 questions/arm, paired McNemar + sign test |
| §5.7 representation axis | first exercise of the axis; underpowered pilot, direction reported |
| §5.8 protocol | ten defects + the controls that catch them |

Every number in the paper is asserted against its artifact by `../scripts/check_paper_numbers.py`; run it
after regenerating any experiment.
