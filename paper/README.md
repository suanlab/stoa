# STOA paper source

**Target (2026-10-02): ICDE 2027, research track, Experiment, Analysis & Benchmark (EAB), round 2 —
deadline 2026-11-11, 5 PM PT.** Title: *[Experiment, Analysis, and Benchmark] Decide on Arrival:
Placement Timing Beats Placement Prediction on Production KV Traces.* The paper reports what production
traces say about KV-block placement: split-instant evaluation leaves ~95% of a hindsight oracle's
advantage unreachable, and moving the decision to each block's arrival raises the reachable share from
5.3% to 83.5% (conversation) and from 4.1% to 19.0% (toolagent), where first-come-first-served
admission captures 87.4% and 50.2% of it. On Mooncake's synthetic control, where the benefit is not on
new blocks, timing recovers nothing — which the same diagnostic predicts.

The PVLDB-formatted version is `main_pvldb.tex`; its exact last revision is git tag
`pvldb-final-2026-09`. It was never submitted. The ICDE version drops the two LLM-dependent sections
(they need a paid hosted model, and EAB requires every result to be reproducible "no exceptions") and
demotes the formulation from contribution to frame. See `../docs/claims_dependency.md` §AO.

**Five claims were retracted on the way here** and two further sub-claims falsified, including the
previous framing ("Ranking Is Not Placement"). See `../docs/claims_dependency.md` §I, §T, §U (the
retractions), §AC (the two falsifications), and §AD onward (the guards added so each class fails
loudly next time).

The experimental record is `../docs/claims_dependency.md` — the lab notebook, one section per finding,
in the order they were found. `../docs/research_plan.md` is the **pre-registration**, written 2026-07-15
before any of these experiments ran; read it for what we intended to measure, not for what we found.

## Build

IEEE conference format (`IEEEtran.cls`, `IEEEtran.bst`, both from the local TeX installation).

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"   # userspace TeX already installed here
pdflatex main && bibtex main && pdflatex main && pdflatex main   # -> main.pdf
python3 ../scripts/make_figures.py                     # regenerate figs/ from experiments/*.json
python3 ../scripts/check_paper_numbers.py              # every number, guard and requirement below
```

ICDE requirements, each enforced by `check_paper_numbers.py` rather than remembered:

1. **Title starts with `[Experiment, Analysis, and Benchmark]`** (removed at camera-ready).
2. **Real author names** — single-blind. `\author{STOA Project}`, the git user name, sat on page 1
   through every PVLDB pass; `verify.author_block_problems` refuses it and the current placeholder.
3. **AI-generated content disclosed in the acknowledgments**, which ICDE excludes from the page count.
   The section is drafted from what the commit history records; the author's own statement of their
   role is a marked slot only the author can fill, and the checker refuses it while it is empty.
4. **12-page body, excluding references and the acknowledgement.** The acknowledgement precedes the
   references, so the body ends at whichever heading comes first; `verify.body_pages` matches headings
   as whole lines after collapsing IEEEtran's small-caps letter-spacing ("R EFERENCES").
5. **No appendix.**
6. **Artifact URL in the paper**: `\artifacturl` must be defined *and* rendered with
   `\url{\artifacturl}`; `STOA_CHECK_URL=1` fetches it with no credentials.
7. **BibTeX's own log is clean** (`main.blg`), and **the PDF publishes no internal notes** — both were
   violated by the PVLDB build for its whole life, unseen because every build discarded BibTeX's output
   and every check read the source rather than the PDF.

`main_cidr.tex` preserves the superseded CIDR 2027 version; `main_simple.tex` is a plain-`article`
fallback.

## Venue status — target: ICDE 2027 EAB, round 2

Verified from the official call (icde2027.github.io/cf-research-papers.html, 2026-10-02):

| Item | Value |
|---|---|
| Deadline | **2026-11-11, 5 PM PT**; no separate abstract deadline |
| Review | **single-blind**; at least three reviewers; rebuttal 2027-01-08 to 01-15 |
| Notification | 2027-02-10 |
| Page limit | **12 pages, excluding references and the AI-generated content acknowledgement**; IEEE format; no appendix |
| EAB requirement | "MUST provide all artifacts necessary to reproduce the results. No exceptions." Supplemental material via an openly accessible URL |
| Concurrent submission | not permitted |

Category fit: EAB papers provide "fundamentally new insights into the strengths and weaknesses of
existing methods" through extensive evaluation — here, of the split-instant evaluation protocol itself,
and of ARC, LeCaR, CACHEUS and LRB on production KV traces.

Reproducibility package: `../REPRODUCIBILITY.md` and `make verify`.

## Length

Body ends partway down page 12 under the ICDE rule; the PDF is 12 pages including references. Additions
now cost space directly. Measure the body, not the file: `stoa.verify.body_pages` with the ICDE end
headings is the quantity the venue constrains. A calibration pass against a live vLLM+LMCache deployment
is still the open gate (`docs/claims_dependency.md` §B, §P).

## Figures

Generated into `figs/` from `experiments/*.json` by `../scripts/make_figures.py`. **Two are embedded:**
`fig_reactive.pdf` (reactive envelope, ARC and LRB) and `fig_architecture.pdf` (the control-plane
schematic, which plots no data). The other six are produced for a longer version and are *not* in the
paper — notably `fig_locomo_baselines.pdf` (leak-free LoCoMo: per-query retrieval dominates any
query-agnostic placement) and `fig_capacity_ladder.pdf`.

An earlier revision of this paragraph named `fig_capacity_ladder.pdf` as one of the two used. It
appears in no `.tex` file and never has; its description was edited twice without the premise being
checked. `check_paper_numbers.py` now derives figure coverage from the `\includegraphics` calls
themselves, so a rule aimed at an unused figure — or a used figure with no rule — fails the run.

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
| placement vs retrieval, representation axis | **omitted from the ICDE version** (LLM-dependent; in `main_pvldb.tex`) |
| §5.2 synthetic control | Mooncake's third workload; unreachable share 45.3–83.5%; timing boundary |
| §5.8 protocol | ten defects + the controls that catch them |

Every number in the paper is asserted against its artifact by `../scripts/check_paper_numbers.py`; run it
after regenerating any experiment.
