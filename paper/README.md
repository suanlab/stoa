# STOA paper source

**Target (2026-10-20): PVLDB Volume 20 (VLDB 2027), research track, Experiment, Analysis & Benchmark
(EA&B).** Monthly deadlines on the 1st at 5 PM PT, with a mandatory abstract by the 25th of the month
before. Title: *The Unseen Block: What Split-Instant Evaluation Hides in KV-Cache Placement [Experiment,
Analysis & Benchmark]*. It was retitled on 2026-10-20 from "Decide on Arrival: Placement Timing Beats
Placement Prediction on Production KV Traces", which the corrected results no longer support. The
target moved from ICDE 2027 (EAB, round 2), whose IEEE-formatted version is kept as `main_icde.tex`; it
was never submitted. The paper reports what production
traces say about KV-block placement: split-instant evaluation leaves 84.4% and 87.6% of a hindsight
oracle's advantage on blocks that do not yet exist, so the best split-time policy reaches 15.6%
(conversation) and 12.4% (toolagent) of it. Deciding at each block's arrival captures 2.3–4.9x as much
when the trust placed in an unseen block is estimated from the prefix; fixed by convention, that decision
moves the result by 14–29 points and can leave it at the split-time ceiling. This is a static analysis;
replayed online on the same cost metric, a multi-tier LRU beats every arrival-time arm when migration is
free, and none of the replayed policies that move blocks beats the static heuristic once a move is priced.

An earlier PVLDB version, with a different framing, is `main_pvldb.tex` (git tag `pvldb-final-2026-09`);
it was never submitted. Since the ICDE version, the paper drops the two LLM-dependent sections
(they need a paid hosted model, and EA&B requires all experimental data and software, "there are no
excuses") and
demotes the formulation from contribution to frame. See `../docs/claims_dependency.md` §AO.

**Five claims were retracted on the way here**, three of which the paper counts (the other two predate
its framing), including the previous framing ("Ranking Is Not Placement"). Two further sub-claims were
falsified (§AC) and then withdrawn with two more readings when an audit found the arrival-time
experiment leaky (§AP); re-verifying that fix withdrew two more (§AQ), six in all. See
`../docs/claims_dependency.md` §I, §T, §U (the retractions), §AC, §AP, §AQ, and
§AD onward (the guards added so each class fails loudly next time).

The experimental record is `../docs/claims_dependency.md` — the lab notebook, one section per finding,
in the order they were found. `../docs/research_plan.md` is the **pre-registration**, written 2026-07-15
before any of these experiments ran; read it for what we intended to measure, not for what we found.

## Build

The official PVLDB template (github.com/vldbproceedings/VLDB-Template, commit 39c95f5): `acmart.cls`
v2.19, `pvldb.sty` and `ACM-Reference-Format.bst`, vendored here byte-identical to upstream so the build
does not depend on the local TeX installation's acmart.

```bash
# any TeX Live; or simply `make paper` from the repository root
python3 ../scripts/make_figures.py                     # FIRST: figs/ from experiments/*.json (deterministic)
pdflatex main && bibtex main && pdflatex main && pdflatex main   # -> main.pdf
python3 ../scripts/record_pdf_build.py                 # content digests, so a fresh clone can verify the PDF
python3 ../scripts/check_paper_numbers.py              # every number, guard and requirement below
```

PVLDB requirements, each enforced by `check_paper_numbers.py` rather than remembered:

1. **Title ends with `[Experiment, Analysis & Benchmark]`**, in the PDF and in the CMT title (dropped
   at camera-ready). The running head uses the short title without it.
2. **Real author names** — single-blind. `\author{STOA Project}`, the git user name, sat on page 1
   through every earlier PVLDB pass; `verify.author_block_problems` refuses it.
3. **12 pages, excluding references only.** Appendices and acknowledgements count, so the body ends at
   the REFERENCES heading (`verify.body_pages`).
4. **The template's VLDB blocks unmodified** (`\usepackage{pvldb}`, `\vldbtopmatter`); DOI and page
   placeholders are allowed at submission, and the leaked-annotation guard exempts exactly those strings.
5. **Reproducibility package linked** in `\vldbavailabilityurl` (rendered as "PVLDB Artifact
   Availability"); EA&B requires it in the initial submission. `STOA_CHECK_URL=1` fetches it with no
   credentials.
6. **AI-generated content disclosed in the acknowledgments** (counted in the 12 pages).
7. **BibTeX's own log is clean** (`main.blg`), and **the PDF publishes no internal notes** — both were
   violated by an earlier PVLDB build for its whole life, unseen because every build discarded BibTeX's
   output and every check read the source rather than the PDF.

`main_cidr.tex` preserves the superseded CIDR 2027 version; `main_simple.tex` is a plain-`article`
fallback.

## Venue status — target: PVLDB Vol. 20, EA&B

Verified from the official guidelines (vldb.org/2027/submission-guidelines.html, 2026-10-20):

| Item | Value |
|---|---|
| Deadlines | the **1st of each month, 5 PM PT**, until 2027-03-01; **abstract mandatory by the 25th of the previous month** |
| Review | **single-blind**; initial reviews by the 15th of the next month; up to 2.5 months to revise |
| VLDB 2027 | papers accepted by 2027-07-01 form the research track (Athens, 2027-08-23 to 27); final revision deadline for it 2027-06-01 |
| Page limit | **12 pages excluding references**; appendices and acknowledgements count; PVLDB template mandatory |
| EA&B requirement | "required to make available all experimental data and related software (there are no excuses)"; the initial submission links the full reproducibility package with instructions |
| Caps | two submissions per author per month, twelve per volume; concurrent submissions must be declared |

Category fit: EA&B papers give "new ways to evaluate existing methods and systems" and "characterization
of workloads in real-world deployments" — here, of the split-instant evaluation protocol itself, and of
ARC, LeCaR, CACHEUS and LRB on production KV traces.

Reproducibility package: `../REPRODUCIBILITY.md` and `make verify`.

## Length

In the PVLDB template the body, acknowledgement included, ends on page 11; references start on page 11
and the PDF is 12 pages. Measure the body, not the file: `stoa.verify.body_pages` with REFERENCES as the
end heading is the quantity the venue constrains. A calibration pass against a live vLLM+LMCache
deployment is still the open gate (`docs/claims_dependency.md` §B, §P).

## Figures

Generated into `figs/` from `experiments/*.json` by `../scripts/make_figures.py`. **One is embedded:**
`fig_reactive.pdf` (reactive envelope, ARC and LRB). The control-plane schematic `fig_architecture.pdf`
was dropped with §IV's compression (2026-10-20). The other ten are produced for a longer version and are *not* in the
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
| §V-G length sweep (headroom 1,500 requests → full) | both traces run to exhaustion |
| §V-D capacity ladder | linear / GBDT / MLP / oracle, full traces |
| §V-F reaction | full traces; ARC, LeCaR, CACHEUS + LRB, with ghost-hit and censoring diagnostics |
| placement vs retrieval, representation axis | **omitted** since the ICDE version (LLM-dependent; in `main_pvldb.tex`) |
| §V-B synthetic control | Mooncake's third workload; unreachable share 29.4–77.9%; no arrival-time arm beats the split ceiling |
| §V-C arrival-time admission | causal (§AP); conventions vs an estimated prior for unseen blocks (§AQ–§AS): estimated, 2.3–4.9x the split ceiling (static; needs free migration online) |
| §V-G protocol | twenty-seven defects + the controls that catch them |

Every number in the paper is asserted against its artifact by `../scripts/check_paper_numbers.py`; run it
after regenerating any experiment.
