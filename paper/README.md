# STOA paper source

**Target (2026-10-20): PVLDB Volume 20 (VLDB 2027), research track, Experiment, Analysis & Benchmark
(EA&B).** Monthly deadlines on the 1st at 5 PM PT, with a mandatory abstract by the 25th of the month
before. Title: *The Unseen Block: How Evaluation Conventions Decide KV-Cache Results [Experiment, Analysis &
Benchmark]* (retitled 2026-11-24 from "… What Split-Instant Evaluation Hides in KV-Cache Placement", notebook
BN). It was first retitled on 2026-10-20 from "Decide on Arrival: Placement Timing Beats
Placement Prediction on Production KV Traces", which the corrected results no longer support. The
target moved from ICDE 2027 (EAB, round 2), whose IEEE-formatted version is kept as `main_icde.tex`; it
was never submitted. The paper reports what an offline evaluation of KV-block placement has to fix. On the production Mooncake
traces 92.8% and 95.0% of the blocks seen before the split are never read again, and how an evaluation
orders these and the not-yet-created blocks, all without evidence, is a tie rule: the same clairvoyant
split-time reference reaches 20.4% / 15.3% of the oracle's advantage with dead blocks first and 77.8% /
76.2% with them last (tier-only). Knowing which seen blocks are dead is most of the prize; a seen-once
rule recovers about half of it; online, at measured move prices, nothing that moves blocks beats a static
frequency heuristic. **Reframed 2026-11-06** after notebook §AZ showed the earlier headline ("84.4% /
87.6% out of a split-time policy's reach", "2.3–4.9x the split ceiling") was that tie rule; sections 3–4
(the MDP formulation and control-plane sketch) and Table 1 were removed, and the paper was then organized
around blocks without evidence. Since 2026-11-24 it is organized around three conventions: the tie rule (§5),
the matching rule (§6) and the write path (§7), with the remaining findings in §8 (notebook BN).

An earlier PVLDB version, with a different framing, is `main_pvldb.tex` (git tag `pvldb-final-2026-09`);
it was never submitted. Since the ICDE version, the paper drops the two LLM-dependent sections
(they need a paid hosted model, and EA&B requires all experimental data and software, "there are no
excuses") and
demotes the formulation from contribution to frame. See `../docs/claims_dependency.md` §AO.

**Six claims were retracted on the way here**, four of which the paper counts (three in its "Retracted
claims" row and the split-time "ceiling" of §AZ; the other two predate its framing), including the previous framing ("Ranking Is Not Placement"). Two further sub-claims were
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

The paper's body, acknowledgements included, ends on page 9, where the references begin: about three of
the twelve pages are unused, left for what reviewers will ask. Measure the body,
not the file: `stoa.verify.body_pages` with REFERENCES as the end heading is the quantity the venue
constrains. Block-move costs are measured on one A100 PCIe host (§AX); a full serving deployment
(vLLM+LMCache), NVMe and the remote RDMA tier remain unmeasured.

## Figures

Generated into `figs/` from `experiments/*.json` by `../scripts/make_figures.py`. **Three are embedded:**
`fig_ties.pdf` (the reference under both tie rules across splits), `fig_price.pdf` (captured share against
the price of a move) and `fig_reactive.pdf` (reactive envelope, ARC and LRB). The control-plane schematic `fig_architecture.pdf`
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
| §1 Introduction, §2 Background (with the evaluation-protocol table) | complete; references verified (notebook §BC) |
| §3 What an evaluation has to fix | the three conventions (tie rule, matching rule, write path), then split, billing, references, move ledgers |
| §4 Workloads | Mooncake x3, Alibaba Qwen x4, PrefixSim's two traces; A100 host for move timings, A6000 host for the serving replay |
| §5 Convention 1: the tie rule (5.1–5.6) | dead shares, tie-rule references and their capacity dependence (`ties_by_capacity.json`, scale 0.05 ≈ one GPU), evidence ladder, why (Robbins / Good–Turing, age and session rules), the ranker ladder, arrival-time admission |
| §6 Convention 2: which hits count | PrefixSim on its own traces: counted as vLLM/SGLang match, RandomCompute's +10.4-point lead over LRU becomes −19.6 (`run_prefixsim_audit.py`) |
| §7 Convention 3: what a store costs (7.1–7.2) | vLLM + LMCache per-block prices on the A100 (`serving_tiers_a100.json`); store admission replayed into vLLM through a blocking (LMCache) and an asynchronous (offloading) connector on one RTX A6000, and SGLang HiCache's three write policies (`write_contract_a6000.json`, `run_write_contract.py`, `serving_replay/`) |
| §8 With the conventions fixed (8.1–8.2) | recency, measured move price, five ledgers, ski rental, the measured latency table; reaction: ARC/LeCaR/CACHEUS/LRB + S3-FIFO/SIEVE/W-TinyLFU and libCacheSim's implementations |
| §9 What the measurement took | thirty-one defects, four retracted claims |

Every number in the paper is asserted against its artifact by `../scripts/check_paper_numbers.py`; run it
after regenerating any experiment.
