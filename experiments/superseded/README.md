# Superseded artifacts

These are not current results. They are kept because they are the evidence behind retractions
recorded in `docs/claims_dependency.md`, and deleting them would remove the record of what was
believed and why. Nothing in the paper, in `scripts/check_paper_numbers.py`, or in
`REPRODUCIBILITY.md` reads anything in this directory.

| file | what it was | why it is not current |
|---|---|---|
| `kvct_placement*.json` | learned placement on agentic (kv-cache-tester) traces | §I: the headline "+66–88% captured" reversed sign at a larger sample; §J showed randomized draws span hundreds of points, so the quantity was never measurable under that protocol |
| `mooncake_rq2_full.json`, `mooncake_session_rq2_full*.json` | intermediate length-sweep runs | superseded by `mooncake_length_sweep.json`, and by §T: the "captured" denominator they use is 97% unreachable by construction |
| `reactive_kvct*.json` | reactive tiering on agentic traces | probe runs at two conversation counts; the sampling study (§J) showed that corpus cannot support a point estimate at these sizes |
| `lrb_sweep.json` | LRB hyperparameter grid (12 points, 2 traces) | §U: it was run on the implementation whose training buffer was truncated to its tail, so from the third retrain the policy was uniform random eviction. The sweep therefore tested the robustness of random replacement, not of LRB. Not repeated. |
| `eval_locomo_multi.json`, `eval_memqa_multihop.json` | early multi-hop QA probes | superseded by `eval_locomo_powered.json`, which is the run at the sample size the power analysis demanded |

If you are reproducing the paper, ignore this directory entirely.
