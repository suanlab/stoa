"""Artifact -> source modules each one encodes. Shared by check_paper_numbers.py and
record_provenance.py.

This used to live inline in check_paper_numbers.py, and record_provenance.py read it by
`exec`-ing a slice of that file's source -- fragile (any edit near the dict could change what
was executed) and a bad pattern to ship (audit L2). Both scripts now import it.

`verify.producer_problems` audits it against each producing script's imports, so an omission
fails the check rather than exempting an artifact from staleness tracking.
"""

DEPENDS_ON = {
    # artifact -> modules whose behaviour it encodes. Audited against each producing script's
    # direct imports by verify.producer_problems, which is how the omissions below were found:
    # four artifacts that load the Mooncake traces had not declared mooncake.py (§AO).
    "arrival_admission.json": ("sequential.py", "mooncake.py", "causal.py"),
    "arrival_diagnostics.json": ("sequential.py", "mooncake.py", "causal.py"),
    "weight_sensitivity.json": ("sequential.py", "mooncake.py", "simulator.py"),
    "source_stats.json": ("mooncake.py", "kvct.py"),
    "split_sensitivity.json": ("sequential.py", "mooncake.py", "gbdt.py"),
    "reactive_kvct_sweep.json": ("eval/online.py", "kvct.py"),
    "capacity_ladder_fixed.json": ("sequential.py", "gbdt.py", "mooncake.py"),
    "calibration_sensitivity.json": ("sequential.py", "calibration.py", "simulator.py",
                                     "environment.py", "mooncake.py"),
    "belief_controls.json": ("sequential.py", "mooncake.py"),
    "reactive_real_full_lrb.json": ("eval/online.py", "experts.py", "lrb.py", "mooncake.py"),
    "lrb_retraction.json": ("lrb.py", "eval/online.py", "mooncake.py"),
    "eval_locomo_powered.json": ("eval/memqa.py", "stats.py", "eval/locomo.py",
                                 "eval/retrieval.py", "llm.py"),
    "representation_axis.json": ("eval/representation.py", "stats.py", "eval/locomo.py",
                                 "eval/memqa.py", "llm.py"),
    "locomo_power.json": ("stats.py", "eval/locomo.py", "eval/memqa.py",
                          "eval/retrieval.py", "llm.py"),
    "lrb_sweep.json": ("lrb.py", "eval/online.py", "mooncake.py"),
    "sampling_study.json": ("kvct.py",),
    "reachable_ceiling.json": ("sequential.py", "mooncake.py", "simulator.py"),
    "headroom_by_length.json": ("sequential.py", "mooncake.py"),
}
