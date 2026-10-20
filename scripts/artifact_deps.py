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
    "arrival_admission.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "credit.py",
                               "eval/oracle.py", "gnn.py", "simulator.py"),
    "tier_only.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "credit.py", "eval/oracle.py",
                       "gnn.py", "simulator.py", "environment.py"),
    "arrival_diagnostics.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py"),
    "weight_sensitivity.json": ("traces.py", "sequential.py", "mooncake.py", "simulator.py"),
    "source_stats.json": ("traces.py", "mooncake.py", "kvct.py"),
    "split_sensitivity.json": ("traces.py", "sequential.py", "mooncake.py", "gbdt.py"),
    "reactive_kvct_sweep.json": ("traces.py", "eval/online.py", "kvct.py"),
    "capacity_ladder_fixed.json": ("traces.py", "sequential.py", "gbdt.py", "mooncake.py"),
    "calibration_sensitivity.json": ("traces.py", "sequential.py", "calibration.py", "simulator.py",
                                     "environment.py", "mooncake.py"),
    "belief_controls.json": ("traces.py", "sequential.py", "mooncake.py"),
    "reactive_real_full_lrb.json": ("traces.py", "eval/online.py", "experts.py", "lrb.py", "mooncake.py"),
    "lrb_retraction.json": ("traces.py", "lrb.py", "eval/online.py", "mooncake.py"),
    "eval_locomo_powered.json": ("eval/memqa.py", "stats.py", "eval/locomo.py",
                                 "eval/retrieval.py", "llm.py"),
    "representation_axis.json": ("eval/representation.py", "stats.py", "eval/locomo.py",
                                 "eval/memqa.py", "llm.py"),
    "locomo_power.json": ("stats.py", "eval/locomo.py", "eval/memqa.py",
                          "eval/retrieval.py", "llm.py"),
    "lrb_sweep.json": ("traces.py", "lrb.py", "eval/online.py", "mooncake.py"),
    "sampling_study.json": ("traces.py", "kvct.py", "sequential.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "kvct_placement_fixed.json": ("traces.py", "kvct.py", "sequential.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "kvct_placement_n70.json": ("traces.py", "kvct.py", "sequential.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "kvct_placement_n100.json": ("traces.py", "kvct.py", "sequential.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "kvct_placement_n150.json": ("traces.py", "kvct.py", "sequential.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    # 2026-10-19 experiments (notebook AS)
    "hazard_prior.json": ("traces.py", "sequential.py", "mooncake.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "hazard_prior_tier.json": ("traces.py", "sequential.py", "mooncake.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py", "environment.py", "causal.py"),
    "arrival_admission_reuse.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "tier_only_reuse.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py", "environment.py"),
    "bailian_reachable_birthx.json": ("traces.py", "sequential.py", "mooncake.py", "simulator.py", "environment.py"),
    "bailian_hazard_birthx.json": ("traces.py", "sequential.py", "mooncake.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py", "environment.py"),
    "reactive_cost.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "simulator.py", "environment.py", "credit.py", "eval/oracle.py", "gnn.py"),
    "hazard_prior_rescaled.json": ("traces.py", "sequential.py", "mooncake.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py"),
    "hazard_prior_tier_rescaled.json": ("traces.py", "sequential.py", "mooncake.py", "credit.py", "eval/oracle.py", "gnn.py", "simulator.py", "environment.py"),
    "prefix_features.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "simulator.py"),
    "prefix_features_tier.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "simulator.py", "environment.py"),
    "unreachability_law.json": ("traces.py", "mooncake.py"),
    "online_admission.json": ("traces.py", "sequential.py", "mooncake.py", "causal.py", "simulator.py", "environment.py", "credit.py", "eval/oracle.py", "gnn.py"),
    "reachable_by_length.json": ("traces.py", "sequential.py", "mooncake.py", "simulator.py"),
    "equal_horizon.json": ("traces.py", "mooncake.py", "causal.py"),
    "admission_filter.json": ("traces.py", "eval/online.py", "mooncake.py"),
    "reachable_ceiling.json": ("traces.py", "sequential.py", "mooncake.py", "simulator.py"),
    "headroom_by_length.json": ("traces.py", "sequential.py", "mooncake.py"),
}
