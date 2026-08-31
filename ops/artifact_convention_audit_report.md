# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 3 |

## experiment_6794_temporal_exchange_cold_hardware_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The cold replay supports the source null result across sampler arms, while showing denominator-dependent performance differences.

## WHAT IS MISSING
The per-seed, per-arm comparison `"rows"` underlying `"cold_recomputed_metrics"` and `"denominator_sensitivity.comparisons"`; `"field_principles.rows"` and `"preconditions_checked"` mention 360 rows, but the actual rows and their metric values are not present.

## THE CHECK A READER CANNOT DO
Were the reported null and denominator-dependent differences broad across the 20 seeds in each stratum, or driven by outliers, degenerate controls, or units with no headroom?

## experiment_6795_v592_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V592 has durable-checkpoint and positive grouped fixed-point evidence, while dispatch and cold CSL causality remain incomplete and temporal exchange is null.

## WHAT IS MISSING
Per-unit paired rows containing each seed/topology’s arm values and exact-valid outcome are missing; `"rows"` contains only task inventories and aggregate `"branch_claim"` records, while `"paired_exact_valid_delta"`, `"exact_valid_rate_by_arm"`, and `"held_topology_exact_valid_delta"` report summaries only. The blocked checks are adequately identified in `"gate_check_summary"` with `"failed_check"`, `"expected"`, and `"observed"`.

## THE CHECK A READER CANNOT DO
Was the grouped fixed-point advantage broad across the 320 paired units, or driven by a few outliers, degenerate controls, or units with no headroom?

## experiment_6796_agent_model_dispatch_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The dispatch requalification was blocked because `research-roadmap-next.yaml` and `scripts/agent_model_compatibility.py` were missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6797_canonical_transaction_byte_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The replay completed successfully, with all 3,189 commits retaining verified canonical parent and new-state byte snapshots.

## WHAT IS MISSING
nothing; `"gate_check_summary.checks"` records each check’s `"expected"`, `"observed"`, and `"passed"` values, while `"failed_checks"` and `"failures"` are empty.

## THE CHECK A READER CANNOT DO
none

## experiment_6798_csl_causal_safety_byte_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The compositional-online arm beat the frozen controller on held-future utility across all five orders, with a positive bootstrap confidence interval, and the causal audit completed successfully.

## WHAT IS MISSING
nothing; per-unit values are present in `"online_minus_frozen_order_effects"`, `"held_future_utility_by_arm_order"`, `"hard_case_harm_by_arm_order"`, and `"retention_by_arm_order"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6799_model_output_formal_constraint_probes.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The replay and dual-encoding checks succeeded, with expected and observed results matching per case and no translation disagreements.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6801_real_output_fixed_point_cold_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The grouped fixed-point arm has near-zero exact-valid-rate effects versus the flat recurrent control, including a positive restructuring point estimate of 0.0015432099 whose confidence interval crosses zero.

## WHAT IS MISSING
Per-source-case or paired-seed arm metrics for `refinement` and `restructuring`; `clustered_confidence_intervals` contains only aggregate estimates, while `exact_recomputed_metrics.metrics_by_transformation_model_family.base.by_case` provides case-level detail only for `base` in the artifact shown.

## THE CHECK A READER CANNOT DO
Is the positive restructuring estimate spread across many source cases, or caused by one outlier while most paired units are identical or pinned at zero?
