# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 3 |

## experiment_6788_soft_fixed_point_structural_control_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible `"decision_gates"` claim that grouped fixed-point beat the flat recurrent control and passed the positive-effect gate.

## WHAT IS MISSING
The artifact is truncated inside `"frozen_manifest.paired_keys"`; although `"field_principles"` promises `"rows"` and `"honest_verdict"`, their actual contents are unavailable, so per-unit arm metrics cannot be inspected.

## THE CHECK A READER CANNOT DO
Were improvements broad across unit-seed pairs, or driven by a few outliers or units pinned at metric bounds?

## experiment_6789_soft_fixed_point_cold_authority_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The cold destructive audit supports the V592 claim that the grouped fixed-point arm outperformed the flat recurrent control under matched budgets and passed the destructive-control gates.

## WHAT IS MISSING
nothing; the artifact includes per-unit evidence in `"rows"` with `"paired_key"`, `"arm"`, `"exact_valid_rate"`, and exact outcomes, plus diagnostic gate details in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6790_chronological_constraint_routing_stream.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The constraint-routing stream passed all readiness gates, with exhaustive routing outperforming the frozen policy by 0.35 accuracy in every order.

## WHAT IS MISSING
The supplied artifact is truncated before showing the actual `"rows"`, `"gate_check_summary"`, `"honest_verdict"`, `"status"`, and `"verdict_class"` values; these names appear only in `"field_principles"`, so their presence and contents cannot be verified.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the reported 0.35 accuracy gaps were broad across event-order rows or driven by a small number of outliers.

## experiment_6791_compositional_online_constraint_routing_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The `"honest_verdict"` claims compositional online routing beat both the frozen controller and matched placebo without preregistered harm.

## WHAT IS MISSING
The complete `"rows"` array: per-row fields such as `"route_utility"`, `"arm"`, `"order_id"`, and `"pair_key"` are present, but the artifact cuts off mid-row and does not expose all 4,800 rows asserted by `"all_planned_rows_present"`.

## THE CHECK A READER CANNOT DO
Can all five `"online_minus_frozen_order_effects"`, all five `"online_minus_placebo_order_effects"`, and the 0.34375 lower confidence bound be recomputed from the complete paired unit-level results?

## experiment_6792_csl_causal_safety_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The causal safety audit was blocked because the `transaction_byte_snapshots` check failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6793_temporal_exchange_ising_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Temporal exchange did not pass both the paired-efficiency and target-law gates in the CPU simulation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6794_temporal_exchange_cold_hardware_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The cold replay supports the source null comparison between temporal exchange and ordinary Gibbs, while showing that any favorable efficiency result depends on the denominator used.

## WHAT IS MISSING
The per-seed, per-arm metric rows referenced as `"rows"` are missing; `"cold_recomputed_metrics"` and `"denominator_sensitivity.comparisons"` contain only `mean`, `ci95`, and `n` aggregates, while `"gate_check_summary"` is present but empty.

## THE CHECK A READER CANNOT DO
Were the arm-level null and denominator-dependent differences broad across the 20 seeds, or caused by outliers, cancellations, or degenerate control rows?

## experiment_6795_v592_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V592 provides positive bounded fixed-point evidence—grouped fixed-point proposals beat the matched flat control—while dispatch and cold CSL causality remain incomplete and temporal exchange is null.

## WHAT IS MISSING
Per-unit paired rows containing each seed/topology’s arm-level metrics and deltas; `"rows"` contains only task receipts and branch summaries, while `"paired_exact_valid_delta"`, `"exact_valid_rate_by_arm"`, and `"held_topology_exact_valid_delta"` are aggregates. The blocked checks are adequately diagnosed in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Did grouped fixed-point outperform the flat control broadly across the 320 paired keys, or was the positive mean driven by a few outliers, degenerate controls, or units with no headroom?
