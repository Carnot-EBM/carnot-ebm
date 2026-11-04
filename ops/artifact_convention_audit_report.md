# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8806_v754_consensus_challenge_energy.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Learned energy improves Brier error and false accepts but fails the coverage-loss gate, yielding a completed null.

## WHAT IS MISSING
The artifact ends mid-entry in "per_candidate_rows". Visible candidate rows cover only one "unit_id"; complete paired "metric", "covered", and "false_accept" rows across the 120 worlds in "independent_count" cannot be established. "brier_rows" and "coverage_rows" provide pooled metrics per seed.

## THE CHECK A READER CANNOT DO
Do the reported improvements occur broadly across independent worlds, or come from a small subset?

## experiment_8807_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because upstream `runtime_repair_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8808_v754_archived_observer_costs.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run completed with a null result: observer disturbance failed acceptance for the constructed workload, leaving component costs unready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8809_v754_delayed_feedback_capacity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
"acceptance_gates" marks all four checks as passed, but the excerpt shows no headline outcome or final verdict.

## WHAT IS MISSING
The artifact is truncated inside "admission_rows"; the final verdict and any subsequent per-unit outcome metrics or blocker diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Do the reported gate passes have supporting per-unit measurements in the complete artifact?

## experiment_8810_v754_causal_learning_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The completed audit reports no qualifying causal learning benefit, with the future-Brier gate failing.

## WHAT IS MISSING
Actual per-stream Brier metrics or paired differences. "coverage_rows" and "familywise_intervals" provide aggregates; "stream_comparison_rows" and "rows" appear only as descriptions in "field_principles". The supplied artifact truncates inside "retention_checks", so their presence later cannot be determined.

## THE CHECK A READER CANNOT DO
Is the combined arm’s reported Brier disadvantage spread across the 30 streams or driven by a few outliers?

## experiment_8811_new_arc_outcome_analysis.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `arc_new_outcomes_available_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8812_learning_operation_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `exp8808-archived-observer-costs.component_cost_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8813_v754_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports 13 completed administrative slots and passing acceptance gates, but the final headline verdict is unavailable.

## WHAT IS MISSING
The artifact ends mid-string inside "full_task_objects". Actual values for "honest_verdict", "verdict_class", "rows", and "gate_check_summary" are unavailable; their names in "field_principles" do not supply those values.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit outcomes?
