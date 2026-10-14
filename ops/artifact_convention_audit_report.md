# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_8518_objective_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8518 was blocked because the upstream `objective_stream_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8519_v732_append_import_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The append-import runner passed historical-replay qualification, with no new benchmark measurements, model calls, or claimed performance benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8520_v732_append_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Append beats snapshot on complete-service time by 2.16× at batch 1 and 1.59× at batch 8, conditional on this same-host fixture.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8521_bounded_qwen_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8521 was blocked because `runtime_repair_ready_score` and `comparator_inputs_ready_score` both observed 0 when each required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8522_v732_arc_effect_receipt_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Private effect-receipt qualification completed with action parity across nine cases, documented in "action_parity_rows", and no scientific benefit claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8523_v732_arc_effect_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The panel completed as a null result, claiming no scientific benefit and granting no arm-refinement authorization.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8524_v732_board_cost_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt records failed NPU and TSU qualification, but truncation prevents identifying the artifact’s full headline claim.

## WHAT IS MISSING
The remainder of the artifact: it ends mid-string in "board_operation_rows" → "primitive_reference" → "path"; visible "acceptance_gates" already records failed compatibility and clock checks.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative claim whose metric lacks corresponding per-unit rows?

## experiment_8525_v732_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The preserved V726 audit reports a “circular_positive” Brier gain of 0.01969866346324697 for spline9 over logistic9.

## WHAT IS MISSING
Paired evaluation rows containing unit identifiers and each arm’s Brier loss underlying "brier_gain_vs_logistic" and "brier_gain_ci95"; "A1_metrics", "raw_operands", and "calibration" contain pooled metrics, totals, and bin summaries.

## THE CHECK A READER CANNOT DO
Does the reported Brier gain occur across most evaluation units, or come from a few unusually large improvements?
