# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7588_v663_evidence_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7589_v663_arc_output_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7590_evidence_pilot.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream dependency exp7588 failed its readiness gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7596_v663_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because required upstream scientific producer artifacts were missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7597_v663_arc_history_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No comparative claim; the artifact reports an honest null result (`complete_null_insufficient_history_support`) with no policy benefit claimed due to insufficient history support across games.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7598_v663_rust_consumer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The benefit gate failed because `strongest_comparator_lower95` was 0.6797261332013704, below the required value greater than 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7599_v663_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null (`complete_null_board_continuity_placement_unmeasured`): board continuity is tracked across three targets with hardware placement benefit unmeasured, GateMate blocked on an unfulfilled physical prerequisite, and no new accelerator purchase justified.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7600_v663_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Consumer parity passed, but the registered Rust-versus-Python aggregate speed gate failed.

## WHAT IS MISSING
Per-pair timing rows for the 120 independent comparisons are missing; only aggregate fields such as `"comparison_summary"`, `"independent_pair_count"`, `"paired_ratio"`, `"positive_count"`, `"negative_count"`, and percentile summaries are present.

## THE CHECK A READER CANNOT DO
Were the speed-gate results broad across individual pairs, or driven by a few extreme timing observations?
