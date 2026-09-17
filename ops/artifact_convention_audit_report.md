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

## experiment_7361_v646_fresh_plan_capture.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capture met its scientific-value gate by producing 128 publicly faithful proposals, while promotion remained unauthorized.

## WHAT IS MISSING
Per-unit proposal-validity results for each evaluation call supporting `"scientific_value":{"observed":128,"passed":true}`; `"call_manifest.evaluation_calls"` contains call IDs, hashes, and runtime receipts, but no per-call metric or faithful-proposal count.

## THE CHECK A READER CANNOT DO
Did the 128 qualifying proposals arise broadly across evaluation units, or from only a few duplicated or unusually productive calls?

## experiment_7362_v646_prospective_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed but was disqualified because required validation failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7363_learning_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 2 of 3 upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7364_v646_acquisition_adjudication.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The acquisition adjudication is null because the scientific cost gate failed: CI95 upper 0.9652615826317683 was not below 0.90.

## WHAT IS MISSING
The complete `"paired_cost_rows"` array is missing: the artifact truncates mid-row despite `"independent_reduction.complete_cost_ratio_ci95.context_clusters": 30`; `"gate_check_summary.failed_checks"` supplies the failed check and observed value.

## THE CHECK A READER CANNOT DO
Do all 30 per-context cost rows reproduce the reported CI95 upper bound of 0.9652615826317683?

## experiment_7365_v646_supervisor_support.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed audit found insufficient supported outcomes, so the support, scientific-value, and promotion gates were not met.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7366_supervisor_live.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `supervisor_trial_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7367_v646_board_disposition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no qualifying operator-authored GateMate physical-state change receipt existed after Exp6559, while all three board dispositions were recorded and readiness, value, and promotion remained zero.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7368_v646_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All twelve V646 dispositions were accounted for, but advancement was disqualified because Exp7362 failed required validation and the independent learning audit was pre-gated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
