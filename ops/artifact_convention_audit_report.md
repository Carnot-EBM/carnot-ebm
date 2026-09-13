# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_7266_semantic_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7265-mention-heldout.mention_capture_complete_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7267_v639_recognition_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The active recognition fixture is ready, while held-out learning value was not scored.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7268_v639_recognition_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Autonomous recognition completed, but one or more frozen value gates failed, yielding a null verdict.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7269_v639_recognition_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Several comparative acceptance gates passed or failed, including error and false-accept comparisons against reset, random, shuffle, and frozen arms.

## WHAT IS MISSING
The artifact is truncated mid-field at `"retained_co"`, so it is impossible to determine whether later fields contain per-unit rows for the aggregate metrics in `"future_error_vs_reset"`, `"recurrence_degradation_vs_frozen"`, `"recurrence_error_vs_random"`, and `"recurrence_error_vs_shuffle"`; the visible `"bound_rows"` provide per-unit `"false_accept_rate"` but not those other metrics.

## THE CHECK A READER CANNOT DO
Do the recurrence-error comparisons hold broadly across streams, or are their reported aggregate bounds driven by a few outlier streams?

## experiment_7270_v639_durable_profile.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Durable synchronization costs and insufficient replaceable snapshot work make the proposed delta-log prototype unwarranted.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7271_delta_log.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7270-durable-profile.journal_optimization_warranted_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7272_v639_board_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Three authenticated board dispositions are complete: KV260 FPGA and PolarFire CPU graduations remain preserved, while GateMate is blocked because the required post-Exp6559 operator receipt is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7273_v639_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V639 capstone is blocked because the required semantic audit is unavailable, with only 5 of 7 same-milestone gates passing and the independent Markdown/YAML contract mismatching.

## WHAT IS MISSING
nothing; the failed checks and observed values are recorded in `acceptance_gate_results`, while task-level outcomes appear in `branch_decisions`, per-task evidence in `evidence_matrix`, and contract mismatches in `contract_rows`.

## THE CHECK A READER CANNOT DO
none
