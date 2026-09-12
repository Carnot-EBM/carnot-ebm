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

## experiment_7238_v637_mention_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked without running because the upstream structured-quarantine check found `flagged_adversarial|quarantined|fabricated` to be true instead of the required false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7239_v637_semantic_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The semantic audit was blocked without running because upstream experiment 7237 had `flagged_adversarial|quarantined|fabricated` observed as `true` when `false` was required.

## WHAT IS MISSING
nothing; `gate_check_summary` records `failed_check`, `artifact_field`, `expected_value`, `observed_value`, and `upstream`.

## THE CHECK A READER CANNOT DO
none

## experiment_7240_v637_recurrence_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The recurrence fixture, controller, and positive controls are runnable.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7241_v637_recurrence_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Validated archive reuse did not pass every frozen learning gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7242_v637_recurrence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The recurrence audit completed, but the promotion criteria did not all pass.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7243_v637_native_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native arm achieved exact parity but failed every performance gate, with a reported lower-CI speedup of 0.3632×.

## WHAT IS MISSING
nothing; `"cost_rows"` records per-unit `"arm"`, `"block"`, `"metric"`, and `"total_event_ns"`, while `"acceptance_gate_results"` records each gate’s `"actual"`, `"expected"`, and `"pass"` values.

## THE CHECK A READER CANNOT DO
none

## experiment_7244_v637_board_disposition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three board dispositions are recorded: KV260 and PolarFire retain authenticated graduations, while GateMate remains blocked because no operator-authored physical-state receipt after Exp6559 exists.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7245_v637_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The capstone claims the thirteen-task matrix is complete, but at least one recomputed claim does not match and several component results are null, disqualified, or prerequisite-blocked.

## WHAT IS MISSING
The artifact is truncated inside `"evidence_matrix"` at `"observed_sha256": "sha256:04ba7b2566939d0`; the remaining evidence rows, any final `"honest_verdict"`, and any per-unit metric rows supporting comparative claims such as the native path being `"slower"` cannot be found.

## THE CHECK A READER CANNOT DO
Do the unseen per-unit measurements actually support the claim that the native-memory path was slower, rather than that conclusion being based only on aggregate values?
