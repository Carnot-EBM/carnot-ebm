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

## experiment_7529_v658_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone experiment is blocked because required upstream V658 science is absent or externally gated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7531_b2_induction_gate_measurement.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible in the provided, truncated artifact

## WHAT IS MISSING
The complete artifact, including any top-level verdict/headline and `"gate_check_summary"`; the text ends mid-entry inside `"episode_rows"`, which records execution data but no visible claim.

## THE CHECK A READER CANNOT DO
Did the experiment ultimately claim that a comparative gate was met, fail it with a recorded diagnostic, or remain blocked?

## experiment_7532_v659_contract_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task is blocked from contract readiness because the v659 roadmap authorities are incomplete, repository guards failed, and terminal readers failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7533_v659_tool_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7534_v659_count_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7535_v659_native_pilot.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked prior to model execution because the `owned_gpu_available` external precondition check failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7536_fit_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because upstream dependency exp7535-native-pilot failed two required gates (`native_tool_ready_score` and `fit_capture_feasible_score` both observed as 0 instead of expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7537_eval_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream checks in exp7535-native-pilot failed on native_tool_ready_score and eval_capture_feasible_score (observed 0, expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
