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

## experiment_7518_source_pilot.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the pre-gate layer because the upstream check on `exp7517-source-protocol.source_protocol_ready_score == 1` failed with an observed value of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7525_v658_decision_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The decision audit is blocked from evaluating claims because required upstream science inputs are missing or incomplete.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7526_v658_arc_eligibility.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null where eligibility was observed in shadow mode without applied effect support, failing benefit gates due to zero applied mutations.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7527_v658_arc_opportunities.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked before model execution because the `owned_gpu` precondition check failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7528_v658_service_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run is blocked at the service boundary due to a missing count-memory module prototype (`python/carnot/experiment_7523_v658_count_memory.py`), with no comparative performance claims made.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7530_b2_induction_gate_telemetry.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7529_v658_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because required upstream v658 science was absent or externally gated (`honest_verdict`: "complete_blocked_required_v658_science_absent_or_externally_gated"), making no comparative claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7531_b2_induction_gate_measurement.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible in the truncated artifact

## WHAT IS MISSING
The artifact ends mid-`episode_rows` object, so any final headline/verdict, gate result, and diagnostic fields are missing; visible fields include `"selection_role"`, `"episode_rows"`, `"disposition"`, and `"error"`.

## THE CHECK A READER CANNOT DO
Did the complete artifact ultimately claim that a comparative gate passed or that execution was blocked, and what evidence or diagnostic supported that verdict?
