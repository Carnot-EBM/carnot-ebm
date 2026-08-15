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

## experiment_6436_v554_terminal_handoff_and_queue_preflight.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted field(s) ['gate_check_summary'] do not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The artifact claims completion is blocked because the V554 queue is incomplete.

## WHAT IS MISSING
A `gate_check_summary` or equivalent V554 diagnostic identifying the failed queue check and its observed value; `"status": "complete_blocked_v554_queue_incomplete"` is present, but only V553 artifact details and blockers are recorded.

## THE CHECK A READER CANNOT DO
Which V554 queue item or completeness check failed, and what actual value caused the blocked verdict?

## experiment_6437_generation_to_verdict_receipt_replay_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `v554_queue_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6440_held_factor_revocation_binding_shift_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream Exp6439 artifact was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6442_skill_misevolution_quarantine_rollback_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the Exp6441 upstream artifact required to check `prospective_csl_ready_score == 1.0` was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger was added and implemented.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1767_e2e_qwen.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 1736 achieved simulated Vivado success and generated a bitfile.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6444_csl_lifecycle_recomputation_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V554 CSL lifecycle recomputation audit is blocked by missing or failed upstream evidence.

## WHAT IS MISSING
nothing; `"blocked_reason"`, `"csl_ineligibility_reasons"`, and `"gate_check_summary.failed_checks"` identify the failed checks and observed evidence.

## THE CHECK A READER CANNOT DO
none
