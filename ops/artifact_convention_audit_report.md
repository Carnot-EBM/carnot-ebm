# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 11 |
| CANNOT_DETERMINE | 1 |

## experiment_6433_csl_row_recomputation_safety_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed as a null result: the prospective CSL claim remains ineligible due to `DURATION_TOO_SHORT` on `exp6432`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6434_arc_state_key_reachability_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
the artifact body itself; no fields are present, including `gate_check_summary` or per-unit rows

## THE CHECK A READER CANNOT DO
A reader cannot tell whether the artifact made a comparative claim, reported a blocker, or recorded enough data to check either.

## experiment_6435_v553_adversarial_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the V553 capstone is complete but blocked/gated: 10 of 11 upstream tasks completed, exp6434 is missing, and only the public factor claim is eligible while other claim classes are blocked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6440_inducer_h2h_qwen38_vs_gemma31b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims no valid head-to-head comparison was made because only the qwen27b arm completed while gemma31b/qwen38_27b did not.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6436_v554_terminal_handoff_and_queue_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
V553 narrow factor evidence is eligible, while other claim families remain blocked by flagged, underpowered, missing, or unauthenticated evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6437_generation_to_verdict_receipt_replay_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because 1 of 1 gate failed: `v554_queue_ready_score` was `0.0` but expected `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6440_held_factor_revocation_binding_shift_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because 1 of 1 pre-gate failed: the upstream artifact for `exp6439-factor-clause-influence-ab` was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6442_skill_misevolution_quarantine_rollback_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the pre-gate check failed: the upstream artifact was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the CERCE ledger was added/implemented and the task completed.

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
The artifact claims experiment 1736 succeeded with `honest_verdict` of `vivado_simulated_success`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6444_csl_lifecycle_recomputation_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit is blocked because required upstream evidence is missing or failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
