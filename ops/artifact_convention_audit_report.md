# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6553_prospective_sota_continuous_self_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the GPU VRAM contract failed, so the mandated models were not loaded and learning outcomes were not evaluated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6554_continuous_self_learning_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit is blocked by missing live evidence and failed preconditions, so no scientific conclusion is allowed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6555_proof_preserving_constraint_saturation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The 144-unit constraint-saturation fixture achieved a ready score of 1.0, with all proof, attack-matrix, and gate checks passing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6556_sota_constraint_saturation_intervention_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The bounded exact fallback beats flat and longer-flat controls while causing no harmful regressions and accounting for charged cost.

## WHAT IS MISSING
The complete `"per_unit_rows"` array: `"saved_row_count"` reports 540 rows, but the artifact is truncated mid-row and does not provide all model/unit/arm outcomes needed to check the claim.

## THE CHECK A READER CANNOT DO
Across all 540 rows, did the fallback arms broadly outperform both controls with no harmful regressions, rather than only doing so in the displayed subset?

## experiment_6557_constraint_saturation_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the required `prior_failures` entry naming `exp6556-sota-constraint-saturation-intervention-ab` was missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6558_arc_live_redirect_ledger_reachability.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live receipt path was reachable, seven firings were inspected across six runs, and the per-arm evidence supported no policy-order change.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6559_gatemate_changed_state_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no operator-authored GateMate physical-state receipt newer than Exp6525 was found, so zero hardware commands were run.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6560_v567_independent_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The artifact claims Exp6557’s independent audit was blocked because a gate check failed.

## WHAT IS MISSING
For Exp6557, the failed check identifier and its observed value are missing; `"honest_verdict": "blocked_gate_check_failed"`, `"status": "blocked"`, and `"blocked_gate_artifact": true` are present, but no `"gate_check_summary"` or equivalent diagnostic appears.

## THE CHECK A READER CANNOT DO
Which gate check failed for Exp6557, and what value did it observe?
