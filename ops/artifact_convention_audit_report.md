# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8362_v721_threshold_guard.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact’s ending, including any headline claim or final verdict. It stops mid-value in "bound_derivation.cells" at "second_derivative_exact". "acceptance_gates", "assumptions", and per-cell bounds are present; "assumptions" records the missing authenticated rounding/monotonicity contract.

## THE CHECK A READER CANNOT DO
Does the completed artifact make a comparative claim beyond the per-cell bounds shown?

## experiment_8363_atomic_table_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Publication was blocked because upstream `guard_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8368_v721_typed_runtime_closure.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be identified because the supplied artifact is truncated.

## WHAT IS MISSING
The remainder of the artifact, including its final verdict and any associated evidence or blocker diagnostic. It ends mid-string inside "cited_upstream_artifacts"; "acceptance_gates" and per-unit "change_evidence" are present.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim, report a diagnosed block, or make no claim?

## experiment_8369_changed_runtime_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen canary was blocked because `runtime_changed_score` and `cuda_context_ready_score` were both 0, while each gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8370_v721_arc_outcome_delta.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports reader readiness with zero outcome support and zero current game/model calls.

## WHAT IS MISSING
The artifact is truncated inside "field_principles.exposure_scope". The current artifact’s final verdict and any "gate_check_summary" are unavailable. "acceptance_gates" and empty "arm_support_rows" are visible; the recorded "verdict_class" check concerns an upstream artifact.

## THE CHECK A READER CANNOT DO
Does the missing remainder declare a comparative result or a blocked verdict requiring supporting evidence?

## experiment_8371_v721_hardware_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
V721 hardware acquisition is deferred, with "acceptance_gates" explicitly recording failed compatible-kernel, complete-transport, owned-check and scientific-benefit checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8372_v721_gatemate_missing_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The reopening obligation was recorded, but historical authentication and hardware execution remain blocked by missing exact source bytes and physical evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8373_v721_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Spline34’s reported mean cost gain over RBF34 is −0.00390625 across 128 intended units.

## WHAT IS MISSING
The complete "paired_cost_rows" array: slot 100 is truncated and slots 101–128 are absent, despite "intended_count": 128 and "changed_decision_count": 3.

## THE CHECK A READER CANNOT DO
Do all 128 paired cost differences reproduce the −0.00390625 "mean_gain" reported in "bootstrap_summary.all_intended"?
