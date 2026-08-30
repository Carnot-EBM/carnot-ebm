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

## experiment_6768_targetable_proof_panel_expansion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that a 126-row, cold-replayed exact-invalid fixture is ready, explicitly not that one model or arm outperformed another.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6769_environment_indexed_proof_grammar_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The environment-indexed grammar kept valid SAT and UNSAT support reachable with zero ghost violations, and every readiness gate passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6770_dccd_environment_grammar_ab_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
the headline claim cannot be determined because the artifact ends mid-record

## WHAT IS MISSING
Actual values for `"status"`, `"honest_verdict"`, `"verdict_class"`, `"gate_check_summary"`, and `"rows"`; these names appear only in `"field_principles"` before the artifact is truncated.

## THE CHECK A READER CANNOT DO
Did the experiment make a comparative claim supported by per-unit outcome rows, or was it blocked with a recorded failed check and observed value?

## experiment_6771_proof_transport_localization_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `proof_transport_ab_completed` gate observed `false` but required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6773_csl_owned_lease_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live preflight was blocked because no RTX 3090 met the required minimum of 22,610 MB free VRAM.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6776_arc_shadow_supervisor_accrual.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked at preflight because exclusive GPU access without unrelated compute was unavailable, so no live model was invoked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6777_arc_tool_gap_transport.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `shadow_supervisor_transport_ready` was `false` while the gate expected `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6780_v590_branch_disposition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims V590 produced fixture and grammar readiness, while comparative proof, repair, continuous-memory, and ARC evidence remained blocked or missing, with no branch metrics pooled.

## WHAT IS MISSING
nothing; blocked outcomes identify failed checks and observed values in `"gate_check_summary"`, while comparative fields record `"no_eligible_comparative_rows"` with `"pair_count": 0`.

## THE CHECK A READER CANNOT DO
none
