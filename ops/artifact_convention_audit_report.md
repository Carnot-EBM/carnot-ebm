# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8498_v730_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims historical FoVer publication gates passed, but the capstone’s headline verdict is not visible.

## WHAT IS MISSING
The artifact ends mid-"historical_model_provenance"; values for "honest_verdict", "scientific_claims", and "rows" are not supplied, although those names appear in "field_principles". "gate_check_summary" does record failed checks and observed values, so missing blocker diagnostics is not supported.

## THE CHECK A READER CANNOT DO
Does the complete artifact supply per-unit metrics supporting its comparative headline claim?

## experiment_8499_v731_contract_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports current-contract readiness of 0, but the artifact is truncated before its headline verdict.

## WHAT IS MISSING
The actual "honest_verdict" or "verdict_class" value; both appear only as names in "field_principles" before truncation mid-"future_dependencies". "contract_rows" does record diagnostics: "independent_authority": false and "missing_reason": "authority_absent_or_mismatched".

## THE CHECK A READER CANNOT DO
Does the missing headline verdict assert a comparative result requiring additional per-unit evidence?

## experiment_8500_strongest_baseline_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because upstream `methods_ready_score` was 0, failing the gate requiring 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8503_objective_publication_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream `methods_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8506_append_import_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because upstream `methods_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8509_v731_arc_redirect_effect_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed with all 32 events right-censored, so scientific benefit is not estimable and arm refinement lacks sufficient support.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8510_v731_board_cost_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
“NPU_qualified” and “TSU_qualified” are false, with failed “acceptance_gates” supported by recorded missing-clock, compatibility, and missing-field diagnostics.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8511_v731_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports administrative completion while current authority and independent scientific generalization remain unmet.

## WHAT IS MISSING
The artifact cuts off inside "historical_model_provenance"; actual "honest_verdict", "scientific_claims", and "rows" values are not shown, although "field_principles" names them. "gate_check_summary" does record failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the omitted portion contain per-unit metrics supporting any comparative headline claim?
