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

## experiment_8470_v728_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be established from the truncated artifact.

## WHAT IS MISSING
The JSON ends inside "historical_model_provenance"; actual values for "honest_verdict", "verdict_class", "scientific_claims", and "rows" are not visible, although "field_principles" names them. "gate_check_summary" records failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the omitted portion contain a comparative headline claim and the per-unit metric rows needed to check it?

## experiment_8471_v729_contract_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports zero current contract readiness, but the artifact cuts off before its headline verdict.

## WHAT IS MISSING
The actual "honest_verdict", "verdict_class", and any "rows" payload; "field_principles" only describes them. "gate_check_summary" supplies failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the missing headline verdict make a comparative claim supported by per-unit metrics?

## experiment_8472_source_control_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because upstream `methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8475_objective_checkpoint_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8475 was blocked because the upstream `methods_ready_score` was 0, failing the required `== 1` check recorded in `gates_evaluated` and `gate_check_summary`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8478_publication_cost_model.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp8471-contract-methods.methods_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8482_v729_arc_panel_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Panel qualification was disqualified before live collection, with all 16 episodes unstarted and failed authority checks documented in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8483_v729_board_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware qualification gates remain unmet, with diagnostic reasons recorded in "acceptance_gates", "append_cost_boundary", and "board_operation_rows".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8484_v729_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be identified from the truncated artifact.

## WHAT IS MISSING
Actual values for "honest_verdict", "verdict_class", "scientific_claims", and "rows" are unavailable; their names appear in "field_principles". The excerpt ends inside "gate_check_summary", which already records blocker diagnostics.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit metric rows?
