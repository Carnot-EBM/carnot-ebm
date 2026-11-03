# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8768_v751_evidence_set_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The evidence_set arm has lower macro error than both comparators across the recorded evaluation worlds, but readiness is disqualified because scoped_validation is false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8769_evidence_set_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `evidence_set_methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8770_v751_optimized_cost_attribution.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible gates report complete optimized diagnostic execution but failure of the low-observer-overhead gate.

## WHAT IS MISSING
The artifact cuts off inside "component_cost_rows"; the final conclusion and observer-overhead measurements are unavailable. "acceptance_gates" identifies the failed check, and per-process rows are present.

## THE CHECK A READER CANNOT DO
What observed observer overhead supports the failed "low_observer_overhead" gate?

## experiment_8771_v751_lifecycle_archive_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" reports all five gates as passed.

## WHAT IS MISSING
The artifact cuts off inside "cited_upstream_artifacts"; gate criteria, observed check values, per-unit measurements, and the final verdict are not visible.

## THE CHECK A READER CANNOT DO
Do the underlying measurements justify all five passed gates in "acceptance_gates"?

## experiment_8772_v751_ordinary_durable_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The “bounded_durable_benefit” acceptance gate was not met, with “bounded_durable_benefit_score” equal to 0.

## WHAT IS MISSING
The artifact ends mid-“sha256” inside “cited_upstream_artifacts”; the remaining JSON is missing, so any verdict, blocker diagnostic, or per-unit measurements cannot be assessed.

## THE CHECK A READER CANNOT DO
Does the complete artifact record measurements explaining why “bounded_durable_benefit” is false?

## experiment_8773_arc_outcome_frontier.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `arc_new_outcomes_available_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8774_v751_hardware_operation_decision.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims board authority and owned validation passed, but the terminal verdict is not visible.

## WHAT IS MISSING
The artifact cuts off inside `"authority.rows[3].task_object.prompt"`. Actual result fields `"honest_verdict"` and `"verdict_class"` and any subsequent evidence are unavailable; `"acceptance_gates"` and partial `"authority.rows"` are present.

## THE CHECK A READER CANNOT DO
Does the terminal result claim success or blockage, and does its supporting evidence justify that verdict?

## experiment_8775_v751_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible summary withholds activation and reports no supported policy change, but the capstone’s final headline is unavailable.

## WHAT IS MISSING
The remainder of the artifact: it cuts off inside "code_config_hashes", before any final verdict or later per-unit results. "acceptance_gates" and "branch_gate_values" already identify failed checks and their values, so BLOCKED_WITHOUT_DIAGNOSTIC is not established.

## THE CHECK A READER CANNOT DO
Does the capstone’s final verdict assert a comparative improvement supported by per-unit measurements?
