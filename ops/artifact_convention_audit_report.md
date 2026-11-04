# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| CANNOT_DETERMINE | 2 |

## experiment_8789_v753_branch_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim is not visible because the supplied artifact is truncated.

## WHAT IS MISSING
The artifact cuts off inside "full_task_objects". Actual values for "honest_verdict", "verdict_class", "gate_check_summary", and "rows" are not visible; those names appear in "field_principles". "acceptance_gates" does identify failed checks, so missing blocker diagnostics cannot be inferred.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit measurements?

## experiment_8790_delayed_memory_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because upstream `memory_protocol_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8791_v753_evidence_intervention_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The method-qualification run completed but was disqualified, with "gate_check_summary" recording "decision_audit" and "scoped_validation" as false against expected true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8794_evidence_intervention_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `evidence_interventions_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8795_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both prerequisite readiness gates observed 0 when each required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8796_v753_observer_intervention.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8798_arc_new_outcome_decision.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because upstream `arc_new_outcomes_available_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8800_v753_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "G2.detail" claims independent reproduction of FoVer AUROC and positive learning contribution, but the overall artifact’s headline is not visible.

## WHAT IS MISSING
The artifact’s remainder: it cuts off mid-path inside "cited_upstream_artifacts". The overall verdict, any blocker diagnostics, and any per-unit metric rows are unavailable; "G2.detail" provides only summary numbers.

## THE CHECK A READER CANNOT DO
Does the reported positive learning contribution hold across individual seeds or conditions, or come from one outlier?
