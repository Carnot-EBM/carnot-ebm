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

## experiment_9141_v779_human_contrast_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports 90.33% annotator agreement across 100 observed QIDs.

## WHAT IS MISSING
The artifact ends mid-record at `"sibling_"`; complete `"agreement_rows"` and any final verdict or blocker diagnostics are unavailable, although `"agreement_intervals.ALL"` reports 271 agreements out of 300.

## THE CHECK A READER CANNOT DO
Do the complete per-QID rows reproduce the reported 271/300 agreements?

Unchecked: truncated remainder; risk: unseen claims or diagnostics could change the classification.

## experiment_9142_logit_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9142 was blocked because upstream `forward_validation_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9150_v779_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an administrative availability inventory only; unavailable evidence is explained in "cost_rows" by "censor_reason": "external_inputs_unavailable" and in "clock_boundary_map" by "reason": "no qualified complete service clock".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9151_v779_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are complete, while scientific predictions and complete costs remain unmeasured.

## WHAT IS MISSING
The artifact ends mid-entry in "confirmed_branch_results". Several "branch_readiness" entries say "blocked" with "readiness": {}, but the visible "gate_check_summary" describes unblocked exp9138; the blocked branches’ diagnostics are not visible.

## THE CHECK A READER CANNOT DO
Which check blocked exp9142-logit-runtime, and what value did it observe?

The truncated remainder could contain that diagnostic; it could not be checked.

## experiment_9152_v780_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The fourteen-task administrative contract is ready; scientific benefit is explicitly unmeasured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Referenced files and reruns were not checked; this verdict covers only the supplied administrative evidence.

## experiment_9153_v780_validation_schedule.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Administrative validation is recorded, but full qualification remains deferred because the retained slow subprocess’s cause is unknown.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External files and execution were not checked; this assessment covers the supplied excerpt.

## experiment_9154_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `schedule_ready_score` was 0, failing the required `== 1` gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9165_v780_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are complete, while measured validation remains blocked and scientific predictions and complete costs remain unmeasured.

## WHAT IS MISSING
The artifact ends mid-field inside "confirmed_branch_results". "branch_readiness" includes blocked verdicts and "schedule_ready_score": 0, but the supplied portion shows "gate_check_summary" and "failed_gate_value" only for the null contract branch; remaining blocker diagnostics cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the omitted remainder identify the failed checks and observed values for the blocked branches?

External artifacts were not checked; truncation prevents a definitive audit.
