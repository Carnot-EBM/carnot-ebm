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

## experiment_9137_v778_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are complete, while two static and seven online bounds remain unmeasured and no independent verifier benefit or ARC solve is claimed.

## WHAT IS MISSING
The remainder of "confirmed_branch_results": the supplied artifact ends mid-"observed" value inside "gate_check_summary", before the remaining outcomes are visible; "completed_count": 14 records only their total.

## THE CHECK A READER CANNOT DO
Do the omitted branch outcomes identify a failed check and observed value wherever they report being blocked?

No rerun or external verification performed; truncation prevents a complete audit.

## experiment_9138_v779_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The fourteen-task administrative contract is ready, with no measured scientific benefit claimed.

## WHAT IS MISSING
The artifact truncates inside "full_task_objects". Although "contract_ready_score" is 1, actual values for "honest_verdict", "verdict_class", "gate_check_summary", and "rows" are unseen; their mentions in "field_principles" provide no results.

## THE CHECK A READER CANNOT DO
Does the terminal verdict report a blocker without identifying the failed check and observed value?

The omitted remainder could not be checked; truncation prevents a definitive classification.

## experiment_9139_v779_validation_schedule.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports administrative validation with 32 completed units and deferred full qualification, while disclaiming scientific benefit.

## WHAT IS MISSING
The artifact cuts off mid-entry in "gate_check_summary"; the ending containing any terminal verdict and per-unit outcomes is unavailable.

## THE CHECK A READER CANNOT DO
What final verdict does the artifact reach, and what recorded checks support it?

No rerun performed; the omitted remainder could change the classification.

## experiment_9140_v779_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Forward qualification is not ready: "acceptance_gates" records incomplete execution, coverage, schedule projection, and unsuccessful normal-child checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

No rerun or external verification; this assessment covers the supplied text only.

## experiment_9141_v779_human_contrast_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The excerpt reports 90.33% annotator agreement on 100 observed QIDs, but its headline verdict is not visible.

## WHAT IS MISSING
The artifact’s remainder: it ends mid-key at `"sibling_"` inside `"agreement_rows"`. `"agreement_intervals"` and `"acceptance_gates"` are present, but the final outcome and any blocker diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder declare a blocked outcome without identifying which check failed and what value it observed?

## experiment_9142_logit_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9142 was blocked because upstream `forward_validation_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9150_v779_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an administrative availability inventory only, with no measured performance or cost benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Execution was not checked. The supplied text ends mid-row, so this assessment covers only the visible artifact.

## experiment_9151_v779_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are complete; scientific predictions and complete costs remain unmeasured.

## WHAT IS MISSING
The supplied text truncates inside "confirmed_branch_results[0].gate_check_summary". "branch_readiness" lists blocked tasks, but their diagnostic entries are not visible.

## THE CHECK A READER CANNOT DO
Which check blocked "exp9142-logit-runtime", and what value did it observe? The omitted remainder is unchecked and may contain that diagnostic.
