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

## experiment_9153_v780_validation_schedule.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Full qualification is deferred because the retained slow subprocess’s cause remains unknown, explicitly recorded in "blocked_prerequisite" and "computational_root_cause".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not rerun; assessment is limited to the supplied text.

## experiment_9154_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `schedule_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9165_v780_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are recorded; scientific results remain unmeasured, and measured validation is blocked with `"schedule_ready_score": 0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Only the supplied text was assessed; it ends mid-entry, so omitted content remains unchecked.

## experiment_9166_v781_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact reports fourteen completed administrative tasks and contract readiness (`"contract_ready_score": 1`), without claiming measured scientific benefit.

## WHAT IS MISSING
The artifact cuts off inside `"full_task_objects"` → `"previous_outcome_rows"` → `"gate_check_summary"`. Current-run `"honest_verdict"`, `"verdict_class"`, `"gate_check_summary"`, and `"rows"` values are not visible; names listed in `"field_principles"` are not evidence.

## THE CHECK A READER CANNOT DO
Does the current run declare a blocked verdict with a recorded reason, or complete administrative readiness?

The missing tail was not checked and could change the classification.

## experiment_9167_v781_causal_validation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Corrective validation remains incomplete: “before_after_receipts” records a corrected-run timeout, and “unfinished_module_nodes” lists outstanding checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Assessment uses only the supplied artifact; no rerun or external verification.

## experiment_9168_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `causal_fix_ready_score` was 0, failing the required `== 1` check recorded in "gates_evaluated" and "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9178_v781_arc_mechanic_routing.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible acceptance targets are reference-equivalent routing for 52 recognized conditions and unchanged routing for 72 legacy controls.

## WHAT IS MISSING
The artifact cuts off after "consumer_reachability", before any final verdict or gate results. "acceptance_gates" supplies conditions, "completed_count" reports 124 completions, and "changed_behavior_rows" supplies 22 comparisons, leaving the remaining conditions undocumented in the visible excerpt.

## THE CHECK A READER CANNOT DO
Did all 52 recognized conditions match their references and all 72 legacy controls remain unchanged?

Execution and external files were not checked; the truncated artifact leaves the final outcome unknown.

## experiment_9179_v781_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports 124/124 routing equalities and successful FoVer reproduction, while withholding activation.

## WHAT IS MISSING
The supplied artifact ends mid-entry in "arc_summary.rows"; remaining evidence and any diagnostic explaining "required_checks_passed": false are unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder identify the failed check and its observed value?

The truncated remainder was not checked; its absence prevents establishing either failure category.
