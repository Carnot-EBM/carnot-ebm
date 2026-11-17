# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_9168_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `causal_fix_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9178_v781_arc_mechanic_routing.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The stated target is routing invariance for 52 recognized rows and unchanged behavior for 72 legacy controls.

## WHAT IS MISSING
The artifact is truncated after "consumer_reachability"; "acceptance_gates" gives conditions and "changed_behavior_rows" gives individual comparisons, but the final verdict and complete gate evidence are unavailable.

## THE CHECK A READER CANNOT DO
Did all 52 recognized rows match their references and all 72 legacy controls remain unchanged?

The omitted remainder was not assessed and could change this classification.

## experiment_9179_v781_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports successful FoVer AUROC and learning-contribution checks, plus operator-list equality on 124/124 routing cases.

## WHAT IS MISSING
The artifact ends mid-string inside "arc_summary.rows", leaving remaining evidence and any later blocker diagnostics unavailable. "G2.detail" reports aggregate FoVer intervals; visible rows measure "operator_list_equality", not FoVer performance.

## THE CHECK A READER CANNOT DO
Was the positive FoVer learning contribution consistent across individual seeds and conditions?

The omitted remainder could change this verdict; it could not be checked.

## experiment_9180_v782_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The fourteen-task administrative contract is ready (“contract_ready_score”: 1), with no measured scientific benefit claimed.

## WHAT IS MISSING
The artifact is truncated inside “full_task_objects”; current top-level “honest_verdict”, “verdict_class” and “gate_check_summary” values are not visible. Their descriptions in “field_principles” do not supply those values.

## THE CHECK A READER CANNOT DO
Did this run finish blocked, and if so, which check failed and what value did it observe?

## experiment_9181_v782_causal_validation.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The excerpt reports original-versus-changed diagnostic observations, but its final headline claim and verdict are not visible.

## WHAT IS MISSING
The artifact cuts off at `"duration_s": 1898`, leaving the final verdict and any current blocker summary unavailable; `"before_after_rows"` and `"historical_failures"` are present.

## THE CHECK A READER CANNOT DO
Was the task ultimately blocked, and which current check failed?

The truncated remainder is unassessed; it may contain the decisive diagnostics.

## experiment_9182_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `causal_fix_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9192_v782_arc_mechanic_routing.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

“acceptance_gates” requires 52 recognized rows to match the empty-game reference and 72 legacy controls to remain unchanged, but no final outcome is visible.

## WHAT IS MISSING

The artifact’s remainder: “changed_behavior_rows” ends mid-string. Visible rows contain “before”, “after”, and “reference”, so they are not aggregate-only evidence; the final verdict and gate outcomes are unavailable.

## THE CHECK A READER CANNOT DO

Did all 124 attempted conditions satisfy the stated gates?

The omitted remainder was not checked; it could contain the outcome and diagnostics.

## experiment_9193_v782_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt claims routing mismatches fell from 22 to 0, with operator-list equality passing on 124/124 units.

## WHAT IS MISSING
The complete `"arc_summary.rows"`: the supplied artifact ends mid-row at `"arm": "explicit_mecha"`, leaving insufficient rows to reconcile `"regression_result"`, `"equality_numerator"`, and `"denominator"`.

## THE CHECK A READER CANNOT DO
Do the results for all 124 units substantiate the reported reduction from 22 mismatches to zero?

The omitted remainder was not checked; it may contain the supporting evidence.
