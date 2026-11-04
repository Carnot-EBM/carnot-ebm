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

## experiment_8778_delayed_memory_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `memory_protocol_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8779_v752_evidence_set_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
On the exposed replication set, evidence_set achieved lower macro error than both comparators, with "paired_world_rows" recording all three arms’ metrics for each of 120 worlds.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8782_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8783_v752_durable_producer_identity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment marks all eight "acceptance_gates" true, but does not reveal the headline verdict.

## WHAT IS MISSING
The remainder of the artifact: the JSON ends mid-entry in "cited_upstream_artifacts", leaving any results, per-unit rows, or blocker diagnostics unseen.

## THE CHECK A READER CANNOT DO
Does the artifact’s headline verdict have supporting measurements or a recorded blocker reason?

## experiment_8784_v752_durable_cost_components.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact’s remainder: it cuts off mid-"path" inside "cited_upstream_artifacts". No headline verdict, per-unit metrics, or blocker diagnostic is visible. "acceptance_gates" includes "component_attribution": false, but does not establish a blocked outcome.

## THE CHECK A READER CANNOT DO
Does the complete artifact claim a comparative result, report a blocked outcome, or merely record a receipt?

## experiment_8786_arc_new_outcome_decision.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `arc_new_outcomes_available_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8787_v752_hardware_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment asserts administrative contract readiness through `"authority.ready": true`.

## WHAT IS MISSING
The artifact ends mid-string inside `"authority.rows"`; its remaining results and current `"honest_verdict"` / `"verdict_class"` are unavailable.

## THE CHECK A READER CANNOT DO
Does the completed artifact report only administrative readiness, a comparative outcome, or a blocked verdict with a diagnostic?

## experiment_8788_v752_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
"G2.detail" claims independent reproduction of FoVer’s AUROC and learning contribution within published confidence intervals.

## WHAT IS MISSING
The artifact is truncated inside "cited_upstream_artifacts"; any subsequent per-unit results, overall verdict, or blocker diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Does the reported learning contribution reflect improvement across units or an effect concentrated in a few units?
