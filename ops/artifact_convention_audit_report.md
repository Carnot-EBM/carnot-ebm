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

## experiment_9068_v774_dispatch_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact’s remainder: JSON cuts off inside "authority_manifest". No headline claim or overall verdict is visible. The visible "arc_frontier.checks" records a diagnostic: "optional_arc_support" failed with "observed": 0.

## THE CHECK A READER CANNOT DO
Does the complete artifact assert a comparative result without per-unit metrics?

The omitted remainder was not checked; it could change this assessment.

## experiment_9069_v774_source_custody.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

no claim

## WHAT IS MISSING

The remainder of the artifact: it cuts off mid-path inside `"b_attempt_inventory.files"`. No verdict or `"gate_check_summary"` is visible. `"a_custody_ready_score": 0` and `"complete_no_attempt_authority": false` alone do not establish that the task was blocked.

## THE CHECK A READER CANNOT DO

Does the missing remainder report a comparative claim or a blocked verdict with a diagnostic?

Only the supplied excerpt was assessed; omitted content could change the classification.

## experiment_9070_v774_adjudication_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Human-target construction is blocked by `"base_question_family_authority"`; `"construction_evidence.missing"` identifies the absent author mapping from source-version/question/template IDs to common base families.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: external files or reruns; assessment covers only the supplied text.

## experiment_9071_collect_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Collection was blocked because `exp9069-source-custody.b_collection_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9073_human_features.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because both upstream readiness gates, "human_source_ready_score" and "human_methods_ready_score", recorded 0 when each required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9079_v774_hardware_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The hardware assessment is blocked: "gate_check_summary" records "current_cost" failing with "observed": "missing_file"; no actual-device speedup is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: experiment execution or referenced files; this verdict covers the supplied text only.

## experiment_9080_arc_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9080 was blocked because upstream `arc_new_outcomes_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9081_v774_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Fourteen administrative outcomes were completed, without claiming measured scientific benefit or authorizing publication.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External references were not checked; this assessment covers only the supplied text.
