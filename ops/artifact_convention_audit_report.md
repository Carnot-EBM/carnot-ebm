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

## experiment_9123_v777_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are complete, with conditional exposed-development evidence but no independent human-benefit or generalization claim.

## WHAT IS MISSING
The supplied artifact cuts off inside the first `"confirmed_branch_results"` entry’s `"gate_check_summary"`; remaining branch results, verdicts, and any per-unit metric rows are unavailable.

## THE CHECK A READER CANNOT DO
Does the reported FoVer `"learning_contribution"` have per-unit evidence supporting its positive aggregate interval?

Only the supplied fragment was assessed; truncation prevents determining whether those rows are recorded.

## experiment_9124_v778_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied fragment.

## WHAT IS MISSING
The artifact truncates inside "authority_snapshots", leaving its final claim and verdict unavailable; "activation_authorized": false alone does not establish that the task was blocked.

## THE CHECK A READER CANNOT DO
Does the complete artifact report a comparative claim or a blocked verdict with a diagnostic?

Unchecked: the unavailable remainder; it may contain the evidence needed to classify this artifact.

## experiment_9125_v778_receipt_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“claim_scope” limits the claim to administrative receipt qualification, but the final qualification verdict is not visible.

## WHAT IS MISSING
The final verdict: the supplied artifact ends mid-string inside “execution_partition_manifests”; “acceptance_gates” shows “complete_bounded_validation”: false but does not establish the final outcome.

## THE CHECK A READER CANNOT DO
Was receipt qualification ultimately accepted or blocked?

The truncated remainder could not be checked and may contain the answer.

## experiment_9126_v778_logit_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All nine synthetic method controls passed, supporting methods readiness without claiming human advantage.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: referenced files or reruns; this assessment covers only the supplied text.

## experiment_9127_logit_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Exp9127 was blocked because upstream exp9125’s "receipt_ready_score" was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9132_v778_online_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Frozen delayed logit methods passed qualification checks and exact CPU-fixture recovery; human benefit remains unmeasured.

## WHAT IS MISSING
The supplied artifact ends mid-hash inside `"gate_check_summary"`. `"fixture_metrics"` contains arm-level summaries, and `"exact_recovery_rows"` supplies recovery comparisons, but actual `"rows"` and terminal `"honest_verdict"`/`"verdict_class"` values are unavailable; those names appear in `"field_principles"`.

## THE CHECK A READER CANNOT DO
Was the positive-fixture advantage of adaptive spline over frozen spline spread across observations or driven by an outlier?

Only the supplied fragment was checked; the omitted remainder may contain the necessary rows.

## experiment_9136_v778_complete_cost.json

**CHECKABLE**

## VERDICT

CHECKABLE

## WHAT THE CLAIM IS

Administrative availability inventory only; no comparative scientific benefit is claimed, and `"censor_reason": "external_inputs_unavailable"` records the blocker.

## WHAT IS MISSING

nothing

## THE CHECK A READER CANNOT DO

none

Referenced files and execution were not checked; scientific performance remains unmeasured.

## experiment_9137_v778_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports fourteen administrative outcomes, with scientific bounds unmeasured and no independent verifier benefit or new ARC solve claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External artifacts and reruns were not checked; this verdict covers only the supplied text.
