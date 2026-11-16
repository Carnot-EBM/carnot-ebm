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

## experiment_9107_v776_qwen_judge.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The run completed 450 frozen-Qwen judgments but reports disqualification through `"honest_verdict": "complete_disqualified_artifact_checks"`.

## WHAT IS MISSING
The failed artifact-check entry and its observed value: visible `"gate_check_summary"` entries all pass, while `"judge_ready_score"` is 0; the supplied JSON cuts off inside `"qid_rows"`.

## THE CHECK A READER CANNOT DO
Which artifact check caused disqualification, and what value did it observe?

The truncated remainder was not checked; it may contain that diagnostic.

## experiment_9109_v776_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes were completed, retaining descriptive evidence without claiming independent factuality, learned-head performance, or causal benefit.

## WHAT IS MISSING
The artifact’s remainder: it cuts off inside the first `"confirmed_branch_results"` entry’s `"gate_check_summary"`, despite `"completed_count": 14`; the remaining branch verdicts and evidence are unavailable.

## THE CHECK A READER CANNOT DO
Do any remaining branches report blocked outcomes without a failed-check diagnostic?

No rerun or external records checked; the truncated artifact limits this verdict.

## experiment_9110_v777_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied fragment.

## WHAT IS MISSING
The remainder of the artifact: it ends mid-path inside "authority_snapshots". "acceptance_gates" and "activation_authorized" are present, but no final verdict or "gate_check_summary" is shown.

## THE CHECK A READER CANNOT DO
Does the artifact ultimately report a blocked verdict and identify its cause?

Omitted content was not checked; truncation prevents a reliable classification.

## experiment_9111_v777_forward_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The administrative audit is disqualified from readiness: “owned_tests” and “direct_consumers” timed out after 600 seconds with exit code −9.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Assessment limited to the supplied text; referenced files were not checked.

## experiment_9112_masked_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9112 was blocked because upstream `forward_evidence_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9113_energy_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9113 was blocked because upstream `forward_evidence_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9122_v777_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims administrative inventory readiness only; complete services and adaptation remain unmeasured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: execution or external files. The recorded "censor_reason" identifies missing authenticated runtime/head service artifacts; no comparative performance claim is made.

## experiment_9123_v777_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible `"claim_scope"` reports fourteen administrative outcomes and conditional exposed-development evidence, without independent human benefit or generalization.

## WHAT IS MISSING
The artifact is truncated inside the first `"confirmed_branch_results"` entry’s `"gate_check_summary"`; the remaining outcomes, metrics, verdicts, and diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Do the remaining outcomes contain comparative claims without per-unit rows or blocked verdicts without reasons?

The omitted tail was not checked; it could contain the evidence needed to resolve this audit.
