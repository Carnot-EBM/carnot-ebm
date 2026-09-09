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

## experiment_7150_v628_grounding_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The grounding preflight was blocked because `real_qwen_canary` failed with `canary_cuda_layers_missing` and `canary_cuda_placement_unconfirmed`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7151_v629_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V629 contract preflight completed but was disqualified because the active YAML contains 5 of the 14 expected tasks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7152_v629_source_delta.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated mid-JSON; actual values for `"verdict_class"`, `"honest_verdict"`, and `"gate_check_summary"` cannot be found, although their descriptions are present in `"field_principles"`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the omitted verdict declared the task blocked and, if so, which check failed and what value it observed.

## experiment_7153_v629_grounding_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The specified local Qwen model successfully completed a CUDA-backed runtime canary, establishing runtime readiness without claiming verifier quality.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7154_v629_qwen_dual_side_grounding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen dual-side pilot was incomplete and did not run to completion.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7156_v630_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V630 task contract is disqualified because the Markdown/YAML contract mismatches, including a Markdown task count of 14 where `"expected_value"` is 13.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7157_v630_qwen38_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runtime-readiness task was blocked because no idle RTX 3090 was available.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7158_v630_entity_evidence_fixture.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports `"counterfactual_fixture_ready_score": 1`, but the artifact ends before any headline verdict or claim is recorded.

## WHAT IS MISSING
The artifact is truncated mid-`"text_sha256"` inside `"entity_evidence_rows"`; the remainder containing any verdict/status, `"gate_check_summary"`, comparative metrics, or per-unit decision rows cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked claim, and—if so—does it include the per-unit metrics or failed-check diagnostic needed to verify it?
