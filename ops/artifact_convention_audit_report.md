# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7153_v629_grounding_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The specified local Qwen model successfully ran a CUDA-backed canary and returned `RUNTIME_OK`, establishing runtime readiness without claiming comparative verifier performance.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7154_v629_qwen_dual_side_grounding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen dual-side pilot is partial and incomplete because `experiment_complete` was observed as `false`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7156_v630_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V630 task contract is disqualified because the Markdown contains 14 tasks while `"expected_task_count"` is 13.

## WHAT IS MISSING
nothing; `"gate_check_summary"` names `"markdown_task_count"` and records `"expected_value": 13` and `"observed_value": 14`, while `"markdown_task_rows"` provides the task-level rows.

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

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The counterfactual fixture readiness score is 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7159_v631_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V631 Markdown/YAML contract is disqualified because Markdown specifies 14 tasks while the active YAML contains only 7.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `failed_check` as `yaml_task_count`, with `expected_value` 14 and `observed_value` 7, and the discrepancy is detailed in `markdown_task_rows`, `observed_id_order`, and `rows`.

## THE CHECK A READER CANNOT DO
none

## experiment_7160_v631_qwen38_lease_diagnosis.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen3.8 runtime preflight was blocked because no idle RTX 3090 was available: conflicting process PID 233772 occupied both GPUs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7161_qwen38_bounded_structured_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `qwen38_runtime_preflight_ready_score` was 0 rather than the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
