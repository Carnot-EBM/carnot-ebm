# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7156_v630_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V630 contract is disqualified because the Markdown task count was 14 instead of the expected 13.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7157_v630_qwen38_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runtime canary was blocked because no idle RTX 3090 was available.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `check: "idle_rtx_3090"`, `expected_value.minimum_count: 1`, `observed_value.count: 0`, `observed_value.indices: []`, and `passed: false`, corroborated by `preconditions_checked`.

## THE CHECK A READER CANNOT DO
none

## experiment_7158_v630_entity_evidence_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The counterfactual fixture is ready, with `"counterfactual_fixture_ready_score": 1`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7159_v631_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V631 contract is disqualified because Markdown specifies 14 tasks while the active YAML contains only 7.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7160_v631_qwen38_lease_diagnosis.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen3.8 runtime preflight was blocked because no idle RTX 3090 was available, with conflicting process PID 233772 occupying both GPUs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7161_qwen38_bounded_structured_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `qwen38_runtime_preflight_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7166_v632_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V632 Markdown/YAML task contract is disqualified because Markdown specifies 13 tasks while YAML contains only the first 4.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "yaml_task_count"` with `"expected_value": 13` and `"observed_value": 4`, while `"task_contract_rows"` records each task-level result.

## THE CHECK A READER CANNOT DO
none

## experiment_7167_v632_claim_evidence_trace_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The generation run was blocked because no idle, task-ownable RTX 3090 was available.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"idle_task_ownable_rtx_3090"` as failed, records `"passed": false`, and provides the expected and observed resource states, including conflicting PID 233772.

## THE CHECK A READER CANNOT DO
none
