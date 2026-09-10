# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7156_v630_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V630 contract is disqualified because the Markdown/YAML task contract mismatches the expected 13-task contract.

## WHAT IS MISSING
nothing; `"gate_check_summary"` names `"markdown_task_count"` with `"expected_value": 13` and `"observed_value": 14`, while `"markdown_task_rows"` and `"task_contract_rows"` provide per-task evidence.

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

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The counterfactual fixture is ready, reported as `"counterfactual_fixture_ready_score": 1`.

## WHAT IS MISSING
Per-fixture readiness metric or pass/fail rows keyed by `"fixture_id"`; `"entity_evidence_rows"` contains span provenance, while `"calibration_row_count"` and `"counterfactual_fixture_ready_score"` are aggregates.

## THE CHECK A READER CANNOT DO
Did readiness hold across the fixtures, or was the score driven by a small number of nondegenerate cases while others failed or had no headroom?

## experiment_7159_v631_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V631 contract is disqualified because the active YAML contains 7 tasks instead of the 14 required by the Markdown contract.

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
nothing; `"honest_verdict"`, `"gate_check_summary"`, `"preconditions_checked"`, and per-GPU `"gpu_process_rows"` record the failed check and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7161_qwen38_bounded_structured_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because the upstream `qwen38_runtime_preflight_ready_score` was 0 but the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7166_v632_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V632 contract is disqualified because the YAML contains only 4 of the 13 expected tasks, mismatching the Markdown contract.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "yaml_task_count"` with `"expected_value": 13` and `"observed_value": 4`, while `"rows"` records task-level presence and parity checks.

## THE CHECK A READER CANNOT DO
none

## experiment_7167_v632_claim_evidence_trace_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because no idle, task-ownable RTX 3090 was available.

## WHAT IS MISSING
nothing; `"gate_check_summary"` and `"preconditions_checked"` identify the failed `"idle_task_ownable_rtx_3090"` check, its expected state, and the conflicting PID, GPU UUIDs, and memory usage observed.

## THE CHECK A READER CANNOT DO
none
