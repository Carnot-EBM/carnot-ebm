# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_7156_v630_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V630 task contract is disqualified because the Markdown and YAML task contracts mismatch.

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
The visible artifact reports `counterfactual_fixture_ready_score` as 1, but the headline claim cannot be determined because the artifact is truncated.

## WHAT IS MISSING
The remainder of the artifact, including any verdict or gate fields and any per-unit metric rows supporting them; the visible portion contains `"counterfactual_fixture_ready_score"`, `"energy_term_contract"`, and `"entity_evidence_rows"` but ends mid-row.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked verdict, and if so, does it record the per-unit results or failed-check diagnostic needed to verify that verdict?

## experiment_7159_v631_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V631 contract is disqualified because Markdown specifies 14 tasks while the active YAML contains only 7.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `failed_check` as `yaml_task_count` with `expected_value: 14` and `observed_value: 7`, while `markdown_task_rows`, `observed_id_order`, and `rows` provide per-task evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_7160_v631_qwen38_lease_diagnosis.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen3.8 runtime preflight was blocked because no idle RTX 3090 was available; PID 233772 conflicted on both GPUs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7161_qwen38_bounded_structured_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `qwen38_runtime_preflight_ready_score` was 0 when the gate required it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7166_v632_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V632 contract is disqualified because Markdown specifies 13 tasks while YAML contains only 4.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "yaml_task_count"` with `"expected_value": 13` and `"observed_value": 4`, and `"task_contract_rows"` records the per-task mismatches.

## THE CHECK A READER CANNOT DO
none

## experiment_7167_v632_claim_evidence_trace_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The generation task was blocked because no idle, task-ownable RTX 3090 was available; two GPUs had conflicting compute process PID 233772.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
