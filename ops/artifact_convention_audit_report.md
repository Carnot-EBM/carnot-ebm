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

## experiment_7076_v620_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V620 contract preflight was disqualified because Markdown contract parsing failed on a malformed task row.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7077_v620_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V620 SOTA ingestion completed successfully and met its gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7078_v620_gpu_lease_migration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GPU lease compatibility gate passed all checks, with both real legacy journals safely migrated and all synthetic safety cases producing their expected outcomes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7079_v620_gpu_lease_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GPU lease cold audit passed every gate and is ready, while making no model-quality claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7080_v620_three_family_entrance_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-row before any headline verdict or summary

## WHAT IS MISSING
The remainder of the artifact, including any top-level verdict and `gate_check_summary`; fields present include `entrance_proposal_bank_complete_score`, `exact_label_rows`, `cleanup_rows`, and `checkpoint_rows`.

## THE CHECK A READER CANNOT DO
Did the experiment ultimately declare a gate met or the task blocked, and what recorded evidence supported that verdict?

## experiment_7084_v621_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V621 contract is disqualified because the active YAML contains 7 tasks instead of the 12 required by the Markdown contract.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "yaml_task_count"`, `"expected_value": 12`, and `"observed_value": 7`, while `"rows"` provides per-task evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_7085_v621_chat_transport_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three local GGUF model families passed the bounded chat-transport gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7086_v621_three_family_entrance_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no complete headline claim is visible because the artifact is truncated mid-row

## WHAT IS MISSING
The remainder of the artifact, including any headline/verdict or gate fields; the visible `"all_models_real"` and `"causal_witness_rows"` fields do not state a comparative or blocked verdict.

## THE CHECK A READER CANNOT DO
Does the omitted verdict make a comparative or blocked claim, and if so, do the omitted fields provide the required per-unit metrics or failed-check diagnostic?
