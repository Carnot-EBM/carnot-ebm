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

## experiment_7080_v620_three_family_entrance_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-row before any final verdict or headline statement

## WHAT IS MISSING
The remainder of the artifact, including any final `"verdict"`, `"status"`, `"headline_claim"`, or `"gate_check_summary"`; present fields inspected include `"entrance_proposal_bank_complete_score"`, `"cleanup_rows"`, and `"exact_label_rows"`.

## THE CHECK A READER CANNOT DO
Did the experiment ultimately claim success, report a comparison, or declare itself blocked—and, if blocked, which check failed at what value?

## experiment_7084_v621_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V621 contract is disqualified because the active YAML contains 7 tasks instead of the expected 12.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7085_v621_chat_transport_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three local GGUF model families passed the bounded chat-transport gate.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records each check’s expected and observed values, while `"exact_label_rows"`, `"chat_template_rows"`, and `"finish_reason_rows"` provide per-unit evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_7086_v621_three_family_entrance_bank.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims `"all_models_real": true` and records per-unit, per-seed causal-witness results.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7091_v622_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V622 contract is disqualified because the active YAML contains 6 tasks instead of the 12 required by the Markdown contract.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "yaml_task_count"`, `"expected_value": 12`, and `"observed_value": 6`, while `"rows"` records per-task presence and outcomes.

## THE CHECK A READER CANNOT DO
none

## experiment_7092_v622_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V622 literature ingestion completed successfully with full source coverage and no newly promoted experiment hooks or reference delta.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7093_v622_entrance_bank_sufficiency_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The recorded budget-parity and causal-witness replay checks passed for the listed per-model and per-raw-key units.

## WHAT IS MISSING
nothing; `"budget_parity_rows"` and `"causal_witness_replay_rows"` provide unit-level fields including `"model_id"`, `"raw_key"`, `"entrance_id"`, `"replayed_causal_witness"`, and `"passed"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7094_matched_hardness_entrance_diagnostic.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream `entrance_support_audit_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
