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

## experiment_6615_v576_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone claims that live projection had zero effect and prospective self-learning had zero benefit over both controls, while decoding remained blocked.

## WHAT IS MISSING
Actual per-game or per-pair measurement rows containing unit ID, arm, and metric value; the present `"per_unit_rows"` contains task/comparative-group summaries, while `"arm_summaries"`, `"row_count_by_arm"`, `"held_future_benefit_over_shuffled"`, and `"held_future_benefit_over_static"` are aggregates, and `"row_hash_replay"` reports zero checked rows for live projection.

## THE CHECK A READER CANNOT DO
Were the reported zero comparative effects consistent across units, or produced by offsetting wins and losses, degenerate controls, or units pinned at floors or ceilings?

## experiment_6616_v577_execution_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims execution is blocked because the expected Exp6616–Exp6628 YAML contracts are not all present.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` identifies `"exact_task_set"` and `"model_policy"` failures with expected and observed values, including the missing task IDs.

## THE CHECK A READER CANNOT DO
none

## experiment_6617_gpu_lease_phase_receipts.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `execution_contract_ready_score` was `0.0`, failing the required equality check against `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6619_v578_activation_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The activation contract is not ready because required roadmap contract checks failed.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` records the failed checks with `"expected"` and `"observed"` values, while `"document_yaml_diff.matches"` identifies the mismatched contract categories.

## THE CHECK A READER CANNOT DO
none

## experiment_6620_gpu_lease_phase_receipts.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream `activation_contract_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6622_qwen36_direct_headroom.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the required upstream artifact `exp6621-mandated-model-admission` was not found, so `qwen_admission_ready_score == 1.0` could not be verified.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6624_delayed_two_level_decoding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6623-headroom-support-reducer` was not found, so its `constrained_decoding_ready_score` could not be checked against `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6627_spectral_integrity_repair.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `activation_contract_ready_score` was `0.0`, failing the required equality to `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
