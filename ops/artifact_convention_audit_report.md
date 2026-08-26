# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_6620_gpu_lease_phase_receipts.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream `activation_contract_ready_score` was `0.0`, failing the required equality to `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6622_qwen36_direct_headroom.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact for `exp6621-mandated-model-admission.qwen_admission_ready_score` was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6624_delayed_two_level_decoding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream artifact `exp6623-headroom-support-reducer` was missing, so `constrained_decoding_ready_score == 1.0` could not be verified.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6627_spectral_integrity_repair.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `activation_contract_ready_score` was `0.0`, failing the requirement that it equal `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6633_gpu_lease_phase_journal.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GPU lease scheduler is not ready because the focused-tests gate was observed as false, so the artifact is blocked and makes no model-quality claim.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `check: "focused_tests"`, `expected: true`, and `observed: false`, while `tests_run` records the individual commands, exit codes, and summaries.

## THE CHECK A READER CANNOT DO
none

## experiment_6634_mandated_model_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `gpu_lease_scheduler_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6636_delayed_two_level_decoding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6635-matched-direct-headroom` was not found, so its `headroom_ready_score == 1.0` gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6605_qwen36_direct_headroom.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Each recorded attack failed closed, with candidate integrity-readiness scores matching the expected score of 0.0.

## WHAT IS MISSING
nothing; `"attack_rows"` provides per-attack `"candidate_integrity_ready_score"`, `"expected_integrity_ready_score"`, `"failed_checks"`, and `"failed_closed"` values.

## THE CHECK A READER CANNOT DO
none
