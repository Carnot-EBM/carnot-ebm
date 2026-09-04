# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6957_smt_mapping_certification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The enumeration and Z3 authorities agree for each recorded attempt.

## WHAT IS MISSING
nothing; per-attempt `authority_agreement_rows` include `attempt_key`, `authorities_agree`, `enumeration_status`, `enumeration_label`, `z3_status`, and `z3_label`.

## THE CHECK A READER CANNOT DO
none

## experiment_6958_convex_factor_energy_canary.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The input-convex factor sum did not outperform either learned control: both comparisons report `"mean_delta": 0.0`, with `"strictly_above_zero": false`.

## WHAT IS MISSING
Per-unit metric or delta rows for the 45 units summarized by `"paired_unit_count"`; `"confidence_interval_rows"` contains only aggregate `"mean_delta"` and CI values, while `"calibration_rows"` and `"baseline_rows"` provide only per-seed aggregates.

## THE CHECK A READER CANNOT DO
Were all 45 paired units ties, or did wins, losses, outliers, and ceiling effects cancel to a zero mean?

## experiment_6959_certified_energy_selection.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6960_certified_selection_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Convex-factor energy did not outperform syntax validity: the paired mean delta was -0.0370 with a 95% interval of [-0.0926, 0.0], yielding a null verdict.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6961_certified_event_sequence.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims the certified event sequence is ready, via `"certified_event_sequence_ready_score": 1`.

## WHAT IS MISSING
The complete `"chronology_rows"` field: the artifact terminates inside ordinal 26, so complete rows for ordinals 26–71—including `"eligible_prior_certificate_ids"` and `"prohibited_future_certificate_ids"`—cannot be found.

## THE CHECK A READER CANNOT DO
Do all 72 events permit only prior certificates while prohibiting their current and every future certificate?

## experiment_6962_queue_regulated_self_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `all_model_arm_workers` check failed when the Qwen no-memory worker could not load its GGUF model file.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6963_queue_memory_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `queue_learning_run_complete_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6964_v609_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V609 capstone is complete but disqualified because it contains flagged or conflicting evidence.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "flagged_or_conflicting_evidence"` and lists `"disqualified_task_numbers": [6958, 6964]`, while `"adversarial_verify_rows"` records the critical flag for task 6958 and `"gate_replay_rows"` records the failed upstream gate for task 6963.

## THE CHECK A READER CANNOT DO
none
