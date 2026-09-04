# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_6962_queue_regulated_self_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the Qwen3.6 no-memory worker failed while loading its GGUF model file.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_check"` identifies `"all_model_arm_workers"`, and `"gate_check_summary.observed_value"` records the model-loading exception.

## THE CHECK A READER CANNOT DO
none

## experiment_6963_queue_memory_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit was blocked because `queue_learning_run_complete_score` was 0 when the gate required it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6964_v609_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V609 capstone is complete but disqualified because of flagged or conflicting evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6965_v610_contract_advisory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic contract audit completed and found v6.10 contract defects.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6966_gguf_load_envelope_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because foreign GPU compute processes were occupying the two GPUs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6967_certified_error_headroom_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The chronological event stream is ready (`"chronological_event_stream_ready_score": 1`), and the sampled proposal failures are non-equivalent.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6968_arc_post_refit_induction_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The induction audit was blocked because the `immutable_transition_source` precondition failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6969_error_structured_prompt_bank.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6966-gguf-load-envelope-canary.gguf_runtime_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
