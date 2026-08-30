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

## experiment_6770_dccd_environment_grammar_ab_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports zero exact-valid rate for all three arms, but the headline claim is not included.

## WHAT IS MISSING
The actual `"honest_verdict"`, `"status"`, `"gate_check_summary"`, and `"rows"` fields are missing from the truncated artifact; only their descriptions appear under `"field_principles"`, while aggregate fields such as `"exact_valid_rate_by_arm"` are present.

## THE CHECK A READER CANNOT DO
Does the omitted verdict make a comparative or blocked claim, and—if so—do per-unit rows or a specific failed-gate diagnostic support it?

## experiment_6771_proof_transport_localization_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream field `proof_transport_ab_completed` was `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6773_csl_owned_lease_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live preflight was blocked because no RTX 3090 met the required minimum of 22,610 MB free VRAM.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6776_arc_shadow_supervisor_accrual.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before live inference because the exclusive-GPU-without-unrelated-compute preflight check failed.

## WHAT IS MISSING
nothing; `"status"`, per-row `"failure_class"`, `"live_model_invoked"`, and `"actions_observed"` record the blocked outcome and diagnostic.

## THE CHECK A READER CANNOT DO
none

## experiment_6777_arc_tool_gap_transport.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `shadow_supervisor_transport_ready` was observed as `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6780_v590_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V590 partially narrowed the proof gap because the 126-row targetable panel and 45-row dynamic proof grammar were ready, while downstream experiments remained blocked or missing.

## WHAT IS MISSING
The per-unit `"rows"` underlying the aggregate `"panel_rows": 126`, `"grammar_rows": 45`, `"targetable_panel_ready": true`, and `"dynamic_proof_grammar_ready": true`; only aggregate `"branch_rows"` and `"gate_check_summary"` are present.

## THE CHECK A READER CANNOT DO
Did every panel and grammar condition satisfy the readiness criteria, or were the reported gates driven by duplicated, degenerate, or selectively successful rows?

## experiment_6782_sequential_sota_runtime_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because no model completed owned CUDA admission after the Qwen 3.6 runtime-admission lease wait expired.

## WHAT IS MISSING
nothing; `gate_check_summary.failed_check` identifies `"runtime_admission:unsloth/Qwen3.6-35B-A3B-GGUF"` and `gate_check_summary.observed` records `"lease_wait_deadline_expired"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6783_owned_proof_generation_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6782-sequential-sota-runtime-admission.all_mandated_runtime_ready` was `false` instead of the required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
