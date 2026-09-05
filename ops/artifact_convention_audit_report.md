# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7025_belief_shadow_live_trace.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live belief-shadow trace was blocked because `live_trace_execution` raised `ValueError: invalid ARC evaluation provenance: model_filename must be one GGUF filename`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7026_held_mechanic_belief_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7025-belief-shadow-live-trace.belief_shadow_trace_ready_score` was 0 rather than the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7027_v615_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V615 capstone is blocked because required live evidence is absent.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records `"failed_check": "required_live_upstream_complete"` with exact `"expected_value"` and `"observed_value"`, while `"scientific_gap_summary"` records the live-trace error and downstream gate failure.

## THE CHECK A READER CANNOT DO
none

## experiment_7028_v616_active_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V616 task contract is disqualified because only 5 of the expected 10 roadmap tasks were observed.

## WHAT IS MISSING
nothing; `"rows"` provides per-task observations, and `"gate_check_summary"` identifies `"failed_check": "observed_task_count"`, `"expected_value": 10`, and `"observed_value": 5`.

## THE CHECK A READER CANNOT DO
none

## experiment_7029_v616_sota_scope_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V616 scope audit completed successfully and found no verified post-marker scientific improvement requiring a ledger or task-contract change.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7030_arc_gguf_model_identity_bridge.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC model identity bridge is ready because its positive, negative, legacy-compatibility, producer/consumer wiring, hash-join, and hub/revision checks all passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7031_arc_model_identity_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC model-identity cold audit completed positively and met its readiness gate because all recorded identity, mutation, regression, isolation, reachability, and command checks passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7032_repaired_belief_shadow_live_trace.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live belief-shadow trace was blocked because `live_trace_execution` raised a `ValueError` when `observed_server_model_path` was an alias rather than the required canonical path.

## WHAT IS MISSING
nothing; `honest_verdict`, `verdict_class`, and `gate_check_summary.failed_check`, `gate_check_summary.expected_value`, and `gate_check_summary.observed_value` identify the blocker and observed error.

## THE CHECK A READER CANNOT DO
none
