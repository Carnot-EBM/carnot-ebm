# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7141_v627_csl_event_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the CSL event stream is ready, reporting `"csl_event_stream_ready_score": 1` for 108 events.

## WHAT IS MISSING
nothing; `"event_rows"`, `"chronological_order_rows"`, `"constraint_family_rows"`, and `"event_count"` provide unit-level supporting records, and no blocked verdict is present.

## THE CHECK A READER CANNOT DO
none

## experiment_7139_v627_symbolic_grounding_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the native `llama-server` binary was not CUDA-enabled.

## WHAT IS MISSING
nothing; `gate_check_summary` records `"failed_check": "native_llama_server"`, the expected `"cuda_build": true`, and the observed `"cuda_build": false`.

## THE CHECK A READER CANNOT DO
none

## experiment_7142_v627_flowbalance_memory_csl.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because initialization checks had not started.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7143_flowbalance_memory_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `flowbalance_memory_csl_complete_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7144_v627_rebudgeted_arc_loo.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified because the common-arm configuration gate failed.

## WHAT IS MISSING
nothing; `"honest_verdict"` records `"disqualified_common_arm_configuration"`, and `"gate_check_summary"` identifies `"failed_check": "common_arm_configuration"` with `"expected_value": true` and `"observed_value": false`.

## THE CHECK A READER CANNOT DO
none

## experiment_7145_v627_rust_multiscale_sampler.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before execution because toolchain and input precondition checks had not started.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7146_v627_gatemate_changed_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The hardware run was blocked because no valid receipt newer than experiment 6559 was available.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7147_v627_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V627 evidence matrix is structurally complete but scientifically blocked by unavailable upstream terminal evidence.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `failed_check` as `upstream_terminal_availability` and records missing artifacts 7140 and 7143 plus the `gatemate_operator_receipt` external-state block.

## THE CHECK A READER CANNOT DO
none
