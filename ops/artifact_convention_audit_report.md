# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

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
The experiment was disqualified because the common-arm configuration gate failed.

## WHAT IS MISSING
nothing; `"honest_verdict"`, `"gate_check_summary.failed_check"`, and each check’s `"expected_value"`, `"observed_value"`, and `"passed"` are present.

## THE CHECK A READER CANNOT DO
none

## experiment_7145_v627_rust_multiscale_sampler.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before execution because prerequisite toolchain and input checks had not begun.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7146_v627_gatemate_changed_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked before hardware access because `receipt_newer_than_exp6559` expected `1.0` but observed `0.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7147_v627_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V627 evidence matrix is structurally complete but blocked by upstream terminal availability.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7148_v628_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V628 contract preflight was blocked because `research-roadmap-next.yaml` was missing and therefore failed the `v628_yaml_readable` prerequisite.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7149_v628_source_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because local prerequisites were not checked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7150_v628_grounding_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The grounding preflight was blocked because the real Qwen canary lacked confirmed CUDA layers and CUDA placement.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records `"failed_check": "real_qwen_canary"` and `"observed_value": ["canary_cuda_layers_missing", "canary_cuda_placement_unconfirmed"]`.

## THE CHECK A READER CANNOT DO
none
