# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

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
The V628 contract preflight was blocked because `research-roadmap-next.yaml` was missing and therefore unreadable.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7149_v628_source_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V628 source-and-cache delta run was blocked because its local prerequisites were not checked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7150_v628_grounding_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The grounding preflight was blocked because `real_qwen_canary` failed with `canary_cuda_layers_missing` and `canary_cuda_placement_unconfirmed`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7151_v629_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V629 contract preflight was disqualified because the active YAML contained 5 tasks instead of the 14 expected from the Markdown contract.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7152_v629_source_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7153_v629_grounding_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The local Qwen model successfully loaded on CUDA and produced the expected `RUNTIME_OK` canary, establishing runtime readiness without claiming verifier performance.

## WHAT IS MISSING
nothing; `gate_check_summary`, `canary_raw_output_rows`, `runtime_evidence_rows`, and `gpu_rows` record the observed evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_7154_v629_qwen_dual_side_grounding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen dual-side pilot was only partial and did not run because `experiment_complete` was false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
