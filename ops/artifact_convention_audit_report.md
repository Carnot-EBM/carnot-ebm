# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8632_complete_source_comparison.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `streaming_extractor_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8633_v741_installed_lean_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The installed prototype met its mechanical readiness gate, including compatibility checks on 128 responses.

## WHAT IS MISSING
The supplied artifact cuts off inside "historical_model_provenance". "installed_prototype_ready_score" is 1, but actual "prototype_parity_rows" and "rows" are only named in "field_principles"; their contents are unavailable, so aggregate-only reporting cannot be established.

## THE CHECK A READER CANNOT DO
Did each of the 128 installed responses match its original-worker counterpart?

## experiment_8634_v741_installed_compatibility.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

The visible portion reports failed “exact_node_union” and “installed_compatibility” gates, but the headline verdict is unavailable.

## WHAT IS MISSING

The artifact cuts off inside “historical_model_provenance”, before the current “honest_verdict”, “verdict_class”, and remaining results. “acceptance_gates” identifies failed checks, and “historical_8609.gate_check_summary” supplies a historical blocker diagnostic.

## THE CHECK A READER CANNOT DO

Does the current invocation’s final verdict have supporting per-unit results or a recorded blocker diagnostic?

## experiment_8635_installed_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `installed_measurement_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8636_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `runtime_repair_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8637_v741_arc_observer_transfer.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be identified from the truncated artifact.

## WHAT IS MISSING
The artifact ends inside "actual_redirect_rows"; the remaining content and any headline claim or final verdict are unavailable. "acceptance_gates" records pass flags.

## THE CHECK A READER CANNOT DO
Does the artifact’s headline conclusion follow from its per-unit results?

## experiment_8638_v741_hardware_operation_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware admission remains gated because compatible physical execution and a new complete-cost boundary have not been established.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8639_v741_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone reports administrative completion while substantive acceptance gates remain unmet, with blocked tasks explained by failed readiness checks or missing producer artifacts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
