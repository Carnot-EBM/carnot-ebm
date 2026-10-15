# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_8531_causal_brier_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `online_adapter_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8533_v733_nine_input_native_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports that finite parity and complete-request gates passed, while owned validation failed.

## WHAT IS MISSING
The artifact cuts off inside "historical_model_provenance"; "honest_verdict", "verdict_class", and actual "rows" are named in "field_principles" but their values are unavailable. "finite_parity_rows" contains only the count 1180.

## THE CHECK A READER CANNOT DO
Do per-unit results support "acceptance_gates.finite_parity": true across all 1,180 comparisons?

## experiment_8534_nine_input_inference_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `nine_input_native_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8535_bounded_qwen_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8535 was blocked because `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8536_v733_arc_capability_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Action parity passes across the recorded case/mask conditions; no scientific benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8537_v733_arc_capability_panel.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied fragment.

## WHAT IS MISSING
The completed artifact: it ends mid-entry at `"game"` inside `"capability_exclusion_rows"`. `"acceptance_gates"` and per-episode `"action_costs"` are present, but no headline verdict is visible. `"capability_reason": "runtime_explicitly_disabled"` supplies a diagnostic for the exclusions shown.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit outcome metrics?

## experiment_8538_v733_nine_input_board_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Nine-input board deployment and hardware acquisition are gated by missing qualified costs and compatible installed operations.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8539_v733_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The historical five-input host service achieved a qualified pooled speedup of approximately 2.163×.

## WHAT IS MISSING
Paired per-input baseline and service timings supporting "pooled_speedup", "primary_p95_ratio", and "primary_speedup_ci95". "V732_history.rows" contains task dispositions, not measurement rows; "imported_fields" lists field names without their data.

## THE CHECK A READER CANNOT DO
Did the speedup occur across all five inputs, or was the pooled improvement driven by one outlier?
