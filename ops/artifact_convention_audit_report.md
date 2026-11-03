# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| CANNOT_DETERMINE | 7 |

## experiment_8755_v750_verifier_family_surface.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports all eight "acceptance_gates" as true.

## WHAT IS MISSING
The artifact is truncated mid-entry in "call_cost_rows"; no final headline verdict or complete comparison results are visible. "analysis_protocol.transfer" specifies a criterion but records no outcome.

## THE CHECK A READER CANNOT DO
Does the targeted arm improve new-family detection across the 30 evaluation worlds without additional correct-verifier false rejections?

## experiment_8756_v750_feature_reuse_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Feature reuse is ready for engineering parity on the original exposed64 panel, with "feature_reuse_ready_score" equal to 1.

## WHAT IS MISSING
The artifact cuts off inside "frozen_8744_workload.requests". "feature_extraction_count_rows" is present, but actual "wire_parity_rows" and "honest_verdict" values are not visible; their names appear only in "field_principles".

## THE CHECK A READER CANNOT DO
Did reuse preserve baseline outputs for every request across both arms and all batch sizes?

## experiment_8757_v750_feature_reuse_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Feature reuse improves complete-process throughput by about 1.52× in both native and Python arms.

## WHAT IS MISSING
The artifact cuts off inside `"probe_rows"`. Complete per-repeat `"duration_s"` and `"throughput_items_s"` measurements for the baseline/reuse pairs underlying `"paired_process_intervals"` are not visible; they may appear in the omitted remainder.

## THE CHECK A READER CANNOT DO
Do the ten paired process measurements reproduce the reported throughput benefit consistently across repeats?

## experiment_8758_v750_lifecycle_event_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Ordinary/restart and constructed-threshold qualification passed, but lifecycle-event qualification was disqualified.

## WHAT IS MISSING
The supplied artifact cuts off inside "natural_event_rows". "acceptance_gates.owned_validation" is false and "gate_check_summary" is empty; the remaining qualification rows and any later validation diagnostic are unavailable.

## THE CHECK A READER CANNOT DO
Which check caused owned validation to fail, and what value did it observe?

## experiment_8759_ordinary_durable_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8759 was blocked because upstream `ordinary_window_ready_score` was 0, failing the gate requiring 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8760_v750_arc_horizon_sensitivity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports readiness scores of 1 and no identifiable comparative benefit, but the artifact is truncated.

## WHAT IS MISSING
The remaining artifact, including any final claim or blocker diagnostics: the text ends mid-value in "callback_action_join"; "per_game_per_arm_rows" and comparison "reason" fields are present.

## THE CHECK A READER CANNOT DO
Does the omitted remainder assert a comparative success or report a blocker without a diagnostic?

## experiment_8761_v750_hardware_cost_boundaries.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims administrative authority readiness through `"authority.ready": true`, but the overall verdict is unavailable.

## WHAT IS MISSING
The artifact is truncated inside `"authority.rows[3].task_object.prompt"`; top-level `"honest_verdict"`, `"verdict_class"`, and subsequent results are unavailable. `"acceptance_gates"` and `"authority.rows"` are present.

## THE CHECK A READER CANNOT DO
Does the terminal result claim success, report a null, or identify a blocker?

## experiment_8762_v750_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“G2.detail” claims independent reproduction of FoVer’s headline AUROC and positive learning contribution.

## WHAT IS MISSING
Per-unit production/control metrics and paired learning contributions are not visible: “G2.detail” reports aggregate values and intervals, while “G4.detail” merely asserts seeds exist elsewhere. The supplied artifact ends mid-“code_config_hashes”, so whether supporting rows appear later cannot be determined.

## THE CHECK A READER CANNOT DO
Is the reported positive learning contribution spread across units or driven by one outlier?
