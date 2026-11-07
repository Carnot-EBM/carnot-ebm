# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 6 |

## experiment_4381_biprm_detector_localization_abstention.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Bidirectional fusion provides no localization improvement over L2R and no useful abstention point, although the corrigendum calls the null uninformative because labels are degenerate.

## WHAT IS MISSING
Per-trace IDs, true first-error steps, predictions for each direction, correctness labels, and rejection scores. "localization_f1_by_direction", "localization_delta_ci95", and "abstention_curve.points" contain only aggregates; "corrigendum_2026_11_07" supplies no per-trace rows.

## THE CHECK A READER CANNOT DO
Which traces were retained at 75% coverage, and do their outcomes reproduce the reported retained accuracy of 0.999796?

No experiment was rerun; the per-trace behavior must be trusted.

## experiment_8897_v761_source_bound_decisions.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment marks all "acceptance_gates" true, but the headline research claim is unavailable.

## WHAT IS MISSING
The remainder of the artifact: the supplied text ends mid-entry in "cited_upstream_artifacts", leaving any final verdict and per-unit results unseen.

## THE CHECK A READER CANNOT DO
Does the artifact contain per-unit evidence supporting a comparative headline claim?

The missing remainder was not checked; truncation prevents a definitive classification.

## experiment_8898_v761_signed_benefit_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The signed mechanism failed the required future-Brier improvement gate and should be retired unchanged.

## WHAT IS MISSING
The supplied artifact truncates inside "paired_stream_rows" while showing only "feature_arm": "old"; signed per-stream "metric" values supporting the headline are not visible, although aggregate results appear in "joint_gate_rows".

## THE CHECK A READER CANNOT DO
Does the signed gate failure reflect broad results across the 24 streams or a few outliers?

No experiment was rerun; the truncated remainder remains unchecked.

## experiment_8899_v761_buffered_cost_attribution.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No overall headline claim is visible in the truncated artifact.

## WHAT IS MISSING
The remainder after the incomplete "unit_id" in "closure_residuals", including any final claim or verdict; "acceptance_gates" already identifies several failed checks.

## THE CHECK A READER CANNOT DO
Does the complete artifact claim a comparative result supported by per-unit metrics?

The omitted remainder could not be checked and could change this classification.

## experiment_8900_qwen_source_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8901_v761_signed_adaptation_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The excerpt asserts adaptation-cost readiness through "adaptation_cost_ready_score": 1, but no headline verdict is visible.

## WHAT IS MISSING
The artifact cuts off mid-key inside "durable_commit_rows"; its remaining claim and verdict fields are unavailable. "complete_parent_costs" supplies per-"unit_id" measurements, so those costs are not aggregate-only.

## THE CHECK A READER CANNOT DO
Does the omitted remainder report a blocked verdict and explain its cause? The truncated remainder cannot be checked.

## experiment_8902_v761_hardware_operation_map.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact reports that the audit gate passed, physical acceleration failed, and acquisition was deferred.

## WHAT IS MISSING
The remainder of the artifact: “adaptation_durable_commits” ends mid-“duration_ns” value, leaving any subsequent claim, results, or verdict unavailable.

## THE CHECK A READER CANNOT DO
Does the complete artifact supply per-unit metrics supporting its headline claim? The truncated remainder could not be checked.

## experiment_8903_v761_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The shared-terminal branch reports mean learned-energy label savings of 2,844 across 40 replicates, with zero estimated coverage differences.

## WHAT IS MISSING
Complete `"per_replicate_results"`: the supplied artifact stops midway through replicate 23, leaving that row incomplete and replicates 24–39 unavailable; `"paired_intervals"` reports `"independent_count": 40.

## THE CHECK A READER CANNOT DO
Do the 40 replicate observations reproduce the reported mean savings of 2,844?

Unseen rows were not checked; truncation leaves the full claim unverified.
