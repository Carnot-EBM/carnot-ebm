# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8883_v760_evidence_loss_boundary.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

The evidence-loss boundary experiment passed its controls, with exact-view arms eliminating the learned arm’s missing-hop semantic errors.

## WHAT IS MISSING

Per-world, per-head, per-arm decisions and scored metrics are not visible: "arm_view_summaries" and "paired_intervals" contain aggregates; "full_source_truth" contains labels only. The supplied artifact cuts off inside "paired_intervals", so outcome rows might follow.

## THE CHECK A READER CANNOT DO

Which heads and worlds account for the learned arm’s reported 80% missing-hop error rate?

The truncated remainder was not checked; whether it contains supporting rows remains unknown.

## experiment_8884_v760_topology_causal_streams.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied, truncated artifact.

## WHAT IS MISSING
The artifact ends mid-entry in "admission_rows"; the visible rows contain admission events and hashes, while "acceptance_gates" lists criteria without reporting whether they were met.

## THE CHECK A READER CANNOT DO
Does the completed artifact claim a comparative benefit, and provide per-unit metrics supporting it?

Only the supplied fragment was checked; truncation prevents a definitive classification.

## experiment_8885_v760_sequential_label_cost.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

no claim

## WHAT IS MISSING

The supplied artifact ends mid-entry in "calibration_bound_rows", leaving any subsequent headline claim, final verdict, and gate outcomes unavailable; "acceptance_gates" records thresholds only.

## THE CHECK A READER CANNOT DO

Did sequential calibration save labels while meeting the coverage and risk gates?

The omitted remainder was not checked; its claim and supporting evidence remain unknown.

## experiment_8886_v760_label_benefit_audit.json

**CHECKABLE**

## VERDICT

CHECKABLE

## WHAT THE CLAIM IS

Matched labels failed the joint benefit gates: online Brier improvement and checkpoint-24 retention error failed their thresholds.

## WHAT IS MISSING

nothing

## THE CHECK A READER CANNOT DO

none

Not checked: experiment execution or external artifacts; this verdict covers the supplied text.

## experiment_8887_qwen_source_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `exp8878-branch-contract.runtime_repair_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8888_v760_matched_adaptation_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact reports adaptation-cost readiness through `"adaptation_cost_ready_score": 1`.

## WHAT IS MISSING
The artifact is truncated inside `"durable_commit_rows"` at an unfinished `"hash"` field. `"complete_parent_costs"` supplies per-unit timings, but the remainder needed to inspect any further claims and gate diagnostics is unavailable.

## THE CHECK A READER CANNOT DO
Which acceptance-check results justify `"adaptation_cost_ready_score": 1`?

The truncated remainder could not be checked; it may contain the needed evidence.

## experiment_8889_v760_hardware_operation_map.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports that the audit gate passed and the physical acceleration gate failed.

## WHAT IS MISSING
The artifact’s remainder: it ends mid-`"hash"` inside `"adaptation_durable_commits"`. `"acceptance_gates"` is present, but the complete headline verdict and any subsequent comparative results cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the complete artifact provide per-unit metrics supporting any comparative headline claim?

The truncated remainder could not be checked; a definitive classification would be unreliable.

## experiment_8890_v760_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports qualified branch readiness with activation unauthorized, but the overall headline claim is not shown.

## WHAT IS MISSING
The artifact cuts off inside "branch_conclusions[2].reduction.world_ids"; the remaining conclusions and any overall verdict or supporting metric rows are unavailable. Visible "continuation_gate" entries record "expected", "observed", and "passed".

## THE CHECK A READER CANNOT DO
Does the omitted remainder, which could not be checked, make a comparative claim and provide per-unit metrics supporting it?
