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

## experiment_7107_v623_continual_memory_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The continual-memory cold audit passed, with delayed procedural memory outperforming the comparison arms across the reported capacities.

## WHAT IS MISSING
nothing; `"capacity_rows"` provides aggregates, while `"event_replay_rows"` records per-event correctness for every arm, and `"crash_recovery_rows"` records per-attack outcomes and hashes.

## THE CHECK A READER CANNOT DO
none

## experiment_7108_v623_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims a positive, promotion-worthy delayed-commit procedural-memory A/B result within a complete V623 evidence matrix.

## WHAT IS MISSING
The per-unit A/B metric rows for experiment 7106 are missing; `"per_unit_presence_rows"` records only `"row_count": 720`, while `"headline_recomputation_rows"` records only aggregate readiness scores, and no actual `"self_learning_rows"` are provided.

## THE CHECK A READER CANNOT DO
Did the procedural-memory arm improve broadly across the 720 units, or was the claimed advantage driven by a few outliers or degenerate control units?

## experiment_7109_v624_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V624 contract is disqualified because `yaml_task_count` was 6 rather than the expected 12.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7110_v624_evidence_ingress_quarantine.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no complete headline claim is present in the supplied excerpt

## WHAT IS MISSING
The artifact is truncated inside `"reference_paths"` after `"json_path": "flagged_artifacts_skipped.0.experiment_id",`; the remaining fields, closing structure, and any top-level verdict or claim fields are missing.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the omitted portion contains a comparative claim, a blocked verdict, or its diagnostic.

## experiment_7111_v624_arc_provenance_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC forward-provenance canaries passed, including writer validation, headline eligibility, dashboard consumption, and registry stability checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7112_v624_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded V624 source audit completed successfully, passed its ingestion gate, and adopted one decision-relevant candidate into an existing experiment.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7113_v624_arc_generation_liveness.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC generation-liveness run was blocked because required preconditions failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7114_adapter_withheld_arc_loo_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `arc_generation_liveness_ready_score` was observed as 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
