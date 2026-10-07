# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_8226_learning_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked at the conductor pre-gate because upstream experiment exp8225-delayed-utility-learning failed prerequisite gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8227_v711_concurrency_canary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8228_concurrent_service.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream dependency `exp8227-concurrency-canary` failed three of four evaluation gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8229_v711_arc_outcome_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8230_v711_kv260_workload_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8231_v711_polarfire_state_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8232_v711_gatemate_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate hardware execution remains blocked due to the absence of recorded operator physical change evidence following historical JTAG IDCODE failure.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8233_v711_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Hypothesis H1 failed because treatment arm `energy_group` achieved zero cost gain (`lower_gain` = 0.0) and zero improved sources (`improved_sources` = 0 vs. >= 5 expected) relative to comparator `energy_global`.

## WHAT IS MISSING
Per-unit records showing individual outcomes for each of the 97 complete sources or 128 slots (`cluster_unit`: "original_source", `complete_sources`: 97, `intended_count`: 128). Present are only pooled arm-level aggregates in `calibration_and_cost_comparisons` (`all_slot_cost`, `complete_case_brier`, `complete_case_cost`, `treatment_brier_increase`, and `treatment_cost_increase`), summary counts in `complete_class_support`, and summary diagnostics in `bootstrap_diagnostics`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether `energy_group` and `energy_global` produced identical predictions across all 97 sources or whether individual positive and negative source-level gains canceled out to zero.
