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

## experiment_8240_v712_qualified_delayed_learning.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8241_v712_delayed_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8242_v712_independent_concurrent_service.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8243_v712_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8244_v712_kv260_decision_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8245_v712_polarfire_state_dispatch.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8246_v712_gatemate_change_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate hardware execution remains blocked because no authenticated physical change evidence has been recorded since experiment 8232.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8247_v712_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims hypothesis H1 failed because `energy_margin` did not meet the required thresholds for `lower_gain`, `improved_sources`, and `original_frozen_v707_radial_cost_increase` relative to the calibration-selected primary arm and baseline controls.

## WHAT IS MISSING
While aggregate summary values are reported in `statistics.H1.operands` and `action_switch_matrices`, the artifact's `action_switch_rows` contains per-unit entries exclusively for `"arm": "energy_uniform"`. Per-unit/per-source rows giving metrics, costs, and decisions for the actual arms evaluated in the comparison (`energy_margin`, `primary`, `original_frozen_v707_radial`, `additive_margin`, and `logistic_margin`) are missing entirely.

## THE CHECK A READER CANNOT DO
A reader cannot determine which 2 source clusters actually improved under `energy_margin`, nor whether the observed `lower_gain` of -0.04296875 was a consistent effect across sources or driven by extreme outliers.
