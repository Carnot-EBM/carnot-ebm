# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7480_v655_source_eval_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7481_v655_typed_calibration.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The candidate met the decision-benefit gate but failed the probability-benefit gate, leaving `"deployment_certificate_valid": false`.

## WHAT IS MISSING
Per-group evaluation rows containing each of the 74 groups’ candidate and comparator costs; `"decision_cost_grid.cells[].comparison"` provides only aggregate `"delta"`, `"ci95"`, `"group_count"`, and p-values, while `"rows"` is mentioned only in `"field_principles"` and no actual per-unit rows are present.

## THE CHECK A READER CANNOT DO
Was the reported decision-cost benefit broad across the 74 groups, or driven by a few outliers or degenerate groups?

## experiment_7482_v655_importance_anchor.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The importance anchor arm passes the scientific fixture benefit gate by improving adaptation loss by ~0.150 (above the 0.005 threshold) while maintaining a retention drift ratio of ~0.736 relative to the unanchored arm (below the 0.80 maximum threshold) across all three evaluation seeds.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7483_v655_continuous_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed validly but found no registered importance-anchor benefit because the Holm-adjusted comparisons against the unanchored residual failed at delays 0 and 8.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7484_v655_decision_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent audit completed with a null result because `static_and_online_benefit` was not demonstrated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7485_v655_arc_cost_panel_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7487_v655_learning_placement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed placement and fixed-point replay validation, but did not establish the 100× end-to-end service target because the service denominator was incomplete.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7488_v655_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is disqualified because required evidence failed validation, panel B is missing, and the ARC evidence is a sample-limited null without established intervention benefit.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` records each failed check with `"field"`, `"expected"`, `"observed"`, `"path"`, and `"upstream"`, while `"arc_combined_reduction.episode_rows"` provides per-game/per-seed rows.

## THE CHECK A READER CANNOT DO
none
