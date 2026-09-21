# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7464_v654_semif_e6_decision_cost_profile.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest scientific null where E6 acceptance gates were not met due to insufficient sample size (8 of 30 episodes, 2 of 10 game clusters) and low trace time attribution (10.3%).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7465_source_option_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at `conductor_pre_gate` because upstream dependency `exp7462-option-protocol` failed three evaluated gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7467_v654_factual_span_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The result is null because the development gate stayed closed: the factual canary produced only 1 usable span output and 2 usable verbatim outputs, below the required 3 per arm.

## WHAT IS MISSING
nothing; `"development_gate"`, `"factual.failures"`, `"usable_by_arm"`, `"required_usable_per_arm"`, and per-unit `"development_rows"` record the failed check and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7468_v654_residual_learner.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7470_v654_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
An honest null independent audit concluding that two upstream branches were blocked due to missing producer evidence while the extraction branch produced a null result.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7471_v654_arc_seam_observation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live ARC seam observation resulted in an honest complete null with zero reproduced progress across all four games and eight episodes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7473_v654_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null finding that historical continuity is preserved for KV260 and PolarFire, while GateMate remains blocked by an unchanged physical prerequisite.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7474_v654_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
