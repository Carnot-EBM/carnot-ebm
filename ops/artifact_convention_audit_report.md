# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| AGGREGATE_ONLY | 1 |

## experiment_3392_archive_v311_activate_v312.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims archive v311 is complete and activation of v312 is ready.

## WHAT IS MISSING
Per-artifact rows/checks behind `"completed_artifacts"`, `"blocked_artifacts"`, `"missing_artifacts"`, `"duration_flagged_artifacts"`, and `"archive_v311_activate_v312_ready"`.

## THE CHECK A READER CANNOT DO
Which upstream artifacts were actually checked, and what result did each one produce before declaring the archive complete?

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The headline claim is that the root cause was `write_path_missing`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6260_goal_only_induction_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Goal-only beat the combined call on the gate: 1 of 4 goal-only predicates fired on a real win where 0 of 4 combined-call predicates did, with the gain on `ls20`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
