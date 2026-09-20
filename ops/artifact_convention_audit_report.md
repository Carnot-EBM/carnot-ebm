# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| UNKNOWN | 4 |

## experiment_7453_energy_calibration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before running by a conductor pre-gate check because upstream dependency `exp7452-source-embeddings` reported `embedding_capture_ready_score` = 0 instead of the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7454_v653_continuous_learning.json

**UNKNOWN**

jetski: no output produced — a tool required the "command" permission that headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under permissions.allow in settings.json (e.g. command(<target>)). Alternatively, re-run with --dangerously-skip-permissions to auto-approve all tools.

## experiment_7455_v653_decision_audit.json

**UNKNOWN**

jetski: no output produced — a tool required the "command" permission that headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under permissions.allow in settings.json (e.g. command(<target>)). Alternatively, re-run with --dangerously-skip-permissions to auto-approve all tools.

## experiment_7456_v653_extraction_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an honest null disposition (`honest_verdict`: "complete_null_span_capture_development_gate_closed") with zero span value score, failing the evaluation benefit gate because no evaluation effect was established.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7457_v653_arc_exposure.json

**UNKNOWN**

jetski: no output produced — a tool required the "command" permission that headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under permissions.allow in settings.json (e.g. command(<target>)). Alternatively, re-run with --dangerously-skip-permissions to auto-approve all tools.

## experiment_7458_v653_durable_updates.json

**UNKNOWN**

jetski: no output produced — a tool required the "command" permission that headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under permissions.allow in settings.json (e.g. command(<target>)). Alternatively, re-run with --dangerously-skip-permissions to auto-approve all tools.

## experiment_7459_v653_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null audit finding that GateMate remains blocked by an unchanged physical prerequisite while KV260 and PolarFire retain historical continuity with no new hardware value.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7460_v653_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
