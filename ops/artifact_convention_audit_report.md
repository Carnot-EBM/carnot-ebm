# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8667_v744_contract_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Administrative contract readiness is not established: `methods_ready_score` is 0, and `gate_check_summary` records failed checks and observed values.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8668_frozen_learning_core.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `exp8667-contract-methods.methods_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8670_relational_evidence_controls.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `exp8667-contract-methods.methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8674_installed_batch_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because upstream `methods_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8677_v744_arc_observation_ownership.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The recorded acceptance gates passed, including exact policy parity across eight game/seed units.

## WHAT IS MISSING
The artifact ends mid-value inside "callback_ownership_rows", leaving the remainder unavailable; "action_parity_rows" provides eight per-unit parity results, but supporting results for the other "acceptance_gates" are not visible.

## THE CHECK A READER CANNOT DO
What per-condition results establish that "acceptance_gates.horizon_controls" passed?

## experiment_8678_v744_arc_observer_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Exact policy parity passed for all eight game–seed pairs, with per-unit results recorded in "action_parity_rows".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8679_v744_hardware_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports no compatible physical execution, incomplete new cost boundaries, and unchanged acquisition priority.

## WHAT IS MISSING
The artifact ends mid-path inside "cited_upstream_artifacts"; the headline verdict and any subsequent measurements are unavailable. "acceptance_gates" and "board_status_rows" do record failed checks and board-specific prerequisites, so BLOCKED_WITHOUT_DIAGNOSTIC is not supported.

## THE CHECK A READER CANNOT DO
Does the missing remainder assert a comparative result without recording per-unit metrics?

## experiment_8680_v744_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“G2.detail” claims the FoVer headline reproduced within published AUROC and positive learning-contribution intervals.

## WHAT IS MISSING
FoVer per-unit production/control metrics are absent from the visible text; “arc_decision_rows.horizon_rows” records ARC observations instead. The artifact cuts off mid-“observation_sha256”, so whether supporting FoVer rows appear later is unknown.

## THE CHECK A READER CANNOT DO
Was the positive FoVer learning contribution broadly supported across units or driven by a few outliers?
