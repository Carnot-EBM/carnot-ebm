# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| CANNOT_DETERMINE | 2 |

## experiment_8444_source_control_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8447_brier_online_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `methods_ready_score` was 0, failing the gate requiring it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8449_v727_persistence_localization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Persistence localization was blocked, with “gate_check_summary” recording failed authority checks and their observed values, and “rows” recording all 20 intended runs as censored.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8450_journal_equivalence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `persistence_profile_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8453_v727_arc_seal_and_support.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fields claim receipt readiness passes, while support readiness fails and arm refinement is unauthorized.

## WHAT IS MISSING
The artifact cuts off inside "future_dependencies"; "honest_verdict", "rows", and "per_game_results" are named in "field_principles", but their actual contents are not supplied.

## THE CHECK A READER CANNOT DO
Does the complete verdict make a comparative claim supported by per-unit results?

## experiment_8454_v727_board_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Board-operation readiness is gated by unmet contract, operation-compatibility, and transfer/device-clock requirements, with failed checks and observed values recorded in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8455_v727_gatemate_obligation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GateMate obligation remains blocked, with failed checks and observed values recorded in "gate_check_summary" and individual missing reasons in "obligation_rows".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8456_v727_capstone.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

“g2” claims historical FoVer results, including a positive learning contribution, reproduce; the capstone’s headline verdict is not visible.

## WHAT IS MISSING

The supplied JSON cuts off inside “historical_model_provenance”. Actual “honest_verdict”, “scientific_claims”, and “rows” values are not visible; “field_principles” merely names them. “gate_check_summary” does provide failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO

Does the complete artifact contain per-unit paired metrics supporting the positive learning contribution reported in “g2”?
