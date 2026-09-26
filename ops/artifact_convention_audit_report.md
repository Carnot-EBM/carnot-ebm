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

## experiment_7705_v671_constraint_bank_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact reports valid fixture mechanics without empirical acquisition and shows a lower aggregate Brier score for `weight_only`.

## WHAT IS MISSING
The supplied artifact cuts off mid-`lifecycle_rows`, so I cannot tell whether metric-bearing per-unit `rows` appear later. The visible `mean_brier_by_arm` and `mean_base_brier_by_arm` contain only aggregates; the visible `lifecycle_rows` contain events, not Brier scores.

## THE CHECK A READER CANNOT DO
Did `weight_only` improve Brier scores across many units, or did a few units account for its lower mean?

## experiment_7706_continuous_acquisition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because upstream prerequisite `exp7705-constraint-bank-protocol` failed its `constraint_bank_ready_score` gate check (observed 0, expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7707_v671_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked from completion and activation due to failing required upstream readiness and validity checks (`complete_blocked_required_evidence`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7708_v671_arc_generalization_runner.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7709_v671_arc_first_contact.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run completed but was disqualified because required validation was incomplete: 11 of 12 receipts passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7710_v671_native_record_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native-record checks completed but did not meet readiness, despite Python–Rust parity on fixture units and a passing durable-replay check.

## WHAT IS MISSING
nothing — `parity_rows` records per-unit outcomes, and `acceptance_gate_results` identifies failed gates and their measured operands.

## THE CHECK A READER CANNOT DO
none

## experiment_7711_whole_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked from executing at the conductor pre-gate layer because upstream dependency `exp7710-native-record-contract` failed its gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7712_v671_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact reports a decision-cost advantage for the matched-logistic candidate over its control across 40 static families.

## WHAT IS MISSING
Per-family, paired decision-cost rows for both arms, with family identifiers. The artifact gives `"decision_cost"` contrasts and `"cost_mean"` values; its `"interval_inputs"` contain per-family Brier values, not decision costs.

## THE CHECK A READER CANNOT DO
Was the reported 0.165 decision-cost advantage spread across the 40 families or driven by a few outliers?
