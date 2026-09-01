# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 3 |

## experiment_6853_risk_sensitive_memory_opportunity_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; `"action_manifest"` defines comparison baselines, while `"decision_headroom_rows"` records per-decision observations, but no comparative result or blocked verdict is present.

## THE CHECK A READER CANNOT DO
none

## experiment_6854_risk_sensitive_abstention_memory_controller.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact reports a `"controller_benefit_gate_score"` of 1 despite an `"abstention_rate"` of 1.0.

## WHAT IS MISSING
Per-unit controller-versus-comparator benefit values and the selected controller action for each unit; `"exact_feedback_receipts"` provides `"decision_id"`, hashes, and `"signed_direction"`, but no per-unit gate contribution or comparative metric.

## THE CHECK A READER CANNOT DO
Was the gate score supported broadly across seeds and conditions, or produced by a degenerate comparator or a small number of influential units?

## experiment_6855_counterfactual_memory_credit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact records per-decision convergence outcomes for seeded permutation approximations.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6856_sealed_risk_sensitive_learning_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The system is not ready because abstention is uncalibrated and 12 counterfactually supported writes are harmful.

## WHAT IS MISSING
The per-write metric rows supporting `"harmful_write_count": 12`; only `"per_write_rows_sha256"` is present, while `"rows"` contains per-decision arm metrics rather than per-write credit evidence.

## THE CHECK A READER CANNOT DO
Do the 12 harmful writes each have supported negative counterfactual effects, or is the result driven by a small number of anomalous decisions?

## experiment_6857_dynamic_live_arc_receipt_router.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The read-only receipt-routing pass completed successfully, with all declared checks passing and no game-level solve claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6858_supervisor_counterfactual_credit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `supervisor_headroom_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6859_first_party_tool_gap_receipt_wiring.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The first-party tool-gap receipt contract is ready, while no live effect or solve is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6860_v599_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims a positive held-future benefit of 0.054901960784313725, although continuous self-learning readiness remains disqualified.

## WHAT IS MISSING
Per-unit held-future rows containing unit identifiers and treatment/control metric values underlying `"held_future_effect"` are missing; `"rows"` provides only aggregate `"held_future_row_count": 255` and `"held_future_effect": 0.054901960784313725`. The blocked branches do have diagnostics in `"gate_check_summary"`, `"failed_condition"`, and `"observed_value"`.

## THE CHECK A READER CANNOT DO
Was the reported held-future benefit broad across the 255 units, or driven by a few outliers or units with unequal headroom?
