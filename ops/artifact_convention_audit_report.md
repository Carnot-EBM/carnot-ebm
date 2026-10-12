# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_8435_v726_capacity_benefit_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Frozen and capacity-8 FIFO fail the positive-gain gates against uniform, while decision-loss and retention gates pass.

## WHAT IS MISSING
Per-seed, per-arm outcome metrics and per-window Brier differences: "acceptance_gates" contains only booleans, while "admission_rows" records admission metadata without performance outcomes.

## THE CHECK A READER CANNOT DO
Did frozen versus uniform show consistent effects across the five seeds, or did one seed drive the failed gain gate?

## experiment_8436_v726_transaction_attribution.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports attribution readiness with all five “acceptance_gates” true.

## WHAT IS MISSING
The artifact cuts off inside “historical_model_provenance”; actual “honest_verdict”, “verdict_class”, and “rows” values are unavailable, although “field_principles” names them. “attribution_rows_path” provides a pointer, not measurements.

## THE CHECK A READER CANNOT DO
Does the final headline verdict make a comparative claim supported by per-unit measurements?

## experiment_8437_v726_acceleration_ceiling.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Arithmetic acceleration alone cannot achieve a 10× speedup under fixed service assumptions, in any primary repeat.

## WHAT IS MISSING
The supplied artifact is truncated inside "historical_model_provenance". "fraction_accelerable" and "descriptive_interval" provide aggregates; actual "rows" and "per_repeat_bounds" are not visible, although "field_principles" names them. Per-repeat identities, total service times, and accelerable arithmetic times are needed.

## THE CHECK A READER CANNOT DO
Does each primary repeat’s arithmetic fraction independently imply a maximum speedup below 10×?

## experiment_8438_bounded_qwen_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing — `"gates_evaluated"` and `"gate_check_summary"` identify the failed check, observed value, and required value.

## THE CHECK A READER CANNOT DO
none

## experiment_8439_v726_arc_supervisor_support.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Supervisor support readiness is blocked by an upstream hash mismatch and insufficient refinement support, so no arm refinement is authorized.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8440_v726_board_reader_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Board qualification is gated by missing compatible kernels and incomplete transport/cost evidence, with those blockers explicitly recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8441_v726_gatemate_obligation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GateMate obligation is recorded, while historical authentication and hardware execution remain blocked by explicitly identified missing source bytes and physical/device evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8442_v726_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports passing reconciliation gates while independent scientific generalization remains unestablished.

## WHAT IS MISSING
The JSON ends mid-string inside "historical_model_provenance"; actual "honest_verdict", "scientific_claims", and "rows" values are unavailable, although "field_principles" names them. "gate_check_summary" records failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the missing remainder provide per-unit metrics supporting the artifact’s final comparative claims?
