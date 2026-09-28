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

## experiment_7783_source_view_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked from executing because upstream dependency `exp7782-historical-compatibility` failed required pre-flight gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7787_v677_qwen_event_confidence.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
This exposed pilot found no demonstrated decision benefit; the decision-benefit and probability-quality gates failed.

## WHAT IS MISSING
The artifact ends mid-field in `panel`, so the rest of the record is unavailable. `paired_improvements.brier.per_family` and `paired_improvements.cost.per_family` are present, but I cannot tell whether per-arm unit rows appear later.

## THE CHECK A READER CANNOT DO
Do the per-arm results for each family support the reported gate failures?

## experiment_7784_v677_training_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The training runtime candidate is disqualified from qualification readiness (`honest_verdict`: "complete_disqualified_required_validation") due to failed validation gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7789_v677_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run is blocked from readiness because the required Exp7786 and Exp7788 producer artifacts are missing.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies both failed `producer_path` checks and records `observed: "missing"`, while `rows` contains per-unit metrics for the available comparison.

## THE CHECK A READER CANNOT DO
none

## experiment_7790_v677_arc_runner_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runner was disqualified because required validation failed.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `full_python_suite` as the failed check, with `observed: -15` versus `expected: 0`, and `probe_rows` records per-unit results.

## THE CHECK A READER CANNOT DO
none

## experiment_7791_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because two upstream gate checks failed.

## WHAT IS MISSING
nothing; `gates_evaluated` records each check and its observed value, and `gate_check_summary` identifies the failures.

## THE CHECK A READER CANNOT DO
none

## experiment_7793_v677_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No comparative claim; the evaluation is blocked because upstream producer Exp7792 failed the service producer eligibility check (`eligible_service_producer == false`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7794_v677_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The exposed Qwen event-confidence arm showed no qualified benefit over the generic arm, so its unchanged scope should be retired.

## WHAT IS MISSING
Per-family Brier and cost values for **each arm** are missing. `paired_improvements.brier.per_family` and `paired_improvements.cost.per_family` give differences, while `semantic_comparison_rows` gives arm-level aggregates; the `rows` array accounts for tasks rather than recording each family’s arm metrics. `gate_check_summary` does record the failed checks, so the block has a diagnostic.

## THE CHECK A READER CANNOT DO
For each of the 24 families, was the generic arm already at a metric floor or ceiling, leaving no headroom for the event arm?
