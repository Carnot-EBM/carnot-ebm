# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_7812_view_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked from running at the conductor pre-gate because upstream qualification checks in exp7811-training-runtime failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7814_v679_counter_evidence_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified from acceptance due to required validation failures (`complete_disqualified_required_validation`), specifically failing coverage combine and report steps.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7815_qwen_counter_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because 2 of 3 upstream gate checks failed on experiment 7814.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7817_v679_arc_runner_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The circular positive runner qualified the readiness gate, while claiming no benefit.

## WHAT IS MISSING
The artifact cuts off inside the second `probe_rows` entry. The remaining per-probe results needed to check `organic_runner_ready_score` and `acceptance_gate_results.readiness` are not visible. `gate_check_summary` is present and empty, but `honest_verdict` does not say the task was blocked.

## THE CHECK A READER CANNOT DO
Did the results for all six scored probes support the reported readiness gate?

## experiment_7818_v679_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was disqualified due to failed required validation checks (`coverage_report` and `strict_row_lint`), with no positive comparative benefit claimed (`organic_benefit_score`: 0).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7820_v679_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7821_v679_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The evaluation is blocked due to missing required upstream V679 science evidence artifacts and failed upstream counter-evidence checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7822_v679_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment is blocked on missing or disqualified upstream evidence with zero readiness (honest_verdict: "complete_blocked_required_v679_evidence", capstone_complete_score: 0).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
