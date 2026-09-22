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

## experiment_7490_v656_historical_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No claim; the artifact records an honest null historical audit where benefit gates failed and chronological leakage was detected in the negative control.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7491_v656_window_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7492_v656_window_pilot.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The listed acceptance gates passed, capture was forecast feasible, and `"predictive_benefit_claimed"` was false.

## WHAT IS MISSING
The artifact is truncated inside `"current_invocation_events"` and omits the remainder, including the referenced `"honest_verdict"` and potentially additional per-unit evidence or blocker diagnostics; `"acceptance_gate_results"` and `"capture_budget_forecasts"` are present.

## THE CHECK A READER CANNOT DO
Does the omitted remainder contain the complete per-unit measurements needed to verify the gate results rather than trust their aggregate summaries?

## experiment_7493_v656_window_fit_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7494_v656_window_eval_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7498_v656_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment failed multiple acceptance gates across readiness and benefit categories because upstream calibration and causal learning artifacts were absent.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7500_v656_arc_opportunity_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The opportunity audit is blocked because Panel B source artifacts are missing and pooling support gates failed (18 valid episodes observed versus 30 required).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7502_v656_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone completed but is blocked because required V656 evidence is missing or upstream tasks are externally gated.

## WHAT IS MISSING
nothing; `gate_check_summary.failed_checks`, `acceptance_gate_results`, and `preconditions_checked` record the failed checks, paths, expected values, and observed values.

## THE CHECK A READER CANNOT DO
none
