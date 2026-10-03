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
| CANNOT_DETERMINE | 5 |

## experiment_8033_v696_scoring_isolation.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8034_fit_likelihood_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8038_v696_windowed_online_learning.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8039_v696_learning_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8040_v696_native_transaction_cost.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The native transaction cost implementation achieves an arithmetic speedup of ~17.4x to 17.9x across 30 paired repetitions and satisfies all acceptance gates (`measurement`, `owned_checks`, `parity`, `regression`).

## WHAT IS MISSING
Individual per-repetition baseline and candidate execution times or speedups for each of the 30 paired runs in `arithmetic_speedup` (which only reports `speedup`, `lower_95`, `upper_95`, `paired_repetitions`, and `independent_streams`), as well as the underlying measurements and thresholds for the boolean flags in `acceptance_gate_results` (`measurement`, `owned_checks`, `parity`, `regression`).

## THE CHECK A READER CANNOT DO
Did the individual run times across the 30 paired repetitions show consistent speedups, or was the aggregate speedup driven by timing outliers and variance?

## experiment_8041_v696_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8042_v696_precision_fallback_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8043_v696_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
