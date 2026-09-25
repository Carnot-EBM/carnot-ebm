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

## experiment_7645_v667_arc_validation_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an honest null: the CPU goal guard meets readiness qualification across six exact regression fixtures with zero hidden-game probability benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7646_v667_source_feature_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The feature corpus was completed but disqualified because required validation failed: 15 of 18 receipts passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7647_witness_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 7647 was blocked at the conductor pre-gate because upstream dependency exp7646-source-feature-corpus failed required gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7650_v667_independent_source_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit is complete but downstream benefit evaluation is blocked because upstream producer artifacts are disqualified, flagged adversarial, pre-gate blocked, or missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7651_v667_qwen_witness_challenge.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed eight-group paired pilot found no conservatively supported claim intersection and makes no confirmatory benefit claim.

## WHAT IS MISSING
nothing; `paired_pilot_rows` and `rows` provide per-unit results, while `acceptance_gate_results` records which gates were and were not assessed.

## THE CHECK A READER CANNOT DO
none

## experiment_7652_v667_arc_wrapper_measurement.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The wrapper showed no added induced successes and was disqualified because validation failed.

## WHAT IS MISSING
The artifact cuts off inside `per_game_results.dc22`; the remaining per-unit rows needed to check `independent_reduction.induced_new_successes: 0` are missing. `gate_check_summary.validation_failures` does identify `full_python_suite`.

## THE CHECK A READER CANNOT DO
Did every remaining game and window show zero added induced successes?

## experiment_7653_v667_arc_live_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Six episodes completed, but required validation failed and the three public games showed no paired level gain.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7656_v667_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is complete but blocked from supporting a V667 benefit claim because required upstream evidence failed eligibility checks or was absent.

## WHAT IS MISSING
nothing; `gate_check_summary.failed_checks` records each failed check and its `expected` and `observed` values.

## THE CHECK A READER CANNOT DO
none
