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

## experiment_7126_v626_arc_loo_phase_receipts.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Exp7123 was a blocked no-run whose auditable progress stopped before setup because required phase and execution receipts were absent.

## WHAT IS MISSING
nothing; the blocker is identified in `"first_absent_start_receipt"`, `"missing_receipt_rows"`, `"phase_timing_rows"`, and diagnostic `"detail"` values for `"artifact_postflight_failure"`, `"conductor_timeout"`, and `"task_exit"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7127_v626_adapter_withheld_arc_loo.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed paired comparison found no difference: both adapter-withheld and adapter-visible-control arms achieved zero levels on game `r11l`, with `level_delta` 0 and no solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7128_v626_arc_loo_causal_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit disqualifies the experiment because the withheld arm failed the `adapter_access_clean` gate: target adapter code remained accessible.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7129_v626_sota_constraint_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated inside `"exact_outcome_rows"` at `"cell_key": "unsloth`, so any headline or verdict fields and the remaining per-cell rows cannot be found; present fields include `"MODEL_SPECS"`, `"base_instance_rows"`, `"completed_cell_count"`, and partial `"exact_outcome_rows"`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the missing portion makes a comparative or blocked headline claim, much less check that claim.

## experiment_7130_v626_verifier_committed_routing.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible in the provided fragment

## WHAT IS MISSING
The artifact is truncated mid-value after `"reason": "exact_rejection_is_f`; any headline verdict, comparative summary, gate result, and remaining per-unit rows are missing, although `"abstention_rate"` and partial `"abstention_rows"` are present.

## THE CHECK A READER CANNOT DO
Did the complete artifact claim that one arm beat another or that a gate was blocked, and did it include the per-unit metrics or blocker diagnostic needed to verify that claim?

## experiment_7133_v626_multiscale_sampler_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The corrected host-software multiscale proposal achieves exact finite-law parity on both recorded 2×2 fixtures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7134_v626_multiscale_sampler_benchmark.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be determined because the artifact ends mid-record.

## WHAT IS MISSING
The remainder of the artifact, including any headline verdict or gate result; `"acceptance_rows"`, `"autocorrelation_rows"`, and `"asymptotic_scaling_claimed"` are present, but the JSON is incomplete.

## THE CHECK A READER CANNOT DO
Did the experiment ultimately claim a comparative win, report a blocked result, or make no claim?

## experiment_7135_v626_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Corrected multiscale sampling improves the preregistered bounded host ESS statistic.

## WHAT IS MISSING
The actual per-seed sampler rows with each arm’s ESS metric are missing; `"row_recompute_rows"` merely says `"support": "per-seed sampler rows"`, while `"artifact_verdict_rows"` reports only aggregate headline fields.

## THE CHECK A READER CANNOT DO
Was the claimed ESS improvement broad across seeds, or driven by one outlier or degenerate control?
