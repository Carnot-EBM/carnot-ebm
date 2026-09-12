# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_7224_span_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7223-span-canary.span_canary_ready_score` was observed as 0 but required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7226_v636_belief_compiler.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The packed belief compiler matches the circular finite reference with zero mismatches.

## WHAT IS MISSING
Per-state/per-case parity rows containing the packed and reference outputs; `"parity_rows"` records only aggregate `"mismatch_count"`, `"state_count"`, and case counts, while `"rows"` contains only per-seed `"fresh_stream_readiness"` results.

## THE CHECK A READER CANNOT DO
Did every tested prediction, query, and energy case actually agree, or does the reported zero-mismatch aggregate conceal omitted or miscounted cases?

## experiment_7227_v636_belief_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Packed online memory did not pass every fixed prospective learning gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7228_v636_belief_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Packed online memory beat frozen warmup on future error and false accepts, increased error when learned state was removed, and matched the reference online version space.

## WHAT IS MISSING
nothing; `"comparison_recomputation_rows"` includes `"seed_differences"` for all 20 `"stream_seed"` units, and `"causal_control_rows"` and `"cold_reload_rows"` provide per-seed diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_7229_v636_rare_event_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit preserved Exp7216’s failed gate and found the fixed rare-event accuracy target infeasible within the stated transition budget.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7230_v636_native_belief.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The native PyO3 implementation matched the Python reference, restored state across a fresh process, and passed the paired end-to-end cost gate.

## WHAT IS MISSING
The artifact is truncated inside `"parity_rows"`, so the complete per-unit parity and fresh-process restoration rows cannot be found; `"cost_rows"` does contain per-repetition arm measurements, and `"gate_check_summary"` records no failed check.

## THE CHECK A READER CANNOT DO
Did every parity and state-restoration unit pass, rather than merely the visible aggregate exhaustive-domain check?

## experiment_7231_v636_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three board dispositions are authenticated: KV260’s prior graduation is preserved, GateMate remains blocked by the absence of a post-Exp6559 physical-state receipt, and one PolarFire CPU smoke dispatch completed with retained transcript hashes, without claiming FPGA sampling or hardware performance.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7232_v636_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The capstone claims that its complete fourteen-row evidence matrix is stored and validated.

## WHAT IS MISSING
The artifact is truncated mid-value inside `"evidence_matrix"`, so the remaining matrix rows, their `"gate_check_summary"` fields, and any per-unit measurement rows are unavailable.

## THE CHECK A READER CANNOT DO
Do all fourteen matrix entries either include the per-unit rows needed for their comparative claims or record the specific failed check and observed value for every blocked verdict?
