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

## experiment_7224_span_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7223-span-canary.span_canary_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7226_v636_belief_compiler.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The packed belief compiler matches the circular finite reference across the tested parity cases and replay seeds.

## WHAT IS MISSING
Per-case packed-versus-reference outputs and per-seed replay metrics are missing; `"parity_rows"` contains only aggregate case counts and `"mismatch_count": 0`, while `"rows"` contains only single-arm `"fresh_stream_readiness"` records.

## THE CHECK A READER CANNOT DO
For any specific tested state, input, or replay seed, did the packed compiler and reference produce the same prediction, energy, and query result?

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
nothing; `"comparison_recomputation_rows"` includes per-seed `"seed_differences"` for all reported comparative metrics, while `"causal_control_rows"` and `"cold_reload_rows"` provide additional per-seed results.

## THE CHECK A READER CANNOT DO
none

## experiment_7229_v636_rare_event_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit preserved Exp7216’s failed gate, classified every failed criterion, and concluded that the fixed rare-event accuracy target is infeasible within the stated budget.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7230_v636_native_belief.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native PyO3 packed-belief implementation matched the Python reference, restored state across a fresh process, and passed the paired end-to-end native cost gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7231_v636_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three board dispositions are authenticated: KV260’s graduation is preserved, GateMate remains blocked by the absence of a post-Exp6559 physical-state receipt, and one hash-verified PolarFire CPU dispatch completed with retained raw evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7232_v636_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The capstone claims that a complete fourteen-row evidence matrix supports its branch retirement, continuation, and prerequisite decisions.

## WHAT IS MISSING
The artifact is truncated mid-`evidence_matrix`, inside an `observed_value`; despite present fields such as `capstone_complete_score`, `raw_row_count`, `gate_check_summary`, and `acceptance_gate_replay_rows`, the remaining matrix entries and any per-unit metric rows or blocker diagnostics they may contain are unavailable.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether every comparative or gate claim in the fourteen-row matrix is backed by per-unit evidence rather than aggregate summaries.
