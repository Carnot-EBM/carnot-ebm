# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 2 |

## experiment_6840_residual_memory_chronological_shard_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The chronological shard completed with a null result for residual memory, supported by rows and receipts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6841_residual_memory_delayed_correction_shard_b.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The `"honest_verdict"` claims a complete null comparative result for residual memory on shard B.

## WHAT IS MISSING
Per-unit outcome rows containing each event/seed/arm’s residual-error or decision-accuracy metric are missing; `"held_future_results"`, `"delayed_correction_results"`, and `"residual_error_results"` are aggregate summaries, while `"memory_state_transitions"` records transition receipts rather than the compared outcome metrics.

## THE CHECK A READER CANNOT DO
Did the null result hold broadly across events and seeds, or was it produced by outliers, degenerate controls, or the many no-headroom rows reported in `"headroom_summary"`?

## experiment_6842_sealed_memory_pathway_portability_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Continuous self-learning is not ready (`continuous_self_learning_ready_score: 0.0`) because the calibrated-dose and held-future-benefit gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6843_live_arc_evidence_stratum_freeze.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims a complete read-only inventory of live ARC terminal evidence, with all inventory gates passing and no solve or mechanism-effect claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6844_supervisor_action_outcome_credit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The supervisor outcome-credit audit was blocked because the `headroom_nonzero` gate observed `false` when `true` was required.

## WHAT IS MISSING
nothing; `gate_check_summary` records `failed_check`, `observed`, and `expected`, while `per_game_results` provides per-action outcome rows.

## THE CHECK A READER CANNOT DO
none

## experiment_6845_tool_gap_causal_support_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The causal-support audit is blocked because the required tool-gap obligations were absent.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "tool_gap_obligations"`, with `"expected": ">0"` and `"observed": 0`.

## THE CHECK A READER CANNOT DO
none

## experiment_6846_typed_arc_shadow_monitor.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The typed ARC shadow monitor passed all readiness gates while remaining default-off, preserving action bytes, and matching external receipt facts on every recorded unit without making a solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6847_v598_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Chronological residual-memory shards completed, but memory underperformed no-memory and failed the held-future benefit, dose, and family-portability gates.

## WHAT IS MISSING
Per-unit comparison rows containing each game, seed, cell, or condition’s memory and no-memory metrics, headroom, dose, effect, and family are missing; the present `"rows"` are acceptance-criterion summaries, while `"gate_check_summary.failed_checks"` records only aggregates such as 540 rows, 5 wins, 69 losses, and mean effect −0.118519.

## THE CHECK A READER CANNOT DO
Did the negative pooled effect occur broadly across eligible units, or was it driven by a few outliers or units with no headroom?
