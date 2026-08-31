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

## experiment_6811_operational_obligation_automaton_v3.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic source-free operational-obligation fixture is ready, with all readiness gates passed and no live-benefit or level-solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6812_sota_operational_handoff_corpus_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the artifact contains receipt fields such as `"cell_id"`, `"checkpoint_sha256"`, `"row_sha256"`, `"atomic"`, and `"resumed"`, but no comparative claim or blocked verdict.

## THE CHECK A READER CANNOT DO
none

## experiment_6813_selective_priority_arbiter_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Selective priority passed the positive gate on held exact replay, outperforming flat reject-retry on accepted progress.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6823_v595_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The selective-priority arm beat flat-reject-retry by 0.125 mean paired progress and passed its positive gate, although the overall synthesis remains partial and branches are blocked by named missing evidence.

## WHAT IS MISSING
Per-unit paired rows for the 144 comparisons, including each unit’s identifier, arm-level progress values, retry costs, false-intervention outcome, and eligibility status; `"row_recomputed_claims"` provides only aggregates, while `"rows"` contains task/branch summaries rather than experimental-unit metrics.

## THE CHECK A READER CANNOT DO
Was the reported 0.125 improvement broad across the 144 pairs, or driven by a small number of outliers or units with unequal headroom?

## experiment_6824_selective_arbiter_cold_row_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The cold row replay supports the positive verdict that `selective_priority` outperformed `flat_reject_retry` and passed the acceptance gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6825_selective_arbiter_authority_attacks.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Independent mutation results support the hard authority boundary, while adoption was not evaluated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6826_selective_arbiter_sealed_adoption.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The selective-priority arm improved paired progress over flat-reject-retry sufficiently to pass utility and enable deployment adoption.

## WHAT IS MISSING
Per-pair progress metrics or deltas for all 144 units are missing; `"rows"` contains only criterion-level summaries, while `"paired_progress_delta"` records only `"estimate"`, bounds, and `"pair_count"`.

## THE CHECK A READER CANNOT DO
Did improvement occur broadly across the 144 pairs, or was the positive pooled estimate driven by a few outliers while most pairs were unchanged or worse?

## experiment_6827_chronological_causal_edge_memory_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The frozen chronological causal-edge memory stream is complete and ready, and no learning ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
