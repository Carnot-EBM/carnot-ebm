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
The deterministic source-free operational-obligation fixture is complete and ready, with all gates passed and no live-benefit or level-solve claim.

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
nothing; the artifact records receipt fields including `"cell_id"`, `"checkpoint_sha256"`, `"row_sha256"`, `"atomic"`, and `"resumed"`, but states no comparative result or blocked verdict.

## THE CHECK A READER CANNOT DO
none

## experiment_6813_selective_priority_arbiter_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Selective priority passed the positive gate on held exact replay, outperforming flat reject/retry in accepted progress without violating the stated safety conditions.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6823_v595_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The selective-priority arm passed its positive gate, outperforming flat-reject-retry by a mean paired-progress delta of 0.125 across 144 pairs.

## WHAT IS MISSING
The 144 per-pair metric rows underlying `"row_recomputed_claims.selective_arbiter"` are missing; `"rows"` contains only task and branch summaries, while `"paired_progress_delta"`, `"accepted_progress_by_arm"`, and `"false_intervention"` provide aggregates.

## THE CHECK A READER CANNOT DO
Were improvements broad across the 144 pairs, or driven by a few outliers while many controls were degenerate or had no headroom?

## experiment_6824_selective_arbiter_cold_row_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The cold row replay supports the positive verdict that `selective_priority` outperformed `flat_reject_retry` while satisfying all acceptance-gate conditions.

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
The selective-priority arm achieved a positive paired progress improvement over flat-reject-retry, supporting deployment adoption.

## WHAT IS MISSING
Per-unit rows for all 144 pairs, including each unit’s identifier, both arm-level progress values, and paired delta; the present `"rows"` are criterion summaries, while `"paired_progress_delta"`, `"harmful_selections_by_arm"`, and `"legal_support_by_model_and_arm"` contain only aggregates.

## THE CHECK A READER CANNOT DO
Did improvement occur broadly across the 144 pairs, or was the positive mean driven by a few outliers while many units were unchanged or had no headroom?

## experiment_6827_chronological_causal_edge_memory_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The frozen chronological causal-edge memory stream is complete and ready, and no learning experiment ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
