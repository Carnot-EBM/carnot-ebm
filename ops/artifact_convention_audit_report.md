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

## experiment_6824_selective_arbiter_cold_row_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Cold per-unit replay supports the claim that `selective_priority` outperformed `flat_reject_retry` and passed the positive acceptance gate.

## WHAT IS MISSING
nothing; `"rows"` contains per-unit arm metrics and `"gate_check_summary"` records every check’s expected and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_6825_selective_arbiter_authority_attacks.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Independent mutation results support the hard-authority boundary, while adoption was not evaluated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6826_selective_arbiter_sealed_adoption.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The sealed audits justify enabling the selective arbiter because it passes safety, preservation, certificate, and comparative utility criteria.

## WHAT IS MISSING
Per-pair `cold_replay_rows` containing each unit’s progress metric for both `flat_reject_retry` and `selective_priority`; the present `"rows"` are criterion-level summaries, while `"paired_progress_delta"` records only an aggregate estimate, interval, and pair count.

## THE CHECK A READER CANNOT DO
Did the 0.125 paired progress improvement occur broadly across the 144 pairs, or was it driven by a few outliers or units with no headroom?

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

## experiment_6831_v597_evidence_admissibility_contract.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The selective-arbiter authority and causal-edge inputs are procedurally admissible, with all gates passed and a positive utility effect of `paired_progress_delta: 0.125`.

## WHAT IS MISSING
Per-pair or per-unit utility rows containing each arm’s progress metric and paired delta; the present `"rows"` provide only aggregate `"observed"` values such as `"pair_count": 144` and `"paired_progress_delta": 0.125`.

## THE CHECK A READER CANNOT DO
Was the reported positive utility delta broad across the 144 pairs, or driven by a small number of outliers or units with unequal headroom?

## experiment_6832_operational_obligation_saturation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6833_sota_operational_obligation_saturation_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the artifact contains only `"MODEL_SPECS"` configuration and provenance metadata, with no comparative result or blocked verdict.

## THE CHECK A READER CANNOT DO
none

## experiment_6834_operational_saturation_identifiability_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All listed robustness attacks passed, and the detailed `"collision_witnesses"` demonstrate missing cells whose alternative values produce identical observed signatures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
