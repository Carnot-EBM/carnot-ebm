# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 3 |

## experiment_6522_chronological_conflict_self_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Valid reuse delivered positive charged held-future benefit beyond scratch and frozen controls while preserving exact-answer safety, earning a candidate score of 1.0.

## WHAT IS MISSING
The actual per-arm, per-event benefit outcomes—described as `"per_game_results"`, `"immediate_metric_rows"`, and `"held_future_support_rows"`—are missing; only aggregate conclusions appear in `"aggregate_row_recomputation"`, while `"exact_answer_equality_rows"` covers safety and `"chronological_stream_commitment.stream_rows"` records planned event benefit scores rather than observed arm comparisons.

## THE CHECK A READER CANNOT DO
Was the claimed advantage over scratch and frozen controls broad across held-future events and chains, or driven by one outlier while most units showed no improvement?

## experiment_6523_adaptive_validation_csl_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent full audit supports continuous self-learning, while adaptive validation reduces charged checks from 180 to 108 without changing the full-set winner or decision.

## WHAT IS MISSING
nothing; the artifact includes unit-level evidence in `"final_full_audit_rows"`, `"inclusion_probability_rows"`, `"exact_sentinel_rows"`, and `"adaptive_attack_matrix.rows"`, plus comparative summaries in `"cost_and_decision_agreement_rows"` and diagnostics in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6524_arc_supervisor_redirect_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Supervisor refinement was blocked because there were zero outcome-bearing live trajectory-supervisor receipts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6525_gatemate_changed_state_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no valid operator-authored, dated GateMate physical-state receipt newer than Exp6325 was found, so zero hardware commands ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6526_v564_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Structural headroom, learned routing, continuous self-learning, and adaptive validation produced positive comparative benefits, while ARC and GateMate remained blocked.

## WHAT IS MISSING
The actual per-game/per-seed/per-condition arm metrics underlying the positive comparisons are missing; `"comparative_claim_rows"` and `"per_unit_rows"` contain only claim summaries and `"row_support"` counts pointing to external containers such as `"per_game_results"` and `"independent_csl_row_recomputation"`, not those rows themselves.

## THE CHECK A READER CANNOT DO
Were the reported comparative benefits broad across units, or driven by a few outliers or units with unequal/no headroom?

## experiment_6527_v565_evidence_eligibility_corrigendum.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V565 evidence root is eligible because the adopted structural-headroom, learned-router, conflict-memory, continuous-self-learning, and adaptive-validation claims are ready.

## WHAT IS MISSING
The underlying per-game/per-seed arm metrics from the referenced `"per_game_results"` and `"per_unit_rows"` containers are missing; the present `"per_unit_rows"` contains only task summaries, eligibility rows, attacks, receipts, and hashes such as `"observed_value": 1.0` and `"row_count"`, not the experimental measurements supporting the comparative claims.

## THE CHECK A READER CANNOT DO
Did the learned router beat structural headroom broadly across games and seeds, or was the reported advantage caused by outliers, degenerate controls, or units without headroom?

## experiment_6528_v565_source_model_method_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V565 source-model method contract is blocked because required OpenReview primary sources could not be verified.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` identifies `"required_primary_sources_verified"` as false and records the affected source IDs, expected condition, and observed retrieval values.

## THE CHECK A READER CANNOT DO
none

## experiment_6530_external_constraint_corpus_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The external-constraint corpus audit was blocked because seven named prerequisites failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
