# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 4 |

## experiment_6519_structural_headroom_certificate.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims `static_analytical` is the best arm, saving 667 held charged-work units versus `native_dynamic`, and that all certification gates were met.

## WHAT IS MISSING
Actual `"per_unit_rows"` containing each unit’s arm-level charged-work metric and paired native-versus-arm effect are missing; only aggregates such as `"held_total_charged_work_units_by_arm"` and replay booleans in `"exact_receipt_replay_rows"` are present.

## THE CHECK A READER CANNOT DO
Was the 667-unit advantage broadly distributed across held units, or driven by a small number of extreme wins while most units tied or lost?

## experiment_6520_safety_net_branch_router_ab.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The learned `linear_router_seed_6520002` arm met the positive/ready gates with 695 held charged-benefit units, outperforming the upstream best structural arm by 28 units.

## WHAT IS MISSING
Per-unit held-evaluation rows containing each `unit_id` or problem seed, arm-level charged work, native/control charged work, and resulting benefit; only aggregate fields such as `"held_total_charged_work_units"`, `"held_charged_benefit_units"`, `"held_win_count"`, and `"held_loss_count"` are present, while `"candidate_preservation_rows"` contain no outcome metric.

## THE CHECK A READER CANNOT DO
Was the claimed 28-unit advantage broad across held units, or produced by one outlier while most units tied, lost, or had no headroom?

## experiment_6521_transactional_refinement_conflict_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The transactional refinement conflict memory passed all readiness gates, including exact replay, refinement gating, transactions, rollback, quarantine, fallback, and CPU mapping, with zero unsafe admissions or uses.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6522_chronological_conflict_self_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Continuous self-learning achieved positive charged held-future benefit beyond the scratch and frozen-memory controls while preserving exact-answer safety.

## WHAT IS MISSING
Populated `"per_game_results"` or `"per_unit_rows"` containing the observed benefit metric for every `"arm"` and `"event_id"`; the artifact shows aggregate booleans such as `"benefit_beyond_scratch_and_frozen_controls"` and `"charged_held_future_benefit_positive"`, while `"exact_answer_equality_rows"` only establish answer equality and `"chronological_stream_commitment.stream_rows"` record planned stream benefit scores rather than observed arm comparisons.

## THE CHECK A READER CANNOT DO
Was the claimed held-future advantage broad across events and chains, or produced by one outlier while the scratch and frozen controls were otherwise identical or had no headroom?

## experiment_6523_adaptive_validation_csl_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent full audit supports continuous self-learning, and adaptive validation saves charged checks while preserving the full-set decision.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6524_arc_supervisor_redirect_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The supervisor-refinement task was blocked because none of the three accepted live receipts was outcome-bearing.

## WHAT IS MISSING
nothing; `"honest_verdict"`, `"status"`, `"no_firings_receipt"`, `"gate_check_summary"`, and the three `"live_path_receipts"` identify the blocker and record the observed outcome-bearing receipt count as 0.

## THE CHECK A READER CANNOT DO
none

## experiment_6525_gatemate_changed_state_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no operator-authored, dated GateMate physical-state receipt newer than Exp6325 was found, so zero hardware commands ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6526_v564_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Structural headroom, learned routing, continuous self-learning, and adaptive validation produced positive comparative benefits, while ARC generalization and GateMate continuity remained blocked.

## WHAT IS MISSING
The actual per-game, per-seed, per-cell, or per-condition arm metrics are missing; `"row_support"` only names external containers and counts, while the included `"per_unit_rows"` contains summary `"comparative_claim"`, `"task_inventory"`, `"gate_contract"`, and audit rows rather than unit-level measurements.

## THE CHECK A READER CANNOT DO
Were the reported benefits broad across units, or driven by a few outliers, degenerate controls, or units with no headroom?
