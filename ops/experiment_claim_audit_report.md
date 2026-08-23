# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 3 |
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6519_structural_headroom_certificate.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent row recomputation certifies real structural-control headroom: static analytical control reduces charged work while preserving correctness, live influence, breadth, and replay validity.

## WHAT WOULD REFUTE IT
The static analytical arm tying or exceeding native dynamic—or the serious analytical-enumeration baseline—in held charged work, losing correctness, showing no live decision influence, or depending on censored/invalid rows would refute the claim.

## WAS THAT CHECKED
Yes. The held charged-cost comparison includes native dynamic and analytical enumeration; the certificate also gates correctness, exact replay, live influence, breadth, censoring, and row recomputation. Static analytical wins rather than ties, including against the serious analytical-enumeration baseline.

## EVIDENCE
`best_arm`: `static_analytical`; `best_arm_held_charged_benefit_units`: `667`; `held_total_charged_work_units_by_arm`; `static_analytical`: `3313`; `analytical_enumeration`: `3366`; `native_dynamic`: `3980`; `correctness_discrepancy_count`: `0`; `live_influence_passed`: `true`; `exact_replay_passed`: `true`; `uncensored_held_rows`: `42`; `worst_case_missing_rows`: `0`; `censored_count`: `0`; `timeout_count`: `0`; `best_arm_support_family_count`: `3`; `best_arm_support_seed_count`: `3`; `certification_conditions_met`: `true`.

## RECOMMENDATION
KEEP

## experiment_6520_safety_net_branch_router_ab.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6521_transactional_refinement_conflict_memory.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The transactional exact-conflict memory is ready because refinement gating and exact replay produced zero unsafe admissions or uses while all lifecycle and recovery checks passed.

## WHAT WOULD REFUTE IT
An independent correctness oracle finding that a verifier-accepted record was unsafe—especially one admitted or used after passing exact replay—would refute the readiness claim.

## WAS THAT CHECKED
No. Invalid-reuse and failure-path rows were exercised, but correctness was defined by the same exact-replay verifier that gates acceptance; no independent oracle could disagree with it.

## EVIDENCE
`verifier_is_oracle` `true` `verdict_class` `circular_positive` `exact_replay_required_before_use` `true` `exact_replay_required_before_write` `true` `unsafe_admission_count` `0` `unsafe_use_count` `0` `complete_transactional_refinement_conflict_memory_ready`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6522_chronological_conflict_self_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Chronological exact-conflict self-learning reduces charged held-future cost versus scratch and frozen-empty controls while preserving exact answers, prefix support, and safe operation.

## WHAT WOULD REFUTE IT
A valid reuse arm tying or exceeding the controls’ held-future charged cost—making its charged benefit zero or negative—or any exact-answer mismatch, unsafe reuse/write, or prefix-retention failure would falsify the claim.

## WAS THAT CHECKED
Yes. Matched-dose scratch and frozen-empty arms provide direct cost comparators; held-future rows measure both unbounded and bounded reuse against them. Separate rows check exact answers, unsafe activity, prefix retention, chronology, and terminal completeness.

## EVIDENCE
`valid_unbounded_reuse` has `charged_cost` `40` and `charged_benefit_vs_scratch` `84`; `valid_bounded_reuse` has `charged_cost` `44` and `charged_benefit_vs_scratch` `80`. Both `scratch` and `frozen_empty_memory` have `charged_cost` `124`. The artifact records `matched_dose` as `true`, `held_future_unread_until_boundary` as `true`, `thresholds_frozen_before_execution` as `true`, `uses_future_outcomes_for_stream` as `false`, `exact_answer_equality` as `true`, `unsafe_use_count` as `0`, `unsafe_write_count` as `0`, `prefix_retention_within_margin` as `true`, and `all_planned_rows_terminal` as `true`.

## RECOMMENDATION
KEEP

## experiment_6523_adaptive_validation_csl_audit.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['; under ', ', both have ', '. Nevertheless, ', ', versus ', ', with '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The independent full audit supports CSL, while adaptive validation reduces checks without changing the full-set decision.

## WHAT WOULD REFUTE IT
A serious comparator tying or beating the claimed CSL winner on held-future benefit and charged cost would refute the implied added-value claim; adaptive validation would be refuted by a different decision or no check savings.

## WAS THAT CHECKED
Yes. The final full audit compared all candidate arms and found that `restart` exactly tied `valid_unbounded_reuse`. The adaptive arm was separately compared with full-set validation and did save checks while agreeing, so only that narrower result survives.

## EVIDENCE
The `winner_tie_set` contains `valid_unbounded_reuse` and `restart`. Under `benefit_vs_scratch`, both have `84`; under `charged_cost_totals`, both have `40.0`. Nevertheless, `full_audit_winner` is `valid_unbounded_reuse` and `verdict_class` is `positive`. For validation cost, `variance_weighted_adaptive` has `charged_checks` of `108`, versus `180` for `full_set`, with `decision_agreement_with_full` equal to `true`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_6524_arc_supervisor_redirect_generalization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no positive comparative or value claim; outcome-bearing live receipts would be required to test whether any supervisor arm improves outcomes.

## WAS THAT CHECKED
No. All three accepted receipts were non-outcome-bearing, and there were no redirect outcome rows; the artifact correctly reports itself as blocked.

## EVIDENCE
`honest_verdict` `blocked: missing outcome-bearing live trajectory-supervisor receipts` `outcome_bearing_receipt_count` `0` `redirect_outcome_rows` `[]` `solve_credit_claimed` `false` `verdict_class` `blocked`

## RECOMMENDATION
KEEP

## experiment_6525_gatemate_changed_state_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made; operationally, a valid post-Exp6325 physical-state receipt or any executed hardware command would contradict the reported blocked status.

## WAS THAT CHECKED
Yes. The receipt audit examined 24 candidates, found zero valid receipts, and the command rows recorded zero commands.

## EVIDENCE
`hardware_speedup_claim`: `false`; `no_performance_or_availability_claim`: `true`; `receipt_candidate_count`: `24`; `valid_receipt_count`: `0`; `command_row_count`: `0`; `status`: `blocked_missing_new_physical_receipt`; `verdict_class`: `blocked`

## RECOMMENDATION
KEEP

## experiment_6526_v564_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Four row-supported lineages are positive, while ARC generalization and GateMate continuity remain blocked.

## WHAT WOULD REFUTE IT
A held-row recomputation in which the best certified structural arm tied or beat the learned router would refute the router component—and therefore the composite headline.

## WAS THAT CHECKED
Yes. The learned-router comparison used the best structural arm and reports 28 held units of additional benefit, with held contamination excluded. The decision process also preserved two failed lineages instead of forcing every lineage positive.

## EVIDENCE
`held_benefit_beyond_best_structural_units`: `28`; `held_contamination_free`: `true`; `authority`: `oracle_distinct_row_reduction`; `positive_lineage_count`: `4`; `blocked_lineage_count`: `2`; `blocked_lineages`: `arc_generalization`, `hardware_continuity`; `outcome_bearing_receipt_count`: `0`; `new_post_exp6325_physical_receipt_found`: `false`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
