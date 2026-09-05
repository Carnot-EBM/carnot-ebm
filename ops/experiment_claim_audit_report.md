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
| CLAIM_SUPPORTED | 6 |
| NO_CLAIM | 2 |

## experiment_7020_counterexample_belief_ledger.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The deterministic counterexample belief ledger is operationally ready on safety, replay, audit, and isolation, without claiming future utility.

## WHAT WOULD REFUTE IT
A failed gate, accepted future-field write, mutation after an authority conflict, eviction of protected facts, replay mismatch, failed recovery or rollback, or missing executable belief state would falsify readiness.

## WAS THAT CHECKED
Yes. The gate summary, leakage, authority-conflict, poison, retention, restart, rollback, and four-state checks exercised failure-capable paths; all reported the required outcomes. No comparative or generalization claim requires a rival baseline or held-out utility test.

## EVIDENCE
`belief_ledger_ready_score`: `1`; `observed_value`: `all checks pass`; `failed_check`: `null`; `construction_event_count`: `27`; `future_field_leakage_count`: `0`; `fresh_process_replay`: `true`; `reason`: `authority_conflict`; `state_unchanged`: `true`; `reason`: `capacity_protected`; `protected_state_unchanged`: `true`; `reason`: `forbidden_updater_fields`; `committed_restart_passed`: `true`; `parent_bytes_restored`: `true`; `verifier_is_oracle`: `false`; `inference_substrate`: `deterministic_arc_belief_ledger_replay_no_llm`; `honest_verdict`: `complete_positive_counterexample_belief_ledger_ready_no_future_utility_claim`

## RECOMMENDATION
KEEP

## experiment_7021_prospective_belief_utility.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Prospective utility from counterexample belief memory was not demonstrated.

## WHAT WOULD REFUTE IT
The counterexample-belief arm passing the predeclared positive gate: strictly beating both frozen and capacity-matched recency-only controls, with a nonnegative paired interval lower bound and no protected-retention regression.

## WAS THAT CHECKED
Yes. The frozen positive gate, paired comparisons, capacity checks, leakage checks, and terminal gate summary directly tested it; the serious recency-only baseline was not significantly beaten.

## EVIDENCE
`"honest_verdict"`: `"complete_null_prospective_belief_utility_not_demonstrated"`; `"belief_future_utility_positive_score"`: `0`; `"failed_check"`: `"belief_future_utility_positive_score"`; `"comparison"`: `"counterexample_belief_minus_recency_only"`; `"point_estimate"`: `0.111111111111`; `"lower"`: `-0.166666666667`; `"upper"`: `0.5`; `"same_capacity"`: `true`; `"current_outcome_used_by_policy"`: `false`; `"verifier_is_oracle"`: `false`.

## RECOMMENDATION
KEEP

## experiment_7022_belief_ledger_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The belief ledger is safe for bounded shadow use, but its prospective value is not established strongly enough for promotion.

## WHAT WOULD REFUTE IT
Either a failed safety control would refute shadow safety, or nonnegative paired-interval lower bounds against both controls would refute the claim that value remains non-promotable.

## WAS THAT CHECKED
Yes. The artifact tested leakage, replay, rollback, capacity, retention, conflict, and retrieval safety, and compared action-ranking accuracy against both frozen and recency-only controls. The serious recency-only comparison had a negative lower bound, so the promotion criterion genuinely could have passed but did not.

## EVIDENCE
`honest_verdict`: `complete_null_belief_ledger_shadow_safe_value_not_promotable`; `verdict_class`: `null`; `belief_shadow_safe_score`: `1`; `belief_promotion_ready_score`: `0`; `shadow_safety`: `passed`: `true`; `value_promotion`: `passed`: `false`; `counterexample_belief_minus_recency_only`; `point_estimate`: `0.111111111111`; `lower`: `-0.166666666667`; `upper`: `0.5`; `paired_interval_lower_bounds_nonnegative`; `observed_value`: `false`; `passed`: `false`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7023_belief_query_api.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The deterministic bounded, game-blind belief-query API is ready for interface use across the tested contract.

## WHAT WOULD REFUTE IT
Any valid fixture producing the wrong abstention behavior, exceeding its event or byte budget, leaking prohibited identity fields, omitting contradiction evidence, mutating durable state after an invalid request, or changing output across restart would refute readiness.

## WAS THAT CHECKED
Yes. The artifact checks named fixtures including conflict, semantic-tie, stale, empty, and truncated cases; explicit byte/event budgets; prohibited fields; mutation rejection; contradiction preservation; and restart stability. Each reported row is terminal and passed.

## EVIDENCE
`"honest_verdict": "complete_positive_bounded_game_blind_belief_query_api_ready"`; `"belief_query_api_ready_score": 1`; `"inference_substrate": "deterministic_bounded_belief_query_no_llm"`; `"verifier_is_oracle": false`; `"observed_value": "all checks pass"`; `"failed_check": null`; `"fixture": "semantic_tie"` with `"abstention_reason": "tied_active_evidence"`; `"fixture": "truncated_conflict"` with `"omitted_contradiction_event_count": 0`; `"fresh_process_match": true`; `"journal_unchanged": true`; `"state_unchanged": true`; `"verdict_class": "positive"`.

## RECOMMENDATION
KEEP

## experiment_7024_belief_aware_e3_selector.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The belief-aware E3 selector is correctly wired into the live policy path, can influence ranking when supported, and fails closed when evidence is unusable.

## WHAT WOULD REFUTE IT
An enabled live-path fixture where the selector is unreachable, the query does not fire, supported evidence cannot change ranking or the emitted action, rejected evidence changes ranking, or disabling the feature changes the frozen action trace.

## WAS THAT CHECKED
Yes. Factory and constructor wiring, live-path reachability, supported ranking influence, action provenance, abstention and malformed-evidence behavior, and disabled-path trace equivalence were checked. These observables could have disagreed with their expected behavior, so failure was genuinely possible. This establishes wiring readiness, not improved action quality.

## EVIDENCE
The artifact reports `honest_verdict` as `complete_positive_belief_aware_e3_selector_live_path_ready` and `verifier_is_oracle` as `false`. In `factory_to_selector_query`, `query_fired` and `ranking_changed` are `true`. The `supported` ranking changes from `base_actions` with action `1` first to `ranked_actions` with action `2` first, while `base_candidate_set_preserved` is `true`. The `enabled_action_row` records `belief_influence` as `true` and `belief_final_action` as action `2`. All listed `abstention_rows` and `malformed_evidence_rows` have `ranking_changed` as `false`. The `frozen_disabled_trace` has identical `expected_trace_sha256` and `observed_trace_sha256`. `shipped_default_unchanged` is `true`.

## RECOMMENDATION
KEEP

## experiment_7025_belief_shadow_live_trace.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked execution receipt, not a comparative or value claim. A successful owned live trace with populated execution and shadow rows would contradict only its blocked status.

## WAS THAT CHECKED
Yes, execution readiness was checked and failed at `live_trace_execution`; the method’s value was not tested because execution produced no result rows.

## EVIDENCE
`honest_verdict` `blocked_belief_shadow_live_trace:live_trace_execution` `verdict_class` `blocked` `belief_shadow_trace_ready_score` `0` `failed_check` `live_trace_execution` `passed` `false` `rows` `[]` `shadow_rows` `[]` `control_rows` `[]` `per_action_results` `[]`

## RECOMMENDATION
KEEP

## experiment_7026_held_mechanic_belief_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No claim is made to falsify; an A/B outcome showing the held mechanic loses or ties a serious baseline would refute a future positive comparative claim.

## WAS THAT CHECKED
No. The A/B experiment did not run; only prerequisite gates were checked.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_observed": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7027_v615_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V615 capstone is blocked because the required live evidence is absent.

## WHAT WOULD REFUTE IT
Completed live upstream runs—specifically a ready Exp7025 shadow trace and completed Exp7026—together with matched compute receipts and complete row-derived live A/B results would falsify the claimed evidence absence.

## WAS THAT CHECKED
Yes. `gate_check_summary` compares required and observed upstream states, while `compute_receipt_summary` and `live_ab_recomputation` check for matched live compute and per-game evidence. The refuting conditions were possible but did not occur.

## EVIDENCE
`honest_verdict`: `complete_blocked_v615_capstone_required_live_evidence_absent`; `verdict_class`: `blocked`; `failed_check`: `required_live_upstream_complete`; `exp7025.belief_shadow_trace_ready_score`: `0`; `exp7026.status`: `blocked`; `matched_live_compute`: `false`; `live_ab_per_game_row_count`: `0`; `missing_cells`: `all_planned_live_cells`; `promoted_claims`: `[]`.

## RECOMMENDATION
KEEP
