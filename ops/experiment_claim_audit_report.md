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
| CLAIM_SUPPORTED | 2 |
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 4 |

## experiment_6530_external_constraint_corpus_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. Treating the blocked status as an operational claim, it would be refuted if the intake artifact, fixture, and pinned source existed; the revision, license, schema, and corruption boundary were verified; and all gates passed.

## WAS THAT CHECKED
Yes. The existence receipts, revision-and-license receipt, and gate summary explicitly checked those preconditions and recorded their failures. No corpus performance or verifier-added-value claim was tested.

## EVIDENCE
`status`, `blocked_external_constraint_corpus_audit`, `verdict_class`, `blocked`, `all_gates_passed`, `false`, `fixture_row_count`, `0`, `blocked_preconditions`, `true`, `external_constraint_corpus_audited_ready_score`, `0.0`

## RECOMMENDATION
KEEP

## experiment_6541_v566_direct_source_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a readiness receipt, not a method-value claim. Its operational conclusion would be refuted by any required source, replay, cache, split, field, or dependency check failing.

## WAS THAT CHECKED
Yes. The artifact reports per-requirement rows, aggregate recomputation, and a failed-check summary; no model or comparative outcome was evaluated.

## EVIDENCE
`complete_v566_direct_source_contract_ready: direct DRIFT source, local replay, model cache, split, field, and dependency contracts are implementable`; `States direct-source readiness without declaring a scientific result.`; `direct_primary_source_cache_and_dependency_preflight_no_llm`; `model_loaded_or_run`; `false`; `all_gates_passed`; `true`; `failed_checks`; `[]`

## RECOMMENDATION
KEEP

## experiment_6542_drift_bench_external_intake_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s operational readiness assertion, a source-hash mismatch, cross-split family overlap, missing or censored replay row, rejected assignment, corrupt shard accepted, failed leakage attack, or fixture round-trip mismatch would refute completion. None would refute a scientific value claim because none is made.

## WAS THAT CHECKED
Yes. Those failure modes were checked in the source comparisons, replay rows, split commitment, shard corruption probe, leakage attack matrix, terminal-unit accounting, fixture round-trip, and aggregate recomputation. The oracle-defined Z3 results establish execution receipts only; no added-value claim is advanced.

## EVIDENCE
`"positive_scientific_class_declared"`: `false`; `"verdict_class"`: `null`; `"inference_substrate"`: `"content_pinned_drift_intake_and_local_z3_replay_no_llm"`; `"verifier_is_oracle"`: `true`; `"all_gates_passed"`: `true`; `"failed_checks"`: `[]`; `"base_problem_overlap_count"`: `0`; `"censored_count"`: `0`; `"corrupt_resume_rejected"`: `true`; `"roundtrip_matches_expected"`: `true`

## RECOMMENDATION
KEEP

## experiment_6543_external_corpus_independent_audit_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative claim to falsify; the narrower audit-readiness receipt would fail if any source, chronology, split, replay, transaction, aggregate, or attack check failed.

## WAS THAT CHECKED
Yes, for audit readiness: per-row checks, aggregate recomputation, and gate checks could record failures. No value or comparative claim was made or tested.

## EVIDENCE
`"verdict_class": null`, `"Names audit readiness without declaring a scientific result."`, `"True only for audit checks; the artifact makes no positive scientific class."`, `"all_gates_passed": true`, `"failed_checks": []`, `"all_audit_rows_passed": true`

## RECOMMENDATION
KEEP

## experiment_6544_external_structural_headroom.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Analytical ordering provides positive charged-work value over native and random ordering on held data across three families.

## WHAT WOULD REFUTE IT
Analytical ordering tying or costing more than a serious deterministic baseline that simply checks the deepest/current-turn full-state prefix first under identical overhead accounting would refute added analytical value; totals at or above native or random would also refute the stated comparison.

## WAS THAT CHECKED
No, not in a genuinely falsifiable way. The candidate set always includes the current target prefix, analytical ordering selects the largest prefix first, and evaluation stops upon reaching that full target state. Native chronological and random ordering therefore cannot seriously challenge the method. The nominally closest arm, one-shot enumeration, was not an identical deepest-first comparator with identical overhead accounting.

## EVIDENCE
`primary_metric`: `held_total_charged_work_units_beyond_native_and_random`; `candidate_set_definition`: `all cumulative prefix states from turn zero through the current target turn`; `analytical`: `Largest structural prefix first, using constraint and assertion counts.`; `stop_rule`: `stop_after_full_target_state_or_native_fallback`; `native_exact_fallback_required`: `true`; `one_shot_enumeration`: `Enumerate the structurally largest prefix first, then descending prefixes.`; `held_total_charged_work_by_arm`: `native`: `2532`, `random`: `2532`, `analytical`: `1986`; `headroom_pair_count`: `54`; `no_headroom_pair_count`: `0`; `verifier_is_oracle`: `false`; `held_rows_used_for_calibration`: `false`; `learned_parameters`: `false`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6545_external_safety_net_router.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The frozen selected safety-net router lowers held charged cost versus the certified structural control while preserving exact equality, calibrated abstention, immutable train-only exceptions, and reachable exact fallback.

## WHAT WOULD REFUTE IT
A held charged cost at least as high as the certified structural control, a nonpositive paired effect, any exact-result mismatch, held-data leakage, mutable held exceptions, or unavailable fallback would falsify the claim.

## WAS THAT CHECKED
Yes. The comparison rule was frozen before held evaluation; held rows were excluded from fitting, calibration, and model selection; costs were recomputed from per-unit rows; and equality, fallback, exception-table, shortcut, and rollback gates were checked. The certified structural control is a serious safety-equivalent baseline. Cheaper plain-router arms are marked ineligible for the claimed safety-net package and therefore do not refute this specific comparison.

## EVIDENCE
`selection_rule`: `eligible_arm_must_beat_certified_structural_control_on_held_charged_cost`; `selection_rule_frozen_before_held`: `true`; `held_rows_used_for_fitting`: `false`; `held_rows_used_for_calibration`: `false`; `held_rows_used_for_model_selection`: `false`; `linear_compact_router_abstention_exception_exact_fallback`: `1965.0`; `exp6544_certified_structural_control`: `1986.0`; `held_effect_vs_certified_control_units`: `21.0`; `paired_unit_count`: `54`; `all_costs_recomputed_from_rows`: `true`; `all_exact_equal`: `true`; `mismatch_count`: `0`; `native_exact_fallback_reachable`: `true`; `verifier_is_oracle`: `false`; `linear_compact_router`: `eligible_safety_net_arm`: `false`.

## RECOMMENDATION
KEEP

## experiment_6546_smt_cost_guard_sota.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Guarded exact dispatch reduces held token or time cost without worse exact completion across at least two mandated model families.

## WHAT WOULD REFUTE IT
Fewer than two model families showing both a cost benefit and non-inferior exact completion—for example, guarded tokens and time both equaling or exceeding unguarded, or a negative exact-completion delta. An always-Z3 arm tying or beating guarded dispatch would also refute the guard’s added-value interpretation.

## WAS THAT CHECKED
No. Guarded versus unguarded was computed, but the success path was structurally favored: the corpus included a row above the dispatch threshold, routed exact-Z3 calls incur no model-token charge, token OR time savings sufficed, and exact Z3 replacement cannot be less correct than the model under the stated evaluation. The cheapest serious control—always dispatch to Z3—was not tested as an arm.

## EVIDENCE
`route_rule`: `z3_direct_when_conflict_count_ge_threshold`; `conflict_threshold`: `1`; `conflict_count`: `40`; `direct_tool`: `z3_checker.py exact satisfiability replay`; `token_accounting`: `embedded GGUF tokenizer or llama.cpp usage only`; `guarded_direct_tool_rows`: `2`; `unguarded_total_charged_tokens`: `4636`; `guarded_total_charged_tokens`: `3029`; `exact_completion_delta`: `0`, `1`, `1`; `z3_evaluation_authority`: `true`; `token_or_time_benefit`: `true`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6547_external_transfer_independent_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent row-level auditing supports adopting both the router and cost-guard lanes.

## WHAT WOULD REFUTE IT
A nonpositive held router effect against the certified structural control, guarded cost equal to or worse than unguarded cost, inferior exact completion, held-data fitting or selection, or failed row recomputation would refute the claim.

## WAS THAT CHECKED
Yes. The artifact compares the selected router with a certified structural control on held rows, compares guarded with unguarded execution, checks exact-completion noninferiority, audits held-data exclusion, and recomputes costs and effects from rows. The oracle defines audit correctness, but the artifact does not claim that the verifier itself adds value.

## EVIDENCE
`selected_eligible_arm`: `linear_compact_router_abstention_exception_exact_fallback`; `held_effect_vs_certified_control_units`: `21.0`; `held_rows_used_for_fitting`: `false`; `held_rows_used_for_model_selection`: `false`; `held_rows_used_for_calibration`: `false`; `guarded_token_savings_total`: `1607`; `guarded_time_savings_total_s`: `5.569483`; `exact_completion_noninferior`: `true`; `cost_recomputation_passed`: `true`; `exact_equality_passed`: `true`; `router`: `adopted_passed`; `cost_guard`: `adopted_passed`; `verifier_is_oracle`: `true`; `True only for audit checks; the artifact makes no new positive scientific claim.`

## RECOMMENDATION
KEEP
