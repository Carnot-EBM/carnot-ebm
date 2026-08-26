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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6611_live_arc_invariant_projection.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Selected invariant projection had no effect on held exact-next-frame prediction error, with no game- or level-solve claim.

## WHAT WOULD REFUTE IT
A different charged exact-mismatch total or mean for `selected_invariant_projection` versus `no_projection`, or a projection-induced runtime-validity difference, would refute the no-effect claim.

## WAS THAT CHECKED
Yes. The held-arm and runtime-validity summaries compare the selected projection directly with no projection across 52 rows per arm, retain invalid rows, and charge failures; both arms have identical mismatch and validity results. The observed next frames were opened only after prediction, so the null result was not forced by candidate access to held outcomes.

## EVIDENCE
`honest_verdict`: `complete_held_world_model_prediction_projection_no_effect_no_solve_claim`; `verdict_class`: `null`; `candidate_held_win`: `false`; `strictly_lower_than_no_projection`: `false`; `arm`: `no_projection`; `arm`: `selected_invariant_projection`; `charged_exact_mismatch_total`: `9376`; `charged_exact_mismatch_mean`: `180.30769230769232`; `row_count`: `52`; `runtime_invalid_count`: `2`; `invalid_rows_retained`: `true`; `selected_runtime_validity_loss`: `0`; `observation_opened_after_prediction`: `true`; `held_outcomes_used`: `0`; `game_disjoint`: `true`

## RECOMMENDATION
KEEP

## experiment_6612_spectral_k_block_scale_rust_parity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact explicitly withholds stationary, transition, wall-time, and parity conclusions rather than asserting a positive comparative result.

## WAS THAT CHECKED
No affirmative claim required refutation; the blocking condition is documented by an incomplete test suite and a failed gate.

## EVIDENCE
`honest_verdict`: `blocked_reference_parity_cost_or_protection: stationary, transition, wall, and parity conclusions are withheld; CPU software only`; `verdict_class`: `blocked`; `status`: `blocked_reference_parity_cost_or_protection`; `tests_complete`: `false`; `gate`: `toolchain_or_test_failure`; `exit_code`: `3`

## RECOMMENDATION
KEEP

## experiment_6613_invariant_memory_lifecycle.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation can refute a nonexistent utility or comparative claim; a conformance failure would only refute the lifecycle receipt.

## WAS THAT CHECKED
No utility comparison was attempted; the artifact checks only lifecycle conformance through transition, attack, recovery, integrity, and test receipts.

## EVIDENCE
`honest_verdict`: `complete_invariant_memory_lifecycle_conformance_no_utility_claim`; `no_utility_claim`: `true`; `utility_claimed`: `false`; `verdict_class`: `null`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6614_prospective_invariant_self_learning.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6605_qwen36_direct_headroom.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The direct Qwen baseline’s integrity did not complete because GPU process receipts failed.

## WHAT WOULD REFUTE IT
A complete integrity recomputation with the GPU receipt check passing and all GPU sessions authenticated would refute the blocked-integrity claim.

## WAS THAT CHECKED
Yes. The gate, integrity recomputation, and GPU session receipts explicitly checked this; the GPU receipt check failed and the sessions were not all authentic.

## EVIDENCE
`"honest_verdict": "blocked_gpu_process_receipts: direct baseline integrity did not complete"`; `"all_passed": false`; `"failed_condition": "gpu_process_receipts"`; `"gpu_process_receipts": false`; `"complete": false`; `"failed_checks": ["gpu_process_receipts"]`; `"all_sessions_authentic": false`

## RECOMMENDATION
KEEP

## experiment_6615_v576_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The available evidence replays to blocked or partial decoding and sampler results, null projection and self-learning utility, and conformant lifecycle behavior without a positive scientific result.

## WHAT WOULD REFUTE IT
A nonzero held-future benefit over the serious static-projector baseline, a nonzero selected-projection effect over no projection, or a failed lifecycle invariant would refute the corresponding headline conclusions.

## WAS THAT CHECKED
Yes. Held-future comparisons used static and shuffled controls across 36 pairs; projection used no-projection and random-projection controls; and lifecycle transitions, chronology, restart, rollback, recovery, and weight immutability were checked. The verifier is the conformance oracle, but the artifact claims conformance—not independent scientific value from that verifier.

## EVIDENCE
`held_future_benefit_over_static`: `0.0`; `held_future_benefit_over_shuffled`: `0.0`; `held_future_pairs`: `36`; `selected_minus_no_projection_mean_error`: `0.0`; `selected_minus_random_mean_error`: `0.0`; `all_lifecycle_transitions_passed`: `true`; `scientific_verdict_class`: `null`; `verdict_class`: `partial`; `oracle_defined_win_promoted_to_positive`: `false`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6616_v577_execution_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports contract readiness rather than scientific or comparative value. Its administrative blocked status would be contradicted if all 13 expected task contracts were present and all gate checks passed.

## WAS THAT CHECKED
Yes. `exact_task_set` and `model_policy` explicitly checked completeness and recorded missing tasks and failed checks.

## EVIDENCE
`honest_verdict`: `blocked_roadmap_contract_incomplete: expected Exp6616-Exp6628 YAML contracts are not all present`; `verdict_class`: `blocked`; `execution_contract_ready_score`: `0.0`; `all_passed`: `false`; `expected_task_count`: `13`; `yaml_task_count`: `3`; `verifier_is_oracle`: `true`; `v577_roadmap_evidence_and_phase_receipt_contract_no_llm`; `The verdict reports contract readiness without claiming scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_6617_gpu_lease_phase_receipts.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt and reports no experimental, comparative, or value claim to falsify.

## WAS THAT CHECKED
No. The method was not evaluated because execution stopped at the conductor pre-gate; the artifact contains only the failed gate result.

## EVIDENCE
`status` `blocked`  
`honest_verdict` `blocked_gate_check_failed`  
`failed_field` `execution_contract_ready_score`  
`failed_expected` `1.0`  
`failed_observed` `0.0`  
`passed` `false`  
`blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP
