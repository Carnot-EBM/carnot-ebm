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
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 1 |

## experiment_7252_v638_semantic_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A semantic-value claim would be refuted if a serious direct or offset comparator tied or won, fidelity or coverage missed its threshold, selective error failed to improve, or contradiction controls were falsely accepted.

## WAS THAT CHECKED
No. The semantic criteria were not evaluated: the upstream artifact was missing, no units or rows were completed, and no model was invoked. Only the blocked prerequisite status was checked.

## EVIDENCE
`honest_verdict`: `blocked_exp7252_missing_upstream_artifact`; `status`: `blocked`; `verdict_class`: `blocked`; `inference_substrate`: `blocked_no_run`; `observed_value`: `missing_artifact`; `evaluated`: `false`; `observed`: `null`; `rows`: `[]`; `attempted_independent_units`: `0`; `completed_independent_units`: `0`; `model_invoked`: `false`; `semantic_audit_complete_score`: `0`; `semantic_value_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_7253_v638_coverage_memory.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no learning or comparative-value claim to falsify. Interpreting fixture readiness as an operational claim, it would be refuted by a memory-cap breach, a failed controller control, incomplete arm rows, or a shuffle diagnostic that did not change the selected snapshot.

## WAS THAT CHECKED
Yes, for operational readiness: the acceptance gates check bounded memory, seven controller controls, 256 eight-arm rows, and an identity-changing shuffle diagnostic. Learning gain and causal attribution were explicitly not assessed.

## EVIDENCE
`science_learning_gain_scored`: `false`; `causal_attribution_eligible`: `false`; `science_learning_gain`: `not_scored`; `This phase builds a fixture and does not score learning gain.`; `coverage_fixture_ready_score`: `1`; `bounded_memory`; `pass`: `true`; `controller_controls`; `observed`: `7`; `eight_arm_rows`; `observed`: `256`; `selection_changed`: `true`

## RECOMMENDATION
KEEP

## experiment_7254_v638_coverage_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded coverage memory did not pass every frozen gate.

## WHAT WOULD REFUTE IT
Every frozen scientific gate passing, yielding a coverage-learning value score of 1.

## WAS THAT CHECKED
Yes, in `acceptance_gate_results`; several gates were evaluated and failed. The oracle-defined correctness does not circularly support a positive value claim here because the reported claim is a measured null.

## EVIDENCE
`"honest_verdict": "complete_null: bounded coverage memory did not pass every frozen gate"`; `"coverage_learning_value_score": 0`; `"effective_prospective_shuffle_intervention"` has `"pass": false`; `"future_error_vs_reset_upper_ci95_lt_zero"` has `"pass": false`; `"recurrence_error_vs_coverage_shuffled_upper_ci95_lt_zero"` has `"observed": 0.0` and `"pass": false`; `"verdict_class": "null"`.

## RECOMMENDATION
KEEP

## experiment_7255_v638_coverage_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The coverage audit completed, but the method failed the criteria required for promotion.

## WHAT WOULD REFUTE IT
A complete audit in which every preregistered promotion criterion passed and the promotion score was `1`.

## WAS THAT CHECKED
Yes. The separate `acceptance_gate_results` include observed values and pass states; several criteria failed, including comparisons against reset, frozen, destructive, and shuffled controls. The oracle verifier does not circularly support a positive value claim because the reported verdict is null.

## EVIDENCE
`honest_verdict` `complete_null: coverage audit completed but promotion criteria did not all pass`; `coverage_audit_complete_score` `1`; `coverage_promotion_score` `0`; `effective_prospective_shuffle_intervention` `pass` `false`; `future_error_vs_reset_upper_ci95_lt_zero` `pass` `false`; `recurrence_error_increase_vs_frozen_lte_0_02` `observed` `0.1318359375` `pass` `false`; `recurrence_error_vs_coverage_shuffled_upper_ci95_lt_zero` `observed` `0.0` `pass` `false`; `verdict_class` `null`; `verifier_is_oracle` `true`

## RECOMMENDATION
KEEP

## experiment_7256_v638_native_controller.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['; the parity rows report '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The persistent native controller is ready because it achieved exact oracle parity and persistent native ownership.

## WHAT WOULD REFUTE IT
Any semantic mismatch, non-identical rollback, failed restart decision/state check, or nonzero hot-path parsing or active reconstruction would refute the measured component claims; a serious existing-controller baseline tying or winning on end-to-end throughput or cost would refute the broader readiness/value claim.

## WAS THAT CHECKED
No. Semantic parity, restart behavior, rollback, and ownership counters were checked and could fail, but the broader value conclusion remained oracle-defined and throughput was explicitly deferred; the old native wrapper was not compared on throughput or cost.

## EVIDENCE
`honest_verdict` is `complete_circular_positive: exact oracle parity and persistent native ownership passed; throughput deferred to Exp7257`. `verifier_is_oracle` is `true`. `verdict_class` is `circular_positive`. `native_controller_ready_score` is `1`. `throughput_value_score` is `null`. `model_invoked` is `false`. For `persistent_native_controller`, `active_reconstruction_count` and `hot_path_json_parse_count` are both `0`; the parity rows report `mismatch_count` as `0` and `rollback_byte_identical` as `true`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7257_v638_native_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Complete event measurement found no accepted persistent-native interactive cost value.

## WHAT WOULD REFUTE IT
Both interactive-capacity lower confidence bounds exceeding 1 while exact parity and equal durability hold, producing a native event cost value score of 1.

## WAS THAT CHECKED
Yes. The acceptance gates tested both interactive capacities, exact parity, and equal durability across the complete 540-row roster; capacity four failed even though capacity one passed.

## EVIDENCE
`"honest_verdict": "complete_null: complete event measurement found no accepted persistent-native interactive cost value"`; `"interactive_capacity_one"` with `"observed": true`; `"interactive_capacity_four"` with `"observed": false`; `"interactive_persistent_lower_ci95_min": 0.9817458323703578`; `"native_event_cost_value_score": 0`; `"equal_durability": true`; `"parity_failure_count": 0`; `"safety_failure_count": 0`; `"completed_independent_block_arm_units": 540`; `"censored_independent_block_arm_units": 0`; `"verdict_class": "null"`.

## RECOMMENDATION
KEEP

## experiment_7258_v638_board_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made. For the receipt’s descriptive assertion, refutation would be fewer than three authenticated board dispositions, a missing exact next condition, mislabeling PolarFire CPU dispatch as FPGA sampling, or any hardware operation issued during this invocation.

## WAS THAT CHECKED
Yes, as receipt consistency checks in `board_rows`, `acceptance_gate_results`, `sample_size_budget`, and `hardware_operations_issued`; no method-versus-rival test was applicable.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts`; `model_invocation_count`: `0`; `hardware_operations_issued`: `[]`; `board_disposition_complete_score`: `1`; `One records three authenticated dispositions and exact next conditions, not three working samplers.`

## RECOMMENDATION
KEEP

## experiment_7259_v638_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The fourteen-task matrix is complete, but held-out mention and live ARC evidence are absent, coverage learning shows no value over controls, and durable native-cost value is null.

## WHAT WOULD REFUTE IT
A matrix count other than 14; an authenticated held-out mention capture; successful live ARC policy consumption with a non-null score; coverage learning beating matched reset and shuffled controls; or durable native performance meeting parity and the 10× target.

## WAS THAT CHECKED
Yes. Matrix size and external-science availability were explicit acceptance gates; held-out and ARC prerequisites were replayed; coverage was assessed against reset and shuffled controls; and native cost was tested against parity and the 10× target. Invalid, missing, and oracle-based rows were not promoted as positive evidence.

## EVIDENCE
`"fourteen_task_matrix_complete"`, `"expected": 14`, `"observed": 14`, `"passed": true`; `"external_science_available"`, `"observed": false`, `"passed": false`; `"results/experiment_7251_v638_mention_heldout.json is absent"`; `"arc_witness_ready_score"`, `"expected_value": 1`, `"observed_value": 0`; `"The learner did not beat reset and shuffled controls or meet recurrence retention."`; `"value_observed": false`; `"One interactive capacity CI crossed below parity, and the 10x target failed."`; `"durable_cost_value": false`; `"positive_promoted": false`; `"verdict_class": "blocked"`.

## RECOMMENDATION
KEEP
