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
| CLAIM_SUPPORTED | 5 |
| NO_CLAIM | 3 |

## experiment_7413_v650_source_calibration.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Source-aware calibration produced no registered decision benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted if all registered benefit gates passed: statistically better Brier scores against the serious logistic, response-only, and training-prevalence baselines; non-worse mean log loss; and certified coverage of at least 0.25.

## WAS THAT CHECKED
Yes. The explicit scientific-benefit reduction tested those criteria on the `final_test` partition. Brier improvements passed, demonstrating that success was possible, but non-worse log loss and minimum coverage failed; coverage was zero for every arm.

## EVIDENCE
`"honest_verdict": "complete_null_source_calibration_no_registered_decision_benefit"`; `"registered_scientific_value"`; `"observed": false`; `"passed": false`; `"calibration_value_score": 0`; `"three_simultaneous_brier_upper_bounds_below_zero": true`; `"non_worse_mean_log_loss": false`; `"certified_coverage_at_least_0_25": false`; `"coverage": 0.0`; `"full_registered_benefit_passed": false`; `"partition": "final_test"`; `"training_prevalence"`; `"l2_logistic_six_input"`; `"verifier_is_oracle": false`; `"scored": false`; `"brier_contribution": null`; `"log_loss_contribution": null`

## RECOMMENDATION
KEEP

## experiment_7414_v650_selected_feedback.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment completed validly but found insufficient online support for the adaptive method’s registered value claim.

## WHAT WOULD REFUTE IT
Adequate support plus statistically significant Brier improvement over every serious comparator, without worse log loss, coverage, or action risk, would refute the null headline.

## WAS THAT CHECKED
Yes. The registered multi-metric gate compared the adaptive arm with a frozen calibrated baseline, online L2 logistic regression, and a recent-frequency baseline across all 200 completed replay units. The gate failed, including against the serious frozen and logistic controls.

## EVIDENCE
`honest_verdict`: `complete_null_insufficient_online_support`; `registered_online_value`: `observed` `false`; `scientific_benefit_passed`: `false`; `online_value_score`: `0`; `support_passed`: `false`; `independent_online_groups`: `149`; `label_counts`: `0` `5`, `1` `144`; `brier_simultaneous_upper_below_zero`: `false`; `control_arm`: `frozen_calibrated_gibbs`; `control_arm`: `online_l2_logistic`; `completed`: `200`; `failed`: `0`; `censored`: `0`.

## RECOMMENDATION
KEEP

## experiment_7415_v650_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed independent static and online audits found no joint registered value.

## WHAT WOULD REFUTE IT
Valid, complete static and online branches both satisfying their registered value criteria—or a serious no-feedback/frozen comparator losing to the adaptive method on those criteria—would refute the claim.

## WAS THAT CHECKED
Yes. Both branch summaries separately report availability, validity, completion, and value; all 230 planned comparative units were completed, and the online audit included verified no-feedback and erased-update controls. The serious no-feedback comparator tied the frozen arm on the displayed metrics rather than revealing adaptive value.

## EVIDENCE
`honest_verdict` `complete_null_independent_decision_audits_no_joint_registered_value` `producer_verdict_class` `null` `valid` `true` `complete` `true` `value` `false` `planned` `230` `completed` `230` `no_feedback_control_verified` `true` `erased_update_control_verified` `true` `frozen_calibrated_gibbs` `no_feedback_adaptive_copy` `0.03528096605283069` `0.1389895006947763` `0.028288543140028287` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_7416_v650_anchored_extraction.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked precondition and contains no comparative or scientific result to falsify.

## WAS THAT CHECKED
No; no model load, generation, rows, or acceptance-gate results occurred.

## EVIDENCE
`honest_verdict`: `blocked_one_owned_rtx3090_slot`; `status`: `blocked_precondition`; `verdict_class`: `blocked`; `model_invoked`: `false`; `inference_substrate`: `no_model_load`; `rows`: `[]`; `acceptance_gate_results`: `[]`; `attempted`: `0`; `completed`: `0`; `unstarted`: `96`

## RECOMMENDATION
KEEP

## experiment_7417_extraction_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive experimental claim exists to falsify; this is a blocked-gate receipt, not an audit result.

## WAS THAT CHECKED
No. The audit was blocked before execution at `conductor_pre_gate`.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_reason": "actual=0 == expected=1"`, `"gate_check_summary": "2 of 3 gate(s) failed; first failure: exp7416-anchored-extraction.extraction_capture_complete_score (actual=0 == expected=1)"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7418_v650_revision_memory.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment completely captured memory-revision behavior but demonstrated no full-cost value.

## WHAT WOULD REFUTE IT
Incomplete, censored, failed, unsafe, truth-mismatched, restart-mismatched, or invalid-erasure rows would refute complete capture; a memory arm whose registered total-cost confidence bound beat the reset exact-solver baseline would refute the no-full-cost-value conclusion.

## WAS THAT CHECKED
Yes. All 7,680 planned rows were completed without censoring or failure; safety, truth, restart, and erasure checks passed. The serious reset-exact-solver comparator was present, and both candidate memory methods failed the registered total-cost gate. Oracle circularity does not invalidate this headline because it makes no positive value claim.

## EVIDENCE
`complete_memory_revision_capture_no_full_cost_value`; `verdict_class`: `null`; `memory_revision_capture_complete_score`: `1`; `memory_revision_value_score`: `0`; `planned`: `7680`; `completed`: `7680`; `censored`: `0`; `failed`: `0`; `truth_mismatches`: `0`; `cold_restart_mismatches`: `0`; `stale_proof_acceptances`: `0`; `valid_erasure_witness_count`: `792`; `cost_gate_passed`: `false`; `reset_exact_solver`; `persistent_incremental_solver`: `3.706797926244237`; `persistent_graph_reachability`: `5.180409718672519`; `registered_gate_not_met`; `promotion_score`: `0`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7419_v650_precision_placement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Int8 emulation provides no registered full-service benefit over float32 vectorized execution.

## WHAT WOULD REFUTE IT
Quality-preserving int8 execution that beats the serious float32-vectorized baseline on total service time—specifically, a primary-batch int8/float32 ratio and confidence interval below 1 with the speed gate passing—would refute the claim.

## WAS THAT CHECKED
Yes. Paired, rotated timing blocks compared `int8_float32_accum` directly with `float32_vectorized` at three batch sizes, including the registered primary batch size of `128`; int8 was slower at every size. Precision and action parity were also checked on `final_test` rows. Although `verifier_is_oracle` is `true`, the artifact makes no positive claim about verifier value, and the independently measured timing loss is sufficient for its null full-service-benefit claim.

## EVIDENCE
`honest_verdict`: `complete_null_int8_no_registered_full_service_benefit`; `verdict_class`: `null`; `scientific_benefit_passed`: `false`; `precision_value_score`: `0`; `primary_batch_size`: `128`; `estimate`: `1.0343090331264715`; `ci95`: `[1.0307835092811668, 1.0377982860931962]`; `speed_gate_passed`: `false`; `action_parity_passed`: `true`; `measured_action_flip_count`: `0`; `invalid_numeric_count`: `0`; `partition`: `final_test`; `heldout_labels_used`: `false`; `scale_partition`: `train`.

## RECOMMENDATION
KEEP

## experiment_7420_v650_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The capstone accounts for twelve task dispositions and reports required V650 science as blocked, without claiming scientific benefit or promotion.

## WHAT WOULD REFUTE IT
Fewer than twelve unique ordered dispositions, an unavailable branch reported as scientifically complete, or required science being available while the terminal verdict remained blocked would refute the accounting claim.

## WAS THAT CHECKED
Yes. Explicit equality and availability gates could fail, and the availability gate actually failed, producing the blocked verdict. Invalid or disqualified rows were marked as not accepted for science. No comparative value claim was made for the oracle to circularly validate.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v650_science_with_twelve_dispositions`; `twelve_ordered_dispositions`; `passed`: `true`; `required_science_available`; `observed`: `false`; `passed`: `false`; `scientific_value_score`: `0`; `promotion_score`: `0`; `verdict_class`: `blocked`; `accepted_for_science`: `false`; `verdict_class`: `disqualified`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP
