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
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 1 |

## experiment_7321_v643_batch_measurement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The batched verifier failed the frozen value gates relative to the direct comparator.

## WHAT WOULD REFUTE IT
All value gates passing—particularly an accuracy-difference lower bound of at least -0.02 versus the direct arm—and a batch value score of 1 would refute the claim.

## WAS THAT CHECKED
Yes. The artifact reports the paired accuracy comparison against the direct arm, its prespecified threshold, the observed lower bound, and the resulting value-gate score.

## EVIDENCE
`complete_null_batch_value_gates_failed`; `batch_value_score`; `0`; `accuracy_difference_lower_vs_direct`; `paired_intervals.accuracy_difference_vs_direct.one_sided_95_lower`; `expected_value`; `-0.02`; `observed_value`; `-0.296875`; `estimate`; `-0.171875`; `evaluation_labels_read_during_prediction`; `false`; `predictions_sealed_before_labels`; `true`

## RECOMMENDATION
KEEP

## experiment_7322_v643_batch_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The same-mechanism batched verifier failed to demonstrate value over the direct comparator and should not be promoted.

## WHAT WOULD REFUTE IT
A passing promotion result—most concretely, the batched verifier’s one-sided 95% accuracy-difference lower bound meeting or exceeding the frozen −0.02 threshold, alongside acceptable coverage and cost—would refute the null headline.

## WAS THAT CHECKED
Yes. The artifact compares the batched verifier against the serious joint-direct baseline over 128 units and 16 source groups, applies a paired bootstrap, and reports the failed accuracy gate. The oracle relationship does not make this negative value finding circular.

## EVIDENCE
`"honest_verdict": "complete_null_same_mechanism_batch_value_comparison_failed"`; `"comparison": "batched_verifier_vs_joint_direct"`; `"unit_denominator": 128`; `"group_count": 16`; `"accuracy_difference": -0.171875`; `"coverage_difference": -0.1875`; `"expected_value": -0.02`; `"observed_value": -0.296875`; `"failed_check": "accuracy_difference_lower_vs_direct"`; `"batch_promotion_score": 0`; `"verdict_class": "null"`; `"decision": "stop"`

## RECOMMENDATION
KEEP

## experiment_7323_v643_addition_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
No comparative efficacy claim; the artifact claims only that a bounded addition fixture is ready for later held-out evaluation.

## WHAT WOULD REFUTE IT
A failed fixture-readiness requirement—such as broken stream isolation, failed controls, exceeded resource bounds, or no causal later use—would refute readiness. If efficacy were claimed, a held-out cache/reset comparator tying or beating the method while preserving coverage and utility would refute it.

## WAS THAT CHECKED
Yes for fixture readiness, through the development acceptance gates and controls. No for efficacy: all held-out comparisons were unexecuted. Development data also shows the frozen and label-shuffled arms exactly tying the method on oracle calls and utility, so those rows cannot establish learning value.

## EVIDENCE
`honest_verdict`: `complete: bounded addition fixture ready under shared Boolean executor authority; held-out efficacy not executed`; `addition_fixture_ready_score`: `1`; `addition_promotion_score`: `0`; `held_out_oracle_ratio_vs_cache`: `not_executed`; `held_out_oracle_ratio_vs_reset`: `not_executed`; `held_out_utility`: `not_executed`; `attempted_environments`: `0`; `executed`: `false`; `persistent_structural_acquisition`; `frozen_after_four_request_warmup`; `label_shuffled_diagnostic`; `total_oracle_calls`: `504`; `mean_utility_fraction`: `1.0`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`

## RECOMMENDATION
KEEP

## experiment_7324_v643_addition_learning.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Persistent structural addition reduced held-out executor work and therefore demonstrated positive value.

## WHAT WOULD REFUTE IT
A paired work ratio at or above 1.0 against a serious baseline, or reduced work accompanied by worse coverage, utility, or feasibility, would refute the execution claim; an independent correctness authority is required to support the broader value claim.

## WAS THAT CHECKED
No, not fully. Work, coverage, utility, and feasibility were checked against reset and exact-cache controls, and the work reduction could have failed there. But correctness used the same exact executor that generated the feedback, so the positive value claim received no independent test. Moreover, the frozen-after-warmup rival actually won on aggregate executor work.

## EVIDENCE
`honest_verdict`: `complete: persistent structural addition reduced held-out executor work under shared exact executor authority`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`; `oracle_work_vs_cache`: `estimate` `0.14898810929994488`, `ci95_upper` `0.15715804394046776`; `oracle_work_vs_frozen`: `estimate` `1.6458333333333333`, `ci95_lower` `1.4375`, `ci95_upper` `1.8489583333333333`; `model_invoked`: `false`; `censored_stream_count`: `0`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7325_v643_addition_audit.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Persistent structural acquisition reduces total calls without weakening utility, coverage, feasibility, causality, or version safety.

## WHAT WOULD REFUTE IT
A verifier-independent evaluator finding no call reduction or weaker outcomes would refute the value claim; an aggregate tie or loss against the cheap serious baseline that freezes learned structure after warmup would also refute added value.

## WAS THAT CHECKED
No. Call reduction versus reset and cache controls was checked, but correctness and utility remained under shared executor authority, so the verifier-independent refutation could not occur. The artifact includes a frozen-after-warmup comparator, but the visible data show it tying the learned method, and no aggregate comparison against that arm is reported.

## EVIDENCE
`verifier_is_oracle`; `Shared executor authority forbids a positive scientific class.`; `persistent_structural_acquisition`; `frozen_after_four_request_warmup`; `total_query_attempts`: `40`; `mean_utility_fraction`: `1.0`; `control_arm`: `reset_each_request_acquisition`; `control_arm`: `exact_plan_cache_reset_learner`; `further_work_condition`: `independent non-oracle executor replication or a changed information contract`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7326_v643_constraint_kernel.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The Rust kernel achieved bit-exact parity with Python, but failed the predeclared 10× paired host service-boundary performance gate.

## WHAT WOULD REFUTE IT
Any Rust/Python mismatch would refute the parity claim; lower 95% speedup bounds of at least 10 at every tested batch size would refute the performance-null claim.

## WAS THAT CHECKED
Yes. Parity was checked across 3,304 rows, and performance was checked in 30 paired blocks at each batch size of 1, 32, and 256. The Python comparator is a serious direct baseline; Rust did not merely fail 10×, but had lower-bound speedups below 1 at every size. The oracle relationship limits the result to Python/Rust parity, but the headline makes no independent-correctness or added-value claim.

## EVIDENCE
The artifact reports `complete_null: acquired integer constraints have bit-exact Rust parity, but the predeclared 10x paired host service-boundary gate did not pass`; `mismatches`: `0`; `rows`: `3304`; `all_outputs_matched`: `true`; `paired_rows`: `90`; `expected`: `>=10 at sizes 1, 32, and 256`; observed lower bounds `0.5869928629436495`, `0.39793675305710036`, and `0.43296225006025074`; `ten_x_lower_bound_passed`: `false`; `verifier_is_oracle`: `true`; `verdict_class`: `null`; `whole_learning_speedup_claimed`: `false`.

## RECOMMENDATION
KEEP

## experiment_7327_v643_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The task is blocked because no qualifying post-Exp6559 GateMate physical-change receipt exists, while prior KV260 fabric and PolarFire CPU results remain preserved without asserting present availability or performing new hardware operations.

## WHAT WOULD REFUTE IT
A qualifying operator-authored GateMate receipt dated after 20260823, an authentication or quarantine failure in the preserved upstream evidence, a row asserting present board availability, or any newly issued hardware operation would refute the corresponding part of the claim.

## WAS THAT CHECKED
Yes. The artifact searched the specified receipt sources under an explicit eligibility contract, authenticated the upstream artifact and hash chains, recorded per-board availability assertions, and counted hardware operations. These checks could have returned a qualifying receipt, failed authentication, asserted current availability, or recorded nonzero operations.

## EVIDENCE
`"accepted_receipt_count": 0`; `"exists": false`; `"newer_than_exp6559": false`; `"gatemate_changed_physical_state_receipt"`; `"passed": false`; `"terminal_criterion_met": true`; `"historical_graduation_preserved": true`; `"present_availability_asserted": false`; `"checksum_matches": true`; `"quarantined": false`; `"hardware_operations_issued_count": 0`; `"status": "blocked"`; `"verdict_class": "blocked"`; `"inference_substrate": "aggregation_from_upstream_artifacts"`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7328_v643_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All thirteen task dispositions are complete, but capstone readiness remains blocked because the required changed-physical-state GateMate receipt is unavailable.

## WHAT WOULD REFUTE IT
Any incomplete or censored disposition, or an accepted post-Exp6559 operator-authored GateMate physical-state-change receipt, would refute the headline.

## WAS THAT CHECKED
Yes. The artifact checks planned, attempted, completed, and censored disposition counts, and separately checks the GateMate receipt gate. The oracle-defined structural-learning result is explicitly classified as circular and does not promote scientific efficacy, so it is not used to turn the blocked headline into a value claim.

## EVIDENCE
`"honest_verdict": "blocked_exp7327_board_continuity: all thirteen dispositions are complete; the named external prerequisite remains unavailable"`; `"attempted": 13`; `"complete": 13`; `"planned": 13`; `"censored": 0`; `"accepted_receipt_count": 0`; `"selected_source_path": null`; `"terminal_blocking": true`; `"verdict_class": "circular_positive"`; `"promotes_scientific_efficacy": false`; `"capstone_promotion_score": 0`; `"capstone_readiness_score": 0`

## RECOMMENDATION
KEEP
