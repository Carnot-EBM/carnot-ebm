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
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7426_v651_static_decisions.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed static source-support experiment found no registered decision benefit.

## WHAT WOULD REFUTE IT
A registered arm achieving certified official-test coverage of at least 0.25 while satisfying the registered predictive, risk, and non-inferiority checks would refute the null claim.

## WAS THAT CHECKED
Yes. The artifact evaluated registered arms and policy thresholds on separate fit, probability-calibration, policy-calibration, and official-test partitions; the benefit reduction failed specifically because certified coverage did not reach 0.25. All arms escalated every official-test row, producing zero coverage and no enabled accept or reject action.

## EVIDENCE
`"honest_verdict": "complete_null_static_source_support_no_registered_decision_benefit"`; `"decision_value_score": 0`; `"passed": false`; `"certified_coverage_at_least_0_25": false`; `"coverage": 0.0`; `"escalate_count": 2675`; `"accept_count": 0`; `"reject_count": 0`; `"scientific_benefit_passed": false`; `"selection_partition": "policy_calibration"`; `"independent_test_groups": 450`; `"completed_units": 75`; `"failed_units": 0`

## RECOMMENDATION
KEEP

## experiment_7427_v651_randomized_feedback.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed replay found no registered online value.

## WHAT WOULD REFUTE IT
All prespecified benefit clauses passing—including non-worse loss, risk, coverage, and cost, with every simultaneous upper Brier-delta bound below zero—producing an online value score of 1.

## WAS THAT CHECKED
Yes. The registered reduction checked eight combinations against the serious `frozen_spline` and `online_raw_logistic` controls across both orderings and delays; the required upper-bound condition failed in every displayed check.

## EVIDENCE
`honest_verdict` `complete_null_no_registered_online_value` `scientific_benefit_passed` `false` `online_value_score` `0` `passed` `false` `upper_brier_deltas_below_zero` `false` `control_arm` `frozen_spline` `online_raw_logistic` `no_feedback_benefit_disappeared` `true` `attempted_units` `300` `completed_units` `300` `failed_units` `0` `censored_units` `0`

## RECOMMENDATION
KEEP

## experiment_7428_v651_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed static and online audits found no registered benefit.

## WHAT WOULD REFUTE IT
A valid, eligible branch showing a registered method’s benefit over a serious comparator—such as better static held-out performance or an online Brier improvement whose simultaneous upper bound clears the registered benefit threshold—would refute the claim.

## WAS THAT CHECKED
Yes. The static audit used separate fit, calibration, and final-test partitions and compared spline methods with logistic controls; the online audit compared adaptive methods with frozen and no-feedback spline controls using simultaneous intervals. Both branches were valid and complete, yet both recorded no value. Exact static and online comparator ties also gave the claimed added value a real opportunity to fail.

## EVIDENCE
`honest_verdict` `complete_null_static_and_online_audits_reproduce_no_registered_benefit` `branch` `static` `valid` `true` `complete` `true` `value` `false` `final_test` `fit` `policy_calibration` `probability_calibration` `dense_spline_logistic` `sparse_spline_49` `brier` `0.1686380774939802` `log_loss` `0.5101727535875094` `branch` `online` `valid` `true` `complete` `true` `value` `false` `frozen_spline` `no_feedback_spline` `full_brier` `0.19228843334343437` `full_log_loss` `0.5669878795037654` `point_delta` `-0.00029067370218418756` `upper_brier_delta` `0.002075269495644716` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_7429_v651_anchored_capture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7430_extraction_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing; the artifact is a blocked-gate receipt and makes no substantive claim about extraction quality.

## WAS THAT CHECKED
No. The audit did not run because all three upstream gates failed.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"gate-unsat(final): 3 of 3 gate(s) failed"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7431_v651_arc_live_sentinel.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The live ARC sentinel run reached and recorded valid terminal dispositions for both scheduled episodes, without claiming efficacy or comparative benefit.

## WHAT WOULD REFUTE IT
Either scheduled episode failing to obtain a valid terminal disposition, or the model-generation/request chain failing to reach a recorded terminal state, would refute the reachability claim.

## WAS THAT CHECKED
Yes. Both scheduled rows retain terminal dispositions, invocation events record attempted and completed generations, and request permits are joined to terminal outcomes. The design explicitly allowed failed, censored, or unstarted units, so failure was possible. No efficacy claim was tested; therefore the oracle and absent control do not make this limited operational claim circular.

## EVIDENCE
`honest_verdict` = `complete_null_arc_live_sentinel_reachability_no_efficacy_claim`; `arc_sentinel_capture_complete_score` = `1`; `planned_units` = `2`; `completed_units` = `2`; `failed_units` = `0`; `censored_units` = `0`; `unstarted_units` = `0`; both episode rows have `disposition` = `complete`; `generation_calls_attempted` = `2`; `generation_calls_completed` = `2`; `permit_to_terminal_chains` = `2`; `live_efficacy_score` = `0`; `treatment_effect_claimed` = `false`; `promotion_score` = `0`; `verdict_class` = `null`.

## RECOMMENDATION
KEEP

## experiment_7432_v651_update_placement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Sparse/fixed-point update placement provides no registered complete-service benefit over the dense float32 baseline.

## WHAT WOULD REFUTE IT
A registered sparse arm preserving action and proper-score quality while achieving a paired whole-service 95% confidence-interval upper bound below 1 at every tested batch size would falsify the null.

## WAS THAT CHECKED
Yes. The artifact compares sparse arms against the float32-dense baseline across batch sizes 1, 32, and 128 using 30 paired timing blocks per batch; the complete-service speed gate failed because every reported confidence-interval upper bound exceeded 1.

## EVIDENCE
`honest_verdict`: `complete_null_sparse_update_no_registered_complete_service_benefit`; `baseline_arm`: `float32_dense`; `batch_sizes`: `[1, 32, 128]`; `paired_timing_blocks`: `30`; `check`: `paired_whole_service_speed`; `field`: `timing_summary.ci95_upper`; `observed`: `[1.0470479326842994, 1.0659364797751325, 4.167691682605176, 1.080045472608621, 1.003538599768254, 2.435059441808669]`; `passed`: `false`; `update_placement_value_score`: `0`; `verdict_class`: `null`.

## RECOMMENDATION
KEEP

## experiment_7433_v651_capstone.json

**SKIPPED_ALREADY_FLAGGED**
