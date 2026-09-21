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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 1 |

## experiment_7480_v655_source_eval_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation showing that the artifact claims or relies upon a comparative performance advantage or predictive improvement without establishing it against a valid control. Because this artifact is an evaluation capture receipt that explicitly disclaims predictive benefit, there is no comparative hypothesis to refute.

## WAS THAT CHECKED
No; comparative predictive benefit was explicitly not tested or evaluated. The artifact documents an isolated data capture run designed to record raw logit shards for downstream evaluation rather than to evaluate a comparative claim.

## EVIDENCE
- `honest_verdict`: `complete_null_evaluation_capture_ready_predictive_benefit_not_tested`
- `status`: `complete_null_evaluation_capture_ready_predictive_benefit_not_tested`
- `verdict_class`: `null`
- `benefit_measured`: `false`
- `check`: `scientific_benefit_not_claimed`
- `promotion_score`: `0`
- `numbered_runtime_e2e`: `not_applicable_isolated_capture_experiment`

## RECOMMENDATION
KEEP

## experiment_7481_v655_typed_calibration.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The candidate Gibbs calibration head achieves a statistically significant typed decision benefit over the best simple logistic baseline under multiplicity-controlled cost evaluation.

## WHAT WOULD REFUTE IT
The claim would be refuted if the Holm-adjusted upper confidence bound for cost reduction was non-negative (`holm_upper >= 0`) or the adjusted p-value failed to reach significance (`holm_adjusted_p >= holm_alpha`) across all cost grid operating cells—demonstrating that the candidate could not outperform the logistic baseline—or if the observed `decision_benefit` gate was 0.

## WAS THAT CHECKED
Yes. It was checked in `decision_cost_grid` across 9 operating points (`holm_family_size: 9`) against a competitive parametric baseline (`best_simple_arm: "logistic"`). Refutation was genuinely possible and was observed in other cells (such as `cell_id: "fa=1|fr=1|esc=0.5"` and `cell_id: "fa=1|fr=1|esc=1"`, where `benefit_passed: false`) as well as in `probability_benefit` (`observed: 0`). However, refutation did not occur in cell `fa=1|fr=1|esc=0.1`.

## EVIDENCE
- `"check": "decision_benefit"`
- `"expected": 1`
- `"observed": 1`
- `"passed": true`
- `"decision_benefit_score": 1`
- `"cell_id": "fa=1|fr=1|esc=0.1"`
- `"benefit_passed": true`
- `"best_simple_arm": "logistic"`
- `"delta": -0.08756756756756758`
- `"holm_adjusted_p": 0.0008999100089991002`
- `"holm_alpha": 0.005555555555555556`
- `"holm_upper": -0.024472972972972774`
- `"heldout_labels_consumed": false`
- `"frozen": true`

## RECOMMENDATION
KEEP

## experiment_7482_v655_importance_anchor.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Importance anchoring provides a positive scientific benefit by achieving an adaptation improvement of 0.15 and reducing retention drift ratio to 0.74 compared to unanchored and uniform baselines.

## WHAT WOULD REFUTE IT
The claim of scientific benefit would be refuted if the retention drift ratio exceeded 0.80, if adaptation improvement fell below 0.005, or if importance anchoring failed to outperform uniform anchoring on independent, held-out evaluation tasks where the scoring verifier does not define the task fixture.

## WAS THAT CHECKED
No. Refutation was not given a real chance to happen because all evaluated rows use synthetic in-sample fixture data where the verifier is identical to the oracle, making the fixture benefit circular by design. No held-out or non-oracle evaluation was performed.

## EVIDENCE
- `verifier_is_oracle`: `true`
- `verdict_class`: `circular_positive`
- `honest_verdict`: `complete_circular_positive_importance_anchor_fixture_benefit`
- `status`: `complete_circular_positive_importance_anchor_fixture_benefit`
- `check`: `scientific_fixture_benefit`
- `passed`: `true`
- `data_role`: `analytic_fixture`
- `principle`: `A small sample, favorable seed, or analytic fixture cannot substitute for held-out value.`
- `field_principles` for `verifier_is_oracle`: `True forbids positive status because the analytic fixture defines the answer.`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7483_v655_continuous_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Continuous learning with an importance anchor provides no registered performance benefit over unanchored residual learning.

## WHAT WOULD REFUTE IT
A statistically significant, Holm-adjusted reduction in Brier score for `importance_anchor` over `unanchored_residual` across prequential delays, which would be observed as negative upper delta bounds (`delay_0_holm_upper_deltas` and `delay_8_holm_upper_deltas` with `unanchored_residual` < 0.0) and a passing scientific benefit gate.

## WAS THAT CHECKED
Yes. Checked under `online_reduction` across delays 0 and 8 against `unanchored_residual`, `affine`, and `frozen`. The observed upper deltas against `unanchored_residual` were positive (+0.0003895 and +0.0003566), which properly caused the scientific benefit gate to fail and supported the null verdict.

## EVIDENCE
- `honest_verdict`: `"complete_null_no_registered_importance_anchor_benefit"`
- `status`: `"complete_null_no_registered_importance_anchor_benefit"`
- `verdict_class`: `"null"`
- `all_required_validity_passed`: `true`
- `all_scientific_benefit_passed`: `false`
- `online_benefit_score`: `0`
- `online_complete_score`: `1`
- `verifier_is_oracle`: `false`
- `failed_checks`:
  - `check`: `"delay_0_holm_upper_deltas"`
  - `unanchored_residual`: `0.0003895057091183904`
  - `check`: `"delay_8_holm_upper_deltas"`
  - `unanchored_residual`: `0.00035660906355394577`

## RECOMMENDATION
KEEP

## experiment_7484_v655_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The v655 decision and learning mechanisms provide no statistically significant predictive calibration or online continuous learning benefit over baseline comparators on held-out data, resulting in an independent complete null audit finding.

## WHAT WOULD REFUTE IT
The claim of a null result would be refuted if the artifact's data showed the candidate methods outperforming standard baselines on held-out evaluations with statistical significance—specifically, `gibbs` achieving lower Brier and log loss than `temperature` and `logistic` calibration on external test groups (`external_gibbs_beats_temperature` observing `true`), and online `importance_anchor` significantly outperforming `unanchored_residual` across delay intervals (`multiplicity_direction_passed` observing `true`), allowing `static_and_online_benefit` to observe `true` and pass.

## WAS THAT CHECKED
Yes. Static probability calibration was evaluated against `logistic` and `temperature` baselines on 74 external test groups in `static_summary.probability_comparisons` and `static_summary.proper_scores.external`. Online replay was evaluated against `unanchored_residual`, `affine`, and `frozen` baselines across 159 independent groups in `online_summary.delays`. Both were synthesized in `acceptance_gate_results` under `static_and_online_benefit`.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_independent_v655_decision_and_learning_audit"`
- `"verdict_class"`: `"null"`
- `"check"`: `"static_and_online_benefit"`
- `"expected"`: `true`
- `"observed"`: `false`
- `"passed"`: `false`
- `"external_gibbs_beats_temperature"`: `false`
- `"probability_benefit_passed"`: `false`
- `"multiplicity_direction_passed"`: `false`
- `"scientific_benefit_passed"`: `false`
- `"all_required_validity_passed"`: `true`
- `"verifier_is_oracle"`: `false`
- `external` `gibbs` `"brier"`: `0.5783718222014341`
- `external` `temperature` `"brier"`: `0.3179760819813062`
- `external` `logistic` `"brier"`: `0.4853637188480826`
- `unanchored_residual` `"delta"`: `0.00022375403677868752`
- `unanchored_residual` `"holm_adjusted_p"`: `0.9987001299870013`

## RECOMMENDATION
KEEP

## experiment_7485_v655_arc_cost_panel_a.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The live ARC cost panel A evaluation resulted in a complete null finding with no intervention effect or reproduced task progress.

## WHAT WOULD REFUTE IT
Any observation of positive ARC task progress, non-zero intervention effect size, or passing scientific benefit gates in the artifact's own data—specifically, non-zero `reproduced_levels`, `any_reproduced_progress` being true, `actions_to_progress` resolving to a non-null step count, or `all_passed` being true in `gate_check_summary`.

## WAS THAT CHECKED
Yes, across live model evaluation episodes under `per_game_results` for games `sk48` and `tr87`, and evaluated against the scientific benefit criteria in `gate_check_summary`.

## EVIDENCE
`"honest_verdict": "complete_null_live_arc_cost_panel_a"`
`"all_passed": false`
`"first_failed_check": "e6_episode_support_floor"`
`"check": "effect_size_threshold"`
`"check": "retention_threshold"`
`"observed": false`
`"any_reproduced_progress": false`
`"reproduced_levels": 0`
`"actions_to_progress": null`
`"actions_to_progress_censored": true`
`"offline_reproduced": false`
`"e0_parity_claim": false`
`"e4_e5_readiness_claim": false`
`"hidden_game_efficacy_claim": false`
`"verifier_is_oracle": "Declare that the public environment is the evaluation oracle, which forbids a positive verdict."`

## RECOMMENDATION
KEEP

## experiment_7487_v655_learning_placement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
A measured 100x end-to-end service speedup from retained-state placement is not established.

## WHAT WOULD REFUTE IT
An observation of `service_envelope.compatible_complete_denominator` evaluating to `true` with a `service_envelope.measured_end_to_end_speedup` of at least `100.0`, satisfying the scientific benefit acceptance gate and generating a positive `scientific_benefit_score`.

## WAS THAT CHECKED
Yes; acceptance gates in `gate_check_summary` explicitly evaluated `complete_service_denominator` and `one_hundred_x_service_target`, observing `false` and `null` respectively against the required thresholds, confirming the null verdict.

## EVIDENCE
- `"honest_verdict": "complete_null_retained_state_placement_measured_service_100x_not_established"`
- `"verdict_class": "null"`
- `"scientific_benefit_score": 0`
- `"check": "complete_service_denominator"`
- `"expected": true`
- `"observed": false`
- `"check": "one_hundred_x_service_target"`
- `"expected": 100.0`
- `"observed": null`
- `"failed_count": 2`
- `"compatible_complete_denominator": false`
- `"measured_end_to_end_speedup": null`
- `"observed_accelerator_performance": false`
- `"status": "complete_null_retained_state_placement_measured_service_100x_not_established"`

## RECOMMENDATION
KEEP

## experiment_7488_v655_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V655 milestone is disqualified from establishing a positive aggregate because required present upstream evidence is invalid and fails required acceptance gates.

## WHAT WOULD REFUTE IT
Observing that all required present upstream evidence is valid and passes validation gates—specifically, observing `gate_check_summary.passed` as `true` with `failed_count` as `0`, `required_present_evidence_valid` as `true`, upstream validation receipts (`validation_receipts.overdue_priority.passed` and `validation_receipts.adversarial_verify.passed`) as `true`, and required panel B producer evidence present.

## WAS THAT CHECKED
Yes; checked in `acceptance_gate_results`, `gate_check_summary`, `task_dispositions`, `panel_states`, and `science_reductions`.

## EVIDENCE
- `"honest_verdict"`: `"complete_disqualified_required_present_v655_evidence"`
- `"status"`: `"complete_disqualified_required_present_v655_evidence"`
- `"verdict_class"`: `"disqualified"`
- `"positive_aggregate"`: `false`
- `"positive_aggregate_reason"`: `"Required present evidence is invalid and the independent audit failed a required reader."`
- `"gate_check_summary"`: `"passed"`: `false`, `"failed_count"`: `3`
- `"acceptance_gate_results"`:
  - `"check"`: `"required_present_evidence_valid"`, `"expected"`: `true`, `"observed"`: `false`, `"passed"`: `false`
  - `"check"`: `"arc_support_floor"`, `"expected"`: `{"episodes": 30, "games": 10}`, `"observed"`: `{"episodes": 18, "games": 6}`, `"passed"`: `false`
  - `"check"`: `"independent_scientific_benefit"`, `"expected"`: `true`, `"observed"`: `false`, `"passed"`: `false`
- `"failed_checks"`:
  - `"upstream"`: `"exp7475-contract-methods"`, `"field"`: `"validation_receipts.overdue_priority.passed"`, `"expected"`: `true`, `"observed"`: `false`, `"passed"`: `false`
  - `"upstream"`: `"exp7484-decision-audit"`, `"field"`: `"validation_receipts.adversarial_verify.passed"`, `"expected"`: `true`, `"observed"`: `false`, `"passed"`: `false`
  - `"upstream"`: `"exp7486-arc-cost-panel-b"`, `"field"`: `"path"`, `"op"`: `"exists"`, `"expected"`: `true`, `"observed"`: `false`, `"passed"`: `false`
- `"evidence_state"`: `"invalid"`
- `"evidence_state"`: `"missing"`
- `"support_floor_passed"`: `false`

## RECOMMENDATION
KEEP
