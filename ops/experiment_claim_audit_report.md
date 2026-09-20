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
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_7439_v652_certified_decisions.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed experiment found no registered decision benefit for the tested method.

## WHAT WOULD REFUTE IT
A valid sparse-spline certificate meeting the risk and coverage requirements while delivering positive coverage or utility improvement over both the old policy and raw L2 logistic baseline would refute the null claim.

## WAS THAT CHECKED
Yes. The independent reduction evaluated certificate validity, coverage, utility, calibration, and paired coverage differences against both the old-fixed policy and the serious raw L2 logistic baseline. The spline instead failed its risk and coverage checks, while both paired coverage contrasts were exactly zero.

## EVIDENCE
`honest_verdict` `complete_null_no_registered_decision_benefit`; `decision_value_score` `0`; `scientific_benefit_passed` `false`; `spline_certificate_valid` `false`; `certificate_coverage_lower_bound` `0.14352750364168598`; `coverage_floor` `0.25`; `diagnosis` `excessive_observed_risk`; `harmful_outcomes` `12`; `raw_l2_logistic:tuned`; `tuned_spline_minus_logistic`; `observed` `0.0`; `lower` `0.0`; `upper` `0.0`; `registered_utility` `0.0`; `certificate_scope` `exploratory_reused_corpus`

## RECOMMENDATION
KEEP

## experiment_7440_v652_mixture_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment completed but found insufficient evidence of online benefit from mixture learning.

## WHAT WOULD REFUTE IT
All prespecified simultaneous confidence-interval upper bounds for later log-loss deltas being below zero, with the causal, cost, control, and loss gates passing—especially against the equal-weight adaptive mixture and frozen-spline baselines—would refute the null claim.

## WAS THAT CHECKED
Yes. The prespecified acceptance gates tested whether every simultaneous upper later-loss delta was negative across 12 family contrasts; serious equal-weight and frozen comparators were included, all 140 planned units completed, and no validity failures were reported.

## EVIDENCE
`honest_verdict`: `complete_null_insufficient_online_benefit`; `online_value_score`: `0`; `promotion_score`: `0`; `benefit_failures`; `validity_failures`: `[]`; `primary_comparisons`: `3`; `contrasts`: `12`; `comparator`: `equal_weight_adaptive_mixture`; `upper`: `0.0023975552315087414`; `comparator`: `frozen_spline`; `observed`: `0.0021845753196735223`; `upper`: `0.006539421293975578`; `planned`: `140`; `completed`: `140`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7441_v652_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The static branch shows no registered decision benefit, while the online-learning branch is disqualified because its prediction-time expert evidence and weight-update replay are incomplete.

## WHAT WOULD REFUTE IT
A valid static policy certificate meeting both coverage and risk requirements, or complete online expert predictions enabling a successful weight-update replay, would refute the corresponding headline conclusion.

## WAS THAT CHECKED
Yes. The static branch evaluates policy certificates, coverage, risk, and registered utility across spline, logistic, and Gibbs arms. The online validity gate explicitly checks evidence completeness and reports the missing expert predictions and incomplete replay that cause disqualification.

## EVIDENCE
`honest_verdict`: `complete_disqualified_invalid_branch_evidence`; `static_evidence_valid`: `observed` `true`, `passed` `true`; `static_audit`: `valid` `true`, `value` `false`, `verdict_class` `null`; `complete_null_no_registered_decision_benefit`; `registered_utility`: `0.0`; `tuned_spline_minus_logistic`: `observed` `0.0`, `lower` `0.0`, `upper` `0.0`; `online_evidence_valid`: `observed` `false`, `passed` `false`; `online_audit`: `valid` `false`, `value` `false`, `verdict_class` `disqualified`; `missing_expert_predictions`; `weight_update_replay_incomplete`; `updates_replayed`: `0`; `complete expert prediction losses are required`

## RECOMMENDATION
KEEP

## experiment_7442_v652_span_capture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7443_v652_span_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7444_v652_arc_supervisor_evidence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The two archived episodes contained no supervisor firings, no arm-effect evidence, and no basis for a policy change.

## WHAT WOULD REFUTE IT
Any row showing a supervisor firing, a consumed intervention, outcome evidence attributable to an arm, or a resulting policy change would refute the literal null claim.

## WAS THAT CHECKED
Yes, in the per-game rows, arm rows, and private supervisor ledger. Those checks support the observational null, although the episodes did not reach the firing threshold, so they provided no real test of supervisor efficacy.

## EVIDENCE
`"honest_verdict": "complete_null_no_supervisor_firings_no_arm_effect_evidence"`; `"supervisor_firings": 0`; `"arm_effect_evidence": "none_no_firing"`; `"firing_threshold_reached": false`; `"actions": 62`; `"shipped_firing_threshold": 120`; `"outcome_evidence_count": 0`; `"live_policy_changed_by_this_task": false`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7445_v652_hardware_envelope.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
With the measured persistence fraction and unchanged acknowledgement/crash semantics, numeric hardware acceleration cannot deliver 100× complete-service speedup and persistence/orchestration must be redesigned.

## WHAT WOULD REFUTE IT
An authenticated complete-service row showing an unaccelerated fraction below 0.01—or a measured 100× service speedup while preserving equivalent acknowledgement and crash semantics—would refute the claim.

## WAS THAT CHECKED
Yes. The necessary 100× condition was explicitly evaluated in `amdahl_rows` and the benefit gate; it failed because the measured unaccelerated fraction exceeded 0.01. No new hardware execution tested the second route, but the infinite-acceleration bound already gives hardware maximal opportunity under the stated unchanged semantics.

## EVIDENCE
`"formula": "f < 1 / target_speedup"`; `"required_unaccelerated_fraction": 0.01`; `"observed_unaccelerated_fraction": 0.37696053267914603`; `"condition_met": false`; `"service_speed_limit_x": 2.6527976095873163`; `"condition": "idealized_infinite_acceleration_of_all_other_work"`; `"interpretation": "necessary_idealized_condition_not_sufficient_measurement"`; `"requires_changed_persistence_orchestration_design_for_100x": true`; `"equivalent_acknowledgement_and_crash_semantics_required": true`; `"hardware_value_score": 0`; `"new_hardware_measurement": false`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7446_v652_capstone.json

**SKIPPED_ALREADY_FLAGGED**
