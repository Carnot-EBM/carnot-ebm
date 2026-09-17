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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |

## experiment_7361_v646_fresh_plan_capture.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The fresh live-LLM capture produced 128 publicly faithful proposals constituting a positive scientific result.

## WHAT WOULD REFUTE IT
Independent execution or a non-oracle evaluator rejecting the proposals’ semantic correctness, or a cheapest-valid-plan baseline tying or beating the model, would refute the value claim.

## WAS THAT CHECKED
No. Public fidelity was scored, but the verifier defines correctness, and no serious non-model baseline or independent outcome oracle is reported; therefore the value claim had no genuine opportunity to fail.

## EVIDENCE
`honest_verdict` `complete_circular_positive_fresh_plan_capture_with_public_fidelity` `verifier_is_oracle` `True when the evaluator defines truth; independent code alone cannot remove circularity.` `public_semantic_correct_count` `128` `hidden_rule_accepted` `false` `promotion_ready_score` `0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7362_v646_prospective_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed experiment is disqualified because a required validation check failed.

## WHAT WOULD REFUTE IT
All required safety and validation checks passing, including the complete required affected-validation command set.

## WAS THAT CHECKED
Yes. `acceptance_gate_results.affected_validation` compared the required and observed command sets and failed; the artifact also reports multiple failed value gates.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_safety_or_validation_failure`; `affected_validation`; `passed`: `false`; `complete_service_cost_ratio`; `query_ratio_vs_exact_cache`; `query_ratio_vs_reset`; `later_distinct_request_erasure_witness`; `learning_value_passed`: `false`; `promotion_score`: `0`; `verdict_class`: `disqualified`

## RECOMMENDATION
KEEP

## experiment_7363_learning_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing: the artifact makes no substantive comparative or value claim to falsify; it only records a blocked gate.

## WAS THAT CHECKED
No. Execution stopped at `conductor_pre_gate`, so the intended structural-learning and total-cost adjudication did not occur.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_reason`: `actual=0 == expected=1`; `learning_capture_complete_score`; `actual`: `0`; `passed`: `false`; `verdict_class`; `actual`: `disqualified`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7364_v646_acquisition_adjudication.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The archival adjudication is complete but establishes no acquisition value, preserves the original disqualification, and fails the complete-cost gate.

## WHAT WOULD REFUTE IT
A complete-cost 95% interval with an upper bound below 0.90, positive acquisition value or readiness, or evidence that the original disqualification was cleared rather than preserved.

## WAS THAT CHECKED
Yes. The scientific-cost gate compares the finite-bias method with conservative acquisition; its upper bound is 0.9652615826317683 against the required threshold of 0.90. The artifact also explicitly checks and preserves the original disposition. Coverage and utility tie between those arms, so no hidden effectiveness gain rescues the failed cost claim. Oracle circularity does not invalidate this null result because no positive verifier-value claim is made.

## EVIDENCE
`honest_verdict`: `complete_null_acquisition_adjudication_preserves_disqualification_and_failed_cost_gate`; `verdict_class`: `null`; `acquisition_value_score`: `0`; `acquisition_readiness_score`: `0`; `complete_cost_gate_passed`: `false`; `complete_cost_gate_threshold`: `0.9`; `upper`: `0.9652615826317683`; `finite_bias_coverage`: `360`; `conservative_coverage`: `360`; `finite_bias_utility`: `327.5`; `conservative_utility`: `327.5`; `preserved`: `true`; `historical_disqualification_cleared`: `false`; `verifier_is_oracle`: `true`.

## RECOMMENDATION
KEEP

## experiment_7365_v646_supervisor_support.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found insufficient supported outcomes to establish scientific value or trial readiness.

## WHAT WOULD REFUTE IT
A candidate ordering meeting the minimum of 10 supported decisions for every arm, passing leave-one-game-out support, and consequently passing the support, scientific-value, or promotion gate would refute the null claim.

## WAS THAT CHECKED
Yes. All three planned candidate orderings were completed; support-floor and leave-one-game-out checks failed, and the support, efficacy, and promotion gates recorded zero. The oracle defines observed outcomes, but the artifact makes no positive claim about the verifier’s added value.

## EVIDENCE
`"honest_verdict"`: `"complete_null_insufficient_supported_outcomes"`; `"candidate_orderings_planned"`: `3`; `"candidate_orderings_completed"`: `3`; `"minimum_supported_decisions_per_arm"`: `10`; `"arm_support_floor_passed"`: `false`; `"leave_one_game_out_passed"`: `false`; `"support_floor"` with `"observed"`: `0`; `"scientific_value"` with `"observed"`: `0`; `"promotion"` with `"observed"`: `0`; `"supervisor_trial_ready_score"`: `0`; `"scientific_value_score"`: `0`; `"verdict_class"`: `"null"`.

## RECOMMENDATION
KEEP

## experiment_7366_supervisor_live.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the experiment produced no substantive comparative result.

## WAS THAT CHECKED
No; execution stopped at the pre-gate, before any live-game rows or comparator results were produced.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "supervisor_trial_ready_score"`, `"failed_expected": 1`, `"failed_observed": 0`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7367_v646_board_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify; the operational disposition would be contradicted by a qualifying post-Exp6559 GateMate physical-change receipt or by incomplete accounting of the three named boards.

## WAS THAT CHECKED
Yes. The artifact checked the GateMate receipt contract and accounted for KV260, PolarFire, and GateMate, while explicitly withholding readiness, hardware-value, and promotion claims.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts`; `diagnostic_only`: `true`; `authorizes_promotion`: `false`; `board_disposition_complete_score`: `1`; `row_count`: `3`; `accepted_receipt_count`: `0`; `selected_source_path`: `null`; `hardware_ready_score`: `0`; `hardware_value_score`: `0`; `promotion_score`: `0`; `new_hardware_runs_completed`: `0`; `model_invoked`: `false`

## RECOMMENDATION
KEEP

## experiment_7368_v646_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All twelve V646 dispositions were accounted for, but the capstone was disqualified because required science and validation failed and the independent learning audit was pre-gated.

## WHAT WOULD REFUTE IT
A complete twelve-row accounting showing required validation passed, required science completed, and an eligible independent learning audit succeeded—or evidence that fewer than twelve dispositions were accounted for—would refute the headline.

## WAS THAT CHECKED
Yes. Explicit expected-versus-observed gates checked disposition count, required validation, required science, scientific value, and learning-audit eligibility. The required checks and learning capture failed, while all twelve dispositions were reported accounted for. The oracle-defined verifier does not rescue or circularly establish a value claim because the artifact assigns zero scientific value and expressly disqualifies promotion.

## EVIDENCE
`milestone_accounting` `expected` `12` `observed` `12` `passed` `true`; `required_validation` `expected` `true` `observed` `false` `passed` `false`; `required_science` `expected` `1` `observed` `0` `passed` `false`; `scientific_value` `expected` `1` `observed` `0` `passed` `false`; `learning_capture_complete_score` `expected` `1` `observed` `0` `passed` `false`; `required_checks_passed` `false`; `value_score` `0`; `promotion_score` `0`; `publication_ready_score` `0`; `verdict_class` `disqualified`; `verifier_is_oracle` `true`; `complete_disqualified_required_science_or_validation_failure: all twelve V646 dispositions are accounted for, but Exp7362 failed required validation and the independent learning audit was pre-gated`

## RECOMMENDATION
KEEP
