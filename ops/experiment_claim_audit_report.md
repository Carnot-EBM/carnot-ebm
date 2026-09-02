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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 6 |

## experiment_6866_canonical_tokenizer_binding_requalification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6867_tokenizer_aware_semantic_preregistration_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific-effect claim to falsify; the procedural readiness disposition would be refuted if any required model had fewer than 20 accepted held groups.

## WAS THAT CHECKED
Yes, in `sample_size_power_rows` and the preregistered `missing_cell_rule`; all required held splits exceeded the 20-group floor.

## EVIDENCE
`scientific_effect_claimed` `false`; `token_likelihood_call_count` `0`; `generated_answer_count` `0`; `held_label_access_count` `0`; `preregistered_design_target_not_observed_result`; `minimum_group_count` `20`; `accepted_group_count` `50`; `accepted_group_count` `29`; `floor_passed` `true`; `complete_positive_tokenizer_aware_semantic_preregistration_v2_ready_no_scores`

## RECOMMENDATION
KEEP

## experiment_6868_three_family_semantic_scoring_stream_v2.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6869_calibration_only_paired_semantic_rule.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6870_sealed_independent_semantic_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt and reports no semantic-audit or comparative result to falsify.

## WAS THAT CHECKED
No. The audit stopped at the conductor pre-gate after one prerequisite failed; no method, oracle, rival, or scored rows were evaluated.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `duration_s` `0.0` `failed_field` `semantic_contrast_rule_ready_score` `failed_observed` `0` `failed_expected` `1` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6871_observable_reliability_opportunity_stream.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6872_bounded_reliability_controller_quarantine.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6873_prospective_sealed_self_learning_audit.json

**SKIPPED_ALREADY_FLAGGED**
