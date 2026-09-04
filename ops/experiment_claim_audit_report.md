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
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6957_smt_mapping_certification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Frozen SOTA mapping proposals failed to demonstrate a certifiable positive gain over the syntax-only baseline.

## WHAT WOULD REFUTE IT
A positive certification score supported by paired accuracy gains whose 95% confidence intervals exclude zero, while maintaining acceptably low false acceptance, would refute the null claim.

## WAS THAT CHECKED
Yes. All 162 frozen proposals received terminal outcomes, were compared attempt-by-attempt with the frozen syntax-only baseline, and had model-family paired confidence intervals computed. The intervals showed losses for two families and an exact tie for the third, so the method had a real opportunity to win but did not.

## EVIDENCE
`honest_verdict`: `complete_null_sota_mapping_certification`; `verdict_class`: `null`; `sota_mapping_positive_score`: `0`; `proposal_count`: `162`; `terminal_count`: `162`; `baseline`: `frozen_syntax_only_claimed_relation`; `mean_delta`: `-0.3148148148148148`; `ci95_upper`: `-0.14814814814814814`; `mean_delta`: `-0.16666666666666666`; `ci95_upper`: `-0.037037037037037035`; `mean_delta`: `0.0`; `ci95_lower`: `0.0`; `ci95_upper`: `0.0`; `false_acceptance_count`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6958_convex_factor_energy_canary.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6959_certified_energy_selection.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed frozen evaluation found no certified positive evidence that convex-factor energy improves top-one selection over the strongest non-oracle baseline.

## WHAT WOULD REFUTE IT
Convex-factor energy achieving higher paired top-one accuracy than syntax validity, with the pair-clustered 95% confidence-interval lower bound strictly above zero and capture of at least 20% of nonzero oracle headroom.

## WAS THAT CHECKED
Yes. The same 54 groups were compared against the serious syntax-validity baseline using paired deltas and a pair-clustered confidence interval. The method instead lost by 0.037 top-one accuracy, and the interval was not strictly positive. The realized corpus had zero available oracle headroom, so the headroom-capture gate had no live opportunity; appropriately, the artifact reports only a null result, not a general claim that energy selection can never help.

## EVIDENCE
`honest_verdict`: `complete_null_certified_energy_selection`; `inference_substrate`: `frozen_candidate_label_blind_energy_selection`; `strongest_non_oracle_baseline`: `syntax_validity`; `top1_accuracy`: `0.09259259259259259`; `top1_accuracy`: `0.12962962962962962`; `mean_delta`: `-0.037037037037037035`; `ci95_lower`: `-0.09259259259259259`; `ci95_upper`: `0.0`; `strictly_above_zero`: `false`; `available_oracle_headroom`: `0`; `certified_energy_positive_score`: `0`; `verifier_is_oracle`: `false`; `evaluation_labels_sealed`: `true`; `exact_label_opened_after_selection_freeze`: `true`

## RECOMMENDATION
KEEP

## experiment_6960_certified_selection_cold_audit.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Convex-factor energy showed no positive certified-selection value over the strongest non-oracle baseline.

## WHAT WOULD REFUTE IT
A positive paired top-one delta over syntax validity, with a 95% confidence interval strictly above zero, on groups where a correct candidate existed that the baseline missed.

## WAS THAT CHECKED
No—not with a real opportunity to succeed. The paired comparison and confidence interval were computed, but the artifact reports zero available oracle headroom, so no selector could outperform the baseline on this candidate corpus.

## EVIDENCE
`honest_verdict`: `complete_null_certified_selection_cold_audit`; `strongest_non_oracle_baseline`: `syntax_validity`; `available_oracle_headroom`: `0`; `captured_oracle_headroom`: `0`; `headroom_capture_rate`: `null`; `comparison`: `convex_factor_energy_minus_syntax_validity`; `mean_delta`: `-0.037037037037037035`; `ci95_lower`: `-0.09259259259259259`; `ci95_upper`: `0.0`; `strictly_above_zero`: `false`; `raw_positive_gate_passed`: `false`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6961_certified_event_sequence.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The certified event sequence conforms successfully and demonstrates the value of the certificate-based method.

## WHAT WOULD REFUTE IT
An independently produced candidate failing exact verification, leaking a current or future certificate, or performing no better than the cheapest serious baseline—direct derivation from the visible formulations without memory—would refute the value claim.

## WAS THAT CHECKED
No. The shown rows report no inference-generated candidate: success is attached to a sealed answer mapping judged by the correctness-defining exact verifier. The no-memory comparator is acknowledged as capable of solving the tasks, but no independent comparative outcomes give the certificate method a real opportunity to lose.

## EVIDENCE
`certified_event_sequence_ready_score` `1`; `inference_ran` `false`; `sealed_answer_mapping`; `exact_success` `true`; `correct_proposal_possible_without_copying` `true`; `no_memory`; `retrieved_memory` `[]`; `split` `train`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6962_queue_regulated_self_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no headline claim to falsify. Any future positive claim would be refuted if a matched serious comparator tied or beat queue-regulated memory, or if safety outcomes failed.

## WAS THAT CHECKED
No. The run was blocked at `all_model_arm_workers`, and all outcome and comparator row collections are empty.

## EVIDENCE
`honest_verdict`: `blocked_queue_regulated_self_learning`; `verdict_class`: `blocked`; `failed_check`: `all_model_arm_workers`; `queue_learning_positive_score`: `0`; `queue_learning_run_complete_score`: `0`; `arm_rows`: `[]`; `paired_metric_rows`: `[]`; `exact_outcome_rows`: `[]`; `rows`: `[]`

## RECOMMENDATION
KEEP

## experiment_6963_queue_memory_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact is a blocked-gate receipt and makes no comparative or safety-performance claim.

## WAS THAT CHECKED
No; the safety audit did not run because its sole upstream gate failed.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "queue_learning_run_complete_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6964_v609_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed V609 capstone is disqualified because it contains flagged or conflicting evidence.

## WHAT WOULD REFUTE IT
A completed replay showing no disqualified task rows, no circular or conflicting evidence, and a passing capstone gate would falsify the claim.

## WAS THAT CHECKED
Yes. The completed gate replay evaluated all 12 terminal classification and contract rows, identified disqualified tasks 6958 and 6964, and failed the gate for flagged or conflicting evidence.

## EVIDENCE
`honest_verdict` `complete_disqualified_v609_capstone_flagged_or_conflicting_evidence` `status` `complete_disqualified` `verdict_class` `disqualified` `replay_complete` `true` `terminal_classification_rows` `12` `terminal_contract_rows` `12` `disqualified_task_numbers` `6958` `6964` `failed_check` `flagged_or_conflicting_evidence` `passed` `false`

## RECOMMENDATION
KEEP
