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
| NO_CLAIM | 4 |

## experiment_8962_v766_graph_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a corpus collection receipt, with no comparative detector or verifier claim. Its narrower readiness assertion would fail if required responses were missing, unauthenticated, unusable, or an owned check failed.

## WAS THAT CHECKED
Yes, for readiness: the artifact reports 512 completed responses, zero failures, usable signals, and passing owned checks. Those checks could have failed.

Oracle labels establish answer correctness, not verifier benefit; no such benefit is asserted. The frozen protocol names prevalence and logistic controls, but reports no comparative results here. The generalization score is not presented as demonstrated generalization.

## EVIDENCE
- `title`: `Matched graph corpus shard B`
- `Current first-response custody only; oracle truth does not establish detector benefit.`
- `methodology_note`: `Frozen matched-population shard B only. Independent graph labels are evaluation-only; detector features use backend confidence alone. No fitting, balancing, retries or pooled admission claim.`
- `completed_count`: `512`; `failed_count`: `0`
- `usable_signals`: `true`; `required_checks_passed`: `true`
- `verdict_class`: `circular_positive`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

Not checked: elided rows or external receipts. Risk: treating corpus readiness as detector benefit would exceed this evidence.

## experiment_8963_v766_corpus_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a corpus-admission receipt. Its readiness assertion would fail if required class support were insufficient, scheduled first responses were missing, or oracle-sidecar mutations changed detector inputs or prelabel scores.

## WAS THAT CHECKED
Yes, for admission: intended, completed, and independent counts are all 1,024; both classes exceed the minimum in every role; and four oracle-mutation controls pass. These checks could fail independently of oracle agreement.

No comparative verifier benefit or generalization claim is made. Review is limited to the supplied artifact; referenced files were not inspected.

## EVIDENCE
- `task_id`: `exp8963-corpus-admission`
- `metric`: `response_admitted`
- `Every admission gate is required; oracle agreement certifies labels, not learned benefit.`
- `independent_generalization_score`: `0`
- `mutate_label`, `mutate_expected`, `mutate_distances`, `mutate_answer_set`
- `required_checks_passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8964_v766_relational_decisions.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Relational energy did not demonstrate the registered comparative benefit on 128 fresh matched graphs.

## WHAT WOULD REFUTE IT
Passing all five registered superiority bounds: Brier improvements against prevalence and both logistic controls, plus lower accepted error at 50% coverage against both logistic controls.

## WAS THAT CHECKED
Yes. All five paired comparisons were evaluated using 10,000 graph bootstrap draws with multiplicity correction; all failed. Serious inexpensive rivals were included: relational logistic achieved accepted error of 0.59375 versus relational energy’s 0.765625, and both prevalence and relational logistic had better Brier point estimates.

The reported controls separate calibration from test evaluation, seal predictions before test labels, and isolate the correctness oracle. The comparison could therefore have supported benefit; the reported null was not predetermined. No excluded or censored observations are reported, and typed acceptance with zero coverage correctly has an undefined error rate.

## EVIDENCE
- `honest_verdict`: `complete_null_fresh_matched_graph_final_answers`
- `verdict_class`: `null`; `detector_benefit_score`: `0`
- `benefit_passed`: `false`
- `multiplicity`: `Bonferroni five one-sided percentile bounds`
- `relational_energy`: `diagnostic_50`, `value`: `0.765625`
- `relational_logistic`: `diagnostic_50`, `value`: `0.59375`
- `state_and_predictions_sealed_before_truth`: `true`
- `verifier_is_oracle`: `false`
- `excluded_count`: `0`; `censored_count`: `0`
- `typed_accept`: `coverage`: `0.0`, `value`: `null`

## RECOMMENDATION
KEEP

Elided rows were not individually audited; this null establishes no demonstrated benefit, not equivalence.

## experiment_8965_v766_relational_controls.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Integrity controls passed on the frozen 128-graph panel, while added detector benefit was not demonstrated.

## WHAT WOULD REFUTE IT
Required comparisons against prevalence and logistic baselines clearing the corrected benefit gates would overturn the reported null. Predictions changing under private-oracle mutations, failed access restrictions, or permutation drift exceeding the accepted tolerance would refute control readiness.

## WAS THAT CHECKED
Yes. All five retained benefit contrasts failed their gates, including comparisons against serious baselines: prevalence, confidence logistic, and relational logistic. Prevalence and relational logistic have favorable point estimates relative to the detector on Brier score. Oracle-mutation rows report unchanged predictions, custody and invariance checks pass, and all 128 graphs count without exclusions. The artifact reports a null rather than converting successful integrity checks into a positive value claim.

## EVIDENCE
- `honest_verdict`: `complete_null_relational_integrity_controls`; `verdict_class`: `null`.
- `benefit_passed`: `false`; `detector_benefit_score`: `0`; `independent_generalization_score`: `0`.
- `brier_vs_prevalence`: `estimate`: `-0.0011287238588903181`.
- `brier_vs_relational_logistic`: `estimate`: `-0.009519397017716952`.
- `truth_mutations_unchanged`: `true`; `file_and_traversal_denied`: `true`; `baseline_reproduced`: `true`.
- `maximum_probability_drift`: `1.7881393432617188e-07`; `completed_count`: `128`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

Elided rows and execution were not independently checked. The supported null means benefit was not demonstrated; it does not establish equivalence.

## experiment_8966_v766_online_response_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. The operational readiness assertion would fail if mutating unreleased labels changed earlier prediction seals, or restarting produced a different final state.

## WAS THAT CHECKED
Yes, for readiness: displayed mutation controls preserve prediction seals, and displayed restart rows match control and resumed state hashes. Scientific benefit was explicitly unevaluated. The frozen and bias-only arms provide serious comparators, but their displayed initial ties establish no overall result.

## EVIDENCE
- `scientific_benefit_evaluated`: `false`
- `online_benefit_score`: `null`
- `honest_verdict`: `complete_null_sealed_real_error_online_calibration`
- `future_label_seals_unchanged`: `true`
- `retention_label_seals_unchanged`: `true`
- `online_rows_ready_score`: `1`

## RECOMMENDATION
KEEP

Unchecked: elided rows. Risk: readiness does not establish scientific benefit.

## experiment_8967_v766_online_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The fixed V766 adapter failed its prespecified online-benefit criteria, supporting retirement of this exact protocol.

## WHAT WOULD REFUTE IT
Retained paired losses satisfying all nine acceptance bounds, with discriminating feedback and nonzero frozen headroom, would contradict the reported null. Failure to establish benefit does not establish that the true effect is zero.

## WAS THAT CHECKED
Yes. The displayed bounds compare adaptive performance against frozen, bias-only and permuted arms across 24 independent histories. All three fail; any one failure defeats the all-nine criterion. Bias-only is a cheap serious baseline, and adaptive superiority over it is not established.

Feedback differences and frozen headroom show that improvement had a real opportunity to occur. Oracle-defined correctness does not make this comparative null circular. The artifact explicitly limits its exposure claims, and reported counts show no exclusions or censoring.

## EVIDENCE
- `honest_verdict`: `complete_null_real_response_online_benefit_audit`
- `principle`: `All nine prespecified bounds must pass; retention shares one graph draw across histories.`
- `adaptive_minus_frozen`: `upper`: `0.0008872641892848348`; `threshold`: `-0.01`; `passed`: `false`
- `adaptive_minus_bias_only`: `upper`: `0.0004791929617041229`; `threshold`: `0`; `passed`: `false`
- `adaptive_minus_permuted`: `estimate`: `1.3710351995199682e-05`; `passed`: `false`
- `feedback_blocks_differ`: `116`; `frozen_future_brier_headroom`: `0.19539885350820918`
- `intended_count`: `39936`; `completed_count`: `39936`; `excluded_count`: `0`; `censored_count`: `0`
- `action`: `retire_exact_protocol`; `self_learning_in_general_retired`: `false`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream execution; this verdict supports only the scoped failure to meet acceptance criteria.

## experiment_8968_v766_complete_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Adaptive service does not demonstrate a favorable complete-cost/quality tradeoff against confidence logistic on the cached workload.

## WHAT WOULD REFUTE IT
An adaptive-minus-confidence-logistic timing interval with its upper bound below zero, together with all three authenticated quality gates equal to one, would overturn this null conclusion.

## WAS THAT CHECKED
Yes. Across 24 paired process seeds, adaptive was slower by 2.539 seconds on average, with an interval entirely above zero: 2.122–2.956 seconds. The serious, cheap baseline—two-feature confidence logistic—wins this cost comparison.

The quality scores also fail the stated benefit requirements. The verifier is not the oracle, and the artifact claims no independent generalization. All 96 scheduled units completed without exclusions or censoring. The reported null agrees with these results.

## EVIDENCE
- `honest_verdict`: `complete_null_cached_relational_complete_service_cost`; `verdict_class`: `null`.
- `arm`: `adaptive`; `against`: `confidence_logistic`; `metric`: `complete_process_s`; `independent_count`: `24`.
- `mean`: `2.539286065333333`; `lower`: `2.1221246880805706`; `upper`: `2.9564474425860956`.
- `detector_benefit_score`: `0`; `online_benefit_score`: `0`; `independent_generalization_score`: `0`.
- `completed_count`: `96`; `excluded_count`: `0`; `censored_count`: `0`; `failed_count`: `0`.
- `verifier_is_oracle`: `false`; `favorable`: `false`; `deployment_recommendation`: `false`.

## RECOMMENDATION
KEEP

Elided rows and source code were not inspected; this verdict covers only the supplied cached-workload evidence.

## experiment_8969_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked prerequisite check, not a conclusion about held-family transfer.

## WAS THAT CHECKED
The prerequisite was checked and failed: readiness was 0 against a required 1. Execution stopped before transfer evaluation, so transfer remains unassessed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `arc_new_outcomes_ready_score`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP
