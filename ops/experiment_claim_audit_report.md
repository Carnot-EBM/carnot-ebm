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
| NO_CLAIM | 5 |

## experiment_8971_v767_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An authority mismatch or failed required check accepted as ready would contradict administrative readiness. There is no comparative scientific benefit claim to falsify.

## WAS THAT CHECKED
Yes, within the administrative scope: displayed task rows match, seven authority mutation controls report passing, and owned validation receipts pass. Scientific generalization is explicitly assigned zero support; optional ARC evidence is marked unready.

## EVIDENCE

- `honest_verdict`: `complete_null_v767_branch_contract`
- `exposure_scope`: `Administrative custody and method ingestion only; no new scientific benefit, natural-text or hidden-game generalization.`
- `metric`: `ordered_task_contract_match`; `matched`: `true`
- `independent_generalization_score`: `0`
- `administrative_only`: `true`

## RECOMMENDATION
KEEP

Review limited to supplied text; no rerun or inspection of elided rows. The global test suite timed out.

## experiment_8972_v767_encoder_head_diagnosis.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The exposed-development encoder/head diagnosis does not establish an energy-method benefit.

## WHAT WOULD REFUTE IT
Energy outperforming prevalence and the confidence and relational logistic comparators, with all five original paired-bound criteria passing, would contradict the reported failure to demonstrate benefit.

## WAS THAT CHECKED
Yes. The original bound rows and paired intervals retain all five failed comparisons, including serious cheap baselines. Scientific benefit is separated from diagnostic readiness, so successful execution does not become a positive scientific verdict.

The verifier is not the correctness oracle. Label exposure is acknowledged, and independent generalization receives zero credit. The visible prediction rows are completed; the failed comparison flags represent retained outcomes, not invalid rows silently excluded.

## EVIDENCE
- `honest_verdict`: `complete_null_exposed_encoder_head_diagnosis`
- `verdict_class`: `null`
- `benefit_passed`: `false`
- `brier_vs_prevalence`: `one_sided_bound`: `-0.005943697107204875`
- `brier_vs_relational_logistic`: `estimate`: `-0.009519397017716952`; `one_sided_bound`: `-0.018690732577001417`
- `verifier_is_oracle`: `false`
- `exposure_scope`: `all_V765_V766_graph_labels_exposed_development_only`
- `independent_generalization_score`: `0`
- `methodology_note`: `Original five failed bounds retain their interpretation.`

## RECOMMENDATION
KEEP

Elided rows were not audited; this verdict supports the reported failure to demonstrate benefit, not proof that no benefit exists.

## experiment_8973_v767_shared_feature_methods.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The shared-feature Gibbs classifier failed development admission on 64 exposed guard graphs.

## WHAT WOULD REFUTE IT
Those graphs satisfying all three admission requirements: AUROC ≥ 0.65, Brier improvement over prevalence ≥ 0.005, and Brier excess over shared-feature logistic regression ≤ 0.01.

## WAS THAT CHECKED
Yes. The three development-admission rows explicitly compare measured values with those thresholds; all fail. Predictions are scored against candidate error labels, so passing was not guaranteed by construction.

The cheap serious rival—shared-feature logistic regression—beats Gibbs on Brier score; even prevalence beats Gibbs. This supports the reported admission failure. The verifier is not the correctness oracle, and the scores are explicitly exposed-development measurements, without a positive generalization claim.

The displayed scoring rows are completed, and the admission fields cited exist. Review is limited to the supplied, partly elided artifact; fresh confirmation was not measured.

## EVIDENCE
- `honest_verdict`: `complete_null_development_admission_failed`; `verdict_class`: `null`.
- `shared_energy_auroc`: `value` = `0.3502604166666667`, `threshold` = `0.65`, `passed` = `false`.
- `brier_improvement_over_fit_prevalence`: `value` = `-0.012695407226601063`, `threshold` = `0.005`, `passed` = `false`.
- `brier_excess_over_shared_logistic`: `value` = `0.020722842744012537`, `threshold` = `0.01`, `passed` = `false`.
- `arm_brier`: `shared_energy` = `0.20094308789066356`, `shared_logistic` = `0.18022024514665103`, `prevalence` = `0.1882476806640625`.
- `scope`: `exposed_development_only`; `independent_count`: `64`; `verifier_is_oracle`: `false`; `fresh_collection_permitted`: `false`.

## RECOMMENDATION
KEEP

## experiment_8974_feature_online_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or learning outcome is asserted, so there is no scientific claim to falsify.

## WAS THAT CHECKED
Only the prerequisite gate was checked: readiness had to equal 1, but was 0. The artifact records a block before method evaluation.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `shared_methods_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8975_v767_natural_label_eligibility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The reported eligibility blockage would be contradicted by documented authorization and all 800 valid source-text/label joins.

## WAS THAT CHECKED
Yes, for eligibility: the access search and gate checks report missing authorization and zero joins. All 800 slots are censored, with zero completed pairs. No model was invoked; the artifact makes no performance, generalization, or verifier-value claim requiring a rival baseline.

## EVIDENCE

- `honest_verdict`: `complete_blocked_source_access`
- `verdict_class`: `blocked`
- `completed_count`: `0`
- `censored_count`: `800`
- `model_invoked`: `false`
- `No model fitting, ranking, energy benefit or no-cheap-oracle claim is measured.`

## RECOMMENDATION
KEEP

Not rerun; review is limited to the supplied artifact and establishes no model benefit.

## experiment_8976_graph_corpus_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. This artifact records a blocked prerequisite gate and makes no claim about method performance or added value.

## WAS THAT CHECKED
No performance claim was tested. The prerequisite gate was checked and failed: observed 0 versus required 1. Execution stopped at the recorded pre-gate layer.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `shared_methods_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8983_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. This artifact records a blocked prerequisite gate and makes no claim about transfer performance.

## WAS THAT CHECKED
The readiness gate was checked and failed: expected 1, observed 0. No transfer-performance evaluation is reported.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `arc_new_outcomes_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8984_v767_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The fixed shared-feature Gibbs head failed its exposed-development admission criteria.

## WHAT WOULD REFUTE IT
Passing all three admission gates: AUROC ≥0.65, Brier improvement over fitted prevalence ≥0.005, and Brier excess over shared logistic ≤0.01.

## WAS THAT CHECKED
Yes—in branch_outcomes.shared_feature.admission_rows. All three gates failed. Serious cheap comparators were present: both fitted prevalence and shared-feature logistic achieved lower Brier loss than the energy head.

This supports the bounded negative result. Development exposure is explicit, no positive verifier-value claim depends on the oracle, and failed gates remain failures in the headline.

## EVIDENCE

- `honest_verdict`: `complete_blocked_external_science_and_source_access`
- `scope`: `exposed_development_only`
- `shared_energy_auroc`; `value`: `0.3502604166666667`; `threshold`: `0.65`; `passed`: `false`
- `energy_brier`: `0.20094308789066356`
- `logistic_brier`: `0.18022024514665103`
- `prevalence_brier`: `0.1882476806640625`
- `broader_hypothesis_retired`: `false`

## RECOMMENDATION
KEEP

Not checked: upstream artifacts or elided rows. This conclusion covers only the reported development result.
