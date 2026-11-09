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

## experiment_8949_v765_matched_corpus_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The narrower shard-readiness assertion would fail if intended responses were missing, unauthenticated, or lacked usable confidence features.

## WAS THAT CHECKED
Yes for readiness: intended and completed counts both equal 160, and the acceptance gates pass. Oracle-defined correctness cannot establish detector benefit, but this artifact explicitly limits itself to collection custody. The positive verdict label supplies no comparative finding. Most rows are elided, limiting row-level inspection.

## EVIDENCE
- `title`: `Matched graph corpus shard A`
- `All 160 authentic responses and owned checks; pooled class support belongs to admission.`
- `Current first-response custody only; oracle truth does not establish detector benefit.`
- `methodology_note`: `Frozen matched-population shard A only. Independent graph labels are evaluation-only; detector features use backend confidence alone. No fitting, balancing, retries or pooled admission claim.`
- `honest_verdict`: `complete_circular_positive_matched_corpus_a`

## RECOMMENDATION
KEEP

## experiment_8950_v765_matched_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The collection/readiness result would fail if responses were missing, confidence features unusable, or required owned checks failed. The artifact makes no comparative detector-value claim to falsify.

## WAS THAT CHECKED
Yes, for collection: intended and completed counts are both 160, failures are zero, and usable signals and required checks are reported true. Oracle labels establish correctness labels; they do not establish detector benefit. The generalization score is zero, with no generalization success asserted.

## EVIDENCE
- `title`: `Matched graph corpus shard B`
- `No fitting, balancing, retries or pooled admission claim.`
- `Current first-response custody only; oracle truth does not establish detector benefit.`
- `All 160 authentic responses and owned checks; pooled class support belongs to admission.`
- `completed_count`: `160`
- `required_checks_passed`: `true`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Elided rows were not checked; collection readiness provides no evidence of detector benefit.

## experiment_8951_v765_matched_corpus_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The corpus admission would fail if independent references disagreed, frozen responses were missing, total errors fell below 200, or either class had fewer than five examples in any split. These are contingent admission checks; the artifact makes no comparative claim about verifier value.

## WAS THAT CHECKED
Yes, according to the admission summaries: all 320 responses completed, independent-label checks passed, 233 errors exceeded the required 200, and minimum class support was 22 against five required. Oracle-defined correctness serves labeling here, without establishing verifier benefit. Development comparisons are explicitly scoped to fixtures.

## EVIDENCE
- `title`: `Matched corpus admission`
- `Independent final-answer custody and feasibility only; no detector accuracy or inherited compute claim.`
- `independent_error_count`: `233`
- `minimum_independent_errors_required`: `200`
- `minimum_class_support_achieved`: `22`
- `minimum_required`: `5`
- `independent_reference_agrees`: `true`
- `condition`: `development_only`
- `independent_generalization_score`: `0`
- `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; elided rows were not inspected. This receipt establishes no comparative verifier benefit.

## experiment_8952_v765_matched_calibrated_decisions.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Energy calibration did not demonstrate simultaneous benefit over logistic and prevalence baselines on the 80-graph test split under the frozen acceptance rules.

## WHAT WOULD REFUTE IT
Passing all three frozen gates: positive lower bounds for Brier improvement over both baselines, and a nonpositive upper bound for accepted-error difference versus logistic at 50% diagnostic coverage.

## WAS THAT CHECKED
Yes. The paired intervals check those exact contrasts, and all three gates fail. Logistic and constant prevalence are serious cheap baselines; energy’s point-estimate Brier is slightly worse than prevalence.

Success was possible: fitting used training data, temperature selection used calibration data, and test predictions were sealed before truth opened. The verifier is not the correctness oracle. Reported counts exclude no cases; the displayed prediction rows contain the cited metric fields and show no invalid-row contradiction. This supports failure to demonstrate benefit, not equivalence or absence of any effect.

## EVIDENCE
- `honest_verdict`: `complete_null_matched_population_final_answer_calibration`; `verdict_class`: `null`.
- `test_fitting`: `false`; `temperature_selection`: `calibration Brier, ascending temperature ties`.
- `state_and_predictions_sealed_before_truth`: `true`; `verifier_is_oracle`: `false`.
- `accepted_error_vs_logistic`: `one_sided_bound` = `0.07500000000000007`; `passed` = `false`.
- `brier_vs_logistic`: `one_sided_bound` = `-0.007679164672155063`; `passed` = `false`.
- `brier_vs_prevalence`: `estimate` = `-0.00021831114692899556`; `one_sided_bound` = `-0.010454500233107525`; `passed` = `false`.
- `benefit_passed`: `false`; `detector_benefit_score`: `0`.
- `completed_count`: `80`; `excluded_count`: `0`; `censored_count`: `0`.

## RECOMMENDATION
KEEP

Not checked: elided individual rows. Risk: this population-specific null does not establish equivalence.

## experiment_8953_v765_gradient_causal_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A gradient replay benefit claim would fail if prospective Brier scores tied or lost to guarded current updates or uniform replay, or failed the prespecified retention bounds.

## WAS THAT CHECKED
No completed benefit assessment is shown; this artifact records measurements and readiness, and assigns benefit assessment to Exp8954. Serious comparator arms are present, but their identical initial predictions establish no overall tie. Oracle correctness supplies labels without establishing verifier value, and generalization is expressly disclaimed.

## EVIDENCE

- `honest_verdict`: `complete_null_gradient_causal_streams`
- `benefit_required`: `false`
- `benefit_owner`: `Exp8954`
- `Complete causal rows, unchanged methods and owned checks; loss improvement is not a gate.`
- `27648 Brier query comparisons retain typed error alongside each score;55296 endpoint cells. Exp8954 owns all11 prospective and retention bounds. Oracle graphs establish no natural-language generalization.`
- `guarded_current`, `uniform_replay`, `gradient_replay`

## RECOMMENDATION
KEEP

No rerun performed; deferred benefit conclusions remain unassessed in the supplied excerpt.

## experiment_8954_v765_gradient_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This fixed gradient-replay configuration did not meet the required future-benefit and retention acceptance criteria.

## WHAT WOULD REFUTE IT
Valid paired results passing all eleven bounds: the required future Brier improvements over the comparator arms, together with acceptable retention at every checkpoint. That would contradict the reported failure to qualify; the headline does not claim that the true effect is zero.

## WAS THAT CHECKED
Yes. The acceptance rows compare measured paired losses against explicit thresholds. Three visible future bounds fail, each sufficient to prevent qualification.

The oracle defines correctness, but the headline claims no added verifier value. Retention read-only controls are reported, evaluation units used for selection are zero, and independent generalization is explicitly disclaimed.

The cheapest serious rival, guarded current updates, is present and has lower mean future Brier than gradient replay. Uniform and signature replay controls are also present.

Visible scored rows are complete and contain the cited metric fields. False promotion eligibility concerns update admission, not evaluation validity. No wrong-key or invalid-row inclusion is demonstrated.

## EVIDENCE

- `honest_verdict`: `complete_null_gradient_benefit_audit`; `gradient_benefit_score`: `0`; `verdict_class`: `null`.
- `gradient_replay_minus_frozen_verified`: `upper`: `-0.008752952484137192`; `threshold`: `-0.01`; `passed`: `false`.
- `gradient_replay_minus_guarded_current`: `mean`: `0.007290678525250675`; `upper`: `0.02290960307702598`; `passed`: `false`.
- `gradient_replay_minus_uniform_replay`: `upper`: `0.008236456959305824`; `passed`: `false`.
- `evaluation_units_used_for_selection`: `0`; `check`: `retention_read_only`; `passed`: `true`.
- `methodology_note`: `Read-only constructed-oracle evidence; no live-model self-improvement or independent generalization.`
- `completed_count`: `27648`; `intended_count`: `27648`; `excluded_count`: `0`; `status`: `complete`.

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream execution. Risk: this scoped null does not establish zero benefit or justify rejecting continuous learning generally.

## experiment_8955_v765_gradient_complete_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Complete-service costs were measured on finite constructed workloads, but gradient replay did not establish deployment benefit.

## WHAT WOULD REFUTE IT
A quality-benefit score of 1 together with a paired complete-process cost upper bound below zero against ordinary deployment would overturn the deployment null. Incomplete timing or failed durability controls would undermine cost readiness.

## WAS THAT CHECKED
Yes, within the reported scope. Elapsed process time is measured independently of replay scoring, so a cost advantage was possible. The deployment gate requires both quality benefit and lower cost; authenticated upstream quality is zero.

Ordinary guarded deployment is the cheapest serious baseline and is present. Displayed comparisons against guarded-current and signature replay include zero, supporting no demonstrated speedup—not equivalence.

The verifier defines correctness, but no verifier-value or generalization claim is asserted. Shown cost rows are complete, with no reported exclusions. The ordinary-deployment interval and remaining rows are elided, limiting inspection.

## EVIDENCE

- `honest_verdict`: `complete_null_gradient_complete_cost`; `verdict_class`: `null`.
- `clock`: `time.monotonic_ns`; `observer_subtraction`: `false`.
- `gradient_benefit_score`: `0`; `deployment_recommendation`: `false`.
- `rationale`: `Benefit score must equal1 and the paired complete-process upper bound versus ordinary deployment must be below0; valid costs do not imply deployment benefit.`
- `against`: `guarded_current`; `lower`: `-0.4271415696108175`; `upper`: `0.495916273027484`.
- `against`: `signature_replay`; `lower`: `-0.6407267463776262`; `upper`: `0.48786941837762604`.
- `cost_rows_ready_score`: `1`; `intended_count`: `432`; `completed_count`: `432`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream quality calculations. The null does not establish equivalence or extend beyond these finite workloads.

## experiment_8956_v765_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. This artifact records blocked qualification and administrative reconciliation, so there is no claimed method advantage to falsify.

## WAS THAT CHECKED
Not applicable. The visible adversarial producer is rejected and excluded from scientific reductions. Oracle provenance does not create circularity here because no verifier benefit is claimed. FoVer readiness is explicitly scoped separately from V765 findings.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v765_capstone`
- `scientific_benefit`: `false`
- `upstream_evidence_qualified`: `false`
- `independent_generalization_score`: `0`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `disposition`: `rejected_producer`
- `admitted_to_scientific_reductions`: `false`
- `paper_ready_scope`: `FoVer paper only; stable G1-G4 do not certify V765 findings or authorize publication.`

## RECOMMENDATION
KEEP

Not checked: elided upstream rows; this verdict does not validate upstream scientific conclusions.
