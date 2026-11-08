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

## experiment_8909_v762_guarded_causal_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. The operational claim of complete causal measurement would fail if scheduled outcomes were missing, labels were accessed before predictions were sealed, or retention evaluation changed training state.

## WAS THAT CHECKED
Yes, operationally: intended and completed counts both equal 18,432, with zero failed, excluded, or censored outcomes. Displayed predictions precede label access, and displayed retention read-only checks pass.

Readiness explicitly depends on complete measurement regardless of effect. Oracle supervision therefore supports measurement here without establishing verifier value. Ordinary-update and frozen comparators are present, but no comparative benefit or independent generalization result is asserted.

Not rerun; elided rows limit direct inspection. Treating this artifact as evidence of benefit would exceed its stated scope.

## EVIDENCE

- `honest_verdict`: `complete_null_guarded_causal_streams`
- `benefit_required`: `false`
- `benefit_owner`: `Exp8911`
- `causal_rows_ready_score`: `Complete trustworthy18432 scheduled outcomes release measurement regardless of promotions or effect.`
- `methodology_note`: `Exact constructed truth is oracle supervision. Repeated operational guards are not independent risk certificates. Readiness is complete measurement; Exp8911 owns independent intervals and benefit.`
- `intended_count`: `18432`; `completed_count`: `18432`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8910_v762_finite_budget_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Adding early refusal to early acceptance did not establish shared-union label savings in this finite-budget evaluation.

## WHAT WOULD REFUTE IT
A strictly positive 99% lower bound on shared-union label savings against early acceptance, with unchanged certification decisions, no inclusion violations, and served census risks at or below 0.05.

## WAS THAT CHECKED
Yes. Paired replicate observations and shared-union cost ledgers check that contrast. The displayed replicates show zero savings; all three protocols cost 4800 labels in the displayed shared-union comparison. Early acceptance is the serious baseline, and refusal ties it. The displayed standalone learned-policy comparison also ties at 1920 labels.

Oracle correctness is not promoted into a verifier-value claim, and no independent generalization benefit is asserted. Census labels are evaluator-only after service decisions. Displayed refusals retain their costs; the cited cost fields exist, and no false validity flag is shown.

Most rows and the primary interval are elided, limiting independent inspection of the complete calculation.

## EVIDENCE
- `honest_verdict`: `complete_null_finite_budget_evaluation`; `verdict_class`: `null`
- `primary_contrast`: `shared_union/label_savings`; `shared_union_benefit_score`: `0`
- `per_replicate_results`: `shared_union/label_savings`: `0`
- `shared_union_costs`: `fixed_final`, `early_acceptance`, `early_acceptance_refusal`; `total_labels`: `4800`
- `standalone_costs`: `learned_energy`; `early_acceptance`: `1920`; `early_acceptance_refusal`: `1920`
- `decision_equality_violations`: `0`; `pathwise_inclusion_violations`: `0`; `worst_served_group_risk`: `0.0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; extrapolating this null beyond the frozen population remains unsupported.

## experiment_8911_v762_guarded_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The guarded update mechanism did not establish the prespecified joint benefit on sealed constructed-truth streams.

## WHAT WOULD REFUTE IT
All nine simultaneous bounds satisfying their frozen thresholds, with every validity control passing, would contradict the reported failed acceptance test. This null does not establish that the true effect is zero.

## WAS THAT CHECKED
Yes. Three displayed online gates fail, including comparisons against frozen predictions and ordinary updating—the cheapest serious baseline. Any one failure defeats the conjunction. The oracle flag creates no circular positive value claim here, and generalization is not asserted. Displayed scored rows are complete; the repository-health failure is explicitly separate.

## EVIDENCE

- `honest_verdict`: `complete_null_guarded_benefit_audit`; `verdict_class`: `null`; `guarded_benefit_score`: `0`.
- `guarded_true_minus_frozen_verified`: `upper` = `-0.0034586566115616746`; `threshold` = `-0.01`; `passed` = `false`.
- `guarded_true_minus_guarded_permuted`: `upper` = `0.00017286902539403215`; `passed` = `false`.
- `guarded_true_minus_ordinary_true`: `upper` = `0.018737518867359373`; `passed` = `false`.
- `exposure_scope`: `Sealed finite constructed truth; no unrecorded oracle state, pilot evaluation or unseen-topology claim.`
- `status`: `complete`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

Not checked: elided rows or producer code; this conclusion covers the supplied excerpt.

## experiment_8912_v762_complete_request_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Complete request-cost rows are available, but observer overhead fails qualification, leaving cost attribution unqualified.

## WHAT WOULD REFUTE IT
A missing or invalid scheduled measurement would refute completeness. All 36 required cells having finite simultaneous 99% overhead upper bounds ≤0.05 would refute the reported overhead qualification failure.

## WAS THAT CHECKED
Yes. The artifact reports the complete scheduled roster and compares instrumented requests against requests with the observer disabled—a relevant baseline whose measured overhead could have passed.

Displayed bounds exceed the threshold, supporting the honest null. The oracle supplies correctness without an added-value claim; model generalization is not claimed. False validity flags belong to deliberately corrupted clock controls, which are rejected.

## EVIDENCE

- `honest_verdict`: `complete_null_complete_request_cost`
- `complete_cost_rows_ready_score`: `1`; `completed_count`: `3456`; `intended_count`: `3456`
- `cost_attribution_ready_score`: `0`
- `expected`: `finite simultaneous99-percent upper<=0.05`
- `arm`: `indexed`; `accepted`: `false`; `upper`: `0.4055290861537708`
- `arm`: `scan`; `accepted`: `false`; `upper`: `0.21743967679344614`
- `instrumented`: `false`; `instrumented`: `true`
- `scope`: `injected_reduction_only`
- `scientific_benefit_claim`: `false`; `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Not independently checked: elided rows or interval calculations; this assessment is limited to the supplied excerpt.

## experiment_8913_qwen_natural_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. This artifact records a blocked prerequisite and asserts no experimental success or comparative value.

## WAS THAT CHECKED
Only prerequisites were checked. Runtime repair readiness was 0 against a required 1, blocking execution. Experimental outcomes were not checked; this receipt supports no performance conclusion.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8914_v762_guarded_durable_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The run remains disqualified for durable-cost readiness and establishes no speed/quality benefit.

## WHAT WOULD REFUTE IT
Complete valid cost measurements with all required owned checks passing would contradict cost disqualification. An authenticated positive benefit gate together with a guarded-versus-frozen service-time interval entirely below zero would contradict the absence of established benefit.

## WAS THAT CHECKED
Yes, at the reported gate level: required checks fail, authenticated benefit is zero, and the guarded-versus-frozen interval spans zero. The frozen verified arm supplies a serious comparator. Oracle-defined correctness supports no added-verifier-value claim here, and independent generalization is explicitly unclaimed. The failed qualification is preserved in the verdict.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_guarded_durable_cost`
- `adaptation_cost_ready_score`: `0`; `required_checks_passed`: `false`; `censor_reason`: `unfinished_owned_work`
- `benefit_gate`: `authenticated`: `true`; `expected`: `1`; `observed`: `0`; `check`: `observed_zero`
- `paired_total_service_intervals`: `arm`: `guarded_true`; `control`: `frozen_verified`; `lower_ns`: `-1039895692.2749208`; `upper_ns`: `650763317.9415876`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Not checked: elided rows or implementation code. Risk: detailed filtering and gate calculations remain unaudited in this excerpt.

## experiment_8915_v762_hardware_operation_budget.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An admitted compatible-kernel measurement demonstrating matched whole-service acceleration would contradict the reported absence of acceleration evidence. The artifact makes no comparative performance claim.

## WAS THAT CHECKED
No physical comparison was performed. Evidence admission was checked: the rejected producer remains inadmissible, its clocks are diagnostic only, and compatible fractions and acceleration ceilings remain unknown. The outcome is blocked rather than positive.

## EVIDENCE

- `honest_verdict`: `complete_blocked_hardware_operation_budget`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `physical_acceleration_demonstrated`: `false`
- `physical_probes_performed`: `0`
- `claim`: `analytical ceiling unknown; no measured device speed or service gain`
- `source_admitted`: `false`
- `missing_reason`: `rejected_producer`

## RECOMMENDATION
KEEP

Not checked: hardware performance or elided rows; this review establishes no speedup.

## experiment_8916_v762_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. A scheduled slot lacking primary or skip custody would contradict the administrative accounting statement, but would not test method value.

## WAS THAT CHECKED
Administrative accounting is marked passed. Scientific benefit is not established: natural generalization fails its gate, and fresh evaluation remains unopened. Oracle-based upstream checks are not presented as evidence of added value.

## EVIDENCE
- `aggregation_from_upstream_artifacts`
- `Administrative custody and method ingestion only; no new scientific benefit, natural-text or hidden-game generalization.`
- `constructed development pilots only; fresh evaluation remains unopened`
- `complete_accounting`: `passed` = `true`
- `natural_generalization`: `passed` = `false`
- `activation_authorized` = `false`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream computations; this administrative receipt establishes no comparative benefit.
