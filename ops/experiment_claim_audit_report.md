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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 6 |

## experiment_8922_v763_interpolated_causal_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. Such a claim would fail if interpolated_true tied or lost to frozen_verified or guarded_true on future Brier loss, or exceeded the retention degradation limit.

## WAS THAT CHECKED
No completed benefit analysis is shown. Those comparisons appear in the evaluation plan, while benefit inference is explicitly assigned to Exp8925. The displayed retention controls address causal custody.

## EVIDENCE
- `honest_verdict`: `complete_null_interpolated_causal_streams`
- `benefit_required`: `false`
- `methodology_note`: `Original three-class Brier loss; readiness measures causal custody only. Exp8925 owns benefit inference.`
- `independent_generalization_score`: `0`
- `check`: `retention_read_only`; `passed`: `true`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; readiness must not be interpreted as demonstrated benefit.

## experiment_8923_label_demand_evaluation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No empirical claim is advanced. A claim of label savings with unchanged certificate decisions would be refuted by no savings or any changed decision.

## WAS THAT CHECKED
No. This is a blocked-gate receipt. The upstream readiness check failed; label savings and certificate-decision equivalence were not evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `label_demand_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8924_v763_qwen_natural_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Missing generations or paired measurements would contradict the operational completeness report. Parse failures, invalid spans, or poor agreement would not: readiness explicitly does not imply output quality or scientific benefit.

## WAS THAT CHECKED
Yes, at the reported operational level: 24 intended cases completed, with 24 completed generation calls. Two parse failures are disclosed. The artifact makes no comparative benefit, verifier-value, or independent generalization claim; its null designation does not assert statistical equivalence.

## EVIDENCE
- `honest_verdict`: `complete_null_paired_criterion_canary`
- `verdict_class`: `null`
- `Complete operational paired evidence, independent of output quality or scientific benefit.`
- `intended_count`: `24`; `completed_count`: `24`; `generation_calls_completed`: `24`
- `parse_failures`: `2`
- `independent_count`: `0`; `preferred_criterion`: `null`
- `Exact offsets locate retained source bytes; semantic correctness is unmeasured.`

## RECOMMENDATION
KEEP

Elided rows were not inspectable; this review is limited to the supplied excerpt.

## experiment_8925_v763_interpolated_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The fixed interpolated optimizer, guard and schedule did not meet the prespecified prospective-benefit and retention conjunction.

## WHAT WOULD REFUTE IT
All nine simultaneous bounds passing: future Brier improvement versus frozen of at least 0.01, strict improvement versus both permuted and guarded arms, and retention Brier/error increases bounded by 0.01 at every checkpoint.

## WAS THAT CHECKED
Yes. Paired-stream intervals test those gates across 24 streams. All three future gates fail; retention gates pass. This supports failure to establish the required benefit, not proof of zero benefit.

The cheapest serious baseline—keeping the frozen predictor—is present. The guarded comparator also achieves better mean Brier than the interpolated candidate. Oracle-defined correctness creates no circular verifier-value claim here, and no generalization success is asserted.

Displayed rows contain the scored metric and show no false validity flag. Elided rows cannot be independently checked from this excerpt.

## EVIDENCE

- `honest_verdict`: `complete_null_interpolated_benefit_audit`; `interpolated_benefit_score`: `0`; `verdict_class`: `null`.
- `interpolated_true_minus_frozen_verified`: `upper`: `-0.0006505695069113319`; `threshold`: `-0.01`; `passed`: `false`.
- `interpolated_true_minus_interpolated_permuted`: `upper`: `0.011558705667271879`; `threshold`: `0`; `passed`: `false`.
- `interpolated_true_minus_guarded_true`: `mean`: `0.012901702455741654`; `upper`: `0.03395365867612195`; `passed`: `false`.
- `retention_bounds`; `retention_checkpoints`: `0`, `24`, `48`; `retention_upper`: `0.01`.
- `independent_generalization_score`: `0`; `continuous_learning_in_general_retired`: `false`.

## RECOMMENDATION
KEEP

## experiment_8926_v763_durable_cost_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The retained historical guarded-service measurements establish no speed–quality advantage.

## WHAT WOULD REFUTE IT
A reliable whole-service cost reduction against frozen_verified with preserved quality, or an authenticated quality improvement sufficient to justify its measured cost.

## WAS THAT CHECKED
Yes, within the historical scope: all three paired cost intervals against frozen_verified span zero, and the authenticated benefit gate reports zero benefit. This supports the limited null conclusion, not equivalence.

The oracle verifier is not credited with added value, and no generalization benefit is asserted. Displayed comparison rows contain the cited cost fields; the failed repository-health receipt is explicitly outside comparisons.

## EVIDENCE
- `honest_verdict`: `complete_null_durable_cost_qualification`
- `verdict_class`: `null`
- `benefit_gate`: `authenticated`: `true`, `observed`: `0`
- `arm`: `guarded_true`, `control`: `frozen_verified`, `lower_ns`: `-1039895692.2749208`, `upper_ns`: `650763317.9415876`
- `speed_quality_recommendation_score`: `0`
- `independent_generalization_score`: `0`
- `scope`: `unrelated_repository_health`

## RECOMMENDATION
KEEP

Not checked: elided rows or linked artifacts.
Risk: these historical null results do not establish equivalence.

## experiment_8927_inline_request_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. This is a blocked-gate receipt, not a claim about measured request costs or the observer’s added value.

## WAS THAT CHECKED
No request-cost claim was tested. The recorded readiness gate failed: observed 0 versus required 1. The artifact contains no cost measurements or comparisons to assess.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `inline_clock_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8928_v763_hardware_operation_budget.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A valid matched-workload receipt demonstrating whole-service acceleration, including host, transfer and durability costs, would contradict the inventory’s recorded absence of measured acceleration.

## WAS THAT CHECKED
No hardware performance comparison is shown. The artifact records zero physical probes, unknown compatible fractions and a failed acceleration gate, then reports a blocked outcome. Its successful mapping audit does not assert acceleration. Rejected producers remain separate from admitted evidence.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_operation_budget`
- `Valid mapping evidence does not establish acceleration`
- `physical_acceleration_demonstrated`: `false`
- `physical_probes_performed`: `0`
- `feasibility`: `unknown_without_measured_compatible_fraction`
- `inference_substrate`: `aggregation_from_upstream_artifacts`

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream contents. Hardware performance remains unmeasured.

## experiment_8929_v763_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. This capstone records administrative reconciliation and blocked scientific gates; it does not claim verifier benefit or generalization.

## WAS THAT CHECKED
Not applicable. Failed producer gates are recorded, independent generalization remains zero, and the inherited circular pilot result is not promoted to a positive capstone verdict.

## EVIDENCE
`honest_verdict`: `complete_blocked_v763_capstone`; `scientific_benefit`: `false`; `milestone_paper_ready`: `false`; `independent_generalization_score`: `0`; `inference_substrate`: `aggregation_from_upstream_artifacts`; `prior_verdict`: `circular_positive`.

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts; this assessment covers only the supplied excerpt.
