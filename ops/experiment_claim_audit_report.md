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

## experiment_8435_v726_capacity_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This retrospective attempt did not establish a capacity-8 uniform-update benefit over FIFO updates and a frozen predictor.

## WHAT WOULD REFUTE IT
Both FIFO-minus-uniform and frozen-minus-uniform Brier contrasts having strictly positive conditional 95% lower bounds, while the decision-loss, retention, support and control gates also passed. This is a failure-to-demonstrate claim, not proof that benefit is impossible.

## WAS THAT CHECKED
Yes. Both contrasts were measured and their intervals cross zero; both benefit gates failed. The other scientific gates passed, so a positive result had a real chance to qualify.

Issued predictions are scored against evaluator targets and compared across nondegenerate controls; benefit is not guaranteed by parameter updates. The cheapest serious baseline is the frozen predictor. Both frozen and FIFO outperform uniform on measured Brier, supporting the reported null.

The verifier defines correctness, but the artifact claims no positive verifier value. It explicitly limits uncertainty to exposed histories and reports no generalization benefit. Block 0’s false later-loss eligibility identifies its exclusion from later contrasts, not invalid data. The cited metric fields exist; intended and completed counts match, with zero reported exclusions or censoring.

## EVIDENCE

- `honest_verdict`: `complete_null_v726_capacity_benefit_audit`; `scientific_outcome`: `null`.
- `fifo_minus_uniform_brier`: `-0.000839919183498522`; `conditional_ci95` endpoints: `-0.0032621722408660804`, `0.0013780320158046908`.
- `gain_vs_frozen`: `estimate` of `-0.00018457138267420078`; `conditional_ci95` endpoints: `-0.0010567593336667806`, `0.0007039343563264691`.
- `primary_lower95_positive`: `false`; `gain_lower95_positive`: `false`; `controls_qualified`: `true`.
- `verifier_is_oracle`: `true`; `generalized_learning_benefit_score`: `0`.
- `uncertainty_scope`: `Conditional descriptive uncertainty for exposed issued histories; adaptive streams are dependent; not a population generalization confidence interval.`
- `intended_count`: `12000`; `completed_count`: `12000`; `excluded_count`: `0`; `censored_count`: `0`.

## RECOMMENDATION
KEEP

## experiment_8436_v726_transaction_attribution.json

**CLAIM_SUPPORTED**

## VERDICT

CLAIM_SUPPORTED

## THE HEADLINE CLAIM

The retrospective transaction attribution is complete but establishes no net speedup, native consumer qualification, or independent generalization benefit.

## WHAT WOULD REFUTE IT

A valid transaction showing end-to-end speedup above 1 against the original complete-work baseline after charging all transaction costs, or a native transaction passing the required downstream consumer checks.

## WAS THAT CHECKED

Yes, for the scoped transaction and native claims. Valid Python rows compare transaction costs with primitive savings and original complete-work receipts; all reported speedups are below 1. Native rows fail consumer validation and carry null speedups, so they support rejection rather than performance measurement.

Administrative readiness is gated separately from measured gain. The verifier is not the correctness oracle. No independent generalization test is shown, and the artifact claims no such benefit from its exposed retrospective inputs.

## EVIDENCE

- `honest_verdict`: `complete_null_transaction_attribution`
- `verdict_class`: `null`
- `point_speedup`: `0.28312438882742363`, `0.18081672883543048`, `0.08174781686626166`
- `end_to_end_speedup_supported`: `false`
- `cost_valid`: `false`; `point_speedup`: `null`
- `native_consumer_schema_failure_reproduced`: `true`
- `qualification_or_speed_claim_allowed`: `false`
- `verifier_is_oracle`: `false`
- `exposure_scope`: `retrospective_exposed_cached_cost_operands`
- `independent_count`: `0`
- `ready_score_depends_on_gain`: `false`
- `completed_count`: `135`; `failed_count`: `0`

## RECOMMENDATION

KEEP

## experiment_8437_v726_acceleration_ceiling.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
For the measured primary workload, arithmetic-only acceleration cannot deliver a 10× service speedup while the remaining work and durability stay unchanged.

## WHAT WOULD REFUTE IT
A valid primary repeat with arithmetic fraction K/T above 0.9 would permit 10× service acceleration at a sufficiently large finite arithmetic speedup, contradicting the claimed ceiling.

## WAS THAT CHECKED
Yes, through the reported accelerable-fraction range and primary-repeat ceiling checks. The maximum reported fraction is approximately 0.0008053, far below 0.9. The supplied excerpt provides summaries rather than individual repeat rows.

This is a conditional analytical null whose outcome depends on measured time allocation. It makes no verifier-value, generalization, or comparative-method claim requiring a rival arm.

## EVIDENCE

- `arithmetic_10x_feasible_under_fixed_service`: `false`
- `every_primary_repeat_rules_out_arithmetic_10x`: `true`
- `fraction_accelerable`: `range` endpoints `0.0007178643853389116` and `0.000805280851658791`
- `descriptive_interval` → `arithmetic` → `infinity` → `pooled_point`: `1.000766832972139`
- `unchanged_service`: `true`; `identical_work_and_durability`: `true`
- `native_speed_measured`: `false`; `independent_deployment_samples`: `false`

## RECOMMENDATION
KEEP

## experiment_8438_bounded_qwen_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The recorded blocking outcome would be contradicted if both prerequisite gate rows showed passing results.

## WAS THAT CHECKED
Yes, for the blocking outcome: the runtime-repair gate failed with actual 0 versus expected 1; the feature-panel gate passed. This is a prerequisite-check receipt, not a canary performance result.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_reason`: `actual=0 == expected=1`
- `gate_check_summary`: `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8429-contract-methods.runtime_repair_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8439_v726_arc_supervisor_support.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. The artifact records blocked qualification; it does not claim supervisor superiority, ineffectiveness, or generalization.

## WAS THAT CHECKED
Not applicable. A serious comparator—supervisor off—is present and ties supervisor on at zero progress and solve credit. All redirect outcomes remain censored, so this establishes neither benefit nor definitive ineffectiveness. The artifact leaves support blocked and refinement unauthorized.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v726_arc_supervisor_support`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `arc_support_ready_score`: `0`; `arm_refinement_authorized`: `false`
- `arm`: `off` and `on`; `progress_count`: `0`; `headline_solve_credit`: `0`
- `outcome_window_status`: `pending_censored`
- `methodology_note`: `Reconstruct original live-wrapper engineering receipts from independently pinned primitive observations. Shadow redirects are counterfactual diagnostics; all pending outcomes remain right-censored. Two exposed games cannot prove hidden-game generalization. No justified policy change.`

## RECOMMENDATION
KEEP

## experiment_8440_v726_board_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a reader-qualification receipt that makes no comparative benefit claim. An accepted corrupted source would contradict reader qualification, but would not establish anything about hardware or learning benefit.

## WAS THAT CHECKED
Yes, qualification checks are recorded in historical_consumer_receipts: historical assertions and source-drift/tamper controls passed. No comparative value claim is made, so the oracle flag does not create a circular value claim.

## EVIDENCE
- `actual_substrate`: `host_CPU_aggregation`
- `Bind exact task, producer hashes and primitive operands; qualification grants no hardware or scientific benefit.`
- `scientific_benefit`: `false`
- `full_service_speedup`: `null`
- `generalized_learning_benefit_score`: `0`
- `original_source_drift_and_tamper_controls`, `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8441_v726_gatemate_obligation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The descriptive blocked status would be contradicted by authenticated historical sources and completed physical-change, IDCODE, flash, and device-smoke evidence in the artifact’s own rows.

## WAS THAT CHECKED
No comparative test applies. The rows explicitly record one completed continuity item and six censored obligations. No new evidence was supplied to challenge the blocked status. Oracle-defined verification is not presented as added value, and no generalization benefit is asserted.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_obligation`
- `verdict_class`: `blocked`
- `scientific_benefit`: `false`
- `generalized_learning_benefit_score`: `0`
- `completed_count`: `1`
- `censored_count`: `6`
- `missing_reason`: `exact historical bytes not supplied`, `physical change not supplied`, `future device observation not supplied`
- `supplied_evidence_delta`: `[]`

## RECOMMENDATION
KEEP

## experiment_8442_v726_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied artifact reports administrative reconciliation and blocked scientific readiness, without asserting comparative benefit or independent generalization.

## WAS THAT CHECKED
No comparative scientific test is shown or claimed. The visible readiness checks could fail and did fail; scheduling-slot completion is kept separate from scientific success. The oracle flag therefore does not make an asserted value claim circular.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v726_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `scientific_claims_independently_generalize`: `false`
- `independent_count`: `0`
- `independent_generalization_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `arm`: `scheduling_slot`
- `upstream_scientific_verdict`: `expected`: `measured_qualified_branch`; `observed`: `blocked_gate_check_failed`; `passed`: `false`

## RECOMMENDATION
KEEP
