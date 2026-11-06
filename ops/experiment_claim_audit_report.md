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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |

## experiment_8868_v759_conditional_risk_methods.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The scoped conditional-risk evaluation reports an exact paired coverage tie and checked methods readiness, without claiming superiority or independent generalization.

## WHAT WOULD REFUTE IT
A nonzero paired coverage difference would contradict the reported exact tie. Failed separation, failed controls, or an incorrectly constructed certificate would undermine methods readiness.

Any future added-value claim would need to beat a simple query-only acceptance rule using the same exact certification without the trained energy head.

## WAS THAT CHECKED
Yes, at the reported-summary level: 40 independent replicates report zero paired differences, controls passed, and no certificate failures were recorded. Selection precedes independent certification, with the population census evaluated afterward, giving discrepancies a chance to appear.

The oracle defines correctness, but the artifact claims neither verifier benefit nor independent generalization. The displayed stress rows with false validity flags cannot establish guarantees; no universal guarantee is claimed.

The elided rows prevent checking comparator identities and aggregation filters directly. That limits auditability without contradicting this descriptive null.

## EVIDENCE

- `honest_verdict`: `complete_null_conditional_risk_methods`
- `verdict_class`: `null`
- `paired_coverage_difference`: `0.0`
- `independent_replicates`: `40`
- `certificate_failure_count`: `0`
- `benefit_required`: `false`
- `methodology_note`: `Familywise .01 applies per replicate. Readiness is valid methods, not benefit. Oracle labels cannot establish independent generalization.`
- `verifier_is_oracle`: `true`
- `independent_generalization_score`: `0`
- `validity_guarantee`: `false`
- `scope`: `descriptive finite census; no superiority or universal guarantee claim`

## RECOMMENDATION
KEEP

Unchecked: elided comparator rows and aggregation filters; support remains limited to the stated descriptive null.

## experiment_8869_matched_memory_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite gate and makes no comparative or performance claim.

## WAS THAT CHECKED
The prerequisite gate was checked and failed: diagnosis_ready_score was 0 against required 1. Method performance was not tested; interpreting this receipt as successful qualification would be unsupported.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `diagnosis_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8870_v759_conditional_risk_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Learned-energy policies meet the conditional utility gates, but grouping demonstrates no robustness advantage over pooling in these constructed oracle-labeled worlds.

## WHAT WOULD REFUTE IT
A grouped-versus-pooled excess-risk reduction with simultaneous 99% lower bound above zero, while grouped utility passes, would overturn the comparative null. Coverage below the required floor or a served-group risk above 0.05 would contradict the utility claim.

## WAS THAT CHECKED
Yes, through paired replicate comparisons and policy-specific utility summaries. The displayed paired reductions are zero; both learned-energy protocols pass utility. Pooling the same learned scores is the cheapest serious rival, and it ties.

Selection precedes certification labels, and certification precedes census labels, allowing coverage and safety failures to appear. Oracle correctness supports conditional measurement; no independent generalization or added verifier value is claimed. Rows flagged outside the validity guarantee establish observed outcomes only, not guaranteed safety.

## EVIDENCE
- `honest_verdict`: `complete_null_conditional_risk_evaluation`
- `learned_energy`: `comparison`: `parity`; `robustness_advantage`: `0`
- `utility_by_protocol`: `grouped`: `1`; `pooled`: `1`
- `paired_replicate_rows`: `worst_mixture_excess_reduction`: `0.0`; `balanced_coverage_gain`: `0.0`
- `methodology`: `Reexecute all40 sealed paired calibration replicates on authenticated fixed scores and groups; selection before certification labels, certificates before census labels; compute exact accepted/error masses for five fixed mixtures. Bonferroni bounded Hoeffding intervals over36 fixed quantities. Seeds average within replicate. Separate corruption changes within-group evidence/outcome distribution and...(418 chars)`
- `independent_generalization_score`: `0`
- `exposure_scope`: `constructed signed relational worlds; exact oracle labels; no natural-language or oracle-distinct generalization`

## RECOMMENDATION
KEEP

No rerun or inspection of elided rows; the reported null does not establish equivalence.

## experiment_8871_v759_size_aware_verifier.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The size-aware verifier earns a positive verdict from qualification against its own exact-oracle task.

## WHAT WOULD REFUTE IT
A decision or witness-multiset disagreement would refute correctness. For added value, the existing indexed verifier achieving equal correctness and completion at equal or lower complete cost would defeat the positive interpretation.

## WAS THAT CHECKED
Correctness parity was checked: displayed boundary and witness-multiset checks pass. Added value was not established: the verifier defines correctness, and the methodology explicitly has no benefit gate. These checks support implementation qualification, but cannot justify a positive value verdict.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_size_aware_verifier`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `size_dispatch_ready_score`: `1`
- `semantic_parity_rows`: `check`: `exact_witness_multiset`, `passed`: `true`
- `exposure_scope`: `constructed signed-record exact oracle; no natural language, speed or Rust claim`

## RECOMMENDATION
NARROW_CLAIM

Not checked: elided rows or referenced files. Risk: the positive verdict can promote execution parity into unsupported evidence of value.

## experiment_8874_v759_size_aware_complete_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Size-aware dispatch did not demonstrate complete-cost benefit under the declared mixture and guard requirements.

## WHAT WOULD REFUTE IT
A resident mixture speed-ratio lower bound above 1, alongside size-8 and size-16 latency-regression upper bounds at most 5%, exact completed parity, no coverage loss, and no increased false acceptance, would overturn this reported null.

## WAS THAT CHECKED
Yes. Both boundaries report failed mixture-speed gates. The visible cold mixture interval crosses 1, and its size-8 regression upper bound exceeds 5%.

Timing is independently measured; oracle-defined correctness does not predetermine speed. No model generalization benefit is claimed. The existing indexed implementation is a serious baseline and is present alongside scan. Censored observations are explicitly identified and stress is reported separately. These findings support failure to establish benefit, not proof that every workload receives no speedup.

## EVIDENCE
- `honest_verdict`: `complete_null_size_aware_complete_cost`.
- `benefit_gates`: both `cold` and `resident` report `mixture_speed_lower_above_one`: `false`.
- `paired_intervals` → `cold` → `size_aware` → `mixture` → `speed_ratio`: `lower`: `0.9863942115219428`; `upper`: `1.0291241675870688`.
- `cold` → `size_aware` → `8` → `latency_regression`: `upper`: `0.05056101728964857`.
- `cold_speed_claim`: `false`; `independent_generalization_score`: `0`; `inference_substrate`: `no_model_load`.
- `arm`: `indexed`, `size_aware`, `scan`; `censored_count`: `960`; `status`: `censored`; `stress_separate`: `true`.

## RECOMMENDATION
KEEP

Not rerun; elided rows limit independent checks of aggregation and row inclusion.

## experiment_8875_qwen_source_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No source-fidelity conclusion is asserted. This artifact records a blocked prerequisite check, so there is no experimental claim to falsify.

## WAS THAT CHECKED
Source fidelity was not checked; execution stopped at `conductor_pre_gate`. The prerequisite failure is explicitly recorded: the readiness score was 0 against a required 1. Fidelity remains undetermined.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8876_v759_hardware_operation_budget.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no asserted hardware advantage to falsify. A compatible device workload beating the best installed CPU arm on complete end-to-end latency would contradict the stated absence of measured benefit if such evidence appeared here.

## WAS THAT CHECKED
No physical comparison was performed. This artifact inventories upstream CPU evidence, records missing learning inputs, and leaves hardware qualification blocked. Its audit pass does not assert acceleration, verifier value, or generalization. The displayed unsupported board rows are explicitly CPU inventory.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_operation_budget`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `reason`: `no measured complete supported physical workload benefit`
- `count_scope`: `CPU inventory only; absent counts and component times are unknown`
- `measured_board_time_s`: `null`
- `physical_probes_performed`: `0`
- `research_100x`: `unknown end-to-end gain; first measure a disjoint compatible time fraction, then kernel parity, transfers and full durable latency`

## RECOMMENDATION
KEEP

Review limited to supplied text; elided rows were not checked. Physical acceleration remains unmeasured.

## experiment_8877_v759_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. An unaccounted scheduled slot or rejected evidence counted as scientific support would contradict the narrower administrative accounting.

## WAS THAT CHECKED
Yes, administrative admission and dependency checks are shown: the disqualified producer is rejected, missing dependencies fail their gates, and natural generalization remains unestablished. Historical publication readiness is explicitly scoped to the historical paper. Elided rows prevent independently checking all twelve slots; no comparative benefit should be inferred.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v759_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `Administrative custody and method ingestion only; no new scientific benefit, natural-text or hidden-game generalization.`
- `disposition`: `rejected_producer`; `science_admitted`: `false`
- `missing_reason`: `missing_file`; `observed`: `null`; `passed`: `false`
- `natural_generalization_count`: `0`
- `historical_publication_scope`: `historical FoVer paper only`
- `publication_authorized`: `false`
- `Original bytes bind each scoped conclusion; accounting completion does not establish science or authorize deployment.`

## RECOMMENDATION
KEEP
