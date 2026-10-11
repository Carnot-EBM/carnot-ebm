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
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8410_v724_cuda_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked qualification receipt with no comparative claim. An authenticated execution change and a successful owned context/copy with cleanup would contradict its reported blocked status.

## WAS THAT CHECKED
No—the context/copy unit was censored. Environment comparisons and prerequisite gates were checked and remained unmet. The artifact does not claim current CUDA failure, recovery, or scientific benefit.

## EVIDENCE

- `honest_verdict`: `complete_blocked_cuda_reader_qualification`
- `exposure_scope`: `exposed development runtime qualification; no scientific samples`
- `arm`: `context_copy`; `status`: `censored`; `missing_reason`: `authenticated_execution_delta_or_dependency_absent`
- `acceptance_gates`: `actual_execution_delta`: `false`; `owned_context_copy_cleanup`: `false`
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`
- `root_cause_status`: `cause_unproved`

## RECOMMENDATION
KEEP

## experiment_8411_bounded_qwen_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental headline claim to falsify. This artifact records a blocked prerequisite check, not a result establishing canary performance or CUDA recovery.

## WAS THAT CHECKED
Prerequisite gates were checked: runtime change and CUDA-context readiness each scored 0 against a required 1. The canary was blocked before execution, so its performance was not checked.

## EVIDENCE
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`duration_s`: `0.0`  
`blocked_at_layer`: `conductor_pre_gate`  
`gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8410-cuda-reader-qualification.runtime_changed_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8412_v724_arc_panel_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8413_v724_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. A valid, compatible board operation meeting the acceptance gates would challenge the recorded disqualification; absent board measurements cannot establish zero performance benefit.

## WAS THAT CHECKED
No current board performance comparison is shown. The visible rows account for host operation costs and compatibility. This is an eligibility and provenance receipt, not a measured comparative null.

## EVIDENCE
- `actual_substrate`: `host_CPU_aggregation`
- `current_device_execution_count`: `0`
- `eligible_operation_count`: `0`
- `kv260_supported`: `false`
- `compatible_fraction_status`: `unmeasured`
- `full_service_speedup`: `null`

## RECOMMENDATION
KEEP

## experiment_8414_v724_gatemate_obligation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim exists to falsify. Authenticated evidence satisfying obligations that the corresponding rows still mark missing would contradict the administrative record.

## WAS THAT CHECKED
No comparative falsification test is applicable. The rows record one completed continuity item and six censored obligations. The positive documentation score is accompanied by a blocked verdict and zero execution and generalization scores.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_obligation`
- `verdict_class`: `blocked`
- `Qualified documentation records the unchanged obligation and grants no scientific benefit.`
- `scientific_benefit`: `false`
- `execution_ready_score`: `0`
- `independent_generalization_score`: `0`
- `completed_count`: `1`; `censored_count`: `6`

## RECOMMENDATION
KEEP

## experiment_8415_v724_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Neither H1 nor H2 showed an observed decision-cost improvement on the exposed development data, and scientific benefit remains blocked.

## WHAT WOULD REFUTE IT
A positive paired mean cost gain for spline34 over RBF34, or for online_sparse over frozen_spline, after accounting for unqualified rows and their cost bounds. This would contradict the reported utility null; it would not establish generalization.

## WAS THAT CHECKED
Yes. H1’s paired-cost aggregation reports negative gain against RBF34; H2’s paired-cost aggregation reports zero gain against frozen_spline. Different predictions could produce different outcome-based action costs: H1 records three changed decisions, and H2 records 66 changed probabilities.

Serious cheaper controls are present: linear6 ties spline34’s action cost, and calibration_only ties the online arms’ cost. These ties support the stated null. Oracle-based controls are explicitly classified as circular and do not become a verifier-value claim. The comparisons are labelled exposed and descriptive, and qualified versus intended populations and missing-row bounds are tracked explicitly.

## EVIDENCE

- H1: `spline34`, `cost_mean`, `0.453125`; `RBF34`, `cost_mean`, `0.44921875`; `mean_gain`, `-0.00390625`; `science_disposition`, `null_optimization_limited`.
- H1: `linear6`, `cost_mean`, `0.453125`; `changed_decision_count`, `3`.
- H2: `frozen_spline`, `online_sparse`, `calibration_only` each report `cost_mean`, `0.4431818181818182`; `mean_gain`, `0.0`; `science_disposition`, `null_delayed_decision_benefit`.
- H2: `probability_changes`, `66`; `missing_bounds`; `qualified`, `false`; `gain_lower`, `0.0`; `gain_upper`, `0.0`.
- `descriptive_exposed_development_not_confirmatory`; `fixed_exposed_trajectory_descriptive_no_IID_or_conformal_guarantee`; `verdict_class`, `circular_positive`.
- `semantic_benefit`, `false`; `whole_service_benefit`, `false`; `geometry_qualified`, `false`.

## RECOMMENDATION
KEEP

## experiment_8416_v725_contract_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is asserted. The administrative blocked status would be contradicted by a complete, independently bound 13-task contract satisfying all required readiness gates.

## WAS THAT CHECKED
Yes, administrative readiness was checked in gate_check_summary and rows: only 4 of 13 task objects were present, the independent design binding was absent, and the runtime-repair gate failed. These failures are reflected in the blocked verdict. Oracle qualification and exposed labels are not presented as evidence of scientific benefit or generalization.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v725_contract_methods`
- `verdict_class`: `blocked`
- `scientific_benefit`: `false`
- `current_contract_ready_score`: `0`
- `independent_generalization_score`: `0`
- `methodology_note`: `Oracle contract custody only; 4 actual task objects, 9 absent. Sealed retrospective methods grant no semantic or independent benefit.`

## RECOMMENDATION
KEEP

## experiment_8417_criterion_feature_panel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No method-value claim exists to refute; this artifact records a blocked prerequisite check.

## WAS THAT CHECKED
The prerequisite was checked and failed: readiness was 0 against a required 1. No method-performance result is reported.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `methods_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP
