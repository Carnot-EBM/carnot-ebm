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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8408_native_transaction_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title describes an intended comparison, but the artifact asserts no comparative result.

## WAS THAT CHECKED
No transaction-cost comparison is reported. The prerequisite gates were checked; native parity qualification failed, blocking the experiment.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `native_parity_ready_score`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8409_v724_retrospective_label_criterion.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The artifact declares a positive verifier result from a retrospective reduction in false positives measured against released A1 labels.

## WHAT WOULD REFUTE IT
A zero or negative frozen factorial contrast would refute the reported reduction. Verifier added value would require surviving independent correctness assessment and beating a cheap fixed-prompt judge at comparable false-negative rates; a tie or loss would defeat that value claim.

## WAS THAT CHECKED
Yes for the descriptive reduction: the frozen contrast averages two differences in A1-defined false-positive rates, and its bootstrap interval is entirely positive. That contrast could have failed.

No for independent verifier value: A1 defines correctness and is explicitly the oracle. A2 sensitivity checks annotation agreement, not an independent value claim. The positive verdict therefore exceeds what that measurement establishes.

The artifact explicitly limits exposure to retrospective agreement and assigns generalization scores zero, so there is no supported generalization claim to attack. A cheap comparator evaluated at matched false-negative rates is not identified. Displayed outputs are valid and contain the scored labels; the elided rows prevent a complete row audit.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_retrospective_label_criterion`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `A1 is the criterion authority, so a positive agreement replication remains circular_positive.`
- `primary_contrast`: `0.46793587174348694`
- `primary_ci95`: `0.4229957805907173`, `0.5126711839820907`
- `exposure_scope`: `released previously opened A1 targets; retrospective within-release agreement only`
- `independent_generalization_score`: `0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8410_v724_cuda_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The reported blocked qualification would be contradicted by authenticated execution-change evidence together with a successful owned context/copy and cleanup receipt.

## WAS THAT CHECKED
No current context/copy test occurred: prerequisite checks failed in the gate summary, and the context/copy row was censored. The artifact records incomplete qualification; it does not claim current CUDA failure, recovery, or comparative benefit.

## EVIDENCE

- `honest_verdict`: `complete_blocked_cuda_reader_qualification`
- `exposure_scope`: `exposed development runtime qualification; no scientific samples`
- `runtime_changed_score`: `0`; `cuda_context_ready_score`: `0`
- `current_device_probe_count`: `0`
- Row `context_copy` has `status`: `censored` and `missing_reason`: `authenticated_execution_delta_or_dependency_absent`
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

Not applicable: this is a blocked prerequisite receipt, with no canary performance or comparative value claim.

## WAS THAT CHECKED

No canary performance was tested. Prerequisites were checked in `gates_evaluated`: two required gates failed, and the artifact records blocking before execution.

## EVIDENCE

`blocked_gate_check_v1`, `blocked_gate_check_failed`, `runtime_changed_score`, `cuda_context_ready_score`, `actual=0 == expected=1`, `conductor_pre_gate`

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
No comparative claim is asserted. An authenticated current board execution satisfying compatibility, transport, and owned-check requirements would contradict the recorded disqualification.

## WAS THAT CHECKED
No current board trial is shown. The artifact aggregates upstream receipts and records unmet qualification gates, unsupported operations, and unmeasured speedup. It does not claim verifier value, generalization, or superiority over a baseline, so no comparative refutation opportunity is required.

## EVIDENCE

- `actual_substrate`: `host_CPU_aggregation`
- `compatible_kernel`: `false`; `complete_transport`: `false`; `owned_checks`: `false`
- `current_device_execution_count`: `0`
- `eligible_operation_count`: `0`
- `compatible_fraction_status`: `unmeasured`
- `full_service_speedup`: `null`
- `scientific_benefit`: `false`

## RECOMMENDATION
KEEP

## experiment_8414_v724_gatemate_obligation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is asserted. Completed, authenticated evidence for obligations still marked missing would contradict the bookkeeping, rather than refute a claimed method benefit.

## WAS THAT CHECKED
Not applicable to comparative value. The rows document one completed continuity record and six censored obligations. The blocked verdict preserves those limitations; passing validation does not assert scientific benefit.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_obligation`
- `verdict_class`: `blocked`
- `scientific_benefit`: `false`
- `completed_count`: `1`
- `censored_count`: `6`
- `generalized_learning_benefit_score`: `0`
- `Qualified documentation records the unchanged obligation and grants no scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8415_v724_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This exposed development evaluation demonstrates neither a spline decision-cost advantage over RBF34 nor an online decision-cost advantage over frozen, without resolving representation superiority.

## WHAT WOULD REFUTE IT
A reliably positive paired cost gain for spline34 over RBF34, or online updating over frozen, would contradict the absence of demonstrated decision benefit. Such gains must survive the reported uncertainty and missing-row bounds.

## WAS THAT CHECKED
Yes. H1 reports negative mean gain with uncertainty spanning zero; H2 reports exactly zero gain. These comparisons could have favored the methods: costs depend on actions and outcomes, rather than merely reproducing model probabilities.

Serious controls are present: linear6 ties spline34’s decisions and costs, while frozen and calibration_only tie online decision cost. The artifact acknowledges these nulls.

The oracle controls receive circular classifications, without scientific benefit credit. Evaluation is explicitly exposed and descriptive. Unqualified H2 rows have explicit equal costs and zero gain bounds; probability metrics use qualified counts. Missing retention probability outcomes receive broad bounds. Nothing here establishes generalization or optimizer-qualified representation superiority, and the reported dispositions respect those limits.

## EVIDENCE

- H1: `science_disposition` = `null_optimization_limited`; `mean_gain` = `-0.00390625`; `lower_one_sided_975` = `-0.01953125`; `upper_descriptive_975` = `0.0078125`.
- H2: `science_disposition` = `null_delayed_decision_benefit`; `mean_gain` = `0.0`; `action_changes` = `0`.
- Controls: `linear6`; `calibration_only`; `changed_decisions_by_arm`; `cost_mean` = `0.4431818181818182`.
- Limits: `geometry_qualified` = `false`; `representation_null_informative` = `false`; `descriptive_exposed_development_not_confirmatory`.
- Qualification: `qualified` = `false`; `frozen_cost` = `0.5`; `online_cost` = `0.5`; `gain_lower` = `0.0`; `gain_upper` = `0.0`.
- Scientific credit: `verdict_class` = `circular_positive`; `semantic_benefit` = `false`; `whole_service_benefit` = `false`.

## RECOMMENDATION
KEEP
