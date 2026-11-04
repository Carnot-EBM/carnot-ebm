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

## experiment_8806_v754_consensus_challenge_energy.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Learned energy did not meet the joint success criteria on constructed consensus candidates.

## WHAT WOULD REFUTE IT
Passing all three gates: improved Brier score and false accepts while losing at most 0.02 coverage against both non-oracle baselines, including fixed evidence.

## WAS THAT CHECKED
Yes. Brier and false-accept gates pass, but coverage fails. Against both baselines, the reported coverage differences average approximately −0.114, with their entire 97.5% intervals below −0.02. The experiment could succeed or fail; it records an honest null.

The exact oracle is explicitly identified as circular. Calibration and displayed test rows use different cohort identifiers, and no independent-generalization success is claimed. No displayed scored row is flagged invalid, and the cited metrics exist in their respective tables.

## EVIDENCE
- `honest_verdict`: `complete_null_constructed_consensus_energy`
- `brier_lower_97_5`: `true`
- `false_accept_lower_97_5`: `true`
- `coverage_loss_at_most_02`: `false`
- `coverage_agreement`: `-0.11358024691358022`, `-0.18166743827160492`, `-0.04740663580246906`
- `coverage_fixed`: `-0.11358024691358025`, `-0.13605015432098763`, `-0.09135802469135802`
- `verdict_class`: `circular_positive`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8807_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt with no claim about the canary’s performance or value.

## WAS THAT CHECKED
No experimental result was tested. The prerequisite checks in `gates_evaluated` recorded one failure and one pass, consistent with the blocked status.

## EVIDENCE
`status` = `blocked`; `honest_verdict` = `blocked_gate_check_failed`; `failed_field` = `runtime_repair_ready_score`; `failed_expected` = `1`; `failed_observed` = `0`; `blocked_at_layer` = `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8808_v754_archived_observer_costs.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Observer disturbance failed the acceptance gate on this constructed workload, leaving component costs unvalidated.

## WHAT WOULD REFUTE IT
Valid plain/sham/clock comparisons satisfying the frozen 5% absolute-disturbance limit throughout, with the remaining component-readiness gates satisfied, would contradict the reported null.

## WAS THAT CHECKED
Yes. The paired rows compare separately measured parent times for plain execution, sham dispatch, and buffered clocks. Visible differences exceed 5%, and the disturbance failures appear in the aggregate acceptance gate and null verdict.

Plain execution and identical-dispatch sham provide meaningful controls. The verifier’s oracle status does not make these timing differences inevitable. No verifier-value or generalization benefit is claimed. The failed disturbance flags support the headline; successful execution flags do not establish acceptable disturbance.

## EVIDENCE

- `honest_verdict`: `complete_null_archived_observer_costs`
- `maximum_absolute_disturbance`: `0.05`
- `observer_disturbance`: `false`
- First Python paired row: `sham_absolute_fraction`: `0.06370975466750639`; `clock_vs_sham_absolute_fraction`: `0.11103157171111122`; `disturbance_passed`: `false`
- `component_cost_ready_score`: `0`
- `descriptive_only`: `true`; `independent_workloads`: `1`; `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8809_v754_delayed_feedback_capacity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No corrected useful policy advantage was found in the tested delayed-feedback capacity settings.

## WHAT WOULD REFUTE IT
An interpretable comparison at matched delay and capacity showing a practically useful prediction improvement after multiplicity correction, with retention satisfying the usefulness criterion.

## WAS THAT CHECKED
Yes. Six simultaneous FIFO-versus-uniform comparisons report no useful effect. Overloaded queues and differing admissions allowed policies to differ; the null was not forced by the tied low-delay controls.

Two corrected Brier intervals favor FIFO, but their retention intervals cross zero. Thus the supported claim is no demonstrated useful advantage, not no measurable difference. The artifact makes no positive verifier-value or independent-generalization claim. The numerical usefulness threshold is not supplied, limiting independent assessment of that classification.

## EVIDENCE
- `honest_verdict`: `complete_null_no_corrected_useful_policy_effect`
- `queue_controls`: `differing_admissions`: `true`; `overflow_positive`: `true`; `overloaded_peaks`: `true`
- `simultaneous_intervals`: `FIFO_minus_uniform`; `family_size`: `6`; `independent_count`: `30`; every listed `interpretable`: `true` and `useful`: `false`
- At `delay`: `16`, `capacity`: `8`: `lower`: `-0.00038594713770827554`; `upper`: `-4.945834117148112e-05`; `retention_interval`: `lower`: `-1.6366419447828533e-05`; `upper`: `0.00025632970602485833`
- `exposure_scope`: `constructed_signed_graphs_exact_verifier_labels; fresh resets; no independent generalization`

## RECOMMENDATION
KEEP

## experiment_8810_v754_causal_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found no qualifying causal learning benefit on the constructed exact-oracle streams.

## WHAT WOULD REFUTE IT
The combined arm clearing the required future-Brier improvement margins against both read-only and calibration-only, while satisfying retention, error, and validity gates, would overturn this scoped null.

## WAS THAT CHECKED
Yes. Brier scores come from sealed probabilities and labels. The reported familywise intervals for combined minus each comparator are entirely positive: combined performs worse. Read-only supplies the cheapest serious baseline; the conclusion does not depend on beating shuffled feedback.

The oracle defines correctness, but the artifact makes no positive verifier-value or independent-generalization claim. Complete denominators and zero exclusions are reported. Nothing shown demonstrates a false row-validity flag or wrong-key error. The measured comparisons support the null without establishing a universal absence of learning benefit.

## EVIDENCE

- `honest_verdict`: `complete_null_causal_learning_audit`; `verdict_class`: `null`; `learning_benefit_ready_score`: `0`.
- `future_combined_minus_read_only`: `lower`: `0.0003348147308100831`; `upper`: `0.00143417653264363`.
- `future_combined_minus_calibration_only`: `lower`: `7.696004880677928e-05`; `upper`: `9.488482634675866e-05`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`.
- `complete_denominators`: `true`; `excluded_count`: `0`; `censored_count`: `0`.

## RECOMMENDATION
KEEP

## experiment_8811_new_arc_outcome_analysis.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records a blocked prerequisite, without asserting a generalization result.

## WAS THAT CHECKED
No generalization test occurred. The prerequisite availability gate was checked and failed: observed 0 against required 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `arc_new_outcomes_available_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8812_learning_operation_budget.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, not a reported acceleration result.

## WAS THAT CHECKED
No acceleration claim was tested here. The prerequisite checks were evaluated, and one failed: component-cost readiness was 0 against a required 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `component_cost_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8813_v754_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. Treating failed prerequisites or circular results as independent scientific success would contradict the artifact’s stated administrative scope.

## WAS THAT CHECKED
Yes, for that limited scope: gate_check_summary records failed readiness operands and missing producers, and the aggregate remains blocked. Upstream null and circular outcomes are retained without promotion to independent generalization. Publication readiness is explicitly scoped to historical work.

## EVIDENCE
- `exposure_scope`: `administrative thirteen-slot accounting; upstream constructed science remains scoped`
- `honest_verdict`: `complete_blocked_v754_capstone`
- `scientific_measurement_count`: `0`
- `independent_generalization_score`: `0`
- `observed`: `missing_file`; `passed`: `false`
- `historical_publication_scope`: `historical FoVer paper only`

## RECOMMENDATION
KEEP
