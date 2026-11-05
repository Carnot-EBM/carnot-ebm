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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8831_v756_ordered_evidence_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On this constructed evaluation cohort, the selected ordered schedule failed to achieve the required paid-operation savings against always_joint despite matching decision quality.

## WHAT WOULD REFUTE IT
A paired evaluation showing at least 20% operation reduction against always_joint, with the confidence interval’s lower bound meeting that threshold and all decision-quality gates passing, would contradict the reported failure to qualify for benefit.

## WAS THAT CHECKED
Yes. Selection was frozen before evaluation, and whole-world intervals measured operation reduction separately from decision quality. The reduction estimate was negative, with an entirely negative interval. The serious baseline, always_joint, was cheaper in all three displayed world summaries.

The oracle defines correctness, but this is an honest cost-benefit null, not a positive claim about verifier value. No model was invoked, and no generalization success is asserted. The displayed false valid_alternative flags concern intact inputs; they do not establish invalid cost measurements. The compared operations fields exist in the displayed world summaries.

## EVIDENCE

- `honest_verdict`: `complete_null_ordered_evidence_cost`; `verdict_class`: `null`.
- `frozen_before_evaluation`: `true`; `schedule`: `joint_first`.
- `paid_operation_reduction`: `threshold`: `0.2`; `passed`: `false`; `estimate`: `-0.23715857852844158`; `lower`: `-0.2610656529889671`; `upper`: `-0.21847799982965366`.
- `correct_decisive_coverage_loss`: `estimate`: `0.0`; `semantic_false_accept_increase`: `estimate`: `0.0`.
- `always_joint`: `operations`: `4540.833333333333`; `joint_first`: `operations`: `7536.666666666667`.
- `verifier_is_oracle`: `true`; `model_invoked`: `false`; `independent_generalization_score`: `0`.
- `condition`: `intact`; `status`: `completed`; `valid_alternative`: `false`.

## RECOMMENDATION
KEEP

## experiment_8832_v756_selective_availability.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8833_causal_objective_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. This artifact records a prerequisite failure, not a result about future loss or the method’s value.

## WAS THAT CHECKED
Only the prerequisite gate was checked. Its observed value was 0 against a required value of 1, so execution was blocked before the experiment could test its objective.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `memory_execution_ready_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8835_v756_complete_adaptation_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The artifact reports complete-operation CPU costs on constructed streams without establishing a speed gain.

## WHAT WOULD REFUTE IT
A paired cost interval entirely below zero would contradict the reported absence of demonstrated speed gain for that comparison. A missing scheduled window, failed state control, or omitted operation inside the declared boundary would contradict completeness.

## WAS THAT CHECKED
Yes, through parent timing, completion/state gates, and two cost comparisons against read-only and calibration-only controls. Both comparisons report 30 complete pairs, with 99% intervals spanning zero. Timing outcomes could have supported a gain; the null was not guaranteed.

The oracle defines correctness, but the claim concerns measured costs rather than verifier value. No generalization benefit is asserted. Completion checks report all 90 windows valid; elided rows limit direct inspection. The null does not establish equivalence.

## EVIDENCE

- `honest_verdict`: `complete_null_complete_adaptation_cost`
- `scientific_speed_gain`: `passed`: `false`
- `principle`: `Benefit is not required for cost readiness.`
- `control`: `read_only`; `treatment`: `calibration_only`; `lower99_ns`: `-1255811193.0838332`; `upper99_ns`: `749533387.0483329`
- `control`: `calibration_only`; `treatment`: `combined`; `lower99_ns`: `-1624028711.688`; `upper99_ns`: `83562880.78149894`
- `paired_blocks`: `30`; `missing_pairs`: `0`
- `complete_valid_windows`: `passed`: `true`; `principle`: `All90 windows and exact state controls must pass.`
- `intended_count`: `90`; `completed_count`: `90`; `failed_count`: `0`; `excluded_count`: `0`; `censored_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8836_qwen_witness_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no extraction-performance claim to refute. The blocking diagnosis would be contradicted if all evaluated gates passed.

## WAS THAT CHECKED
Yes, for the blocking diagnosis: `gates_evaluated` records runtime readiness failing and protocol readiness passing. This is a prerequisite-check receipt; it provides no extraction results or comparative claim.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `gate_check_summary`: `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8827-branch-contract.runtime_repair_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8837_arc_outcome_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this artifact records a blocked prerequisite gate and makes no comparative claim about supervisor selection.

## WAS THAT CHECKED
The prerequisite was checked and failed: observed 0 versus required 1. No outcome-transfer assessment is reported.

## EVIDENCE
`blocked_gate_check_v1`, `blocked_gate_check_failed`, `arc_new_outcomes_available_score`, `actual=0 == expected=1`, `passed`: `false`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8838_v756_hardware_operation_budget.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audited evidence does not establish hardware acceleration; the 100× latency figures are future requirements.

## WHAT WOULD REFUTE IT
A correctness-preserving, compatible board execution demonstrating acceleration against the same complete CPU workload, including host, transfer, startup, and durability costs.

## WAS THAT CHECKED
Yes, as an evidence audit: the physical-acceleration gate failed, and the displayed board rows report host-only work with null board timings. No physical performance trial occurred, so this supports the scoped conclusion that acceleration remains unestablished, not that acceleration is impossible. The artifact does not promote its CPU comparisons or scheduling comparators into a positive benefit claim.

## EVIDENCE

- `honest_verdict`: `complete_null_hardware_operation_budget`
- `physical_acceleration`: `passed`: `false`
- `threshold`: `measured compatible complete board work`
- `operation_status`: `host_only`
- `measured_board_time_s`: `null`
- `principle`: `Valid mapping evidence does not establish acceleration`
- `claim`: `future100x latency requirement; no predicted gain`
- `inference_substrate`: `aggregation_from_upstream_artifacts`

## RECOMMENDATION
KEEP

## experiment_8839_v756_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no current comparative scientific claim to falsify. A scheduled slot lacking primary or skip custody would refute the administrative accounting statement.

## WAS THAT CHECKED
Yes, administrative accounting is checked through the completion gate and task outcome rows. The capstone preserves null and circular outcomes, excludes rejected evidence from scientific support, and withholds publication and activation. The linear comparator beats the categorical candidate in every displayed seed, but categorical superiority over linear is not claimed.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v756_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `complete_accounting`: `Every scheduled slot needs primary or skip custody`
- `verdict_class`: `null`, `circular_positive`
- `science_admitted`: `false`
- `natural_generalization_count`: `0`
- `publication_authorized`: `false`
- `activation_authorized`: `false`

## RECOMMENDATION
KEEP
