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

## experiment_8816_v755_selective_evidence_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted here. A future benefit claim would fail if coverage gains missed the prespecified margins against the strongest budgeted comparator or fixed_evidence, or violated the risk bounds.

## WAS THAT CHECKED
No. Those comparisons are prospective. The displayed rows document protocol sealing; oracle-based infrastructure checks do not establish added value, and the artifact explicitly leaves benefit unmeasured.

## EVIDENCE

- `scientific_benefit_measured`: `false`
- `constructed protocol seal; no policy evaluation`
- `prospectively frozen; Exp8818/8819 must select before evaluation; no study predictions in this slot`
- `Exact-oracle construction and mutation controls qualify infrastructure only. Readiness is valid evidence, not measured benefit. No natural-text or hidden-game generalization.`
- `protocol_seal`
- `independent_topology_sealed`

## RECOMMENDATION
KEEP

## experiment_8817_categorical_memory_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific result is asserted. A passing sole prerequisite gate would contradict the reported reason for blocking, but would not establish that the objective qualified.

## WAS THAT CHECKED
Yes, the prerequisite gate was checked and failed: actual 0 versus expected 1. This is a blocked-gate receipt; it reports no objective qualification test or comparative result.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `objective_candidate_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8818_v755_evidence_decisive_coverage.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The evaluated joint_symbolic arm shows no decisive-coverage gain over the preselected evidence_set comparator, leaving the benefit claim blocked.

## WHAT WOULD REFUTE IT
A positive evaluation coverage gain for joint_symbolic over evidence_set while satisfying the stated error constraints would contradict this bounded no-gain finding.

## WAS THAT CHECKED
Yes. The reported coverage-gain interval across 240 evaluation worlds is zero throughout, and the displayed world gives both arms identical coverage and error rates. The serious symbolic comparator ties evidence_set; the artifact records that tie as a failed benefit gate.

Oracle-based correctness cannot establish independent semantic value, but the artifact makes no positive value claim. It also assigns generalization a zero score and acknowledges overlapping graph structures. The displayed scored rows contain explicit truth, decision, metric, and denominator fields; no visible row-validity exclusion or wrong-key error contradicts the null.

## EVIDENCE

- `honest_verdict`: `complete_blocked_evidence_decisive_coverage`; `verdict_class`: `blocked`.
- `coverage_gain`: `estimate` = `0.0`, `lower` = `0.0`, `upper` = `0.0`, `independent_worlds` = `240`.
- Displayed `per_world_results` for `evidence_set` and `joint_symbolic`: `correct_decisive_coverage` = `0.5`; `wrong_decision_rate` = `0.0`.
- `selection_before_evaluation`: `true`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`; `graph_structures_disjoint`: `false`.
- `methodology_note`: `Exact encoded reasoning remains circular. No learned-semantic, natural-text, LLM or hidden-game improvement claim. Readiness means valid evidence, not benefit.`

## RECOMMENDATION
KEEP

## experiment_8819_v755_consensus_selective_energy.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The learned energy selector demonstrated no selective benefit under the specified gates on this constructed corpus.

## WHAT WOULD REFUTE IT
A calibration-selected learned cutoff that subsequently passed all five evaluation gates: risk upper bound ≤0.05, coverage lower bound ≥0.30, insufficient-to-decision upper bound ≤0.05, Brier gain lower bound ≥0.01, and useful coverage gain lower bound ≥0.05 against fixed evidence.

## WAS THAT CHECKED
Yes. The threshold rows test six cutoffs across five learned seeds. Cutoffs offering sufficient coverage fail the risk requirement; safe cutoffs provide insufficient coverage. All five seeds therefore select no candidate. This outcome follows measured errors and coverage, rather than an automatic success criterion.

The exact oracle is excluded from learned benefit. The artifact makes no independent generalization claim. Fixed evidence, the cheap substantive comparator, also selects no candidate. That tie supports the reported null.

The false eligibility flags reject cutoffs while retaining their metrics. Undefined risk and unestimable benefit contrasts remain explicit; they are not presented as successful safety or comparative gains.

## EVIDENCE

- `honest_verdict`: `complete_null_consensus_selective_energy`; `benefit_passed`: `false`; `selective_benefit_score`: `0`.
- `selection`: `safe risk upper <=.05 and coverage lower >=.30; maximize correct coverage, then lowest cutoff; none => escalate`.
- For `learned_energy`, seed `7558819`, threshold `0.6`: `coverage_lower` is `0.3333333333333333`; `risk_upper` is `0.3488430232558136`.
- For `learned_energy`, seed `7558821`, threshold `0.9`: `risk_upper` is `0.0`; `coverage_lower` is `0.061111111111111116`; `eligible` is `false`.
- `status`: `no_candidate`; `risk`: `undefined_denominator`; `brier_gain` and `useful_coverage_gain`: `not_estimable`.
- `scope`: `separate exact reference; excluded from policy benefit`.

## RECOMMENDATION
KEEP

## experiment_8823_qwen_witness_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, with no extraction-performance or comparative-value claim.

## WAS THAT CHECKED
No performance claim was tested. The prerequisite checks recorded a failed runtime-readiness gate and a stop before the experiment.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8824_arc_outcome_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, with no generalization result or comparative claim to falsify.

## WAS THAT CHECKED
No generalization assessment occurred. The prerequisite was checked in `gates_evaluated`: observed 0 versus required 1, so execution was blocked.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `arc_new_outcomes_available_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8825_v755_hardware_operation_map.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a static audit and blocker receipt, with no comparative performance or added-value claim to falsify.

## WAS THAT CHECKED
No comparative performance test was conducted. The operation rows record unmeasured timings, and the terminal verdict remains blocked. Audit readiness is explicitly limited to evidence validity.

## EVIDENCE

- `honest_verdict`: `complete_blocked_hardware_operation_map`
- `operation`: `static source/design audit; no model or device execution`
- `physical_acceleration`: `false`
- `operation_status`: `unmeasured`; `measured_time_s`: `null`
- `All three board boundaries stay separate. Source counts are not timing; unknown is not zero. Readiness is evidence validity only.`

## RECOMMENDATION
KEEP

## experiment_8826_v755_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative method-benefit claim is asserted to falsify. This is a custody and reconciliation artifact reporting blocked science, not demonstrated superiority or generalization.

## WAS THAT CHECKED
No comparative benefit test is shown here. The branch conclusions and gate checks record rejected producers, failed prerequisites, and protocol qualification without measured benefit. The overall verdict remains blocked; administrative completion is not presented as scientific success.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v755_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `exposure_scope`: `thirteen-task custody; admitted constructed evidence and CPU nulls only`
- `scientific_benefit_measured`: `false`
- `methodology_note`: `Independent registered row reductions retain constructed nulls. Exact oracle success remains circular. Missing natural extraction, ARC outcomes and qualified hardware costs block overall science.`
- `publication_authorized`: `false`

## RECOMMENDATION
KEEP
