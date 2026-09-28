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

## experiment_7783_source_view_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A claim that source custody was qualified would be refuted by a failed prerequisite gate.

## WAS THAT CHECKED
Yes. The gate check found two failures and recorded the run as blocked. The artifact makes no qualification claim.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate-unsat(final): 2 of 3 gate(s) failed"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7787_v677_qwen_event_confidence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Event-based confidence elicitation does not provide a statistically significant decision benefit over generic confidence elicitation on the exposed development pilot panel (`complete_null_exposed_pilot`).

## WHAT WOULD REFUTE IT
A statistically significant paired improvement where the lower bound of the 95% confidence interval for Brier score reduction and/or decision cost is strictly positive (`lower95 > 0`), causing `decision_benefit.passed` to evaluate to `true`.

## WAS THAT CHECKED
Yes. The artifact executed 24 paired families across both `generic` and `event` arms under live LLM inference. In `acceptance_gate_results.decision_benefit.measured_operands` and `paired_improvements`, both Brier score improvement (`lower95` of -0.113) and decision cost improvement (`lower95` of -0.333) span zero and include negative deltas, resulting in `passed: false` for `decision_benefit`.

## EVIDENCE
- `"honest_verdict": "complete_null_exposed_pilot"`
- `"verdict_class": "null"`
- `"decision_benefit"`
- `"passed": false`
- `"principle": "Paired family gains must exceed uncertainty."`
- `"lower95": -0.11297083333333334`
- `"mean": 0.023158333333333326`
- `"upper95": 0.15390416666666665`
- `"lower95": -0.3333333333333333`
- `"mean": 0.16666666666666666`
- `"upper95": 0.625`
- `"benefit_passed": false`
- `"claim_scope": "exposed_natural_development_pilot; no held-out generalization or production activation"`
- `"production_promotion": false`

## RECOMMENDATION
KEEP

## experiment_7784_v677_training_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. All required validation checks passing would contradict the reported disqualification.

## WAS THAT CHECKED
Yes, for validation status: required checks were attempted and failed. No natural benefit test is claimed.

## EVIDENCE
`claim_scope`: `synthetic_fixture_only_no_natural_benefit`; `decision_benefit`: `null`; `honest_verdict`: `complete_disqualified_required_validation`; `validity`: `false`; `coverage_combine`: `passed`: `false`; `coverage_report`: `passed`: `false`; `full_python_suite`: `timed_out`: `true`.

## RECOMMENDATION
KEEP

## experiment_7789_v677_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The required independent evidence is blocked because two declared upstream producers are missing.

## WHAT WOULD REFUTE IT
The artifact showing both required producers present and qualifying, while still reporting the evidence as blocked.

## WAS THAT CHECKED
Yes. The audit checked the declared producer paths and recorded their states; both required producers were missing.

## EVIDENCE
`honest_verdict` `complete_blocked_required_v677_evidence`; `producer_states` `Exp7786` `missing` `Exp7787` `eligible` `Exp7788` `missing`; `independent_evidence_ready_score` `0`; `readiness` `0`.

## RECOMMENDATION
KEEP

## experiment_7790_v677_arc_runner_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An assertion of runner qualification readiness or comparative performance benefit would be refuted by failed qualification gates, repository validation timeouts, unstarted evaluation rows, and unmeasured quality/benefit metrics. However, the artifact asserts no comparative or performance claim.

## WAS THAT CHECKED
Yes. Prerequisite runner validation and diagnostic checks were executed (`gate_check_summary`, `validation_receipts`, and `repository_health`), where `full_python_suite` timed out with exit code -15, resulting in the artifact recording a disqualified status.

## EVIDENCE
`honest_verdict`
`complete_disqualified_required_runner_validation`
`verdict_class`
`disqualified`
`organic_runner_ready_score`
`0`
`acceptance_gate_results`
`decision_benefit`
`null`
`efficiency`
`probability_quality`
`readiness`
`false`
`retention`
`validity`
`gate_check_summary`
`full_python_suite`
`observed`
`-15`
`passed`
`status`
`unstarted`
`sample_size_budget`
`completed`
`effective_independent_n`

## RECOMMENDATION
KEEP

## experiment_7791_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no substantive or comparative experimental claim because execution was blocked at the pre-gate qualification stage.

## WAS THAT CHECKED
no; no experimental run or evaluation occurred because gate checks failed prior to execution.

## EVIDENCE
`schema`: `"blocked_gate_check_v1"`
`status`: `"blocked"`
`honest_verdict`: `"blocked_gate_check_failed"`
`duration_s`: `0.0`
`blocked_at_layer`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7793_v677_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An empirical observation demonstrating that a comparative performance or speedup claim was asserted despite absent or failing hardware execution. Because this artifact makes no comparative claim, sets all performance metrics to null, and records that execution was blocked at the pre-gate, there is no affirmative claim to refute.

## WAS THAT CHECKED
No. No experimental trials or comparative evaluations were conducted; the artifact is an accounting and gating receipt recording that execution was blocked before running any workloads due to a missing upstream producer (`results/experiment_7792_v677_service_cost.json`).

## EVIDENCE
`hardware_advantage_claimed`: `false`
`honest_verdict`: `complete_blocked_missing_service_evidence`
`verdict_class`: `blocked`
`service_measured`: `false`
`hardware_advantage`: `unmeasured`
`decision`: `unmeasured`
`learning`: `unmeasured`
`probability`: `unmeasured`
`arm`: `historical_accounting`
`completed`: `false`
`started`: `false`
`metric`: `null`
`gate_check_summary`: `A missing producer differs from a scientific null.`

## RECOMMENDATION
KEEP

## experiment_7794_v677_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A claim that this capstone established benefit would be refuted by failed validation or ineligible producers. The artifact reports those conditions and does not claim benefit.

## WAS THAT CHECKED
Yes. The validation receipts and task rows record failed required checks and ineligible producers. This is a disqualification record, not a comparative result.

## EVIDENCE
`actual_inference_substrate_class`: `aggregation`; `honest_verdict`: `complete_disqualified_v677_capstone_validation`; `capstone_complete_score`: `0`; `validity`: `false`; `required_checks_passed`: `false`; `producer_eligible`: `false`.

## RECOMMENDATION
KEEP
