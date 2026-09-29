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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7879_v684_contract_methods.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V684 is disqualified because the staged authority is missing and required validation did not pass.

## WHAT WOULD REFUTE IT
A present staged authority, eligible passing contract rows, and passing required validation would refute the disqualification.

## WAS THAT CHECKED
Yes. The precondition and contract rows check staged authority and eligibility; the validation receipts record a required test failure.

## EVIDENCE
`honest_verdict`: `complete_disqualified_v684_required_validation`; `staged_present`: `false`; `missing_staged_authority`; `contract_ready_score`: `0`; `eligible`: `0`; `excluded`: `12`; `affected_pytest`; `passed`: `false`; `verdict_class`: `disqualified`.

## RECOMMENDATION
KEEP

## experiment_7880_v684_source_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. This artifact records a disqualified custody check.

## WAS THAT CHECKED
No comparative refutation was applicable. The required coverage check was run and failed.

## EVIDENCE
`claim_scope`: `exposed_development_custody_only`; `decision_benefit`: `null`; `honest_verdict`: `complete_disqualified_required_checks`; `coverage_report`; `observed`; `passed`: `false`; `source_boundary_ready_score`: `0`

## RECOMMENDATION
KEEP

## experiment_7881_v684_intervention_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because no comparative, capability, or superiority claim is asserted, there is no scientific hypothesis to refute. If the artifact had claimed successful protocol readiness or execution validity, that would be refuted by any failing required gate check (such as coverage test failure or pytest timeouts) or circular verifier evaluation (`verifier_is_oracle` being `true`).

## WAS THAT CHECKED
No comparative or model claims were evaluated (zero model calls or loads occurred). Execution mechanics were checked under `gate_check_summary` and `historical_required_failures`, and the artifact recorded required check failures (`coverage_report.passed` observed `false` and pytest timed out), resulting in a disqualified run.

## EVIDENCE
`claim_scope`: `CPU fixture mechanics; exposed_development sources; no independent verifier claim`
`current_work`: `Measured CPU fixture construction and validation`
`honest_verdict`: `complete_disqualified_required_checks`
`verdict_class`: `disqualified`
`intervention_protocol_ready_score`: `0`
`target_model`: `none (no pretrained model)`
`model_calls`: `0`
`acceptance_gate_results`:
`decision_benefit`: `null`
`efficiency`: `null`
`probability_quality`: `null`
`readiness`: `false`
`retention`: `null`
`validity`: `false`
`gate_check_summary`:
`artifact_field`: `coverage_report.passed`
`expected`: `true`
`observed`: `false`
`passed`: `false`

## RECOMMENDATION
KEEP

## experiment_7882_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative energy-fit claim to refute. A passing upstream gate would contradict the artifact’s finding that the run was blocked.

## WAS THAT CHECKED
Yes. The prerequisite gates were evaluated; the energy fit did not run.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"source_boundary_ready_score"`; `"expected": 1`; `"actual": 0`; `"passed": false`; `"actual": "disqualified"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7884_qwen_sufficiency.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no Qwen risk finding to refute. The title describes a planned measurement, and this artifact records a blocked gate check.

## WAS THAT CHECKED
No risk measurement was performed. The prerequisite gates were checked, and four of six failed.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `duration_s` `0.0` `gate-unsat(final): 4 of 6 gate(s) failed; first failure: exp7880-source-boundary.source_boundary_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7887_v684_arc_supervisor_delta.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7889_v684_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is a receipt and custody tracking document that makes no comparative claim and explicitly disclaims any performance speedup or hardware advantage.

## WAS THAT CHECKED
No; no comparative hypothesis was evaluated because required checks failed and the artifact disqualified itself from live device execution.

## EVIDENCE
`hardware_speedup_claimed`: `false`  
`hardware_advantage`: `unmeasured`  
`board_evidence`: `historical custody`  
`current_measurement`: `host receipt analysis only`  
`current_device_execution_count`: `0`  
`hardware_evidence_ready_score`: `0`  
`workload_attachment_available`: `false`  
`honest_verdict`: `complete_disqualified_required_checks`  
`verdict_class`: `disqualified`  
`arm`: `historical_accounting`  
`status`: `historical_read_only`  

## RECOMMENDATION
KEEP

## experiment_7890_v684_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V684 capstone is disqualified by failed required validation and establishes no current milestone benefit.

## WHAT WOULD REFUTE IT
Required validation passing, with valid eligible evidence establishing readiness or benefit, would refute the disqualification and null benefit assessment.

## WAS THAT CHECKED
Yes. The artifact records required validation results and counts eligible evidence; a required check failed and the eligible count is zero.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_v684_validation`; `verdict_class`: `disqualified`; `name`: `affected_pytest`, `classification`: `required`, `exit_code`: `1`, `passed`: `false`; `sample_size_budget` → `eligible`: `0`; `acceptance_gate_results` → `validity`: `false`, `readiness`: `0`; `milestone_benefit_score`: `0`.

## RECOMMENDATION
KEEP
