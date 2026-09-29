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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7867_v683_natural_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or natural-performance claim to refute. A failed fixture run would challenge the artifact’s limited completion record.

## WAS THAT CHECKED
Yes for fixture completion: the artifact records completed rows and an online probe. It reports no natural measurement or decision benefit.

## EVIDENCE
`claim_scope`: `fixture_mechanics_only`; `honest_verdict`: `complete_circular_positive_fixture_runtime`; `natural_measurement_performed`: `false`; `decision_benefit`: `null`; `verifier_is_oracle`: `true`; `sample_size_budget`: `completed`: `9`.

## RECOMMENDATION
KEEP

## experiment_7868_v683_intervention_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any affirmative claim of capability, intervention benefit, or production readiness would be refuted if required gate validation failed or if model inference was never executed. However, because this artifact is an execution receipt for a disqualified run that asserts no comparative advantage or positive empirical finding, there is no headline claim to refute.

## WAS THAT CHECKED
No. The artifact is a test and validation receipt where required suite checks failed, assigning a readiness score of zero and disqualifying the protocol without evaluating any comparative hypothesis or executing model inference.

## EVIDENCE
`claim_scope`
`CPU scripted fixture conformance; exposed development evidence only`
`current_work_receipt`
`model_invoked`
`false`
`acceptance_gate_results`
`validity`
`readiness`
`decision_benefit`
`efficiency`
`probability_quality`
`retention`
`null`
`gate_check_summary`
`artifact_field`
`full_pytest.passed`
`expected`
`true`
`observed`
`passed`
`honest_verdict`
`complete_disqualified_required_checks`
`intervention_protocol_ready_score`
`0`
`target_model`
`none (no pretrained model)`
`validation_receipts`
`verdict_class`
`disqualified`

## RECOMMENDATION
KEEP

## experiment_7874_v683_arc_supervisor_delta.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7876_v683_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A measurement demonstrating hardware speedup, whole-service benefit, or active device execution would contradict the artifact's categorization as an unexecuted receipt analysis, but the artifact itself asserts no comparative or performance claim.

## WAS THAT CHECKED
No; the artifact is a custody and validation receipt recording historical provenance and failed required validation checks, not an empirical test of a comparative claim.

## EVIDENCE
`hardware_speedup_claimed`
`false`
`current_measurement`
`receipt analysis only`
`hardware_advantage`
`unmeasured`
`board_evidence`
`historical custody`
`arm`
`historical_accounting`
`current_device_execution_count`
`0`
`current_hardware_execution`
`false`
`honest_verdict`
`complete_disqualified_required_checks`
`verdict_class`
`disqualified`

## RECOMMENDATION
KEEP

## experiment_7877_v683_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to refute. A contrary qualification record would show the required validation passing and the prerequisite gaps resolved.

## WAS THAT CHECKED
The artifact checked qualification and recorded failures. It did not test comparative benefit.

## EVIDENCE
`receipt_delta_only`, `new_generalization`, `false`, `decision_benefit`, `null`, `readiness`, `0`, `validity`, `0`, `full_pytest`, `passed`, `false`, `timed_out`, `true`, `verdict_class`, `disqualified`

## RECOMMENDATION
KEEP

## experiment_7878_v683_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. Passing the required validation would contradict the recorded disqualification.

## WAS THAT CHECKED
Yes, for validation: the artifact records a failed required check and failed owned validation. It does not assert a scientific benefit.

## EVIDENCE
`"honest_verdict": "complete_disqualified_required_v683_validation"`; `"verdict_class": "disqualified"`; `"status": "failed_owned_validation"`; `"milestone_benefit_score": 0`; `"passed": false`

## RECOMMENDATION
KEEP

## experiment_7869_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; this artifact is a pre-execution gate receipt and asserts no empirical or comparative claim regarding source energy or matched controls.

## WAS THAT CHECKED
No; the experiment never executed and was aborted at `conductor_pre_gate` due to upstream prerequisite gate check failures.

## EVIDENCE
- `schema`: `"blocked_gate_check_v1"`
- `status`: `"blocked"`
- `honest_verdict`: `"blocked_gate_check_failed"`
- `blocked_at_layer`: `"conductor_pre_gate"`
- `duration_s`: `0.0`
- `gate_check_summary`: `"gate-unsat(final): 3 of 6 gate(s) failed; first failure: exp7866-source-boundary.source_boundary_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7871_qwen_sufficiency.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None, as this artifact is a pre-execution receipt documenting a blocked run rather than asserting an empirical or comparative claim.

## WAS THAT CHECKED
No; upstream dependency checks failed, blocking execution at the conductor pre-gate stage before any experiment data could be collected or evaluated.

## EVIDENCE
`schema`
`"blocked_gate_check_v1"`
`status`
`"blocked"`
`honest_verdict`
`"blocked_gate_check_failed"`
`duration_s`
`0.0`
`blocked_at_layer`
`"conductor_pre_gate"`
`gate_check_summary`
`"gate-unsat(final): 5 of 6 gate(s) failed; first failure: exp7866-source-boundary.source_boundary_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP
