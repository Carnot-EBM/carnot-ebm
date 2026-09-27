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

## experiment_7757_view_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence of experimental results or empirical comparisons; however, because this artifact is a pre-execution gate receipt making no empirical or comparative claim, there is no headline claim to refute.

## WAS THAT CHECKED
No; the artifact lacks execution and evaluation data because the run was halted at the conductor pre-gate check and never executed.

## EVIDENCE
`schema`: `"blocked_gate_check_v1"`
`status`: `"blocked"`
`duration_s`: `0.0`
`honest_verdict`: `"blocked_gate_check_failed"`
`blocked_at_layer`: `"conductor_pre_gate"`
`gate_check_summary`: `"gate-unsat(final): 6 of 9 gate(s) failed; first failure: exp7754-sentence-protocol.sentence_protocol_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7759_v675_qwen_evidence_views.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed evidence-view run is a development diagnostic disqualified from readiness because required checks failed.

## WHAT WOULD REFUTE IT
All required checks passing, with the readiness gate passing.

## WAS THAT CHECKED
Yes. The artifact records the required checks and readiness gate; a required coverage check failed.

## EVIDENCE
`claim_scope`: `exposed_development_diagnostic_only; no learned_head_gate`; `honest_verdict`: `complete_disqualified_required_checks`; `required_checks_passed`: `false`; `changed_module_coverage_report`: `exit_code`: `2`, `passed`: `false`; `readiness`: `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_7760_v675_online_runner.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The artifact reports readiness of a synthetic delayed-feedback fixture; it makes no comparative benefit claim.

## WHAT WOULD REFUTE IT
A cold-restart mismatch, invalid row reduction, or failed end-to-end fixture run would refute readiness. A tie or loss against a serious baseline would matter to an added-value claim, which this artifact does not make.

## WAS THAT CHECKED
Yes for fixture readiness: the restart, row reduction, and end-to-end checks passed. No comparative benefit result is reported.

## EVIDENCE
`claim_scope`: `synthetic_delayed_feedback_fixture_only; natural_benefit_unmeasured`; `decision_benefit`: `null`; `online_runtime_ready_score`: `1`; `parity`: `true`; `raw_reduction`: `valid`: `true`; `task_e2e`: `passed`: `true`.

## RECOMMENDATION
KEEP

## experiment_7762_v675_independent_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no benefit claim to refute. An existing declared producer would contradict the artifact’s reported missing-producer status.

## WAS THAT CHECKED
Yes, for both declared producers in the precondition and gate checks. Neither was available, so no scientific result was tested.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_validation`; `producer_states`: `Exp7758`: `missing`, `Exp7761`: `missing`; `effective_independent_n`: `0`; `reduction`: `null`.

## RECOMMENDATION
KEEP

## experiment_7763_v675_arc_runner_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative performance claim to refute. A passing required validation check would overturn the reported disqualification.

## WAS THAT CHECKED
Yes for the validation gate: the required check ran and failed. No scored panel result was produced to test comparative performance.

## EVIDENCE
`honest_verdict` `complete_disqualified_required_runner_validation` `field` `full_pytest` `expected` `0` `observed` `2` `passed` `false` `effective_independent_n` `0` `started` `0` `completed` `0`

## RECOMMENDATION
KEEP

## experiment_7764_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no performance claim to falsify. A passing gate record would contradict the reported blocked status.

## WAS THAT CHECKED
Yes. The gate records show two failures, and the experiment stopped before producing scored selection data.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7763-arc-runner-qualification.organic_runner_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7765_v675_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Service cost and readiness remain unmeasured because no qualified fit was available.

## WHAT WOULD REFUTE IT
A qualified current fit in the checked input, followed by eligible service cost rows, would refute the stated reason for blocking measurement.

## WAS THAT CHECKED
Yes. The fit eligibility gate failed, and the artifact records zero eligible runs and no service cost rows.

## EVIDENCE
`honest_verdict`: `complete_blocked_missing_qualified_fit`; `fitted_heads`: `conductor_pre_gate`; `expected`: `qualified_current_fit`; `passed`: `false`; `eligible`: `0`; `service_cost_rows`: `[]`; `batch_1_p95_observed_ms`: `null`.

## RECOMMENDATION
KEEP

## experiment_7766_v675_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V675 capstone is disqualified because required validation failed and the evidence is insufficient to open its acceptance gates.

## WHAT WOULD REFUTE IT
The artifact’s own checks showing that required validation passed and the necessary producers were eligible, while still reporting the capstone as disqualified.

## WAS THAT CHECKED
Yes. The validation receipts report failed required checks, and the gate and producer rows record the resulting ineligibility.

## EVIDENCE
`honest_verdict` `complete_disqualified_v675_capstone_validation`; `required_checks_passed` `false`; `full_python_suite.exit_code` `2`; `producer_eligible` `false`; `validity` `false`; `capstone_complete_score` `0`.

## RECOMMENDATION
KEEP
