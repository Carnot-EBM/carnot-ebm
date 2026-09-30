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
| NO_CLAIM | 8 |

## experiment_7894_v685_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation showing that the artifact asserts a positive capability, readiness score, or decision benefit despite failing required prerequisite checks. Because the artifact explicitly records a disqualified outcome and makes no comparative or capability claims, there is no affirmative claim to refute.

## WAS THAT CHECKED
No. The experiment was disqualified at the gate due to required check failures; consequently, no comparative evaluation of candidate efficacy or readiness was conducted.

## EVIDENCE
- `"honest_verdict"`: `"complete_disqualified_required_checks"`
- `"verdict_class"`: `"disqualified"`
- `"energy_fit_ready_score"`: `0`
- `"validity"`: `false`
- `"readiness"`: `0`
- `"decision_benefit"`: `null`
- `"efficiency"`: `null`
- `"probability_quality"`: `null`
- `"retention"`: `null`
- `"historical_required_failures"`

## RECOMMENDATION
KEEP

## experiment_7895_decision_abstention.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to refute. A successful measurement recorded in this artifact would contradict its blocked status.

## WAS THAT CHECKED
Yes, for the gate status: `gates_evaluated` records two failed gates. The artifact contains no typed-decision or abstention measurements.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7894-energy-fit.energy_fit_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7896_qwen_sufficiency.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is a pipeline gate receipt recording that execution was blocked due to failed upstream gates, making no comparative or empirical claim.

## WAS THAT CHECKED
No; execution was halted at the pre-gate layer before the experiment ran.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`duration_s`
`0.0`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7897_causal_acquisition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact is an execution gate receipt asserting no substantive empirical or comparative claim, there is no experimental hypothesis to refute; refuting the receipt's own recorded state would require demonstrating that upstream prerequisites were satisfied (`energy_fit_ready_score` equal to 1 and `verdict_class` matching the expected set) and that execution was not halted at `conductor_pre_gate`.

## WAS THAT CHECKED
No. The run was aborted at `conductor_pre_gate` before execution began, so no experimental evaluation or comparative test was conducted.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`duration_s`
`0.0`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_at_layer`
`conductor_pre_gate`
`gate_check_summary`
`gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7894-energy-fit.energy_fit_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7899_v685_arc_supervisor_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The observation of non-zero supervisor outcomes, candidate firings, or populated evaluation rows generated during the delta run would refute the status of having no new outcomes, but the artifact is an observational receipt tracking upstream hashes and live state delta rather than advancing a comparative or performance claim.

## WAS THAT CHECKED
Yes, in `precheck_reduce_validate` and `preconditions_checked`, where upstream baseline hashes, the registry precheck, and validation suites were verified; however, the artifact lacks active model loading or an execution harness producing comparative task outcomes.

## EVIDENCE
`"claim_scope": "exposed_development; observational live receipt delta"`
`"honest_verdict": "complete_null_no_new_supervisor_outcomes"`
`"verdict_class": "null"`
`"target_model": "none"`
`"inference_substrate": "aggregation_from_upstream_artifacts"`
`"inference_substrate_class": "no_model_load"`
`"new_outcome_count": 0`
`"new_live_outcome_count": 0`
`"firings": 0`
`"rows": []`
`"outcome_rows": []`
`"decision_benefit": null`
`"efficiency": null`
`"probability_quality": null`
`"retention": null`

## RECOMMENDATION
KEEP

## experiment_7900_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no service-cost result or comparative claim to refute. A passing upstream gate would contradict the reported reason for blocking.

## WAS THAT CHECKED
Yes, for the gate: the artifact records the evaluated conditions and their outcomes. No service-cost measurement was run.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "energy_fit_ready_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7901_v685_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any observed hardware acceleration, speedup, or comparative performance advantage over a baseline demonstrated in the data rows, or an assertion of positive hardware readiness.

## WAS THAT CHECKED
No; no device-level executions were performed (`current_device_execution_count` is 0), actual workloads were absent, and required validation checks failed.

## EVIDENCE
`hardware_speedup_claimed`
`false`
`hardware_evidence_ready_score`
`0`
`current_device_execution_count`
`0`
`hardware_advantage`
`unmeasured`
`current_measurement`
`host receipt analysis only`
`board_evidence`
`historical custody`
`arm`
`historical_accounting`
`status`
`historical_read_only`
`honest_verdict`
`complete_disqualified_required_checks`
`verdict_class`
`disqualified`

## RECOMMENDATION
KEEP

## experiment_7902_v685_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to refute. The reported disqualification would be contradicted if the required checks passed and the required producers supplied qualified evidence.

## WAS THAT CHECKED
Yes for the disqualification status: the gate checks and failure records show failed required checks and blocked producers. The artifact does not assert an independent benefit.

## EVIDENCE
`honest_verdict` `complete_disqualified_required_checks`; `verdict_class` `disqualified`; `independent_benefit` `null`; `decision` `blocked`; `capstone_execution_ready_score` `0`

## RECOMMENDATION
KEEP
