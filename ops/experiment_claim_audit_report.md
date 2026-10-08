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
| NO_CLAIM | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8263_v714_protocol_conformance.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8264_v714_evidence_view_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Since the artifact is an execution receipt recording a preflight-blocked run rather than an empirical study, there is no substantive or comparative performance claim to refute. Any positive capability or comparative advantage claim would be refuted if completed model execution failed to demonstrate statistical separation or benefit over baseline arms under full evaluation.

## WAS THAT CHECKED
No. Execution was blocked at the preflight stage due to CUDA runtime unavailability, resulting in zero completed generation calls and zero evaluated rows.

## EVIDENCE
`"honest_verdict"`: `"complete_blocked_CUDA_runtime_available"`
`"verdict_class"`: `"blocked"`
`"inference_substrate_class"`: `"blocked_no_run"`
`"inference_mode"`: `"no_model_load"`
`"scientific_benefit_measured"`: `false`
`"completed_count"`: `0`
`"censored_count"`: `36`
`"generated_tokens"`: `0`
`"passed"`: `false`
`"actual_exit"`: `1`
`"ready_for_gated_consumers"`: `false`
`"methodology_note"`: `"Frozen public focal selection and length matching; one bounded local model lifetime, exact request/view custody and independent syntax reduction. No human labels or generalization claim. Conservative maximum per-token timing forecasts include full focal rosters, startup, shutdown and retries."`

## RECOMMENDATION
KEEP

## experiment_8265_fit_view_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Data showing that the experimental run executed and evaluated comparative hypotheses, or that upstream prerequisite checks were satisfied rather than failing at the conductor pre-gate.

## WAS THAT CHECKED
No. The artifact is an operational gate receipt showing execution halted before the run began (`duration_s`: 0.0) at the conductor pre-gate layer.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_at_layer`
`conductor_pre_gate`
`gate_check_summary`
`gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8264-evidence-view-canary.view_canary_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8266_tune_view_capture.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8272_v714_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8273_v714_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8274_v714_gatemate_physical_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact explicitly scopes itself as an evidence ledger and disclaims any scientific benefit, it asserts no comparative, performance, or empirical generalization claim to refute. If interpreted as a substantive empirical claim that GateMate hardware is physically non-functional or unreachable, that claim would be refuted by an authenticated physical-change receipt or an active JTAG probe reading a valid GM1Ax IDCODE (`0x20000001`).

## WAS THAT CHECKED
No. The artifact did not probe physical hardware reachability (`current_reachability` was `not_probed` and `current_execution_authorized` was `false`). It executed only document hash comparisons against upstream receipts.

## EVIDENCE
- `"claim_scope"`: `"Evidence ledger; future physical preflight remains unexecuted"`
- `"arm"`: `"documentation_audit"`
- `"methodology"`: `"Authenticated Exp8260 frontier and original transcript; qualified dry-run receipt parsing and document hash comparison. No model load or device execution."`
- `"scientific_benefit_score"`: `0`
- `"generalized_learning_benefit_score"`: `0`
- `"independent_generalization_score"`: `0`
- `"honest_verdict"`: `"complete_blocked_gatemate_physical_change"`
- `"verdict_class"`: `"blocked"`
- `"inference_substrate"`: `"aggregation_from_upstream_artifacts"`
- `"inference_substrate_class"`: `"no_model_load"`
- `"model_invoked"`: `false`
- `"current_hardware_execution"`: `false`
- `"current_reachability"`: `"not_probed"`
- `"status"`: `"excluded"`
- `"completed"`: `false`

## RECOMMENDATION
KEEP

## experiment_8275_v714_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation refuting a positive outcome is inapplicable because the artifact records blocked execution receipts and unmeasured hypotheses without asserting any comparative superiority, empirical gain, or scientific finding.

## WAS THAT CHECKED
No; evaluation was blocked before hypothesis measurements could occur.

## EVIDENCE
`"status": "blocked_unmeasured"`
`"failed_operand": "eligible_registered_audit_primitives"`
`"completed_count": null`
`"supported_transferable_evidence": false`
`"arc_evidence_ready_score": 0`

## RECOMMENDATION
KEEP
