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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 5 |

## experiment_8113_radial_decision_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no comparative or empirical claim, serving only as an operational receipt recording that experiment execution was blocked at pre-run gate evaluation.

## WAS THAT CHECKED
No; execution was blocked prior to running the experiment, so no empirical evaluation occurred.

## EVIDENCE
`schema`: `"blocked_gate_check_v1"`
`status`: `"blocked"`
`honest_verdict`: `"blocked_gate_check_failed"`
`duration_s`: `0.0`
`blocked_at_layer`: `"conductor_pre_gate"`
`gate_check_summary`: `"gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8112-fit-source-capture.fit_capture_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_8116_v702_independent_online_memory.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8117_learning_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact is a gate-check receipt recording an upstream dependency failure and asserts no comparative or empirical claim.

## WAS THAT CHECKED
No; the experiment was halted at the conductor pre-gate layer prior to execution, so no experimental hypothesis or comparator was evaluated.

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

## experiment_8118_v702_fresh_acquisition_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8119_v702_batched_service_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8120_v702_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8121_v702_hardware_batch_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The evaluated FPGA hardware boards demonstrate no measured whole-service acceleration or qualifying batch execution, establishing an honest null hardware batch boundary result.

## WHAT WOULD REFUTE IT
A board row in the artifact recording measured whole-service acceleration or qualifying batch execution (e.g., a non-null `measured_current_latency_ms` demonstrating speedup, `terminal_criterion_met` evaluating to `true` for FPGA batch workloads, or `acquisition_relevance` showing positive whole-service benefit rather than deferral).

## WAS THAT CHECKED
Yes. The audit checked each evaluated board arm (`KV260` and `PolarFire`) for current hardware execution, whole-service speedup, and qualifying batch prerequisites, finding that batch inputs were unavailable, current hardware execution was not performed, and neither board met the whole-service acceleration criteria.

## EVIDENCE
- `batch_input_status`: `unavailable`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `arm`: `historical_accounting`
- `board`: `KV260`
- `historical_verdict`: `historical_fabric_k_max<=5`
- `current_hardware_execution`: `false`
- `terminal_criterion_met`: `false`
- `measured_current_latency_ms`: `null`
- `board`: `PolarFire`
- `historical_verdict`: `historical_linux_cpu_only`

## RECOMMENDATION
KEEP

## experiment_8122_v702_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
