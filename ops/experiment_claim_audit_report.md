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
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 6 |

## experiment_8155_v705_reserved_evidence_capture.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8156_v705_decision_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8157_release_aware_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No empirical observation applies because this artifact is an execution receipt documenting a blocked run rather than an experimental evaluation asserting a comparative or scientific claim.

## WAS THAT CHECKED
No; the experiment did not execute because it was blocked at the conductor pre-gate layer.

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

## experiment_8159_v705_durable_batch_service.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8160_v705_shared_acquisition_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8161_v705_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8162_v705_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acquisition and acceleration remain blocked because upstream acquisition composition qualification failed, host batch software speedup is bounded by Amdahl's law to at most 1.014x without redesign, and historical FPGA boards demonstrate no measured whole-service benefit.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if upstream acquisition composition passed qualification (`composition: true`), or if host batch profiling demonstrated that removable arithmetic accounted for a sufficient fraction of runtime to allow a >= 100x speedup without redesign (`whole_workload_redesign_required: false`), or if any physical FPGA board demonstrated active execution and verified whole-service latency reduction (`terminal_criterion_met: true` or `current_hardware_execution: true`).

## WAS THAT CHECKED
Yes. Upstream composition readiness was evaluated in `acceptance_gates` (where `composition` failed as `false`) and in `amdahl_bounds` (where the `exp8160` arm was marked `unavailable` with `units: 1`). Removable arithmetic fractions and ceilings were evaluated across 540 units in `amdahl_bounds` and 3,566 rows in `workload_rows` (where `outer_ceiling` peaked at `1.0138852476394273` and `whole_workload_redesign_required` remained `true`). Board execution and service timing were checked across KV260, PolarFire, and GateMate in `board_rows` (where all three recorded `terminal_criterion_met: false`, `current_hardware_execution: false`, and `defer: no measured board whole-service benefit`).

## EVIDENCE
- `honest_verdict`: `complete_blocked_acquisition_composition_ready_score`
- `verdict_class`: `blocked`
- `claim_scope`: `Durable host batch and composed request software ceilings; historical board custody only`
- `composition`: `false`
- `host_batch`: `true`
- `outer_ceiling`: `1.0138852476394273`
- `whole_workload_redesign_required`: `true`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `terminal_criterion_met`: `false`
- `current_hardware_execution`: `false`
- `verifier_is_oracle`: `false`
- `hardware_integration_executed`: `false`

## RECOMMENDATION
KEEP

## experiment_8163_v705_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
