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

## experiment_8226_learning_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A substantive claim evaluating utility loss and retention from learning traces would be refuted by degraded utility trajectories or inferior retention relative to baseline traces; however, no empirical or comparative claim is made because this artifact is solely an upstream pre-gate check receipt.

## WAS THAT CHECKED
No. The audit experiment never ran because upstream prerequisite gates failed, resulting in zero execution duration and no evaluation of model traces.

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

## experiment_8227_v711_concurrency_canary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8228_concurrent_service.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An empirical or comparative finding being asserted despite the experiment never having run, or execution logs demonstrating that the workload ran even though preconditions failed. Because this artifact is strictly a pre-flight execution receipt recording a blocked run rather than reporting experimental results, there is no substantive or comparative claim to refute.

## WAS THAT CHECKED
No; the experiment was never executed because upstream gates failed prior to launch at the conductor pre-gate layer.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`
`gate_check_summary`: `gate-unsat(final): 3 of 4 gate(s) failed; first failure: exp8227-concurrency-canary.concurrent_canary_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8229_v711_arc_outcome_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8230_v711_kv260_workload_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8231_v711_polarfire_state_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is an execution receipt and boundary scaffolding record that explicitly disclaims performance benefits and comparative advantages. If construed as an affirmative claim that device or learning benefits were evaluated and found absent, observing non-zero measured device execution benefit, authorized FPGA fabric execution, or passing generalized learning scores would refute it.

## WAS THAT CHECKED
No. The artifact contains no comparative arms, baseline rival evaluations, or active hardware probing. Device execution and reachability were explicitly unprobed (`current_device_execution_count` is 0, `current_reachability` is `"not_probed"`), and the learning branch was disqualified upstream.

## EVIDENCE
`"claim_scope"`: `"Portable state and historical Linux CPU dispatch boundary; no device or learning benefit"`
`"polarfire_boundary_ready_score"`: `"Qualified host inventory remains ready despite blocked learning; no fabric, reachability or benefit claim."`
`"honest_verdict"`: `"complete_disqualified_owned_checks"`
`"verdict_class"`: `"disqualified"`
`"measured_device_benefit"`: `false`
`"current_device_execution_count"`: `0`
`"current_model_calls"`: `0`
`"generalized_learning_benefit_score"`: `0`
`"independent_generalization_score"`: `0`
`"polarfire_boundary_ready_score"`: `0`
`"benefit_score"`: `0`
`"current_board_execution"`: `false`
`"current_reachability"`: `"not_probed"`
`"device_work_authorized"`: `false`
`"fabric_use"`: `false`
`"required_checks_passed"`: `false`

## RECOMMENDATION
KEEP

## experiment_8232_v711_gatemate_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact makes no comparative, algorithmic, or empirical performance claim to refute; it is an administrative scaffolding audit and blocking receipt. If construed as an operational claim that the GateMate device remains physically blocked and unprobed, it would be refuted by data showing an authenticated, valid GM1Ax IDCODE read (such as `0x20000001`), a successful bitstream flash, or a valid, operator-authored physical change receipt.

## WAS THAT CHECKED
No. Physical hardware execution and live JTAG reachability were not probed (`current_hardware_execution` is false, `current_reachability` is `not_probed`, and `current_jtag_retry_count` is 0). The artifact only audited historical documentation and checked candidate markdown entries in `ops/known-issues.md` without running any device-level checks.

## EVIDENCE
- `claim_scope`: `"Documentation audit only; recorded changes require future device preflight"`
- `methodology`: `"Authenticated historical GateMate custody and dry-run structured operator receipt audit. No model, benchmark, JTAG retry, flash or device execution."`
- `arm`: `"documentation_audit"`
- `honest_verdict`: `"complete_blocked_gatemate_physical_change"`
- `verdict_class`: `"blocked"`
- `MODEL_SPECS`: `[]`
- `model_invoked`: `false`
- `scientific_benefit_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `current_hardware_execution`: `false`
- `current_reachability`: `"not_probed"`
- `execution_ready_score`: `"Audit readiness grants no device acceptance or scientific benefit."`

## RECOMMENDATION
KEEP

## experiment_8233_v711_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
