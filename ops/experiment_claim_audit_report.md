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
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7904_v686_training_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7904_v686_training_qualification.json.validators.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact records validator execution receipts rather than an empirical or comparative claim.

## WAS THAT CHECKED
No; no experimental hypothesis or comparative claim is evaluated in this receipt artifact.

## EVIDENCE
`receipts`
`actual_exit`
`0`
`exit_code`
`expected_exit`
`passed`
`true`

## RECOMMENDATION
KEEP

## experiment_7905_v686_intervention_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7906_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is a pipeline receipt recording that pre-flight conductor gate checks failed before the run began; it makes no empirical, comparative, or substantive claim that could be refuted.

## WAS THAT CHECKED
No. Execution was blocked at the pre-gate layer prior to running any experiment.

## EVIDENCE
`schema`
`"blocked_gate_check_v1"`
`status`
`"blocked"`
`honest_verdict`
`"blocked_gate_check_failed"`
`blocked_at_layer`
`"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7908_qwen_sufficiency.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7911_v686_arc_supervisor_delta.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7913_v686_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation would refute the headline claim because the artifact asserts no comparative, empirical, or performance claim; refuting a hardware advantage claim would require measured device execution latency or whole-service throughput outperforming host baselines, which this artifact lacks.

## WAS THAT CHECKED
No; live hardware execution, workload benchmarking, and comparator speedup evaluations were not checked or performed.

## EVIDENCE
`"hardware_speedup_claimed"`: `false`
`"hardware_advantage"`: `"unmeasured"`
`"current_measurement"`: `"host receipt analysis only"`
`"board_evidence"`: `"historical custody"`
`"workload"`: `"unavailable"`
`"arm"`: `"historical_accounting"`
`"claim_class"`: `"historical"`
`"current_device_execution_count"`: `0`
`"honest_verdict"`: `"complete_disqualified_required_checks"`
`"verdict_class"`: `"disqualified"`

## RECOMMENDATION
KEEP

## experiment_7914_v686_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
