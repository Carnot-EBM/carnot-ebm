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
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 6 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8099_v701_fit_source_capture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8100_radial_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence that upstream prerequisite `exp8099-fit-source-capture` actually satisfied `fit_capture_ready_score == 1`, or data showing that the training pipeline and experimental arms were executed and produced comparative results.

## WAS THAT CHECKED
No, the run was halted at the conductor pre-gate layer prior to execution due to an unsatisfied upstream dependency check.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`
`"gate_check_summary": "gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8099-fit-source-capture.fit_capture_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_8102_v701_learning_stream_capture.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8105_v701_native_radial_kernel.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8106_v701_radial_service_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8107_v701_arc_supervisor_evidence.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8108_v701_radial_hardware_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8109_v701_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
