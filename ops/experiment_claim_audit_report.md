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

## experiment_7944_decision_abstention.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7945_qwen_sentence_risk.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7946_evidence_fragility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no empirical or comparative claim. As a gate receipt, the record of a blocked run would only be falsified if the upstream artifact actually satisfied all gating criteria (`energy_fit_ready_score == 1` and `verdict_class` in `['positive', 'circular_positive', 'null']`).

## WAS THAT CHECKED
No; no experimental evaluation occurred because execution was blocked at `conductor_pre_gate`.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`
`gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7943-energy-fit.energy_fit_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7947_causal_acquisition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; no empirical or comparative claim is made because the artifact is a gate check receipt recording that the experiment was blocked prior to execution.

## WAS THAT CHECKED
No; no experimental evaluation occurred because execution was blocked at the pre-gate check layer.

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

## RECOMMENDATION
KEEP

## experiment_7949_v689_arc_supervisor_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact is an observational inventory and reconciliation receipt that explicitly disclaims independent or causal benefit, contains zero rows, zero model invocations, and zero active experimental arms; consequently, there is no comparative hypothesis or performance claim to falsify. If treated strictly as an accounting receipt asserting that no new supervisor outcomes or level solves exist beyond authenticated upstream inventory, the appearance of novel unauthenticated outcome records in `new_event_rows`, non-zero `new_level_solves_claimed`, or mismatched checksums in `preconditions_checked` would refute that accounting.

## WAS THAT CHECKED
No comparative or causal effect was checked because no comparative evaluation or intervention was executed (`arm_outcomes` is empty and `model_invocation_counts` is 0). For the accounting inventory, reconciliation was checked in `preconditions_checked` and `receipt_inventory`, which validated upstream artifact checksums and confirmed zero new outcome rows.

## EVIDENCE
- `claim_scope`: `exposed_development; observational inventory without independent or causal benefit`
- `arm_outcomes`: `{}`
- `rows`: `[]`
- `new_event_rows`: `[]`
- `new_level_solves_claimed`: `0`
- `model_invocation_counts`: `0`
- `honest_verdict`: `complete_null_no_new_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `inference_substrate_class`: `no_model_load`
- `action`: `retain_no_change_terminal_inventory`
- `receipt_inventory`: `Content hashes prevent retries from inventing new events.`
- `solve_provenance`: `Only authenticated input events carry live discovery provenance; aggregation claims zero solves.`
- `MODEL_SPECS`: `Bind executing producer custody and keep observational inventory distinct from benefit.`

## RECOMMENDATION
KEEP

## experiment_7950_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is a gate-check receipt recording an upstream dependency failure and asserts no substantive or comparative claim.

## WAS THAT CHECKED
No; execution was blocked at `conductor_pre_gate` before any experiment or measurement was run.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7951_v689_hardware_evidence.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7952_v689_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
