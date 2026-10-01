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
| CANNOT_DETERMINE | 7 |

## experiment_7953_v690_contract_methods.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7954_v690_training_coverage.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7955_v690_response_targets.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7956_energy_fit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7958_v690_qwen_response_risk.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7962_v690_arc_supervisor_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact is an observational inventory and receipt manifest that explicitly disclaims any causal or independent benefit and reports no model invocations or level solves, there is no substantive empirical claim to refute. If interpreted narrowly as an assertion that no new supervisor outcomes exist beyond upstream experiment 7949, observing authenticated, non-duplicate supervisor outcome events in `new_event_rows` would refute that inventory state.

## WAS THAT CHECKED
No comparative or empirical hypothesis was tested because no comparative claim was made. For delta tracking, yes: the artifact verified upstream hashes and seen receipts in `preconditions_checked` and recorded zero new event rows.

## EVIDENCE
- `claim_scope`: `exposed_development; observational inventory without independent or causal benefit`
- `honest_verdict`: `complete_null_no_new_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `inference_substrate_class`: `no_model_load`
- `model_invocation_counts`: `0`
- `new_level_solves_claimed`: `0`
- `action`: `retain_no_change_terminal_inventory`
- `solve_provenance`: `Only authenticated input events have live discovery provenance; aggregation claims no solves.`

## RECOMMENDATION
KEEP

## experiment_7964_v690_hardware_evidence.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7965_v690_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
