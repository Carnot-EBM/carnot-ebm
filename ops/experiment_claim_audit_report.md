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

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8122_v702_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8123_v703_contract_custody.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact asserts no comparative or scientific claim, explicitly scoping itself to administrative custody validation with zero claim scope. If a downstream consumer were to interpret this artifact as demonstrating scientific performance, capability, or generalization, that interpretation would be refuted by the artifact's own recorded design showing that no model was loaded, no training was conducted, no rival arm was evaluated, and zero independent scientific units were measured.

## WAS THAT CHECKED
No comparative or scientific hypothesis was checked or attempted. The artifact only checked administrative custody preconditions and task contract syntax against authority files (17 checks across 13 tasks in `rows`), while explicitly confirming that no model was invoked and no scientific units were evaluated.

## EVIDENCE
`claim_scope`
`0`
`exposure_scope`
`0`
`honest_verdict`
`"complete_null_contract_custody"`
`methodology_note`
`"Compare complete V703 task bytes with strict authority parsing; independently preserve V702 primaries and conductor skips. No science, training, model loading or service benchmark is measured."`
`inference_substrate`
`"aggregation_from_upstream_artifacts"`
`inference_substrate_class`
`"no_model_load"`
`current_work_receipt`
`"model_invoked": false`
`"performed": false`
`sample_size_budget`
`"administrative_tasks": 13`
`"independent_scientific_units": 0`
`contract_ready_score`
`1`
`"Complete authority agreement and normal owned validation qualify scheduling only."`
`MODEL_SPECS`
`"Record MODEL_SPECS so administrative custody cannot imply scientific benefit."`
`arm`
`"contract_custody"`

## RECOMMENDATION
KEEP

## experiment_8132_v703_service_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8133_v703_arc_reader_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8134_v703_hardware_service_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8135_v703_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
