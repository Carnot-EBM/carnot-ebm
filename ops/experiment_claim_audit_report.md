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
| CANNOT_DETERMINE | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7076_v620_contract_preflight.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7077_v620_sota_ingestion.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7078_v620_gpu_lease_migration.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7079_v620_gpu_lease_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7080_v620_three_family_entrance_bank.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7084_v621_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7085_v621_chat_transport_canary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7086_v621_three_family_entrance_bank.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.149.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write
