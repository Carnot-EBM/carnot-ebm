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
| CLAIM_OVERSTATED | 1 |
| CANNOT_DETERMINE | 7 |

## experiment_7672_v669_bound_relations.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The bound relation verification protocol is ready, achieving 100% verification accuracy across 72 test fixture groups.

## WHAT WOULD REFUTE IT
Any incorrect classification, unhandled error, or failure to outperform baseline methods when evaluated against an independent ground-truth oracle on natural data, rather than against a constructed fixture suite where truth is defined by the verifier itself.

## WAS THAT CHECKED
No. Refutation was not given a chance to occur because correctness was evaluated entirely against constructed fixtures where the verifier served as its own oracle (`verifier_is_oracle` is `true`). The `freshness` gate was not passed (`passed`: `null`) with zero fresh natural groups evaluated (`fresh_natural_groups`: `0`), and all eight exposed pilot natural answers failed extraction with `unhandled_answer_form`.

## EVIDENCE
- `"honest_verdict"`: `"complete_circular_positive_bound_relation_protocol_ready"`
- `"verdict_class"`: `"circular_positive"`
- `"verifier_is_oracle"`: `true`
- `"relation_protocol_ready_score"`: `1`
- `"provenance"`: `"exact_fixture_oracle"`
- `"oracle_scope"`: `"exact_fixture_only"`
- `"prior_exposure"`: `"Eight V664 pilots exposed before this run; fixture oracle truth is constructed."`
- `"limits"`: `"No natural-answer accuracy or learned-verifier advantage."`
- `"fresh_natural_groups"`: `0`
- `"gate"`: `"freshness"`
- `"passed"`: `null`
- `"gate"`: `"readiness"`
- `"passed"`: `true`
- `"reason"`: `"unhandled_answer_form"`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7673_v669_fresh_relation_cohort.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7674_relation_energy.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7676_v669_qwen_quote_relations.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7679_v669_independent_evidence_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7680_v669_arc_probe_protocol.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7681_v669_arc_live_probes.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7684_v669_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [
