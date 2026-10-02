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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8008_v694_conditioned_energy_fit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8009_development_decisions.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8010_v694_source_intervention_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation demonstrating that an empirical, comparative, or causal claim was asserted would refute the classification of this artifact as purely protocol scaffolding. If treated as an empirical evaluation of model intervention or sensitivity, observing non-zero complete pairs and valid comparative measurements between arms (`swap` vs. `original`) showing no intervention effect would falsify efficacy. However, the artifact explicitly disclaims making any such claim.

## WAS THAT CHECKED
No. No empirical model evaluations or comparative hypothesis tests were conducted. All model calls were censored, zero generation calls were attempted, and no model was loaded.

## EVIDENCE
- `claim_scope`: `"This invocation freezes a natural source conditioning panel and qualifies scripted transport only. No natural sensitivity, hallucination accuracy or causal mitigation is measured."`
- `fixture_scope`: `"circular_scripted_HTTP_transport_only"`
- `inference_substrate_class`: `"no_model_load"`
- `generation_calls_attempted`: `0`
- `model_loads_attempted`: `0`
- `protocol_preparation`: `true`
- `scientific_benefit`: `false`
- `natural_sensitivity`: `false`
- `complete_pairs`: `0`
- `duplicate_minus_original`: `null`
- `swap_minus_original`: `null`
- `status`: `"censored"`
- `censor_reason`: `"protocol_only_no_model_requested"`
- `probability`: `null`

## RECOMMENDATION
KEEP

## experiment_8011_v694_qwen_source_sensitivity.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8012_budgeted_online_updates.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or empirical claim is asserted to refute. If the experiment had run and claimed improved efficiency from delayed decision-loss update allocation, observing equal or worse decision loss relative to a standard uniform or greedy update baseline at matched budgets would refute it.

## WAS THAT CHECKED
No. The experiment did not execute because upstream prerequisite gates failed, blocking execution at the pre-gate layer before any updates or evaluations could occur.

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
`gate_check_summary`
`gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8008-conditioned-energy-fit.conditioned_fit_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8014_v694_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8016_v694_hardware_update_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8017_v694_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
