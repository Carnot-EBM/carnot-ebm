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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 5 |

## experiment_7916_v687_training_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7916_v687_training_qualification.json.validators.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7917_v687_intervention_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7918_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation that the experiment executed and produced comparative model or energy fitting results; however, as an operational gate-check receipt recording pre-flight blockage, no comparative or empirical claim is made.

## WAS THAT CHECKED
No; the experiment was never executed because pre-flight gate checks blocked execution at the conductor layer.

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

## experiment_7920_v687_qwen_sufficiency.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Qwen3.8-27B exhibits null source sensitivity across qualified source intervention views, demonstrating no evidence sufficiency or correctness certification.

## WHAT WOULD REFUTE IT
A statistically significant positive source sensitivity score (such as `mean_neighbors_minus_filler` significantly exceeding zero with a confidence interval strictly above zero and non-null scientific or decision benefits), demonstrating that the model systematically distinguishes supporting witness contexts from filler.

## WAS THAT CHECKED
Yes. The experiment executed 170 live generation calls across 28 independent families across multiple source intervention arms (`full_source`, `witness_only`, `witness_neighbors`), giving positive sensitivity a genuine opportunity to appear. Instead, the observed mean difference was negative (-0.0946) with a null interval, honestly confirming the null verdict.

## EVIDENCE
- `"honest_verdict": "complete_null_source_sensitivity"`
- `"verdict_class": "null"`
- `"claim_scope": "exposed_development source sensitivity only"`
- `"mean_neighbors_minus_filler": -0.09464285714285714`
- `"interval": null`
- `"interpretation": "source sensitivity only; no correctness or evidence sufficiency certification"`
- `"independent_families": 28`
- `"scientific_benefit": null`
- `"decision_benefit": null`
- `"verifier_is_oracle": false`
- `"generation_calls_completed": 170`

## RECOMMENDATION
KEEP

## experiment_7924_v687_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7926_v687_hardware_evidence.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7927_v687_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation asserting an empirical comparative scientific benefit, performance improvement over a baseline, or active model inference rather than administrative disposition.

## WAS THAT CHECKED
Yes. The artifact verified upstream readiness and recorded that no model was loaded (`model_invoked` is `false`), all outcome rows are assigned to `administrative_disposition`, and scientific benefit metrics (`decision_benefit`, `efficiency`, `probability_quality`, `retention`) remain `null`.

## EVIDENCE
`honest_verdict`
`complete_blocked_missing_science`
`verdict_class`
`blocked`
`inference_substrate`
`aggregation_from_upstream_artifacts`
`inference_substrate_class`
`no_model_load`
`model_invoked`
`false`
`target_model`
`none`
`performed`
`false`
`independent_benefit`
`false`
`decision_benefit`
`null`
`efficiency`
`null`
`probability_quality`
`null`
`retention`
`null`
`arm`
`administrative_disposition`
`scientifically_independent`
`0`

## RECOMMENDATION
KEEP
