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

## experiment_8033_v696_scoring_isolation.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8034_fit_likelihood_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An empirical or comparative refutation is not applicable because the artifact contains no scientific or comparative claim; it is a gate-check receipt recording that the experiment was blocked prior to execution due to failed upstream prerequisites.

## WAS THAT CHECKED
No; the experiment did not run.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`duration_s`
`0.0`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8038_v696_windowed_online_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Windowed online learning updates across replay window arms yield no generalized learning benefit under equal adaptive budgets (`complete_null_windowed_trajectories`).

## WHAT WOULD REFUTE IT
An observation in the artifact's evaluation rows showing a statistically meaningful reduction in loss or Brier score (a positive generalized learning benefit) for any of the adaptive replay arms (`recent64`, `cumulative`, or `newest16`) over the baseline (`frozen_no_write`).

## WAS THAT CHECKED
Yes; checked across 20 algorithm seeds and four comparator arms with 56 updates per seed, logged in `update_budget_rows`, `rows`, and `feedback_release_rows`, confirming no advantage over baseline and resulting in a `generalized_learning_benefit_score` of `0`.

## EVIDENCE
`honest_verdict`
`complete_null_windowed_trajectories`
`verdict_class`
`null`
`generalized_learning_benefit_score`
`0`
`claim_scope`
`This invocation replays one exposed development stream under equal adaptive budgets. Temporal support changes; generator and importance weights stay fixed. No retention, independent learning benefit or deployment claim.`
`arms`
`recent64`
`cumulative`
`newest16`
`frozen_no_write`
`learning_trajectory_ready_score`
`1`

## RECOMMENDATION
KEEP

## experiment_8039_v696_learning_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8040_v696_native_transaction_cost.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8041_v696_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8042_v696_precision_fallback_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8043_v696_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or substantive scientific claim is made to refute. If the capstone had asserted a positive capability or generalization claim over baselines, that claim would be refuted by observing zero or negative gain, confidence intervals spanning zero or failing required margins, or gate failures. The artifact lacks model evaluation, lacks rival comparison arms, and explicitly declares that zero independent scientific observations were made.

## WAS THAT CHECKED
No. The artifact lacks any comparative testing against rival baselines or models. It is an administrative aggregation and custody receipt that records upstream gate results and validation command receipts rather than evaluating an empirical hypothesis.

## EVIDENCE
- `"honest_verdict": "complete_blocked_v696_capstone"`
- `"verdict_class": "blocked"`
- `"science_ready": false`
- `"claim_scope": "This invocation binds thirteen administrative dispositions and exposed finite-trajectory reductions. Missing source evidence and complete deployment remain unavailable; historical publication is separate."`
- `"positive_claim": false`
- `"inference_substrate": "aggregation_from_upstream_artifacts"`
- `"inference_substrate_class": "no_model_load"`
- `"MODEL_SPECS": []`
- `"sample_size_budget": "Thirteen administrative dispositions are zero new independent scientific observations."`
- `"generalized_learning_benefit_score": 0`
- `"validation_receipts"`

## RECOMMENDATION
KEEP
