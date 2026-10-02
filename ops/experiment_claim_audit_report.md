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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_7997_v693_typed_development_decisions.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7998_v693_selective_feedback_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Selective feedback learning over fallible source-support labels yields a complete null result with zero decision benefit.

## WHAT WOULD REFUTE IT
A statistically significant positive decision benefit (`decision_benefit_score` > 0 or `acceptance_gate_results.benefit` evaluating to `true`) from online feedback updates (`targeted_ipw`, `targeted_unweighted`, or `uniform_ipw`) over the non-updating baseline (`frozen_no_write`).

## WAS THAT CHECKED
Yes. The artifact demonstrated genuine headroom existed (`genuine_headroom`: `true`), confirmed the pipeline's detection capability via synthetic positive controls (`positive_control_results`: `passed`: `true`), and evaluated updates across 20 random seeds against equal-cost rival arms over 254 independent stream slots, observing no decision benefit.

## EVIDENCE
- `"honest_verdict": "complete_null_selective_feedback_learning"`
- `"verdict_class": "null"`
- `"benefit": false`
- `"decision_benefit_score": 0`
- `"genuine_headroom": true`
- `"verifier_is_oracle": false`
- `"readiness": true`
- `"claim_scope": "Completed randomized delayed-feedback mechanism over fallible source-support labels. Natural benefit and retention are reserved for Exp7999. Public pretraining and wider historical exposure remain unknown."`

## RECOMMENDATION
KEEP

## experiment_7999_v693_learning_causal_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8000_v693_delayed_confidence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Adaptive delayed confidence adjustment provides no benefit over a frozen calibration baseline across the evaluated delay windows.

## WHAT WOULD REFUTE IT
A statistically significant positive gain over the frozen baseline in any tested delay condition, demonstrated by a positive bootstrap lower bound (`lower95` > 0), `deviation_gain` = true, `lower_positive` = true, and a positive `confidence_benefit_score` without excess false accepts.

## WAS THAT CHECKED
Yes. Checked in `delay_sensitivity` across delays 20, 24, and 36 using 10,000 bootstrap draws, with genuine headroom present (`errors` = 16) and responsive positive controls verified.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_delayed_confidence"`
- `"confidence_benefit_score"`: `0`
- `"benefit"`: `false`
- `"deviation_gain"`: `false`
- `"lower_positive"`: `false`
- `"support"`: `false`
- `"lower95"`: `-0.04259259259259259`
- `"lower95"`: `-0.05`
- `"errors"`: `16`
- `"verifier_is_oracle"`: `false`

## RECOMMENDATION
KEEP

## experiment_8001_v693_arc_supervisor_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any assertion of comparative advantage, positive model performance, or causal efficacy in the data rows, such as non-zero entries in `new_event_rows`, non-zero `new_level_solves_claimed`, non-empty `arm_outcomes`, or a verdict token claiming experimental success (such as `complete_positive`).

## WAS THAT CHECKED
Yes; the artifact explicitly checks and verifies that no models were executed, no new events or solves were produced, and downstream aggregation imports only an authenticated qualification and receipt inventory.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_no_new_outcomes"`
- `"causal_benefit_claimed"`: `false`
- `"claim_scope"`: `"exposed_development; observational inventory without independent or causal benefit"`
- `"no_new_outcomes"`: `true`
- `"inference_substrate"`: `"aggregation_from_upstream_artifacts"`
- `"inference_substrate_class"`: `"no_model_load"`
- `"model_invocation_counts"`: `0`
- `"new_event_rows"`: `[]`
- `"new_level_solves_claimed"`: `0`
- `"arm_outcomes"`: `{}`

## RECOMMENDATION
KEEP

## experiment_8002_v693_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Current CPU engineering cost on cached sources yields no whole-service speed gain or natural learning benefit (`complete_null_service_cost`).

## WHAT WOULD REFUTE IT
A statistically significant, meaningful reduction in service latency or positive runtime acceleration in `complete_service_cost` or `paired_latency_intervals`, resulting in `acceptance_gate_results.benefit` evaluating to `true`.

## WAS THAT CHECKED
Yes. It was checked across 64 independent source groups and 10 repetitions (640 matched observations per arm/case) in `complete_service_cost`, `paired_latency_intervals`, and `acceptance_gate_results`, with positive controls confirming measurement headroom in `positive_control_results`.

## EVIDENCE
`"honest_verdict"`: `"complete_null_service_cost"`
`"acceptance_gate_results"`: `{"benefit": false, "readiness": true}`
`"claim_scope"`: `"Current CPU engineering cost on cached sources. Matching model acquisition is historical; no whole-service speed gain or natural learning benefit is claimed."`
`"genuine_service_headroom"`: `true`
`"maximum_ideal_complete_acceleration"`: `1.0000334132967006`
`"mean_difference_s"`: `0.0`
`"matched_rows"`: `640`
`"missing_rows"`: `0`
`"completed"`: `64`
`"independent"`: `64`

## RECOMMENDATION
KEEP

## experiment_8003_v693_hardware_sparse_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8004_v693_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
