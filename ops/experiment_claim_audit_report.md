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

## experiment_8168_sentence_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation showing that the upstream gate condition was satisfied, that the experiment actually executed rather than terminating at pre-flight check, or any empirical performance claim resulting from the run. Because this artifact is an execution gate receipt recording a blocked run, no empirical or comparative claim is made to refute.

## WAS THAT CHECKED
No; the experiment was blocked at `conductor_pre_gate` prior to execution, so no experimental trials or comparative evaluations were run.

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
`gate_check_summary`
`gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp8167-fit-sentence-capture.fit_trainable_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8171_v706_released_feedback_learning.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8172_v706_learning_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Online learning with error-center updates confers no later typed cost benefit over the fixed-public-center baseline on later stream slots (`complete_null_no_later_typed_cost_benefit`).

## WHAT WOULD REFUTE IT
The headline null claim would be refuted by a statistically significant, non-zero typed cost reduction for `error_center` over `fixed_public_center` on the later stream slots—concretely, an observation where `paired_gain_interval` yields a `lower_bound` ≥ `0.02` (`gain_lower_bound_minimum`), `improved_sources` ≥ `5` (`beneficial_sources_minimum`), or `h2_passed` evaluates to `true`.

## WAS THAT CHECKED
Yes. Refutation was given a real opportunity to occur: headroom was present (`available_typed_cost_headroom` of `0.6265822784810127`), detector sensitivity was verified by a passing positive control (`positive_control`), and 158 independent completed sources were evaluated across multiple moving-block bootstrap resamples (`paired_intervals` at block lengths 8, 16, and 32) in `audit_statistics` and `reductions`.

## EVIDENCE
- `"honest_verdict": "complete_null_no_later_typed_cost_benefit"`
- `"h2_passed": false`
- `"improved_sources": 0`
- `"available_typed_cost_headroom": 0.6265822784810127`
- `"other_control_cost_advantage": 0.0`
- `"lower_bound": 0.0`
- `"mean_gain": 0.0`
- `"support_sufficient": true`
- `"completed_count": 158`
- `"gain_lower_bound_minimum": 0.02`
- `"beneficial_sources_minimum": 5`

## RECOMMENDATION
KEEP

## experiment_8173_v706_service_validation.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8174_v706_complete_request_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native atomic batching fails to achieve the NFR01 speedup threshold (lower 95% bound >= 10x) over Python durable batching for complete request latency, resulting in an honest null.

## WHAT WOULD REFUTE IT
Observing a statistically significant paired speedup ratio with `lower95 >= 10` (`nfr01_met: true`) while preserving behavioral equivalence (`equivalent_behavior_score: 1`).

## WAS THAT CHECKED
Yes; checked across 24 independent source pairs (48 completed requests) under live GPU inference with 10,000 bootstrap resamples in `paired_speed_intervals` and evaluated against `nfr01_lower95`.

## EVIDENCE
- `"honest_verdict": "complete_null_complete_request_cost"`
- `"verdict_class": "null"`
- `"nfr01_met": false`
- `"nfr01_lower95": 10`
- `"estimate": 1.0798600220763859`
- `"lower95": 0.6999568606412714`
- `"upper95": 1.6827817162346008`
- `"completed_count": 48`
- `"independent_count": 24`
- `"equivalent_behavior_score": 1`
- `"claim_scope": "Independent complete durable requests; descriptive pipeline timing when outputs or decisions differ; no model quality or learning claim"`

## RECOMMENDATION
KEEP

## experiment_8175_v706_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8176_v706_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware arithmetic acceleration is bounded to an Amdahl outer ceiling of ≤ 1.014x speedup because retained non-arithmetic work dominates total latency, blocking hardware acceleration without a whole-workload redesign.

## WHAT WOULD REFUTE IT
Observing an execution arm where retained non-arithmetic work is low enough to yield an Amdahl outer ceiling approaching the 100x target (i.e., arithmetic fraction ≥ 0.99 or retained work fraction ≤ 0.01), or observing actual hardware execution with measured whole-service latency reduction and terminal criteria met on attached boards.

## WAS THAT CHECKED
Yes. Amdahl outer ceilings and retained overhead components (including durable fsync, queue wait, and lifecycle times) were profiled across 3,808 rows over five execution arms in `amdahl_bounds`, `workload_rows`, and `rows`, and physical hardware execution criteria were explicitly evaluated in `board_rows`.

## EVIDENCE
- `honest_verdict`: `complete_blocked_composition_replay_ready_score`
- `verdict_class`: `blocked`
- `claim_scope`: `Durable host batch and composed request software ceilings; historical board custody only`
- `measured_bottleneck`: `Retained acquisition and queue dominate complete requests; durable storage dominates host batches.`
- `whole_workload_redesign_required`: `true`
- `outer_ceiling`: `1.0138852476394273`
- `outer_ceiling`: `1.0000273149555816`
- `scoring_envelope_fraction`: `0.003088606764739793`
- `terminal_criterion_met`: `false`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `current_hardware_execution`: `false`

## RECOMMENDATION
KEEP

## experiment_8177_v706_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
