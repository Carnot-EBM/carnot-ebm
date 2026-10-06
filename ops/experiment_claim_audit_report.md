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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 3 |

## experiment_8196_v708_selective_sealed_evaluation.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8197_v708_selective_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The local set decision method fails to demonstrate selective utility or energy-specific benefit over the frozen radial baseline, resulting in a complete null verdict.

## WHAT WOULD REFUTE IT
A statistically significant positive cost gain over `frozen_v707_radial` satisfying the pre-registered acceptance gates (specifically, bootstrap one-sided 97.5% lower cost gain > 0.02, zero extra false accepts, and Brier increase <= 0.01), or any performance difference demonstrating an energy-specific advantage over `equivalent_logistic_set`.

## WAS THAT CHECKED
Yes. Checked under `H1`, `acceptance_operands`, `paired_intervals`, `complete_case_metrics`, and `equivalent_logistic_parity` across 128 slots (97 completed pairs) with 10,000 paired bootstrap resamples.

## EVIDENCE
- `"honest_verdict": "complete_null_selective_decision_null"`
- `"claim_scope": "All-slot selective utility on exposed development sources; no energy-specific or independent benefit"`
- `"attribution": "Any wrapper benefit is abstention or calibration; logistic equivalence rules out energy-specific evidence."`
- `"comparator": "frozen_v707_radial"`
- `"treatment": "local_set"`
- `"passed": false`
- `"lower_cost_gain": -0.234375`
- `"minimum_gain_exclusive": 0.02`
- `"extra_false_accepts": 2`
- `"maximum_extra_false_accepts": 0`
- `"mean_gain": -0.11328125`
- `"equivalent_logistic_parity"`
- `"passed": true`
- `"generalized_learning_benefit_score": 0`
- `"h1_development_signal_score": 0`

## RECOMMENDATION
KEEP

## experiment_8198_calibrated_online_memory.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact makes no empirical or comparative performance claim, there is no scientific hypothesis to refute; refutation of its execution state would require artifact records showing that upstream requirements passed (`passed` is true or `actual` equals `expected`) while marking the run as blocked, or execution proceeding past the gate.

## WAS THAT CHECKED
No empirical test was run because execution halted prior to execution; gate conditions were checked against the upstream artifact in `gates_evaluated` and both failed.

## EVIDENCE
`"schema"`
`"blocked_gate_check_v1"`
`"status"`
`"blocked"`
`"duration_s"`
`0.0`
`"honest_verdict"`
`"blocked_gate_check_failed"`
`"blocked_reason"`
`"actual=0 == expected=1"`
`"gate_check_summary"`
`"gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8193-learning-qualification.calibrated_memory_ready_score (actual=0 == expected=1)"`
`"blocked_at_layer"`
`"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_8200_v708_request_trace_census.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8201_observed_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no empirical or comparative claim, functioning strictly as an execution receipt recording a blocked pre-flight gate check.

## WAS THAT CHECKED
No; the experiment was blocked at the pre-gate phase and never executed.

## EVIDENCE
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"schema": "blocked_gate_check_v1"`
`"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_8202_v708_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a receipt inventory and frontier aggregation run that was blocked at an acceptance gate, it asserts no comparative, causal, or performance claims. If a positive claim regarding supervisor efficacy or level solves had been made, refutation would require observing zero improvement over an unredirected baseline across held-out games under live model execution.

## WAS THAT CHECKED
No. The run was halted at an acceptance gate check (`authenticate_live_state_is_file`), executed zero model invocations, conducted zero live game runs, and evaluated no competing arms.

## EVIDENCE
- `"honest_verdict"`: `"complete_blocked_authenticate_live_state_is_file"`
- `"causal_benefit_claimed"`: `false`
- `"new_solve_claim"`: `false`
- `"new_level_solves_claimed"`: `false`
- `"no_new_outcomes"`: `true`
- `"arm_outcomes"`: `{}`
- `"inference_substrate"`: `"aggregation_from_upstream_artifacts"`
- `"inference_substrate_class"`: `"no_model_load"`
- `"current_model_invocation_count"`: `0`
- `"current_game_runs"`: `0`
- `"scientific_benefit"`: `null`
- `"check"`: `"authenticate_live_state_is_file"`
- `"passed"`: `false`

## RECOMMENDATION
KEEP

## experiment_8203_v708_hardware_decision_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration purchase and integration should be deferred because Amdahl's law bounds complete-workload speedup to at most ~1.002x (failing the 100x threshold) while candidate hardware targets remain blocked or exhibit no measured whole-service benefit.

## WHAT WOULD REFUTE IT
An Amdahl bound showing an optimistic speedup ceiling of at least 100x (`supports_100x: true`, requiring the accelerated portion to constitute at least 99% of total service latency), or any candidate board meeting its terminal qualification criteria (`terminal_criterion_met: true`) with measured whole-service latency reduction over the CPU baseline.

## WAS THAT CHECKED
Yes. Amdahl speedup limits were evaluated in `amdahl_bounds` across 96 qualified runs spanning 4 operational conditions (`changed_source`, `cold_miss`, `exact_repeat`, `forced_refresh` detailed in `workload_rows`), and candidate hardware continuity was evaluated in `board_rows` across five platforms (`KV260`, `PolarFire`, `GateMate`, `NPU`, `TSU`).

## EVIDENCE
`hardware_spending_decision`: `"defer; no purchase authorized; no100x complete-workload evidence"`
`honest_verdict`: `"complete_blocked_exists"`
`verdict_class`: `"blocked"`
`required_arithmetic_fraction_for_100x`: `0.99`
`supports_100x`: `false`
`optimistic_ceiling`: `1.000283194465957`
`optimistic_ceiling`: `1.0002748073696213`
`optimistic_ceiling`: `1.0021319989821222`
`optimistic_ceiling`: `1.0002795251631347`
`required_redesign`: `"Reduce original acquisition and durable commit costs; kernel acceleration alone cannot deliver100x."`
`terminal_criterion_met`: `false`
`acquisition_relevance`: `"defer: no measured board whole-service benefit"`
`blocker`: `"0xffffffff"`
`status`: `"blocked_authenticated_access"`

## RECOMMENDATION
KEEP

## experiment_8204_v708_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
