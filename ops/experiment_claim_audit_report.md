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
| CLAIM_SUPPORTED | 5 |
| CLAIM_OVERSTATED | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_8210_v709_restricted_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The restricted energy decision policy fails to achieve the registered H1 cost gain over baseline controls, resulting in a disqualified decision audit.

## WHAT WOULD REFUTE IT
Observing a lower cost gain exceeding threshold (`lower_gain > 0.02`), at least 5 improved sources (`improved_sources >= 5`), and a passing repository health suite (`repository_health.passed: true`), which would produce `H1.passed: true`, `required_checks_passed: true`, and a positive verdict rather than disqualification.

## WAS THAT CHECKED
Yes. Evaluated in `H1.operands` (where `lower_gain` observed `-0.01171875` vs `> 0.02` and `improved_sources` observed `3` vs `>= 5`), `all_slot_metrics` (where `energy` tied `additive` at `0.359375` and trailed `original_frozen_v707_radial` at `0.3515625`), and `repository_health` (where execution timed out after 180s).

## EVIDENCE
- `"honest_verdict": "complete_disqualified_restricted_decision_audit"`
- `"verdict_class": "disqualified"`
- `"passed": false`
- `"failed_conditions": [\n   "lower_gain",\n   "improved_sources"\n  ]`
- `"lower_gain": {\n    "expected": 0.02,\n    "observed": -0.01171875,\n    "op": ">",\n    "passed": false\n   }`
- `"improved_sources": {\n    "expected": 5,\n    "observed": 3,\n    "op": ">=",\n    "passed": false\n   }`
- `"required_checks_passed": false`
- `"scientific_H1": false`
- `"timed_out": true`
- `"claim_scope": "Exposed development source utility only; qualified negative evidence remains available."`

## RECOMMENDATION
KEEP

## experiment_8211_v709_calibrated_memory_trajectory.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The calibrated-memory trajectory establishes an honest null for causal learning benefit while confirming trajectory qualification, reserving any benefit claim for independent audit.

## WHAT WOULD REFUTE IT
The claim of an honest null would be refuted if the artifact's own rows and reductions demonstrated that the calibrated memory arms achieved a clear and statistically meaningful advantage over the baseline comparator (`calibration_only`), or if the artifact asserted a positive benefit score or non-null verdict (`generalized_learning_benefit_score` > 0 or `verdict_class` != `null`) without comparative superiority. Additionally, the readiness claim would be refuted by unhandled execution failures, such as exit code mismatches or state parity discrepancies upon recovery from hard exit 73.

## WAS THAT CHECKED
Yes. Comparative evaluation across arms was checked in `reductions` and `rows` over 158 later-stream and 61 retention independent sources. The simple `calibration_only` baseline was evaluated alongside the calibrated memory arms and achieved superior retention performance (lower Brier score and lower typed cost), confirming the absence of memory-added causal benefit and directly supporting the null verdict. Trajectory execution readiness and recovery parity were also explicitly verified in `child_exit_rows` and `restart_state_hashes`.

## EVIDENCE
- `honest_verdict`: `complete_null_causal_trajectory_benefit_reserved_for_8212`
- `verdict_class`: `null`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `claim_scope`: `Natural causal calibrated-memory trajectory only; benefit requires the separate frozen audit.`
- `benefit`: `reserved for independent Exp8212 audit; readiness does not imply benefit`
- `learning_trajectory_ready_score`: `1`
- `retention/calibrated_fixed_center/brier`: `0.14411130138918005`
- `retention/calibration_only/brier`: `0.143321984290345`
- `retention/calibrated_fixed_center/typed_cost`: `0.4098360655737705`
- `retention/calibration_only/typed_cost`: `0.4016393442622951`
- `later_stream/calibrated_fixed_center/brier`: `0.15467268699121256`
- `later_stream/calibration_only/brier`: `0.1550793639659329`
- `recovery`: `genuine exit73, complete state and issued-row parity`
- `required_checks_passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8212_v709_memory_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Calibrated error-center memory provides no independent benefit over the fixed-public-center baseline on later stream slots, yielding an honest null audit.

## WHAT WOULD REFUTE IT
A statistically significant positive gain from error-center memory over fixed-public-center that satisfied the pre-specified acceptance criteria: `h2_passed` evaluating to true, `mean_gain` > 0 with `lower_bound` >= 0.02, `improved_sources` >= 5, `extra_false_accepts` <= 0, and `brier_increase` <= 0.01.

## WAS THAT CHECKED
Yes; checked in `H2` across 158 completed independent sources with 10,000 moving original-slot bootstrap draws against pre-registered `acceptance_gates`, with sufficient headroom (`available_typed_cost_headroom` = 0.5126582278481012) and sample support (`support_sufficient` = true).

## EVIDENCE
`"honest_verdict": "complete_null_independent_memory_benefit_audit"`
`"verdict_class": "null"`
`"h2_passed": false`
`"improved_sources": 0`
`"mean_gain": -0.03164556962025317`
`"lower_bound": -0.090625`
`"extra_false_accepts": 0.006329113924050633`
`"brier_increase": 0.00025206210654502974`
`"available_typed_cost_headroom": 0.5126582278481012`
`"completed_count": 158`
`"support_sufficient": true`
`"decision_benefit_claim": false`
`"generalized_learning_benefit_score": 0`
`"independent_generalization_score": 0`

## RECOMMENDATION
KEEP

## experiment_8213_v709_prospective_request_recorder.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The prospective research request recorder is qualified and ready for prospective request capture with a readiness score of 1.

## WHAT WOULD REFUTE IT
Refutation would occur if request serialization, envelope validation, or durable event logging encountered failures or state corruption during live model transport (yielding non-zero `failed_count` or schema rejections), or if verified against an independent external ground truth rather than the internal oracle.

## WAS THAT CHECKED
No. Refutation was never given an opportunity to occur. No live model generation calls or model loads were attempted. Because testing was conducted strictly against scripted HTTP fixtures where the verifier was the oracle defining correctness, the qualification outcome was true by construction for the scripted boundary.

## EVIDENCE
- `honest_verdict`: `"complete_circular_positive_prospective_recorder_qualified"`
- `verdict_class`: `"circular_positive"`
- `verifier_is_oracle`: `true`
- `request_recorder_ready_score`: `1`
- `qualification_scope`: `"schedule envelope preparation and scripted recorder boundary; no prospective Qwen call"`
- `generation_calls_attempted`: `0`
- `model_loads_attempted`: `0`
- `inference_substrate_class`: `"no_model_load"`
- `benchmark_performed`: `false`
- `model`: `"scripted_http_fixture"`
- `verdict`: `"Owned failures disqualify; external missing operands block; oracle success is circular."`
- `methodology`: `"Fixed source identity order; full original condition; one sentence per source; max_tokens128; one future transport attempt. Current qualification uses a scripted HTTP peer, durable issue and terminal events, real Python/Rust signatures and disjoint spans. Oracle fixtures certify plumbing only."`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8214_v709_prospective_service_measurement.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8215_v709_arc_authoritative_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8216_v709_hardware_workload_obligations.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration obligations cannot achieve the target 100x speedup or deliver whole-service acceleration because the research workload is over 99.98% serial (dominated by acquisition and model startup) and physical board execution requirements remain unfulfilled.

## WHAT WOULD REFUTE IT
An empirical measurement showing that parallelizable scoring time (`removable_scoring_ns`) dominates total service runtime over serial acquisition and startup overhead, yielding an Amdahl serial fraction `<= 0.01` and a maximum theoretical speedup `>= 100`; or an authenticated device execution transcript showing satisfied terminal criteria and verified whole-service speedup on a physical board (KV260, PolarFire, or GateMate).

## WAS THAT CHECKED
Yes. Checked in `amdahl_bounds` across `native_atomic_batch` and `python_durable_batch` using measured stage timings from `workload_rows`, in `board_rows` across historical FPGA targets (KV260, PolarFire, GateMate), and in `access_obligations` for NPU and TSU hardware.

## EVIDENCE
- `honest_verdict`: `complete_null_hardware_workload_obligations`
- `verdict_class`: `null`
- `target_speedup`: `100`
- `supports_100x`: `false`
- `maximum_speedup`: `1.0001599271488204`
- `serial_fraction`: `0.9998400984237827`
- `removable_scoring_ns`: `12888571`
- `acquisition_ns`: `48574070284`
- `model_startup_ns`: `41413713642.88423`
- `total_ns`: `94673400713.88423`
- `claim_scope`: `Software reducer and host primitive limits; no current model or board execution`
- `terminal_criterion_met`: `false`
- `status`: `blocked_authenticated_access`
- `current_device_execution_count`: `0`
- `current_model_calls`: `0`

## RECOMMENDATION
KEEP

## experiment_8217_v709_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Neither the energy increment policy (H1) nor structural memory (H2) achieves statistically significant cost or decision gains over baseline comparators, resulting in an honest null outcome and disqualified owned validation.

## WHAT WOULD REFUTE IT
The headline claim would be refuted by observed statistically significant positive gains exceeding the pre-specified protocol thresholds in the resampled evaluation data—specifically, an H1 bootstrap lower bound `lower_gain` > 0.02 with `improved_sources` >= 5, and an H2 paired bootstrap gain interval with a lower bound > 0 and `improved_sources` > 0, leading to passing hypothesis conditions and passing validation gates.

## WAS THAT CHECKED
Yes. Statistical tests were executed with 10,000 bootstrap draws over 97 completed source clusters for H1 in `statistics.H1` and across 158 completed sources in `statistics.paired_gain_interval` for H2; the tests failed to demonstrate positive gains, confirming the null result.

## EVIDENCE
`"status": "completed_null"`
`"passed": false`
`"failed_conditions"`
`"lower_gain"`
`"improved_sources"`
`-0.01171875`
`3`
`"h2_passed": false`
`0`
`-0.03164556962025317`
`"decision_benefit_claim": false`
`"population_safety_claim": false`
`"disqualified"`
`"owned_validation"`

## RECOMMENDATION
KEEP
