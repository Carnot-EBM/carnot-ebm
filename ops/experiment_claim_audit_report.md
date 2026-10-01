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
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 2 |

## experiment_7968_v691_response_role_targets.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7969_v691_qwen_calibration_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A comparative claim asserting calibration or decision benefit would be refuted by observed Brier score degradation, zero cost reduction against an uncalibrated or source-erased baseline, or failure to separate distinct risk strata. However, the artifact lacks any comparative performance hypothesis or evaluative claim, explicitly stating that capture does not establish science benefit and that evaluation calls are reserved for later analysis.

## WAS THAT CHECKED
No. The artifact lacks evaluation checks; `current_evaluation_call_count` is `0`, `paired_family_rows` is empty, and `comparison_status` is `insufficient_data`.

## EVIDENCE
`honest_verdict`: `complete_null_qwen_calibration_capture`
`comparison_status`: `insufficient_data`
`current_evaluation_call_count`: `0`
`Zero: existing evaluation responses are reserved for later analysis.`
`Bind exact producer identity, public protocol, owned work or measured validity; capture does not establish science benefit.`
`brier_gain`: `null`
`cost_gain`: `null`
`calibration`: `null`
`decision_benefit`: `null`
`efficiency`: `null`
`retention`: `null`
`paired_family_rows`: `[]`

## RECOMMENDATION
KEEP

## experiment_7970_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a pre-execution gate receipt asserting no empirical, comparative, or performance claim, there is no substantive headline claim to refute. Observing training execution outputs, fitted energy parameters, or a claim of successful execution despite failed upstream qualifications would contradict the artifact's blocked status.

## WAS THAT CHECKED
No. The fitting procedure was never executed; execution was halted at the pre-gate check layer prior to running any experiment.

## EVIDENCE
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

## experiment_7972_v691_qwen_energy_calibration.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Gibbs energy calibration provides no statistically significant decision benefit or Brier score improvement over baseline calibration controls, resulting in an honest complete null verdict.

## WHAT WOULD REFUTE IT
Statistically significant positive gain in decision cost and Brier score by the Gibbs arm over all three baseline controls (`platt`, `isotonic`, and `raw_qwen`) on evaluation data, yielding Holm-adjusted p-values below threshold (adjusted p < 0.05) and `decision_benefit` evaluating to true.

## WAS THAT CHECKED
Yes. Evaluated on independent held-out evaluation data (`role_label_access_events` sequence 3, `evaluation_support` with 62 independent clusters) comparing Gibbs against `platt`, `isotonic`, and `raw_qwen` across Brier score and decision cost. A synthetic positive control (`positive_control_rows`) confirmed that the evaluation pipeline had statistical power and was capable of detecting positive gain (`detected`: true).

## EVIDENCE
- `"honest_verdict": "complete_null_qwen_energy_calibration"`
- `"verdict_class": "null"`
- `"qwen_calibration_benefit_score": 0`
- `"decision_benefit": false`
- `"adjusted_p_values": { "isotonic_brier": 1.0, "isotonic_cost": 1.0, "platt_brier": 1.0, "platt_cost": 1.0, "raw_qwen_brier": 0.80991900809919, "raw_qwen_cost": 0.80991900809919 }`
- `"gain": -0.01334683485130828`
- `"gain": -0.0029893479577797486`
- `"gain": 0.0`
- `"detected": true`

## RECOMMENDATION
KEEP

## experiment_7975_v691_arc_supervisor_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this is a scaffolding and receipt artifact recording an observational inventory with disqualified prerequisites, no comparative claim is asserted to refute. If the artifact had claimed positive intervention impact or valid supervisor progress, that would be refuted by observations of zero supervisor activity, empty execution rows, or failed prerequisite validation gates.

## WAS THAT CHECKED
No. The artifact executed validation gates and upstream integrity checks (which failed), but conducted no experimental arm comparisons and ran no model trials.

## EVIDENCE
- `honest_verdict`: `"complete_disqualified_supervisor_delta_prerequisites"`
- `claim_scope`: `"exposed_development; observational inventory without independent or causal benefit"`
- `inference_substrate`: `"aggregation_from_upstream_artifacts"`
- `inference_substrate_class`: `"no_model_load"`
- `acceptance_gate_results`: `"validity": false`, `"readiness": 0`, `"decision_benefit": null`
- `arm_outcomes`: `{}`
- `rows`: `[]`
- `new_firing_count`: `0`
- `new_helped_count`: `0`
- `new_level_solves_claimed`: `0`
- `solve_provenance`: `"Only authenticated input events have live discovery provenance; aggregation claims no solves."`

## RECOMMENDATION
KEEP

## experiment_7976_v691_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The service evaluation demonstrates a complete null service cost outcome (`complete_null_service_cost`) with no fresh inference speedup over the reference CPU execution baseline.

## WHAT WOULD REFUTE IT
A statistically meaningful latency or throughput advantage of `exclusive` mode over `reference` mode, a non-zero `compatible_fraction` allowing hardware offload, an `ideal_amdahl_bound` greater than 1.0, or a true `fresh_inference_speedup_claim`.

## WAS THAT CHECKED
Yes. In `rows` (evaluating 640 timing requests across 64 independent source clusters for `exclusive` vs `reference` mode), `cached_incremental_cost`, `exclusive_phase_spans`, and `compatible_fraction`.

## EVIDENCE
- `honest_verdict`: `complete_null_service_cost`
- `verdict_class`: `null`
- `efficiency`: `descriptive_only`
- `fresh_inference_speedup_claim`: `false`
- `compatible_fraction`: `0.0`
- `ideal_amdahl_bound`: `1.0`
- `mode`: `exclusive`, `p50_s`: `0.000114555`, `throughput_requests_s`: `8769.493213179583`, `timing_requests`: `640`
- `mode`: `reference`, `p50_s`: `0.00011453`, `throughput_requests_s`: `8781.023889063617`, `timing_requests`: `640`
- `independent`: `64`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7977_v691_hardware_evidence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No hardware acceleration or whole-service board execution benefit is measured over host CPU execution, yielding an ideal Amdahl speedup bound of 1.0 and deferring hardware acquisition across evaluated boards.

## WHAT WOULD REFUTE IT
Any of the following appearing in the artifact's own rows or summaries would refute the headline claim:
- Any entry in `board_rows` reporting hardware execution or achieving its terminal criteria.
- A non-zero count of device executions.
- Any operation in `operation_map` evaluated as fabric-compatible with non-zero kernel share, producing a modeled kernel bound or Amdahl upper bound greater than 1.0.
- Non-zero compatible fraction or ideal Amdahl bound exceeding 1.0 in modeled acceleration bounds.
- Affirmative assertion of hardware speedup.

## WAS THAT CHECKED
Yes:
- Checked across hardware targets in `board_rows` (KV260, PolarFire, GateMate) evaluating execution status, physical blockers, and prerequisites.
- Checked in `operation_map` evaluating candidate service operations against board fabric constraints and computing Amdahl acceleration bounds.
- Checked in `current_device_execution_count` and `modeled_acceleration_bounds`.

## EVIDENCE
- `honest_verdict`: `complete_null_historical_board_scope_current_host_mapping`
- `hardware_speedup_claimed`: `false`
- `current_device_execution_count`: `0`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `current_hardware_execution`: `false`
- `terminal_criterion_met`: `false`
- `hardware_advantage`: `unmeasured`
- `compatible_fraction`: `0.0`
- `ideal_amdahl_bound`: `1.0`
- `modeled_100x_bound`: `1.0`
- `fabric_compatible`: `false`
- `kernel_share`: `0.0`
- `amdahl_estimate_upper_bound`: `1.0`
- `wishlist_decision`: `CPU timings cannot establish measured full-service accelerator benefit or justify a purchase.`

## RECOMMENDATION
KEEP

## experiment_7978_v691_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
