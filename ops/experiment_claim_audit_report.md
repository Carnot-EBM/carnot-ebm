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
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_8183_v707_sentence_energy_fit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Calibrated sentence energy fitting produces a complete null result with zero independent generalization and no demonstrated learning benefit over comparator or ablation baselines.

## WHAT WOULD REFUTE IT
An observation of a strictly positive generalization score (`independent_generalization_score > 0` or `generalized_learning_benefit_score > 0`), or out-of-sample performance where `local_evidence_radial16` achieves a lower typed decision cost than the `local_feature_ablation` control on held-out evaluation clusters.

## WAS THAT CHECKED
Yes. Performance was evaluated across cross-validation folds on held-out clusters (`held_source_ids` in `fit_fold_rows`), benchmarked against baseline arms (`local_max` and `local_feature_ablation` in `frozen_thresholds` and `calibration_receipts`), and both generalization metrics were evaluated and recorded as `0`, with the ablation baseline tying the candidate model at a `typed_cost` of `0.265625`.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_sentence_energy_fit"`
- `"verdict_class"`: `"null"`
- `"independent_generalization_score"`: `0`
- `"generalized_learning_benefit_score"`: `0`
- `"claim_scope"`: `"sealed calibrated source energy decisions; exposed fit/tune only; reserved H1 untested"`
- `"comparator_id"`: `"radial16"`
- `"arm"`: `"local_evidence_radial16"`
- `"arm"`: `"local_feature_ablation"`
- `"typed_cost"`: `0.265625`
- `"verifier_is_oracle"`: `false`

## RECOMMENDATION
KEEP

## experiment_8184_v707_reserved_sentence_capture.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8185_v707_sentence_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The candidate sentence decision policy `local_evidence_radial16` fails the acceptance gates against baseline `radial16`, yielding a complete null decision audit verdict (`complete_null_sentence_decision_null`).

## WHAT WOULD REFUTE IT
The null claim would be refuted by observed data where `local_evidence_radial16` achieved a strictly positive lower-bound cost gain exceeding `0.02` at the 97.5% one-sided confidence level while incurring `0` extra false accepts relative to `radial16`, thereby satisfying all acceptance gates and yielding `safety_passed` as true.

## WAS THAT CHECKED
Yes. It was checked across all 128 intended slots (and 97 complete cases) evaluated via 10,000 bootstrap draws in `acceptance_operands`, `all_slot_metrics`, `paired_intervals`, and `acceptance_gates`.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_sentence_decision_null"`
- `"verdict_class"`: `"null"`
- `"safety_passed"`: `false`
- `"extra_false_accepts"`: `2`
- `"maximum_extra_false_accepts"`: `0`
- `"lower_cost_gain"`: `-0.13671875`
- `"minimum_gain_exclusive"`: `0.02`
- `"lower_one_sided_975"`: `-0.13671875`
- `"mean_gain"`: `-0.01953125`
- `"cost"`: `0.37109375` (for `"arm"`: `"local_evidence_radial16"`)
- `"cost"`: `0.3515625` (for `"arm"`: `"radial16"`)
- `"primary_denominator"`: `"all_intended_slots"`

## RECOMMENDATION
KEEP

## experiment_8186_calibrated_online_memory.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any comparative experimental results, execution metrics, or downstream performance data within the artifact demonstrating that the experiment actually executed despite the gate failure.

## WAS THAT CHECKED
No; execution was halted at the pre-gate validation layer before any experimental condition or hypothesis could be evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8188_v707_exact_request_service.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8189_v707_arc_supervisor_frontier.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No new authenticated ARC supervisor outcomes, firings, or level solves were observed beyond the upstream frontier cutoff.

## WHAT WOULD REFUTE IT
The observation of any authenticated new supervisor event rows, firings, or level solves past the upstream cutoff timestamp (`2026-10-05T21:51:37.962846+00:00`), non-zero outcome reduction from the empty ledger control, or positive control failure indicating that the reduction reader is blind to genuine progress.

## WAS THAT CHECKED
Yes; the reader verified 156 upstream preconditions, executed negative controls in `historical_required_failures` confirming corrupt records fail, ran positive control fixtures in `reader_conformance_rows` confirming that genuine progress is accepted (`supported_progress_accepted: true`), evaluated `empty_ledger_control`, and scanned the upstream receipt frontier after the cutoff timestamp, detecting zero new outcomes (`new_outcome_count: 0`).

## EVIDENCE
- `"honest_verdict"`: `"complete_null_no_new_outcomes"`
- `"no_new_outcomes"`: `true`
- `"causal_benefit_claimed"`: `false`
- `"new_outcome_count"`: `0`
- `"new_firing_count"`: `0`
- `"new_level_solves_claimed"`: `0`
- `"new_solve_claim"`: `false`
- `"fixture_rejected"`: `true`
- `"supported_progress_accepted"`: `true`
- `"claim_scope"`: `"exposed_development_reader_conformance_and_observational_frontier"`

## RECOMMENDATION
KEEP

## experiment_8190_v707_hardware_service_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration provides no whole-workload service benefit over software execution (with Amdahl outer speedup ceilings bounded below 1.0022x across conditions), justifying the decision to defer hardware spending.

## WHAT WOULD REFUTE IT
A measured arithmetic fraction approaching the required 0.99 threshold, an Amdahl speedup ceiling substantially exceeding 1.0x toward the 100x target, or an authenticated board deployment exhibiting actual execution and whole-service latency reduction.

## WAS THAT CHECKED
Yes. In `amdahl_bounds` and `rows`, 288 completed requests across four conditions (`cold_miss`, `exact_repeat`, `forced_refresh`, `changed_source`) were decomposed into constituent latencies, demonstrating that non-accelerable overhead accounts for >99.7% of total runtime and bounding maximum potential speedup to at most 1.00213x; additionally, `board_rows` audited five hardware platforms and confirmed zero active, compatible execution.

## EVIDENCE
- `honest_verdict`: `complete_null_hardware_service_boundary`
- `hardware_spending_decision`: `defer: no measured whole-workload hardware benefit`
- `claim_scope`: `Independent cold, hit and invalidation software ceilings; historical board custody only`
- `target_speedup`: `100`
- `required_arithmetic_fraction_for_100x`: `0.99`
- `outer_ceiling`: `1.0002748073696213`
- `outer_ceiling`: `1.0021319989821222`
- `outer_ceiling`: `1.0002795251631347`
- `outer_ceiling`: `1.000283194465957`
- `retained_ns`: `43014021117`
- `total_ns`: `43025841687`
- `retained_ns`: `4897856935`
- `total_ns`: `4908299161`
- `retained_ns`: `39015768304`
- `total_ns`: `39026674193`
- `retained_ns`: `40295229504`
- `total_ns`: `40306640890`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `current_hardware_execution`: `false`
- `measured_bottleneck`: `Acquisition on misses/invalidation and durable storage/acknowledgement on cache hits.`

## RECOMMENDATION
KEEP

## experiment_8191_v707_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
