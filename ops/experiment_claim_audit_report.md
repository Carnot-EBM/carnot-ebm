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
| CLAIM_SUPPORTED | 7 |
| NO_CLAIM | 1 |

## experiment_8139_fit_evidence_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No empirical or comparative claim is made to refute; the artifact is an execution receipt documenting that an experiment was blocked prior to running.

## WAS THAT CHECKED
No. No experimental data or comparisons were collected because execution was halted at the pre-gate check layer.

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
`gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp8137-source-protocol.source_protocol_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8143_v704_delayed_energy_memory.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The execution completed a conforming delayed-energy memory learning trajectory under causal persistent constraints while correctly establishing an honest null result for learning benefit.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if:
1. Crash recovery or persistence failed, showing state divergence or non-identical predictions across crash/resume cycles.
2. Causal constraints were violated through feedback queue overflow, unreleased updates, or lost feedback rows.
3. The artifact asserted a positive benefit verdict (`complete_positive` or a non-zero benefit score) despite comparator arms tying.
4. Conformance or required gate checks failed.

## WAS THAT CHECKED
Yes. Persistence and crash resumption were tested via real crash and fresh resume validation commands, verifying identical predictions and durable state. Causal feedback queues and deadlines were tracked across slots with zero lost feedback. Comparative arms (`error_center`, `fixed_public_center`, `random_past_center`, and `frozen_qwen_offset`) were evaluated and shown to tie, which directly produced and backed the null verdict and zero benefit score.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_trajectory_complete_audit_pending"`
- `"claim_scope"`: `"Causal persistent constraints on exposed development; statistical benefit reserved for the independent audit"`
- `"generalized_learning_benefit_score"`: `0`
- `"independent_generalization_score"`: `0`
- `"learning_trajectory_ready_score"`: `1`
- `"required_checks_passed"`: `true`
- `"passed"`: `true`
- `"predictions_and_state_identical"`: `true`
- `"uninterrupted_state_hash"`: `"sha256:f4710daca491af452da933887e85df9c6d50088f7836f53d10118de7af17a8fa"`
- `"lost_feedback_rows"`: `[]`
- `"failed_count"`: `0`
- `"later_stream/error_center/brier"`: `0.1698573670891582`
- `"later_stream/fixed_public_center/brier"`: `0.1698573670891582`
- `"later_stream/frozen_qwen_offset/brier"`: `0.1698573670891582`
- `"later_stream/random_past_center/brier"`: `0.1698573670891582`

## RECOMMENDATION
KEEP

## experiment_8144_v704_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Adaptive error-center updates provide no later typed-cost benefit over the fixed-public-center baseline on later stream slots, establishing a complete null result (`complete_null_no_later_typed_cost_benefit`).

## WHAT WOULD REFUTE IT
A statistically credible typed-cost reduction for `error_center` over `fixed_public_center` meeting the pre-registered acceptance gate (`lower_bound` >= `0.02` with `improved_sources` >= `5`, resulting in `h2_passed` being `true`), or zero available headroom (`available_typed_cost_headroom` = 0) preventing detection of a real difference.

## WAS THAT CHECKED
Yes. It was evaluated in `audit_statistics` and `per_source_results` across 158 completed sources with 10,000 bootstrap draws across block lengths 8, 16, and 32. Non-zero headroom was verified (`available_typed_cost_headroom` is `0.6265822784810127`), and `positive_control` confirmed that signal detection was functional (`gain_lower_bound` of `0.5`).

## EVIDENCE
- `"honest_verdict"`: `"complete_null_no_later_typed_cost_benefit"`
- `"verdict_class"`: `"null"`
- `"h2_passed"`: `false`
- `"available_typed_cost_headroom"`: `0.6265822784810127`
- `"improved_sources"`: `0`
- `"lower_bound"`: `0.0`
- `"mean_gain"`: `0.0`
- `"other_control_cost_advantage"`: `0.0`
- `"completed_count"`: `158`
- `"support_sufficient"`: `true`

## RECOMMENDATION
KEEP

## experiment_8145_v704_natural_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Natural host service and update costs yield a complete null verdict because measured natural rejected admission costs are unavailable and speedup acceptance gates are unmet.

## WHAT WOULD REFUTE IT
Observation of measured natural rejected admissions (`rejected` > 0 with valid measured `rejected_update_cost`) satisfying update gates, or contender execution times meeting speedup criteria (`speed_claim` true with `one_sided_95_lower` > 1 and `NFR-01` >= 10).

## WAS THAT CHECKED
Yes; checked in `natural_update_categories`, `natural_update_cost_ready_score`, `rejected_update_cost`, and across 90 paired comparisons in `paired_speed_intervals` and `reduction`.

## EVIDENCE
`"honest_verdict"`
`"complete_null_natural_host_cost_measured_rejection_unavailable"`
`"verdict_class"`
`"null"`
`"natural_update_cost_ready_score"`
`0`
`"rejected_update_cost"`
`null`
`"accepted"`
`60`
`"rejected"`
`0`
`"speed_claim"`
`false`
`"nfr01_met"`
`false`
`"completed_count"`
`900`

## RECOMMENDATION
KEEP

## experiment_8146_v704_live_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native scalar execution yields no statistically significant speedup over python scalar execution under live LLM inference service costs, resulting in a bounded complete null.

## WHAT WOULD REFUTE IT
A paired speed ratio bootstrap 95% lower bound meeting or exceeding the acceptance threshold of 1.0 (`lower95` >= 1), demonstrating statistically significant speedup of `native_scalar` over `python_scalar` with `speed_supported` evaluating to `true`.

## WAS THAT CHECKED
Yes; checked across 23 matched pairs of live LLM inference requests in `request_pair_rows` and analyzed via 10,000 bootstrap resamples in `paired_speed_interval`.

## EVIDENCE
- `honest_verdict`: `complete_null_bounded_live_service_cost`
- `claim_scope`: `bounded retention-source complete miss latency; no correctness claim`
- `estimate`: `1.0004645876184797`
- `lower95`: `0.9611681420737964`
- `lower95_speed`: `1`
- `speed_supported`: `false`
- `nfr01_met`: `false`
- `sources`: `23`
- `matched_pair_count`: `23`
- `complete_service_ready_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8147_v704_arc_reader_renewal.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No new live supervisor outcomes or level solves occurred beyond the authenticated historical frontier, resulting in an honest complete null verdict while reader qualification and conformance checks passed.

## WHAT WOULD REFUTE IT
Any authenticated new outcome or solve appearing in the current run (`new_outcome_count > 0` or `new_level_solves_claimed > 0`), failure of the reader on positive control fixtures (`passed: false` or inability to accept supported progress, which would indicate false-negative risk), or corrupted upstream boundary hashes in precondition checks.

## WAS THAT CHECKED
Yes. Reader sensitivity and headroom were verified in `reader_conformance_rows` (where condition `supported` showed `expected`: 1, `observed`: 1, `passed`: true) and `positive_control_results` (`genuine_headroom`: true, `supported_progress_accepted`: true); upstream inputs were verified in `preconditions_checked` (all passed); and zero new outcomes were confirmed across the current run (`new_outcome_count`: 0, `new_level_solves_claimed`: 0, `no_new_outcomes`: true).

## EVIDENCE
- `"honest_verdict": "complete_null_no_new_outcomes"`
- `"no_new_outcomes": true`
- `"causal_benefit_claimed": false`
- `"new_outcome_count": 0`
- `"new_level_solves_claimed": 0`
- `"new_firing_count": 0`
- `"new_helped_count": 0`
- `"condition": "supported"`
- `"expected": 1`
- `"observed": 1`
- `"passed": true`
- `"reader_accepts_supported_progress": true`
- `"supported_progress_accepted": true`
- `"genuine_headroom": true`
- `"inference_substrate": "aggregation_from_upstream_artifacts"`
- `"inference_substrate_class": "no_model_load"`

## RECOMMENDATION
KEEP

## experiment_8148_v704_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Host workload overhead bounds potential arithmetic acceleration to an outer ceiling under 1.05x with no measured device-side acceleration, blocking whole-service hardware acceleration without a redesign of the workload.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. Latency measurements in `natural_workload_rows` or `amdahl_bounds` showed non-arithmetic retained components (`decision_serialization_fsync_ns`, `hash_lookup_feature_preparation_ns`, `lifecycle_ns`) accounting for a small enough fraction of `total_ns` to permit substantial arithmetic speedup (e.g., an arithmetic fraction ≥ 0.99 or an `outer_ceiling` reaching the 100x redesign threshold).
2. Any entry in `board_rows` demonstrated active hardware execution (`current_hardware_execution` true) with verified whole-service latency reduction satisfying its terminal qualification (`terminal_criterion_met` true).
3. The upstream service readiness gate passed (`complete_service` true in `acceptance_gates`) alongside measured end-to-end device acceleration rather than blocking service qualification.

## WAS THAT CHECKED
Yes:
- In `natural_workload_rows` and `amdahl_bounds`, host execution latencies were decomposed across 4,173 completed rows covering four distinct arms (`native_batch`, `native_scalar`, `python_batch`, `python_scalar`), each over 930 units; retained non-arithmetic work consumed 95%–98% of runtime in every arm, empirically constraining outer speedup ceilings to between 1.018x and 1.049x.
- In `board_rows`, hardware continuity across three physical platforms (`KV260`, `PolarFire`, `GateMate`) was evaluated, confirming no active hardware execution (`current_hardware_execution: false`), no satisfied terminal criteria (`terminal_criterion_met: false`), and board acquisition deferred due to zero measured benefit.
- In `acceptance_gates` and `gate_check_summary`, upstream service readiness failed (`complete_service: false`), properly triggering a blocked honest verdict (`honest_verdict: complete_blocked_complete_service_ready_score`).

## EVIDENCE
- `claim_scope`: `Natural exposed host workload ceilings and historical board custody; no measured device acceleration.`
- `honest_verdict`: `complete_blocked_complete_service_ready_score`
- `verdict_class`: `blocked`
- `hardware_boundary_ready_score`: `1`
- `measured_workload_bound_ready_score`: `1`
- `complete_service`: `false`
- `completed_count`: `4173`
- `eligible_count`: `4173`
- `arm`: `native_batch`, `total_ns`: `26280033116`, `retained_ns`: `25048707582`, `outer_ceiling`: `1.0491572481322282`, `target_100x`: `whole_workload_redesign_before_board_port`, `units`: `930`
- `arm`: `native_scalar`, `total_ns`: `25233102604`, `retained_ns`: `24441053234`, `outer_ceiling`: `1.0324065154810178`, `target_100x`: `whole_workload_redesign_before_board_port`, `units`: `930`
- `arm`: `python_batch`, `total_ns`: `21874914260`, `retained_ns`: `21488376904`, `outer_ceiling`: `1.0179882062627097`, `target_100x`: `whole_workload_redesign_before_board_port`, `units`: `930`
- `arm`: `python_scalar`, `total_ns`: `24093567580`, `retained_ns`: `23183868780`, `outer_ceiling`: `1.0392384380981645`, `target_100x`: `whole_workload_redesign_before_board_port`, `units`: `930`
- `acquisition_relevance`: `defer: no measured board whole-service benefit`
- `current_hardware_execution`: `false`
- `terminal_criterion_met`: `false`
- `verifier_is_oracle`: `false`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8149_v704_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Candidate development reductions yield zero measurable gain over baseline controls (an honest null on H2 with H1 blocked) while required governance and contract custody checks pass.

## WHAT WOULD REFUTE IT
The claim would be refuted if the artifact's own evaluation showed statistically significant positive gain over control arms (`observed_gain` > 0.0, `beneficial_changed_sources` > 0, or `h2_passed` set to true), a safety violation (`safety_passed` set to false), an absence of cost headroom (`available_typed_cost_headroom` equal to 0.0), or if any mandatory governance gates (G1–G4 `pass` set to false) or authority contract checks failed.

## WAS THAT CHECKED
Yes. H2 evaluated 158 completed sources across balanced classes with 0.6266 cost headroom against four arms (`error_center`, `fixed_public_center`, `frozen_qwen_offset`, `random_past_center`) and 10,000 bootstrap draws, confirming zero gain and zero improved sources; H1 was evaluated and tracked as blocked; gates G1–G4 verified reproducibility, AUROC artifacts, and seeds; and 14 authority contract tasks each passed all 17 structural verification checks.

## EVIDENCE
`status`
`completed_null`
`h2_passed`
`false`
`observed_gain`
`0.0`
`beneficial_changed_sources`
`0`
`improved_sources`
`available_typed_cost_headroom`
`0.6265822784810127`
`completed_count`
`158`
`support_sufficient`
`true`
`safety_passed`
`blocked`
`generalized_learning`
`independent_generalization`
`exposure_scope`
`Previously exposed RAGTruth; oracle protocol controls give no natural credit`
`honest_verdict`
`Completed nulls and external blocks are terminal and recorded once.`

## RECOMMENDATION
KEEP
