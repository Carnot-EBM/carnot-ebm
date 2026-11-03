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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8755_v750_verifier_family_surface.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On these previously exposed worlds, targeted probing shows no detection gain over baseline, and the tested surface transformations preserve truth and correct-verifier decisions.

## WHAT WOULD REFUTE IT
Targeted probing meeting the declared transfer criterion—a new-family detection delta with lower95 above zero and no extra correct-verifier false rejections—would contradict the reported null. Any paired surface truth discrepancy or correct-verifier decision disagreement would refute the invariance result.

## WAS THAT CHECKED
Yes, within the reported scope.

1. Probe outcomes are compared with expected truth; the displayed reject-all variant fails. Comparative success requires a separate detection advantage. Both old- and new-family paired intervals instead report zero estimates and zero endpoints.
2. The verifier is the oracle, which prevents a positive added-value inference. The overall verdict is explicitly null.
3. Exposure is acknowledged, model generation calls are zero, and independent generalization is explicitly unclaimed.
4. The unchanged five-call baseline is the relevant inexpensive rival. It ties targeted probing; that supports the null.
5. Excluded and censored counts are zero. The displayed failed reject-all row measures an intentional defect, rather than invalidating the observation. The cited interval and surface fields exist; surface summaries report zero discrepancies.

This supports the observed null on this workload, without establishing equivalence on unseen worlds.

## EVIDENCE

- `honest_verdict`: `complete_null_v750_verifier_family_surface`
- `criterion`: `new-family detection delta lower95 > 0 and no extra correct-verifier false rejection`
- `paired_world_intervals`: `new`, `old`; both have `estimate`, `lower95`, and `upper95` equal to `0.0`
- `correct_control_extra_false_rejections`: `0`
- `surface_family_rows`: `truth_discrepancies` and `decision_disagreements` are `0`
- `independent_generalization`: `false`; `independent_generalization_score`: `0`
- `exposure_scope`: `Sixty constructed and previously exposed worlds; thirty original canonical evaluation clusters. Surface siblings, variants and ten timing repeats add no independent science units.`
- `generation_calls_completed`: `0`
- `excluded_count`: `0`; `censored_count`: `0`
- `variant`: `reject_all`; `passed`: `false`; `raw_numerator`: `1`; `raw_denominator`: `5`

## RECOMMENDATION
KEEP

## experiment_8756_v750_feature_reuse_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Feature reuse halves feature extractions while preserving baseline serialized decisions and digests on the original exposed 64-item panel.

## WHAT WOULD REFUTE IT
Reuse failing to reduce successful extractions from 128 to 64, producing a different decision or digest for the same input, or breaking required input and ownership contracts would refute qualification.

## WAS THAT CHECKED
Yes, within the stated engineering scope. Extraction counts distinguish expected from successful operations; parity rows check outputs; input and ownership checks report success. The failure control also produces a zero readiness score.

The baseline’s output tie is the required preservation result. Oracle status supplies no independent correctness evidence, but this claim concerns extraction counts and execution parity. No speed or generalization benefit is claimed. The unsuccessful full-suite diagnostic is explicitly outside qualification dependencies.

## EVIDENCE

- `exposure_scope`: `original exposed64 panel; engineering parity only`
- `feature_extraction_count_rows`: `baseline`, `successful_extractions`, `128`; `reuse`, `successful_extractions`, `64`; `intended_items`, `64`
- `wire_parity_rows`: `passed`, `true`
- `owned_failure_control`: `feature_reuse_ready_score`, `0`; `actual_exit`, `1`; `passed`, `true`
- `verifier_is_oracle`: `true`
- `throughput_benefit_score`: `0`; `independent_generalization_score`: `0`
- `repository_health_is_dependency`: `false`; `full_suite_pass`: `false`

## RECOMMENDATION
KEEP

## experiment_8757_v750_feature_reuse_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Feature reuse improves complete-process throughput on the exposed workload while meeting the registered latency limits.

## WHAT WOULD REFUTE IT
Reuse matching or underperforming the same-provider baseline would refute the throughput benefit. A throughput-ratio lower confidence bound at or below 1.05 would fail the registered promotion criterion. Changed outputs, p95 above 50 ms, or p95 regression above 5% would defeat the qualified claim.

## WAS THAT CHECKED
Yes. Equal-work process rows measure elapsed time independently of the reuse setting; reuse could have incurred enough overhead to lose. Ten paired repeats yield a native throughput ratio of approximately 1.517, with a lower bound of 1.502. Native p95 decreases, and the shown native baseline/reuse output hashes match.

The verifier is explicitly not the oracle. The workload is explicitly exposed, and the artifact assigns zero generalization credit.

The cheapest serious baseline is the same implementation without reuse; both native and Python comparisons are present. Optimized Python approximately ties optimized native, and the separate 10× promotion gate fails. That tie does not undermine the within-provider reuse benefit.

The shown measurement rows pass and contain the cited timing fields. The failed historical full-suite receipt concerns repository health, which is explicitly excluded as a dependency. Most individual process rows are elided, so their validity cannot be inspected individually here.

## EVIDENCE

- `honest_verdict`: `complete_positive_feature_reuse_cost`
- `exposure_scope`: `original64 exposed requests; process timing only`
- `native_reuse`: `estimate`: `1.5165077375588447`; `lower95`: `1.5016786270448874`
- `native_reuse_throughput_lower95_min_exclusive`: `1.05`
- `native_steady_p95_ms`: `baseline`: `18.80990305`; `reuse`: `9.945570149999998`
- `python_reuse`: `lower95`: `1.5081887979010027`
- `optimized_native_python`: `lower95`: `0.9935943114642525`; `upper95`: `1.010817068844449`
- `same_batch_10x_score`: `0`
- `verifier_is_oracle`: `false`
- `independent_generalization_score`: `0`
- `repository_health_is_dependency`: `false`

## RECOMMENDATION
KEEP

## experiment_8758_v750_lifecycle_event_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8759_ordinary_durable_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt with no comparative or performance conclusion.

## WAS THAT CHECKED
No experiment result was tested. The prerequisite was checked in `gates_evaluated` and failed: observed 0 versus required 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `ordinary_window_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8760_v750_arc_horizon_sensitivity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This censored panel identifies no intervention benefit and supports no policy change.

## WHAT WOULD REFUTE IT
An admissible comparison with completed outcomes and cross-game support identifying a positive benefit, or an independent reconstruction showing that usable outcomes were incorrectly classified as missing.

## WAS THAT CHECKED
Yes. The arm summaries check completion, overlap rows check comparison admissibility, and independent reductions reconstruct outcome counts and bounds. The reported outcomes remain unknown, comparisons fail admission, and shown reconstructions pass. This supports non-identification; it does not establish zero effect.

## EVIDENCE

- `honest_verdict`: `complete_null_arc_horizon_sensitivity`
- `policy_decision`: `no_supported_policy_change`
- `completed_count`: `0`; `censored_count`: `16`
- `complete_case_mean`: `null`
- `admission_passed`: `false`; `identifiable_benefit`: `false`; `generalization_claim`: `false`
- `usable_cross_game_support`: `0`
- `intervals_reconstructed_from_frozen_rows`: `true`; `unknown_count`: `16`; `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8761_v750_hardware_cost_boundaries.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An authenticated, workload-compatible device execution with precision/quality parity and complete host, transfer, durability, and recovery costs would contradict the administrative conclusion that hardware qualification remains blocked.

## WAS THAT CHECKED
Yes, as an evidence inventory: gate_check_summary records failed prerequisites, while board_status_rows and operation_support_rows document execution and compatibility status. No comparative hardware experiment was performed or claimed. Oracle-defined readiness is not presented as verifier value, and imported CPU timings are not promoted to hardware acceleration or generalization.

## EVIDENCE

- `honest_verdict`: `complete_blocked_hardware_cost_boundaries`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `physical_hardware_ready_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `interpretation`: `Conditional1/(1-f); neither observed speed nor a100x complete-learning result`

## RECOMMENDATION
KEEP

## experiment_8762_v750_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V750 remains blocked, with limited batching benefit insufficient to establish broader readiness.

## WHAT WOULD REFUTE IT
Qualified evidence satisfying all mandatory acceptance gates would contradict the blocked decision. The batching subclaim would fail if its paired lower confidence bound were ≤1.05 or native B16 p95 exceeded 50 ms.

## WAS THAT CHECKED
Yes, at the aggregation level. Gate rows distinguish observed failures from missing evidence, and disqualified producers retain failed qualification status. The batching interval and latency satisfy their stated thresholds. The near-tie against Python at B16 is reported, while the stronger same-batch 10× claim receives zero. Independent generalization is explicitly unmeasured.

These rows support an evidence-readiness decision; they do not establish that the blocked capabilities are impossible.

## EVIDENCE
- `activation_authorized`: `false`
- `future_loss_and_retention`: `false`; `independent_semantics`: `false`
- `operand_state`: `observed_zero`, `missing_file`; `producer_qualified`: `false`
- `generalization_measured`: `false`
- `batching_threshold`: `paired lower95 > 1.05 and native B16 p95 <= 50ms`
- `lower95`: `1.1218473605571395`; `native_B16_steady_p95_ms`: `18.42018975`
- `contrast`: `native_python_B16`; `estimate`: `1.0053949467902554`
- `same_batch_10x_score`: `0`

## RECOMMENDATION
KEEP
