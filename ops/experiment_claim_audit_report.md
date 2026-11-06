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
| CLAIM_SUPPORTED | 6 |
| NO_CLAIM | 2 |

## experiment_8858_v758_installed_indexed_verifier.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The installed wheel passes installation and ownership checks and preserves reference semantics and operation counts on the constructed request roster.

## WHAT WOULD REFUTE IT
An outside-checkout import failure, any installed/reference semantic or operation-count mismatch, or acceptance of stale or cross-query state would refute this qualification claim.

## WAS THAT CHECKED
Yes. Installed import checks, semantic and operation parity rows, and ownership checks report passes. Packaging or state-handling failures could have occurred even though the verifier defines correctness.

Oracle agreement establishes execution parity here; the artifact asserts no comparative verifier benefit. The failed repository-wide test receipt is explicitly scoped to global health and does not contradict the narrower qualification result.

## EVIDENCE

- `methodology`: `Build a private wheel; compare its outside-checkout public calls against all 3600 original requests and 14 pinned adversaries. No model, training, or timing benefit measurement.`
- `exact_installed_parity`: `true`; `installed_index_ready_score`: `1`.
- `installed_outside_checkout_no_pythonpath`; `semantic_parity_rows`; `operation_parity_rows`; `passed`: `true`.
- `stale_reuse_rejected`; `cross_query_rejected`; `passed`: `true`.
- `completed_count`: `3614`; `failed_count`: `0`; `verifier_is_oracle`: `true`.
- `repository_health`; `scope`: `global_health_only`; `passed`: `false`.

## RECOMMENDATION
KEEP

Execution was not independently checked, and elided rows were not inspected. These results support installation qualification only.

## experiment_8859_v758_split_certificate_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This evaluation did not demonstrate a split-certification benefit over simultaneous certification under the prespecified gates.

## WHAT WOULD REFUTE IT
A split-minus-simultaneous coverage gain with a 99% lower bound above 0.05, with useful rules achieving coverage at least 0.3 and census risk at most 0.05, would contradict the reported null.

## WAS THAT CHECKED
Yes. The protocol separates fitting, selection, certification, and subsequent independent evaluation. It compares equal calibration-label budgets across 60 paired replicates. The reported coverage contrast and both confidence bounds are zero: the simultaneous comparator ties.

Shown refusals contribute zero coverage, while empty risk denominators remain null. The oracle defines correctness, but the headline claims neither verifier value nor generalization. Zero certificate failures therefore need not establish safety to support this limited null.

## EVIDENCE
- `honest_verdict`: `complete_null_split_certificate_evaluation`; `verdict_class`: `null`.
- `primary_contrast`: `estimate`: `0.0`; `lower_99`: `0.0`; `upper_99`: `0.0`; `independent_replicates`: `60`.
- `split_protocol_benefit_score`: `0`; `method_superiority_score`: `0`; `independent_generalization_score`: `0`.
- `refusal_reason`: `no_valid_certificate`; `certificate_issued`: `false`; `served_risk`: `null`.
- `conditional_failure_intervals`: `denominator`: `0`; `estimate`: `null`.
- `verifier_is_oracle`: `true`.

## RECOMMENDATION
KEEP

Review limited to supplied text; elided rows were not independently checked. This null establishes no universal safety guarantee.

## experiment_8860_v758_linear_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The combined linear method did not establish causal benefit under the audit’s registered criteria.

## WHAT WOULD REFUTE IT
All registered causal-benefit gates passing, including a simultaneous upper bound below zero for combined-minus-shuffled loss, alongside the required future gain, addition-specific gain, integrity and retention checks.

## WAS THAT CHECKED
Yes, through the reported paired contrasts and gate scores. The shuffled comparison’s interval crosses zero, preventing a positive causal verdict despite other gains.

The cheapest serious rival, additions-only, is present: combined shows no demonstrated improvement over it. This supports the restrained verdict.

The verifier defines correctness, but the headline claims no verifier-added value. Prediction-before-feedback checks are reported, and independent generalization receives zero. The false validation receipt concerns repository health; it is not an outcome-validity flag. No wrong-key error is demonstrated.

## EVIDENCE
- `honest_verdict`: `complete_null_linear_benefit_audit`; `verdict_class`: `null`.
- `causal_benefit_score`: `0`; `true_label_specificity_score`: `0`; `independent_generalization_score`: `0`.
- `future_gain_score`: `1`; `addition_benefit_score`: `1`; `retention_pass_score`: `1`.
- `linear_combined_minus_linear_shuffled_combined`: `lower`: `-0.05751561162773932`; `upper`: `0.004353599730100326`.
- `linear_combined_minus_linear_additions_only`: `estimate`: `0.0002922020206789777`; `lower`: `-0.0155661859704758`; `upper`: `0.01786551296739734`.
- `verifier_is_oracle`: `true`; `prediction_before_feedback`: `true`.
- `scope`: `repository_health`; `passed`: `false`; `timed_out`: `true`.

## RECOMMENDATION
KEEP

Uninspected: elided rows and aggregation code; this verdict is limited to the supplied artifact.

## experiment_8861_v758_installed_index_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Indexed verification did not establish the declared installed-cost benefit over always_joint.

## WHAT WOULD REFUTE IT
Passing every benefit gate, including a size-16 latency-regression upper bound at most 5%, while establishing mixture speedup and preserving quality, would contradict the reported null.

## WAS THAT CHECKED
Yes. The paired comparison against always_joint provides a substantive baseline, and measured elapsed costs allow either arm to lose. The cold mixture speed-ratio lower bound exceeds one, but the size-16 regression upper bound is approximately 30.18%, failing the 5% requirement. That establishes failure to demonstrate the required benefit, not proof of a 30.18% regression.

Oracle-defined correctness does not predetermine elapsed cost. No model generalization benefit is claimed. Censored requests remain in denominators, with no exclusions reported.

## EVIDENCE

- `honest_verdict`: `complete_null_installed_index_cost`; `verdict_class`: `null`; `installed_cost_benefit_score`: `0`.
- `size16_latency_regression`: `threshold`: `0.05`; `upper`: `0.3017865229990757`.
- `size16_latency_regression_upper_at_most_five_percent`: `false`.
- Under `cold`, `mixture`, `speed_ratio`: `lower`: `1.063403991993194`.
- `always_joint`; `indexed_witness`; `Paired random arm order within each independently constructed world.`
- `All censors remain in denominators.`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

No rerun or external inspection; conclusions are limited to the supplied artifact.

## experiment_8862_v758_linear_durable_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The paired durable-cost run completed and measured service costs, without establishing scientific benefit from learning.

## WHAT WOULD REFUTE IT
A failed required window, missing paired measurement, or incomplete preparation accepted as complete would contradict this limited claim. The null must not be interpreted as proof that all arms have equal costs.

## WAS THAT CHECKED
Yes. The artifact reports all 60 intended windows completed, no failures or censoring, and no missing pairs in the six cost comparisons.

The comparisons could produce different outcomes: the combined arm’s timing interval excludes zero at occupancy 0 but crosses zero at occupancy 32. Frozen and calibration-only arms provide serious controls. These results support cost measurement, without establishing a learning benefit. Oracle-defined correctness and repeated prepared structures do not invalidate that limited conclusion.

## EVIDENCE
- `honest_verdict`: `complete_null_linear_durable_cost`
- `completed_count`: `60`; `intended_count`: `60`; `failed_count`: `0`; `censored_count`: `0`
- `paired_blocks`: `10`; `missing_pairs`: `0`
- `linear_combined`, occupancy `0`: `lower_ns`: `-5244562150.563442`; `upper_ns`: `-3326911059.0365577`
- `linear_combined`, occupancy `32`: `lower_ns`: `-3005945763.149974`; `upper_ns`: `1775536773.3499737`
- `independent_generalization_score`: `0`
- `Bind actual complete work to retained operands; cost readiness does not establish scientific benefit.`

## RECOMMENDATION
KEEP

Elided rows were not individually checked; row-level validity remains unconfirmed beyond the supplied summaries.

## experiment_8863_qwen_source_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No extraction-performance claim is asserted. This artifact records a prerequisite failure, so there is no substantive result to falsify.

## WAS THAT CHECKED
No extraction performance was checked. The prerequisite checks appear in `gates_evaluated`; runtime readiness failed, blocking execution.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8864_v758_hardware_budget.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The available evidence establishes no measured complete supported physical workload benefit, so hardware acquisition is deferred.

## WHAT WOULD REFUTE IT
A compatible board workload showing correct results and lower complete latency than the same CPU workload, including host, transfer, startup and durable costs.

## WAS THAT CHECKED
Yes, within the evidence audit: board-operation rows record compatibility and measured timing, and the physical-acceleration gate fails. The visible rows supply no contradicting measurement. This supports the limited evidence claim.

No new physical comparison was performed; actual hardware benefit remains unknown.

## EVIDENCE

- `honest_verdict`: `complete_null_hardware_budget`
- `reason`: `no measured complete supported physical workload benefit`
- `physical_acceleration`: `passed`: `false`
- `fabric_supported`: `false`
- `measured_board_time_s`: `null`
- `end_to_end_gain_bound`: `null`
- `physical_probes_performed`: `0`
- `count_scope`: `CPU inventory only; absent counts and component times are unknown`

## RECOMMENDATION
KEEP

## experiment_8865_v758_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. The administrative accounting assertion would fail if a scheduled slot lacked primary or skip custody; scientific benefit would require a separate, falsifiable comparison.

## WAS THAT CHECKED
Yes for administrative accounting: its gate passes. The natural-generalization gate fails, and the capstone retains a blocked verdict with activation and publication disabled. Historical paper readiness is explicitly scoped to the historical FoVer paper. Recorded circular branch outcomes are not promoted into a positive capstone verdict.

## EVIDENCE

- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `honest_verdict`: `complete_blocked_v758_capstone`
- `complete_accounting`: `passed`: `true`
- `natural_generalization_count`: `0`
- `activation_authorized`: `false`
- `publication_authorized`: `false`
- `historical_publication_scope`: `historical FoVer paper only`
- `capstone_next_decision`: `Keep independent branches separate; do not activate or publish externally`

## RECOMMENDATION
KEEP

Elided upstream rows were not assessed; this verdict covers only the capstone’s claim scope.
