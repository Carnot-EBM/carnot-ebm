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
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7508_v657_static_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7509_v657_causal_online.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The causal online adaptation experiment is a valid measurement that failed its pre-registered support and benefit gates, establishing an honest complete null result without demonstrated causal benefit over controls.

## WHAT WOULD REFUTE IT
The headline null claim would be refuted by either:
1. Observed data demonstrating causal benefit: all five primary contrasts beating controls with Holm-adjusted p < 0.05 and upper 95% confidence bound < 0, a local Brier effect size <= -0.01, and primary support meeting the >= 12 permutable labels threshold across all seeds.
2. Observed data indicating an invalid run rather than a valid null: failed restart parity, non-zero chronology violations, or missing execution units.

## WAS THAT CHECKED
Yes. Primary support, effect size thresholds, multi-arm Holm contrasts against serious rival controls (including affine and intercept baselines), and measurement validity/restart parity were all explicitly computed and checked under `acceptance_gate_results`, `gate_check_summary`, `measurement_reduction`, and `restart_parity_rows`.

## EVIDENCE
- `honest_verdict`: `complete_null_causal_online_measurement_valid_benefit_gate_failed`
- `verdict_class`: `null`
- `causal_information_value_score`: `0`
- `online_benefit_score`: `0`
- `failed_checks`: `primary_support`, `local_brier_effect_size`, `five_holm_primary_contrasts`
- `local_brier_effect_size`: `observed` `-0.0022371781201150523`
- `affine_brier`: `holm_adjusted_p` `0.5847076461769115`, `upper95_delta` `0.0002389333129294682`, `holm_passed` `false`
- `local_log_loss`: `mean_delta` `0.0021875455431429837`, `holm_adjusted_p` `1.0`, `holm_passed` `false`
- `valid_complete_measurement`: `passed` `true`, `observed` `1`
- `restart_parity`: `passed` `true`, `observed` `1`

## RECOMMENDATION
KEEP

## experiment_7510_v657_causal_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Causal online adaptation fails the audit benefit gate, yielding no qualified online benefit over baseline comparators.

## WHAT WOULD REFUTE IT
The claim of a failed benefit gate would be refuted if the causal online model demonstrated statistically significant superiority under Holm adjustment against all primary comparators (specifically non-trivial rivals `affine_brier` and `local_log_loss`), satisfied all audit label delivery thresholds (`support_passed`), and achieved a non-zero `qualified_online_benefit_score`.

## WAS THAT CHECKED
Yes; checked in `gate_check_summary.failed_checks`, `independent_reduction.primary_contrasts` across five comparator arms, and `independent_reduction.support_rows`.

## EVIDENCE
- `honest_verdict`: `complete_null_v657_causal_audit_benefit_gate_failed`
- `verdict_class`: `null`
- `all_benefit_passed`: `false`
- `check`: `primary_support`, `observed`: `false`
- `check`: `five_primary_contrasts`, `observed`: `false`
- `primary_passed`: `false`
- `support_passed`: `false`
- `comparator`: `affine_brier`, `holm_passed`: `false`
- `comparator`: `local_log_loss`, `holm_passed`: `false`
- `producer_online_benefit_score`: `0`
- `qualified_online_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_7511_v657_arc_evidence_recovery.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The historical ARC Panel B evaluation evidence is an authenticated, fully qualified complete null result with zero progress achieved across all 18 scheduled episodes.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if:
1. Any episode row demonstrated positive task advancement or progress (e.g., `progressed` is true, `peak_level` > 0, `new_level_credit` > 0, or `scientific_benefit_score` > 0).
2. Validity or qualification preconditions failed (e.g., `required_validity_and_readiness_passed` is false, `arc_panel_b_qualified_score` is 0, source hashes mismatched, interval timing failed reconciliation, or schedule rows were missing).
3. Any unit was censored or aborted due to an execution fault or crash rather than exhausting the full action cap without progress.

## WAS THAT CHECKED
Yes. All 18 scheduled episodes across 6 game clusters were checked in `per_episode_results` and `sample_size_budget`. Each unit was verified to have completed its 180-action cap without progress (`censor_reason`: "no_progress_within_action_cap", `progressed`: false). The acceptance gate `reproduced_progress_support` explicitly tested for progress (`expected`: 1, `observed`: 0), and all integrity/readiness preconditions passed (`required_validity_and_readiness_passed`: true, `bounds_valid`: true, `reconciles_within_timestamp_resolution`: true).

## EVIDENCE
- `"honest_verdict": "complete_null_arc_panel_b_evidence_recovered"`
- `"arc_panel_b_qualified_score": 1`
- `"scientific_benefit_score": 0`
- `"solve_claim_made": false`
- `"required_validity_and_readiness_passed": true`
- `"check": "reproduced_progress_support"`
- `"field": "progressed"`
- `"expected": 1`
- `"observed": 0`
- `"passed": false`
- `"principle": "Complete zero progress remains a valid null and not a benefit claim."`
- `"planned_independent_units": 18`
- `"attempted_independent_units": 18`
- `"completed_independent_units": 18`
- `"censored_independent_units": 18`
- `"failed_independent_units": 0`
- `"action_count": 180`
- `"action_limit": 180`
- `"progressed": false`
- `"censor_reason": "no_progress_within_action_cap"`
- `"new_level_credit": 0`
- `"terminal_level": 0`
- `"bounds_valid": true`
- `"reconciles_within_timestamp_resolution": true`

## RECOMMENDATION
KEEP

## experiment_7512_v657_arc_opportunity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The artifact claims a complete null result (`complete_null_zero_eligible_supervisor_opportunity`), establishing that zero authenticated eligible supervisor firing opportunities occurred across 36 audited ARC episodes, requiring supervisor efficacy tuning to be retired until live reachable choices exist.

## WHAT WOULD REFUTE IT
Any audited episode row or supervisor opportunity row exhibiting `eligible_firing_opportunity_count > 0`, `eligible_firing_opportunity: true`, or any applied supervisor redirection producing valid level progression or positive causal outcome would refute the null claim.

## WAS THAT CHECKED
Yes. It was checked across all 36 completed episodes (12 unique games) in `panel_results` and `rows`, across 6,480 gate evaluations summarized in `supervisor_disposition`, and in `gate_check_summary.failed_checks` under check `eligible_supervisor_opportunity_present` (which observed `0`).

## EVIDENCE
- `honest_verdict`: `"complete_null_zero_eligible_supervisor_opportunity"`
- `status`: `"complete_null_zero_eligible_supervisor_opportunity"`
- `eligible_firing_opportunity_count`: `0`
- `applied_redirection_count`: `0`
- `gate_evaluation_count`: `6480`
- `diagnosis`: `"zero_authenticated_eligible_opportunities_with_shadow_firings"`
- `supervisor_efficacy_tuning`: `"retired_until_live_reachable_choices"`
- `efficacy_estimate`: `null`
- `broad_generalization_supported`: `false`
- `progressed_game_count`: `0`
- `check`: `"eligible_supervisor_opportunity_present"`
- `all_passed`: `false`

## RECOMMENDATION
KEEP

## experiment_7513_v657_placement_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Integer quantization achieves deterministic numeric placement parity with float32 across training and calibration rows with zero policy disagreements, while whole-service speedup remains unmeasured.

## WHAT WOULD REFUTE IT
The headline claim would be refuted by any policy action disagreement between integer and float32 arithmetic (`action_disagreement_count > 0`), any probability discrepancy exceeding the threshold (`max_abs_probability_error > 0.01`), any accumulator overflow (`overflow_count > 0`), or an unverified assertion of whole-service speedup despite lacking measured execution timing for the end-to-end pipeline.

## WAS THAT CHECKED
Yes. Deterministic numeric equivalence was tested across 472 groups (covering int16 and int8 formats evaluated against 9 decision policies, totaling 4,248 comparisons) in `quantization_rows` and `independent_reduction.numeric`. Refutation was genuinely possible if integer rounding shifted probabilities across policy thresholds (maximum observed error was ~0.00295 against a 0.01 limit). Accumulator overflow was audited per row. Furthermore, whole-service speedup was evaluated via acceptance gate `whole_service_denominator_measured`, failed as unmeasured, and was honestly reported as an unmeasured null rather than claimed.

## EVIDENCE
- `honest_verdict`: `"complete_null_numeric_placement_ready_whole_service_speedup_unmeasured"`
- `numeric_placement_ready_score`: `1`
- `action_disagreement_count`: `0`
- `max_abs_probability_error`: `0.002948939800262451`
- `probability_error_limit`: `0.01`
- `overflow_count`: `0`
- `observed_row_count`: `472`
- `planned_row_count`: `472`
- `check`: `"whole_service_denominator_measured"`
- `observed`: `false`
- `passed`: `false`
- `failed_count`: `1`
- `whole_service_speedup`: `null`
- `fpga_performance_measured`: `false`
- `methodology_note`: `"Zero action disagreement is deterministic numeric parity on training and calibration rows. It is not held-out predictive benefit or FPGA performance."`

## RECOMMENDATION
KEEP

## experiment_7514_v657_service_trace.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The numeric update kernel contributes an insignificant fraction of end-to-end execution time (~0.0053%, yielding an ideal Amdahl speedup ceiling of 1.00005x) and introduces a median paired overhead of 450.4 µs, supporting a valid null service-level speedup result with no predictive efficacy or production SLA claimed.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. `measured_update_kernel_share` was substantial enough to enable meaningful end-to-end service speedup, or `ideal_infinite_speed_update_kernel_ceiling` meaningfully exceeded 1.0.
2. Timing coverage failed (`interval_coverage_passed` was false or any row showed `unaccounted_ns` > 0), indicating unmeasured or double-counted execution time.
3. Durable parity failed (`durable_parity_passed` was false or `fsync_completed` was false), showing that the `update` and `no_update` control arms bypassed identical persistence boundaries.
4. Predictive efficacy, 100x speedup, or production SLAs were claimed (`ebm_efficacy_claimed`, `measured_100x_claimed`, or `production_sla_claimed` observed as true).
5. Paired sample support fell below required thresholds (`paired_support_passed` was false or `paired_request_count` < 20).

## WAS THAT CHECKED
Yes.
- Kernel duration share and speedup ceilings were directly computed from measured stage timestamps in `amdahl_bounds` and `service_reduction`.
- Exclusive timing coverage was verified for each request in `request_rows` (each reporting `unaccounted_ns` of 0) and verified by gate check `interval_coverage`.
- Durable fsync completion and acknowledgement parity across both arms were verified in `request_rows` and by gate check `durable_parity`.
- Disclaiming of predictive efficacy from timing was verified by gate check `efficacy_not_inferred_from_timing` as well as explicit boundary flags.
- Support requirements were evaluated and satisfied across 24 matched request pairs (48 requests total) with zero failed or censored rows in `sample_size_budget`.

## EVIDENCE
`"honest_verdict"`: `"complete_null_controlled_service_trace_ready_no_efficacy_or_sla_claim"`
`"verdict_class"`: `"null"`
`"ebm_efficacy_claimed"`: `false`
`"measured_100x_claimed"`: `false`
`"production_sla_claimed"`: `false`
`"measured_update_kernel_share"`: `5.2558091618348854e-05`
`"ideal_infinite_speed_update_kernel_ceiling"`: `1.0000525608541164`
`"durable_parity_passed"`: `true`
`"interval_coverage_passed"`: `true`
`"paired_support_passed"`: `true`
`"paired_request_count"`: `24`
`"unaccounted_ns"`: `0`
`"fsync_completed"`: `true`
`"check"`: `"efficacy_not_inferred_from_timing"`
`"observed"`: `false`
`"passed"`: `true`
`"median"`: `450435.5`
`"p95"`: `1146742.5499999998`

## RECOMMENDATION
KEEP

## experiment_7515_v657_capstone.json

**SKIPPED_ALREADY_FLAGGED**
