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
| NO_CLAIM | 3 |

## experiment_7550_v660_count_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Local count learning fails to demonstrate a qualified benefit over control baselines, failing the exploratory effect gate.

## WHAT WOULD REFUTE IT
The observation that would refute this null claim is `local_count` achieving a statistically significant reduction in Brier score over both `frozen` and `global_count` baselines (Holm-adjusted p < 0.05 and simultaneous upper 95% confidence limit on delta Brier < 0) while keeping retention deterioration within threshold limits, thereby passing the `exploratory_effect` gate.

## WAS THAT CHECKED
Yes; checked in `independent_reduction` across `primary_contrasts`, `retained_absolute_brier`, and `gate_check_summary`.

## EVIDENCE
- `"positive_claim"`: `false`
- `"honest_verdict"`: `"complete_null_count_claims_qualified_benefit_gate_failed"`
- `"verdict_class"`: `"null"`
- `"failed_checks"`: `["exploratory_effect"]`
- `"effect_passed"`: `false`
- `"retention_passed"`: `false`
- `"arm"`: `"local_count"`, `"mean_brier"`: `0.14997917983634546`
- `"arm"`: `"global_count"`, `"mean_brier"`: `0.14120923420057496`
- `"comparator"`: `"global_count"`, `"holm_passed"`: `false`, `"mean_delta"`: `0.008769945635770493`
- `"comparator"`: `"frozen"`, `"holm_passed"`: `false`, `"mean_delta"`: `-0.007269976501000086`
- `"no_headroom_annotation"`: `"No no-headroom claim is made. The local arm lost to the global control and exceeded retention limits."`

## RECOMMENDATION
KEEP

## experiment_7551_native_pilot.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation indicating that all upstream gate criteria were satisfied (e.g., `gpu_capacity_observed_score` evaluating to 1 and `verdict_class` belonging to the expected set) while the run was recorded as blocked, or any substantive empirical outcome regarding source-intervention feasibility despite execution never having occurred.

## WAS THAT CHECKED
No. The artifact is a pre-execution gate receipt that evaluated upstream dependencies and halted at `conductor_pre_gate` without executing an experiment or testing any empirical claim.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`
`gate_check_summary`: `gate-unsat(final): 2 of 4 gate(s) failed; first failure: exp7548-capture-runner.gpu_capacity_observed_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7556_v660_arc_corrected_custody.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Authenticated B2 evidence under the corrected token budget demonstrates no induction efficacy.

## WHAT WOULD REFUTE IT
Any observed induction attempt yielding an executed plan (`planned` = true), an observed or passing verifier outcome (`verifier_result` != "not_observed"), non-zero level progress attributed to induction (`credited_level_count` > 0 or `level_up_progress` = true), or the acceptance check `induction_efficacy_observed` observing true.

## WAS THAT CHECKED
Yes; checked across 33 completed induction attempts logged in `induction_attempt_rows` and `per_game_results`, summarized in `join_summary`, and evaluated in `acceptance_gate_results`.

## EVIDENCE
- `honest_verdict`: `complete_null_corrected_b2_authenticated_no_efficacy`
- `verdict_class`: `null`
- `positive_claim`: `false`
- `check`: `induction_efficacy_observed`
- `expected`: `true`
- `observed`: `false`
- `passed`: `false`
- `planned_count`: `0`
- `credited_level_count`: `0`
- `verifier_observed_count`: `0`
- `planned`: `false`
- `verifier_result`: `not_observed`
- `level_up_progress`: `false`

## RECOMMENDATION
KEEP

## experiment_10008_b2_induction_gate_measurement_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No claim is asserted to refute. If the artifact had claimed usable induction output or gate efficacy, that claim would be refuted by observing empty model generations, unobserved verifier outcomes, and unmet sample floors; here, the artifact explicitly disclaims measuring usable induction output or asserting gate quality.

## WAS THAT CHECKED
No. Refutation was not evaluated because the artifact explicitly disclaims comparative quality and efficacy claims, serving solely as a preserved execution receipt of think-mode token budget exhaustion.

## EVIDENCE
- `"numeric_gate_quality_claim": false`
- `"hidden_game_efficacy_claim": false`
- `"publication_mode": "feasibility_only"`
- `"honest_verdict": "complete_feasibility_only_sample_floor_not_met"`
- `"gate_ready_to_ship": false`
- `"issue": "All 33 durable responses spent the 4,096-token budget in hidden reasoning and emitted empty content, so this artifact does not measure usable induction output."`
- `"citation_instruction": "Preserve this artifact as the think-mode budget-exhaustion record. Cite Experiment 10009 for the codeonly-suppressed B2 measurement; it is not live-default think-mode parity."`
- `"met": false`

## RECOMMENDATION
KEEP

## experiment_10009_b2_induction_gate_measurement_v3.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact asserts no comparative, performance, or gate-quality claim, there is no empirical claim to refute. To refute an empirical claim of gate efficacy or headroom, the artifact's own rows would need to demonstrate that gating suppressed progress-yielding attempts, produced negative headroom relative to an un-gated baseline, or failed to meet pre-specified statistical power floors.

## WAS THAT CHECKED
No. The artifact explicitly disclaims numeric gate quality, game efficacy, readiness to ship, and live parity, while explicitly recording that the sample floor was not met and the positive-control progress signal was saturated.

## EVIDENCE
- `numeric_gate_quality_claim`: `false`
- `hidden_game_efficacy_claim`: `false`
- `gate_ready_to_ship`: `false`
- `live_default_parity_claim`: `false`
- `publication_mode`: `feasibility_only`
- `honest_verdict`: `complete_feasibility_only_sample_floor_not_met`
- `met`: `false`
- `positive_control_headroom_exists`: `false`
- `interpretation`: `The progress proxy is saturated and cannot establish that B2 has no positive-control headroom.`

## RECOMMENDATION
KEEP

## experiment_7557_v660_arc_generalization.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
ARC generalization testing demonstrates a complete null result with feasibility only, as causal endpoints and plan-linked level gains were completely unavailable across all attempts.

## WHAT WOULD REFUTE IT
Any observation of positive level progress (`level_gains > 0` or `positive_count > 0`), attributable plan-linked useful attempts (`plan_linked_useful_attempts > 0`), supervisor intervention benefit (`helped > 0`), identifiable plan-to-action causal joins (`plan_linked_execution.available = true`), or satisfaction of the gate check (`plan_linked_efficacy_identifiable = true`).

## WAS THAT CHECKED
Yes. Across 35 live agent attempts across multiple games (including `ar25`, `bp35`, and `dc22`) recorded in `rows`, `per_game_results`, `endpoint_identifiability`, `supervisor_arm_rows`, and `gate_check_summary`, level gains and causal attribution were tracked and repeatedly evaluated to 0.

## EVIDENCE
- `honest_verdict`: `complete_null_feasibility_only_causal_endpoint_unavailable`
- `verdict_class`: `null`
- `positive_claim`: `false`
- `numeric_gate_quality_claim`: `false`
- `gate_ready_to_ship`: `false`
- `new_level_solve_claimed`: `false`
- `actual_level_progress`: `positive_count`: `0`
- `level_gains`: `0`
- `disposition`: `complete_observed_feasibility`
- `plan_linked_execution`: `disposition`: `unavailable_missing_causal_joins`
- `gate_check_summary`: `all_passed`: `false`
- `failed_count`: `5`
- `helped`: `0`

## RECOMMENDATION
KEEP

## experiment_7558_v660_service_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Durable service costs are successfully measured, but hardware acceleration benefit remains unmeasured and does not justify new hardware purchases because updates account for a non-dominant fraction of full service time.

## WHAT WOULD REFUTE IT
Observing that `measured_dominant_hardware_cost` evaluated to `true` (such as `measured_update_fraction` accounting for a dominant share of service latency to yield meaningful Amdahl speedup and set `new_hardware_purchase_justified` to `true`), or observing durable service and crash recovery failures (such as `lost_acknowledged_event_count` > 0 or `passed: false` in `crash_rows`).

## WAS THAT CHECKED
Yes. Durability and crash resilience were tested across 60 trials in `service_rows` and 3 crash boundary trials in `crash_rows`. Hardware acceleration headroom was directly tested against measured latency breakdown in `hardware_acceleration_bound` and evaluated via the `measured_dominant_hardware_cost` gate in `acceptance_gate_results`.

## EVIDENCE
`"honest_verdict": "complete_null_durable_service_measured_hardware_benefit_unmeasured"`
`"positive_claim": false`
`"verdict_class": "null"`
`"new_hardware_purchase_justified": false`
`"check": "measured_dominant_hardware_cost"`
`"expected": true`
`"observed": false`
`"passed": false`
`"measured_update_fraction": 0.05784469960043681`
`"ideal_update_only_speedup": 1.061396140929107`
`"service_cost_complete_score": 1`
`"crash_recovery_complete_score": 1`
`"lost_acknowledged_event_count": 0`
`"reconstruction_parity": true`
`"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7559_v660_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Milestone v660 required source science is externally blocked by unavailable GPU capacity, while all completed research branches establish null benefit with no positive claim asserted.

## WHAT WOULD REFUTE IT
The claim would be refuted if the artifact's own data demonstrated that GPU capacity was available (`gpu_capacity_observed_score` equal to 1), that the conductor pre-gates passed, that fresh source capture executed, or that any completed branch demonstrated positive measured benefit (`benefit` > 0 or `hardware_benefit` > 0).

## WAS THAT CHECKED
Yes. GPU capacity was checked against the process inventory in `gate_check_summary` (observing 0 vs expected 1), resulting in external gate blocks for the native pilot and capture tasks. Furthermore, each completed branch audited its upstream tasks and verified null benefit: count learning and audit verified `benefit` of 0; ARC live agent generalization verified lack of causal efficacy and `benefit` of 0; and CPU service/board continuity verified unmeasured hardware acceleration with `hardware_benefit` of 0.

## EVIDENCE
- `"honest_verdict": "complete_blocked_required_v660_source_science_externally_gated"`
- `"positive_claim": false`
- `"check": "gpu_capacity_available"`
- `"field": "gpu_capacity_observed_score"`
- `"expected": 1`
- `"observed": 0`
- `"passed": false`
- `"conclusion": "Fresh source capture and its independent reduction did not run after the external GPU-capacity gate failed."`
- `"verdict_class": "blocked"`
- `"conclusion": "Independent arithmetic qualified the completed count measurement. The registered exploratory benefit failed."`
- `"benefit": 0`
- `"conclusion": "Corrected bytes are qualified, but plan-linked efficacy is not identifiable and support floors failed."`
- `"conclusion": "Durable CPU service and board accounting completed. No measured dominant cost supports hardware benefit."`
- `"hardware_benefit": 0`
- `"no_headroom_annotation": "No aggregate no-headroom claim is made. Count retention failed, source science is absent, ARC causal joins are absent, and hardware benefit is unmeasured."`

## RECOMMENDATION
KEEP
