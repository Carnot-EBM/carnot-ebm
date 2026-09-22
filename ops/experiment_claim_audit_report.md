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
| NO_CLAIM | 4 |

## experiment_7490_v656_historical_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Historical probability calibration shows null benefit over simple baselines, and the historical pre-replay shuffled negative control is chronologically invalid due to leakage from future events.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. The candidate probability method (`gibbs`) demonstrated statistically significant improvement over simple calibration baselines (`temperature` and `logistic`) on proper scoring rules (substantially lower Brier loss and log loss with significant p-values, resulting in `probability_benefit_passed` being true and `historical_probability_benefit` passing as `positive`), or
2. The historical shuffled negative control showed zero lookahead leakage into future timestamps (`future_origin_assignment_count` of 0, `from_future_event` false across all rows, and `shuffled_negative_control_chronology` passing with observed value 0).

## WAS THAT CHECKED
Yes. Probability quality was checked against competitive baselines across 74 evaluated groups with 10,000 bootstrap draws in `historical_probability_summary` and `historical_probability_rows`. Control chronology was checked by tracking timestamps across all 764 label assignments in `feedback_chronology_rows` and summarized in `feedback_chronology_summary`.

## EVIDENCE
- `"honest_verdict": "complete_null_historical_probability_limit_and_invalid_shuffled_control"`
- `"verdict_class": "null"`
- `"check": "historical_probability_benefit"`
- `"expected": "positive"`
- `"observed": "null"`
- `"passed": false`
- `"check": "shuffled_negative_control_chronology"`
- `"expected": 0`
- `"observed": 376`
- `"probability_benefit_passed": false`
- `"gibbs"`
- `"brier": 0.5783718222014341`
- `"log_loss": 2.524944140767486`
- `"temperature"`
- `"brier": 0.3179760819813062`
- `"log_loss": 0.90299758804979`
- `"logistic"`
- `"brier": 0.4853637188480826`
- `"log_loss": 1.5621939806892695`
- `"one_sided_p": 1.0`
- `"future_origin_assignment_count": 376`
- `"assignment_count": 764`
- `"from_future_event": true`
- `"valid": false`

## RECOMMENDATION
KEEP

## experiment_7491_v656_window_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a protocol-sealing and readiness receipt that explicitly disclaims measuring any empirical benefit or comparative performance, there is no comparative claim to refute. If evaluated purely as a structural readiness receipt, it would be refuted by failing acceptance gates (`acceptance_gate_results`), failed import or test suites (`validation_receipts`), unsealed or leaking role splits (`role_manifest`), or token budget violations.

## WAS THAT CHECKED
No comparative or model benefit hypothesis was checked, as no live model inferences were attempted (`forward_calls_attempted` is 0). Structural integrity, deterministic token accounting, preconditions, and test receipts were checked and passed.

## EVIDENCE
- `honest_verdict`: `complete_null_structural_window_protocol_ready`
- `verdict_class`: `null`
- `window_protocol_ready_score`: `1`
- `field_principles`: `The bare score means complete sealed roles and lossless label-blind prompts, not benefit.`
- `small_ebm_training`: `Protocol sealing performs no model or EBM fit.`
- `random_seed`: `No stochastic fit or benefit estimate occurs.`
- `e2e_scope`: `Pure reporting changed no shared runtime, binding, ARC, telemetry, or Rust path.`
- `inference_substrate_class`: `no_model_load`
- `model_invoked`: `false`
- `forward_calls_attempted`: `0`
- `disposition`: `planned_unstarted`
- `unstarted_forwards`: `4128`

## RECOMMENDATION
KEEP

## experiment_7492_v656_window_pilot.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native window transport and capture budget forecasts are ready and feasible within time caps, supporting a complete null with no predictive benefit claimed.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. Any forward pass in native window transport failed, generated non-finite logits, failed KV state resets, or had invalid token mappings, resulting in `failed_forwards` > 0 or `transport_reduction.passed` evaluating to false.
2. Empirical runtime forecasts exceeded the 3300.0s hard cap (`forecast_s` > `hard_cap_s`), setting `feasible` to false in `fit_capture` or `evaluation_capture`.
3. Predictive benefit was claimed without held-out generalization evidence (`predictive_benefit_claimed` being true), failing the `predictive_benefit_not_claimed` acceptance gate.
4. Execution was incomplete or aborted (`terminal_status` != "complete" or `pilot_complete_score` < 1).

## WAS THAT CHECKED
Yes. Native transport execution was checked across 58 live forward calls with zero failures in `transport_reduction`, budget feasibility was checked against the 3300.0s hard cap using empirical p90 token prefill rates in `capture_budget_forecasts`, and the absence of predictive benefit claims along with validity and readiness requirements was checked and verified in `acceptance_gate_results`.

## EVIDENCE
`"honest_verdict": "complete_null_window_native_transport_and_capture_forecasts_ready"`
`"verdict_class": "null"`
`"predictive_benefit_claimed": false`
`"window_native_ready_score": 1`
`"pilot_complete_score": 1`
`"terminal_status": "complete"`
`"verifier_is_oracle": false`
`"transport_reduction"`:
`"passed": true`
`"complete_forwards": 58`
`"failed_forwards": 0`
`"finite_logits_valid": true`
`"fresh_kv_valid": true`
`"capture_budget_forecasts"`:
`"capture": "fit_capture"`
`"feasible": true`
`"forecast_s": 2350.4010691859157`
`"hard_cap_s": 3300.0`
`"capture": "evaluation_capture"`
`"feasible": true`
`"forecast_s": 2745.3543594869366`
`"hard_cap_s": 3300.0`
`"acceptance_gate_results"`:
`"check": "authenticated_identity_and_accounting"`, `"passed": true`
`"check": "native_window_transport"`, `"passed": true`
`"check": "required_validation"`, `"passed": true`
`"check": "capture_forecasts_fit"`, `"passed": true`
`"check": "capture_forecasts_evaluation"`, `"passed": true`
`"check": "predictive_benefit_not_claimed"`, `"passed": true`

## RECOMMENDATION
KEEP

## experiment_7493_v656_window_fit_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact asserts no comparative or predictive claim, there is no empirical efficacy claim to refute. If a claim of predictive benefit or downstream window-fit efficacy were asserted, it would be refuted by showing that the captured window-fit representations fail to outperform a standard baseline (such as unwindowed full-prompt scoring or a trivial heuristic) on held-out tasks.

## WAS THAT CHECKED
No. Predictive benefit was explicitly not tested or measured, and no comparative evaluation against any baseline was performed.

## EVIDENCE
`predictive_benefit_claimed`
`false`
`honest_verdict`
`complete_null_window_fit_capture_ready_predictive_benefit_not_tested`
`verdict_class`
`null`
`check`
`predictive_benefit_not_measured`
`principle`
`A favorable seed, fixture or low-support result cannot replace held-out value; this check prevents capture data from becoming a benefit claim.`
`deterministic_null_seed_reason`
`Capture performs no fit, interval, or outcome-guided retry.`
`inference_substrate_class`
`model_load_no_generation`
`validation_receipts`

## RECOMMENDATION
KEEP

## experiment_7494_v656_window_eval_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no predictive-benefit or comparative headline to falsify; this artifact only asserts capture completeness and readiness. That operational assertion would be refuted by incomplete or invalid eligible calls, insufficient role support, or failed capture validation.

## WAS THAT CHECKED
Yes. Capture completeness, eligible-call validity, role support, per-group status, and required validation were checked; predictive benefit and serious comparator performance were not checked because no such claim was made.

## EVIDENCE
`"honest_verdict": "complete_null_window_evaluation_capture_ready_predictive_benefit_not_tested"`, `"predictive_benefit_claimed": false`, `"verdict_class": "null"`, `"capture_complete_score": 1`, `"role_support_score": 1`, `"all_eligible_calls_valid": true`, `"failed": 0`, `"censored": 0`, `"terminal_status": "complete"`

## RECOMMENDATION
KEEP

## experiment_7498_v656_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation can refute an unmade comparative benefit claim. If the blocked-status finding is treated as the headline, it would be refuted by valid scientific outputs from producers 7495–7497, including populated probability, utility, and learning results.

## WAS THAT CHECKED
Yes, for the blocked-status finding: `claim_dispositions`, `gate_check_summary`, and the three independent result arrays check those inputs and record them as absent. No comparative benefit was tested.

## EVIDENCE
`honest_verdict`: `complete_blocked_missing_v656_scientific_inputs`; `verdict_class`: `blocked`; `benefit_state`: `not_claimed`; `availability`: `absent`; `all_benefit_passed`: `false`; `all_readiness_passed`: `false`; `independent_learning_rows`: `[]`; `independent_probability_rows`: `[]`; `independent_utility_rows`: `[]`; `model_invoked`: `false`; `inference_substrate`: `aggregation_from_upstream_artifacts`

## RECOMMENDATION
KEEP

## experiment_7500_v656_arc_opportunity_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found no authenticated eligible supervisor opportunity in the available valid Panel A rows and appropriately blocked any pooled benefit or acceleration claim because Panel B and adequate compatible support were unavailable.

## WHAT WOULD REFUTE IT
A valid included row with nonzero eligible supervisor service or a triggered opportunity would refute the observed-null finding; a present, identity-compatible Panel B yielding sufficient pooled support and an estimable intervention benefit would refute the stated reasons for blocking broader claims.

## WAS THAT CHECKED
Yes for the observed-null finding: 18 valid completed Panel A episodes were reduced using explicit eligibility and trigger fields, and none showed eligible service or a triggered opportunity. The broader conclusion was deliberately not tested to completion because Panel B was missing, pooling was disallowed, and no intervention occurred; the artifact reports those limitations rather than converting them into a general null.

## EVIDENCE
`honest_verdict` = `complete_blocked_panel_b_blocked_missing_opportunity_audit`; `verdict_class` = `blocked`; `panel_results.A.state` = `valid`; `complete_independent_units` = `18`; `excluded_independent_units` = `0`; `eligible_service_upper_ns` = `0`; `triggered_opportunity_count` = `0`; `panel_results.B.state` = `blocked_missing`; `source_artifact_missing`; `pooling_allowed` = `false`; `valid_episode_count` = `18`; `episode_support_floor` = `30`; `identities_compatible` = `false`; `efficacy_estimate` = `null`; `observational_benefit_inference_allowed` = `false`; `hardware_acceleration_claim` = `false`

## RECOMMENDATION
KEEP

## experiment_7502_v656_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Progression on the v656 milestone is blocked because required evidence is absent or externally gated.

## WHAT WOULD REFUTE IT
Observing that the required upstream producer artifacts exist on disk, that upstream audit tasks resolve to non-blocked terminal dispositions, and that all acceptance benefit gates evaluate to passing or positive.

## WAS THAT CHECKED
Yes, in `gate_check_summary` and `acceptance_gate_results`, which explicitly verified upstream artifact existence, terminal verdict classes, and benefit criteria, recording 7 failed producer checks and 4 failed benefit gates.

## EVIDENCE
- `"honest_verdict": "complete_blocked_required_v656_evidence_absent_or_externally_gated"`
- `"verdict_class": "blocked"`
- `"status": "complete_blocked_required_v656_evidence_absent_or_externally_gated"`
- `"failed_count": 7`
- `"results/experiment_7495_v656_window_calibration.json"`
- `"results/experiment_7496_v656_causal_update_fixture.json"`
- `"results/experiment_7497_v656_causal_online_learning.json"`
- `"results/experiment_7498_v656_independent_audit.json"`
- `"results/experiment_7499_v656_arc_panel_b.json"`
- `"results/experiment_7500_v656_arc_opportunity_audit.json"`
- `"results/experiment_7501_v656_service_placement.json"`
- `"current_probability_support_effect_and_multiplicity"`
- `"causal_feedback_benefit_and_retention"`
- `"arc_cross_game_support"`
- `"durable_service_measurement"`
- `"blocked_missing"`
- `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP
