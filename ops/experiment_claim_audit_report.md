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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |

## experiment_7579_v662_decision_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The decision-learning candidate demonstrates a complete null result with no statistically significant benefit over count-based or raw baselines across static calibration and online feedback, and fails retention non-regression.

## WHAT WOULD REFUTE IT
The null claim would be refuted if the artifact's own data showed the candidate achieving statistically significant reductions in Brier loss and decision cost against serious baseline controls (`local_count`, `global_count`, `raw_original`) with one-sided Holm-adjusted p-values below 0.05 (`reject_at_0_05` being true) alongside non-negative retention degradation, thereby passing the benefit and retention gates (`static_probability_benefit`, `learning_feedback_benefit`, and `retention_non_regression`).

## WAS THAT CHECKED
Yes. Refutation was given a real chance to happen across:
- `acceptance_gate_results` and `gate_check_summary`, specifically evaluating `static_probability_benefit`, `learning_feedback_benefit`, and `retention_non_regression`.
- `learning_interval_reduction` over 5,000 bootstrap replays, explicitly testing contrasts and Holm-adjusted one-sided p-values against serious rivals (`brier_vs_local_count`, `brier_vs_global_count`, `brier_vs_raw`) as well as `brier_vs_shuffled_feedback`.
- `retention_reduction` over 5,000 replays tracking `retention_degradation`.
- `static_reconstruction` computing contrasts against `raw_original` and `temperature_original` across 80 units and 1,000 draws.
- Sensitivity verification in `mutation_rows`, confirming that corrupted fixtures (such as label leakage, ordering mismatches, missing rows, duplicate updates, and sign errors) are actively detected and cause test failures rather than silently passing.

## EVIDENCE
- `"honest_verdict": "complete_null_independent_static_and_learning_audit"`
- `"verdict_class": "null"`
- `"verifier_is_oracle": false`
- `"flagged_adversarial": false`
- `"fresh_confirmatory_claim_allowed": false`
- `"descriptive_exposed_data_only": true`
- `"gate_check_summary"`: `"passed": false`
- `"failed_count": 3`
- `"failed_checks": [ "static_probability_benefit", "learning_feedback_benefit", "retention_non_regression" ]`
- `"branch_conclusions"`:
  - `"causal"`: `"benefit": false`, `"failure_source": "no_signal_against_counts_and_uncertainty"`
  - `"retention"`: `"benefit": false`, `"failure_source": "harmful_recalibration"`
  - `"static"`: `"benefit": false`, `"failure_source": "harmful_recalibration_and_uncertainty"`
- `"learning_interval_reduction"`:
  - `"benefit_passed": false`
  - `"retention_passed": false`
  - `"brier_vs_global_count"`: `"reject_at_0_05": false`, `"holm_adjusted_p": 0.9818036392721455`
  - `"brier_vs_local_count"`: `"reject_at_0_05": false`, `"holm_adjusted_p": 0.9818036392721455`
  - `"brier_vs_raw"`: `"reject_at_0_05": false`, `"holm_adjusted_p": 0.19436112777444509`
  - `"brier_vs_shuffled_feedback"`: `"reject_at_0_05": false`, `"holm_adjusted_p": 0.07678464307138572`

## RECOMMENDATION
KEEP

## experiment_7580_v662_arc_verifier_support.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The verifier integrity guard achieves full readiness (`verifier_support_ready_score` = 1) by reproducing verifier defects, passing positive controls, and rejecting impure engines across deterministic fixtures.

## WHAT WOULD REFUTE IT
The claim would be refuted if the integrity guard were evaluated against an independent, external correctness oracle on live panel rollouts or active model generations, and was observed either generating false-positive rejections of valid models or failing to catch execution defects.

## WAS THAT CHECKED
No. Refutation was not given a chance to happen. Zero live model loads or forward passes were executed, all evaluations were run against deterministic hand-crafted CPU fixtures where the verifier serves as its own oracle, and the live panel protocol was left unexecuted (`frozen_protocol_unstarted`).

## EVIDENCE
- `honest_verdict`: `complete_circular_positive_fixture_guard_support_ready_no_live_benefit_claim`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `benefit`: `false`
- `benefit_reason`: `No live panel or model call ran; fixture success is circular.`
- `readiness`: `true`
- `verifier_support_ready_score`: `1`
- `inference_substrate`: `deterministic_cpu_verifier_fixtures_and_protocol_freeze`
- `inference_substrate_class`: `no_model_load`
- `model_invoked`: `false`
- `methodology_note`: `Exact 1.0 is expected on the code-defined positive fixture. It is a circular plumbing control, not a model-capability or live-benefit result.`
- `provenance`: `frozen_protocol_unstarted`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7581_v662_arc_bounded_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation that the artifact asserted a positive capability, solve rate, or operational readiness claim despite failing preflight acceptance gates.

## WAS THAT CHECKED
Yes; acceptance gates were evaluated under `acceptance_gate_results` and `gate_check_summary`, where `arc_e2e` failed, properly setting the status to blocked and halting execution before model invocation.

## EVIDENCE
`positive_claim`: `false`
`world_model_quality_claim`: `false`
`solve_provenance`: `canary_no_solve_claim`
`verdict_class`: `blocked`
`honest_verdict`: `complete_blocked_arc_e2e`
`status`: `complete_blocked_arc_e2e`
`inference_substrate`: `blocked_no_run`
`model_invoked`: `false`
`The canary has no solve claim; live_agent_self_discovery begins only in later episodes.`

## RECOMMENDATION
KEEP

## experiment_7582_arc_panel_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports a blocked pre-gate check, not a result about verifier integrity.

## WAS THAT CHECKED
No; the intended experiment did not run because upstream readiness gates failed.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"gate_check_summary"`: `"gate-unsat(final): 3 of 7 gate(s) failed; first failure: exp7581-arc-bounded-canary.arc_transport_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7584_v662_arc_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact disclaims any performance score, supervisor refinement, or empirical induction benefit, there is no headline comparative claim to refute. To refute the artifact's characterization as a pre-gate transport block, the artifact's own data would need to show uncensored episode rows with valid scores (`raw_numerator` not `null`), completed model calls (`completed_calls` > 0), or passing producer verification gates (`producer_exists` observing `true`).

## WAS THAT CHECKED
No. The upstream panel producers `results/experiment_7582_v662_arc_panel_a.json` and `results/experiment_7583_v662_arc_panel_b.json` were absent, tripping the `producer_exists` acceptance gate and blocking live execution before any comparative evaluation could take place.

## EVIDENCE
- `honest_verdict`: `complete_blocked_live_panel_producers_missing_or_invalid`
- `verdict_class`: `blocked`
- `official_score_claimed`: `false`
- `pooled_independence_claim`: `false`
- `supervisor_refinement_supported`: `false`
- `arc_claims_qualified_score`: `0`
- `solve_provenance`: `no_new_solve_credit`
- `prior_verdict_disposition`: `narrow_pre_gate_transport_block_preserved_not_scientific_hypothesis_retirement`
- `model_invoked`: `false`
- `check`: `producer_exists`
- `observed`: `false`
- `passed`: `false`
- `benefit`: `not_measured`
- `censored`: `true`
- `censoring_reason`: `producer_absent_or_invalid`
- `raw_numerator`: `null`

## RECOMMENDATION
KEEP

## experiment_10010_b2_think_on_pilot.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
In this 10-window pilot, think-on achieved mean held-out change fidelity of 0.574 versus 0.128 for the code-only baseline.

## WHAT WOULD REFUTE IT
A matched code-only arm tying or exceeding the think-on arm on mean held-out change fidelity would refute the claim.

## WAS THAT CHECKED
Yes. The artifact scores both approaches against recorded held-out transitions using the same primary metric, reproduces the preregistered code-only baseline, checks for held-out leakage, and reports 0.574 versus 0.128. Independent identity and expert controls demonstrate that the scoring could produce both failure and success.

## EVIDENCE
`honest_verdict`: `complete_think_on_pilot_10_windows_mean_change_fidelity_0.574_vs_codeonly_0.128`

`primary`: `masked symmetric-union change fidelity over held-out changing rows; a raised, wrong-type, or wrong-shape row scores 0`

`codeonly_baseline_mean_reproduced`: `0.12753842112150332`

`codeonly_baseline_ok`: `true`

`identity_all_zero`: `true`

`expert_all_one`: `true`

`heldout_leak_check`: `passed`: `true`

`verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7585_v662_portable_service.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Porting the recalibration service to a standalone Rust binary achieves a statistically significant whole-service latency improvement over the incumbent Python service while preserving numerical parity and board continuity.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. The lower 95% confidence bound of the warm whole-service speedup was <= 1.0x (indicating no demonstrable whole-service latency advantage over the Python baseline),
2. The Rust service failed numerical parity against the Python implementation (`typed_decision_mismatch_count` > 0, `max_probability_absolute_error` exceeding numerical tolerance, or any mismatched states/acknowledgments across the evaluated event streams), or
3. Core build, compilation, or test validation checks failed for the Rust binary.

## WAS THAT CHECKED
Yes:
1. Service speedup was empirically measured on host processes over 30 paired runs comparing the Rust service binary against the incumbent Python service (`pair_count`: 30), verifying that the lower 95% confidence interval exceeded 1.0x (evaluated in `acceptance_gate_results` under `whole_service_improvement_lower95` and reported in `whole_service_speedup.warm`).
2. Numerical parity was evaluated across 1,000 streams and 16,000 events between Python and Rust processes (evaluated in `acceptance_gate_results` under `kernel_numerical_readiness` and detailed in `parity_summary` and `rows`).
3. Rust release compilation, code formatting, and module imports were executed and validated (evaluated in `acceptance_gate_results` under `required_validation` and detailed in `rust_validation_receipts`).

## EVIDENCE
- `honest_verdict`: `"complete_positive_portable_service_improvement"`
- `verdict_class`: `"positive"`
- `verifier_is_oracle`: `false`
- `check`: `"whole_service_improvement_lower95"`, `expected`: `1.0`, `op`: `"gt"`, `observed`: `31.122338001012775`, `passed`: `true`
- `whole_service_speedup`: `warm`: `estimate`: `33.59268279121724`, `lower95`: `31.122338001012775`, `pair_count`: `30`
- `check`: `"kernel_numerical_readiness"`, `field`: `"portable_parity_score"`, `expected`: `1`, `observed`: `1`, `passed`: `true`
- `parity_summary`: `stream_count`: `1000`, `event_count`: `16000`, `typed_decision_mismatch_count`: `0`, `max_probability_absolute_error`: `4.440892098500626e-15`, `all_states_match`: `true`, `all_acknowledgments_match`: `true`

## RECOMMENDATION
KEEP

## experiment_7586_v662_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Required v662 external evidence remains blocked, despite completion of the capstone’s accounting and validation work.

## WHAT WOULD REFUTE IT
Terminal, valid, unblocked live-panel producer artifacts—along with a passing transport pre-gate—would refute the blocked-evidence claim.

## WAS THAT CHECKED
Yes. The acceptance and producer gates explicitly checked artifact existence, producer terminal status, and transport readiness; four required checks failed. The invalid live-verifier branch was not promoted into a benefit claim, and the circular fixture result was identified as circular.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v662_external_evidence`; `positive_claim`: `false`; `failed_count`: `4`; `arc_transport_ready_score`; `observed`: `0`; `path`: `results/experiment_7583_v662_arc_panel_b.json`; `observed`: `false`; `validity`: `false`; `benefit`: `not_measured`; `fixture_positive_is_circular`: `true`; `oracle_distinct_positive_claimed`: `false`; `semantic_null_claimed`: `false`

## RECOMMENDATION
KEEP
