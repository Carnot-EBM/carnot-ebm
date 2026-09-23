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

## experiment_7566_v661_energy_fit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Contrastive energy model fitting and decision control freezing completed with verified readiness, but predictive benefit over the baseline remains unmeasured and yields an honest null result.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. Candidate energy heads had demonstrated superior, verified predictive benefit over the baseline on held-out evaluation data, contradicting the null verdict; or
2. Preconditions, custody verification, optimization loss convergence, or readiness gates had failed or crashed, contradicting the claim that the energy fit and baseline controls are verified and ready; or
3. The artifact had asserted a positive empirical benefit (`positive_claim: true` or `complete_positive`) despite the temperature baseline outperforming all candidate energy models on tuning Brier score.

## WAS THAT CHECKED
Yes:
- Validity, preconditions, custody (`raw_custody` observed 240), and readiness scores were evaluated and passed in `acceptance_gate_results`, `validation_receipts`, and `challenge_controls`.
- Baseline superiority was verified in `frozen_head_manifest.strongest_comparator` and `frozen_head_manifest.temperature_baseline`, where the temperature baseline attained a lower `tune_brier` (`0.11930674576400344`) than any fitted energy head (lowest candidate `tune_brier` was `0.13525139808043912`).
- The absence of unearned held-out claims was explicitly checked in `acceptance_gate_results` under `heldout_benefit_unmeasured`, confirming `predictive_benefit_measured` is `false`.

## EVIDENCE
- `"honest_verdict": "complete_null_energy_fit_ready_benefit_unmeasured"`
- `"positive_claim": false`
- `"verdict_class": "null"`
- `"benefit_measured": false`
- `"predictive_benefit_measured": false`
- `"complete": true`
- `"ready": true`
- `"energy_fit_ready_score": 1`
- `"baseline_ready_score": 1`
- `"principle": "A valid null remains reusable."`
- `"principle": "Completion cannot substitute for empirical value."`
- `"strongest_comparator"`: `"family": "temperature_original"`, `"tune_brier": 0.11930674576400344`
- `"source_contrast_energy"`: `"tune_brier": 0.13570556747592802`
- `"unconstrained_equal_capacity_energy"`: `"tune_brier": 0.13525139808043912`
- `"required_checks_passed": true`

## RECOMMENDATION
KEEP

## experiment_7567_v661_source_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The candidate method `source_contrast_energy` demonstrates no supported probability or decision benefit over the baseline comparator `temperature_original` in source-grounded tool-error evaluation.

## WHAT WOULD REFUTE IT
Statistically significant empirical improvement of `source_contrast_energy` over `temperature_original` on the registered metrics (achieving Brier score improvement $\ge 0.01$ with Holm-adjusted $p < 0.05$, log loss non-regression, and primary cost reduction) resulting in non-zero `probability_benefit_score` and `decision_benefit_score`.

## WAS THAT CHECKED
Yes. Statistical comparison was executed across 80 groups with 2,000 bootstrap draws under `paired_intervals` and `probability_metrics`, while evaluation sensitivity/power was separately verified via the analytical positive control panel in `positive_control_results`.

## EVIDENCE
- `honest_verdict`: `complete_null_source_evaluation_no_supported_benefit`
- `positive_claim`: `false`
- `verdict_class`: `null`
- `probability_benefit_score`: `0`
- `decision_benefit_score`: `0`
- `failed_benefit_gates`: `fresh_confirmatory_claim_forbidden`, `registered_brier_family`, `log_loss_nonregression`, `primary_cost_nonregression`, `primary_cost_improvement`
- `gate_check_summary` -> `failed_checks`: `fresh_confirmatory_claim_allowed`, `probability_benefit`, `decision_benefit`
- `temperature_original` Brier comparison: `delta`: `0.10277710332291774`, `holm_adjusted_p`: `1.0`, `one_sided_p`: `0.9995002498750625`
- `positive_control_results` -> `panel`: `analytical_oracle_defined_separate_from_empirical_rows`, `passed`: `true`, `probability_benefit_score`: `1`, `decision_benefit_score`: `1`

## RECOMMENDATION
KEEP

## experiment_7568_continuous_recalibration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An empirical measurement or comparative finding reported despite failed execution gates, or proof that the upstream prerequisite checks had actually passed.

## WAS THAT CHECKED
No; no experimental execution or comparative evaluation occurred because prerequisite gates failed at the conductor pre-gate layer.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`
`"gate_check_summary": "gate-unsat(final): 2 of 11 gate(s) failed; first failure: exp7561-recalibration-prototype.recalibration_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7569_v661_decision_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Decision learning is blocked by an upstream recalibration gate failure, while source evaluation establishes an honest null result with zero probability or decision benefit over baselines.

## WHAT WOULD REFUTE IT
The blocked-learning and source-null verdict would be refuted if:
1. Upstream producer `exp7561` reported `recalibration_ready_score` equal to `1`, unblocking the learning branch.
2. The candidate source model demonstrated statistically significant probability improvement (lower Brier score or log loss with Holm-adjusted p < 0.05) or lower decision cost compared to `temperature_original` or `raw_original`.
3. The raw reconstruction checks failed or private corruption mutations did not fail closed.

## WAS THAT CHECKED
Yes. Upstream readiness was checked under `learning_producer_available` (observed `blocked_pre_gate` due to upstream `recalibration_ready_score` observing `0` against expected `1`). Comparative benefit was evaluated across 80 source components and 480 rows in `static_reconstruction`, where the candidate underperformed `temperature_original` on Brier score (`delta` of `0.10277710332291774`, `holm_adjusted_p` of `1.0`), yielding observed values of `0` on both `source_probability_benefit` and `source_decision_benefit`. Data integrity and fail-closed behaviors were confirmed via `private_corruptions_fail_closed` (8 of 8 passed) and `source_raw_reconstruction`.

## EVIDENCE
- `honest_verdict`: `complete_blocked_learning_external_source_null`
- `verdict_class`: `blocked`
- `check`: `learning_producer_available`
- `observed`: `blocked_pre_gate`
- `field`: `recalibration_ready_score`
- `observed`: `0`
- `expected`: `1`
- `check`: `source_probability_benefit`
- `observed`: `0`
- `expected`: `1`
- `check`: `source_decision_benefit`
- `observed`: `0`
- `expected`: `1`
- `check`: `learning_effect_and_retention`
- `observed`: `null`
- `expected`: `1`
- `producer_probability_benefit_score`: `0`
- `producer_decision_benefit_score`: `0`
- `source_claims_qualified_score`: `1`
- `learning_claims_qualified_score`: `0`
- `qualified_source_benefit_score`: `0`
- `qualified_learning_benefit_score`: `0`
- `strongest_comparator`: `temperature_original`
- `delta`: `0.10277710332291774`
- `holm_adjusted_p`: `1.0`
- `check`: `private_corruptions_fail_closed`
- `observed`: `8`
- `expected`: `8`
- `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_7570_v661_arc_live_lineage.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7571_v661_portable_calibration.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The task completed in a blocked state because the upstream Exp7561 recalibration prototype was not ready.

## WHAT WOULD REFUTE IT
An upstream `recalibration_ready_score` of 1—or completed kernel-parity work despite that prerequisite—would refute the claimed block.

## WAS THAT CHECKED
Yes. The readiness gate explicitly tested for 1 and observed 0 from the named upstream artifact; the independent reduction also records that parity and service trials never started. No portability or benefit claim was made.

## EVIDENCE
`honest_verdict` `complete_blocked_exp7561_recalibration_ready_score`; `check` `exp7561_recalibration_ready_score`; `expected` `1`; `observed` `0`; `passed` `false`; `verdict_class` `disqualified`; `portable_kernel_ready_score` `0`; `parity_cases_started` `false`; `paired_service_trials_started` `false`; `positive_claim` `false`

## RECOMMENDATION
KEEP

## experiment_10009_b2_induction_gate_measurement_v3.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The owned native-CUDA model feasibly produced nonempty code-only induction responses under the bounded 4,096-token protocol, without claiming gate quality or efficacy.

## WHAT WOULD REFUTE IT
Scoped code-only requests yielding no distinct nonempty responses, or evidence that the model was not actually invoked and GPU-offloaded, would refute the feasibility claim.

## WAS THAT CHECKED
Yes. Durable response files were joined to the scoped code-only requests: 42 responses were counted as nonempty, while invocation and real CUDA offload were separately recorded. The artifact expressly limits publication to feasibility because the efficacy sample floor was not met.

## EVIDENCE
`honest_verdict`: `complete_feasibility_only_sample_floor_not_met`; `publication_mode`: `feasibility_only`; `numeric_gate_quality_claim`: `false`; `hidden_game_efficacy_claim`: `false`; `model_invoked`: `true`; `response_count`: `42`; `content_nonempty_count`: `42`; `all_content_nonempty`: `true`; `offload_real`: `true`; `passed`: `true`; `met`: `false`

## RECOMMENDATION
KEEP

## experiment_7572_v661_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact makes no comparative efficacy or value claim.

## WAS THAT CHECKED
No; not applicable to this read-only aggregation and disposition receipt.

## EVIDENCE
`"positive_claim": false`; `"model_invoked": false`; `"inference_substrate": "aggregation_from_upstream_artifacts"`; `"numbered_runtime_e2e": "not_applicable_read_only_reporting"`; `"honest_verdict": "complete_disqualified_required_v661_evidence"`; `"hardware_benefit_claimed": false`

## RECOMMENDATION
KEEP
