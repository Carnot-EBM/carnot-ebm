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
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7453_energy_calibration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation that the upstream dependency condition (`actual=1 == expected=1` for `embedding_capture_ready_score`) was met or that the experiment proceeded past pre-flight gating would refute the receipt's report of a blocked gate failure; however, because the artifact is purely a pre-execution gate receipt, no comparative or substantive empirical claim is made.

## WAS THAT CHECKED
No; no experimental evaluation or comparison was checked because execution halted immediately at `conductor_pre_gate` with a duration of 0.0 seconds.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `duration_s`: `0.0`
- `blocked_reason`: `actual=0 == expected=1`
- `failed_upstream`: `exp7452-source-embeddings`
- `failed_field`: `embedding_capture_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `gate_check_summary`: `gate-unsat(final): 1 of 6 gate(s) failed; first failure: exp7452-source-embeddings.embedding_capture_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7454_v653_continuous_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Online continuous learning provides insufficient online benefit over baseline models, supporting an honest null verdict and the retirement of mixture construction.

## WHAT WOULD REFUTE IT
The null claim would be refuted if the candidate method demonstrated a statistically significant online loss reduction clearing the protocol threshold—specifically, passing the `upper_log_loss_delta` benefit check, recording an `online_value_score` of 1, and registering zero benefit failures across the prespecified comparisons.

## WAS THAT CHECKED
Yes. Online benefit was evaluated across all 140 planned and completed evaluation units across 753 independent source groups and 12 prespecified contrasts, where the `upper_log_loss_delta` check failed to meet the required benefit threshold.

## EVIDENCE
- `"honest_verdict": "complete_null_insufficient_online_benefit"`
- `"verdict_class": "null"`
- `"status": "complete_null_retired"`
- `"online_value_score": 0`
- `"benefit_failures": [`
- `"upper_log_loss_delta:insufficient_online_benefit"`
- `"required_checks_passed": true`
- `"completed": 140`
- `"independent_source_groups": 753`
- `"mixture_construction_retired": true`
- `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7455_v653_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The decision audit disqualifies the milestone under `complete_disqualified_required_validation` because required validation failed, decision value benefit failed against the equal-weight mixture rival, and upstream static inputs were blocked.

## WHAT WOULD REFUTE IT
The disqualified and null findings would be refuted if:
1. `required_validation` had passed (`observed: true`) along with complete audit verification (`decision_audit_complete == 1`).
2. `learned_mixture` demonstrated statistically significant decision benefit over the serious rival `equal_weight_adaptive_mixture`, achieving an upper paired confidence interval strictly below zero (`primary_log_loss_upper:learned_mixture_minus_equal_weight_adaptive_mixture < 0.0`).
3. The upstream static dependency had met readiness conditions (`embedding_capture_ready_score == 1`), allowing the static branch to execute rather than being pre-gated.

## WAS THAT CHECKED
Yes. All three refutation pathways were checked:
1. `required_validation` was evaluated in `acceptance_gate_results` and `gate_check_summary.failed_required_checks`, observing `false` against expected `true`.
2. Online decision benefit was evaluated against `equal_weight_adaptive_mixture` in `acceptance_gate_results` and `independent_update_replay.paired_confidence_intervals`, observing an upper bound of `0.007041313118560075` (failing `< 0.0`) across 753 groups and 940 replayed updates, where the rival achieved a lower mean log loss (`0.6750019394051517` vs `0.676994305506869`).
3. Static upstream pre-conditions were checked in `preconditions_checked` and `static_audit.gate_check_summary`, observing `embedding_capture_ready_score == 0` against expected `1`.

## EVIDENCE
- `honest_verdict`: `"complete_disqualified_required_validation"`
- `status`: `"complete_disqualified_required_validation"`
- `verdict_class`: `"disqualified"`
- `check`: `"required_validation"`
- `observed`: `false`
- `passed`: `false`
- `failed_required_checks`: `["required_validation"]`
- `check`: `"decision_value_benefit"`
- `expected`: `1`
- `observed`: `0`
- `check`: `"primary_log_loss_upper:learned_mixture_minus_equal_weight_adaptive_mixture"`
- `expected`: `0.0`
- `observed`: `0.007041313118560075`
- `op`: `"<"`
- `primary_benefit_passed`: `false`
- `arm`: `"equal_weight_adaptive_mixture"`
- `log_loss_mean`: `0.6750019394051517`
- `arm`: `"learned_mixture"`
- `log_loss_mean`: `0.676994305506869`
- `disposition`: `"complete_null_reproduced"`
- `honest_verdict`: `"complete_null_insufficient_online_benefit"`
- `disposition`: `"blocked_upstream_pre_gated"`
- `honest_verdict`: `"blocked_gate_check_failed"`
- `capstone_action`: `"retire_mechanism"`
- `mixture_construction_retired`: `true`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7456_v653_extraction_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The extraction audit verified that upstream span capture produced a null result with no established evaluation effect, correctly keeping the development gate closed without promotion.

## WHAT WOULD REFUTE IT
The claim of an honest null with a closed development gate would be refuted if the artifact's own rows recorded an established evaluation effect (`evaluation_effect_established` observing true or a positive `paired_completion_effect`), an unauthorized promotion (`promotion_score` greater than zero), or audit tampering/failure (such as failed checks in `audit_mutation_rows`, `flagged_adversarial` marked true, or `preserved_without_laundering` marked false).

## WAS THAT CHECKED
Yes. It was checked in `gate_check_summary` (which explicitly confirmed `evaluation_effect_established` failed with `observed: false`), `audited_scientific_disposition` (confirming `span_value_score: 0`, `flagged_adversarial: false`, and `preserved_without_laundering: true`), `audit_mutation_rows` (where all six injection attacks passed verification), and `sample_size_budget` (showing evaluation remained unstarted with zero completed evaluation pairs).

## EVIDENCE
- `"honest_verdict": "complete_null_extraction_audit_preserved_development_gate_closed"`
- `"verdict_class": "null"`
- `"promotion_score": 0`
- `"check": "evaluation_effect_established"`
- `"observed": false`
- `"expected": true`
- `"passed": false`
- `"all_passed": false`
- `"failed_checks": ["evaluation_effect_established"]`
- `"flagged_adversarial": false`
- `"preserved_without_laundering": true`
- `"span_value_score": 0`
- `"pairs": 0`
- `"estimate": null`
- `"attempted": 8`
- `"completed": 8`
- `"unstarted": 96`
- `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7457_v653_arc_exposure.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded exposure of the supervisor intervention arm across paired episodes yielded a complete null effect with zero banked progress, justifying no arm promotion.

## WHAT WOULD REFUTE IT
Any applied episode exhibiting post-redirect progress or level advancement—specifically an observed `banked_progress` greater than 0, `resolved_by_levelup` evaluating to true, `terminal_level` or `peak_level` exceeding 0, or positive attribution in `helped` or `helped_sole` for the applied arm—would refute the null claim.

## WAS THAT CHECKED
Yes. In `rows` and `per_game_results`, across four paired applied and shadow episodes covering games `bp35` and `cn04` under seeds 7457001 and 7457002, the applied arm (`drop_goal_bias`) was triggered at action 120 and allowed 60 post-redirect actions (`actions_after_redirect`: 60) to achieve progress. Every applied run recorded 0 banked progress, reached no level advancement, and logged 0 helped outcomes.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_bounded_exposure_no_arm_promotion"`
- `"arm_value_score"`: `0`
- `"automatic_arm_promotion"`: `false`
- `"new_arm_evidence_present"`: `false`
- `"paired_control_present"`: `true`
- `"applied_banked_progress"`: `0`
- `"shadow_banked_progress"`: `0`
- `"banked_progress"`: `0`
- `"peak_level"`: `0`
- `"terminal_level"`: `0`
- `"actions_after_redirect"`: `60`
- `"resolved_by_levelup"`: `false`
- `"helped"`: `0`
- `"solve_provenance"`: `"no_level_reached"`

## RECOMMENDATION
KEEP

## experiment_7458_v653_durable_updates.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Delta journal durable updates do not achieve the required speed advantage or host work reduction over the whole-state baseline, resulting in an honest null outcome.

## WHAT WOULD REFUTE IT
An observed paired total service ratio upper 95% confidence bound below 0.90 (`paired_total_service_ratio_ci95_upper < 0.9`) combined with a residual host fraction of at most 1% (`one_hundred_x_residual_host_fraction <= 0.01`), or rows showing durability/parity violations during fault recovery.

## WAS THAT CHECKED
Yes. Durability and recovery parity were verified across a 10-condition crash fault matrix (`fault_matrix_complete`), and performance was evaluated across 30 paired independent blocks (3,840 updates per arm) against the `whole_state` baseline. The observed upper bound on the service ratio was 1.0047 (failing `< 0.9`) and the residual host fraction was 0.6121 (failing `<= 0.01`), honestly yielding a null result.

## EVIDENCE
- `honest_verdict`: `"complete_null_durable_delta_speed_gate_not_met"`
- `verdict_class`: `"null"`
- `durable_update_value_score`: `0`
- `check`: `"paired_total_service_ratio_ci95_upper"`, `expected`: `0.9`, `observed`: `1.0046567702574078`, `passed`: `false`
- `check`: `"one_hundred_x_residual_host_fraction"`, `expected`: `0.01`, `observed`: `0.6120927319090697`, `passed`: `false`
- `total_service_ratio`: `1.0006514391986738`
- `speed_gate_passed`: `false`
- `completed_independent_units`: `30`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7459_v653_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
GateMate remains blocked by an unchanged physical prerequisite after Exp6559 while KV260 and PolarFire preserve their historical status, yielding a complete null audit with zero new hardware value.

## WHAT WOULD REFUTE IT
The observation of any dated operator-authored physical change receipt (cable, port, power, board, or DirtyJTAG) after Exp6559 that unblocks GateMate, any issued hardware operations, or a non-zero hardware value score.

## WAS THAT CHECKED
Yes. The artifact executed a changed-state evidence search against declared operator provenance files recorded at `results/raw/experiment_7459_v653_board_continuity/gatemate_changed_state_evidence.json`, and explicitly tested the acceptance gates `gatemate_changed_physical_prerequisite` and `new_board_value`.

## EVIDENCE
- `honest_verdict`: `"complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite"`
- `status`: `"complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite"`
- `verdict_class`: `"null"`
- `all_validity_checks_passed`: `true`
- `all_passed`: `false`
- `failed_checks`: `["gatemate_changed_physical_prerequisite", "new_board_value"]`
- `check`: `"gatemate_changed_physical_prerequisite"`, `expected`: `true`, `observed`: `false`, `passed`: `false`
- `check`: `"new_board_value"`, `expected`: `1`, `observed`: `0`, `passed`: `false`
- `accepted_receipt_count`: `0`
- `error`: `"no operator-authored dated GateMate cable, port, power, board, or DirtyJTAG change after Exp6559"`
- `hardware_value_score`: `0`
- `hardware_ready_score`: `0`
- `hardware_operations_issued`: `[]`

## RECOMMENDATION
KEEP

## experiment_7460_v653_capstone.json

**SKIPPED_ALREADY_FLAGGED**
