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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7464_v654_semif_e6_decision_cost_profile.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The decision cost profile is a complete null due to sample-limited observations and unattributed trace time, failing acceptance gate checks and halting queue authorization for downstream rungs.

## WHAT WOULD REFUTE IT
The null claim would be refuted if the upstream traces showed high attributed execution time for directly replaceable decisions, a positive lower bound on replaceable share, sufficient game clusters, and complete candidate set snapshots with start and end timestamps across decision seams, resulting in passing all gate checks (`gate_check_summary.all_passed = true`).

## WAS THAT CHECKED
Yes. It was checked across all 8 independent episodes and their constituent decision seam rows in `episode_summaries`, `raw_unit_rows`, `independent_reduction`, and `gate_check_summary`. Only 10.3% of trace time was attributed, leaving 89.7% unclassified residual time; candidate sets were absent; and 4 acceptance checks failed, confirming the null.

## EVIDENCE
- `"honest_verdict": "complete_null_sample_limited_decision_cost_unattributed"`
- `"all_passed": false`
- `"failed_checks": [ "episode_count", "game_cluster_count", "trace_time_attribution", "positive_lower_replaceable_share" ]`
- `"trace_time_fraction": 0.10312798503228131`
- `"unclassified_residual_s": 1338.4132452930085`
- `"total_episode_s": 1492.312418`
- `"lower": 0.0`
- `"lower_assignment": "unclassified_time_is_nonreplaceable"`
- `"downstream_generation_directly_replaceable": false`
- `"e6_gate_passed": false`
- `"queue_authorized": false`
- `"decision": "insufficient_evidence"`
- `"reason": "E6 lacks candidate sets and sufficient attributed trace time"`
- `"broad_transfer_claim_allowed": false`
- `"sample_limited": true`

## RECOMMENDATION
KEEP

## experiment_7465_source_option_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is an operational receipt recording a pre-execution gate failure; it asserts no substantive empirical or comparative claim to falsify.

## WAS THAT CHECKED
No. The run was aborted at the conductor pre-gate before execution, so no experimental evaluation took place.

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

## RECOMMENDATION
KEEP

## experiment_7467_v654_factual_span_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The factual span canary failed the development gate due to insufficient usable outputs, keeping evaluation capture closed with a complete null verdict that retires the construction.

## WHAT WOULD REFUTE IT
The claim of development-gate failure and null closure would be refuted if the canary trials demonstrated at least 3 usable factual extractions per arm (with `usable_by_arm` meeting or exceeding `required_usable_per_arm`), which would have caused `passed: true` and opened evaluation capture (`capture_open: true`).

## WAS THAT CHECKED
Yes; checked in `development_gate` and `development_rows`. Twelve canary calls were executed against the model, observing 5 malformed responses and resulting in only 1 usable output for `span` and 2 for `verbatim` against the required threshold of 3 per arm. This failed the factual gate, kept evaluation unstarted, and closed the run as a null.

## EVIDENCE
- `honest_verdict`: `"complete_null_factual_span_canary_development_gate_closed"`
- `capture_open`: `false`
- `passed`: `false`
- `required_usable_per_arm`: `3`
- `usable_by_arm`: `{"span": 1, "verbatim": 2}`
- `disposition_counts`: `{"correct_empty": 4, "malformed": 5, "usable_nonempty": 3}`
- `retired_by_current_result`: `true`

## RECOMMENDATION
KEEP

## experiment_7468_v654_residual_learner.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The residual learner is verified and ready for deployment based on analytic fixtures.

## WHAT WOULD REFUTE IT
Refutation would require observing performance degradation or a failure to outperform comparator arms (`frozen_residual`, `matched_linear_residual`) on the task evaluation rows (`source_support_binary_decision`), an observed failure on real source-support benefit, or failing validation under an independent verifier where the verifier is not the oracle.

## WAS THAT CHECKED
No. All 80 planned evaluation rows were left unstarted (`attempted`: 0, `unstarted`: 80), empirical benefit was explicitly bypassed (`expected`: `"not_tested"`, `observed`: `"not_tested"`), and the completed checks were restricted to synthetic analytic fixtures where the verifier was the oracle (`verifier_is_oracle`: true).

## EVIDENCE
- `"honest_verdict": "complete_circular_positive_analytic_residual_learner_ready"`
- `"status": "complete_circular_positive_analytic_residual_learner_ready"`
- `"verdict_class": "circular_positive"`
- `"verifier_is_oracle": true`
- `"residual_learner_ready_score": 1`
- `"check": "real_source_support_benefit"`
- `"expected": "not_tested"`
- `"observed": "not_tested"`
- `"attempted": 0`
- `"unstarted": 80`
- `"methodology_note": "Analytic fixtures establish implementation evidence only. They do not measure real source-support value, so a positive verdict is forbidden."`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7470_v654_independent_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent audit yields an honest null verdict across all evaluated branches, finding two branches blocked due to missing producer evidence and the single available extraction branch showing no scientific benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted if any evaluated branch demonstrated verifiable improvement passing the benefit acceptance criteria, if missing producer artifacts were observed with valid checksums, if non-zero promotion scores were recorded, or if the audit failed to catch synthetic mutations injecting fabricated improvements or label leakage.

## WAS THAT CHECKED
Yes. In `acceptance_gate_results` and `gate_check_summary`, the audit checked for expected improvement across candidate branches and recorded failures for missing branches. In `branch_rows` and `extraction_audit`, candidate tasks were evaluated and verified to have zero promotion score and preserved null status. In `mutation_rows`, six corruptions—including fabricated improvement and label leakage—were explicitly tested and caught.

## EVIDENCE
- `honest_verdict`: `complete_null_independent_audit_two_missing_branches_extraction_null`
- `gate_check_summary`: `all_passed`: `false`, `failed_count`: `2`
- `failed_checks`: `typed_decision_benefit`, `residual_learning_benefit`
- `expected`: `measured_improvement`
- `observed`: `not_evaluated_missing_branch`
- `promotion_score`: `0`
- `branch`: `typed_decision`, `availability`: `missing`, `verdict_class`: `blocked`
- `branch`: `residual_learning`, `availability`: `missing`, `verdict_class`: `blocked`
- `branch`: `extraction`, `availability`: `available`, `verdict_class`: `null`
- `honest_verdict`: `complete_null_extraction_audit_preserved`
- `check`: `mutations_rejected`, `expected`: `6`, `observed`: `6`
- `mutation`: `fabricated_improvement`, `caught`: `true`
- `mutation`: `leaked_labels`, `caught`: `true`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7471_v654_arc_seam_observation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Live inference with Qwen3.8-27B on the ARC seam benchmark produced a complete null observation with zero level progress across all tested games.

## WHAT WOULD REFUTE IT
Any episode achieving level progress or positive credit (such as `progressed` being true, `peak_level` exceeding 0, or `new_level_credit` being greater than 0) under the live generation protocol.

## WAS THAT CHECKED
Yes; refutation was given a real chance across 8 completed episodes (2 seeds each across 4 independent games: `wa30`, `ar25`, `sp80`, and `su15`) with 16 live generation calls dispatched to the local server and 180 environment actions executed per episode, all ending with zero progress.

## EVIDENCE
- `"honest_verdict": "complete_null_live_arc_seam_observation"`
- `"hidden_game_efficacy_claim": false`
- `"new_level_credit": 0`
- `"arc_observation_complete_score": 1`
- `"generation_calls_completed": 16`
- `"any_reproduced_progress": false`
- `"progressed": false`
- `"peak_level": 0`
- `"solve_provenance": "no_level_reached"`

## RECOMMENDATION
KEEP

## experiment_7473_v654_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
GateMate hardware execution remains blocked due to unchanged physical prerequisites with zero new hardware operations issued, while historical continuity for KV260 and PolarFire is preserved.

## WHAT WOULD REFUTE IT
The headline null claim would be refuted if:
1. An operator-authored physical change receipt newer than Exp6559 (cutoff 20260823) existed matching the eligibility contract, resulting in `accepted_receipt_count >= 1` and passing the `gatemate_changed_physical_prerequisite` gate.
2. Hardware operations were issued during this read-only audit (`hardware_operation_count > 0` or non-empty `hardware_operations_issued`), causing the `zero_current_hardware_operations` check to fail.
3. Upstream source authentication or historical continuity broke (e.g., mismatched SHA-256 hashes, unauthenticated venues, or ungrounded FPGA claims).

## WAS THAT CHECKED
Yes. The search artifact at `results/raw/experiment_7473_v654_board_continuity/gatemate_changed_state_evidence.json` checked declared provenance paths (`ops/known-issues.md`, `research-hardware-wishlist.md`, `ops/hardware-bringup-prep.md`, `ops/operator-followup.md`) for physical change receipts dated after 20260823 and found none (`accepted_receipt_count` was 0). The acceptance gate `gatemate_changed_physical_prerequisite` evaluated this condition and failed as expected. Furthermore, `zero_current_hardware_operations` and all validity preconditions were checked and passed.

## EVIDENCE
- `honest_verdict`: `complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite`
- `status`: `complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite`
- `verdict_class`: `null`
- `all_validity_checks_passed`: `true`
- `check`: `gatemate_changed_physical_prerequisite`
- `field`: `accepted_receipt_count`
- `expected`: `1`
- `observed`: `0`
- `passed`: `false`
- `check`: `zero_current_hardware_operations`
- `hardware_operation_count`: `0`
- `hardware_operations_issued`: `[]`
- `board`: `GateMate`
- `disposition`: `blocked`
- `current_disposition`: `blocked_unchanged_physical_prerequisite`
- `exists`: `false`
- `board`: `KV260`
- `current_disposition`: `graduated_preserved`
- `board`: `PolarFire`
- `current_disposition`: `graduated_cpu_dispatch_preserved`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7474_v654_capstone.json

**SKIPPED_ALREADY_FLAGGED**
