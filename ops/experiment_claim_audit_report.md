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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6840_residual_memory_chronological_shard_a.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6841_residual_memory_delayed_correction_shard_b.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Shard B and its delayed-correction receipts are complete, while verified residual memory yields a null comparative result.

## WHAT WOULD REFUTE IT
Missing planned rows or receipts would refute completeness; a consistent held-future advantage for verified residual memory over the serious no-memory and read-only-memory baselines would refute the null result.

## WAS THAT CHECKED
Yes. Completion was checked against planned row and event counts, while held-future accuracy, residual error, negative transfer, and correction latency were compared across all four arms. Verified residual memory improved residual error but lost on held-future decision accuracy and tied every arm on correction latency, so no unambiguous value advantage appeared.

## EVIDENCE
`honest_verdict` `complete_null_residual_memory_shard_b_rows_and_delayed_correction_receipts_complete`; `verdict_class` `null`; `planned_row_count` `3672`; `total_rows` `3672`; `csl_shard_b_complete_score` `1.0`; `failed_checks` `[]`; `verified_residual_memory` `decision_accuracy` `0.083333`; `no_memory` `decision_accuracy` `0.226852`; `read_only_memory` `decision_accuracy` `0.101852`; `verified_residual_memory` `held_future_mean_residual_error` `0.405093`; `no_memory` `held_future_mean_residual_error` `0.476852`; `mean_correction_latency_events` `2.039216`.

## RECOMMENDATION
KEEP

## experiment_6842_sealed_memory_pathway_portability_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The sealed audit completed and found the memory pathway not ready for continuous self-learning, with readiness score zero.

## WHAT WOULD REFUTE IT
A nonzero readiness score with every conjunctive readiness gate passing, supported by positive held-future performance of verified residual memory over no memory.

## WAS THAT CHECKED
Yes. The artifact reports the conjunctive readiness score, dose-calibration and deletion gates, held-future comparisons, and family/order/seed breakdowns. Outcomes were capable of varying—the random arm recorded both wins and losses—while verified residual memory tied the serious no-memory baseline after deletion.

## EVIDENCE
`honest_verdict`: `complete_null_sealed_memory_audit_complete_ready_score_zero`; `continuous_self_learning_ready_score`: `0.0`; `calibrated_dose_gate_passed`: `false`; `mean_effect_vs_no_memory`: `-0.118519`; `readiness_gate_passed`: `false`; `verified_residual_memory`; `held_future_rows`: `540`; `wins`: `0`; `losses`: `0`; `ties`: `540`; `mean_effect_vs_no_memory`: `0.0`; `random_admission`; `wins`: `30`; `losses`: `42`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_6843_live_arc_evidence_stratum_freeze.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying outcome applies because the artifact asserts only inventory completeness, not solving or mechanism value.

## WAS THAT CHECKED
No comparative refutation was checked or required; only provenance, eligibility counts, source identity, and inventory gates were checked.

## EVIDENCE
`solve_claim`: `false`; `verdict_class`: `"null"`; `inference_substrate`: `"read_only_live_artifact_inventory"`; `honest_verdict`: `"complete_live_arc_inventory_terminal_evidence_only_no_solve_claim"`; `mechanism_effect_claim`: `false`

## RECOMMENDATION
KEEP

## experiment_6844_supervisor_action_outcome_credit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The causal audit is complete, but supervisor-effect inference is blocked because the matched outcomes have no nonzero headroom.

## WHAT WOULD REFUTE IT
At least one same-stratum redirect or control outcome score differing from the others, producing nonzero headroom and effect eligibility, would refute the claimed blockage.

## WAS THAT CHECKED
Yes. The headroom check compared 30 redirects with 30 matched controls across 60 exact later-outcome receipts; all observed scores were zero, and the eligibility gate failed specifically on nonzero headroom.

## EVIDENCE
`"honest_verdict": "complete_blocked_supervisor_outcome_credit_audit"`; `"failed_check": "headroom_nonzero"`; `"observed": false`; `"passed": false`; `"nonzero_headroom": false`; `"rule": "same-stratum redirect/control exact outcome scores are not all equal"`; `"redirect_count": 30`; `"matched_control_count": 30`; `"exact_later_outcome_count": 60`; `"effect_delta": 0.0`; `"effect_eligible": false`; `"supervisor_effect_eligible_score": 0`; `"solve_claim": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6845_tool_gap_causal_support_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify; the blocked conclusion would be contradicted by one or more valid tool-gap obligation rows passing the eligibility gates.

## WAS THAT CHECKED
Yes. The obligation ledger, request-receipt joins, precondition gates, and effect-eligibility score explicitly check this; all show zero eligible observations.

## EVIDENCE
`"solve_claim": false`; `"verdict_class": "blocked"`; `"failed_check": "tool_gap_obligations"`; `"observed": 0`; `"passed": false`; `"row_count": 0`; `"rows": []`; `"joined_count": 0`; `"tool_gap_effect_eligible_score": 0`; `"utility_results": []`

## RECOMMENDATION
KEEP

## experiment_6846_typed_arc_shadow_monitor.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a readiness receipt, it would fail if the hook were unreachable, readiness gates failed, the default armed the monitor, or shadow execution changed returned actions; these would refute readiness and non-interference, but not an effectiveness claim because none is made.

## WAS THAT CHECKED
Yes. Reachability, gate completion, default-off configuration, action identity, external-label agreement, latency, and row validity were checked across 20 rows.

## EVIDENCE
`honest_verdict`: `complete_null_typed_arc_shadow_monitor_ready_default_off_no_solve_claim`; `verdict_class`: `null`; `solve_claim`: `false`; `reachable`: `true`; `default_enabled`: `false`; `monitor_constructed_by_default`: `false`; `all_identical`: `true`; `row_count`: `20`; `passed`: `true`; `invalid_row_hash_count`: `0`; `missing_fields`: `[]`

## RECOMMENDATION
KEEP

## experiment_6847_v598_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s bookkeeping assertion, a source branch state, failed gate, missing artifact, or producer disagreement being omitted or recorded differently in the capstone would refute faithful preservation; there is no comparative method-value claim to falsify.

## WAS THAT CHECKED
Yes, for receipt integrity: source hashes, expected-versus-observed rows, failed checks, producer disagreements, and missing-artifact preservation were recorded. No comparative baseline was checked because no comparative claim was made.

## EVIDENCE
`aggregation_from_upstream_artifacts_no_llm`; `complete_null_v598_independent_capstone_all_branch_states_preserved`; `complete_terminal_null`; `null`; `passed`; `false`; `producer_disagreement_count`; `2`; `missing_artifacts_preserved`; `exp6838`; `Completeness measures branch coverage, not scientific or operational positivity.`

## RECOMMENDATION
KEEP
