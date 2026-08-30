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
| CLAIM_SUPPORTED | 8 |

## experiment_6788_soft_fixed_point_structural_control_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The grouped fixed-point proposal has a positive paired structural effect relative to the matched flat recurrent control.

## WHAT WOULD REFUTE IT
A grouped-minus-flat exact-valid delta at or below zero, a 95% lower confidence bound at or below zero, or a tie/win by the flat control under matched budgets would refute the claimed structural benefit.

## WAS THAT CHECKED
Yes. The artifact uses paired unit-seed comparisons, an independently evaluated exact-validity outcome, equal candidate and optimization budgets, equal parameter counts, and a substantive recurrent control. The overall and topology-specific confidence intervals exclude zero. The claim does not assert convergence or generalization; those stronger claims are not established here.

## EVIDENCE
`honest_verdict`: `complete: grouped fixed-point proposal has a positive paired structural effect`; `paired_exact_valid_delta`: `0.1020833333`; `lower`: `0.0760416667`; `flat_recurrent_control`: `0.565625`; `grouped_fixed_point`: `0.6677083333`; `held_topology_test`: `0.0854166667`; `lower`: `0.0416666667`; `candidate_budget_by_arm`: `960`; `parameter_counts_by_arm`: `91`; `optimization_steps_by_arm`: `30`; `verifier_is_oracle`: `false`; `evaluated_after_proposal`: `true`; `model_feedback_applied`: `false`; `support_contraction`: `0.0`; `all_rows_attributable`: `true`; `cold_recompute_agreement`: `true`

## RECOMMENDATION
KEEP

## experiment_6789_soft_fixed_point_cold_authority_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The grouped fixed-point method improves exact-valid candidate production over a matched flat recurrent control, with the advantage depending on genuine group and dependency structure.

## WHAT WOULD REFUTE IT
A nonpositive paired exact-validity delta—especially on the held-topology split—or a confidence interval reaching zero; a tie or win by the matched flat recurrent control; survival of the advantage after group, label, or dependency destruction; or proposal-time access to the exact checker would refute the claim.

## WAS THAT CHECKED
Yes. The artifact compares paired arms with matched parameter, update, iteration, and candidate budgets; reports an independently positive held-topology result; freezes candidates before exact checking; and applies dependency rewiring, group-ID permutation, label permutation, and topology-ID swapping. The first three remove or reverse the effect, while renaming topology identifiers preserves it.

## EVIDENCE
The source comparison reports `paired_exact_valid_delta` `0.1020833333`, with `lower` `0.0760416667` and `upper` `0.128125`. On `held_topology_test`, the delta is `0.0854166667`, with `lower` `0.0416666667` and `upper` `0.125`. The serious comparator is `flat_recurrent_control`; its `exact_valid_rate` is `0.565625`, versus `0.6677083333` for `grouped_fixed_point`. `parameter_count_by_arm` is `91` for both arms, `candidate_budget_by_arm` is `960` for both, and `optimizer_updates_by_arm` is `30` for both. Under `dependency_rewire_effect`, `paired_exact_valid_delta` is `-0.05` and `positive_effect_survives` is `false`; under `group_id_permutation_effect`, they are `-0.05625` and `false`; under `label_permutation_effect`, they are `-0.00625` and `false`. Under `topology_id_swap_effect`, `positive_effect_survives` is `true`. `exact_checker_after_candidate_freeze` is `true`, `oracle_feature_violations` is `[]`, and `verifier_is_oracle` is `false`.

## RECOMMENDATION
KEEP

## experiment_6790_chronological_constraint_routing_stream.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The artifact is a completed, reproducible chronological constraint-routing stream with held-future isolation, poison cases, and non-saturated headroom—not evidence of online-learning gain.

## WHAT WOULD REFUTE IT
Replay disagreement, failed readiness gates, future-feature leakage, absent held-future or poison cases, saturated performance with no headroom, incomplete orders, or actions chosen after receipts became visible would falsify the readiness claim.

## WAS THAT CHECKED
Yes. Aggregate gate results, per-order counts, chronology fields, denylist auditing, diagnostic headroom, and fresh-process replay directly allowed those failures to appear. The artifact does not test—and explicitly disclaims—an online-learning gain claim.

## EVIDENCE
`"constraint_routing_stream_ready": true`; `"all_passed": true`; `"failed_checks": []`; `"future_feature_violations": []`; `"agreement": true`; `"fresh_process": true`; `"row_count": 1200`; `"actions_fixed_before_receipt": true`; `"receipt_visible_at_action": false`; `"held_future_first_position": 160`; `"held_future_counts"`; `"order_1": 80`; `"poison_counts"`; `"order_1": 24`; `"accuracy_gap": 0.35`; `"exhaustive_decision_accuracy": 1.0`; `"frozen_decision_accuracy": 0.65`; `"verifier_is_oracle": false`; `"this is not an online-learning gain claim"`.

## RECOMMENDATION
KEEP

## experiment_6791_compositional_online_constraint_routing_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Compositional online routing beat the frozen controller and activity-matched placebo without preregistered harm.

## WHAT WOULD REFUTE IT
A nonpositive held-future utility difference versus either comparator, a nonpositive frozen-comparison lower confidence bound, a serious retrieval-disabled online baseline tying or winning, or any compositional-online harm flag.

## WAS THAT CHECKED
Yes. Paired effects were positive against frozen and placebo in all five orders; the frozen-comparison lower bound was positive; the serious `retrieval_disabled_online` baseline lost on held-future utility in every order; and hard-case and retention harm were checked per order. Current receipts were hidden until after actions, so failure was possible.

## EVIDENCE
`online_minus_frozen_lcb`: `0.34375`. `online_minus_frozen_order_effects`: `0.359375`, `0.359375`, `0.34375`, `0.34375`, `0.34375`. `online_minus_placebo_order_effects`: `0.390625`, `0.3625`, `0.346875`, `0.334375`, `0.33125`. The `compositional_online` held-future `mean_utility` values are `0.578125`, `0.578125`, `0.5625`, `0.5625`, `0.5625`, versus `retrieval_disabled_online` values `0.21875`, `0.234375`, `0.28125`, `0.203125`, `0.20625`. Compositional `harm` is `false` in every reported order. `receipt_visible_at_action` is `false`; `future_feature_violations` is `[]`; `verifier_is_oracle` is `false`.

## RECOMMENDATION
KEEP

## experiment_6792_csl_causal_safety_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The causal-safety audit was blocked and does not support the source verdict because required transaction byte snapshots and hash matches were absent.

## WHAT WOULD REFUTE IT
Finding all 3,189 parent-state and new-state byte snapshots with 3,189 matching byte hashes would refute the stated reason for blocking the audit.

## WAS THAT CHECKED
Yes. The `transaction_byte_snapshots` gate directly compared the required counts with the observed counts and failed.

## EVIDENCE
`csl_causal_audit_completed`: `false`; `source_verdict_supported`: `false`; `verdict_class`: `blocked`; `failed_checks`: `transaction_byte_snapshots`; `committed_receipts`: `3189`; `parent_byte_snapshots`: `0`; `new_state_byte_snapshots`: `0`; `byte_hash_matches`: `0`; `all_passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_6793_temporal_exchange_ising_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Temporal exchange did not pass both the paired-efficiency and target-law gates, and the experiment was limited to CPU simulation.

## WHAT WOULD REFUTE IT
Headline-eligible data showing every paired-efficiency lower confidence bound above zero and target-law noninferiority passing every family would refute the performance claim; evidence that physical hardware was invoked would refute the substrate limitation.

## WAS THAT CHECKED
Yes. Temporal exchange was compared against ordinary Gibbs under matched updates, and both prespecified gates could pass or fail independently. The efficiency gate failed in all three high-temperature strata, while the target-law gate failed for the mixed-sparse family. Stress rows were explicitly ineligible and separate from the headline analysis. Hardware invocation was also recorded.

## EVIDENCE
`honest_verdict`: `complete: temporal exchange did not pass both the paired efficiency and target-law gates; CPU simulation only`

`paired_efficiency_lcb`: `passed`: `false`

`minimum_stratum_lcb`: `-0.00884548058204447`

`target_law_noninferiority`: `passed`: `false`

`mixed_sparse`: `upper_confidence_bound`: `0.030513455239037`

`margin`: `0.03`

`ordinary_gibbs`

`design`: `matched single-site updates with common random numbers`

`schedule_tuned_on_headline_graphs`: `false`

`verifier_is_oracle`: `false`

`headline_fidelity_eligible`: `false`

`stress_rows_separate`: `true`

`physical_hardware_invoked`: `false`

`inference_substrate`: `CPU exact-enumerable Ising simulation, no physical hardware`

## RECOMMENDATION
KEEP

## experiment_6794_temporal_exchange_cold_hardware_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
A fresh-process CPU replay reproduces the source null result while exposing temporal overhead and stationary-law sensitivity, without making a hardware-performance claim.

## WHAT WOULD REFUTE IT
A positive cold gate—consistent efficiency gains over ordinary Gibbs across every stratum while preserving the target law—or replay/hash disagreement with the source would refute the claimed null reproduction.

## WAS THAT CHECKED
Yes. `source_gate_recomputation` tests efficiency and target-law gates across graph/temperature strata; replay, row, trajectory, and exact-target hashes were also checked. The serious baseline `ordinary_gibbs` is present. The gates failed rather than being structurally forced to pass.

## EVIDENCE
`cold_verdict_class`: `null`; `efficiency_gate_passed`: `false`; `target_law_gate_passed`: `false`; `source_all_strata_positive_under_attempted_updates`: `false`; `source_verdict_supported`: `true`; `source_row_hash_invalid_count`: `0`; `trajectory_hash_mismatch_count`: `0`; `exact_target_hash_mismatch_count`: `0`; `verifier_is_oracle`: `false`; `physical_hardware_invoked`: `false`; `status`: `complete_null`.

## RECOMMENDATION
KEEP

## experiment_6795_v592_branch_disposition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V592 supports durable checkpoint readiness and a synthetic, bounded fixed-point advantage, while dispatch and cold CSL causality remain incomplete and temporal exchange is null.

## WHAT WOULD REFUTE IT
A matched flat recurrent control tying or beating grouped fixed point on independently audited held-topology exact-valid rate would refute the positive fixed-point portion.

## WAS THAT CHECKED
Yes, in `fixed_point_disposition.evidence`: the matched control was evaluated with matched candidate work and parameter counts, and the held-topology confidence interval remained above zero.

## EVIDENCE
`held_topology_exact_valid_delta`: `{"lower": 0.0510251308751085, "mean": 0.08541666666666667, "n": 160, "upper": 0.11980820245822484}`

`matched_candidate_work`: `true`

`matched_parameter_counts`: `true`

`evidence_authority`: `independent_audit_rows`

`oracle_leakage_free`: `true`

`cold_causal_audit_passed`: `false`

`efficiency_gate_passed`: `false`

`target_law_gate_passed`: `false`

`pooled_score_computed`: `false`

## RECOMMENDATION
KEEP
