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
| CLAIM_SUPPORTED | 1 |
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_6853_risk_sensitive_memory_opportunity_fixture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6854_risk_sensitive_abstention_memory_controller.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
The risk-sensitive contextual-bandit controller provides positive held-future benefit while bounding false-positive memory injection.

## WHAT WOULD REFUTE IT
The controller tying a fixed always-abstain policy on held-future loss, overall loss, and every action—showing that the controller adds no value beyond unconditional abstention—would refute the claimed controller benefit.

## WAS THAT CHECKED
Yes. The `per_arm_summary` directly compares `contextual_bandit` with `abstain_only`, and they tie exactly; the controller also abstained on all 765 decisions.

## EVIDENCE
`honest_verdict`: `complete_positive_risk_sensitive_controller_benefit_gate_passed`; `contextual_bandit`; `abstain_only`; `abstention_count`: `765`; `decision_count`: `765`; `held_future_mean_loss`: `0.254901960784`; `mean_loss`: `0.277777777778`; `total_loss`: `212.5`; `abstention_rate`: `1.0`; `helpful_memory_selection_count`: `0`; `chosen_action`: `abstain`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_6855_counterfactual_memory_credit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed counterfactual audit finds no learned selection advantage over placebo context, while identifying harmful memory writes.

## WHAT WOULD REFUTE IT
A nonzero learned-selection-minus-placebo reward effect would refute the null selection claim; zero supported negative per-write effects would refute the claim that harmful writes are present.

## WAS THAT CHECKED
Yes. The matched placebo comparison reports zero effect, while eligible per-write and coalition analyses include both positive and negative effects and count harmful writes. Unsupported counterfactuals are separately marked ineligible.

## EVIDENCE
`honest_verdict` `complete_null_counterfactual_memory_credit_harmful_writes_present` `comparison` `learned_selection_minus_placebo_context` `mean_reward_effect` `0.0` `learned_selection` `placebo_context` `mean_reward` `-0.277777777778` `harmful_write_count` `18` `aggregate_coalition_credit` `-0.283333333334` `benefit_eligible` `true` `causal_credit_eligible` `true` `credit_class` `interaction_only` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_6856_sealed_risk_sensitive_learning_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6857_dynamic_live_arc_receipt_router.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6858_supervisor_counterfactual_credit_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made; a passed prerequisite gate or completed counterfactual audit would contradict only the reported blocked status.

## WAS THAT CHECKED
No—the method was never evaluated; only the upstream prerequisite gate was checked.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"actual": 0`, `"expected": 1`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6859_first_party_tool_gap_receipt_wiring.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation could refute a comparative or live-effect claim because the artifact explicitly makes none; such a claim would require authentic-live rows showing response use, valid headroom, and outcomes against a serious baseline.

## WAS THAT CHECKED
No. This is a receipt-wiring artifact using fixtures, with zero live-effect-eligible rows and no comparator arm.

## EVIDENCE
`honest_verdict`: `complete_first_party_tool_gap_receipt_contract_ready_no_live_effect_claim`; `solve_claim`: `false`; `solve_claimed`: `false`; `game_level_solve_count`: `0`; `tool_gap_live_effect_claim_eligible_score`: `0`; `provenance_class`: `authentic_live`; `row_count`: `0`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_6860_v599_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific value claim to falsify. Treating this as an administrative receipt, a declared/recomputed mismatch, source-hash mismatch, or advancement of a failed or disqualified branch would refute its disposition-preservation assertion.

## WAS THAT CHECKED
Yes. The aggregate consistency and source-link hash checks could fail, and the gate checks genuinely did fail without being promoted into scientific success.

## EVIDENCE
`honest_verdict` `complete_partial_v599_dispositions_preserved_no_scientific_branch_advance` `solve_claimed` `false` `hardware_speedup_claimed` `false` `game_level_solve_count` `0` `gate_check_summary` `passed` `false` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP
