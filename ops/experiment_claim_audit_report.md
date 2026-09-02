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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6857_dynamic_live_arc_receipt_router.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6858_supervisor_counterfactual_credit_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and makes no substantive comparative or value claim. The blocking assertion itself would be refuted by an observed gate value of 1.

## WAS THAT CHECKED
Yes, but only the prerequisite gate: `supervisor_headroom_ready_score` was checked against 1 and observed as 0. No counterfactual credit audit was run.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `failed_field` `supervisor_headroom_ready_score` `failed_expected` `1` `failed_observed` `0` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6859_first_party_tool_gap_receipt_wiring.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no comparative, solve, generalization, or live-effect claim; a live-effect refutation would require authentic-live, valid-headroom rows showing no benefit or harm.

## WAS THAT CHECKED
No. Live effect was deliberately not tested; only receipt wiring and contract readiness were checked.

## EVIDENCE
`honest_verdict` `complete_first_party_tool_gap_receipt_contract_ready_no_live_effect_claim` `solve_claim` `false` `solve_claimed` `false` `verdict_class` `null` `game_level_solve_count` `0` `tool_gap_live_effect_claim_eligible_score` `0` `provenance_class` `fixture` `valid_headroom` `false`

## RECOMMENDATION
KEEP

## experiment_6860_v599_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative headline to falsify. The narrower administrative completeness statement would be refuted by a planned task lacking either a terminal disposition or an explicit skip, or by a declared disposition disagreeing with its independent recomputation.

## WAS THAT CHECKED
Yes. The artifact checks task and skip manifests, source hashes, aggregate-row consistency, and branch gates; it also preserves blocked, null, partial, and disqualified outcomes rather than converting them into scientific success.

## EVIDENCE
`honest_verdict` = `complete_partial_v599_dispositions_preserved_no_scientific_branch_advance`; `solve_claimed` = `false`; `hardware_speedup_claimed` = `false`; `game_level_solve_count` = `0`; `verdict_class` = `Summarizes milestone evidence completeness, not an ARC solve or hardware result.`; `gate_check_summary` has `passed` = `false`; `typed_compatibility_disposition` has `scientific_result` = `not_identifiable`; `continuous_self_learning_disposition` has `verdict_class` = `disqualified`; `arc_supervisor_disposition` has `verdict_class` = `blocked`.

## RECOMMENDATION
KEEP

## experiment_6861_v600_branch_retirement_evidence_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative claim to falsify; the narrower procedural-readiness assertion would be refuted by a missing prerequisite, source-hash mismatch, invalid replacement mechanism, unverified reference, or unnamed downstream gate consumer.

## WAS THAT CHECKED
Yes, for procedural readiness: the artifact checks source preconditions, mechanism validity, reference verification, downstream consumers, and source hashes. It does not test the future mechanisms’ scientific effects, nor claim that it does.

## EVIDENCE
`honest_verdict`: `complete_positive_v600_branch_retirement_evidence_contract_ready`; `scope`: `procedural_only`; `scientific_branch_advance_count`: `0`; `scientific_claim_eligible`: `false`; `observed`: `all checks pass`; `failed_checks`: `[]`; `v600_evidence_contract_ready_score`: `1`

## RECOMMENDATION
KEEP

## experiment_6862_dual_side_semantic_contrast_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no model-performance or comparative claim to falsify; the limited readiness assertion would fail if accepted groups missed the 96-group floor, accepted rows showed authority disagreement, or single-atom mutations failed to change both labels.

## WAS THAT CHECKED
Yes. The artifact checks the group-count floor, dual-authority agreement, mutation label changes, nuisance-transform invariance, and rejection of checker disagreements; however, it contains no model scores or comparator evaluation.

## EVIDENCE
`honest_verdict` `complete_null_dual_side_semantic_contrast_bank_ready_no_model_scores` `verdict_class` `null` `model_scores_present` `false` `accepted_contrast_group_count` `100` `dual_side_semantic_contrast_bank_ready_score` `1` `both_authorities_passed` `true` `solution_label_changed` `true` `structure_label_changed` `true` `checker_disagreement` `rejected` `true` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no scientific-effect or comparative-value claim; it records a blocked preregistration gate.

## WAS THAT CHECKED
No scientific hypothesis was tested: there are no score rows, no likelihood calls, and no accepted cells.

## EVIDENCE
`scientific_effect_claimed` `false`; `verdict_class` `blocked`; `honest_verdict` `complete_blocked_tokenizer_aware_semantic_contrast_preregistration`; `rows` `[]`; `token_likelihood_call_count` `0`; `accepted_cell_manifest` `[]`; `models_used` `[]`; `semantic_contrast_preregistration_ready_score` `0`; `gate_check_summary` `passed` `false`; `tokenizer_hash_drift`

## RECOMMENDATION
KEEP

## experiment_6864_three_family_semantic_contrast_scoring_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and reports no semantic-scoring result or comparative claim to falsify.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate`, so no method outcomes or rival comparisons were produced.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "semantic_contrast_preregistration_ready_score"`, `"failed_expected": 1`, `"failed_observed": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
