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
| NO_CLAIM | 2 |

## experiment_6995_v612_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V612 capstone is complete but contains no cold-audited scientific positive.

## WHAT WOULD REFUTE IT
Any eligible branch showing both a positive learned result and its required cold audit confirmed, producing a science-positive score of 1, would refute the claim.

## WAS THAT CHECKED
Yes. The capstone’s explicit cold-audited science-positive gate checked all 12 terminal tasks and failed with an observed score of 0; branch rows also record absent cold-audit confirmation and no science positive. Circular oracle-derived positives were correctly denied science credit.

## EVIDENCE
`honest_verdict` `complete_null_v612_capstone_no_cold_audited_science_positive` `v612_capstone_complete_score` `1` `v612_science_positive_score` `0` `failed_check` `cold_audited_v612_science_positive` `expected_value` `1` `observed_value` `0` `passed` `false` `terminal_task_count` `12` `science_positive` `false` `required_cold_audit_confirmed` `false` `verifier_is_oracle` `true` `science_credit` `false` `verdict_class` `null`

## RECOMMENDATION
KEEP

## experiment_6996_v613_source_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All requested V613 source routes produced terminal receipts, and the 13-task Markdown/YAML contract and gate producers conform.

## WHAT WOULD REFUTE IT
A requested source route lacking a terminal receipt, or any mismatch in task count, order, identity, deliverable, title, gate contract, or gate producer resolution would falsify the claim.

## WAS THAT CHECKED
Yes. Terminal status was recorded per source route, while contract parity and gate resolution were checked per task and summarized by the two completion scores. This supports receipt completeness, not successful retrieval from every source or literature-search exhaustiveness.

## EVIDENCE
`v613_source_delta_complete_score` is `1`; `v613_task_contract_conforms_score` is `1`; `expected_task_count` and `observed_task_count` are both `13`; `failed_check` is `null`; `verifier_is_oracle` is `false`. The artifact also openly records `outcome` as `http_error` while `terminal` is `true`, confirming that source completeness means terminal receipts rather than universal retrieval success.

## RECOMMENDATION
KEEP

## experiment_6997_authority_sidecar_rebuild.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the limited completion receipt, fewer than 138 candidates or 414 source rows, a failed gate, incomplete family views, or learner access to authority sidecars would refute successful rebuilding.

## WAS THAT CHECKED
Yes. Candidate and source counts, gate results, family completeness, file-access boundaries, and terminal candidate rows were checked. No comparative performance or verifier-value claim was made.

## EVIDENCE
`honest_verdict` `circular_positive: authority_sidecar_rebuild_complete` `expected_candidate_count` `138` `observed_candidate_count` `138` `expected_source_row_count` `414` `observed_source_row_count` `414` `observed_value` `all checks pass` `sidecar_rebuild_complete_score` `1` `gguf_inference_performed` `false` `verifier_fit_performed` `false` `verifier_is_oracle` `true` `deterministic_feature_sidecar_transform_no_llm`

## RECOMMENDATION
KEEP

## experiment_6998_three_family_commitment_controls.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Across three model families, shortcut commitment from true provenance hints was not detected.

## WHAT WOULD REFUTE IT
A reliably negative true-provenance-hint-minus-clean commitment-latency effect—ideally with its entire confidence interval below zero—while the permuted-decoy control did not show the same effect.

## WAS THAT CHECKED
Yes. The artifact reports paired bootstrap intervals across all families and separately by model family; none of the true-provenance intervals is entirely below zero.

## EVIDENCE
`honest_verdict`: `complete_null_shortcut_commitment_not_detected`; `comparison`: `true_provenance_hint_minus_clean`; `mean`: `-0.027777777777777776`; `ci_low`: `-0.0798611111111111`; `ci_high`: `0.024305555555555556`; `sample_count`: `36`; `comparison`: `permuted_decoy_hint_minus_clean`; `mean`: `0.017361111111111112`; `ci_low`: `-0.027777777777777776`; `ci_high`: `0.0625`; `verifier_is_oracle`: `false`; `learner_feature_allowed`: `false`; `verifier_fit_performed`: `False`; `joined_after_all_model_processes_exit`: `true`.

## RECOMMENDATION
KEEP

## experiment_6999_blinded_feature_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The blinded feature bank is disqualified because at least one prohibited shortcut exceeds the readiness threshold.

## WHAT WOULD REFUTE IT
Every prohibited shortcut probe passing the preregistered gate, particularly the mutation-metadata probe having its shortcut AUROC and confidence interval below the 0.8 threshold, would refute the disqualification.

## WAS THAT CHECKED
Yes. The artifact evaluates prohibited shortcut probes against the explicit threshold, and the mutation-metadata probe fails with even its confidence-interval lower bound above that threshold.

## EVIDENCE
`honest_verdict`: `complete_disqualified_blinded_feature_shortcut_gate`; `blinded_feature_bank_ready_score`: `0`; `probe_name`: `mutation_metadata_only`; `prohibited`: `true`; `shortcut_auroc`: `0.9540740740740741`; `ci95_lower`: `0.859375`; `threshold`: `0.8`; `gate_passed`: `false`; `verdict_class`: `disqualified`

## RECOMMENDATION
KEEP

## experiment_7000_certified_blinded_pwa_kan.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive claim to falsify; any implied ranking-value claim would require the method to tie or lose against a serious baseline on valid blinded held-out rows.

## WAS THAT CHECKED
No. The experiment was blocked before execution at the conductor pre-gate, so no method or comparator results were produced.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7005_arc_live_envelope_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The live-envelope engine does not beat both controls on held-out transition prediction.

## WHAT WOULD REFUTE IT
Positive paired error improvement over both frozen controls, with each 95% interval entirely above zero and no loss of transition coverage.

## WAS THAT CHECKED
Yes. Held-out paired metrics and grouped bootstrap intervals compare the engine against both the inert control and the serious pre-engine observed-action-delta baseline. The inert-control interval crosses zero, so the refutation did not occur.

## EVIDENCE
`honest_verdict`: `complete_null_arc_live_envelope_engine_does_not_beat_both_controls`; `arc_engine_quality_positive_score`: `0`; `control`: `inert_no_change`; `interval_low`: `-0.0001220703125`; `interval_high`: `0.000244140625`; `control`: `observed_action_delta_hypothesis`; `interval_low`: `0.000244140625`; `interval_high`: `0.00152587890625`; `stratum`: `heldout`; `transition_coverage`: `1.0`; `verifier_is_oracle`: `false`; `development_fixture_used_for_quality`: `false`.

## RECOMMENDATION
KEEP

## experiment_7008_v613_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed V613 capstone contains no eligible oracle-distinct science-positive result.

## WHAT WOULD REFUTE IT
An eligible, noncircular oracle-distinct selection result with `science_positive` true, a confirmed certificate, and a nonzero capstone science-positive score would falsify the claim.

## WAS THAT CHECKED
Yes. The capstone explicitly checked the oracle-distinct selection gate, science qualification, eligibility, and circularity classifications. The relevant experiment was missing, so this supports only the narrow inventory claim that no positive result exists—not a broader claim that the method was empirically shown to fail.

## EVIDENCE
`honest_verdict`: `complete_null_v613_capstone_no_oracle_distinct_science_positive`; `v613_science_positive_score`: `0`; `failed_check`: `oracle_distinct_selection_positive`; `observed_value`: `0`; `selection_positive_score`: `0`; `qualified_science_positive_score`: `0`; `certificate_confirmed_score`: `0`; `artifact_present`: `false`; `eligible_for_positive`: `false`; `state`: `missing`; `verifier_is_oracle`: `true`; `science_positive`: `false`; `verdict_class`: `null`.

## RECOMMENDATION
KEEP
