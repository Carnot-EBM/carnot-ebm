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
| NO_CLAIM | 5 |

## experiment_6984_exact_contrast_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. If interpreted narrowly as a fixture-completeness receipt, it would be refuted by fewer observed than expected pairs, an unaccepted pair, failed hash replay, authority disagreement, or incomplete label balance.

## WAS THAT CHECKED
Yes, for fixture completeness: the artifact checks expected versus observed pair counts, pair acceptance, hash replay, authority agreement, and label balance. It does not test verifier added value or model generalization, but it does not claim either.

## EVIDENCE
`controlled_fixture_only` `true` `live_extraction_claimed` `false` `inference_substrate` `deterministic_z3_contrast_fixture_no_llm` `verifier_is_oracle` `true` `verdict_class` `circular_positive` `expected_pair_count` `36` `observed_pair_count` `36` `contrast_fixture_complete_score` `1`

## RECOMMENDATION
KEEP

## experiment_6985_chronological_constraint_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative performance claim is made; this fixture would support such a claim only if a later learner beat a serious no-update baseline on unseen chronological events without oracle access.

## WAS THAT CHECKED
No. The artifact constructs and validates a chronological evaluation stream, but reports no learner outputs, baseline comparison, or performance result.

## EVIDENCE
`"continuous_self_learning_fixture": true`; `"True marks the stream as later learning input, not an update."`; `"inference_substrate": "deterministic_z3_chronological_stream_no_llm"`; `"verifier_is_oracle": true`; `"verdict_class": "circular_positive"`; `"A closed class prevents oracle fixture evidence from becoming a learned win."`

## RECOMMENDATION
KEEP

## experiment_6986_three_family_contrast_features.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or predictive claim to falsify; the receipt’s narrower completeness assertion would fail if required candidate-family rows were missing, source counts mismatched, or model execution checks failed.

## WAS THAT CHECKED
No comparative refutation was checked because no verifier was fitted or evaluated; only feature-bank completeness, provenance, and execution integrity were checked.

## EVIDENCE
`complete_three_family_contrast_feature_bank`; `three_family_feature_bank_complete_score`; `Completion measures evidence integrity, not predictive value.`; `verifier_fit_performed`; `False prevents feature collection from becoming hidden verifier selection.`; `expected_feature_row_count`; `414`; `all checks pass`; `live_local_llama_cpp_three_family_teacher_forced_cuda`

## RECOMMENDATION
KEEP

## experiment_6987_contrast_feature_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The contrast feature bank is disqualified because mutation metadata alone predicts the oracle label above the shortcut threshold.

## WHAT WOULD REFUTE IT
Mutation-metadata-only discrimination at or below the 0.8 threshold, producing a passing shortcut gate, would refute the stated reason for disqualification.

## WAS THAT CHECKED
Yes, in `shortcut_interval_rows`; the mutation-metadata probe was evaluated out of fold with source-pair grouping and failed the preregistered threshold, while multiple other shortcut probes passed it.

## EVIDENCE
`honest_verdict`: `complete_disqualified_contrast_feature_bank_shortcut_gate`; `verdict_class`: `disqualified`; `contrast_feature_bank_ready_score`: `0`; `probe_name`: `mutation_metadata_only`; `feature_fields`: `fault_family`, `schedule_id`, `event_type`; `shortcut_auroc`: `0.9532275132275133`; `ci95_lower`: `0.8548580567772891`; `threshold`: `0.8`; `gate_passed`: `false`; `group_overlap_count`: `0`; `preprocessing_fit_on_train_only`: `true`; `verifier_is_oracle`: `true`.

## RECOMMENDATION
KEEP

## experiment_6988_certified_pwa_kan_ranker.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact is a gate-failure receipt and reports no result for the constraint ranker.

## WAS THAT CHECKED
No. The method was blocked before execution at `conductor_pre_gate`, so no comparative or performance claim was tested.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_observed": 0`, `"failed_expected": 1`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6993_arc_producer_evidence_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact asserts only operational evidence-contract completeness, while explicitly declining model-quality, level-completion, solve, and leaderboard claims.

## WAS THAT CHECKED
No substantive claim required checking. The limited operational contract was nevertheless exercised against interruptions, tampering, legacy rows, and prohibited transition sources.

## EVIDENCE
`honest_verdict`: `positive: arc_producer_evidence_contract_complete`; `inference_substrate`: `deterministic_arc_producer_contract_fixture_no_llm`; `model_quality_claimed`: `false`; `level_claimed`: `false`; `solve_claimed`: `false`; `submitted_to_leaderboard`: `false`

## RECOMMENDATION
KEEP

## experiment_6994_arc_producer_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
A fresh-process replay confirms the ARC producer contract’s integrity, routing, and source agreement.

## WHAT WOULD REFUTE IT
An untampered success fixture being rejected, a tampered or interrupted fixture remaining eligible, routing selecting an engine inconsistent with the evidence envelope, source disagreement, or incomplete fixture coverage would falsify the claim.

## WAS THAT CHECKED
Yes—in the eligibility, tamper, atomicity, hash, routing, source-disagreement, and per-fixture replay rows. The negative fixtures could fail individual checks and were required to become ineligible; the success fixture was required to remain eligible.

## EVIDENCE
`arc_producer_contract_confirmed_score`: `1`; `arc_contract_audit_complete_score`: `1`; `fixture_id`: `success`; `eligible`: `true`; `expected_eligible`: `true`; `fixture_id`: `tamper_prompt`; `passed`: `false`; `eligible`: `false`; `failed_checks`: `raw_prompt_sha256`; `fixture_id`: `tamper_engine`; `passed`: `false`; `eligible`: `false`; `failed_checks`: `engine_sha256`; `source_disagreement_rows`: `[]`; `engine_hash_matches_envelope`: `true`; `factory_constructed_e3_policy`: `true`; `verifier_is_oracle`: `false`; `model_quality_claimed`: `false`; `solve_claimed`: `false`; `honest_verdict`: `complete_positive_arc_producer_contract_confirmed`.

## RECOMMENDATION
KEEP

## experiment_6995_v612_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V612 capstone completed, but produced no cold-audited scientific positive.

## WHAT WOULD REFUTE IT
Any branch showing a genuine learned result with science-positive status and its required cold audit confirmed—equivalently, a V612 science-positive score of 1—would falsify the claim.

## WAS THAT CHECKED
Yes. The capstone enumerated each branch, recorded science-positive and cold-audit status, checked expected deliverables, and evaluated the explicit cold-audited-science-positive gate. The oracle-derived positives were correctly denied science credit rather than used as counterexamples.

## EVIDENCE
`honest_verdict` `complete_null_v612_capstone_no_cold_audited_science_positive`; `v612_capstone_complete_score` `1`; `v612_science_positive_score` `0`; `failed_check` `cold_audited_v612_science_positive`; `observed_value` `0`; `passed` `false`; `science_positive` `false`; `required_cold_audit_confirmed` `false`; `verifier_is_oracle` `true`; `science_credit` `false`; `metrics` `null`; `state` `blocked`; `artifact_present` `false`

## RECOMMENDATION
KEEP
