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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6824_selective_arbiter_cold_row_replay.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Cold row replay supports the positive verdict that selective priority outperforms flat reject/retry under matched budgets while satisfying the stated safety gates.

## WHAT WOULD REFUTE IT
A nonpositive paired progress lower bound, greater harmful-selection or hard-violation rates for selective priority, a false-intervention upper bound above the acceptance limit, or unmatched work budgets would refute the claim.

## WAS THAT CHECKED
Yes. The artifact recomputed all 576 source rows, compared 144 matched units per arm, checked paired progress, false interventions, harmful selections, hard violations, legal support, and budget equality, and included both development and held scenarios.

## EVIDENCE
`"direction": "selective_minus_flat"`, `"estimate": 0.125`, `"lower_bound": 0.0763888888888889`, `"pair_count": 144`, `"selective_priority"`, `"mean": 0.19444444444444445`, `"flat_reject_retry"`, `"mean": 0.06944444444444445`, `"numerator": 0`, `"upper_bound": 0.12064330476584559`, `"false_intervention_upper_limit": 0.2`, `"mismatched_pair_ids": []`, `"matched_fields"`, `"no_harmful_selection_increase": true`, `"zero_accepted_hard_violations": true`, `"all_producer_rows_recomputed": true`, `"expected_identity_count": 576`, `"observed_identity_count": 576`, `"fit_split": "development"`, `"held_count": 24`, `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6825_selective_arbiter_authority_attacks.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent mutations support the hard authority boundary; adoption was not evaluated.

## WHAT WOULD REFUTE IT
An applicable mutation producing a failed attack, an accepted hard violation, forbidden-feature influence, altered safe-action bytes, fail-open row-integrity behavior, or a non-identical fresh-process replay would falsify the bounded claim.

## WAS THAT CHECKED
Yes. The artifact reports applicable-case outcomes across priority, certificate, prohibited-feature, safe-action, and row-integrity attacks, plus fresh-process replay. Every reported attack has zero failures and passes; replay is byte-identical. Non-applicable rows are separated through the applicability field rather than counted as substantive successes.

## EVIDENCE
`"honest_verdict": "complete: independent mutations support the hard authority boundary; adoption was not evaluated"`; `"verifier_is_oracle": false`; `"inference_substrate": "deterministic_verifier_plus_replay (fresh-process deterministic CPU mutation audit, no LLM)"`; `"failed_count": 0`; `"passed": true`; `"accepted_hard_violation": false`; `"influence_detected": false`; `"byte_identity_enforced": true`; `"failed_closed": true`; `"byte_identical": true`; `"adoption_decision": "not_evaluated"`

## RECOMMENDATION
KEEP

## experiment_6826_selective_arbiter_sealed_adoption.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6827_chronological_causal_edge_memory_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The frozen chronological causal-edge memory stream is complete and ready; no learning ran.

## WHAT WOULD REFUTE IT
A required gate failing, expected and observed chronology or row counts differing, sealed fields appearing, or operation/read counts being zero would refute readiness.

## WAS THAT CHECKED
Yes, in `gate_check_summary`; the artifact checks chronology, canonical rows, sealed fields, family isolation, headroom, and operation outcomes. No learning-effect or comparative claim was made, so no rival arm was required.

## EVIDENCE
`honest_verdict` is `complete: frozen chronological causal-edge memory stream is ready; no learning ran`; `status` is `complete_chronological_causal_edge_memory_stream`; `verified_memory_stream_ready` is `true`; `failed_checks` is `[]`; `verifier_is_oracle` is `false`.

## RECOMMENDATION
KEEP

## experiment_6831_v597_evidence_admissibility_contract.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The selective-arbiter authority and causal-edge inputs are procedurally admissible, with no learning performed.

## WHAT WOULD REFUTE IT
A failed admissibility gate, an incomplete or inadmissible source row, mutable or learned-on causal-edge inputs, or use of the flagged Exp6826 receipt as authority would refute the claim.

## WAS THAT CHECKED
Yes. The gate summary permits failure, the stream validation checks immutability and whether learning ran, source rows carry admissibility/completeness flags, and the flagged receipt’s authority disposition is recorded.

## EVIDENCE
`"failed_check": null`; `"passed": true`; `"csl_inputs_admissible": true`; `"selective_arbiter_receipt_admissible": true`; `"learning_ran": false`; `"weights_immutable": true`; `"row_count": 4320`; `"admissible": true`; `"complete": true`; `"authority_consumed": false`; `"disposition": "quarantined_comparator_only"`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6832_operational_obligation_saturation_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The operational readiness receipt would fail if any readiness gate failed, checker mutations behaved unexpectedly, leakage appeared, or required legal-action headroom was absent; none would refute a model-value claim because no model-value claim is made.

## WAS THAT CHECKED
Yes. The artifact reports readiness gates in `gate_check_summary`, mutation behavior in `checker_mutation_results`, leakage checks in `leakage_audit`, and headroom checks in `legal_action_headroom`.

## EVIDENCE
`honest_verdict`: `complete_operational_obligation_saturation_fixture: deterministic source-free fixture ready; no model ran`; `inference_substrate`: `deterministic CPU fixture generation, no LLM`; `verdict_class`: `null`; `operational_saturation_fixture_ready`: `true`; `failed_check`: `null`; `passed`: `true`; `all_expected`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6833_sota_operational_obligation_saturation_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. For the corpus-readiness assertion, missing, duplicate, or invalid rows; failed budget parity; incomplete checkpoints; or unauthenticated inference would refute readiness.

## WAS THAT CHECKED
Yes, for readiness: the artifact checks row coverage, validation errors, budget parity, checkpoint completeness, and process authenticity. It explicitly reserves inference for a later experiment, so no superiority claim is tested here.

## EVIDENCE
`descriptive_only_exp6834_owns_inference`; `A closed class prevents readiness from becoming an inferential claim.`; `operational_saturation_corpus_ready`: `true`; `passed`: `true`; `row_validation_errors`: `[]`; `rows`: `900`; `duplicates`: `[]`; `expected`: `900`; `observed_unique`: `900`

## RECOMMENDATION
KEEP

## experiment_6834_operational_saturation_identifiability_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The observed support does not identify binary operational-field preservation across every finite fixture cell under unrestricted binary finite-field policies.

## WHAT WOULD REFUTE IT
Observed support that separates all compatible target policies—meaning no two policies agree on every observed cell while differing on any target cell—would refute the claim.

## WAS THAT CHECKED
Yes. The audit explicitly allowed either a separated or not-separated disposition, evaluated all missing binary coordinates, and produced constructive collision witnesses with identical observed signatures but different target values.

## EVIDENCE
`honest_verdict`: `complete_null_operational_field_preservation_not_identified`; `identifiability_disposition`: `["separated", "not_separated"]`; `disposition`: `not_separated`; `policy_class`: `unrestricted binary finite-field policies`; `observed_cell_count`: `395`; `target_cell_count`: `18900`; `missing_cell_count`: `18505`; `compatible_policy_count`: `2^18505`; `observed_signature_left`: `sha256:3f1c9c5701bc07ac92d62b93d6c791f96e9525760ee4a32487c847b57b3626f9`; `observed_signature_right`: `sha256:3f1c9c5701bc07ac92d62b93d6c791f96e9525760ee4a32487c847b57b3626f9`; `target_value_left`: `0`; `target_value_right`: `1`; `coverage_valid`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
