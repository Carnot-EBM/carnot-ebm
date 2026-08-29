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
| NO_CLAIM | 6 |
| CANNOT_DETERMINE | 1 |

## experiment_6484_non_generation_representation_receipt_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or model-quality claim to falsify; the receipt-status assertion would fail if required rows were invalid or absent, a prohibited generation call occurred, or a mutation attack was falsely accepted.

## WAS THAT CHECKED
Yes, for receipt readiness: aggregate row checks, generation-call receipts, and the mutation attack matrix checked those failure conditions. No model-quality hypothesis was tested or asserted.

## EVIDENCE
`honest_verdict`: `complete: non-generation representation receipt contract is ready; no model-quality claim is made`; `inference_substrate`: `deterministic_representation_contract_no_llm`; `model_loaded`: `false`; `false_accept_count`: `0`; `generation_call_count`: `0`; `failed_checks`: `[]`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6487_representation_integrity_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The representation stream is disqualified because three shortcut features predict correctness and three required length-based controls cannot be evaluated.

## WHAT WOULD REFUTE IT
No shortcut exceeding its stated ceiling, together with complete raw-row token, candidate, and prompt lengths enabling all required controls, would refute the disqualification.

## WAS THAT CHECKED
Yes. Shortcut-control rows tested predictive nuisance features, and the missing-gap audit checked whether the three length controls were selectable. The refuting outcome did not occur. The oracle flag does not make this negative integrity finding a circular claim of verifier value.

## EVIDENCE
`honest_verdict` is `disqualified: candidate_identifier_length,candidate_identity,row_order_modulo_pair,token_length_unavailable_from_raw_rows,candidate_length_unavailable_from_raw_rows,prompt_length_unavailable_from_raw_rows`. `surviving_shortcut_count` is `9`. `surviving_shortcuts` includes `candidate_identifier_length`, `candidate_identity`, and `row_order_modulo_pair`. For `candidate_identifier_length`, `balanced_accuracy` is `1.0`, `ceiling` is `0.75`, and `survived_shortcut` is `true`. Each listed length gap has `blocks_readiness` set to `true`, `selectable` set to `false`, and `missing_count` set to `432`. `representation_integrity_ready_score` is `0.0`, and `status` is `disqualified`.

## RECOMMENDATION
KEEP

## experiment_6491_sota_factor_proposal_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. Treating the receipt statement as an operational assertion, it would be refuted by a missing event–model receipt, an incomplete cross-product, a one-shot violation, missing raw bytes, or absent exact-compilation disposition.

## WAS THAT CHECKED
Yes. The aggregate recomputation checks model-family completion, event–model coverage, one-shot violations, raw-receipt completeness, and compilation outcomes. These checks could have failed, but they establish execution and receipt integrity—not model usefulness or successful factor generation.

## EVIDENCE
`"completed_model_family_count": 2`, `"event_model_cross_product_complete": true`, `"one_shot_violation_count": 0`, `"raw_receipt_missing_count": 0`, `"accept": 0`, `"no_proposal": 2`, `"reject": 2`, `"model_output_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6520_safety_net_branch_router_ab.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['. The ', '. The positive gate instead cites '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Compact learned routers beat certified structural headroom while preserving exact fallback and making no held-table writes.

## WHAT WOULD REFUTE IT
A certified structural comparator tying or outperforming every learned router on the same held charged-work metric would refute the claimed router advantage.

## WAS THAT CHECKED
Yes. The `arm_held_summaries` compare the learned routers directly with the certified static analytical arm, which wins.

## EVIDENCE
The `best_certified_static_analytical` arm has `held_charged_benefit_units` of `714`, `held_total_charged_work_units` of `13382`, and `held_loss_count` of `0`. The `best_learned_arm` is `linear_router_seed_6520002`, with `best_learned_held_charged_benefit_units` of `695`; its summary has `held_total_charged_work_units` of `13401` and `held_loss_count` of `4`. The positive gate instead cites `upstream_best_structural_held_benefit_units` of `667`.

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_6561_v568_evidence_gate_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify. The administrative readiness assertion would fail if an expected artifact were missing or ineligible, a required gate field were undeclared, a contract check failed, or a protected file changed.

## WAS THAT CHECKED
Yes. Per-artifact eligibility rows, aggregate recomputation, gate summaries, and before/after protected-file hashes provide explicit failure paths. No method-value comparison was attempted or claimed; the verifier is merely the contract oracle.

## EVIDENCE
`verdict_class`: `null`; `artifact_kind`: `contract_or_infrastructure`; `inference_substrate`: `immutable_v567_artifact_gate_and_scope_audit_no_llm`; `verifier_is_oracle`: `true`; `exp6561_llm_load_count`: `0`; `exp6561_hardware_command_count`: `0`; `all_rows_contract_eligible`: `true`; `failed_checks`: `[]`; `all_unchanged`: `true`

## RECOMMENDATION
KEEP

## experiment_6565_v569_evidence_and_retirement_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a contract/receipt artifact, its readiness would be falsified by a required source artifact being absent or hash-mismatched, a gate using an undeclared or retired upstream field, an incomplete retirement row, or a protected-file mutation; it makes no comparative method-value claim to refute.

## WAS THAT CHECKED
Yes. Artifact-level eligibility, hashes, gate-field declarations, prior-failure contracts, aggregate recomputation, and protected-file hashes were checked; the artifact also explicitly records adverse and null inputs rather than claiming scientific success.

## EVIDENCE
`inference_substrate`: `immutable_v568_artifact_gate_failure_and_retirement_audit_no_llm`; `verifier_is_oracle`: `true`; `verdict_class`: `null`; `no_llm_load_performed`: `true`; `no_hardware_command_performed`: `true`; `exp6562_disqualified_science_recorded`: `true`; `exp6563_clean_null_production_recorded`: `true`; `exp6564_clean_null_nfr01_recorded`: `true`; `protected_files_unchanged`: `true`; `failed_checks`: `[]`; `Artifact validation is audit authority, so a clean result cannot use a positive class.`

## RECOMMENDATION
KEEP

## experiment_6571_v570_evidence_gate_and_retirement_root.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify. Treating contract readiness as an administrative claim, it would be refuted by an unresolved gate, an unexpected missing prerequisite, inconsistent eligibility rows, or any assertion that extraction, generation, or graph-Potts utility ran despite contrary execution receipts.

## WAS THAT CHECKED
Yes, for administrative consistency: the artifact includes eligibility rows, gate checks, aggregate recomputation, missing-artifact handling, and execution-boundary receipts. It does not test method value, nor claim to do so.

## EVIDENCE
`source_extraction_claim_count` `0`; `graph_potts_utility_claim_count` `0`; `generation_ran` `false`; `llm_load_performed` `false`; `graph_potts_utility_claimed_by_exp6571` `false`; `failed_checks` `[]`; `unexpected_missing_prerequisite` `false`; `verifier_is_oracle` `true`

## RECOMMENDATION
KEEP

## experiment_6716_object_table_fetch_on_demand_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports only a blocked precondition receipt, not an experimental result.

## WAS THAT CHECKED
No; the paired A/B was not checked because inference never started.

## EVIDENCE
`"run_completed": false`, `"inference_substrate": "precondition_only_no_inference"`, `"verdict_class": "blocked"`, `"honest_verdict": "blocked_context_window: CARNOT_ARC_INDUCE_N_CTX is unset; the paired A/B did not start"`, `"table_on": null`, `"fetch_on_demand": null`, `"paired_delta_fetch_minus_on": null`

## RECOMMENDATION
KEEP
