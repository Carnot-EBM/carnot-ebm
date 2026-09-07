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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7080_v620_three_family_entrance_bank.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The three-family entrance proposal bank is partial and incomplete.

## WHAT WOULD REFUTE IT
Complete proposal and forced-prefix rows for all three model families, with every completeness, execution, checkpoint, cleanup, lease, and VRAM-release gate passing and the completion score equal to 1.

## WAS THAT CHECKED
Yes. The explicit completion gates, per-model counts, execution receipts, and completion score could have shown a complete bank; instead, they record missing proposal and forced-prefix keys, two models with zero proposals, failed execution/cleanup checks, and a zero completion score.

## EVIDENCE
`honest_verdict` `partial_three_family_entrance_proposal_bank_incomplete` `entrance_proposal_bank_complete_score` `0` `proposal_row_completeness` `proposal_key_set_mismatch` `forced_prefix_row_completeness` `forced_prefix_key_set_mismatch` `raw_identity_cuda_cleanup` `cleanup_incomplete` `model_execution_incomplete` `checkpoint_incomplete` `vram_release_incomplete` `passed` `false` `proposal_count` `0` `forced_prefix_count` `0` `terminal_state` `failed`

## RECOMMENDATION
KEEP

## experiment_7084_v621_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7085_v621_chat_transport_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All three tested local GGUF model families successfully completed bounded chat transport.

## WHAT WOULD REFUTE IT
Any headline-eligible family failing local generation, producing transport errors, empty or zero-token output, unparseable output, leaked control tokens, length-limited output, or an incomplete execution would refute the claim.

## WAS THAT CHECKED
Yes. The artifact reports per-model execution and transport metrics for all three families, plus aggregate gates covering transport errors, model authenticity, and transport readiness.

## EVIDENCE
`honest_verdict`: `positive: all three local GGUF families passed bounded chat transport`; `generation_invoked`: `true`; `inference_substrate`: `bounded live local SOTA GGUF chat generation`; `transport_error_set`; `observed_value`: `[]`; `chat_transport_ready_score`: `1`; `raw_row_count`: `8`; `terminal_state`: `complete`; `parseable_rate`: `1.0`; `empty_output_rate`: `0.0`; `zero_token_rate`: `0.0`; `leaked_control_token_count`: `0`; `length_limited_count`: `0`; `all_models_real`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7086_v621_three_family_entrance_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the operational receipt, a missing model family, missing expected proposal or forced-prefix row, failed execution, or non-local/fallback generation would refute bank completeness.

## WAS THAT CHECKED
Yes. The gate checks family count, proposal-row completeness, forced-prefix-row completeness, and execution/transport cleanup; all passed. This verifies acquisition, not proposal quality or comparative value.

## EVIDENCE
`positive: complete three-family chat entrance bank acquired`; `One means complete data, not good proposals.`; `model_family_count`; `expected_value`: `3`; `observed_value`: `3`; `proposal_row_completeness`; `forced_prefix_row_completeness`; `all checks pass`; `model_full_generation`; `generation_invoked`: `true`

## RECOMMENDATION
KEEP

## experiment_7091_v622_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V622 Markdown and YAML task contracts mismatch, disqualifying contract conformance.

## WHAT WOULD REFUTE IT
Twelve YAML tasks matching all twelve Markdown tasks in identity, order, and required contract clauses, with every row passing and a conformance score of one.

## WAS THAT CHECKED
Yes. The independent Markdown/YAML replay compared expected and observed task counts, IDs, ordering, and per-task contract fields; it found only six YAML tasks where twelve were expected and recorded failing rows.

## EVIDENCE
`inference_substrate`: `deterministic independent Markdown and YAML contract replay`; `expected_task_count`: `12`; `observed_task_count`: `6`; `failed_check`: `yaml_task_count`; `passed`: `false`; `v622_task_contract_conforms_score`: `0`; `verdict_class`: `disqualified`; `honest_verdict`: `complete_disqualified_v622_markdown_yaml_contract_mismatch`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7092_v622_sota_ingestion.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded V622 source-ingestion audit completed and found no post-cutoff, decision-relevant method warranting adoption.

## WHAT WOULD REFUTE IT
A source within the declared delta window producing a nonduplicate candidate marked decision-relevant and entered into the adoption ledger, or a failed ingestion gate.

## WAS THAT CHECKED
Yes. The artifact reports a post-cutoff search result count, candidate-level relevance and duplication dispositions, an adoption ledger, source-class receipts, and explicit gate outcomes. Within this bounded scope, the refuting observation could have appeared but did not.

## EVIDENCE
The `delta_rule` is `publication_or_revision_date > planner_cutoff_date and <= literature_end_date`; `post_cutoff_arxiv_result_count` is `0`; the displayed candidates have `decision_relevant` equal to `false`; `adoption_rows` is `[]`; `failed_check` is `null`; `passed` is `true`; `v622_sota_ingestion_complete_score` is `1`; `verifier_is_oracle` is `false`; and `inference_substrate_class` is `no_model_load`.

## RECOMMENDATION
KEEP

## experiment_7093_v622_entrance_bank_sufficiency_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed entrance-support audit found the available evidence insufficient.

## WHAT WOULD REFUTE IT
A passing overall gate with `entrance_support_audit_ready_score` equal to 1—especially complete support across every applicable model, seed, source-group, and entrance-family cell—would refute the insufficiency claim.

## WAS THAT CHECKED
Yes. The predeclared support cross-product could have passed, but the overall gate failed specifically at family sufficiency, with applicable family cells recorded as missing.

## EVIDENCE
`"honest_verdict": "null: completed entrance support audit is insufficient"`; `"entrance_support_audit_ready_score": 0`; `"failed_check": "family_sufficiency"`; `"observed_value": false`; `"passed": false`; `"applicable": true`; `"proposal_count": 0`; `"status": "missing"`; `"frozen_before_proposal_outcomes": true`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7094_matched_hardness_entrance_diagnostic.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify; the blocking determination would be refuted if the upstream gate had observed 1 or passed.

## WAS THAT CHECKED
Yes. The sole gate was evaluated and recorded an observed value of 0 against the required value of 1.

## EVIDENCE
`"status"` `"blocked"` `"honest_verdict"` `"blocked_gate_check_failed"` `"actual"` `0` `"expected"` `1` `"passed"` `false` `"blocked_at_layer"` `"conductor_pre_gate"`

## RECOMMENDATION
KEEP
