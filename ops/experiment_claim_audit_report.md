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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7076_v620_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V620 task-contract preflight is disqualified because the Markdown contract could not be parsed, preventing Markdown–YAML conformity from being established.

## WHAT WOULD REFUTE IT
A successful Markdown contract parse followed by two parseable independent contracts whose 13 task rows and eight gates all conform.

## WAS THAT CHECKED
Yes. The contract-parse gate explicitly ran and failed on a malformed Markdown task row; the artifact therefore terminated before downstream parity rows could be evaluated.

## EVIDENCE
`"failed_check": "contract_parse"`; `"passed": false`; `"row_kind": "contract_parse_error"`; `"observed_task_count": 0`; `"v620_task_contract_conforms_score": 0`; `"verdict_class": "disqualified"`; `"honest_verdict": "complete_disqualified_v620_markdown_yaml_contract_mismatch"`

## RECOMMENDATION
KEEP

## experiment_7077_v620_sota_ingestion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation applies: the artifact records completion of a bounded ingestion workflow but asserts no comparative performance, scientific validity, or added-value claim.

## WAS THAT CHECKED
No; no comparative claim was tested. The checks concern source capture, mapping, and claim boundaries.

## EVIDENCE
`science_claim_promoted`: `false`; `hardware_execution_promoted`: `false`; `evidence_role`: `discovery_index_only`; `search_rank_used_as_quality_evidence`: `false`; `citation_count_claimed`: `false`; `inference_substrate`: `web_bibliographic_search_only_no_llm`

## RECOMMENDATION
KEEP

## experiment_7078_v620_gpu_lease_migration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify; the narrower operational-readiness assertion would fail if any safety gate failed, migration corrupted or removed evidence, repeated execution rewrote journals, or post-migration preflight classified either device as unavailable.

## WAS THAT CHECKED
Yes. The artifact includes precondition, migration, source-preservation, atomic-publication, strict-reader, idempotence, and post-migration preflight checks, with outcomes reported for both real devices.

## EVIDENCE
`"honest_verdict": "null_gpu_lease_compatibility_ready_no_science_claim"`; `"verdict_class": "null"`; `"verifier_is_oracle": false`; `"inference_substrate": "deterministic_os_lease_recovery_no_llm"`; `"gpu_lease_compatibility_ready_score": 1`; `"observed_value": "all checks pass"`; `"failed_check": null`; `"files_removed": []`; `"signals_sent": []`; `"action": "idempotent_noop"`; `"classification": "available"`

## RECOMMENDATION
KEEP

## experiment_7079_v620_gpu_lease_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the limited operational-readiness receipt, any failed cold-audit gate, invalid row, unreleased lease, unsafe recovery, or unavailable post-audit device would refute readiness; there is no comparative or model-quality claim to falsify.

## WAS THAT CHECKED
Yes, for operational readiness: the artifact includes success-path and fail-closed checks for contention, checksum mutation, invalid phase transitions, expiry, ownership, crash recovery, fresh-process rereads, release, and post-audit availability. No comparative/model-quality control was needed because no such claim was made.

## EVIDENCE
`honest_verdict`: `null_gpu_lease_cold_audit_ready_no_model_quality_claim`; `inference_substrate`: `fresh_process_os_lease_audit_no_llm`; `model_load_count`: `0`; `gpu_lease_cold_audit_ready_score`: `1`; `expected_value`: `all checks pass`; `observed_value`: `all checks pass`; `failed_check`: `null`; `outcome`: `JournalError`; `reason`: `checksum_mismatch`; `outcome`: `TransitionError`; `reason`: `transition_not_allowed:preflight->loading`; `outcome`: `LeaseExpired`; `reason`: `lease_expired_stale_heartbeat`.

## RECOMMENDATION
KEEP

## experiment_7080_v620_three_family_entrance_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; the operational statement of incompleteness would be contradicted by complete proposal and forced-prefix rows for all three model families with every completion gate passing.

## WAS THAT CHECKED
Yes. Completion was directly checked in `gate_check_summary`, `per_model_rows`, `model_execution_rows`, and `forced_prefix_rows`; the required completeness conditions failed.

## EVIDENCE
`honest_verdict` `partial_three_family_entrance_proposal_bank_incomplete`; `entrance_proposal_bank_complete_score` `0`; `proposal_row_completeness` `passed` `false`; `forced_prefix_row_completeness` `passed` `false`; `raw_identity_cuda_cleanup` `passed` `false`; `terminal_state` `failed`; `forced_prefix_rows` `[]`

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
Any headline-eligible family producing a transport error, no usable tokens, empty or unparsable output, leaked control tokens, length-limited output, or an incomplete execution would refute the claim.

## WAS THAT CHECKED
Yes. Generation was invoked through `create_chat_completion`; the artifact reports transport-error gates, execution status, and per-model empty-output, zero-token, parseability, control-token-leak, and length-limit measurements across eight rows per family.

## EVIDENCE
`"generation_invoked": true`; `"transport_method": "create_chat_completion"`; `"check": "transport_error_set"` with `"observed_value": []`; `"raw_row_count": 8`; `"terminal_state": "complete"`; `"empty_output_rate": 0.0`; `"zero_token_rate": 0.0`; `"parseable_rate": 1.0`; `"leaked_control_token_count": 0`; `"length_limited_count": 0`; `"chat_transport_ready_score": 1`; `"honest_verdict": "positive: all three local GGUF families passed bounded chat transport"`

## RECOMMENDATION
KEEP

## experiment_7086_v621_three_family_entrance_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a completeness receipt, not a comparative performance or value claim. Its operational assertion would be falsified by a missing model family, missing expected proposal or forced-prefix rows, or an incomplete execution.

## WAS THAT CHECKED
Yes. The gate summary checks model-family count and row completeness, while execution rows report completed generation for all three families and both phases.

## EVIDENCE
`honest_verdict`: `positive: complete three-family chat entrance bank acquired`; `entrance_proposal_bank_complete_score`: `1`; `One means complete data, not good proposals.`; `model_family_count`; `expected_value`: `3`; `observed_value`: `3`; `proposal_row_completeness`; `forced_prefix_row_completeness`; `observed_value`: `[]`; `passed`: `true`; `generation_invoked`: `true`; `terminal_state`: `complete`

## RECOMMENDATION
KEEP
