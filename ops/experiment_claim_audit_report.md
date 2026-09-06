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
| NO_CLAIM | 4 |

## experiment_7071_bcit_drift_rollback_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact is a blocked-execution receipt and reports no audit outcome to falsify.

## WAS THAT CHECKED
No; the audit was blocked at the pre-gate because the upstream completion score was 0 rather than 1.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "bcit_comparison_complete_score"`, `"failed_expected": 1`, `"failed_observed": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7072_v619_live_arc_compaction_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact makes no comparative value claim. Its blocked status would be contradicted by at least 30 eligible units and populated A/B result rows.

## WAS THAT CHECKED
Yes. The eligibility gate expected at least 30 units and observed zero; consequently, no paired comparisons were run.

## EVIDENCE
`verdict_class`: `blocked`; `honest_verdict`: `blocked_live_arc_compaction_ab:eligible_hidden_or_rotation_units`; `failed_check`: `eligible_hidden_or_rotation_units`; `expected_value`: `>=30`; `observed_value`: `0`; `rows`: `[]`; `per_game_results`: `[]`; `models_used`: `[]`; `qwen_pair_count`: `0`; `gemma_pair_count`: `0`; `retirement_decision`: `no_decision_precondition_blocked`

## RECOMMENDATION
KEEP

## experiment_7075_v619_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V619 reached a complete null capstone disposition because every scientific branch remained resource-blocked or lacked required evidence.

## WHAT WOULD REFUTE IT
At least one branch having all required upstream tasks available, valid, non-circular, and gate-passing while still being classified as resource-blocked would refute the headline.

## WAS THAT CHECKED
Yes. The artifact checks upstream discovery, schema and hash validity, circularity, gate outcomes, and branch-level evidence classes; those checks could have produced a completed branch, but instead recorded missing, blocked, invalid, or circular evidence. Invalid positive rows were not allowed to overturn the branch dispositions.

## EVIDENCE
`"honest_verdict": "complete_null_v619_terminal_branch_dispositions"`; `"verdict_class": "null"`; `"entrance_branch_decision": "blocked_resource"`; `"self_learning_branch_decision": "blocked_resource"`; `"arc_compaction_branch_decision": "blocked_resource"`; `"ising_branch_decision": "blocked_resource"`; `"reason": "A required task is missing or resource-blocked."`; `"valid": false`; `"errors": ["source_hash_mismatch"]`; `"science_result": "circular_positive"`; `"circular": true`; `"verifier_is_oracle": true`; `"observed_value": 0`; `"passed": false`; `"status": "not_comparative"`; `"default_off": true`

## RECOMMENDATION
KEEP

## experiment_7076_v620_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V620 contract preflight was disqualified because the Markdown task contract could not be parsed as required.

## WHAT WOULD REFUTE IT
Successful parsing of both independent contracts, followed by 13 conforming task rows and all required gates passing, would refute the disqualification.

## WAS THAT CHECKED
Yes. The contract-parse gate attempted the required check and recorded a concrete malformed Markdown row; the corresponding evidence row failed. Downstream parity checks could not run after that terminal failure, but they are unnecessary to sustain the narrow disqualification claim.

## EVIDENCE
`"failed_check": "contract_parse"`; `"passed": false`; `"observed": "ValueError: malformed Markdown task row: | 1 | `exp7076-v620-contract-preflight` | V620 Markdown and YAML task-contract preflight | `results/experiment_7076_v620_contract_preflight.json` | none |"`; `"row_kind": "contract_parse_error"`; `"v620_task_contract_conforms_score": 0`; `"verdict_class": "disqualified"`; `"honest_verdict": "complete_disqualified_v620_markdown_yaml_contract_mismatch"`

## RECOMMENDATION
KEEP

## experiment_7077_v620_sota_ingestion.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded V620 SOTA ingestion completed successfully with dated sources and every promoted hook mapped to a task.

## WHAT WOULD REFUTE IT
A required search failing, a promoted hook lacking a task or defer decision, an undated supporting source, or an inaccessible/disallowed source being used to promote a claim would falsify the bounded completion claim.

## WAS THAT CHECKED
Yes. The gate could record failure through `failed_check`, query rows record access outcomes, task-mapping rows expose mappings and defer decisions, and inaccessible sources are retained with claims disallowed.

## EVIDENCE
`"honest_verdict": "complete_positive_v620_sota_ingestion"`; `"sota_ingestion_complete_score": 1`; `"expected_value": 1`; `"observed_value": 1`; `"failed_check": null`; `"passed": true`; `"access_outcome": "completed_bounded_search"`; `"science_claim_promoted": false`; `"access_outcome": "browser_challenge"`; `"claim_allowed": false`; `"verifier_is_oracle": false`; `"inference_substrate": "web_bibliographic_search_only_no_llm"`

## RECOMMENDATION
KEEP

## experiment_7078_v620_gpu_lease_migration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or scientific claim is asserted; operational readiness would be contradicted by any failed migration, strict-reader rejection of the migrated journal, source-preservation mismatch, non-idempotent rerun, unavailable post-migration device, removed file, or sent signal.

## WAS THAT CHECKED
Yes for the operational receipt: migration, strict-reader acceptance, source preservation, idempotence, post-migration availability, removals, and signals are represented by explicit rows or ledgers. No comparative claim required a rival baseline.

## EVIDENCE
`honest_verdict` = `null_gpu_lease_compatibility_ready_no_science_claim`; `verdict_class` = `null`; `inference_substrate` = `deterministic_os_lease_recovery_no_llm`; `verifier_is_oracle` = `false`; `gpu_lease_compatibility_ready_score` = `1`; `observed_value` = `all checks pass`; `failed_check` = `null`; `files_removed` = `[]`; `signals_sent` = `[]`; `classification` = `available`; `current_accepted` = `true`; `action` = `idempotent_noop`.

## RECOMMENDATION
KEEP

## experiment_7079_v620_gpu_lease_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The GPU lease infrastructure is ready for cold-audit use, without making any model-quality claim.

## WHAT WOULD REFUTE IT
A valid row showing failed same-device exclusion, blocked independent-device progress, acceptance of a corrupted journal or illegal phase transition, recovery over a live owner, failure to recover after a crash, or a device remaining unavailable after release would falsify the readiness claim.

## WAS THAT CHECKED
Yes. The artifact exercises same-device contention, independent-device progress, checksum mutation, phase skipping, lease expiry, live-owner protection, PID reuse, wrong-token ownership, crash recovery, fresh-process rereads, and post-audit availability. These checks contain failure-capable outcomes rather than only successful-path receipts.

## EVIDENCE
`honest_verdict`: `null_gpu_lease_cold_audit_ready_no_model_quality_claim`; `verifier_is_oracle`: `false`; `inference_substrate`: `fresh_process_os_lease_audit_no_llm`; `model_load_count`: `0`; `same_device_exclusion`; `acquired_count`: `1`; `lease_busy_count`: `1`; `independent_device_progress`; `observed_value`: `2`; `checksum_mutation`; `outcome`: `JournalError`; `reason`: `checksum_mismatch`; `phase_skip`; `outcome`: `TransitionError`; `reason`: `transition_not_allowed:preflight->loading`; `matching_live_owner`; `outcome`: `RecoveryError`; `reason`: `recorded_owner_still_live`; `crash_exit_code`: `79`; `recovery_exit_code`: `0`; `recovery_performed`: `true`; `classification`: `available`; `observed_value`: `all checks pass`; `gpu_lease_cold_audit_ready_score`: `1`

## RECOMMENDATION
KEEP

## experiment_7080_v620_three_family_entrance_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made; the artifact explicitly reports an incomplete acquisition.

## WAS THAT CHECKED
No refutation check was applicable. The artifact instead checked completeness and recorded failure in `gate_check_summary` and `per_model_rows`.

## EVIDENCE
`honest_verdict`: `partial_three_family_entrance_proposal_bank_incomplete`; `entrance_proposal_bank_complete_score`: `0`; `failed_check`: `proposal_row_completeness`; `proposal_key_set_mismatch`; `forced_prefix_rows`: `[]`; `terminal_state`: `failed`; `proposal_count`: `0` for two model rows.

## RECOMMENDATION
KEEP
