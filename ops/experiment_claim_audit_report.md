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

## experiment_7038_v617_active_contract_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s factual preflight conclusion, finding the required V617 Markdown source readable and nonempty would refute the reported prerequisite-missing status.

## WAS THAT CHECKED
Yes. The precondition check directly attempted to read the source and recorded its unavailability and terminal failure; no comparative method-value claim was tested.

## EVIDENCE
`honest_verdict` `complete_blocked_v617_active_contract_preflight_prerequisite_missing` `failed_check` `v617_markdown_readable` `expected_value` `readable_nonempty_source` `available` `false` `passed` `false` `verdict_class` `blocked` `rows` `[]`

## RECOMMENDATION
KEEP

## experiment_7039_v617_model_report_forensics.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7040_v617_typed_identity_bridge.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports a blocked prerequisite gate and makes no positive or comparative claim about the typed identity bridge.

## WAS THAT CHECKED
No; bridge evaluation did not proceed because the upstream artifact-validity precondition failed.

## EVIDENCE
`"inference_substrate": "aggregation_from_upstream_artifacts"`; `"arc_typed_identity_bridge_ready_score": 0`; `"raw_report_reproduction_rows": []`; `"identity_obligation_rows": []`; `"positive_fixture_rows": []`; `"negative_fixture_rows": []`; `"passed": false`; `"failed_check": "exp7039_artifact_valid"`; `"verdict_class": "blocked"`; `"honest_verdict": "blocked_exp7039_artifact_invalid"`.

## RECOMMENDATION
KEEP

## experiment_7041_identity_report_channel_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
If the blocking diagnosis were treated as a claim, an observed gate value of 1 with the gate passing would refute it.

## WAS THAT CHECKED
Yes, in `gates_evaluated`; the sole prerequisite gate was evaluated and failed. No method-performance or comparative claim was attempted.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"actual"`: `0`; `"expected"`: `1`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7049_v617_capstone_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation: this is a disposition receipt reporting blocked inputs, not a comparative method, value, or generalization claim.

## WAS THAT CHECKED
No falsification test was applicable; the artifact instead checked input availability and integrity.

## EVIDENCE
`"verdict_class": "blocked"`, `"honest_verdict": "complete_blocked_v617_capstone_input_missing"`, `"inference_substrate": "aggregation_from_upstream_artifacts"`, `"failed_check": "v617_markdown_readable"`, `"observed_value": "missing"`, `"v617_capstone_complete_score": 0`, `"production_default_unchanged": true`

## RECOMMENDATION
KEEP

## experiment_7050_v618_active_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V618 active contract is disqualified because the Markdown and YAML task contracts mismatch.

## WHAT WOULD REFUTE IT
The YAML contract containing all 13 expected tasks in the expected order, with its task count matching the Markdown contract.

## WAS THAT CHECKED
Yes. The artifact directly compares expected and observed task counts and ID order in `gate_check_summary`, `expected_id_order`, and `observed_id_order`.

## EVIDENCE
`honest_verdict`: `complete_disqualified_v618_markdown_yaml_contract_mismatch`; `expected_task_count`: `13`; `observed_task_count`: `3`; `failed_check`: `yaml_task_count`; `passed`: `false`; `v618_task_contract_conforms_score`: `0`; `verdict_class`: `disqualified`

## RECOMMENDATION
KEEP

## experiment_7051_v618_model_report_requalification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no model-performance, solve, generalization, or comparative-value claim to falsify; this is an operational evidence receipt. Its narrower execution record would fail if the launched model hash or process identity mismatched, live inference produced no token, GPU offload was unconfirmed, or the owned-live interval fell below the stated minimum.

## WAS THAT CHECKED
Yes. The artifact checks model-path and hash agreement, process ownership, GPU use, token generation, the minimum live interval, checksum integrity, and cleanup. It does not check task performance or a rival baseline, but it expressly makes no such claim.

## EVIDENCE
`game_level_solve_claim` `false`; `arc_action_count` `0`; `model_report_evidence_ready_score` `1`; `One means evidence validity, not model quality or belief value.`; `inference_substrate` `live_llm_inference`; `generation_request_count` `2`; `minimum_owned_live_interval_s` `75.0`; `owned_live_interval_s` `75.515162603`; `cuda_layer_offload_confirmed` `true`

## RECOMMENDATION
KEEP

## experiment_7052_v618_typed_identity_attack_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The typed-identity attack audit is ready because valid path forms pass, identity attacks fail closed, fresh-process results agree, legacy handling is explicit, and consumers and producers use shared code.

## WHAT WOULD REFUTE IT
Any malformed identity attack being accepted, any valid supported path being rejected, any required identity obligation being unsupported, any fresh-process disagreement, any consumer or producer bypassing shared validation, or any failed upstream gate while the readiness score remained positive.

## WAS THAT CHECKED
Yes. The artifact includes positive fixtures, one-factor attack rows, typed obligation rows, unknown-evidence handling, subprocess execution, upstream gates, and producer/consumer wiring checks. The visible attacks were rejected while all six summary checks passed.

## EVIDENCE
`typed_identity_attack_audit_ready_score`: `1`; `verdict_class`: `positive`; `parent_attacks_fail_closed`; `positive_paths`; `fresh_process_agreement`; `legacy_handling_explicit`; `all_consumers_use_shared_code`; `all_producers_use_shared_code`; `passed`: `true`; `artifact_checksum_change`; `broken_link`; `changed_hub`; `accepted`: `false`; `validation_errors`: `every identity obligation must be supported`; `snapshot_alias`; `canonical_blob`; `direct_file`; `all_obligations_supported`: `true`; `fresh_process`: `true`; `isolated_python`: `true`; `returncode`: `0`; `current_fields_inferred`: `false`; `local_copy_present`: `false`; `shared_builder`: `true`; `shared_validator`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
