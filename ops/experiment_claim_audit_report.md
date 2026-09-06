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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7030_arc_gguf_model_identity_bridge.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC GGUF model-identity bridge is ready because it accepts the reproduced valid topology, rejects specified identity defects, supports legacy records, and is wired into producers and validation.

## WHAT WOULD REFUTE IT
Rejection of the valid Exp7025-shaped snapshot-to-extensionless-blob case, acceptance of any malformed identity case, failure of legacy compatibility, or evidence that producers or consumers bypass the shared bridge.

## WAS THAT CHECKED
Yes. One positive fixture, eight negative fixtures, one legacy compatibility row, and four producer/consumer wiring checks exercised those failure modes.

## EVIDENCE
`arc_model_identity_bridge_ready_score` `1`; `topology_reproduced` `true`; `positive_fixture_rows` `accepted` `true`; `negative_fixture_rows` `accepted` `false`; `row_count` `8`; `legacy_compatibility_rows` `accepted` `true`; `shared_bridge_called` `true`; `canonical_builder_called` `true`; `shared_bridge_recomputed` `true`; `verifier_is_oracle` `false`; `complete_positive_arc_model_identity_bridge_ready`

## RECOMMENDATION
KEEP

## experiment_7031_arc_model_identity_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC model-identity mechanism is audit-ready because it accepts valid identities, rejects identity mutations, preserves legacy cases, and is reachable from the live consumer in an isolated process.

## WHAT WOULD REFUTE IT
Any invalid alias, hash, revision, path type, or stale-server identity being accepted; any valid or legacy identity being rejected; an isolated-process failure; or the live consumer using a local or different implementation would refute the claim.

## WAS THAT CHECKED
Yes. Positive and legacy acceptance rows, multiple mutation and stale-identity rejection rows, fresh-process execution, command receipts, and live-consumer reachability explicitly exercised those failure modes.

## EVIDENCE
`arc_model_identity_audit_ready_score`: `1`; `all_identity_mutations_rejected`; `observed_value`: `true`; `passed`: `true`; `snapshot_symlink_to_extensionless_blob`; `accepted`: `true`; `receipt_round_trip_equal`: `true`; `content_hash`; `same_size_different_bytes`; `repository`; `revision`; `broken_link`; `path_type`; `ambiguous_hard_link`; `stale_server_identity`; `accepted`: `false`; `isolated_python`: `true`; `private_work_dir`: `true`; `round_trip_valid`: `true`; `local_copy_present`: `false`; `shared_function_identity`: `true`; `shared_source_file`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7032_repaired_belief_shadow_live_trace.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. Treating the blocked-status report as an operational assertion, completed live inference and populated trace rows would contradict it.

## WAS THAT CHECKED
Yes. The execution gate failed before live inference, and the artifact records no decision, belief-query, action-parity, or observation rows.

## EVIDENCE
`"verdict_class": "blocked"`; `"honest_verdict": "blocked_belief_shadow_live_trace:live_trace_execution"`; `"belief_shadow_trace_ready_score": 0`; `"live_model_invoked": false`; `"game_level_solve_claim": false`; `"failed_check": "live_trace_execution"`; `"passed": false`; `"belief_query_rows": []`; `"action_parity_rows": []`; `"per_decision_rows": []`

## RECOMMENDATION
KEEP

## experiment_7038_v617_active_contract_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; treating the blocked status as an operational assertion, a readable, nonempty V617 markdown source at execution would refute it.

## WAS THAT CHECKED
Yes. The prerequisite check directly tested markdown readability and terminated before producing task-level or comparative results.

## EVIDENCE
`honest_verdict`: `complete_blocked_v617_active_contract_preflight_prerequisite_missing`; `failed_check`: `v617_markdown_readable`; `passed`: `false`; `available`: `false`; `verdict_class`: `blocked`; `observed_task_count`: `0`; `rows`: `[]`

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
Not applicable; the artifact reports a blocked run rather than claiming the typed identity bridge is ready.

## WAS THAT CHECKED
No; the substantive identity checks were not run because an upstream artifact-validity precondition failed.

## EVIDENCE
`arc_typed_identity_bridge_ready_score` `0`; `positive_fixture_rows` `[]`; `negative_fixture_rows` `[]`; `identity_obligation_rows` `[]`; `verdict_class` `blocked`; `honest_verdict` `blocked_exp7039_artifact_invalid`; `failed_check` `exp7039_artifact_valid`

## RECOMMENDATION
KEEP

## experiment_7041_identity_report_channel_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite gate and makes no comparative or value claim.

## WAS THAT CHECKED
No; the audit did not run because the upstream readiness gate failed.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "arc_typed_identity_bridge_ready_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7049_v617_capstone_disposition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V617 capstone is blocked because required input evidence is missing or unusable.

## WHAT WOULD REFUTE IT
All required contract and upstream artifacts being present, readable, integrity-valid, and sufficient to produce a passing capstone completion score would refute the blocked disposition.

## WAS THAT CHECKED
Yes. The artifact checks contract readability, artifact presence, integrity, gate replay, and completion; the refuting conditions had a real opportunity to appear but did not.

## EVIDENCE
`honest_verdict`: `complete_blocked_v617_capstone_input_missing`; `failed_check`: `v617_markdown_readable`; `observed_value`: `missing`; `passed`: `false`; `artifact_state`: `missing`; `effective_verdict_class`: `blocked`; `producer_evidence_usable`: `false`; `v617_capstone_complete_score`: `0`; `verdict_class`: `blocked`

## RECOMMENDATION
KEEP
