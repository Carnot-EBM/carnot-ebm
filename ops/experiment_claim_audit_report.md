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
| CLAIM_OVERSTATED | 1 |
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 3 |

## experiment_7025_belief_shadow_live_trace.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is made. Treating the blocked execution status as an operational assertion, a successful owned live trace with populated execution and action rows would contradict it.

## WAS THAT CHECKED
Yes, operationally in `gate_check_summary`; execution failed before producing trace rows, so no method-value claim received a chance to succeed or fail.

## EVIDENCE
`verdict_class` is `blocked`; `belief_shadow_trace_ready_score` is `0`; `failed_check` is `live_trace_execution`; `passed` is `false`; `observed_value` is `ValueError: invalid ARC evaluation provenance: model_filename must be one GGUF filename`; `rows`, `model_execution_rows`, `shadow_rows`, and `control_rows` are `[]`; `arc_new_level_banked` is `0`; `verifier_is_oracle` is `false`; `solve_provenance` states `Live self-discovery labels the official observation path without claiming a solve.`

## RECOMMENDATION
KEEP

## experiment_7026_held_mechanic_belief_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made to falsify; an A/B result showing the held-mechanic method losing or tying a serious baseline would refute a future positive claim.

## WAS THAT CHECKED
No. The A/B was blocked before execution at `conductor_pre_gate`; only two prerequisite gates were evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_reason": "actual=0 == expected=1"`; `"gate_check_summary": "1 of 2 gate(s) failed; first failure: exp7025-belief-shadow-live-trace.belief_shadow_trace_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7027_v615_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V615 capstone is blocked because the required live provenance, matched-compute, and held-mechanic comparison evidence is absent.

## WHAT WOULD REFUTE IT
Complete live-shadow evidence, Exp7025 readiness equal to 1, Exp7026 status equal to complete, matched live compute receipts, and populated per-game comparison rows would refute the claimed evidence absence.

## WAS THAT CHECKED
Yes. The artifact checks the upstream gate states, live-shadow provenance, matched live compute, row-derived live value, compute receipts, and per-game rows. Each required live check could have passed but instead recorded a failing or zero-valued observation.

## EVIDENCE
The headline is `complete_blocked_v615_capstone_required_live_evidence_absent` with `verdict_class` `blocked`. The gate expected `exp7025.belief_shadow_trace_ready_score` of `1` and `exp7026.status` of `complete`, but observed `0` and `blocked`. The checks `live_shadow_provenance`, `matched_live_compute`, and `row_derived_live_value` each have `observed_value` `false`. The live recomputation records `per_game_row_count` `0`, `model_call_count` `0`, `compute_receipt_count` `0`, and `missing_cells` `all_planned_live_cells`. The artifact also records `promoted_claims` as `[]`.

## RECOMMENDATION
KEEP

## experiment_7028_v616_active_contract_preflight.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
The active V616 Markdown and YAML roadmaps conform to the expected 10-task contract.

## WHAT WOULD REFUTE IT
A task-count, identity, ordering, or contract-field mismatch between the Markdown roadmap and active YAML would falsify conformity; specifically, observing fewer than the expected 10 tasks is sufficient.

## WAS THAT CHECKED
Yes. The task-count gate compared the expected and observed counts, failed, and propagated the failure into the conformity score and disqualified verdict.

## EVIDENCE
`expected_task_count`: `10`; `observed_task_count`: `5`; `failed_check`: `observed_task_count`; `passed`: `false`; `v616_task_contract_conforms_score`: `0`; `honest_verdict`: `complete_disqualified_v616_markdown_yaml_contract_mismatch`; `verdict_class`: `disqualified`.

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_7029_v616_sota_scope_audit.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The completed V616 SOTA scope audit found zero post-marker delta and no scientific improvement.

## WHAT WOULD REFUTE IT
A verified, nonduplicate primary finding provably published or updated after the exact V616 marker time that required a reference-ledger change or scope expansion.

## WAS THAT CHECKED
No. The audit occurred on the marker date, records the marker only by date, and treats same-day ordering as uncertain. Consequently, a same-day source could not establish the required post-marker ordering, while a later-date source could not yet exist in this audit’s data. The null result therefore lacked a real opportunity to fail.

## EVIDENCE
`complete_positive_v616_sota_scope_audit_zero_delta_no_scientific_improvement`; `marker_date`; `2026-09-05`; `accessed_on`; `2026-09-05`; `cutoff_relation`; `same_day_order_uncertain`; `date_receipt`; `access_date_only_no_page_update_time`; `post_marker_delta_rows`; `[]`; `action`; `no_change`; `No verified primary finding has proved ordering after the V616 marker.`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7030_arc_gguf_model_identity_bridge.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC GGUF model-identity bridge is ready because it accepts valid current and legacy identities, rejects specified identity defects, and is wired into producers and validation.

## WHAT WOULD REFUTE IT
A valid Exp7025-shaped identity being rejected; any malformed identity being accepted; legacy provenance becoming unreadable; or a producer or consumer bypassing the shared bridge would falsify readiness.

## WAS THAT CHECKED
Yes. The artifact includes one positive fixture, eight distinct negative fixtures, one legacy-compatibility check, four wiring checks, and successful command receipts.

## EVIDENCE
`arc_model_identity_bridge_ready_score`: `1`; `positive_fixture_rows`: `accepted`: `true`; `negative_fixture_rows`: `accepted`: `false`; `legacy_compatibility_rows`: `accepted`: `true`; `current_fields_inferred`: `false`; `shared_bridge_recomputed`: `true`; `shared_bridge_called`: `true`; `canonical_builder_called`: `true`; `wrong_hash`; `wrong_hub`; `wrong_revision`; `missing_requested_file`; `broken_symlink`; `directory_path`; `misleading_gguf_basename`; `observed_blob_not_reachable`; `failed_check`: `null`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_7031_arc_model_identity_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC model-identity audit is ready because valid identities pass, identity-confusing mutations fail, and the audited shared validator reaches the live consumer.

## WHAT WOULD REFUTE IT
A malformed or stale identity being accepted, a valid identity being rejected, or the live consumer not using the audited shared implementation would falsify the claim.

## WAS THAT CHECKED
Yes. Positive and legacy-valid cases were accepted; alias, hash, revision, path-type, and stale-server mutations were rejected; consumer reachability was checked for the shared functions.

## EVIDENCE
`"accepted": true`, `"headline_eligible": true`, `"receipt_round_trip_equal": true`, `"validation_errors": []`, `"accepted": false`, `"case": "stale_server_identity"`, `"case": "same_size_different_bytes"`, `"case": "ambiguous_hard_link"`, `"local_copy_present": false`, `"shared_function_identity": true`, `"shared_source_file": true`, `"passed": true`, `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7032_repaired_belief_shadow_live_trace.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked readiness attempt and makes no comparative, performance, or game-level claim. A successful owned live trace would contradict only its blocked status.

## WAS THAT CHECKED
Yes, for readiness: `live_trace_execution` was attempted and failed before model invocation or decision rows could be produced. No method-value claim was tested.

## EVIDENCE
`"verdict_class": "blocked"`; `"honest_verdict": "blocked_belief_shadow_live_trace:live_trace_execution"`; `"belief_shadow_trace_ready_score": 0`; `"game_level_solve_claim": false`; `"live_model_invoked": false`; `"failed_check": "live_trace_execution"`; `"passed": false`; `"per_decision_rows": []`

## RECOMMENDATION
KEEP
