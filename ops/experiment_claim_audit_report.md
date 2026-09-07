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

## experiment_7110_v624_evidence_ingress_quarantine.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The deterministic evidence-ingress quarantine is ready to accept eligible clean evidence, reject flagged or date-invalid evidence, preserve historical artifacts, and reproduce the founding regression gate.

## WHAT WOULD REFUTE IT
A clean eligible fixture being rejected, a flagged or invalid-date fixture being accepted, the flagged experiment 7099 passing ingestion, a historical artifact being rewritten, or the historical census count differing from 55 would falsify the claim.

## WAS THAT CHECKED
Yes. Opposing positive and negative fixtures were exercised in `fixture_rows`; the census, historical non-rewrite condition, and experiment-7099 regression were separately checked in `rows`. These checks had outcomes that could have disagreed with their expected values.

## EVIDENCE
`accepted_clean_input` has `expected_accepted_count` `1`, `observed_accepted_count` `1`, and `passed` `true`. `artifact_level_flag` and `verifier_critical_flag` each have `expected_accepted_count` `0`, `observed_accepted_count` `0`, and `passed` `true`. `all_fixtures_pass` has `observed_value` `true`; `historical_capstone_census_count` has `expected_value` `55` and `observed_value` `55`; `historical_artifacts_rewritten` has `observed_value` `false`; and `exp7108_rejects_flagged_exp7099` has `observed_value` `true`. `verifier_is_oracle` is `false`.

## RECOMMENDATION
KEEP

## experiment_7111_v624_arc_provenance_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V624 ARC forward-provenance writer and consumer canaries passed.

## WHAT WOULD REFUTE IT
Acceptance of missing or invalid provenance, headline eligibility for an unqualified row, rejection of the qualified live row, incorrect dashboard grouping, or mutation of historical bytes or the ARC registry would falsify the claim.

## WAS THAT CHECKED
Yes. Negative provenance cases appear in `missing_provenance_rejection_rows`; qualified and disqualified cases appear in `headline_eligibility_rows`; dashboard results appear in `dashboard_consumer_rows`; and preservation checks appear in `rows` and the registry hashes.

## EVIDENCE
`writer_rejects_missing_or_invalid_provenance` has `observed_value` `true` and `passed` `true`. `only_receipted_live_row_is_headline_eligible` observed `[true, false, false, false]` and `passed` `true`. Every case in `missing_provenance_rejection_rows` has `rejected` `true`. `observed_headline_levels` is `2`, matching `expected_headline_levels` `2`. `historical_row_bytes_unchanged` and `arc_registry_hash_unchanged` both have `passed` `true`. `verifier_is_oracle` is `false`, and `inference_substrate_class` is `no_model_load`.

## RECOMMENDATION
KEEP

## experiment_7112_v624_sota_ingestion.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded V624 SOTA audit completed its gates and adopted one decision-relevant control linked to experiment `exp7118-principle-step-memory-csl`.

## WHAT WOULD REFUTE IT
A failed gate, an incomplete required source-class receipt, no decision-relevant candidate, or an adoption row that did not match that candidate and experiment hook would refute the claim.

## WAS THAT CHECKED
Yes. The gate summary passed; every source-class row is terminal with an honest receipt, including blocked routes; and the decision-relevant candidate matches the sole adoption row by candidate ID and experiment hook. These checks could have failed or yielded no adoption, so the outcome was not forced. This supports only audit completion and adoption—not the control’s eventual value.

## EVIDENCE
`gate_check_summary` `passed` `true`  
`expected_value` `1` `observed_value` `1`  
`honest_receipt` `true`  
`terminal` `true`  
`candidate_id` `arxiv-2609-02750`  
`decision_relevant` `true`  
`classification` `control`  
`experiment_hook` `exp7118-principle-step-memory-csl`  
`v624_sota_ingestion_complete_score` `1`  
`Theorems under stated assumptions and SWE-bench results do not prove Carnot memory value or authorize any write.`

## RECOMMENDATION
KEEP

## experiment_7113_v624_arc_generation_liveness.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; successful preconditions followed by completed generation-request rows would only contradict the artifact’s blocked-run status.

## WAS THAT CHECKED
Yes. The precondition checks recorded terminal failures, so generation was not attempted and no method-performance claim was evaluated.

## EVIDENCE
`honest_verdict`: `complete_blocked_arc_generation_liveness_precondition_failed`; `verdict_class`: `blocked`; `healthy_idle_gpus`; `expected_value`: `2`; `observed_value`: `0`; `passed`: `false`; `rows`: `[]`; `per_model_request_rows`: `[]`; `arc_generation_liveness_ready_score`: `0`; `offline_reproduced`: `false`; `arc_registry_delta`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7114_adapter_withheld_arc_loo_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and reports no measurement or comparative result to falsify.

## WAS THAT CHECKED
No; the experiment was blocked at `conductor_pre_gate` because the sole gate had `passed` set to `false`.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_reason`: `actual=0 == expected=1`; `failed_upstream`: `exp7113-arc-generation-liveness-recovery`; `failed_observed`: `0`; `failed_expected`: `1`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7121_v625_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7122_v625_sota_ingestion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
V625 SOTA ingestion completed successfully.

## WHAT WOULD REFUTE IT
A failed required access, source, model-cache, task-map, date-window, or append-integrity check—or any required row marked nonterminal—would refute completion.

## WAS THAT CHECKED
Yes. The artifact records the ingestion gates, preconditions, source rows, task-method mappings, publication-window checks, and append-marker integrity; all reported checks pass. This is a collection receipt, not a comparative method-value claim.

## EVIDENCE
`"honest_verdict": "positive_v625_sota_ingestion_complete"`; `"v625_sota_ingestion_complete_score": 1`; `"failed_check": null`; `"passed": true`; `"inference_substrate_class": "aggregation"`; `"no model loaded"`; `"carnot_result_claimed": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7123_v625_arc_loo_shard_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made; a future claim of adapter benefit would be refuted if the withheld arm tied or lost to the visible-control arm.

## WAS THAT CHECKED
No. Both experimental arms and all outcome rows are empty; the artifact records only initialization and frozen game selection.

## EVIDENCE
`"inference_substrate_class": "blocked_no_run"`, `"rows": []`, `"adapter_withheld_rows": []`, `"adapter_visible_control_rows": []`, `"paired_delta_rows": []`, `"arc_loo_shard_complete_score": 0`, `"headline_solve_eligible": false`, `"failed_check": "both_arms_complete"`, `"observed_value": 0`, `"verdict_class": "partial"`, `"honest_verdict": "complete_partial: artifact_initialized_and_registry_rank_one_frozen"`

## RECOMMENDATION
KEEP
