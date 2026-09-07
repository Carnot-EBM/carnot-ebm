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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7097_v623_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V623 Markdown and YAML task contracts conform across all 12 tasks and required gate clauses.

## WHAT WOULD REFUTE IT
Any mismatch in task count, ID/order, title, deliverable, routing, execution venue, required fields, gates, producer-consumer relationships, or prompt tails—or any row marked false—would falsify the claim.

## WAS THAT CHECKED
Yes. Independent Markdown and YAML contract replay compared expected and observed values through per-task row families and the aggregate gate check; the checks could fail and prior contract versions recorded mismatches.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts: independent Markdown and YAML contract replay`; `expected_task_count`: `12`; `observed_task_count`: `12`; `gate_check_summary`; `expected_value`: `1`; `observed_value`: `1`; `passed`: `true`; `v623_task_contract_conforms_score`: `1`; `verifier_is_oracle`: `false`; `complete_positive_v623_task_contract_conforms`

## RECOMMENDATION
KEEP

## experiment_7098_v623_sota_ingestion.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The V623 state-of-the-art ingestion completed successfully and found no post-cutoff material requiring adoption.

## WHAT WOULD REFUTE IT
A source with a publication or revision date after the planner cutoff but no later than the literature end date, producing a novel, decision-relevant candidate or a nonempty adoption row.

## WAS THAT CHECKED
No. The cutoff and end date are identical, while eligibility requires a date strictly after the cutoff and no later than the end date. That interval is empty, so the claimed empty delta could not lose.

## EVIDENCE
`honest_verdict`: `complete_positive_v623_sota_ingestion_empty_delta`; `delta_rule`: `publication_or_revision_date > planner_cutoff_date and <= literature_end_date`; `planner_cutoff_date`: `2026-09-07`; `literature_end_date`: `2026-09-07`; `post_cutoff_arxiv_result_count`: `0`; `adoption_rows`: `[]`; `v623_sota_ingestion_complete_score`: `1`; `verdict_class`: `positive`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_7099_v623_adapter_withheld_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7100_adapter_withheld_arc_loo_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite rather than an experimental result. Any implied claim that the measurement succeeded would be refuted by blockage before execution.

## WAS THAT CHECKED
No; the measurement was stopped at the pre-gate, so no comparative or leave-one-game-out result was evaluated.

## EVIDENCE
`status` = `blocked`; `honest_verdict` = `blocked_gate_check_failed`; `failed_observed` = `0`; `passed` = `false`; `blocked_at_layer` = `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7105_v623_exact_constraint_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or verifier-value claim is made; a failed construction, leakage, uniqueness, mutation, or witness-replay gate would refute only the operational readiness receipt.

## WAS THAT CHECKED
Yes. The artifact reports explicit readiness gates, conflict checks, leakage checks, uniqueness checks, mutation attacks, and fresh-process witness replay.

## EVIDENCE
`honest_verdict`: `complete: exact constraint stream ready; exact solvers measure fixture integrity only`; `verdict_class`: `circular_positive`; `verifier_is_oracle`: `true`; `inference_substrate_class`: `no_model_load`; `exact_constraint_stream_ready_score`: `1`; `gate_check_summary`; `passed`: `true`; `fresh_process_match`: `true`; `label_matches`: `true`; `witness_matches`: `true`

## RECOMMENDATION
KEEP

## experiment_7106_v623_procedural_memory_csl.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Delayed procedural memory provides positive later value over the predeclared controls while preserving protected retention.

## WHAT WOULD REFUTE IT
A nonpositive later paired delta or confidence interval reaching zero against a serious baseline such as bounded raw trace, or failure of protected-retention probes in the delayed-procedural arm.

## WAS THAT CHECKED
Yes. Middle-and-late paired comparisons tested four frozen controls, including raw trace; the delayed arm could lose and did lose five events to equal-context replay, yet every aggregate delta and confidence-interval lower bound remained positive. Protected-retention rows separately tested forgetting.

## EVIDENCE
The headline is `complete: delayed procedural memory has positive later value with protected retention`. The inference substrate is `deterministic prospective candidate policy with verifier-signed bounded external memory`, with `inference_substrate_class` equal to `no_model_load` and `verifier_is_oracle` equal to `false`. For `delayed_procedural_vs_raw_trace`, the `middle_and_late` comparison reports `event_count` `96`, `wins` `44`, `losses` `0`, `ties` `52`, and `mean_delta` `0.458333`; its interval has `lower` `0.319723`. For `delayed_procedural_vs_equal_context_replay`, it reports `wins` `59`, `losses` `5`, and `mean_delta` `0.5625`, demonstrating that adverse outcomes were possible and retained. The shown protected probe reports `arm` `delayed_procedural`, `correct_count` `3`, `probe_count` `3`, and `retention_passed` `true`. Decision rows report `exact_feedback_visible_at_decision` `false` and `future_label_accessed` `false`.

## RECOMMENDATION
KEEP

## experiment_7107_v623_continual_memory_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The continual-memory cold audit is complete and ready, while upstream memory value remains independently classified.

## WHAT WOULD REFUTE IT
Any required readiness component failing—such as replay or parity disagreement, target-arm retention failure, future-label leakage, an undetected mutation, or incorrect crash recovery—would refute audit readiness.

## WAS THAT CHECKED
Yes. The artifact checks replay reconstruction, producer–auditor parity, future-label isolation, retention, transaction recovery, and adversarial mutations. Failures were representable, and the serious equal-context comparator recorded five wins over the tested method rather than being structurally unable to win.

## EVIDENCE
`honest_verdict`: `complete: continual memory cold audit ready; upstream value remains independently classified`

`continual_memory_cold_audit_ready_score`: `1`

`observed_value`: `all checks pass`; `failed_check`: `null`; `passed`: `true`

`verifier_is_oracle`: `false`

`current_event_feedback_used`: `false`; `future_label_accessed`: `false`

`fresh_process`: `true`; `llm_disabled`: `true`; `network_disabled`: `true`

`comparison`: `delayed_procedural_vs_equal_context_replay`; `wins`: `59`; `losses`: `5`; `ties`: `32`

`retention_passed`: `false`

## RECOMMENDATION
KEEP

## experiment_7108_v623_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The capstone claims only that the twelve-task V623 evidence matrix is complete and auditable, not that a method beats a rival.

## WHAT WOULD REFUTE IT
A missing or duplicate required artifact, a task-count or contract-order mismatch, absent required fields, or a disposition that promoted value despite failed prerequisites would refute the procedural completeness claim.

## WAS THAT CHECKED
Yes: task counts, artifact presence, contract parity, required fields, gate replay, and dispositions were checked. Failed upstream checks were recorded as blocked or disqualified rather than converted into method-value claims. No comparative method claim or rival arm exists to test.

## EVIDENCE
`complete_positive_v623_evidence_matrix`; `inference_substrate_class`; `aggregation`; `expected_task_count`; `12`; `observed_task_count`; `12`; `completion_separate_from_value`; `true`; `value_promoted`; `false`; `disposition`; `disqualified`; `blocked`; `do not infer energy value from absent rows`

## RECOMMENDATION
KEEP
