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
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 4 |

## experiment_4381_biprm_detector_localization_abstention.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The experiment establishes a clean, powered null: bidirectional fusion does not provide actionable localization and abstention.

## WHAT WOULD REFUTE IT
A strictly positive localization-gain confidence interval together with a useful abstention operating point would refute non-actionability. Both outcomes must have a real chance of occurring for their absence to establish an informative null.

## WAS THAT CHECKED
No meaningful opportunity was provided. The corrigendum identifies degenerate localization labels: every error starts at step zero, and the matches are exactly the one-step traces. All three directions tie, but this does not establish a powered null on meaningful localization targets.

The abstention criterion also requires accuracy above the possible ceiling given the stated base rate. Its failure therefore cannot demonstrate that abstention lacks practical value.

## EVIDENCE

- `honest_verdict`: `complete: clean_powered_null_bidirectional_not_actionable`
- `corrigendum_2026_11_07`: `Input step labels are outcome labels copied to every prefix; first error is step 0 in all 114 error traces; F1 0.096 equals 11/114 one-step traces. Null verdict is uninformative (degenerate labels). See ops/known-issues.md 2026-11-07 and docs/research-notes/label-provenance-audit-step1-2026-11-07.md.`
- `bidirectional_fusion`, `causal_online`, `unidirectional_l2r`: `f1`: `0.096491`
- `base_rate_fraction_correct`: `0.98259`
- `useful_criterion`: `retained_accuracy >= base_rate + 0.05 at coverage >= 0.5`
- `useful_operating_point`: `null`

## RECOMMENDATION
CORRECT_THE_RECORD

Artifact-only review; no rerun. Risk: the terminal verdict still presents the acknowledged uninformative null as powered.

## experiment_8897_v761_source_bound_decisions.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The source-bound decision verifier warrants a positive experimental verdict.

## WHAT WOULD REFUTE IT
A direct signed-record evaluator plus typed-source equality check matching the verifier’s results would refute added value. Separately, accepting a corrupted ledger as complete would falsify the narrower implementation contract.

## WAS THAT CHECKED
No independent test of added value is shown. Ledger validation, provenance rejection and installed-callable checks support execution correctness, but the verifier defines the correctness criterion.

The displayed complete-view rows return support across all three arms; this does not establish a corpus-wide tie. The administrative scope disclaimer appropriately limits the measurements, but cannot turn oracle agreement into evidence supporting the positive verdict.

## EVIDENCE
- `honest_verdict`: `complete_circular_positive_source_bound_decisions`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `CPU exact signed-record decisions and typed-ledger equality`
- `source_boundary_ready_score`: `1`
- `Administrative parity and provenance gates only; no scientific benefit or release claim.`
- `provenance_rejection`: `true`

## RECOMMENDATION
NARROW_CLAIM

Not checked: elided rows or external code. Risk: the positive verdict can overstate what the execution receipt establishes.

## experiment_8898_v761_signed_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit did not establish the prespecified joint benefit of the signed matched-label mechanism on the constructed streams.

## WHAT WOULD REFUTE IT
All signed information and retention gates passing, including a future-Brier difference upper bound below −0.01 against the frozen signed head, would contradict the reported null. Feature-specific benefit additionally requires a strictly negative signed-minus-old upper bound.

## WAS THAT CHECKED
Yes. The decisive signed-versus-frozen gate reports an upper bound of 0.006226604391033034 against a required threshold of −0.01, so the benefit conjunction fails. Keeping the signed head frozen is the cheapest serious baseline, and it is included.

Oracle-defined truth does not invalidate this scoped negative finding; the artifact claims neither verifier-added value nor generalization. No invalid-row or missing-key contradiction is visible. Elided rows prevent a complete independent inspection, but the displayed failed gate suffices to support failure to qualify—not proof that benefit is impossible.

## EVIDENCE

- `honest_verdict`: `complete_null_signed_benefit_audit`; `verdict_class`: `null`.
- `failing_gate`: `feature_arm`: `signed`; `comparison`: `matched_true_minus_frozen_verified`; `endpoint`: `brier`; `upper`: `0.006226604391033034`; `threshold`: `-0.01`; `passed`: `false`.
- `label_benefit_score`: `0`; `feature_specific_benefit_score`: `0`.
- `independent_count`: `24`; `excluded_count`: `0`; `censored_count`: `0`.
- `exposure_scope`: `Constructed signed-record truth only; no natural-language generalization or generator training.`

## RECOMMENDATION
KEEP

## experiment_8899_v761_buffered_cost_attribution.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt making no comparative performance, attribution, or verifier-value claim.

## WAS THAT CHECKED
No empirical comparison ran: all 3,456 intended rows are censored, with zero completions. Injected clock controls were checked, but real-run attribution remains untested. The invalid rows and zero placeholders are not reported as a positive result.

## EVIDENCE
- `honest_verdict`: `complete_blocked_buffered_cost_attribution`
- `verdict_class`: `blocked`
- `inference_substrate`: `blocked_no_run`
- `completed_count`: `0`
- `censored_count`: `3456`
- `cost_attribution_ready_score`: `0`
- `scope`: `injected_reduction_only`

## RECOMMENDATION
KEEP

## experiment_8900_qwen_source_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked-run receipt, with no extraction-performance claim. Its blocking diagnosis would be contradicted if all evaluated gates passed.

## WAS THAT CHECKED
Yes. The gate rows show one failure and one pass, supporting the blocking diagnosis. Extraction performance was not tested.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"failed_field": "runtime_repair_ready_score"`
`"failed_expected": 1`
`"failed_observed": 0`
`"passed": false`
`"passed": true`
`"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_8901_v761_signed_adaptation_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The constructed CPU workload has complete adaptation-cost accounting and exact restart parity, without an established signed-feature speed benefit.

## WHAT WOULD REFUTE IT
A family-adjusted paired cost-ratio interval wholly below 1 would overturn the cost null. An incomplete measured cell, omitted required work, or restart-state mismatch would refute operational readiness.

## WAS THAT CHECKED
Yes. The four old-versus-signed comparisons in per_stream_results each use 12 paired streams; every reported ratio interval includes 1 and every difference interval includes zero. Completion counts and restart receipts support readiness.

The old-feature deployment supplies a substantive comparator. Oracle-defined labels do not make elapsed-cost measurements circular, and no verifier-value or generalization benefit is claimed. The separate repository-health failure does not contradict this scoped result.

## EVIDENCE
- `honest_verdict`: `complete_null_signed_adaptation_cost`
- `scope`: `complete host CPU costs; no efficiency-benefit headline`
- `speed_claim`: `false`
- `frozen_verified`: `lower_ratio`: `0.7957945746343944`; `upper_ratio`: `1.1975095540080383`
- `matched_true`: `lower_ratio`: `0.6530818954642532`; `upper_ratio`: `1.5408937013635962`
- `matched_permuted`: `lower_ratio`: `0.7488985944061255`; `upper_ratio`: `1.2488903032704342`
- `frozen_query_only`: `lower_ratio`: `0.6452860830569456`; `upper_ratio`: `1.2552740990646862`
- `completed_count`: `96`; `intended_count`: `96`; `excluded_count`: `0`
- `restart_receipts`: `exact_state`: `true`; `passed`: `true`

## RECOMMENDATION
KEEP

Unchecked: elided rows and implementation. The broad intervals do not establish cost equivalence.

## experiment_8902_v761_hardware_operation_map.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. An admitted compatible board execution showing a gain over the existing CPU path, including host, transfer and durability costs, would overturn the reported absence of qualified acceleration evidence.

## WAS THAT CHECKED
No physical comparison was performed. This is an operation-accounting audit that reports blocked hardware qualification. Rejected attribution remains diagnostic, speed bounds remain unknown, and unchanged shared-policy label costs are preserved. It asserts neither verifier value nor independent generalization.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_operation_map`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `physical_acceleration`: `passed`: `false`
- `measured_board_time_s`: `null`; `end_to_end_gain_bound`: `null`
- `fixed_total_labels`: `4800`; `sequential_total_labels`: `4800`; `actual_joint_research_labels`: `4800`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream contents; this assessment covers only the supplied text.

## experiment_8903_v761_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific headline is asserted. An unaccounted scheduled slot or a disqualified producer counted as qualified would contradict the administrative accounting.

## WAS THAT CHECKED
Yes, for the visible rejected producer: its failed admission is preserved. The capstone reports blocked status and failed natural generalization rather than promoting upstream circular results into scientific success. Full accounting cannot be assessed from the elided, truncated text.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v761_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `activation_authorized`: `false`
- `natural_generalization_score`: `0`
- `disposition`: `rejected_producer`
- `next_decision`: `Keep qualified signed-feature methods separate from benefit`
- `historical_publication_scope`: `historical FoVer paper only`

## RECOMMENDATION
KEEP

No rerun or upstream inspection; omitted branches remain unreviewed.
