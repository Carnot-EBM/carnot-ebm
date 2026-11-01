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

## experiment_8659_v743_complete_source_comparison.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Streaming extraction increases complete source coverage over the original capped extractor on this fixed roster while preserving existing features and meeting resource bounds.

## WHAT WOULD REFUTE IT
Zero completed recoveries among the 55 former abstentions, missing source spans, changed features on previously usable responses, chunk-dependent results, or resource-limit violations.

## WAS THAT CHECKED
Yes, as reported by the recovery counts, parity checks, window custody, paired resource measurements, and acceptance gates. The artifact reports 55 recoveries and exact feature parity on 135 previously usable responses. That tie establishes preservation; it does not contradict increased coverage.

The capped extractor is the relevant baseline for this limited comparison. No verifier-value or generalization claim is needed. No invalid-row inclusion or wrong-key contradiction is shown, although most individual rows are elided.

## EVIDENCE

- `coverage_gate`: `no missing source spans and at least one of55 former abstentions becomes complete`
- `recovered_count`: `55`
- `old_domain_parity`: `exact eight feature values on all135 originally usable rows`
- `common_parity`: `true`; `chunk_parity`: `true`; `complete_window_custody`: `true`; `resource_bounds`: `true`
- `coverage_benefit_score`: `Absolute engineering reach improvement only; never changed-denominator detector benefit.`
- `independent_generalization_score`: `0`
- `excluded_count`: `0`; `failed_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8660_v743_installed_compatibility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative value claim is made. The limited deployment receipt would be invalidated by an installed response disagreeing with its baseline, an export-contract violation, or failed recovery or cleanup.

## WAS THAT CHECKED
Yes, for deployment compatibility: installed parity, export compatibility, rejection and cleanup rows provide opportunities for failure, and the displayed checks pass. Matching the baseline is the compatibility objective; it does not claim superiority. The oracle flag therefore does not establish a circular added-value claim here.

## EVIDENCE

- `exposure_scope`: `Exposed compatibility operands and synthetic public verification smoke; deployment only`
- `Bind actual immutable invocation evidence; ownership and packaging grant no scientific benefit.`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `installed_measurement_ready_score`: `1`
- `installed_parity_rows`, `baseline_response`, `response`, `passed`
- `honest_verdict`: `complete_circular_positive_v743_installed_compatibility`

## RECOMMENDATION
KEEP

## experiment_8661_v743_installed_complete_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On the exposed original64 workload, installed native execution meets packaging noninferiority but fails the 10× NFR-01 complete-operation throughput requirement.

## WHAT WOULD REFUTE IT
At L16, packaging throughput lower95 below 0.95, p95 latency ratio above 1.05, or a decision mismatch would defeat packaging qualification; installed-native/installed-Python throughput lower95 reaching 10 would contradict the reported NFR-01 failure.

## WAS THAT CHECKED
Yes. The paired throughput intervals report packaging lower95 of 0.9918997155865926 and p95 latency ratio of 1.002628040443349. The arithmetic interval is approximately 1.197–1.225, far below 10.

These timing comparisons could fail independently of the verifier’s decisions. The oracle flag therefore does not make this timing claim circular. The artifact explicitly limits exposure and makes no generalization claim. Private native and installed Python are relevant comparators; a packaging tie is consistent with noninferiority.

The artifact reports all 120 cells completed without failures, censoring, or exclusions. Displayed session rows report exact decisions; most individual rows are elided, limiting direct inspection.

## EVIDENCE

- `honest_verdict`: `complete_null_v743_installed_complete_cost`; `verdict_class`: `null`
- `Only L16 installed_native/installed_python lower95>=10 satisfies NFR-01.`
- `L16 installed_native/private_native lower95>=0.95, p95<=1.05 and exact decisions.`
- `packaging`: `lower95`: `0.9918997155865926`; `p95_latency_ratio`: `1.002628040443349`
- `arithmetic`: `lower95`: `1.197007408532037`; `upper95`: `1.2247656009276773`
- `exposure_scope`: `original64 exposed development requests; original checkpoint; no independent generalization or hardware claim`
- `completed_count`: `120`; `failed_count`: `0`; `censored_count`: `0`; `excluded_count`: `0`
- `exact_decisions`: `true`

## RECOMMENDATION
KEEP

## experiment_8662_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No method-value claim is made. The blocking diagnosis would be contradicted if all evaluated prerequisite gates passed.

## WAS THAT CHECKED
Yes. The gate rows show runtime readiness failed while input readiness passed, consistent with blocking. This is a gate-check receipt, not an experimental result.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`artifact_field`: `runtime_repair_ready_score`  
`reason`: `actual=0 == expected=1`  
`artifact_field`: `canary_inputs_ready_score`  
`reason`: `actual=1 == expected=1`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8663_v743_arc_reset_provenance.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to refute. The bounded engineering witness could fail if an enabled reset lacked its provenance join or enabling instrumentation changed actions, observations, RNG, SDK calls, or termination.

## WAS THAT CHECKED
Yes, within the bounded witness: the enabled arm reports one reset and one join, both arms pass, and the action-parity checks all pass. The equal action counts do not establish policy benefit; none is claimed.

## EVIDENCE

- `methodology_note`: `Bounded engineering witness only. Historical disqualification and quarantines remain unchanged. No solve, policy benefit or transfer-panel qualification. Exp8664 owns the full panel.`
- `honest_verdict`: `complete_null_arc_reset_provenance`
- `independent_scientific_questions`: `0`
- In `sdk_witness_rows`: `arm` = `on`, `reset_count` = `1`, `joined_count` = `1`, `passed` = `true`.
- In `action_parity_rows`: `actions`, `observations`, `rng`, `sdk_calls`, and `termination` are `true`.

## RECOMMENDATION
KEEP

## experiment_8664_v743_arc_reset_transfer.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8665_v743_hardware_operation_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. An admitted, authenticated compatible hardware execution with complete costs would contradict the reported blocked evidence status.

## WAS THAT CHECKED
No hardware benefit test was performed. This is an evidence audit: execution gates remain false, failed source checks are recorded, and speedup estimates remain null. These limitations are reflected in the blocked verdict.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v743_hardware_operation_evidence`
- `compatible_physical_execution`: `false`
- `complete_new_cost_boundaries`: `false`
- `device_commands_executed`: `0`
- `achieved_speedup`: `null`
- `missing_reason`: `binding_residual_does_not_isolate_spline_arithmetic`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8666_v743_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No headline claim of comparative efficacy is advanced. Qualified current learning evidence, independent generalization results, and authenticated physical execution could overturn the blocked readiness assessment.

## WAS THAT CHECKED
Yes, as readiness checks: learning lacks a current producer, learning and reset qualification are false, and every listed board reports no physical execution. These limitations remain reflected in the blocked verdict; administrative completion is not promoted into demonstrated benefit.

## EVIDENCE
- `exposure_scope`: `Previously exposed registered evidence only; administrative reconciliation`
- `honest_verdict`: `complete_blocked_v743_capstone`
- `current_producer_present`: `false`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `physical_execution_this_invocation`: `false`
- `promotion_ready_score`: `0`

## RECOMMENDATION
KEEP
