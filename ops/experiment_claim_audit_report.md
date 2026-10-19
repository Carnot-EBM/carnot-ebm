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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8645_complete_source_comparison.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. The title states a measurement objective; this blocked receipt asserts no coverage, cost, or comparative performance result.

## WAS THAT CHECKED
Only prerequisite gates were checked. Streaming readiness was 0 against a required 1, so the experiment was blocked before comparative measurement.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `streaming_extractor_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8646_v742_installed_compatibility.json

**NO_CLAIM**

## VERDICT

NO_CLAIM

## THE HEADLINE CLAIM

no claim

## WHAT WOULD REFUTE IT

A failed required compatibility or node-union check would defeat positive installed qualification. The artifact reports disqualification, not a comparative benefit.

## WAS THAT CHECKED

Yes. The exact-node-union gate and original-node-union result fail; the terminal verdict remains disqualified and readiness remains zero. Oracle-defined correctness and baseline parity serve compatibility checks here. Exposed operands are disclosed, generalization is not claimed, and historical timing results remain null. The failed checks are not promoted into a positive qualification.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v742_installed_compatibility`
- `acceptance_gates`: `name` = `exact_node_union`, `passed` = `false`
- `original_node_union`: `passed` = `false`
- `installed_measurement_ready_score`: `0`
- `exposure_scope`: `Exposed compatibility operands and synthetic public verification smoke; deployment only`
- `preserved_historical_nulls`: `new_timing_comparison` = `false`, `verdict_class` = `null`

## RECOMMENDATION

KEEP

## experiment_8647_installed_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports a blocked prerequisite check, without asserting successful inference measurement or comparative value.

## WAS THAT CHECKED
No inference outcome was tested. The prerequisite checks in `gates_evaluated` detected failed installed readiness and blocked execution.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `installed_measurement_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8648_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, with no comparative or effectiveness claim to falsify.

## WAS THAT CHECKED
No experimental claim was tested. The prerequisite gate was checked and failed: runtime readiness was 0 against a required 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`failed_field`: `runtime_repair_ready_score`  
`failed_expected`: `1`  
`failed_observed`: `0`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8649_v742_arc_reset_provenance.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A changed action, observation, or RNG trace, or missing or ambiguous reset attribution, would refute the narrower recording qualification. This artifact makes no comparative benefit claim.

## WAS THAT CHECKED
Yes, within the fixture scope: all three action-parity rows pass, the displayed requester-ownership rows report unique attribution, and the missing-cause list is empty. These checks could reveal recording failures; they establish no learning or generalization benefit.

## EVIDENCE

- `methodology_note`: `Passive recording qualification only. Fixtures are not real outcomes; Exp8650 owns diagnostic transfer. No reset timing, supervisor priority or horizon changes.`
- `honest_verdict`: `complete_null_arc_reset_provenance`
- `verdict_class`: `null`
- `headline_solve_credit`: `0`
- `independent_count`: `0`
- `action_parity_rows`, `passed`: `true`
- `requester_ownership_rows`, `status`: `unique`
- `missing_cause_rows`: `[]`

## RECOMMENDATION
KEEP

## experiment_8650_v742_arc_reset_transfer.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8651_v742_hardware_operation_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. An authenticated, qualified complete-cost workload in the artifact would contradict its narrower report that none qualified.

## WAS THAT CHECKED
Yes, for evidence admission: all three entries in source_admission_rows are unqualified, and their corresponding cost rows are censored with null values. This is a blocked evidence receipt; it does not establish a performance null or hardware advantage.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_operation_evidence`
- `methodology_note`: `Read-only authenticated cost mapping. No language model, board, or benchmark runs. Qualified complete primitive closures are reduced separately; unavailable costs remain null.`
- `completed_count`: `0`
- `censored_count`: `3`
- `qualified`: `false`
- `status`: `censored`
- `value`: `null`
- `compatible_physical_execution`: `false`
- `complete_new_cost_boundaries`: `false`

## RECOMMENDATION
KEEP

## experiment_8652_v742_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. Unequal paired actions or mismatched reproduced values would refute the narrower parity or evidence-reproduction assertions.

## WAS THAT CHECKED
Yes, those narrower checks appear in the action-parity and reproduction samples. Scientific qualification failures remain explicit, and benefit/generalization readiness stays zero. Oracle-defined correctness, exposed data, and unqualified timing rows are not promoted into comparative value claims.

## EVIDENCE
- `administrative_complete`: `true`
- `learning_benefit_ready_score`: `0`
- `independent_generalization_score`: `0`
- `action_parity_rows`; `passed`: `true`
- `matches_original`: `true`
- `qualified`: `false`
- `qualified_complete_cost`: `false`
- `exposure_scope`: `Previously exposed development responses and engineering fixtures; historical exposure remains unknown`

## RECOMMENDATION
KEEP
