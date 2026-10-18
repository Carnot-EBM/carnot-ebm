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
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8632_complete_source_comparison.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked prerequisite check and makes no claim about streaming source coverage, cost, or comparative value.

## WAS THAT CHECKED
No coverage or cost comparison was performed. The prerequisite was checked in gates_evaluated: streaming readiness was 0 against required 1, blocking execution.

## EVIDENCE
`blocked_gate_check_v1`; `blocked`; `blocked_gate_check_failed`; `streaming_extractor_ready_score`; `actual=0 == expected=1`; `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8633_v741_installed_lean_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An installed response differing from its corresponding original-worker response, or a failed import/export check, would invalidate this mechanical compatibility receipt. No comparative benefit is asserted.

## WAS THAT CHECKED
Yes, for mechanical compatibility: actual private and installed executions were compared, with parity, import, and export checks. Visible rows pass; totals report 128 completed and zero failures or exclusions.

Matching the original worker is the intended compatibility result. The oracle flag does not establish circular added-value reasoning because no verifier benefit is claimed. Exposed requests are explicitly treated as compatibility operands, with zero generalization credit.

## EVIDENCE
- `methodology`: `Compare actual private and installed Python/native children on identical original64 frames. No fitting, generation, speed or confidence claim.`
- `installed_prototype_ready_score`: `One only when original64 Python/native comparisons, imports, exports and all owned checks pass. No surface qualification or speed credit.`
- `exposure_scope`: `original exposed64 research requests; mechanical compatibility only`
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`
- `prototype_parity_rows`: `parity`: `true`; `status`: `ok`
- `import_trace_rows`: `clean`: `true`; `clean_shutdown`: `true`
- `export_contract_rows`: `passed`: `true`
- `completed_count`: `128`; `failed_count`: `0`; `excluded_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8634_v741_installed_compatibility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An installed response differing from its private baseline would refute parity; a failed mandatory gate would defeat readiness. The artifact makes no comparative value or generalization claim.

## WAS THAT CHECKED
Yes, for deployment compatibility. The displayed response pairs match, and public-contract and rejection/recovery checks are recorded. The mandatory node-union check fails, and the artifact reports disqualification with zero readiness. Baseline ties serve the parity check; they are not presented as superiority. Oracle status and synthetic fixture fitting do not establish added value, but none is claimed.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_installed_compatibility`
- `installed_measurement_ready_score`: `0`
- `required_checks_passed`: `false`
- Within `original_node_union`, `name`: `exact_node_union`; `passed`: `false`
- `exposure_scope`: `Exposed compatibility operands and synthetic public verification smoke; deployment only`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `Bind this invocation to immutable primitives; compatibility supplies no independent scientific credit.`

## RECOMMENDATION
KEEP

## experiment_8635_installed_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt. Its title describes an intended measurement, not a comparative finding.

## WAS THAT CHECKED
No inference-cost comparison is reported. The prerequisite check ran in `gates_evaluated` and failed: installed readiness was 0 against a required 1. The artifact records that failure explicitly.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `installed_measurement_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8636_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check and makes no experimental performance or added-value claim.

## WAS THAT CHECKED
The prerequisite was checked: runtime readiness was 0 against a required 1, so execution was blocked before the experiment. This supplies no comparative result to assess.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8637_v741_arc_observer_transfer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Observer-on and observer-off matched in this run, while causal benefit and independent generalization remained unestablished.

## WHAT WOULD REFUTE IT
An on/off mismatch in actions, observations, RNG, legacy records, or termination would refute parity; an eligible outcome establishing independently attributable benefit would contradict the reported null.

## WAS THAT CHECKED
Yes for parity: all eight paired comparisons explicitly check those quantities and pass. Observer-off is the appropriate cheap comparator and ties throughout.

Benefit was not established: all 16 redirect windows are censored, with zero complete horizons per game and no causal attribution. This supports the bounded null reporting claim, not a conclusion that benefit cannot occur.

## EVIDENCE
- `honest_verdict`: `complete_null_arc_observer_transfer`
- `observer_action_parity_rows`; `equality`; `actions`, `legacy`, `observations`, `rng`, `termination`: `true`
- `censored_count`: `16`; `full_horizons`: `0`
- `classification`: `explicitly_censored`; `causal_attribution`: `null`
- `headline_solve_credit`: `0`; `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`; `promotion_ready_score`: `0`
- `exposure_scope`: `historically_exposed_public_games_adapters_withheld`
- `hidden_game_performance`: `unmeasured`

## RECOMMENDATION
KEEP

## experiment_8638_v741_hardware_operation_delta.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8639_v741_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. Qualified current execution satisfying the required gates would contradict the corresponding blocked-status report. Missing execution cannot establish that a method has no benefit.

## WAS THAT CHECKED
The blocked status is supported by explicit gate failures: expected readiness scores of 1 are observed as 0, and skipped producers are identified. Comparative benefit was not tested in those branches. Historical and prototype measurements are explicitly scoped or unqualified; they are not promoted into current generalization or verifier-value claims.

## EVIDENCE
- `administrative_complete`: `true`
- `branch_outcomes` → `learning`: `blocked_no_current_execution`
- `branch_outcomes` → `native_cost`: `blocked_no_current_complete_cost`
- `acceptance_gates` → `qualified_learning`: `false`
- `acceptance_gates` → `packaging_noninferiority`: `false`
- `current_language_model_calls`: `0`
- `numerical_inference_this_invocation`: `false`
- `current_learning` → `observed`: `null`
- `streaming` rows: `qualified`: `false`; `semantic_correctness_established`: `false`

## RECOMMENDATION
KEEP
