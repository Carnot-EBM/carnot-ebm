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

## experiment_8632_complete_source_comparison.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check, without asserting a coverage, cost, or comparative result.

## WAS THAT CHECKED
Readiness was checked: the streaming extractor gate required 1 and observed 0, blocking execution. The proposed coverage and cost comparison was not performed.

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

## experiment_8633_v741_installed_lean_prototype.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The installed Python/native prototype preserves original-worker responses on the exposed compatibility requests and passes its installation checks.

## WHAT WOULD REFUTE IT
Any installed response differing from its original-worker baseline, an import/export contract failure, or a failed required check would refute the reported readiness.

## WAS THAT CHECKED
Yes. Actual private and installed children were compared in the parity rows; import traces, export-contract rows and validation receipts check installation behavior. Packaging or execution errors could have produced failures.

The original worker is the appropriate compatibility baseline, and equality is the claimed result. The oracle establishes compatibility without establishing verifier value. Exposed requests are explicitly acknowledged, and generalization scores remain zero. Displayed response rows pass; reported failure, exclusion and censoring counts are zero. Some rows are elided, limiting direct inspection.

## EVIDENCE

- `methodology`: `Compare actual private and installed Python/native children on identical original64 frames. No fitting, generation, speed or confidence claim.`
- `completed_count`: `128`
- `baseline_response`; `response`; `parity`: `true`; `status`: `ok`
- `clean`: `true`; `clean_shutdown`: `true`; `required_checks_passed`: `true`
- `failed_count`: `0`; `excluded_count`: `0`; `censored_count`: `0`
- `verifier_is_oracle`: `true`
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8634_v741_installed_compatibility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative value claim to falsify. An installed-readiness claim would be refuted by failed required validation, installed/private response mismatches, or broken public contracts.

## WAS THAT CHECKED
Yes. Installed/private responses and public contracts were checked; the exact-node-union gate failed. The artifact records disqualification and zero readiness.

The matching baseline responses serve the stated parity purpose. Oracle-defined checks and exposed synthetic fixtures are not promoted to verifier benefit or generalization. This is a compatibility receipt reporting failed qualification.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_installed_compatibility`
- `required_checks_passed`: `false`
- `installed_measurement_ready_score`: `0`
- `original_node_union`: `passed`: `false`
- `exposure_scope`: `Exposed compatibility operands and synthetic public verification smoke; deployment only`
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
No comparative result is asserted. The blocking diagnosis would be contradicted if all evaluated gates passed.

## WAS THAT CHECKED
Yes. The evaluated gates record installed measurement readiness as 0 against the required 1, with that gate failing. This is a blocked-run receipt; it provides no inference comparison to assess.

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
Not applicable: this is a blocked-gate receipt, with no performance, generalization, or comparative claim to falsify.

## WAS THAT CHECKED
Yes—the prerequisite was checked in `gates_evaluated`: runtime readiness was 0 against required 1, and the artifact reports the run blocked before execution.

## EVIDENCE
`blocked_gate_check_v1`, `blocked_gate_check_failed`, `runtime_repair_ready_score`, `actual=0 == expected=1`, `blocked`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8637_v741_arc_observer_transfer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This invocation provides no demonstrated causal observer-transfer, solve, or generalization benefit.

## WHAT WOULD REFUTE IT
Valid, uncensored, nonoverlapping outcomes establishing attributable transfer benefit over the observer-off baseline would contradict the null conclusion.

## WAS THAT CHECKED
Yes for the bounded conclusion about demonstrated benefit. The observer-off baseline is present and ties across all eight paired behavior comparisons. All 16 redirect windows are censored, with zero full horizons per game; they receive no causal benefit credit.

No complete-window causal test was achieved, so these results cannot establish that a benefit does not exist. Historically exposed games also cannot establish unseen-game generalization; the artifact awards none. The supplied pre-pass identifies no oracle circularity.

## EVIDENCE

- `honest_verdict`: `complete_null_arc_observer_transfer`
- `methodology_note`: `Natural redirects only; overlapping and censored windows assign no causal arm wins. Fixed budgets are never extended. Instrumentation support grants no solve or priority promotion.`
- `censored_count`: `16`; `full_horizons`: `0`
- `observer_action_parity_rows`: `passed`: `true`; `equality`: `actions`: `true`, `observations`: `true`, `rng`: `true`, `termination`: `true`, `legacy`: `true`
- `descriptive_support`: `false`; `generalized_learning_benefit_score`: `0`; `headline_solve_credit`: `0`
- `exposure_scope`: `historically_exposed_public_games_adapters_withheld`; `hidden_game_performance`: `unmeasured`

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
No comparative headline is asserted to falsify. This capstone accounts for blocked branches; it does not claim that missing measurements establish either benefit or a null effect.

## WAS THAT CHECKED
No current learning or complete-cost benefit test ran. The blocked inputs and deterministic samples explicitly record those limitations rather than declaring comparative success. The verifier’s oracle role therefore does not create a circular value claim here.

## EVIDENCE

- `administrative_complete`: `true`
- `learning`: `blocked_no_current_execution`
- `native_cost`: `blocked_no_current_complete_cost`
- `supervisor_benefit_established`: `false`
- `numerical_inference_this_invocation`: `false`
- `Exp8630 cascade skipped; historical samples are unqualified for current benefit`
- `Exp8635 pre-gated; prototype clocks cannot measure packaging noninferiority`

## RECOMMENDATION
KEEP
