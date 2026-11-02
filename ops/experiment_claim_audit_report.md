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
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 1 |

## experiment_8729_v748_equal_budget_verifier.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['; both have '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Targeted verification adds defect-detection value over baseline at an equal five-call budget.

## WHAT WOULD REFUTE IT
Using independently established correctness, targeted verification detects no more defects than baseline, or gains detections by rejecting valid alternatives or mishandling unknowns. The cheapest serious rival is direct use of the defining oracle, with its cost counted; a tie would undermine added value.

## WAS THAT CHECKED
No for independent verifier value; yes for the narrower oracle-relative schedule comparison. The arm summaries report 130/150 versus 120/150 detections, zero errors on the correct variants’ alternatives and unknowns, and a positive paired interval.

However, the verifier itself defines correctness. Those measurements therefore cannot independently establish its added value. No independent adjudication or direct-oracle comparator is shown. The artifact explicitly disclaims generalization, so exposure is not an additional finding. The displayed failing defective variants are detection outcomes, not evidence of invalid rows.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_v748_equal_budget_verifier`
- `verifier_is_oracle`: `true`
- `inference_substrate`: `deterministic_verifier`
- `cost_matched_detection_gain_score`: `1`
- `baseline`: `defects_detected` = `120`; `targeted`: `defects_detected` = `130`; both have `defective_denominator` = `150`.
- Both schedules have `budget` = `5`.
- Both arms have `correct_valid_alternative_rejections` = `0` and `correct_unknown_to_false_errors` = `0`.
- `lower95`: `0.03333333333333333`
- `exposure_scope`: `All original120 worlds are exposed-development; retain original90 development/30 evaluation IDs; no independent generalization`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8730_qwen_self_score_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records a blocked prerequisite check and asserts no comparative outcome.

## WAS THAT CHECKED
No comparison results are reported. The prerequisite checks detected a failed runtime-readiness gate and recorded the block.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8731_v748_resident_sparse_prototype.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The resident sparse prototype preserves reference arithmetic, counters, and restart state on the tested lifecycle inputs within its stated domain.

## WHAT WOULD REFUTE IT
Any native/reference arithmetic, integer-state, counter, or historical-state mismatch within that domain, or a restarted run producing different state, acknowledgements, or checkpoint bytes.

## WAS THAT CHECKED
Yes—in the native parity, boundary, Python/native arithmetic, and restart comparisons. The displayed rows pass, including restarts after actual process exits. Implementation divergence could have failed these checks.

Reference ties support the claimed equivalence. The verifier defines the reference target, but the artifact makes no claim about its added value. Elided rows limit direct inspection of the complete results.

## EVIDENCE
- `exposure_scope`: `same_original_exposed_lifecycle_inputs_language_port_parity_only`
- `native_parity_rows` and `boundary_rows`: `integer_state_identical`, `counters_identical`, and `historical_identical` are `true`.
- `rejection_rows`: matching `native` and `python` values include `-4` and `-2`.
- `restart_rows`: `real_exit` is `73`; `state_acknowledgement_and_bytes_identical` is `true`.
- `Bind eligibility to exact sequential arithmetic, resident counters and durable restart; no learning or performance claim.`

## RECOMMENDATION
KEEP

## experiment_8732_v748_resident_update_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. A resident-benefit claim would be refuted by valid matched lifecycle timings showing resident_native tying or losing to json_native; speedup below 10× would refute a 10× claim.

## WAS THAT CHECKED
No comparative measurement completed: all 30 planned rows were censored. The budget gate was checked and failed, with projected measurement time of 71,507.5252224 seconds against a 3,600-second cap. The artifact reports a blocked run and asserts no resident benefit. The verifier’s oracle role therefore creates no circular value claim here.

## EVIDENCE
- `honest_verdict`: `complete_blocked_durability_budget`
- `verdict_class`: `blocked`
- `completed_count`: `0`
- `censored_count`: `30`
- `resident_benefit`: `false`
- `complete_10x`: `false`
- `check`: `complete_roster_fits_measurement_cap`
- `observed`: `71507.5252224`
- `expected`: `3600`
- `passed`: `false`

## RECOMMENDATION
KEEP

## experiment_8733_v748_installed_cost_attribution.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Installed B16 runtime costs can be attributed with low instrumentation overhead while preserving same-arm outputs on the fixed exposed panel.

## WHAT WOULD REFUTE IT
Instrumented runs taking twice as long as matched uninstrumented runs, changing scientific outputs, or failing to complete would refute this claim.

## WAS THAT CHECKED
Yes. Instrumented and uninstrumented runs provide a real timing comparator across ten paired repeats per arm. The reported overhead upper confidence bounds are approximately 2.89% for Python and 3.57% for native. The displayed Python pair has identical request and output digests; the methodology reports exact same-arm outputs throughout.

All forty processes reportedly completed without failures or exclusions. Timing overhead could have been large independently of output correctness, so success was not guaranteed by construction. The numerical overhead acceptance cutoff is not visible in this excerpt. The claim concerns runtime diagnostics, with no generalization or measured optimization benefit asserted.

## EVIDENCE
- `honest_verdict`: `complete_positive_installed_cost_attribution`
- `low_overhead_attribution`; `passed`: `true`
- `instrumentation_overhead_interval`; `paired_repeats`: `10`; `upper95`: `0.0356792380507134`
- `instrumented`: `false`; `instrumented`: `true`; matching `output_digest`: `sha256:894dd4674ae5d127fb05d76a0ebd5b857c2e26120e4dfbed5dd42d08b7abefcf`
- `completed_count`: `40`; `failed_count`: `0`; `excluded_count`: `0`
- `exposure_scope`: `fixed exposed original64 panel; runtime diagnostic only`
- `measured_improvement`: `false`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_8734_v748_arc_outcome_frontier.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ledger update contains no new supervisor outcomes and supplies no new support for changing policy.

## WHAT WOULD REFUTE IT
An authenticated supervisor outcome with an event identity outside the retained frontier would refute the no-new-outcomes finding. Qualifying uncensored cross-game intervention results would contradict the reported absence of usable policy-change support.

## WAS THAT CHECKED
Yes, within the ledger’s scope. The stated method compares exact event identities; the before/after frontiers match, the new-event list is empty, and all four decision rows report zero usable cross-game support. This supports the narrow reporting null, without establishing supervisor effectiveness or ineffectiveness.

## EVIDENCE
- `honest_verdict`: `complete_null_no_new_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `new_event_rows`: `[]`
- `new_events`: `0`
- `usable_cross_game_support`: `0`
- `policy_decision`: `no_supported_policy_change`
- `generalized_learning_benefit_score`: `0`
- `solve_provenance_scope`: `reader_credits_no_live_outcomes`

## RECOMMENDATION
KEEP

## experiment_8735_v748_hardware_cost_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. Authenticated compatible physical execution with qualified complete learning and update costs would overturn the reported blocked readiness status.

## WAS THAT CHECKED
Yes, as readiness checks in gate_check_summary and source_admission_rows: learning evidence is missing, update evidence is unqualified, and physical readiness fails. Those producers remain censored rather than supporting a positive result. This aggregation asserts neither measured acceleration nor verifier added value nor generalization, so the oracle flag creates no circular value claim here.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_cost_or_physical_execution`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `physical_hardware_ready_score`: `0`
- `operand_state`: `missing_file`
- `qualified`: `false`
- `censored_count`: `2`
- `achieved_speedup`: `null`
- `complete_100x_learning_ready_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8736_v748_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The available evidence leaves the capstone blocked, without establishing complete 10× acceleration or independent generalized learning benefit.

## WHAT WOULD REFUTE IT
Qualified, authenticated measurements meeting the unmet acceptance gates—for example, a same-batch speedup with a paired 95% lower bound of at least 10× contradicting that gate’s reported failure.

## WAS THAT CHECKED
Yes, for this limited conclusion. Deployment rows compare measured intervals against explicit thresholds; the B16 native/Python interval spans parity. Gate rows reject an upstream positive score when its producer is disqualified and distinguish missing files from observed zeros. Exposed evidence is explicitly excluded from independent-generalization claims. These rows support withholding qualification, rather than concluding benefits are impossible.

## EVIDENCE
- `same_batch_threshold`: `paired lower95 >= 10`
- `contrast`: `native_python_B16`; `lower95`: `0.998091298984346`; `upper95`: `1.0138702202694783`
- `complete_10x_score`: `0`; `generalized_learning_benefit_score`: `0`; `activation_authorized`: `false`
- In `gate_edge_rows`: `observed`: `1`; `producer_qualified`: `false`; `passed`: `false`
- `operand_state`: `missing_file`; `observed`: `null`
- `exposure_scope`: `Constructed and exposed upstream evidence; no natural independent generalization`

## RECOMMENDATION
KEEP
