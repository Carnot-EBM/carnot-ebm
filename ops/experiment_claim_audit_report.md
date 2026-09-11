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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |

## experiment_7210_span_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked pre-gate check and makes no outcome or comparative claim.

## WAS THAT CHECKED
No; the experiment did not proceed beyond the prerequisite gates, so no method result was tested.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_reason": "actual=0 == expected=1"`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7212_v635_refinement_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No learning-value claim is made to falsify; an operational fixture-readiness assertion would be refuted by failed contract checks, stream-conformance errors, incomplete units, or failed commit/rollback behavior.

## WAS THAT CHECKED
Yes, operational readiness was checked through the gate summary, stream-conformance field, sample-size accounting, per-arm fixture-contract rows, and commit-path receipt. Learning value was explicitly not checked.

## EVIDENCE
`complete: refinement fixture ready; learning value is unmeasured and V634 nulls remain unpromoted`; `learning_value_measured`: `false`; `metric`: `fixture_contract_pass`; `stream_conformance_errors`: `[]`; `known_failed_value_promoted`: `false`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`

## RECOMMENDATION
KEEP

## experiment_7213_v635_refinement_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Witnessed committed predicates did not pass the fixed primary learning gate.

## WHAT WOULD REFUTE IT
A passed primary learning gate—meaning every required primary criterion satisfied its fixed threshold—would refute the claim.

## WAS THAT CHECKED
Yes, in `acceptance_gate_learning.primary_criteria` and `acceptance_gate_learning.primary_learning_gate_passed`; several criteria could and did pass, while two required criteria failed.

## EVIDENCE
`honest_verdict` `complete_null: witnessed committed predicates did not pass the fixed primary learning gate`  
`primary_learning_gate_passed` `false`  
`future_error_upper_vs_random_below_zero` `0.00015120967741934915` `passed` `false`  
`false_accept_upper_vs_warmup_nonpositive` `0.002368951612903227` `passed` `false`  
`future_error_upper_vs_warmup_below_zero` `-0.00917338709677419` `passed` `true`  
`verdict_class` `null`

## RECOMMENDATION
KEEP

## experiment_7214_v635_refinement_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent audit completed, but producer refinement showed no added value over a serious passive-query comparator.

## WHAT WOULD REFUTE IT
A prospective comparison against the passive-query committed arm whose 95% confidence interval excluded zero in favor of refinement would refute the null-value claim.

## WAS THAT CHECKED
Yes. The artifact reports an independently recomputed comparison against the passive-query committed arm; its confidence interval crosses zero. The oracle relationship prevents a positive verifier-value claim, but does not invalidate this observed null.

## EVIDENCE
`"honest_verdict": "complete_null: the independent audit completed and producer refinement value was null"`; `"comparison_id": "future_error_change_vs_passive_query_committed"`; `"estimate": -0.00211693548387097`; `"ci95_lower": -0.011139112903225803`; `"ci95_upper": 0.005897177419354836`; `"independent_stream_count": 20`; `"refinement_value_score": 0`; `"verdict_class": "null"`; `"status": "complete"`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7215_v635_down_up_prototype.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
All 90 transition cells and three mutation controls certify the elementary CPU down-up kernel.

## WHAT WOULD REFUTE IT
A transition row exceeding its stationarity, detailed-balance, stochasticity, or empirical-error threshold—or an incorrect mutation escaping detection—would refute execution consistency; independent certification would additionally require a correctness oracle not derived from the verifier itself.

## WAS THAT CHECKED
Yes for execution consistency: 90 transition cells and three deliberately incorrect mutation controls were checked, and the controls were rejected. No for independent certification: the artifact explicitly identifies the verifier as the correctness oracle.

## EVIDENCE
`"honest_verdict"`: `"complete_circular_positive: all 90 finite transition cells and all three mutation controls passed; this certifies the elementary CPU kernel only."`; `"verifier_is_oracle"`: `true`; `"verdict_class"`: `"circular_positive"`; `"completed_transition_cells"`: `90`; `"completed_mutation_cells"`: `3`; `"control_detected"`: `true`; `"passed"`: `false`; `"metric"`: `"target_law_rejection"`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7216_v635_down_up_quality.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All prespecified panels were completed, but the down-up kernel failed the joint exact-fidelity and cost-adjusted quality gate.

## WHAT WOULD REFUTE IT
A passed primary gate—supported by qualified fidelity diagnostics and a cost-adjusted ESS-per-second ratio whose 95% confidence-interval lower bound exceeded 1—or evidence that required panels were incomplete or censored.

## WAS THAT CHECKED
Yes. The artifact reports full planned-versus-completed row counts, zero censored rows, and the primary gate components; the gate failed because chain ESS/R-hat qualification was not achieved, so no paired throughput ratios could qualify. Oracle circularity does not rescue or undermine this null claim because the artifact makes no positive added-value claim.

## EVIDENCE
`"honest_verdict": "complete_null: all prespecified panels were measured, but the joint exact-fidelity and cost-adjusted down-up quality gate did not pass."`; `"down_up_value_score": 0`; `"panel_complete": true`; `"summary_complete": true`; `"passed": false`; `"all_nondegenerate_chain_ess_at_least_200": false`; `"all_split_rhat_at_most_1_05": false`; `"ess_per_second_ratio_ci95": null`; `"paired_graph_ratios": []`; `"planned_exact_authority_rows": 30`; `"completed_exact_authority_rows": 30`; `"planned_matched_budget_rows": 480`; `"completed_matched_budget_rows": 480`; `"planned_quality_rows": 240`; `"completed_quality_rows": 240`; `"censored_exact_authority_rows": 0`; `"censored_matched_budget_rows": 0`; `"censored_quality_rows": 0`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7217_v635_abi_board_readiness.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or performance claim to falsify. The limited readiness receipt would be refuted by a failed selected-extension import, a Rust/Python transition mismatch, or failure to restore the serialized state in the second process.

## WAS THAT CHECKED
Yes, for the limited operational assertions: fresh-process import, cross-language replay, and second-process restoration were all capable of failing and were checked. Oracle circularity does not invalidate an execution receipt because no added-value claim is made.

## EVIDENCE
`hardware_performance_claimed`: `false`; `new_performance_claimed`: `false`; `compiled_execution`: `true`; `python_fallback_used`: `false`; `maximum_energy_error`: `8.881784197001252e-16`; `restore_exit_code`: `0`; `This readiness receipt does not reopen the failed 10x claim or establish device performance.`

## RECOMMENDATION
KEEP

## experiment_7218_v635_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V635 matrix is complete, source-span fidelity and exact-execution value remain blocked, and refinement and mixing show null value.

## WHAT WOULD REFUTE IT
A valid refinement arm outperforming the strong version-space comparator, down/up mixing outperforming the pair-swap baseline, or an unquarantined authenticated span capture enabling an assessable exact-execution result would refute the corresponding headline conclusion.

## WAS THAT CHECKED
Yes. Refinement was compared with the strong version-space rival, which won; down/up mixing tied pair-swap at 0.2; and the span prerequisites were replayed but failed quarantine and capture gates. Exact-execution value itself was not tested because those prerequisites failed, which is consistent with the limited claim that it remains blocked.

## EVIDENCE
`honest_verdict`: `blocked_external: V635 matrix is complete; source-span fidelity and exact-execution value remain blocked, while refinement and mixing value are null`; `capstone_complete_score`: `1`; `refinement_primary_value`: `false`; `strong_version_space_comparator`: `version_space_outperformed_committed_predicates`; `down_up_primary_pass_rate`: `0.2`; `pair_swap_primary_pass_rate`: `0.2`; `source_span_fidelity`: `blocked`; `exact_execution_value`: `blocked`; `failed_check`: `structured_quarantine`; `span_capture_complete_score`: `null`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
