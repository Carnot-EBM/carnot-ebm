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
| CLAIM_SUPPORTED | 6 |
| NO_CLAIM | 2 |

## experiment_7184_v633_revocable_template_csl.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Revocable templates were added and stale versions revoked, but they did not reduce future error versus the static-rule baseline.

## WHAT WOULD REFUTE IT
A paired-block interval for revocable-template minus static-rule future error lying strictly below zero, or rows showing that templates were not actually added and revoked.

## WAS THAT CHECKED
Yes. The paired bootstrap directly compared `revocable_template` with `static_rule`, and the lineage and revocation ledgers recorded the claimed operations. The interval was strictly positive, so the learned method performed worse, not better.

## EVIDENCE
`honest_verdict` `complete_null: templates were added and stale versions were revoked, but future error did not beat the static rule baseline with a strictly negative paired-block interval` `comparison_arm` `static_rule` `target_arm` `revocable_template` `metric` `future_segment_error_delta` `mean_delta` `0.2` `ci95_lower` `0.15` `ci95_upper` `0.255556` `paired_within_block` `true` `operation` `add_template` `operation` `revoke_template` `memory_value_score` `0` `verdict_class` `null`

## RECOMMENDATION
KEEP

## experiment_7185_v633_memory_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit completed successfully, but Exp7184 showed no value because real and shuffled family credit produced identical controller decisions.

## WHAT WOULD REFUTE IT
A nonzero controller-decision difference between real and shuffled family credit, a positive memory-value result, or a failed cold-reload, revocation, rollback, or deletion audit would refute the headline.

## WAS THAT CHECKED
Yes. The credit control held capacity, event stream, and prediction policy constant; cold reload, revocation, rollback, and deletion were audited; and mutation rows demonstrate that audit failures could be detected. The oracle is the verifier, but the headline makes no positive verifier-value claim.

## EVIDENCE
`"verdict_class": "null"`; `"memory_promotion_score": 0`; `"memory_value_score": 0`; `"credit_assignment_difference_count": 7`; `"credit_assignment_effective": false`; `"decision_difference_count": 0`; `"state_hash_difference_count": 0`; `"same_capacity": true`; `"same_event_stream": true`; `"same_prediction_policy": true`; `"changed_decision_count": 54`; `"mechanism_decorative": false`; `"byte_equal": true`; `"fresh_process": true`; `"no_model_load": true`; `"verifier_is_oracle": true`; `"memory_audit_complete_score": 1`

## RECOMMENDATION
KEEP

## experiment_7186_v633_arc_withheld_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked preflight rather than a transfer, generalization, or comparative result.

## WAS THAT CHECKED
No. Model inference and qualifying experimental work did not run, and no result rows were produced.

## EVIDENCE
`honest_verdict`, `blocked_required_source_bytes`, `inference_substrate`, `preflight_only_no_model_load`, `inference_substrate_class`, `blocked_no_run`, `status`, `blocked`, `verdict_class`, `generalization_established`, `false`, `new_solve_claimed`, `rows`, `[]`

## RECOMMENDATION
KEEP

## experiment_7187_v633_slice_sampler.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded pair-swap Metropolis kernel preserves the enumerated fixed-cardinality slice laws and is a feasible CPU baseline.

## WHAT WOULD REFUTE IT
A pair-swap transition row with failed detailed balance, incorrect stationary law, invalid cardinality, non-normalized transitions, or a frozen kernel would refute the claim; failure to detect known implementation mutations would show the checks lacked sensitivity.

## WAS THAT CHECKED
Yes. Exact finite-law and transition rows check cardinality, normalization, detailed balance, stationarity, and freezing. Mutation rows show that asymmetric proposals, edge double-counting, energy-sign reversal, and invalid cardinality are detected. The verifier is not the correctness oracle. The rival kernel ties on law preservation and wins the reported speed metric, but no superiority or speed-win claim is made.

## EVIDENCE
`"bounded_claim": "Validated a feasible pair-swap Metropolis baseline on the fixed CPU roster. No polynomial mixing, specialized-paper-sampler, or hardware claim is made."`; `"arm": "pair_swap_metropolis"`; `"cardinality_valid": true`; `"detailed_balance_error_max": 1.0408340855860843e-17`; `"stationary_law_error": 5.551115123125783e-17`; `"transition_normalization_error_max": 0.0`; `"frozen": false`; `"passed": true`; `"detected": true`; `"verifier_is_oracle": false`; `"speed_win_claimed": false`; `"pair_swap_metropolis": 7528.22458265741`; `"uniform_slice_independence_metropolis": 10887.681287706579`; `"slice_sampler_ready_score": 1`

## RECOMMENDATION
KEEP

## experiment_7188_v633_quantized_transition_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On the tested CPU exact-law corpus, naive quantized-energy MH changed the target in 162/162 conditions, while the two-stage delayed-acceptance kernel preserved the full-precision target.

## WHAT WOULD REFUTE IT
A naive-arm law with zero exact target distance, or a delayed-acceptance law with nonzero exact target distance, moment bias, stationary residual, or material detailed-balance error against the full target.

## WAS THAT CHECKED
Yes. The artifact compares full-precision, naive-quantized, and delayed-acceptance transition laws against the full target using exact enumeration. The full-precision arm is the serious baseline: it ties delayed acceptance on fidelity, as it should, while the artifact does not claim added fidelity or useful acceleration. Refutation was possible because the naive and corrected arms could independently have produced the opposite exact-law results.

## EVIDENCE
`naive_changed_target_law_count`: `162`; `naive_planned_law_count`: `162`; `quantization_defines_different_target`: `true`. The shown `naive_quantized_energy_mh` row has `exact_target_tv_from_full`: `0.008504837766257423` and `full_target_detailed_balance_error_max`: `0.0013283597975251617`. The corresponding `two_stage_delayed_acceptance` row has `exact_target_tv_from_full`: `0.0`, `first_moment_bias_max`: `0.0`, `second_moment_bias_max`: `0.0`, and `full_target_detailed_balance_error_max`: `3.469446951953614e-18`. The `full_precision_pair_swap_mh` comparator also has `exact_target_tv_from_full`: `0.0`. `verifier_is_oracle`: `false`; `useful_acceleration_claimed`: `false`.

## RECOMMENDATION
KEEP

## experiment_7189_v633_rust_slice_parity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Compiled Rust achieved transition, distribution-law, and bounded E2E parity with Python, but failed the measured 10× latency target.

## WHAT WOULD REFUTE IT
A replay mismatch, invalid cardinality, distribution error above its stated limit, failed E2E execution, or all measured Rust latency speedups reaching 10× would falsify part of the claim.

## WAS THAT CHECKED
Yes. Independent Python and Rust replay rows test transition parity; independently seeded distribution rows are compared with the exact finite-slice law; the serialized E2E has an explicit pass result; and Python is the serious performance comparator in every speedup condition. The performance test could fail and did: all displayed speedups are below 1×, much less than 10×.

## EVIDENCE
`"verifier_is_oracle": false`; `"compiled_rust_execution": true`; `"delta_energy_error": 4.440892098500626e-16`; `"cardinality_valid": true`; `"energy_mean_error": 0.1664472808659836`; `"energy_mean_error_limit": 0.25`; `"total_variation": 0.12040395733064468`; `"total_variation_limit": 0.15`; `"scenario": "SCENARIO-SAMPLER-7189-E2E"`; `"passed": true`; `"python_over_rust_latency_speedup": 0.6623730302831603`; `"python_over_rust_latency_speedup": 0.2433507139906358`; `"target": 10.0`; `"target_met": false`; `"nfr_01_10x_met": false`; `"performance_verdict_class": "null"`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7190_v633_board_placement_receipt.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or performance claim is made; the narrower receipt assertion would fail if an attached board lacked a continuity row or a placement row failed its declared compatibility checks.

## WAS THAT CHECKED
Yes, through `board_rows`, `placement_rows`, and `gate_check_summary`; readiness and physical topology were explicitly outside the claim.

## EVIDENCE
`new_board_performance_claimed` = `false`; `hardware_execution_claimed` = `false`; `claim_scope` = `compatibility_only`; `topology_fit` = `topology_unknown`; `board_placement_receipt_complete_score` = `1`; `One means complete visibility, not device readiness.`

## RECOMMENDATION
KEEP

## experiment_7191_v633_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V633 evidence matrix is complete, but exp7186 was blocked because `python/carnot/agentic/arc_eval_runner.py` was empty.

## WHAT WOULD REFUTE IT
Fewer than all 13 contracted tasks being represented, any task being silently excluded, or the required runner being nonempty and exp7186 nevertheless having qualifying execution evidence would falsify the claim.

## WAS THAT CHECKED
Yes. Matrix completeness was checked through contracted and represented task counts plus excluded task IDs; the blocking prerequisite was checked directly through the required source path’s byte count and failed gate.

## EVIDENCE
`"honest_verdict": "blocked: V633 evidence matrix is complete, but exp7186 could not run because python/carnot/agentic/arc_eval_runner.py has zero bytes"`; `"contracted_task_count": 13`; `"represented_task_count": 13`; `"excluded_task_ids": []`; `"all_contracted_tasks_preserved": true`; `"failed_check": "upstream_terminal_evidence"`; `"field": "REQUIRED_SOURCE_PATHS.python/carnot/agentic/arc_eval_runner.py"`; `"expected_value": "nonempty_file"`; `"observed_value": 0`; `"passed": false`; `"upstream": "exp7186-arc-withheld-transfer"`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP
