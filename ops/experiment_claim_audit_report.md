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
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6557_constraint_saturation_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports no experimental result or comparative claim to falsify.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate`, before an audit could test any claim.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6558_arc_live_redirect_ledger_reachability.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The live supervisor receipt path was reachable, seven firings were inspected, and the supported replay did not justify changing the curated arm order.

## WHAT WOULD REFUTE IT
An unreachable live entrypoint, missing outcome-bearing firing receipts, or a support-qualified arm ranking different from the current curated order would refute the claim.

## WAS THAT CHECKED
Yes—in `gate_check_summary`, `redirect_to_next_outcome_rows`, `curated_arm_support_rows`, and `selection_policy_disposition`. The support-qualified lower-priority arms recorded no helpful outcomes, while the sole helpful arm lacked the required three firings; the replay therefore retained the current order. The recorded prior missing-outcome failure also demonstrates that receipt validation could fail closed.

## EVIDENCE
`live_entrypoint_reachable`: `true`; `accepted_live_run_count`: `6`; `fired_total`: `7`; `helped_total`: `1`; `minimum_prospective_firings`: `3`; `drop_goal_bias`: `unsupported_fewer_than_three_firings`; `allow_reinduction`: `supported_no_help_lower_candidate`; `force_exploration_diversity`: `supported_no_help_lower_candidate`; `policy_changed`: `false`; `disposition`: `unchanged`; `reason`: `supported replay does not improve the current curated order`; `exp6524_status`: `blocked_missing_outcome_bearing_live_receipts`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_6559_gatemate_changed_state_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit found no qualifying post-Exp6525 physical-state receipt, therefore ran zero hardware commands and preserved Exp3866’s exclusion.

## WHAT WOULD REFUTE IT
A receipt candidate marked valid that was operator-authored, targeted GateMate plus DirtyJTAG, and dated after Exp6525 would refute the missing-receipt claim; any hardware action row or nonzero command count would refute the zero-command claim.

## WAS THAT CHECKED
Yes. The receipt audit evaluated 24 candidates, recomputed zero valid receipts and zero commands from the emitted rows, checked that the hardware-action list was empty, and separately verified Exp3866 remained excluded.

## EVIDENCE
`"receipt_candidate_count": 24`, `"valid_receipt_count": 0`, `"new_post_exp6525_physical_receipt_found": false`, `"failed_check": "operator_physical_state_receipt.newer_than_exp6525"`, `"hardware_action_rows": []`, `"hardware_command_count_recomputed": 0`, `"command_count_matches_rows": true`, `"preserved": true`, `"status": "blocked_missing_new_physical_receipt"`

## RECOMMENDATION
KEEP

## experiment_6560_v567_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V567 produced a null adoption result: production routing remains default-off, reversible memory and constraint saturation remain experiment-only, CSL and GateMate remain blocked, ARC selection remains unchanged, and V567 did not close the publication requirement.

## WHAT WOULD REFUTE IT
Any row showing a production default flip, adoption of either experimental mechanism, successful CSL or GateMate advancement, changed ARC selection, or closure of the independent-reproducer requirement by V567 would refute the claim.

## WAS THAT CHECKED
Yes. The claim-and-adoption matrix explicitly records each mechanism’s state; the independent recomputation rows test production behavior, CSL value and retention, and constraint saturation; the ARC, hardware, and publication dispositions separately record whether policy, hardware, or publication status advanced.

## EVIDENCE
`"state": "default-off"`; `"state": "experiment-only"`; `"state": "blocked"`; `"current_value_positive"` with `"observed_value": false`; `"retained_family_noninferior"` with `"observed_value": false`; `"policy_changed": false`; `"selection_policy_disposition": "unchanged"`; `"hardware_advanced": false`; `"hardware_command_count_recomputed": 0`; `"v567_integration_closes_independent_reproducer_requirement": false`; `"all_headlines_derived_from_rows": true`; `"verdict_class": "null"`.

## RECOMMENDATION
KEEP

## experiment_6561_v568_evidence_gate_contract.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6562_constraint_saturation_independent_audit_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The constraint-saturation evidence is disqualified because required provenance, replay, release/harm, or cost evidence is non-recomputable.

## WHAT WOULD REFUTE IT
All required audit gates passing, including recomputable live provenance, exact clause-and-joint replay, harm/release auditing, and charged-cost recomputation.

## WAS THAT CHECKED
Yes. The aggregate recomputation and gate summary explicitly evaluated those gates, and four failed.

## EVIDENCE
`"verdict_class": "disqualified"`; `"live_provenance_recomputable": false`; `"exact_clause_and_joint_replay_passed": false`; `"harm_and_release_audit_passed": false`; `"charged_cost_recomputed": false`; `"constraint_saturation_independent_audit_ready_score": 0.0`; `"constraint_saturation_policy_audited_score": 0.0`

## RECOMMENDATION
KEEP

## experiment_6563_production_safety_net_workload_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The disabled adapter preserved native behavior and all safety mechanisms passed, while enabling the adapter delivered no preregistered reduction in work or latency.

## WHAT WOULD REFUTE IT
Any disabled-versus-native mismatch, changed exact output, deleted candidate, unreachable fallback, failed restart or rollback, or an enabled-adapter checker-call or latency improvement meeting the preregistered threshold would falsify the claim.

## WAS THAT CHECKED
Yes. All 48 expected per-unit rows were present; native, disabled, and enabled conditions were compared; identity, output equality, candidate preservation, fallback, restart, and rollback were checked; and enabled work and latency were measured against native.

## EVIDENCE
`status`: `complete_production_safety_net_workload_canary_null`; `verdict_class`: `null`; `expected_per_unit_row_count`: `48`; `observed_per_unit_row_count`: `48`; `complete_rows`: `true`; `disabled_identity_exact`: `true`; `all_exact_outputs_equal`: `true`; `changed_output_count`: `0`; `all_candidates_preserved`: `true`; `fallback_passed`: `true`; `restart_passed`: `true`; `rollback_passed`: `true`; `native_checker_calls`: `14.0`; `enabled_checker_calls`: `14.0`; `enabled_checker_call_delta`: `0.0`; `enabled_wall_time_saved_s`: `-0.001244641`; `tail_latency_regression`: `true`; `measured_enabled_benefit`: `false`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6564_rust_pyo3_safety_net_nfr01.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The Rust/PyO3 implementation preserved exact Python behavior but failed the 10× NFR01 speedup threshold, while remaining within the frozen p99 latency bound.

## WHAT WOULD REFUTE IT
A parity mismatch, a fully charged median batched speedup of at least 10× versus Python scalar, or Rust batch p99 latency exceeding 0.05 seconds would refute one of the claim’s components.

## WAS THAT CHECKED
Yes. Per-unit parity rows compare Python scalar with PyO3 scalar and batch outputs; throughput rows include the serious Python-scalar baseline; and aggregate recomputation evaluates the measured speedup and p99 latency against frozen thresholds. The method was allowed to lose and did lose on throughput.

## EVIDENCE
`"verdict_class": "null"`; `"python_vs_pyo3_batch_bytes_equal": true`; `"all_exact_downstream_equal": true`; `"python_scalar_median_throughput_ops_s": 21532.190854`; `"rust_pyo3_batch_median_throughput_ops_s": 16450.883352`; `"steady_state_median_batched_speedup_vs_python_scalar": 0.764013447`; `"nfr01_threshold_speedup": 10.0`; `"rust_pyo3_batch_p99_latency_s": 6.312575e-05`; `"p99_latency_bound_s": 0.05`; `"nfr01_passed": false`; `"rust_pyo3_nfr01_ready_score": 0.0`

## RECOMMENDATION
KEEP
