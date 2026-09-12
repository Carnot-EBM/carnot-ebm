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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_7224_span_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked gate check, not a measurement result or comparative claim.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate` because an upstream gate failed.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "span_canary_ready_score"`, `"failed_observed": 0`, `"failed_expected": 1`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7226_v636_belief_compiler.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The packed belief compiler matches the circular finite reference within the tested finite domain, without claiming learning benefit or runtime acceleration.

## WHAT WOULD REFUTE IT
Any nonzero prediction, query, energy, replay, or fresh-stream error; a failed parity or mutation check; or a state that does not match after reload would falsify the scoped equivalence claim.

## WAS THAT CHECKED
Yes. The parity rows exercised prediction, query, energy, and delayed replay behavior; the stream rows checked fresh-stream readiness; mutation rows tested boundary failures; and state reload was checked. These checks could have produced mismatches or failures, although they do not establish value beyond the oracle.

## EVIDENCE
`"honest_verdict": "complete: packed belief compiler matches the circular finite reference; no learning or speed claim"`; `"verdict_class": "circular_positive"`; `"verifier_is_oracle": true`; `"mismatch_count": 0`; `"passed": true`; `"error": 0`; `"abstention": 0`; `"reload_match": true`; `"stream_conformance_errors": []`; `"learning_efficacy_claimed": false`; `"runtime_acceleration_claimed": false`

## RECOMMENDATION
KEEP

## experiment_7227_v636_belief_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Packed online memory did not pass every fixed prospective learning gate.

## WHAT WOULD REFUTE IT
All fixed prospective learning gates passing, including the recurrence-error requirement, would refute the claim.

## WAS THAT CHECKED
Yes. The acceptance-gate results show that efficacy failed specifically because the recurrence-error requirement failed, while the other efficacy checks passed. The serious reference-online comparator also tied the packed method exactly, leaving no evidence of added value. Oracle reuse would invalidate a positive value claim, but it does not invalidate this reported null.

## EVIDENCE
`honest_verdict`: `complete_null: packed online memory did not pass every fixed prospective learning gate`; `learning_value_passed`: `false`; `efficacy`: `passed`: `false`; `recurrence_error_increase_lte_0_02`: `false`; `future_error_upper_ci95_lt_zero`: `true`; `false_accept_upper_ci95_lte_zero`: `true`; `comparison_id`: `packed_online_memory_vs_reference_online_version_space`; `future_error_delta`: `estimate`: `0.0`; `false_accept_delta`: `estimate`: `0.0`; `recurrence_error_increase`: `estimate`: `0.0`; `verdict_class`: `null`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7228_v636_belief_cold_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7229_v636_rare_event_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The read-only audit completely classified the upstream failures, preserved the failed Exp7216 gate, and determined that the fixed rare-event accuracy target exceeds the stated budget.

## WHAT WOULD REFUTE IT
Any original failed criterion remaining unclassified, any changed or passing Exp7216 gate value, or every fixed probe satisfying the accuracy requirements within the 40,000,000-transition-per-graph-arm budget would refute the claim.

## WAS THAT CHECKED
Yes. The artifact checks classification completeness in `failure_classification`, compares expected and observed upstream gate fields in `gate_check_summary`, and evaluates the fixed observables against the declared accuracy target and budget in `next_measurement_envelope`. The oracle reuse does not establish independent verifier value, but the headline makes no such value claim.

## EVIDENCE
`"classified_failed_criteria": 142`, `"failed_criteria": 142`, `"unclassified_failed_criteria": 0`, `"down_up_value_score": 0`, `"primary_gate.passed": false`, `"unchanged": true`, `"observables_fixed_before_new_chain": true`, `"max_pooled_transitions_per_graph_arm": 40000000`, `"worst_probe_iid_effective_sample_lower_bound": 169703912`, `"feasible_at_budget": false`, `"verdict_class": "null"`, `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7230_v636_native_belief.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7231_v636_board_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or method-value claim is made; contradictory authentication, dispatch, or board-state rows would invalidate the continuity receipt, but would not refute an absent performance claim.

## WAS THAT CHECKED
Yes. Per-board rows report authentication, terminal status, abstention, and dispatch outcomes, including the blocked GateMate disposition.

## EVIDENCE
`aggregation_from_upstream_artifacts`; `new_performance_claimed`: `false`; `hardware_performance_claimed`: `false`; `programmable_logic_sampling_claimed`: `false`; `graduated_preserved`; `blocked_inherited_no_new_physical_state`; `terminal_cpu_dispatch_raw_transcript_retained`

## RECOMMENDATION
KEEP

## experiment_7232_v636_capstone.json

**SKIPPED_ALREADY_FLAGGED**
