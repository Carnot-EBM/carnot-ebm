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
Not applicable: this is a blocked-gate receipt and reports no source-span grounding result or comparative claim.

## WAS THAT CHECKED
No. The measurement never ran because its upstream canary gate failed.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_reason": "actual=0 == expected=1"`, `"failed_upstream": "exp7223-span-canary"`, `"failed_field": "span_canary_ready_score"`, `"failed_observed": 0`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7226_v636_belief_compiler.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The packed belief compiler matches the circular finite reference on the tested supported domain, without claiming learning efficacy or runtime acceleration.

## WHAT WOULD REFUTE IT
Any nonzero parity mismatch, failed parity or mutation check, stream-conformance error, or per-stream error would falsify the stated execution-parity claim.

## WAS THAT CHECKED
Yes. The parity rows checked prediction, energy, query, and delayed-replay behavior; mutation rows exercised five failure boundaries; stream conformance and 20 fresh-stream rows were also checked. The oracle was circular, but the claim explicitly limits itself to matching that reference and makes no independent-value claim.

## EVIDENCE
`"honest_verdict": "complete: packed belief compiler matches the circular finite reference; no learning or speed claim"`; `"verdict_class": "circular_positive"`; `"verifier_is_oracle": true`; `"mismatch_count": 0`; `"passed": true`; `"error": 0`; `"stream_conformance_errors": []`; `"learning_efficacy_claimed": false`; `"runtime_acceleration_claimed": false`; `"completed_streams": 20`; `"censored_streams": 0`

## RECOMMENDATION
KEEP

## experiment_7227_v636_belief_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Packed online memory did not pass every fixed prospective learning gate.

## WHAT WOULD REFUTE IT
All required learning gates passing, including recurrence-error increase at or below 0.02, would refute the claim.

## WAS THAT CHECKED
Yes. The efficacy gate directly tested recurrence retention alongside future-error and false-accept criteria across 20 independent stream seeds. The serious reference online version-space comparator was also present and tied the packed method.

## EVIDENCE
`honest_verdict`: `complete_null: packed online memory did not pass every fixed prospective learning gate`; `learning_value_passed`: `false`; `efficacy` `passed`: `false`; `recurrence_error_increase_lte_0_02`: `false`; `comparison_id`: `packed_online_memory_vs_reference_online_version_space`; its `false_accept_delta`, `future_error_delta`, and `recurrence_error_increase` estimates are all `0.0`; `independent_unit_count`: `20`; `verdict_class`: `null`.

## RECOMMENDATION
KEEP

## experiment_7228_v636_belief_cold_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7229_v636_rare_event_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The read-only audit classified every failed criterion, preserved Exp7216’s failed gate, and found the fixed rare-event accuracy target infeasible within the stated budget.

## WHAT WOULD REFUTE IT
Any failed criterion remaining unclassified, any alteration or reversal of the original failed gate, or a fixed probe whose required effective sample size and diagnostics could all be satisfied within the stated budget would refute the claim.

## WAS THAT CHECKED
Yes. Failure-classification accounting covers all 142 failed criteria; the audit compares the preserved gate with the upstream gate; and fixed-probe requirements are compared with the declared transition budget. All 240 planned observable rows were completed with none censored. No sampler rerun was needed for this read-only feasibility claim.

## EVIDENCE
`classified_failed_criteria`: `142`; `failed_criteria`: `142`; `unclassified_failed_criteria`: `0`; `primary_gate.passed`: `false`; `unchanged`: `true`; `feasible_at_budget`: `false`; `worst_probe_iid_effective_sample_lower_bound`: `169703912`; `max_pooled_transitions_per_graph_arm`: `40000000`; `planned_observable_rows`: `240`; `completed_observable_rows`: `240`; `censored_observable_rows`: `0`; `sampler_rerun_performed`: `false`; `verdict_class`: `null`

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
For the artifact’s factual receipt assertions, refutation would be an unauthenticated board disposition, a failed or hash-mismatched PolarFire dispatch, or a missing disposition for one of the three boards; there is no comparative performance or added-value claim to falsify.

## WAS THAT CHECKED
Yes. Per-board authentication and terminal-status fields are reported, PolarFire dispatch and hashes are checked, and GateMate’s unresolved physical-state condition is explicitly retained rather than counted as board success.

## EVIDENCE
`new_performance_claimed`: `false`; `hardware_performance_claimed`: `false`; `programmable_logic_sampling_claimed`: `false`; `board_continuity_complete_score`: `All three board dispositions, not all boards successful.`; `disposition`: `blocked_inherited_no_new_physical_state`; `dispatch_completed`: `true`; `input_hash_matches`: `true`; `output_hash_matches`: `true`

## RECOMMENDATION
KEEP

## experiment_7232_v636_capstone.json

**SKIPPED_ALREADY_FLAGGED**
