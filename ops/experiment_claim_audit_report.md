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
| NO_CLAIM | 7 |

## experiment_7141_v627_csl_event_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive comparative or learning claim is made; the narrower readiness receipt would fail if a gate failed, replay hashes mismatched, or a required corruption remained undetected without reducing readiness.

## WAS THAT CHECKED
Yes. Gate checks, independent replay checks, and four mutation tests could fail; every mutation shown was detected and reduced readiness to zero.

## EVIDENCE
`"learning_claim_made": false`; `"positive_csl_event_stream_ready_no_learning_claim"`; `"csl_event_stream_ready_score": 1`; `"failed_check": null`; `"observed_value": "all checks pass"`; `"passed": true`; `"detected": true`; `"readiness_after_mutation": 0`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7139_v627_symbolic_grounding_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no comparative performance or value claim.

## WAS THAT CHECKED
No. The experiment was blocked before inference, leaving all outcome and comparison rows empty.

## EVIDENCE
`honest_verdict`: `blocked_native_llama_server`; `inference_substrate_class`: `blocked_no_run`; `verdict_class`: `blocked`; `symbolic_grounding_complete_score`: `0`; `arm_rows`: `[]`; `metric_rows`: `[]`; `rows`: `[]`; `model_load_receipts`: `[]`.

## RECOMMENDATION
KEEP

## experiment_7142_v627_flowbalance_memory_csl.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No claim is made to falsify; any future value claim would be refuted by a baseline tie or win, non-positive paired uplift, or negative transfer.

## WAS THAT CHECKED
No; checks had not started and all result rows are empty.

## EVIDENCE
`"honest_verdict": "blocked_initial_schema_written_before_checks"`, `"inference_substrate_class": "blocked_no_run"`, `"observed_value": "checks_not_started"`, `"flowbalance_memory_csl_complete_score": 0`, `"future_uplift_supported_score": 0`, `"rows": []`, `"future_success_rows": []`, `"advantage_rows": []`, `"arm_rows": []`, `"paired_interval_rows": []`

## RECOMMENDATION
KEEP

## experiment_7143_flowbalance_memory_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and makes no substantive claim about cold retention or negative transfer.

## WAS THAT CHECKED
No; the audit was blocked before execution at `conductor_pre_gate`.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `failed_observed` `0` `failed_expected` `1` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7144_v627_rebudgeted_arc_loo.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact explicitly declines both a solve claim and a valid comparative conclusion.

## WAS THAT CHECKED
No falsification check is applicable. The artifact instead reports that the comparison was disqualified because the common-arm configuration check failed.

## EVIDENCE
`honest_verdict` `disqualified_common_arm_configuration` `verdict_class` `disqualified` `failed_check` `common_arm_configuration` `all_passed` `false` `solve_claim_made` `false` `game_level_solve_claimed` `false`

## RECOMMENDATION
KEEP

## experiment_7145_v627_rust_multiscale_sampler.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A completed run containing any Python–Rust mismatch in the preregistered parity cells would refute an exact-parity claim, but this blocked initialization artifact makes no such claim.

## WAS THAT CHECKED
No; execution never started, and all comparison row collections are empty.

## EVIDENCE
`"inference_substrate_class": "blocked_no_run"`, `"duration_s": 0.0`, `"rows": []`, `"exact_parity_score": 0.0`, `"rust_multiscale_parity_score": 0.0`, `"failed_check": "execution_not_started"`, `"verdict_class": "blocked"`, `"honest_verdict": "blocked_no_run_pending_preconditions"`

## RECOMMENDATION
KEEP

## experiment_7146_v627_gatemate_changed_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made. The operational blocking conclusion would be refuted by a valid operator-authored physical-state receipt newer than `Exp6559`.

## WAS THAT CHECKED
Yes, for the operational conclusion: `receipt_rows`, `physical_state_receipt`, and `gate_check_summary` evaluate the receipt cutoff. No hardware-performance claim was tested.

## EVIDENCE
`"inference_substrate_class": "blocked_no_run"`, `"receipt_newer_than_exp6559_score": 0.0`, `"command_rows": []`, `"hardware_command_count": 0`, `"verdict_class": "blocked"`, `"honest_verdict": "blocked_no_new_operator_physical_state_receipt_after_exp6559"`

## RECOMMENDATION
KEEP

## experiment_7147_v627_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V627 evidence matrix is structurally complete, but upstream terminal availability is blocked.

## WHAT WOULD REFUTE IT
A complete matrix showing no missing upstream artifacts, no external-state blocks, and a passing upstream-terminal-availability gate would refute the blocked headline.

## WAS THAT CHECKED
Yes. The artifact explicitly recomputed the upstream availability gate and matrix coverage; the gate failed because experiments 7140 and 7143 were missing and the GateMate operator receipt remained externally blocked.

## EVIDENCE
`honest_verdict` `blocked_upstream_terminal_availability_v627_matrix_complete` `matrix_slot_coverage_rate` `1.0` `failed_check` `upstream_terminal_availability` `passed` `false` `missing_artifacts` `7140` `7143` `external_state_blocks` `gatemate_operator_receipt` `scientific_branch_promotion_rate` `0.0`

## RECOMMENDATION
KEEP
