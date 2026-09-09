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
| NO_CLAIM | 5 |

## experiment_7143_flowbalance_memory_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact is a blocked-gate receipt and reports no audit outcome to falsify.

## WAS THAT CHECKED
No; the experiment stopped at the pre-gate before the cold-retention or negative-transfer audit ran.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"duration_s": 0.0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7144_v627_rebudgeted_arc_loo.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment is disqualified because the arms failed the common-arm-configuration gate.

## WHAT WOULD REFUTE IT
A passing common-arm-configuration check, with the observed value matching the expected value and all gates passing, would refute the disqualification.

## WAS THAT CHECKED
Yes, in `gate_check_summary`; the common-arm-configuration check failed, so the refuting observation did not occur.

## EVIDENCE
`honest_verdict` `disqualified_common_arm_configuration` `verdict_class` `disqualified` `all_passed` `false` `failed_check` `common_arm_configuration` `expected_value` `true` `observed_value` `false`

## RECOMMENDATION
KEEP

## experiment_7145_v627_rust_multiscale_sampler.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A completed run showing any Python–Rust mismatch in the preregistered parity cells would refute an exact-parity claim, but this artifact makes no such claim.

## WAS THAT CHECKED
No; execution never started, and every evidentiary row collection is empty.

## EVIDENCE
`"inference_substrate_class": "blocked_no_run"`; `"duration_s": 0.0`; `"rows": []`; `"exact_parity_score": 0.0`; `"rust_multiscale_parity_score": 0.0`; `"failed_check": "execution_not_started"`; `"verdict_class": "blocked"`; `"honest_verdict": "blocked_no_run_pending_preconditions"`

## RECOMMENDATION
KEEP

## experiment_7146_v627_gatemate_changed_state.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment was correctly blocked because no complete operator-authored physical-state receipt newer than Exp6559 was found.

## WHAT WOULD REFUTE IT
A valid, complete, operator-authored GateMate physical-state receipt dated after the Exp6559 cutoff of 20260823.

## WAS THAT CHECKED
Yes. The receipt audit records candidate rows with dates, authorship, validity, and rejection reasons, then reports the latest candidate date and the failed cutoff gate.

## EVIDENCE
`physical_state_receipt`; `exists`; `false`; `no complete operator-authored physical-state receipt newer than Exp6559`; `latest_candidate_date`; `20260811`; `receipt_cutoff_date`; `20260823`; `receipt_newer_than_exp6559`; `observed_value`; `0.0`; `passed`; `false`; `command_rows`; `[]`; `hardware_command_count`; `0`; `verdict_class`; `blocked`; `blocked_no_new_operator_physical_state_receipt_after_exp6559`

## RECOMMENDATION
KEEP

## experiment_7147_v627_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V627 evidence matrix is structurally complete, while upstream terminal availability blocks completion.

## WHAT WOULD REFUTE IT
A matrix-slot coverage below one, or an upstream-availability check showing no missing artifacts and no external-state blocks, would refute the headline.

## WAS THAT CHECKED
Yes. Structural coverage is reported in `rows`, and `gate_check_summary` compares expected and observed upstream availability and records the failed gate.

## EVIDENCE
`"matrix_slot_coverage_rate": 1.0`; `"failed_check": "upstream_terminal_availability"`; `"missing_artifacts": [7140, 7143]`; `"external_state_blocks": ["gatemate_operator_receipt"]`; `"passed": false`; `"status": "blocked"`; `"v627_capstone_complete_score": 1`; `"One means the evidence matrix is structurally complete, not scientifically promoted."`; `"scientific_branch_promotion_rate": 0.0`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7148_v628_contract_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify; the narrower operational finding would be refuted by the YAML prerequisite being readable and nonempty, allowing the 12 task contracts to be evaluated.

## WAS THAT CHECKED
Yes, for the prerequisite finding: `preconditions_checked` and `gate_check_summary` test YAML readability. No method result was tested because the run was blocked before any task rows were produced.

## EVIDENCE
`honest_verdict`: `blocked_v628_contract_preflight_prerequisite_missing`; `inference_substrate_class`: `blocked_no_run`; `check`: `v628_yaml_readable`; `available`: `false`; `passed`: `false`; `observed_task_count`: `0`; `rows`: `[]`; `verdict_class`: `blocked`

## RECOMMENDATION
KEEP

## experiment_7149_v628_source_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked, zero-duration no-run and asserts no comparative or value claim.

## WAS THAT CHECKED
No; there are no execution rows or completed source/cache checks against which a substantive claim could fail.

## EVIDENCE
`inference_substrate_class`, `blocked_no_run`, `duration_s`, `0.0`, `rows`, `[]`, `v628_source_delta_complete_score`, `0`, `failed_check`, `preconditions_not_checked`, `passed`, `false`, `verdict_class`, `blocked`, `honest_verdict`, `blocked_v628_source_delta_precondition`

## RECOMMENDATION
KEEP

## experiment_7150_v628_grounding_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports preflight readiness and a blocked gate, not comparative method value or detection performance.

## WAS THAT CHECKED
No; no game outcomes or comparative scores were produced.

## EVIDENCE
`honest_verdict`: `blocked_real_qwen_canary`; `verdict_class`: `blocked`; `grounding_preflight_ready_score`: `0`; `inference_substrate_class`: `blocked_no_run`; `per_game_results`: `[]`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
