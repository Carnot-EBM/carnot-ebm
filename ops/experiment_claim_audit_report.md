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

## experiment_6926_span_first_relation_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or added-value claim to falsify. Treating fixture readiness as an operational assertion, a failed gate, semantic mismatch, solver disagreement, exposed held-out answer, or unexpected parser acceptance/rejection would refute readiness.

## WAS THAT CHECKED
Yes for operational fixture readiness, through per-row checks and `gate_check_summary`; no added-value comparison was attempted or claimed.

## EVIDENCE
`span_relation_fixture_ready_score` `1` `failed_checks` `[]` `solver_parity` `true` `verifier_is_oracle` `true` `inference_substrate` `deterministic_cpu_span_first_fixture_no_llm` `verdict_class` `circular_positive` `The closed class prevents an oracle result from claiming a moat.`

## RECOMMENDATION
KEEP

## experiment_6927_v607_literature_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or method-value claim is made; the artifact only records completion of a bounded literature review and one correction.

## WAS THAT CHECKED
No—not applicable. There is no method-versus-rival claim to falsify; the artifact checks procedural completion through the gate summary and terminal rows.

## EVIDENCE
`inference_substrate`: `bounded_primary_source_web_research_no_model_inference`; `invoked_models`: `[]`; `status`: `complete`; `expected`: `15 terminal source families and 10 terminal named candidates`; `observed`: `15 source families and 10 candidates`; `honest_verdict`: `complete_v607_literature_delta_with_one_verified_correction`

## RECOMMENDATION
KEEP

## experiment_6928_sota_runtime_receipt_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6941_v608_source_delta.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded source review found no material facts arising after the V608 planner marker.

## WHAT WOULD REFUTE IT
At least one qualifying primary or first-party fact absent from the planner marker, observed after `2026-09-03T16:23:00Z`, and changing a material local execution boundary.

## WAS THAT CHECKED
Yes. The artifact reports terminal outcomes across all 15 required source families and seven selected papers; qualifying discoveries would have appeared in `ledger_append_rows`.

## EVIDENCE
`honest_verdict`: `complete_null_v608_source_delta_no_post_marker_facts`; `verdict_class`: `null`; `ledger_append_rows`: `[]`; `expected`: `15 terminal source families and 7 terminal selected papers`; `observed`: `15 source families and 7 selected papers`; `all_gates_passed`: `true`; `status`: `complete`; `terminal_outcome`: `no_post_marker_update`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6942_v608_contract_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or scientific claim is made. Treating the operational blocked status separately, an all-pass gate summary, zero lint-command failures, and a readiness score of 1 would refute that status.

## WAS THAT CHECKED
Yes. The contract could have passed, but the gate summary and lint-command rows record actual failures. The oracle defines contract conformance only; no claim of verifier added value is made.

## EVIDENCE
`inference_substrate`: `deterministic_roadmap_contract_preflight_no_llm`; `honest_verdict`: `blocked_v608_contract_preflight`; `status`: `blocked`; `v608_execution_contract_ready_score`: `0`; `gate_check_summary`; `passed`: `false`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6943_verifier_density_prefix_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and reports no experimental or comparative result.

## WAS THAT CHECKED
No; the experiment stopped at the upstream pre-gate, so no method, oracle, rival, or scored rows were evaluated.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_observed`: `0`; `failed_expected`: `1`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6948_arc_branch_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact makes no substantive or comparative claim.

## WAS THAT CHECKED
No; execution stopped at the pre-gate, so no branching-corpus results were produced or evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6952_v608_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative method claim is made. The artifact’s status statement would be contradicted by admissible executed scientific rows with non-null positive scores and passing prerequisite gates.

## WAS THAT CHECKED
Yes for the status statement: gate replay, eligibility, availability, and authoritative-score rows were checked. No comparative method or serious baseline was tested because downstream science remained blocked or unavailable.

## EVIDENCE
`honest_verdict`: `partial_v608_capstone_contract_audited_science_incomplete`; `status`: `complete_partial`; `verdict_class`: `partial`; `v608_execution_contract_ready_score`; `observed`: `0`; `passed`: `false`; `eligible`: `false`; `value`: `null`; `prefix_energy_positive_score`: `null`; `causal_hidden_state_positive_score`: `null`; `branch_energy_positive_score`: `null`; `trace_state_positive_score`: `null`; `audited_trace_state_positive_score`: `null`; `inference_substrate`: `independent_artifact_replay_and_contract_reconciliation_no_llm`

## RECOMMENDATION
KEEP
