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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |

## experiment_7825_v680_training_runtime.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The artifact records a positive training and online runtime result for six public fixture families.

## WHAT WOULD REFUTE IT
A simple verifier-free rule applied to the same cached candidates tying the verifier ensemble on decisions and independently judged correctness would refute any claim of added value.

## WAS THAT CHECKED
No. The artifact reports fixture readiness, but the verifier defines correctness and no independent correctness check or serious comparator is shown. The runtime measurements can stand; the positive verdict cannot establish verifier value.

## EVIDENCE
`honest_verdict` = `complete_circular_positive_training_and_online_runtime`; `verdict_class` = `circular_positive`; `verifier_is_oracle` = `true`; `inference_substrate` = `verifier_ensemble_against_cached_candidates`; `decision_benefit` = `null`; `claim_scope` = `six_public_fixture_families_only; no natural fitting or held-out benefit`; `Fixture success is circular evidence.`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_7826_view_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no fit or policy result to refute. A completed fit reported in this artifact would contradict its blocked status.

## WAS THAT CHECKED
No fit was evaluated. The artifact checked six prerequisite gates and reports two failures.

## EVIDENCE
`"status": "blocked"`; `"gate_check_summary": "gate-unsat(final): 2 of 6 gate(s) failed; first failure: exp7824-source-feature-isolation.source_isolation_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7828_v680_counter_evidence_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A concrete refutation cannot be formulated because the artifact asserts no comparative, capability, or generalization claims; it functions as a protocol receipt recording execution disqualification prior to model evaluation.

## WAS THAT CHECKED
No. Execution was blocked and marked disqualified during validation receipts before candidate evaluation commenced, resulting in zero model invocations and zero completed rows.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_required_validation`
- `verdict_class`: `disqualified`
- `claim_scope`: `Exposed development fixture only; no hidden generalization or oracle-distinct benefit.`
- `disposition`: `unstarted_no_model_invocation`
- `calls`: `0`
- `completed`: `0`
- `started`: `0`

## RECOMMENDATION
KEEP

## experiment_7829_qwen_counter_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Empirical observations or comparative sensitivity measurements resulting from execution. Because execution was blocked before running, no empirical hypothesis was evaluated and no comparative claim was made.

## WAS THAT CHECKED
No; the experiment was blocked at the pre-gate stage before any experimental run occurred.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`duration_s`
`0.0`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_at_layer`
`conductor_pre_gate`
`gate_check_summary`
`gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7828-counter-evidence-protocol.counter_evidence_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7831_v680_arc_supervisor_refinement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The inspected sources yielded no eligible new redirect receipts, and the refinement run was disqualified; the artifact makes no solve or causal benefit claim.

## WHAT WOULD REFUTE IT
An eligible new redirect receipt in an inspected source would contradict the inventory finding. A passing result for the failed required validation would contradict the stated disqualification.

## WAS THAT CHECKED
Yes. The artifact records an eligible receipt count for each of the three inspected sources and reports the required validation failure.

## EVIDENCE
`eligible_receipt_count`: `0` in each of the three `rows`; `eligible`: `0`; `excluded`: `3`; `validation_errors`: `failed_required:all_python_tests`; `verdict_class`: `disqualified`; `claim_scope`: `observational_redirect_outcome_ledger_no_solve_or_causal_claim`.

## RECOMMENDATION
KEEP

## experiment_7834_v680_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A qualified board run showing a whole-service advantage would overturn the stated reason to defer acquisition. The artifact makes no comparative performance claim to refute.

## WAS THAT CHECKED
No. It reports historical board receipts and no new board operation or service measurement.

## EVIDENCE
`terminal_scope`: `historical_board_custody_only`; `hardware_operations_issued`: `[]`; `service_measured`: `false`; `hardware_advantage_claimed`: `false`; `honest_verdict`: `complete_disqualified_required_checks`.

## RECOMMENDATION
KEEP

## experiment_7835_v680_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit found no eligible independent evidence and disqualified readiness after required validation failed.

## WHAT WOULD REFUTE IT
An eligible independent evidence row, a qualifying upstream producer, or successful completion of the required validation would contradict the stated grounds for disqualification.

## WAS THAT CHECKED
Yes. The artifact reports upstream dispositions, eligibility counts, discrepancy rows, and required validation results.

## EVIDENCE
`independent_evidence_ready_score`: `0`; `eligible`: `0`; `independent_n`: `0`; `excluded`: `640`; `state`: `disqualified`; `state`: `missing`; `name`: `full_python_pytest`; `timed_out`: `true`; `honest_verdict`: `complete_disqualified_required_validation`.

## RECOMMENDATION
KEEP

## experiment_7836_v680_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The v680 capstone did not qualify a new result because required validation and independent evidence were insufficient.

## WHAT WOULD REFUTE IT
A qualifying producer with independent evidence of benefit, together with passing required validation, would refute the disqualification.

## WAS THAT CHECKED
Yes. The artifact records producer eligibility and benefit across the task rows, and records the required validation outcomes. The qualifying evidence and passing validation did not appear.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_validation`; `capstone_complete_score`: `0`; `independent_n`: `0`; `required_checks_passed`: `false`; `fresh_generalization_eligible`: `false`; `benefit`: `null`; `state`: `blocked`.

## RECOMMENDATION
KEEP
