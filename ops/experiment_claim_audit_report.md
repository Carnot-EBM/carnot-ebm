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

## experiment_7705_v671_constraint_bank_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A claim that the constraint bank adds value would be refuted if a matched control tied or beat it on untouched empirical outcomes.

## WAS THAT CHECKED
No. The artifact reports fixture measurements and leaves the benefit gates unresolved.

## EVIDENCE
`fixture_oracle`: `true`; `qualified_empirical_acquisition`: `false`; `empirical_effect`: `null`; `constraint_bank_ready_score`: `0`; `verdict_class`: `circular_positive`; `honest_verdict`: `complete_circular_positive_fixture_mechanics_no_acquisition`.

## RECOMMENDATION
KEEP

## experiment_7706_continuous_acquisition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A completed measurement contradicting a reported result would refute a substantive claim, but this artifact reports no measurement result.

## WAS THAT CHECKED
No. The experiment stopped at the upstream gate check.

## EVIDENCE
`"status": "blocked"`; `"blocked_at_layer": "conductor_pre_gate"`; `"constraint_bank_ready_score"`; `"actual": 0`; `"expected": 1`; `"passed": false`

## RECOMMENDATION
KEEP

## experiment_7707_v671_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent evidence audit is blocked because required evidence is missing or ineligible.

## WHAT WOULD REFUTE IT
Required producers being present and eligible, with online evidence measured and the readiness and validity gates passing, would refute the blocked verdict.

## WAS THAT CHECKED
Yes. The producer checks and acceptance gates could have passed; they instead record a missing producer and failed readiness and validity gates.

## EVIDENCE
`"honest_verdict": "complete_blocked_required_evidence"`; `"check": "producer_eligible"`, `"observed": 0`, `"expected": 1`; `"check": "producer_exists"`, `"observed": false`, `"expected": true`; `"eligible_producers": 5`, `"required_producers": 7`, `"passed": false`; `"failed_checks": 2`; `"online": null`; `"activation": false`.

## RECOMMENDATION
KEEP

## experiment_7708_v671_arc_generalization_runner.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The ARC generalization runner is ready for evaluation (`arc_runner_ready_score`: 1) based on successful execution of scripted fixtures.

## WHAT WOULD REFUTE IT
Runtime execution failures or state discrepancies when evaluated against an independent external oracle, errors during live model inference, or failure to generalize to unseen, hidden games.

## WAS THAT CHECKED
No. Refutation was not given a real chance to occur. The test executed deterministic scripted CPU fixtures without loading or invoking an LLM (`model_invoked` is `false`, `inference_substrate` is `host_scripted_sdk_cpu_fixture_no_model_call`), against historically public games where the verifier serves as its own oracle (`verifier_is_oracle` is `true`). Live arms and hidden games were explicitly not evaluated (`paired_live_arms` is `0`, `hidden_games` is `0`), rendering the positive readiness score true by construction.

## EVIDENCE
`honest_verdict`
`complete_circular_positive_fixture_runner_ready`
`verdict_class`
`circular_positive`
`verifier_is_oracle`
`true`
`arc_runner_ready_score`
`1`
`model_invoked`
`false`
`inference_substrate`
`host_scripted_sdk_cpu_fixture_no_model_call`
`solve_provenance`
`development_proxy`
`new_solve_credit`
`false`
`paired_live_arms`
`0`
`hidden_games`
`0`
`retained_live_games`
`0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7709_v671_arc_first_contact.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The two live first-contact episodes earned no new solve credit, and the run was disqualified for incomplete required validation.

## WHAT WOULD REFUTE IT
A live episode with a confirmed solve earning new solve credit would refute the solve result; a twelfth passing required receipt would refute the stated validation basis.

## WAS THAT CHECKED
Yes. Both episode rows report the solve outcome and goal support, while the coverage gate counts passed against required receipts.

## EVIDENCE
`"new_solve_credit": false` appears on both episode rows. Both report `"sdk_peak_level": 0`, `"engine_accepted": false`, and `"censoring": "censored_action_limit"`. The coverage gate reports `"passed_receipts": 11`, `"required_receipts": 12`, and `"passed": false`.

## RECOMMENDATION
KEEP

## experiment_7710_v671_native_record_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A Rust/Python action mismatch, a probability difference above the stated threshold, or failed replay would refute the narrow conformance result. The artifact makes no comparative value claim.

## WAS THAT CHECKED
Yes for conformance: the parity rows compare both implementations, and the replay check reports its outcome. Independent quality and utility were not established, and the record disqualifies itself.

## EVIDENCE
`honest_verdict`: `complete_disqualified_native_record_checks`; `verdict_class`: `disqualified`; `native_record_ready_score`: `0`; `fresh_evaluator_labels`: `0`; `held_out_labels`: `0`; `retention_groups`: `0`; `No scientific effect or trained-head claim.`

## RECOMMENDATION
KEEP

## experiment_7711_whole_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation of the experiment executing and asserting comparative performance or cost findings despite failing its upstream prerequisite gates, or the gate check reporting a pass despite unmet conditions. Because this is a pre-execution gate failure receipt, no comparative or empirical hypothesis is claimed.

## WAS THAT CHECKED
No; the run was blocked at `conductor_pre_gate` with duration `0.0` seconds before execution could begin, so no comparative evaluation was performed.

## EVIDENCE
`schema`
`"blocked_gate_check_v1"`
`status`
`"blocked"`
`honest_verdict`
`"blocked_gate_check_failed"`
`duration_s`
`0.0`
`blocked_at_layer`
`"conductor_pre_gate"`
`gate_check_summary`
`"gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7710-native-record-contract.native_record_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7712_v671_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone’s accounting is complete, but the required evidence for a new scientific benefit is blocked.

## WHAT WOULD REFUTE IT
Fresh, eligible results that pass the scientific gates—including a measured benefit against the matched controls, retention, and full-service cost—would refute the blocked verdict.

## WAS THAT CHECKED
Yes, for the verdict: the artifact checks the gates and reports failures. The static comparisons show no registered decision benefit; the prospective online, retention, and full-service measurements needed to overturn the block are absent.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v671_scientific_evidence`; `required_science_eligible`: `false`; `registered_decision_benefit_score`: `0`; `online_brier`: `null`; `online_retention_blocks`: `null`; `whole_service_paired_blocks`: `null`; `verdict_class`: `blocked`.

## RECOMMENDATION
KEEP
