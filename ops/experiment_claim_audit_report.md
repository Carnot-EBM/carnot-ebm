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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6796_agent_model_dispatch_requalification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked precondition audit and makes no comparative or value claim about dispatch requalification.

## WAS THAT CHECKED
No substantive method claim was tested; no pairs were audited and no result rows were produced.

## EVIDENCE
`"status": "blocked"`, `"rows": []`, `"v593_pairs_audited": 0`, `"dispatch_contract_ready": false`, `"verdict_class": "blocked"`, `"honest_verdict": "complete_blocked_dispatch_requalification"`

## RECOMMENDATION
KEEP

## experiment_6797_canonical_transaction_byte_replay.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6798_csl_causal_safety_byte_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent byte replay verified causal route and utility witnesses, with zero admitted or influential poison.

## WHAT WOULD REFUTE IT
Any replay mismatch, a credited same-parent counterfactual showing no action or utility effect, any admitted or influential poison, or the retrieval-disabled control tying or beating the method on utility.

## WAS THAT CHECKED
Yes. Action, utility, receipt, counterfactual-witness, and poison checks could fail; the retrieval-disabled control was also evaluated across five orders and lost on held-future utility in every order. The capacity-pressure harm result would refute a broader “no harm under pressure” claim, but the headline does not make that claim.

## EVIDENCE
`all_actions_replayed`: `true`; `all_utilities_replayed`: `true`; `all_receipt_hashes_matched`: `true`; `all_passed`: `true`; `failed_checks`: `[]`; `action_errors`: `[]`; `utility_errors`: `[]`; `receipt_errors`: `[]`; `admitted_poison_count`: `0`; `credited_factor_count`: `116`; `same_parent_bytes`: `true`; `action_changed`: `true`; `utility_difference`: `1.25`. Under `held_future_utility_by_arm_order`, `compositional_online` has `mean_utility` values `0.578125`, `0.578125`, `0.5625`, `0.5625`, and `0.5625`, versus `retrieval_disabled_online` values `0.21875`, `0.234375`, `0.28125`, `0.203125`, and `0.20625`. `hard_case_harm_after_phase` records `capacity_pressure`: `true`.

## RECOMMENDATION
KEEP

## experiment_6799_model_output_formal_constraint_probes.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s limited receipt assertion, a replay mismatch, unauthenticated or replaced source output, failed pairing gate, or invalid formal-operation proof would refute successful construction; none would test comparative method value because no such value claim is made.

## WAS THAT CHECKED
Yes. Source hashes, fresh-process replay matches, matching tolerances, operation proofs, and adversarial mutation receipts provide failure paths for the receipt assertion. No comparative baseline or added-value test was called for by the headline.

## EVIDENCE
`honest_verdict`: `complete: frozen authentic outputs produced exact paired formal probes`; `inference_substrate`: `deterministic_verifier -- CPU transform of frozen authentic mandated-GGUF outputs; no new LLM inference and no source-output replacement`; `live_llm_invoked`: `false`; `fresh_process`: `true`; `matches`: `true`; `all_tolerances_passed`: `true`; `operation_class_distinction_proved`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The frozen real-output comparison completed without the preregistered positive effect.

## WHAT WOULD REFUTE IT
A completed matched comparison showing the grouped arm’s preregistered exact-valid lower bound above zero, with required harm gates passing and the terminal positive gate true, would refute the null headline; an incomplete grid would refute its completion claim.

## WAS THAT CHECKED
Yes. The decision gates report the effect bounds and terminal positive decision, while the checkpoint receipt compares completed and planned rows and lists pending rows. The matched flat recurrent arm gave the grouped method a real opportunity to win.

## EVIDENCE
The artifact reports `positive` as `false`. The `restructuring_exact_valid_lower_bound` is `-0.0037037037`, and `restructuring_exact_valid_lower_bound_above_zero` is `false`. The refinement confidence interval is `lower` `-0.0037037037`, `point` `0.000617284`, and `upper` `0.0055555556`; restructuring is `lower` `-0.0037037037`, `point` `0.0015432099`, and `upper` `0.0074074074`. It also reports `no_refinement_harm` as `false` and `no_support_harm` as `false`. Both `candidate_budget_by_arm` values are `4365`, and the arms share `parameter_count_per_arm` `91`. Completion is supported by `complete` `true`, `completed_row_count` `2910`, `planned_row_count` `2910`, and `pending_row_ids` `[]`. The terminal value is `complete: frozen real-output comparison finished without the preregistered positive effect`.

## RECOMMENDATION
KEEP

## experiment_6801_real_output_fixed_point_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation can be specified because the supplied artifact states no headline proposition about the comparative value of `grouped_fixed_point`.

## WAS THAT CHECKED
No; the artifact reports comparative measurements and controls but contains no explicit headline claim or success criterion to test.

## EVIDENCE
`grouped_fixed_point` `flat_recurrent_control` `complete_disqualified: authority or shortcut checks failed`

## RECOMMENDATION
KEEP

## experiment_6802_operational_obligation_automaton_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify; the recorded blocker would be contradicted if the required specification existed with the required requirement anchors and the source gate passed.

## WAS THAT CHECKED
Yes, in `gate_check_summary`; the required specification precondition was checked and failed before compilation or row evaluation.

## EVIDENCE
`"solve_claim": false`, `"verdict_class": "blocked"`, `"status": "complete_blocked_operational_obligation_automaton"`, `"operational_automaton_fixture_ready": false`

## RECOMMENDATION
KEEP

## experiment_6803_sota_operational_handoff_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing: this is a blocked-gate receipt, not a completed experiment asserting a comparative result.

## WAS THAT CHECKED
No; execution stopped at the pre-gate, so no method, oracle, rival, or scored rows were evaluated.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP
