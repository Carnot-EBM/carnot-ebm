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
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6796_agent_model_dispatch_requalification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports a blocked precondition audit, not a positive or comparative claim about dispatch requalification.

## WAS THAT CHECKED
No comparative claim was checked; no routes were evaluated and no result rows were produced.

## EVIDENCE
`status` `blocked` `rows` `[]` `v593_pairs_audited` `0` `dispatch_contract_ready` `false` `verdict_class` `blocked` `honest_verdict` `complete_blocked_dispatch_requalification`

## RECOMMENDATION
KEEP

## experiment_6797_canonical_transaction_byte_replay.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6798_csl_causal_safety_byte_audit.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Byte replay verified causal route and utility witnesses with zero admitted or influential poison.

## WHAT WOULD REFUTE IT
A poison-attack phase causing harm despite zero poison admission—for example, capacity pressure evicting active state and changing hard-case outcomes from unharmed to harmed.

## WAS THAT CHECKED
Yes, under `hard_case_harm_after_phase` and `capacity_eviction_receipts`; the refutation occurred after capacity pressure and disappeared after rollback.

## EVIDENCE
`admitted_poison_count`: `0`; `hard_case_harm_after_phase`; `baseline`: `false`; `capacity_pressure`: `true`; `rollback`: `false`; `capacity_eviction_receipts`; `reason`: `capacity_pressure_fifo`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_6799_model_output_formal_constraint_probes.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
Frozen authentic outputs produced exact paired formal probes.

## WHAT WOULD REFUTE IT
A source-output hash mismatch or replacement, a failed fresh-process replay, a missing or invalid probe member, failure to distinguish the formal operations, or failure of the declared pairing tolerances would refute this execution-receipt claim.

## WAS THAT CHECKED
Yes. Source hashes and inference provenance, fresh-process exact replays, operation proofs, matching receipts, gate summaries, and adversarial mutation receipts provided opportunities for those failures. This remains a receipt claim, not a comparative claim of method value or generalization.

## EVIDENCE
`honest_verdict`: `complete: frozen authentic outputs produced exact paired formal probes`; `inference_substrate`: `deterministic_verifier -- CPU transform of frozen authentic mandated-GGUF outputs; no new LLM inference and no source-output replacement`; `live_llm_invoked`: `false`; `fresh_process`: `true`; `matches`: `true`; `operation_class_distinction_proved`: `true`; `all_tolerances_passed`: `true`; `all_passed`: `true`; `failed_checks`: `[]`; `model_output_constraint_probe_ready`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The frozen real-output comparison completed without the preregistered positive effect.

## WHAT WOULD REFUTE IT
A positive grouped-minus-flat restructuring effect whose clustered 95% confidence-interval lower bound exceeded zero, while all preregistered harm and validity gates passed.

## WAS THAT CHECKED
Yes. The artifact reports the paired clustered restructuring interval and the explicit preregistered decision gates; the positive outcome was possible but did not occur.

## EVIDENCE
`"honest_verdict": "complete: frozen real-output comparison finished without the preregistered positive effect"`; `"positive": false`; `"restructuring_exact_valid_lower_bound": -0.0037037037`; `"restructuring_exact_valid_lower_bound_above_zero": false`; `"lower": -0.0037037037`; `"upper": 0.0074074074`; `"no_refinement_harm": false`; `"no_support_harm": false`; `"candidate_budget_by_arm"`; `"flat_recurrent_control": 4365`; `"grouped_fixed_point": 4365`; `"parameter_count_per_arm": 91`; `"transfer_updates": 0`; `"verifier_is_oracle": "False states that exact checking never proposes or fits."`

## RECOMMENDATION
KEEP

## experiment_6801_real_output_fixed_point_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The fixed-point result is disqualified because authority or shortcut checks failed.

## WHAT WOULD REFUTE IT
A model-identity permutation that materially changed the measured effects, plus a robust grouped-fixed-point advantage over the flat recurrent control—especially confidence intervals excluding zero—would refute the disqualification.

## WAS THAT CHECKED
Yes. The model-ID permutation left the global effects unchanged, the identical-arm control tied exactly, the serious flat recurrent comparator matched or beat grouped fixed point on exact validity, and every reported transformation confidence interval includes zero.

## EVIDENCE
`global_effect_unchanged`: `true`; `proposal_input_hash_unchanged`: `true`; `paired_exact_valid_delta`: `0.0`; `flat_recurrent_control`; `grouped_fixed_point`; base `exact_valid_rate` values `0.0323024055` and `0.0302405498`; base interval `lower`: `-0.0074074074`, `upper`: `0.0037037037`; refinement interval `lower`: `-0.0037037037`, `upper`: `0.0058641975`; restructuring interval `lower`: `-0.0043209877`, `upper`: `0.0077160494`.

## RECOMMENDATION
KEEP

## experiment_6802_operational_obligation_automaton_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no comparative, performance, solve, or added-value claim; it records an operationally blocked run.

## WAS THAT CHECKED
No efficacy refutation was applicable or checked because no automaton was built and no rows were evaluated. The operational blocker itself was checked in `gate_check_summary`.

## EVIDENCE
`"solve_claim"`: `false`; `"verdict_class"`: `"blocked"`; `"rows"`: `[]`; `"attack_results"`: `[]`; `"operational_automaton_fixture_ready"`: `false`; `"state"`: `"not_built_because_precondition_failed"`; `"failed_check"`: `"required_agentic_verification_spec_exists"`

## RECOMMENDATION
KEEP

## experiment_6803_sota_operational_handoff_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive comparative or value claim exists to falsify; the artifact is only a blocked-gate receipt.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate` before producing or evaluating the proposed corpus.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"failed_field"`: `"operational_automaton_fixture_ready"`; `"failed_expected"`: `true`; `"failed_observed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP
