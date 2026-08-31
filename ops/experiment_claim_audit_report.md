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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6801_real_output_fixed_point_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The grouped fixed-point method is not qualified as a positive improvement over the flat recurrent control.

## WHAT WOULD REFUTE IT
A positive grouped-minus-control exact-valid effect whose 95% source-case-clustered confidence interval remained above zero, accompanied by a genuine validity or convergence advantage over the flat recurrent control and passing authority/shortcut checks.

## WAS THAT CHECKED
Yes. The serious flat recurrent comparator was evaluated on paired cases, with duplicate removal, source-case-clustered confidence intervals, and transformation-specific effects. Every reported interval includes zero; the base aggregate favors the control, and neither arm converged.

## EVIDENCE
`resampling_unit`: `source_case`; `case_count`: `36`; base `point`: `-0.0018518519`, `lower`: `-0.0074074074`, `upper`: `0.0037037037`; refinement `point`: `0.000617284`, `lower`: `-0.0037037037`, `upper`: `0.0058641975`; restructuring `point`: `0.0015432099`, `lower`: `-0.0043209877`, `upper`: `0.0077160494`. Under `overall`, `flat_recurrent_control` has `exact_valid_rate`: `0.0323024055`, while `grouped_fixed_point` has `exact_valid_rate`: `0.0302405498`; both have `convergence_rate`: `0.0`.

## RECOMMENDATION
KEEP

## experiment_6802_operational_obligation_automaton_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact explicitly declines solve, readiness, and comparative-value claims.

## WAS THAT CHECKED
No; execution stopped at a failed source precondition, so no automaton, adversarial results, canonical-byte receipts, or evaluated rows were produced.

## EVIDENCE
`solve_claim` `false` `verdict_class` `blocked` `operational_automaton_fixture_ready` `false` `rows` `[]` `attack_results` `[]` `not_built_because_precondition_failed` `required_agentic_verification_spec_exists`

## RECOMMENDATION
KEEP

## experiment_6803_sota_operational_handoff_corpus.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact only records a blocked prerequisite gate and makes no comparative or value claim.

## WAS THAT CHECKED
No. The proposed corpus experiment did not run; it stopped at the prerequisite gate.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_at_layer": "conductor_pre_gate"`; `"actual": false`; `"expected": true`; `"passed": false`

## RECOMMENDATION
KEEP

## experiment_6810_v595_contract_manifest_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6811_operational_obligation_automaton_v3.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s limited readiness statement, any failed readiness component, failed gate, non-identical replay receipt, compatibility failure, hard violation, or mutation accepted instead of rejected would refute readiness; none would establish or refute live benefit because no such claim is made.

## WAS THAT CHECKED
Yes. Separate readiness components, gate checks, canonical-byte replay receipts, compatibility receipts, hard-violation counts, and adversarial mutation outcomes could record failures. No comparator is required because the artifact expressly makes no comparative-value or solve claim.

## EVIDENCE
`honest_verdict`: `complete: deterministic source-free operational-obligation fixture ready; no live benefit or level solve claimed`; `verdict_class`: `null`; `solve_claim`: `false`; `operational_automaton_fixture_ready`: `true`; `failed_checks`: `[]`; `hard_violation_count`: `0`; `readiness_components`; `attack_coverage`: `true`; `backward_compatibility`: `true`; `compiler`: `true`; `replay`: `true`; `schema`: `true`; `outcome`: `rejected`; `failed_closed`: `true`; `fresh_process`: `true`.

## RECOMMENDATION
KEEP

## experiment_6812_sota_operational_handoff_corpus_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All 576 authentic rows are ready, and direct-typed inputs achieve higher operational preservation than compressed prose.

## WHAT WOULD REFUTE IT
Direct-typed operational preservation tying or falling below compressed prose, or fewer than 576 complete authentic rows.

## WAS THAT CHECKED
Yes. The artifact compares both arms over 288 rows each, reports their operational-preservation counts and rates, and separately checks row completeness and corpus readiness. The comparator could have tied or beaten direct typed but did not.

## EVIDENCE
`"planned_row_count": 576`; `"rows_complete": true`; `"operational_handoff_corpus_ready": true`; `"direct_typed"`: `"numerator": 23`, `"denominator": 288`, `"rate": 0.0798611111111111`; `"compressed_prose"`: `"numerator": 8`, `"denominator": 288`, `"rate": 0.027777777777777776`; `"verifier_is_oracle": false`; `"effect_sign_controls_readiness": false`

## RECOMMENDATION
KEEP

## experiment_6813_selective_priority_arbiter_ab.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
On held exact replay, selective priority provides a positive effect over flat reject/retry while satisfying the safety and budget gates.

## WHAT WOULD REFUTE IT
A safe-base-preserving first-valid selector tying or beating selective priority on held paired progress under equal work would refute the claimed added value of priority ordering. The existing flat arm tying or winning, a nonpositive paired lower bound, or increased harmful selections would also refute the narrower A/B claim.

## WAS THAT CHECKED
No. Held outcomes, matched budgets, and an external post-selection evaluator were checked, so the narrow comparison could have failed. However, the sole rival accepts the first valid candidate and lacks selective priority’s safe-base preservation. The artifact does not test the cheapest serious control: preserve an already-valid base, otherwise accept the first fully valid candidate. Thus it cannot distinguish priority-ordering value from the built-in preservation advantage.

## EVIDENCE
`"honest_verdict": "complete: selective priority positive gate passed on held exact replay"`

`"flat_reject_retry": "Inspect the same candidates in frozen order. Reject any failed constraint and accept the first fully valid candidate."`

`"selective_priority": "Preserve a valid base byte string. Otherwise choose the exact hard-binding-soft lexicographic minimum with a stable index tie."`

`"direction": "selective_minus_flat"`

`"estimate": 0.125`

`"lower_bound": 0.0763888888888889`

`"headline_split": "held"`

`"equal_observed_work": true`

`"verifier_is_oracle": false`

`"outcome_evaluator": "Exp6811 exact transition replay after selection"`

`"safe_action_identity_by_arm"`

`"flat_reject_retry"`

`"rate": 0.35714285714285715`

`"selective_priority"`

`"rate": 1.0`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6823_v595_branch_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the limited receipt assertion, an available terminal artifact omitted from the rows, a present artifact classified as missing, or a branch marked blocked despite complete eligible independent evidence would refute it.

## WAS THAT CHECKED
Yes, for the receipt assertion: task coverage, artifact hashes, eligibility, terminal classes, manifest agreement, and branch gate failures were recorded. No comparative method-value or solve claim was asserted.

## EVIDENCE
`"solve_claim": false`; `"verdict_class": "partial"`; `"executed_task_count": 14`; `"task_count": 14`; `"differences": []`; `"matches": true`; `"flagged": 2`; `"missing": 9`; `"positive": 2`; `"disposition": "blocked"`; `"independent_cold_audit_required": true`; `"observed": "missing"`; `"eligible_for_positive_claim": false`

## RECOMMENDATION
KEEP
