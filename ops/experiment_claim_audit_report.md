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
| CLAIM_SUPPORTED | 4 |
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7126_v626_arc_loo_phase_receipts.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No value or solve claim exists to refute; the artifact only claims that a future-run phase-receipt contract is ready.

## WAS THAT CHECKED
Not applicable to value or solving. Contract readiness was checked through the synthetic attack rows and the phase-contract gate.

## EVIDENCE
`"value_measurement_run": false`, `"solve_claim_made": false`, `"contract_for_future_run": true`, `"evidence_about_exp7123": false`, `"phase_contract_ready"`, `"passed": true`, `"complete_positive_arc_phase_receipt_contract_ready_no_value_run"`

## RECOMMENDATION
KEEP

## experiment_7127_v626_adapter_withheld_arc_loo.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This completed paired run found no level-progress difference between the adapter-withheld and adapter-visible arms and makes no solve claim.

## WHAT WOULD REFUTE IT
A nonzero paired level difference—particularly control progress while the withheld arm remained at zero—or any claimed solve would refute the bounded null headline.

## WAS THAT CHECKED
Yes. Separate fresh-process arms differed in whether the `r11l` adapter was removed, and executed transitions supplied each arm’s level count; the visible control therefore had a real opportunity to outperform the withheld arm but tied it at zero.

## EVIDENCE
`adapter_withheld`; `removed_adapters`; `r11l`; `adapter_visible_control`; `removed_adapters`; `[]`; `fresh_process`; `true`; `withheld_levels`; `0`; `control_levels`; `0`; `level_delta`; `0`; `executed_transition_count`; `3`; `solve_claim_made`; `false`; `verdict_class`; `null`; `honest_verdict`; `complete_null_executed_pair_zero_withheld_levels_no_solve_claim`

## RECOMMENDATION
KEEP

## experiment_7128_v626_arc_loo_causal_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit disqualifies the experiment because the supposedly adapter-withheld arm retained access to target adapter code.

## WHAT WOULD REFUTE IT
The withheld arm showing no retained target-recipe symbols and passing the adapter-access check would refute the disqualification.

## WAS THAT CHECKED
Yes, in `adapter_access_rows` and `gate_check_summary`; the visible-control arm also passed the same check, showing that failure was not automatic.

## EVIDENCE
`"arm": "adapter_withheld"`; `"target_recipe_symbols_retained": true`; `"mismatches": ["target_adapter_code_retained"]`; `"passed": false`; `"arm": "adapter_visible_control"`; `"passed": true`; `"failed_check": "adapter_access_clean"`; `"observed_value": false`; `"verdict_class": "disqualified"`; `"solve_claim_made": false`

## RECOMMENDATION
KEEP

## experiment_7129_v626_sota_constraint_bank.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7130_v626_verifier_committed_routing.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Verifier-committed exact admission and uncertainty routing constitute a positive result, despite potentially null metric uplift.

## WHAT WOULD REFUTE IT
An exact-failing candidate being executed, a rejected candidate being promoted, an incomplete gate, or an exact-gated single-shot baseline tying or beating the router on safety while matching or improving success and cost would refute the positive-value interpretation.

## WAS THAT CHECKED
No. Procedural compliance was checked, but unsafe execution is prevented using the same `exact_penalty` that defines accepted error, so the safety result is largely true by construction. The artifact includes an exact-gated single-shot arm but does not establish that routing beats it; retry uplift is null.

## EVIDENCE
`honest_verdict` = `positive_verifier_committed_routing_complete_metric_uplift_may_be_null`; `verdict_class` = `positive`; `verifier_committed_routing_complete_score` = `1`; `accepted_error_rate` = `0.0`; `exact_rejected_actions_promoted` = `0`; `rejection_reason` = `exact_rejection_is_final`; `useful_retry_rate` = `0.0`; `arm` = `single_shot`; `exact_penalty` = `1`; `executed` = `false`; `final_action` = `abstain`; `verifier_is_oracle` = `false`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7133_v626_multiscale_sampler_prototype.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The corrected host-software multiscale sampler achieves finite-law parity on the two tested 2×2 Ising fixtures.

## WHAT WOULD REFUTE IT
A fixture with finite-law error exceeding the stated tolerance, failed detailed balance or stationarity, disagreement with the sealed probabilities, or a deliberately corrupted correction that passed verification would refute the claim.

## WAS THAT CHECKED
Yes. Complete finite-state laws and transition matrices were checked against independently sealed enumeration; detailed balance, stationarity, normalization, support, forward/reverse probabilities, and eight targeted mutations were also checked. The proposal was frozen before the sealed reference was opened.

## EVIDENCE
`honest_verdict`: `complete: corrected host-software multiscale proposal has exact finite-law parity`

`methodology`: `A fixed positive coarse-to-fine proposal uses exact-energy Metropolis-Hastings correction. Complete matrices are compared with sealed independent enumeration.`

`reference_opened_after_freeze`: `true`

`verifier_is_oracle`: `false`

`finite_law_error_max`: `2.7755575615628914e-17`

`tolerance`: `2e-12`

`maximum_error`: `1.734723475976807e-18`

`maximum_error`: `1.3877787807814457e-17`

`mutation_id`: `omitted_reverse_probability`

`mutation_id`: `wrong_temperature`

`mutation_id`: `energy_sign_reversal`

`detected`: `true`

`observed_value`: `all_structural_sealed_and_mutation_checks_pass`

`claim_boundaries`: `This is a small host-software correctness result, not a mixing result.`

## RECOMMENDATION
KEEP

## experiment_7134_v626_multiscale_sampler_benchmark.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Corrected multiscale sampling improves the preregistered bounded-host minimum-observable ESS statistic relative to local Gibbs under matched energy-evaluation budgets.

## WHAT WOULD REFUTE IT
A zero or negative mean paired ESS difference, or a tie/win by local Gibbs on the preregistered statistic, would refute the claim.

## WAS THAT CHECKED
Yes. The pooled comparison directly paired multiscale MH against local Gibbs across 40 seed-condition units using the mean paired difference in minimum-observable ESS; the design also retained failures and enforced matched budgets.

## EVIDENCE
`baseline_arm`: `local_gibbs`; `comparison_arm`: `multiscale_mh`; `statistic`: `mean paired difference in minimum observable ESS`; `paired_unit_count`: `40`; `mean_paired_ess_delta`: `42.06842302166267`; `wins`: `31`; `losses`: `9`; `ties`: `0`; `positive_advantage`: `true`; `matched_budget_verified`: `true`; `failure_rate`: `0.0`; `row_consistency_findings`: `[]`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_7135_v626_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No outcome-level observation could refute a method or value claim because none is made. The administrative completion assertion would fail if an expected upstream were unaccounted for, a flagged or missing upstream were treated as positive evidence, or matrix completion were promoted as a science result.

## WAS THAT CHECKED
Yes. The artifact checks expected-versus-observed matrix coverage, explicitly records excluded and missing upstreams, recomputes gates, and limits affected branches to null, blocked, or disqualified dispositions.

## EVIDENCE
`complete_positive_v626_evidence_matrix_without_science_promotion`; `matrix_completion_is_science_claim`: `false`; `expected_value`: `12`; `observed_value`: `12`; `passed`: `true`; `excluded_flagged_upstreams`; `missing_upstream_rows`; `no_claim_disqualified`; `no_claim_missing_or_blocked`; `null_only`

## RECOMMENDATION
KEEP
