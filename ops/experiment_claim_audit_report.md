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
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6590_qwen36_constraint_first_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no CFR-benefit claim to falsify; treating the receipt’s completeness assertion separately, any required unit, arm, raw stage, exact check, checkpoint, cost, failure, model, or GPU receipt being absent or incomplete would refute it.

## WAS THAT CHECKED
No comparative benefit was checked or claimed; the artifact is explicitly completeness and provenance infrastructure.

## EVIDENCE
`honest_verdict`; `complete: every frozen Qwen CFR unit, arm, raw stage, exact check, checkpoint, cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made`; `A complete source stream is null evidence infrastructure.`

## RECOMMENDATION
KEEP

## experiment_6591_gemma4_31b_constraint_first_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation could refute a CFR-benefit claim because none is made; this is a completeness and provenance receipt.

## WAS THAT CHECKED
No comparative benefit was checked or claimed; the artifact instead checks execution completeness through per-unit rows, exact-check receipts, checkpoints, failures, and process receipts.

## EVIDENCE
`honest_verdict` `complete: every frozen Gemma CFR unit, arm, raw stage, exact check, checkpoint, cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made` `The verdict reports Gemma row completeness without claiming CFR benefit.` `A complete source stream is null evidence infrastructure.`

## RECOMMENDATION
KEEP

## experiment_6592_v575_terminal_intake_and_method_lock.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no scientific or comparative benefit claim; a losing or tying CFR arm would matter only if such a value claim were asserted.

## WAS THAT CHECKED
No; it checks receipt binding, replay readiness, method locks, and infrastructure gates, not comparative method value.

## EVIDENCE
`"no science result was created"`; `"v574_terminal_and_v575_method_replay_no_llm"`; `"paper_result_counts_as_carnot_evidence": false`; `"exact-set validity is circular"`; `"A ready intake is null infrastructure, not positive science."`

## RECOMMENDATION
KEEP

## experiment_6593_cfr_independent_row_reducer.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6594_cfr_counterfactual_authority_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The integrity audit completed as preregistered, while CFR showed no exact-success benefit over the direct control.

## WHAT WOULD REFUTE IT
Any applicable attack escaping detection, any clean replay mismatch, or any CFR-versus-direct exact-success win or loss would refute the corresponding audit-completeness or null-effect claim.

## WAS THAT CHECKED
Yes. Clean and attacked rows record replay and release outcomes, while the family-effect rows compare both CFR arms against the direct control. All shown exact-success comparisons are ties; importantly, losses were possible even though the direct arm left no improvement headroom. The oracle defines correctness, but the artifact makes no positive claim about the oracle’s added value.

## EVIDENCE
`clean_replay_matches_exp6593`: `true`; `attack_passed`: `true`; `failed_blocking_check_count`: `0`; `candidate_arm`: `always_on_cfr`; `candidate_arm`: `routed_cfr`; `control_arm`: `direct`; `exact_success_delta`: `0.0`; `wins`: `0`; `losses`: `0`; `ties`: `20`; `authority_integrity_is_scientific_benefit`: `false`; `the CFR scientific effect remains null`

## RECOMMENDATION
KEEP

## experiment_6595_invariant_projection_world_model_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Learned invariant projection improved held conservative frozen-model rollouts over no projection and norm-matched random projection, while producing no benefit when no comparable damped invariant was selected.

## WHAT WOULD REFUTE IT
A nonpositive held paired effect against no projection, an interval lower bound at or below zero, held losses or ties rather than wins, instability, or evidence that held outcomes influenced invariant selection; the random-harm component would be refuted if random projection matched or improved upon no projection.

## WAS THAT CHECKED
Yes. Thirty held conservative pairs compared learned projection with the serious no-projection baseline, with explicit effects, intervals, wins, losses, ties, and failures. Calibration and held membership were disjoint, and selection reported zero held outcomes used. The exact-invariant diagnostic effectively tied learned projection, but it was explicitly headline-ineligible and does not refute the narrower claim of improvement over no projection and random constraints.

## EVIDENCE
`"comparison_id": "conservative_learned_vs_no_projection"`; `"effect": 0.012385895802960328`; `"lower": 0.007749606247072551`; `"wins": 30`; `"losses": 0`; `"ties": 0`; `"underpowered": false`; `"comparison_id": "conservative_learned_vs_random"`; `"lower": 0.9235813561789291`; `"calibration_and_held_disjoint": true`; `"held_outcomes_read": 0`; `"failure_count": 0`; `"headline_eligible": false`; `"constraint_role": "analytic_exact_invariant_diagnostic"`; `"comparison_id": "damped_learned_vs_no_projection"`; `"effect": 0.0`; `"ties": 30`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6596_convergeflow_feasible_token_canary.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The toy convex-hull predictor-projection produced a positive result for exact-set validity and convergence under held predictor errors.

## WHAT WOULD REFUTE IT
The cheapest serious baseline—full-vocabulary nearest-token rounding—matching the projection arm’s validity and convergence while achieving lower distortion and substantially lower charged cost would refute any added-value claim.

## WAS THAT CHECKED
Yes. The `robustness_summary` shows an exact tie on held ordinary rows, and `distortion_and_cost_summary` shows nearest-token rounding winning on distortion and cost. Moreover, validity is defined by the same exact feasible set supplied to the treatment, so it is not an oracle-distinct outcome.

## EVIDENCE
`verdict_class` `circular_positive`; `verifier_is_oracle` `true`; `nearest_token_rounding`; `convex_hull_predictor_projection`; `held_perturbed`; `convergence_rate` `1.0`; `validity_rate` `1.0`; `ordinary_row_count` `30`; `mean_endpoint_distortion` `0.4233103193279481`; `mean_endpoint_distortion` `0.4913191111297726`; `total_charged_work_units` `1110`; `total_charged_work_units` `3840`; `not an oracle-distinct validity result`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6597_spectral_k_block_ising_canary.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [', with ', ' set to ', '. The success rule is '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
On charged CPU exact-enumerable fixtures, every spectral k-block arm receives positive support relative to sequential Gibbs.

## WHAT WOULD REFUTE IT
At equal charged computation—matched wall time or kernel-component updates—sequential Gibbs tying or exceeding any spectral arm’s ESS would refute the every-arm value claim; failure of stationary-distribution noninferiority for any arm would also refute it.

## WAS THAT CHECKED
No. Stationary quality was independently checked, but the efficiency comparison matched 12,000 nominal transitions while spectral arms performed 24,000 component updates versus 12,000 for sequential Gibbs. Every spectral arm lost on charged wall time, yet the OR-gate allowed that loss to be bypassed by ESS per nominal transition. No equal-wall, equal-update, or ESS-per-charged-time comparison gave the serious baseline a fair chance to tie or win.

## EVIDENCE
The headline is `complete: charged CPU software exact-enumerable fixtures support every spectral k-block arm; no FPGA, TSU, PIMI, or general hardware claim`, with `verdict_class` set to `positive`. The success rule is `ESS-per-transition or charged-wall mean > 0 with lower_bound >= 0`. All six receipts report `charged_wall_gain_with_nonnegative_lower_bound` as `false` while `ess_gain_with_nonnegative_lower_bound` is `true`. Both arms report `transitions` of `12000`, but `kernel_component_updates` is `12000` for `sequential_gibbs` and `24000` for `spectral_k_block`. The charged-wall means are `-0.4757049726007444`, `-0.5329179254285619`, `-0.534075468456366`, `-0.5501567175572027`, `-0.5331060356434801`, and `-0.5247695939227265`. The evaluator is labeled `independent_evaluator_not_sampler_input`, and `verifier_is_oracle` is `false`.

## RECOMMENDATION
ADD_MISSING_CONTROL
