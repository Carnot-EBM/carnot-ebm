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
| NO_CLAIM | 4 |

## experiment_8933_v764_compatible_replay_methods.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Compatible replay has not demonstrated comparative benefit in the constructed development runs; independent evaluation remains unopened.

## WHAT WOULD REFUTE IT
Paired Brier differences establishing an advantage over both guarded-current updates and uniform replay, while meeting the frozen-arm improvement and retention requirements, would contradict the descriptive null.

## WAS THAT CHECKED
Yes, descriptively: compatible replay’s mean Brier differences are positive against guarded-current updates and uniform replay, so both serious comparators outperform it. Replay selection uses feature distance; scoring uses Brier, allowing either outcome. The oracle supplies correctness labels, but no added verifier value is claimed.

Independent evaluation was not checked. The result supports a development null, not equivalence or generalization.

## EVIDENCE
- `honest_verdict`: `complete_null_compatible_replay_methods`
- `against`: `guarded_current`; `mean`: `0.01511427512174313`
- `against`: `uniform_replay`; `mean`: `0.010792920669549191`
- `scope`: `development descriptive only; no readiness or benefit gate`
- `evaluation_opened`: `false`
- `methodology_note`: `Oracle constructed streams do not establish natural-language generalization, prospective benefit or a verifier moat.`

## RECOMMENDATION
KEEP

## experiment_8934_v764_depth_control.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The paired control did not establish that depth 3 increases final-answer error relative to depth 2.

## WHAT WOULD REFUTE IT
A positive depth-3-minus-depth-2 error difference with a paired confidence interval wholly above zero would contradict the reported null.

## WAS THAT CHECKED
Yes. Across 50 paired graphs, depth-3 error was 0.78 versus 0.84 at depth 2; the difference was −0.06, with an interval spanning zero. Discordant pairs occurred in both directions, so the outcome was not predetermined.

The oracle supplies correctness labels; no verifier-value claim is made. No generalization benefit is claimed. Depth 2 supplies the relevant comparator. Reported exclusions and censoring are zero, and the displayed paired rows contain both error fields.

## EVIDENCE

- `honest_verdict`: `complete_null_paired_depth_control`
- `error_rate_k2`: `0.84`; `error_rate_k3`: `0.78`
- `mean_difference`: `-0.06`; `low`: `-0.22239701586189017`; `high`: `0.10239701586189018`
- `k2_correct_k3_wrong`: `7`; `k2_wrong_k3_correct`: `10`
- `accepted`: `false`; `collection_authorized`: `false`
- `excluded_count`: `0`; `censored_count`: `0`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; elided rows were not inspected. This null does not establish equivalence.

## experiment_8935_graph_corpus_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check and makes no experimental or comparative claim.

## WAS THAT CHECKED
No experimental claim was tested. The prerequisite gate was checked and failed: the observed readiness score was 0, while 1 was required. No rerun or upstream validation was performed for this review.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `depth_source_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8937_v764_calibration_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a methods-readiness receipt, not a comparative scientific claim. Failed implementation controls or Brier values inconsistent with their probability/error operands would undermine its limited readiness assertion.

## WAS THAT CHECKED
Yes, within the fixture scope: implementation controls and owned validation report passing. Same-feature logistic—the cheapest serious baseline—is present and beats energy on fixture Brier. The benefit gate fails, and the artifact records a null.

Live quality and independent generalization were not checked; readiness cannot establish either.

## EVIDENCE

- `title`: `Frozen response calibration methods`
- `methodology_note`: `Fixtures verify implementation only; graph labels remain outside the response-only scorer. No natural-language generalization or verifier moat is measured.`
- `honest_verdict`: `complete_null_calibration_methods`
- `verdict_class`: `null`
- `brier_vs_logistic`: `estimate`: `-0.057312465264168964`; `one_sided_bound`: `-0.10893172085153909`; `passed`: `false`
- `benefit_passed`: `false`
- `real_quality_evaluated`: `false`
- `independent_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8940_v764_compatible_replay_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. Operational readiness would be refuted by retention mutation, feedback reaching training before prediction sealing, or restart divergence from uninterrupted controls.

## WAS THAT CHECKED
Partly visible: retention controls pass, and the shown prediction precedes label evaluation and feedback release. Restart outcomes and aggregate comparator results are outside the excerpt. Benefit assessment is explicitly assigned to Exp8941; this artifact does not claim verifier value or generalization.

## EVIDENCE
- `honest_verdict`: `complete_null_compatible_replay_streams`
- `benefit_required`: `false`
- `benefit_owner`: `Exp8941`
- `replay_rows_ready_score`: `Complete causal rows, unchanged methods and owned checks; loss improvement is not a gate.`
- `methodology_note`: `23040 Brier query comparisons retain typed error alongside each score;46080 endpoint cells. Exp8941 owns benefit. Oracle graphs establish no natural-language generalization.`

## RECOMMENDATION
KEEP

Not checked: elided rows or external artifacts. Risk: interpreting this readiness receipt as demonstrated benefit.

## experiment_8941_v764_compatible_replay_benefit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This fixed compatible-replay configuration did not demonstrate the prespecified future benefit, warranting retirement of that configuration.

## WHAT WOULD REFUTE IT
All ten acceptance bounds passing on valid evaluation data, including a future Brier upper bound at most −0.01 against frozen verification and strictly below zero against guarded-current and uniform-replay baselines.

## WAS THAT CHECKED
Yes. The artifact reports paired comparisons across 24 streams with multiplicity-adjusted bounds. Three displayed future-benefit gates fail. Guarded-current and uniform replay are serious baselines; both achieve lower mean future Brier than compatible replay. Passing retention bounds cannot compensate.

The oracle defines correctness, but the headline makes no claim about verifier value. Reported controls include zero evaluation units used for selection and read-only retention evaluation; the generalization score is zero. Visible scoring rows are complete. False promotion eligibility flags describe rejected updates, rather than invalid scoring rows; the failed repository-health check is explicitly separate.

This supports failure to demonstrate the specified benefit, without establishing zero effect.

## EVIDENCE

- `honest_verdict`: `complete_null_compatible_replay_benefit`; `compatible_benefit_score`: `0`.
- `compatible_replay_minus_frozen_verified`: `upper`: `6.387491138142021e-05`; `threshold`: `-0.01`; `passed`: `false`.
- `compatible_replay_minus_guarded_current`: `mean`: `0.005552042868489988`; `passed`: `false`.
- `compatible_replay_minus_uniform_replay`: `mean`: `0.0008375842080466413`; `passed`: `false`.
- `evaluation_units_used_for_selection`: `0`; `independent_generalization_score`: `0`; `required_checks_passed`: `true`.
- `completed_count`: `23040`; `intended_count`: `23040`; `excluded_count`: `0`; `censored_count`: `0`.

## RECOMMENDATION
KEEP

Elided rows and upstream execution were not independently checked; this null supports only the stated configuration and acceptance criterion.

## experiment_8942_v764_replay_complete_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Complete-service costs were measured, but compatible replay has no demonstrated cost advantage and does not qualify for deployment.

## WHAT WOULD REFUTE IT
A simultaneous cost-difference interval entirely below zero against guarded current would refute the cost null; combined with a passing quality gate, it would support deployment. Incomplete timing rows or failed durability controls would refute measurement readiness.

## WAS THAT CHECKED
Yes. Across 24 paired process/world seeds, compatible replay’s complete-service difference against guarded current spans zero, so a cost advantage was possible to detect but was not established. The authenticated quality gate also fails.

Elapsed-time measurement is independent of verifier-defined correctness. The artifact claims neither verifier added value nor generalization. Guarded current and uniform replay provide substantive comparators. Completion counts match the intended count, and displayed timing rows are complete; no false validity flag is shown on them.

## EVIDENCE
- `honest_verdict`: `complete_null_replay_complete_cost`; `verdict_class`: `null`.
- `cost_rows_ready_score`: `1`; `Complete costs and valid controls; no speedup or learning prerequisite.`
- `arm`: `compatible_replay`; `against`: `guarded_current`; `metric`: `complete_process_s`; `independent_count`: `24`; `lower`: `-0.2133998699147004`; `upper`: `0.6134695434980337`.
- `compatible_benefit_score`: `0`; `deployment_recommendation`: `false`.
- `completed_count`: `288`; `intended_count`: `288`; `excluded_count`: `0`; `failed_count`: `0`.
- `exposure_scope`: `Finite constructed recurring topology streams only; no natural-language generalization, verifier moat or generator learning.`

## RECOMMENDATION
KEEP

Not checked: reruns or elided rows. Scope limit: the supplied finite-stream evidence.

## experiment_8943_v764_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. The accounting would be contradicted by reporting failed prerequisites as qualified scientific success.

## WAS THAT CHECKED
Yes, within the visible reconciliation: the depth prerequisite failed, collection remained blocked, and detector benefit was not evaluated. The capstone preserves that blocked outcome. Oracle-defined correctness is not presented as evidence of verifier value, and readiness is not presented as generalization.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v764_capstone`
- `scientific_benefit`: `false`
- `independent_generalization_score`: `0`
- `accepted`: `false`
- `missing_reason`: `depth_source_ready_score=0; no collection executed`
- `missing_reason`: `no admitted collection corpus`
- `paper_ready_scope`: `No separately authorized publication process; accounting only.`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream execution. This accounting receipt establishes no comparative scientific benefit.
