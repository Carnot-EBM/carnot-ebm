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
| CLAIM_SUPPORTED | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8060_source_energy_training.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8063_v698_admission_opportunity_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The retrospective admission opportunity audit establishes a complete null result with zero generalized learning benefit, unfeasible statistical safety certificates, and substantial missed opportunity and harmful admission rates.

## WHAT WOULD REFUTE IT
The null claim would be refuted by observing in the artifact's own data:
1. A positive generalized learning benefit score (`generalized_learning_benefit_score` > 0).
2. Statistically significant positive performance gain passing evaluation gates against comparator arms in closed-loop hypothesis testing (`local_passed` true, `gain_margin` true, and `raw_p` below nominal alpha).
3. Any valid, feasible certificate bounds on Brier score or cost margins (`iid_certificate_valid` true, `brier_margin_feasible` true, `cost_margin_feasible` true, or `paired_margin_feasible` true).
4. Efficient selection of useful candidates resulting in a low or zero missed opportunity rate along with zero harmful admissions under the audit contract.

## WAS THAT CHECKED
Yes. Refutation was given a genuine opportunity to occur across multiple independent evaluators and slices:
1. Feasibility certification was evaluated across 3,960 candidate pool attempts (`certificate_feasibility_rows`), where all checks for Brier margin, cost margin, and paired margin feasibility failed.
2. Closed-loop causal performance was tested via 10,000 moving-block bootstrap draws across hypotheses `H3` and `secondary_frozen` against `unconstrained` and `frozen_no_write` baselines (`historical_closed_loop`), where both hypotheses failed all gain margin and retention gates (`raw_p` of 1.0 and 0.89).
3. Candidate pools were evaluated against empirical audit contract thresholds (`pool_category_rows` and `candidate_pool_rows`), discovering 30 missed opportunities out of 37 useful gradient pools (an 81.1% missed opportunity rate) and over 1,470 harmful admissions.

## EVIDENCE
- `"honest_verdict"`
- `"complete_null_admission_opportunity_audit"`
- `"verdict_class"`
- `"null"`
- `"generalized_learning_benefit_score"`
- `0`
- `"claim_scope"`
- `"Finite retrospective common-candidate diagnosis on historically exposed text. Closed-loop causality, iid safety and generalized benefit are unsupported. No Exp8064 tuning."`
- `"iid_assumptions_satisfied"`
- `false`
- `"missed_opportunity_rate"`
- `0.8108108108108109`
- `"missed_opportunity_numerator"`
- `30`
- `"missed_opportunity_denominator"`
- `37`
- `"local_passed"`
- `"raw_p"`
- `1.0`
- `"brier_margin_feasible"`
- `"cost_margin_feasible"`
- `"paired_margin_feasible"`
- `"iid_certificate_valid"`
- `"retention_passed"`

## RECOMMENDATION
KEEP

## experiment_8064_v698_fresh_feedback_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Fresh feedback learning yields a complete null result with zero generalized learning benefit over comparator arms on the historically exposed stream, certifying causal completeness of the trajectory only.

## WHAT WOULD REFUTE IT
Statistically significant outperformance of the `fresh_admission` arm over the comparator arms (`frozen`, `unconditional`, `reused_guard`) by the prespecified hypothesis margin (0.02 in typed cost on `later_prediction_rows`), or a non-zero `generalized_learning_benefit_score`.

## WAS THAT CHECKED
Yes. The multi-arm trajectory was executed across 20 seeds and 256 stream slots, comparing `fresh_admission` against `frozen`, `unconditional`, and `reused_guard` across `candidate_commit_rows`, `alpha_check_rows`, `durable_commit_rows`, `later_prediction_rows`, `retention_rows`, and `per_seed_false_accept_rows`.

## EVIDENCE
- `honest_verdict`
- `complete_null_fresh_feedback_learning`
- `verdict_class`
- `null`
- `claim_scope`
- `Finite historically exposed development stream. Trajectory readiness certifies causal completeness only. No scientific gain, independent environment, future safety or deployment credit.`
- `generalized_learning_benefit_score`
- `0`
- `learning_trajectory_ready_score`
- `1`
- `verifier_is_oracle`
- `false`
- `genuine_headroom`
- `benefit_unassessed`
- `true`
- `measured`
- `false`
- `arms`
- `frozen`
- `unconditional`
- `reused_guard`
- `fresh_admission`
- `later_prediction_rows`
- `per_seed_false_accept_rows`
- `update_budget_rows`

## RECOMMENDATION
KEEP

## experiment_8065_v698_fresh_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Fresh admission demonstrates no qualified learning benefit or safety advantage over the reused guard baseline on the audited stream, resulting in a complete null audit outcome.

## WHAT WOULD REFUTE IT
Fresh admission demonstrating a statistically significant cost gain exceeding the margin threshold (`gain >= 0.02` with `margin_p < 0.05`), satisfying safety criteria (`safety_passed == true`), maintaining guarded retention (`retention.reused_guard.cost <= 0.02`), and beneficially changing source predictions (`beneficial_changed_sources >= 5`), which would yield `qualified_benefit == true`.

## WAS THAT CHECKED
Yes; evaluated in `primary_hypothesis_results` across block bootstrap lengths (16, 32, 64) and tested in `gate_check_summary` against predefined scientific thresholds, where all checks observed failure to beat the baseline or satisfy the margin.

## EVIDENCE
`"honest_verdict"`: `"complete_null_fresh_learning_audit"`
`"learning_benefit_score"`: `0`
`"generalized_learning_benefit_score"`: `0`
`"comparison"`: `"fresh_admission versus reused_guard"`
`"hypothesis"`: `"H3"`
`"qualified_benefit"`: `false`
`"safety_passed"`: `false`
`"beneficial_changed_sources"`: `0`
`"gain"`: `-0.011363636363636364`
`"raw_p"`: `0.998999599839936`
`"check"`: `"H3.safety_passed"`, `"observed"`: `false`
`"check"`: `"H3.beneficial_changed_sources"`, `"observed"`: `0`
`"check"`: `"H3.cost_gain"`, `"observed"`: `-0.011363636363636364`
`"check"`: `"H3.margin_p"`, `"observed"`: `0.998999599839936`
`"check"`: `"H3.retention.reused_guard.cost"`, `"observed"`: `0.07142857142857142`

## RECOMMENDATION
KEEP

## experiment_8066_v698_content_addressed_feature_service.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8067_v698_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8068_v698_hardware_feature_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration for new feature workloads is blocked and board acquisition is deferred due to disqualified upstream feature service inputs and lack of measured whole-service benefit, while historical board custody is maintained.

## WHAT WOULD REFUTE IT
The claim would be refuted if `gate_check_summary` showed upstream feature service prerequisites passing (`passed: true`, `feature_service_ready_score` observed as 1, or non-zero qualified cached rows), if `rows` admitted the new workload candidate under `conditional_reduction` as completed rather than excluded, if any entry in `board_rows` reported active device execution (`current_hardware_execution: true`) with positive whole-service acceleration (`measured_current_latency_ms` measured), or if historical board custody verification failed (`custody_valid: false`).

## WAS THAT CHECKED
Yes. Upstream prerequisites were evaluated in `gate_check_summary` across five explicit checks (all five failed with `passed: false`), the `exp8066` workload was evaluated and marked `status: "excluded"` in `rows` under `conditional_reduction`, zero device executions were confirmed (`current_device_execution_count: 0`), whole-service benefit was evaluated and deferred across all boards in `board_rows`, and cryptographic custody hashes were authenticated (`custody_valid: true`).

## EVIDENCE
- `honest_verdict`: `complete_blocked_new_workload_bound`
- `verdict_class`: `blocked`
- `workload_bound_status`: `blocked`
- `claim_scope`: `Read-only historical custody and conditional bounds only. No current device execution, speed or purchase claim.`
- `purchase_recommendation`: `none; read-only custody and estimates grant no device benefit`
- `gate_check_summary`: `passed`: `false`, `observed`: `0`, `observed`: `disqualified`
- `rows`: `arm`: `conditional_reduction`, `source`: `exp8066`, `status`: `excluded`, `exclusion_reason`: `missing_or_unqualified_current_costs`, `numerator`: `0`
- `board_rows`: `acquisition_relevance`: `defer: no measured board whole-service benefit`, `current_hardware_execution`: `false`, `custody_valid`: `true`, `measured_current_latency_ms`: `null`
- `current_device_execution_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8069_v698_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V698 capstone milestone is blocked and establishes zero qualified generalized learning benefit with unresolved upstream verification gates.

## WHAT WOULD REFUTE IT
The claim of a blocked capstone with null learning benefit would be refuted if the artifact's data showed passing upstream verification gates (e.g. `duplicate_mean_nll_drift` meeting tolerance and `fit_capture_ready_score` equal to 1), primary hypotheses H1 or H2 achieving qualified status, or hypothesis H3 demonstrating a statistically significant cost gain exceeding the 0.02 margin with passed retention safety checks, resulting in `science_ready: true` and `generalized_learning_benefit_score` greater than 0.

## WAS THAT CHECKED
Yes. Upstream gate verification checks were evaluated and recorded failures (e.g. `duplicate_mean_nll_drift` and `fit_capture_ready_score` in `gate_check_summary`), hypotheses H1 and H2 were evaluated and marked missing/unqualified with $p = 1.0$, and hypothesis H3 was subjected to 10,000 bootstrap draws across block lengths 16, 32, and 64 in `primary_hypothesis_results`, showing negative gain, failed retention safety, and zero beneficial changed sources.

## EVIDENCE
- `"honest_verdict"`: `"complete_blocked_v698_capstone"`
- `"science_ready"`: `false`
- `"generalized_learning_benefit_score"`: `0`
- `"claim_scope"`: `"Exact task custody and exposed development reductions only; no generalized learning, live solve or board speed credit."`
- `"hypothesis"`: `"H1"`, `"missing_reason"`: `"absent frozen source heads and evaluation tokens"`, `"raw_p"`: `1.0`, `"qualified"`: `false`
- `"hypothesis"`: `"H2"`, `"raw_p"`: `1.0`, `"qualified"`: `false`
- `"hypothesis"`: `"H3"`, `"comparison"`: `"fresh_admission versus reused_guard"`, `"gain"`: `-0.011363636363636364`, `"raw_p"`: `0.998999599839936`, `"beneficial_changed_sources"`: `0`, `"safety_passed"`: `false`, `"qualified_benefit"`: `false`
- `"check"`: `"duplicate_mean_nll_drift"`, `"observed"`: `0.0003044915178467278`, `"expected"`: `1e-06`, `"passed"`: `false`
- `"check"`: `"fit_capture_ready_score"`, `"observed"`: `0`, `"expected"`: `1`, `"passed"`: `false`
- `"independent"`: `0`
- `"failed"`: `2`
- `"excluded"`: `6`

## RECOMMENDATION
KEEP
