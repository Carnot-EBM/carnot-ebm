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
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_8075_v699_constraint_projection_kernel.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8076_v699_projected_online_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Projected online learning yields a complete null result with zero generalized learning benefit across the evaluated causal online trajectory.

## WHAT WOULD REFUTE IT
Any observation of positive learning benefit for the projected arm, such as candidate updates passing alpha validation on guard slots with improved Brier score over the incumbent (`passed: true`), successful adoption of non-fallback parameters, or recording a `generalized_learning_benefit_score` > 0 or `benefit_credit` > 0.

## WAS THAT CHECKED
Yes. Candidate parameter updates were tested across line-search steps in `alpha_check_rows` against incumbent and initial Brier metrics on guard slots, subject to fresh feedback admission verification in `admission_block_rows` and `admission_consumption_rows`, with outcomes tracked in `durable_commit_rows`, `fallback_rows`, and `behavior_counts`.

## EVIDENCE
- `honest_verdict`: `"complete_null_projected_online_learning"`
- `generalized_learning_benefit_score`: `0`
- `benefit_credit`: `0`
- `noop`: `120`
- `passed`: `false`
- `reasons`: `["incumbent.brier"]`
- `fallback`: `"incumbent"`
- `status`: `"deferred"`
- `claim_scope`: `"Causal finite exposed development trajectory only. No future safety theorem, H2 significance, generalized improvement, live generation or deployment credit."`

## RECOMMENDATION
KEEP

## experiment_8077_v699_projected_learning_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Projected online learning achieves no qualified cost benefit or beneficial changed sources over fresh learning on the audited development trajectory, resulting in a complete null audit.

## WHAT WOULD REFUTE IT
The null claim would be refuted by observing a statistically significant positive cost gain (`gain >= 0.02` with `margin_p < 0.05`) and at least 5 beneficial changed sources (`beneficial_changed_sources >= 5`) for `projected_fresh` over `ray_fresh`.

## WAS THAT CHECKED
Yes; checked via moving-block bootstrap tests across block lengths 16, 32, and 64 (10,000 draws each) in `H2.tests` and evaluated against expected scientific thresholds in `gate_check_summary`.

## EVIDENCE
`honest_verdict`: `complete_null_projected_learning_audit`
`beneficial_changed_sources`: `0`
`qualified_benefit`: `false`
`comparison`: `ray_fresh cost minus projected_fresh cost`
`gain`: `0.0`
`raw_p`: `1.0`
`projected_learning_benefit_score`: `0`
`claim_scope`: `Independent replay of historically exposed cached development sources; conditional H2, retention and private recovery only; no unseen-environment or generalized learning claim.`

## RECOMMENDATION
KEEP

## experiment_8078_v699_feature_cache_core.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8079_v699_feature_cache_lifecycle.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8080_v699_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any claim of positive task performance, causal efficacy, or level solves appearing in this artifact would refute its non-claim status, because upstream gate validation failed and zero evaluation runs were executed.

## WAS THAT CHECKED
Yes. Gate checks verified that upstream input was missing (`gate_check_summary` recorded failure on `validate_is_file`), execution was halted with no game or model runs completed, and the run was explicitly recorded as blocked with no causal credit claimed.

## EVIDENCE
- `"causal_benefit_claimed": false`
- `"claim_scope": "This 20261003 invocation reduces only authenticated content beyond Exp8067; exposed observational evidence grants no causal or solve credit."`
- `"blocked_input_count": 1`
- `"completed_count": 0`
- `"current_game_runs": 0`
- `"current_model_invocation_count": 0`
- `"validity": false`
- `"readiness": 0`
- `"passed": false`
- `"observed": "missing"`
- `"scientific_benefit": null`
- `"solve_provenance": "Only authenticated input events have live discovery provenance; aggregation claims no solves."`

## RECOMMENDATION
KEEP

## experiment_8081_v699_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration provides no whole-service benefit over host execution (bounding hypothetical infinite arithmetic acceleration to ~1.0000001x–1.0000004x), establishing a null workload boundary and warranting no hardware acquisition.

## WHAT WOULD REFUTE IT
The null claim would be refuted if the arithmetic component under test constituted a meaningful fraction of total runtime rather than ~1e-7, if `infinite_arithmetic_ceiling` showed substantive speedup, if `current_device_execution_count` were non-zero with measured whole-service speedup over host, or if any board satisfied its terminal criteria (`terminal_criterion_met` being true).

## WAS THAT CHECKED
Yes. Component-level timing across multiple transaction and constraint conditions was measured in `acceleration_bounds` and `compatible_component_fractions`, and physical board readiness/continuity was evaluated across three platforms in `board_rows`.

## EVIDENCE
`honest_verdict`
`complete_null_conditional_hardware_workload_boundary`
`verdict_class`
`null`
`purchase_recommendation`
`none`
`claim_scope`
`Read-only historical custody and conditional bounds only. No current device execution, speed or purchase claim.`
`eligible_fraction`
`1.098641084965502e-07`
`3.1947395920731243e-07`
`3.900277847277712e-07`
`infinite_arithmetic_ceiling`
`1.0000001098641205`
`1.0000003194740612`
`1.0000003900279368`
`current_device_execution_count`
`0`
`acquisition_relevance`
`defer: no measured board whole-service benefit`
`terminal_criterion_met`
`false`

## RECOMMENDATION
KEEP

## experiment_8082_v699_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The milestone is blocked from science readiness due to failed hypothesis tests, showing null retained causal learning benefit in H2 and source interaction gains that fail the pre-registered margin and safety in H1.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if either primary hypothesis demonstrated statistically significant qualified benefit: specifically, H1 showing an observed gain exceeding the 0.02 margin with safety passed and family p < 0.05, or H2 showing an observed cost reduction gain over `ray_fresh` exceeding 0.02 with at least 5 beneficial changed sources, enabling gap closure and science readiness.

## WAS THAT CHECKED
Yes. It was checked in `H1` (evaluated over 10,000 completed draws across 95 complete source groups against a 0.02 margin), in `H2` (evaluated across block-bootstrap tests with block lengths 16, 32, and 64 over 10,000 draws comparing `ray_fresh` cost to `projected_fresh` cost), and in `gap_decisions` aggregating the 13 audited task reductions.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v699_capstone`
- `science_ready`: `false`
- `H1`:
  - `positive_claim`: `false`
  - `observed_gain`: `0.005263157894736842`
  - `margin`: `0.02`
  - `raw_p`: `1.0`
  - `family_p`: `1.0`
  - `safety_passed`: `false`
  - `benefit_passed`: `false`
- `H2`:
  - `positive_claim`: `false`
  - `observed_gain`: `0.0`
  - `margin`: `0.02`
  - `raw_p`: `1.0`
  - `family_p`: `1.0`
  - `beneficial_changed_sources`: `0`
  - `qualified_benefit`: `false`
- `gap_decisions`:
  - `retained_causal_learning`: `projected_versus_ray_later_benefit_null`, `closed`: `false`
  - `useful_source_verification`: `exposed_development_H1_margin_or_changed_source_floor_failed`, `closed`: `false`
  - `reproducible_deployment`: `bounded_host_cache_transactions_only_complete_acquisition_unpriced`, `closed`: `false`
- `claim_scope`: `Exact task custody and historically exposed development reductions; no generalized learning, new live solves or device speed claims.`

## RECOMMENDATION
KEEP
