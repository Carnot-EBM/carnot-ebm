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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 3 |

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

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8081_v699_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Specialized hardware acceleration yields no whole-service benefit for this workload because accelerable arithmetic accounts for less than 4e-7 of runtime (limiting infinite arithmetic speedup to under 1.0000004x), supporting a null workload boundary and a recommendation to defer hardware acquisition.

## WHAT WOULD REFUTE IT
The claim would be refuted if the profiled execution time showed a substantial fraction of time spent in accelerable gradient arithmetic (producing an `infinite_arithmetic_ceiling` or `arithmetic_only100x_ceiling` well above 1.0x), or if any candidate hardware board in `board_rows` achieved `terminal_criterion_met: true` with measured current latency demonstrating whole-service acceleration over the host baseline.

## WAS THAT CHECKED
Yes. It was checked in `compatible_component_fractions` and `acceleration_bounds`, where component profiling across conditions demonstrated that gradient arithmetic accounted for only ~1.1e-7 to 3.9e-7 of execution time (with ceilings <= 1.0000004x), and in `board_rows`, where KV260, PolarFire, and GateMate all failed their terminal criteria and recorded zero whole-service benefit.

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
`infinite_arithmetic_ceiling`
`1.0000001098641205`
`3.900277847277712e-07`
`1.0000003900279368`
`acquisition_relevance`
`defer: no measured board whole-service benefit`
`terminal_criterion_met`
`false`
`current_device_execution_count`
`0`
`verifier_is_oracle`
`false`

## RECOMMENDATION
KEEP

## experiment_8082_v699_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The milestone is blocked without science readiness, as neither primary hypothesis demonstrates a qualified benefit over baseline (both H1 and H2 yield an honest null).

## WHAT WOULD REFUTE IT
A statistically significant positive treatment gain meeting or exceeding preregistered thresholds on unexposed evaluation data: specifically, H1 showing an observed gain of at least 0.02 with an adjusted p-value below 0.05 and safety passing, or H2 showing a cost gain of at least 0.02 with at least five beneficial changed sources, qualified benefit passing, and retention safety intact.

## WAS THAT CHECKED
Yes. Both hypotheses were evaluated against their preregistered 0.02 margin across 10,000 bootstrap draws (including block lengths 16, 32, and 64 for H2). Refutation was given a real opportunity to occur, but the data showed that neither passed benefit: H1 achieved an observed gain of ~0.0053 with a raw p-value of 1.0 (benefit and safety failed), while H2 showed an observed gain of 0.0 with 0 beneficial changed sources and a raw p-value of 1.0.

## EVIDENCE
- `"honest_verdict"`: `"complete_blocked_v699_capstone"`
- `"science_ready"`: `false`
- `"claim_scope"`: `"Exact task custody and historically exposed development reductions; no generalized learning, new live solves or device speed claims."`
- `"generalized_learning_benefit_score"`: `0`
- Under `H1`: `"positive_claim"`: `false`, `"observed_gain"`: `0.005263157894736842`, `"margin"`: `0.02`, `"raw_p"`: `1.0`, `"benefit_passed"`: `false`, `"safety_passed"`: `false`
- Under `H2`: `"positive_claim"`: `false`, `"observed_gain"`: `0.0`, `"margin"`: `0.02`, `"raw_p"`: `1.0`, `"beneficial_changed_sources"`: `0`, `"qualified_benefit"`: `false`
- Under `gap_decisions`: `"decision"`: `"exposed_development_H1_margin_or_changed_source_floor_failed"`, `"decision"`: `"projected_versus_ray_later_benefit_null"`

## RECOMMENDATION
KEEP

## experiment_8083_v700_contract_custody.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is an administrative custody receipt that explicitly disclaims any fresh scientific outcomes or comparative performance claims.

## WAS THAT CHECKED
No. As a non-comparative administrative custody record, no empirical hypothesis was formulated or subjected to potential falsification.

## EVIDENCE
- `claim_scope`: `administrative authority and historical controls only; zero fresh scientific outcomes`
- `current_work_receipt`
- `model_invoked`: `false`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `inference_substrate_class`: `no_model_load`
- `independent_scientific_units`: `0`
- `eligible_count`: `0`
- `excluded_count`: `14`
- `honest_verdict`: `complete_disqualified_research-roadmap-vNEXT_md`
- `field_principles`: `Preserve MODEL_SPECS so administrative custody cannot imply fresh scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8084_v700_fresh_cohort_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None, because the artifact makes no comparative claim and explicitly disclaims measuring any primary hypothesis. If interpreted as asserting cohort readiness or experimental progress, observation of passing gate checks and successful model evaluations would be required, whereas execution was blocked before any trials were run.

## WAS THAT CHECKED
No. No comparative hypothesis or model performance was evaluated; only pre-run gate preconditions were evaluated and recorded as blocked.

## EVIDENCE
`claim_scope`
`Frozen methods and recorded-history-separated public cohort custody; no current model work, verification benefit or general lifelong learning. Private fixtures establish protocol behavior only.`
`methodology_note`
`Allocation precedes labels. Unknown labels remain excluded in original slots. Gaussian local memory changes the declared mechanism; no primary hypothesis is measured by this experiment.`
`honest_verdict`
`complete_blocked_available_history`
`verdict_class`
`blocked`
`completed_count`
`0`

## RECOMMENDATION
KEEP

## experiment_8085_v700_radial_memory_kernel.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The radial memory kernel achieves complete positive qualification and lifecycle readiness across 64 supplied numerical systems.

## WHAT WOULD REFUTE IT
Failure of candidate kernel updates under an independent, non-oracle verifier or on held-out evaluation data (such as candidate Brier score or objective failing to improve over the fixed-center baseline, or failing admission checks).

## WAS THAT CHECKED
No. The verifier is the oracle defining correctness (`verifier_is_oracle = true`), making the lifecycle verification circular by construction, with no independent verification or out-of-sample generalization testing.

## EVIDENCE
`honest_verdict`
`complete_circular_positive_radial_memory_kernel`
`verifier_is_oracle`
`claim_scope`
`64 supplied numerical systems and private memory lifecycle only; no independent verification or learning benefit`
`kernel_ready_score`
`1`
`generalized_learning_benefit_score`
`0`
`independent_count`
`0`
`inference_substrate`
`aggregation_from_upstream_artifacts`
`inference_substrate_class`
`no_model_load`

## RECOMMENDATION
CORRECT_THE_RECORD
