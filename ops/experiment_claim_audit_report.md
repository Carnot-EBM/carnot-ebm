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
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 5 |

## experiment_8212_v709_memory_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Calibrated error-center memory provides no statistically significant independent benefit over fixed-public-center controls, resulting in a complete null verdict for hypothesis H2.

## WHAT WOULD REFUTE IT
The null claim would be refuted if the artifact's own data showed a statistically significant positive benefit for the error-center method over the control baselines—specifically, if `h2_passed` evaluated to true, if `improved_sources` reached or exceeded the minimum threshold of 5, or if the moving block-bootstrap `lower_bound` met or exceeded the 0.02 acceptance gate.

## WAS THAT CHECKED
Yes; checked in `H2` across 158 completed evaluation sources using 10,000 moving original-slot block-bootstrap draws (tested across block lengths 8, 16, and 32), where non-zero headroom was available (`available_typed_cost_headroom` of 0.5126582278481012) and refutation was given a genuine opportunity to occur.

## EVIDENCE
- `honest_verdict`: `"complete_null_independent_memory_benefit_audit"`
- `verdict_class`: `"null"`
- `h2_passed`: `false`
- `improved_sources`: `0`
- `available_typed_cost_headroom`: `0.5126582278481012`
- `mean_gain`: `-0.03164556962025317`
- `lower_bound`: `-0.090625`
- `gain_lower_bound_minimum`: `0.02`
- `beneficial_sources_minimum`: `5`
- `decision_benefit_claim`: `false`
- `completed_count`: `158`
- `support_sufficient`: `true`

## RECOMMENDATION
KEEP

## experiment_8213_v709_prospective_request_recorder.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8214_v709_prospective_service_measurement.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8215_v709_arc_authoritative_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8216_v709_hardware_workload_obligations.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware workload obligations result in a complete null, with Amdahl speedup bounded at ~1.00016x (failing the 100x target) and zero current device execution benefit across evaluated targets.

## WHAT WOULD REFUTE IT
The claim would be refuted if the artifact's data demonstrated an Amdahl maximum speedup meeting the target (`supports_100x` as `true` with a sufficiently low serial fraction), or if any target board or device showed authenticated current hardware execution delivering whole-service speedup with `terminal_criterion_met` as `true`.

## WAS THAT CHECKED
Yes. It was checked in `amdahl_bounds` across pipeline execution arms (which showed serial fractions > 0.9998 and `supports_100x` as `false`), in `board_rows` across KV260, PolarFire, and GateMate (all showing `current_hardware_execution` as `false` and `terminal_criterion_met` as `false`), and in `access_obligations` (verifying NPU and TSU access remain blocked).

## EVIDENCE
`"honest_verdict"`: `"complete_null_hardware_workload_obligations"`
`"verdict_class"`: `"null"`
`"claim_scope"`: `"Software reducer and host primitive limits; no current model or board execution"`
`"supports_100x"`: `false`
`"maximum_speedup"`: `1.0001599271488204`
`"maximum_speedup"`: `1.000093299462364`
`"serial_fraction"`: `0.9998400984237827`
`"serial_fraction"`: `0.9999067092416137`
`"current_device_execution_count"`: `0`
`"current_hardware_execution"`: `false`
`"terminal_criterion_met"`: `false`
`"generalized_learning_benefit_score"`: `0`
`"independent_generalization_score"`: `0`
`"deployment_claim"`: `false`

## RECOMMENDATION
KEEP

## experiment_8217_v709_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8218_v710_contract_replay_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8219_v710_utility_patch_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation within the artifact's own rows or summary metrics reporting empirical utility improvement, generalization benefit, fitted patches, or a measured win of a utility patch method over a baseline or comparator arm.

## WAS THAT CHECKED
no. The artifact lacks any evaluation of the registered hypotheses H1 and H2, fitted zero patches, selected no primary comparator, and left all 128 rows unmeasured.

## EVIDENCE
`claim_scope`
`"Frozen finite utility method and fit/tune descriptive residuals only; no benefit measurement."`
`scientific_benefit_measured`
`false`
`fitted_patch_count`
`0`
`H1`
`measured_here`
`false`
`H2`
`acceptance_gates`
`"registered_unmeasured"`
`primary_comparator_selected`
`null`
`honest_verdict`
`"complete_null_utility_patch_methods"`
`rows`
`arm`
`"utility_protocol_registration"`
`evaluation_status`
`"unmeasured"`
`methodology_note`
`"Cached source annotations supply fit/tune residuals. No patch was fitted here. Reserved identities and original missing masks are target free. H1/H2 remain registered and unmeasured; exposed development supplies no independent generalization or distribution-free safety claim."`

## RECOMMENDATION
KEEP
