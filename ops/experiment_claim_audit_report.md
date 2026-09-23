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
| CLAIM_SUPPORTED | 1 |
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7529_v658_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The required V658 science is absent or externally gated, despite complete capstone accounting.

## WHAT WOULD REFUTE IT
A valid completed row showing that a required scientific branch ran and produced positive, aggregate-eligible evidence—or that the stated prerequisite was satisfied while the artifact still classified that branch as blocked—would refute the claim.

## WAS THAT CHECKED
Yes. The acceptance gates, claim ledger, and ordered task rows separately check source inventory, producer existence, completion, validity, exclusion, and positive-aggregate eligibility. They record failed prerequisites, missing producers, and unstarted/censored tasks rather than contrary completed science.

## EVIDENCE
`complete_blocked_required_v658_science_absent_or_externally_gated`; `failed_count`; `11`; `fresh_eligible_groups`; `expected`; `480`; `observed`; `0`; `positive_aggregate_eligible`; `false`; `qualified_value`; `0`; `ready_value`; `0`; `artifact_path`; `null`; `attempted`; `false`; `censored`; `true`; `unstarted`; `true`; `model_invoked`; `false`; `verifier_is_oracle`; `false`; `positive_scientific_claim`; `false`

## RECOMMENDATION
KEEP

## experiment_7531_b2_induction_gate_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact explicitly disclaims any numeric gate quality or game efficacy claims and marks the run as feasibility-only with a failed sample floor, there is no substantive empirical claim asserted to refute. If the run had claimed that the induction gate demonstrated positive headroom or was ready to ship, that claim would be refuted by demonstrating that the positive control was degenerate, that generation was truncated by a token cap, or that frame progress occurred independently of induction plans.

## WAS THAT CHECKED
No. No comparative claim was asserted; the harness recorded feasibility telemetry, noted that the sample floor was not met, and explicitly annotated why the positive control was degenerate rather than claiming positive value or readiness.

## EVIDENCE
- `publication_mode`: `"feasibility_only"`
- `numeric_gate_quality_claim`: `false`
- `hidden_game_efficacy_claim`: `false`
- `gate_ready_to_ship`: `false`
- `honest_verdict`: `"complete_feasibility_only_sample_floor_not_met_and_degenerate_positive_control"`
- `sample_floor`: `met`: `false`
- `positive_control_headroom_exists`: `false`

## RECOMMENDATION
KEEP

## experiment_7532_v659_contract_methods.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7533_v659_tool_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact makes no comparative claim. Falsifying an empirical claim about tool-output grounding performance would require observing that candidate intervention outputs fail to outperform a baseline or donor-swapped comparator under the cost policy on evaluation groups.

## WAS THAT CHECKED
No. The artifact lacks any model inference, generation calls, or comparative evaluation between intervention and control arms; benefit was explicitly unmeasured.

## EVIDENCE
`"title"`: `"V659 label-blind tool-output grounding protocol"`
`"benefit_measured"`: `false`
`"positive_claim"`: `false`
`"verdict_class"`: `"null"`
`"honest_verdict"`: `"complete_null_tool_protocol_ready_benefit_unmeasured"`
`"honest_no_headroom_annotation"`: `"protocol_only_benefit_unmeasured"`
`"inference_substrate_class"`: `"no_model_load"`
`"model_invoked"`: `false`
`"forward_calls_attempted"`: `0`
`"generation_calls_attempted"`: `0`
`"model_loads_attempted"`: `0`
`"private_llm_off_real_environment_smoke"`: `"not_applicable_reporting_only_protocol"`
`"tool_protocol_ready_score"`: `1`
`"validation_receipts"`

## RECOMMENDATION
KEEP

## experiment_7534_v659_count_memory.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The count memory mechanism is verified and ready for continuous self-learning deployment based on passing analytical acceptance gates and constructed fixture controls.

## WHAT WOULD REFUTE IT
The claim of functional readiness and learning efficacy would be refuted by demonstrating that the count memory update rule fails to outperform static priors (`frozen`) or permuted controls (`permuted_local`) on natural, non-constructed data streams where the evaluation verifier is not the defining oracle. Furthermore, requiring the method to beat trivial or permuted baselines on the fixture streams would directly refute readiness, as the local count memory arm actively loses to or ties them across multiple evaluated streams.

## WAS THAT CHECKED
No. Refutation was not given a genuine chance to occur. Correctness is evaluated entirely against constructed synthetic controls where the verifier is the oracle (`verifier_is_oracle: true`, `oracle_fixture: true`). The readiness score (`count_memory_ready_score: 1`) is true by construction because its gating criteria only check software mechanics (arithmetic, release chronology, and restart durability) while explicitly excluding empirical performance (`constructed evidence never becomes an empirical claim`). Where comparative predictive metrics were scored, the method underperformed the frozen and permuted baselines, but the readiness verdict was shielded from failure.

## EVIDENCE
- `"honest_verdict": "complete_circular_positive_count_memory_qualified"`
- `"verdict_class": "circular_positive"`
- `"verifier_is_oracle": true`
- `"oracle_fixture": true`
- `"count_memory_ready_score": 1`
- `"continuous_self_learning_task": true`
- `"evidence_class": "constructed_control"`
- `"positive_claim": false`
- `"condition": "constructed evidence never becomes an empirical claim"`
- `"principle": "Oracle fixtures test implementation, not natural-data benefit."`
- `"principle": "Oracle-defined fixtures cannot support an oracle-distinct benefit claim."`
- `"count_memory_ready_score": "A bare 0/1 qualifies arithmetic, chronology, and restart only."`
- `"stream": "conditional_drift_stable_global_prevalence"`
- `"local": {"mean_brier": 0.35762759924385634, "n_scored": 4}`
- `"frozen": {"mean_brier": 0.3400000000000001, "n_scored": 4}`
- `"permuted_local": {"mean_brier": 0.26671029149315884, "n_scored": 4}`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7535_v659_native_pilot.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation that an admissible single GPU was unallocated and available during the admission window, or any assertion of positive model capability or comparative benefit despite execution being blocked.

## WAS THAT CHECKED
Yes. The external precondition gate checked `owned_gpu_available` under `acceptance_gate_results` and `gpu_admission_receipt` over 61 observations across 300 seconds, verified both GPUs were occupied, and blocked execution without evaluating model performance.

## EVIDENCE
`"positive_claim"`: `false`
`"benefit_measured"`: `false`
`"model_invoked"`: `false`
`"execution_device"`: `"none_precondition_blocked"`
`"honest_verdict"`: `"complete_blocked_owned_gpu_available"`
`"verdict_class"`: `"blocked"`
`"rows"`: `[]`

## RECOMMENDATION
KEEP

## experiment_7536_fit_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is a pre-execution gate-check receipt recording that the experiment was blocked prior to running, advancing no empirical or comparative claim.

## WAS THAT CHECKED
No. Execution was blocked at the pre-gate stage before any intervention or experimental evaluation could take place.

## EVIDENCE
- `schema`: `"blocked_gate_check_v1"`
- `status`: `"blocked"`
- `honest_verdict`: `"blocked_gate_check_failed"`
- `duration_s`: `0.0`
- `blocked_at_layer`: `"conductor_pre_gate"`
- `gate_check_summary`: `"gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7535-native-pilot.native_tool_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7537_eval_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; this artifact is an execution gate receipt documenting failed upstream preconditions, not an experimental run making a comparative or empirical claim.

## WAS THAT CHECKED
No; the experiment was blocked before execution at the conductor pre-gate stage.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`
`"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7535-native-pilot.native_tool_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP
