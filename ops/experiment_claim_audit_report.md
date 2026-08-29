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
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_1736_ebt_gradient_refinement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No refuting observation applies because the artifact reports no experimental result or comparative claim.

## WAS THAT CHECKED
No; execution stopped at the pre-gate before the method was tested.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"blocked_at_layer"`: `"conductor_pre_gate"`; `"passed"`: `false`

## RECOMMENDATION
KEEP

## experiment_6275_flagship_asp_constraint_verification_benchmark.formal_sidecar.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact makes no comparative or value claim; such a claim would require candidate outcomes that disagree with the formal answer sets or lose to a serious baseline.

## WAS THAT CHECKED
No. This sidecar records formal programs and their solution sets, but provides no candidate performance, comparator arm, validity flags, success criterion, or headline verdict.

## EVIDENCE
`program_text`, `exact_answer_sets`, `solver_answer_set_count`, `zero_energy_states`, `task_id`

## RECOMMENDATION
KEEP

## experiment_1736_kanele_synth.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6275_flagship_asp_constraint_verification_benchmark.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Energy-guided repair adds semantic-validity value over ordinary one-shot generation across the flagship models and task families.

## WHAT WOULD REFUTE IT
A zero or negative paired semantic-validity delta against the cheapest serious baseline—one-shot generation scored by the same exact solver—or a rival arm tying or outperforming repair would refute added value.

## WAS THAT CHECKED
Yes. The artifact directly pairs `energy_guided_repair` with `one_shot` by model and family. The refutation occurred: every Qwen semantic comparison tied exactly, and several Gemma comparisons also tied; no reported positive interval excludes zero. The value claim is additionally circular because exact ASP solving both guides repair and defines correctness.

## EVIDENCE
`base_arm`: `one_shot`; `repair_arm`: `energy_guided_repair`; `paired_intervals_and_sample_sizes`; `semantic`; `mean_delta`: `0.0`; `ci95`: `0.0`, `0.0`; `sample_size`: `6`; `verifier_is_oracle`; `Discloses exact ASP solving is the correctness oracle.`; `honest_verdict`: `complete_partial: test_exit_codes`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_6751_thermalizer_factor_trajectory_fidelity.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6752_arc_code_carrying_tool_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the narrow readiness receipt, either model observing less than 32768 context, lacking full CUDA offload, failing the live production route, failing typed parsing or dispatch, or exceeding the response bound would refute readiness; there is no comparative, value, generalization, or solve claim to falsify.

## WAS THAT CHECKED
Yes. Both model rows record context, CUDA offload, live-path, parse, dispatch, and bounded-response receipts, and the gate summary checks them per model. The false loop completion does not refute the expressly transport-only claim.

## EVIDENCE
`"title": "Task-owned 32K code-carrying ARC tool preflight"`; `"arc_context_tool_preflight_ready": true`; `"multi_parameter_parse_successes": 2`; `"multi_parameter_dispatch_successes": 2`; `"bounded_response_successes": 2`; `"live_path_reached": true`; `"loop_returned_success": false`; `"solve_claim": false`; `"claim_boundary": "This preflight proves 32K CUDA admission and code-carrying tool transport only. It measures no game quality and claims no level solve."`

## RECOMMENDATION
KEEP

## experiment_6753_object_table_fetch_on_demand_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify. The intended future claim would be refuted by nonpositive prompt-token savings or a change-fidelity confidence-interval lower bound below the negative noninferiority margin, computed from valid completed arm pairs.

## WAS THAT CHECKED
No. Model inference never occurred, every planned row was invalid due to the preflight block, zero pairs entered analysis, and the comparative metrics remained null.

## EVIDENCE
`status` `blocked` `live_model_invoked` `false` `mean_prompt_token_savings` `null` `change_fidelity_ci95` `n_games_paired` `0` `n_seed_pairs` `adoption_gate_passed` `object_table_ab_completed` `complete_blocked_object_table_ab:cuda_device_available`

## RECOMMENDATION
KEEP

## experiment_6754_v588_branch_disposition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V588 produced mixed branch outcomes—blocked handoff and FR12, null continuous self-learning, positive execution activity, simulator-only stochastic evidence, and partial ARC transport—without claiming pooled milestone success.

## WHAT WOULD REFUTE IT
A valid source row contradicting its assigned branch verdict—for example, transactional memory outperforming `no_memory` despite the null CSL verdict, a failed CUDA receipt despite positive activity, or emission of a pooled success claim—would refute the headline.

## WAS THAT CHECKED
Yes. Raw task rows were replayed into `recomputed_headlines`, checked for headline disagreements in `row_headline_mismatches`, and audited by adversarial and row-consistency validators. The design demonstrably allowed failure: handoff and FR12 were blocked, CSL tied its `no_memory` comparator with zero gain, and the ARC A/B remained incomplete.

## EVIDENCE
`complete_partial: V588 preserved blocked handoff, blocked FR12, null CSL, positive activity, simulator-only stochastic evidence, and partial ARC transport without a pooled success claim.`; `pooled_success_claim_emitted`: `false`; `row_headline_mismatches`: `[]`; `verdict_class`: `partial`; `no_memory`: `0.4583333333333333`; `transactional_memory`: `0.4583333333333333`; `mean_delta`: `0.0`; `commits`: `0`; `diagnostic_energy_ready`: `false`; `heldout_reasoning_error_auroc`: `null`; `object_table_ab_completed`: `false`; `hardware_used`: `false`; `simulator_used`: `true`; `bounded_cpu_compiler_fidelity_no_physical_tsu_or_performance_claim`

## RECOMMENDATION
KEEP
