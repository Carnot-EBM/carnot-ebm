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
| NO_CLAIM | 7 |

## experiment_6760_prefix_backtracking_repair_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact makes no comparative claim about prefix-backtracking repair.

## WAS THAT CHECKED
No. The experiment was blocked at `conductor_pre_gate` before any A/B data was produced.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"duration_s": 0.0`; `"failed_observed": null`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6761_procedural_memory_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify; only the procedural-readiness receipt could fail, through any false readiness gate, unequal capacity, chronology leakage, insufficient accept/reject opportunities, restart mismatch, rollback failure, or mishandled poison fixture.

## WAS THAT CHECKED
Yes. Those conditions are covered by `gate_check_summary`, per-order opportunity counts, capacity fields, transaction rows, and restart, rollback, and poison receipts.

## EVIDENCE
`complete_procedural_memory_stream_ready`; `procedural_memory_stream_ready`; `true`; `failed_checks`; `[]`; `verdict_class`; `circular_positive`; `A closed class prevents fixture readiness from becoming a science claim.`; `verifier_is_oracle`; `false`

## RECOMMENDATION
KEEP

## experiment_6762_procedural_vs_trace_csl_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify; a future claim that procedural memory outperforms rivals would be refuted if the procedural arm tied or lost to detailed-trajectory or no-memory controls on valid rows.

## WAS THAT CHECKED
No. Execution stopped at failed preconditions, so no comparative rows or arm outcomes were produced.

## EVIDENCE
`honest_verdict`: `complete_blocked_procedural_csl_ab: owned precondition failed`; `live_model_invoked`: `false`; `prospective_csl_completed`: `false`; `rows`: `[]`; `failed_checks`: `one_model_vram`, `task_owned_lease`.

## RECOMMENDATION
KEEP

## experiment_6763_csl_hard_case_forgetting_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental outcome claim to falsify. The receipt’s implicit gate assertion would be refuted if the prerequisite were observed as true or the gate passed.

## WAS THAT CHECKED
Yes, in `gates_evaluated`; the sole prerequisite was explicitly compared with its expected value. No hard-case, forgetting, or poison audit was executed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_expected`: `true`; `failed_observed`: `false`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6764_arc_exclusive_load_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or quality proposition is asserted. The limited operational receipt would fail if either model lacked a complete CUDA load, 32K runtime context, first token, successful selfparse dispatch, teardown, lease release, VRAM recovery, or if an unrelated process were signaled.

## WAS THAT CHECKED
Yes, for both model admissions in `gpu_receipts` and `gate_check_summary`; this was an operational readiness check, not a method-value experiment.

## EVIDENCE
`claim_boundary` `Transport and teardown admission only. It measures no ARC quality, claims no solve, and keeps model timings unpooled.` `runtime_context` `32768` `first_token_observed` `true` `success` `true` `released` `true` `passed` `true` `unrelated_processes_signaled` `[]` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_6765_object_table_fetch_ab_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify; live paired rows showing fetch-on-demand inferior, superior, or tied would merely answer the planned question.

## WAS THAT CHECKED
No. The model was never invoked, all displayed rows stopped at preflight, and the comparative metrics were not computed.

## EVIDENCE
`status` is `blocked`; `live_model_invoked` is `false`; `stop_reason` is `preflight_blocked`; `mean_prompt_token_savings`, `change_fidelity_delta`, and `change_fidelity_interval` are `null`; `object_table_ab_completed` and `adoption_gate_passed` are `false`; `honest_verdict` is `complete_blocked_object_table_ab_v2`.

## RECOMMENDATION
KEEP

## experiment_6766_thermalizer_independent_trajectory_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
An independent simulator evaluator reproduced a trajectory-TV reduction for context matching, while trajectory refinement received no non-circular value claim.

## WHAT WOULD REFUTE IT
Context matching tying or losing to the independent-factor baseline—shown by a nonpositive paired TV difference or a confidence interval including zero—or the direct sampler materially disagreeing with the exact evaluator would refute the claim.

## WAS THAT CHECKED
Yes. Paired comparisons against `independent_factor` include ties and worsened rows, so failure was possible; the reported intervals exclude zero at the shown depths. A 192-row direct-sampler cross-check also tested the exact evaluator. Circularity was identified specifically for `trajectory_refinement`, not credited as evidence of value.

## EVIDENCE
`honest_verdict`: `complete: the independent evaluator reproduced the context-matching trajectory reduction; trajectory refinement remains exact-objective circular; simulator only`

`method`: `context_matched`

`mean_independent_minus_method_tv`: `0.057389814199323`, `0.0659517153362909`, `0.0537530735194788`

`interval_excludes_zero`: `true`

`worsened_pair_count`: `2`

`methods_consuming_exact_evaluator_outcome`: `trajectory_refinement`

`verifier_is_oracle`: `true`

`exact_in_ci99_count`: `190`

`observed_row_count`: `192`

`passed`: `true`

`claim_boundary`: `simulator-only; no speed, power, X0, Z1, FPGA, or physical-hardware claim`

## RECOMMENDATION
KEEP

## experiment_6767_v589_branch_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a synthesis receipt, it would be contradicted by its own rows showing completed repair, prospective continuous-memory results, live ARC quality measurements, non-oracle portability evidence, or an emitted pooled-success claim while the headline reports those outcomes as blocked, circular, or absent.

## WAS THAT CHECKED
Yes. The branch summaries and recomputed headlines check each disposition, while the artifact reports no row/headline mismatches.

## EVIDENCE
`"The capstone is a synthesis receipt. It is not a verifier or a pooled science claim."`; `"complete_partial: V589 preserved narrowed proof transport, blocked repair, blocked continuous memory, blocked ARC quality, circular simulator portability, and no pooled success claim."`; `"repair_completed": false`; `"prospective_rows": 0`; `"live_quality_rows"` with `"numerator": 0`; `"simulator_only": true`; `"verifier_is_oracle": true`; `"pooled_success_claim_emitted": false`; `"row_headline_mismatches": []`

## RECOMMENDATION
KEEP
