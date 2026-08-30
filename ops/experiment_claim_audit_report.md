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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6770_dccd_environment_grammar_ab_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify; a future value claim would be refuted if `dccd_environment` tied or lost to `repaired_direct` or `static_grammar` on completed, valid rows.

## WAS THAT CHECKED
No. Execution stopped at the failed `one_model_vram` precondition, before model inference or row collection.

## EVIDENCE
`"verdict_class": "blocked"`, `"live_model_invoked": false`, `"models_used": []`, `"rows": []`, `"proof_transport_ab_completed": false`, `"failed_check": "one_model_vram"`, `"rows_complete": false`

## RECOMMENDATION
KEEP

## experiment_6771_proof_transport_localization_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite gate and makes no substantive audit or comparative claim to falsify.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate` because the sole prerequisite gate failed.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `proof_transport_ab_completed`; `failed_expected`: `true`; `failed_observed`: `false`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6773_csl_owned_lease_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify. For the narrower operational record, an eligible or selected GPU, live model invocation, or populated execution receipts would contradict the reported blocked status.

## WAS THAT CHECKED
Yes. The admission gate recorded zero eligible devices, no selected device, no live invocation, and no execution receipts.

## EVIDENCE
`status`: `blocked`; `verdict_class`: `blocked`; `eligible_device_count`: `0`; `selected_device`: `null`; `live_model_invoked`: `false`; `models_used`: `[]`; `gpu_receipts`: `[]`; `lease_receipts`: `[]`; `teardown_receipts`: `[]`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6776_arc_shadow_supervisor_accrual.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No efficacy claim exists to falsify; any implied claim of successful evidence accrual would be refuted by zero live inference, zero observed actions, unmet evidence floors, and failed transport readiness.

## WAS THAT CHECKED
No efficacy test occurred. The artifact checked only preconditions and recorded the blocked run; no live supervisor behavior, invariance pair, or comparative control was evaluated.

## EVIDENCE
`"status": "complete_blocked_shadow_supervisor_accrual"`, `"models_used": []`, `"live_model_invoked": false`, `"actions_observed": 0`, `"method": "not run because preconditions failed"`, `"passed": false`, `"evidence_floor_met_by_arm"`, `"shadow_supervisor_transport_ready": false`, `"solve_claim": false`, `"verdict_class": "blocked"`, `"honest_verdict": "complete_blocked_shadow_supervisor_accrual:exclusive_gpu_without_unrelated_compute"`

## RECOMMENDATION
KEEP

## experiment_6777_arc_tool_gap_transport.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, not a comparative or value claim.

## WAS THAT CHECKED
No. The experiment stopped at the upstream gate before the method under test was executed or evaluated.

## EVIDENCE
`status` is `blocked`; `honest_verdict` is `blocked_gate_check_failed`; `failed_field` is `shadow_supervisor_transport_ready`; `failed_expected` is `true`; `failed_observed` is `false`; `blocked_at_layer` is `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_6780_v590_branch_disposition.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6782_sequential_sota_runtime_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made. Treating the operational status as an assertion, one valid owned-CUDA admission receipt for any listed model would contradict “no model completed owned CUDA admission.”

## WAS THAT CHECKED
No comparative claim required checking. Operationally, admission was attempted but blocked at the lease stage before any model invocation, so successful admission could not occur in this run.

## EVIDENCE
`honest_verdict` `complete_blocked_sequential_sota_runtime: no model completed owned CUDA admission.` `verdict_class` `blocked` `failed_check` `runtime_admission:unsloth/Qwen3.6-35B-A3B-GGUF` `observed` `lease_wait_deadline_expired` `live_model_invoked` `false` `models_used` `[]` `gpu_receipts` `[]` `rows` `[]`

## RECOMMENDATION
KEEP

## experiment_6783_owned_proof_generation_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked pre-gate rather than an A/B result.

## WAS THAT CHECKED
No; the three-model proof-generation A/B was not executed. Only the upstream readiness gate was checked.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "all_mandated_runtime_ready"`, `"failed_expected": true`, `"failed_observed": false`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
