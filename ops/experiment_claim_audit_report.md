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
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_6962_queue_regulated_self_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the intended positive claim would be refuted by paired valid rows showing queue-regulated memory tying or losing to a serious non-queue memory baseline, or failing its utility or safety criteria.

## WAS THAT CHECKED
No. The run failed before producing any arm, outcome, paired-metric, retention, contradiction, or debt rows.

## EVIDENCE
`honest_verdict`: `blocked_queue_regulated_self_learning`; `verdict_class`: `blocked`; `failed_check`: `all_model_arm_workers`; `passed`: `false`; `queue_learning_positive_score`: `0`; `queue_learning_run_complete_score`: `0`; `arm_rows`: `[]`; `paired_metric_rows`: `[]`; `rows`: `[]`.

## RECOMMENDATION
KEEP

## experiment_6963_queue_memory_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made; the artifact is only a blocked-gate receipt.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate` because the sole gate had `actual` `0`, `expected` `1`, and `passed` `false`.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate_check_summary": "1 of 1 gate(s) failed; first failure: exp6962-queue-regulated-self-learning.queue_learning_run_complete_score (actual=0 == expected=1)"`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6964_v609_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is made; treating the administrative status as a claim, it would be contradicted by an incomplete replay or a passing gate with no flagged or conflicting evidence.

## WAS THAT CHECKED
Yes. The capstone separately checked completion and gate validity: replay completed, but the gate failed because tasks were disqualified.

## EVIDENCE
`"inference_substrate": "independent_artifact_replay_and_contract_reconciliation_no_llm"`; `"v609_capstone_complete_score": 1`; `"status": "complete_disqualified"`; `"failed_check": "flagged_or_conflicting_evidence"`; `"passed": false`; `"replay_complete": true`; `"disqualified_task_numbers": [6958, 6964]`; `"certified_energy_positive_score": 0`; `"queue_learning_positive_score": null`

## RECOMMENDATION
KEEP

## experiment_6965_v610_contract_advisory.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6966_gguf_load_envelope_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports a blocked precondition check and makes no positive, comparative, readiness, or value claim to falsify.

## WAS THAT CHECKED
No—the canary never ran; there are no generation, reproduction, checkpoint, runtime, or teardown rows from which such a claim could be tested.

## EVIDENCE
`honest_verdict` `blocked_gguf_load_envelope_canary` `verdict_class` `blocked` `failed_check` `foreign_gpu_compute_processes` `passed` `false` `gguf_load_canary_complete_score` `0` `gguf_runtime_ready_score` `0` `live_duration_s` `0.0`

## RECOMMENDATION
KEEP

## experiment_6967_certified_error_headroom_fixture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6968_arc_post_refit_induction_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked audit rather than a positive or comparative result.

## WAS THAT CHECKED
No. The failed source-immutability precondition stopped evaluation before transition execution, held-out scoring, or control comparison.

## EVIDENCE
`honest_verdict`: `blocked_arc_post_refit_induction_audit`; `verdict_class`: `blocked`; `immutable_transition_source`; `observed_value`: `false`; `engine_execution_rows`: `[]`; `control_rows`: `[]`; `paired_control_delta_rows`: `[]`; `heldout_exact_accuracy`: `null`; `arc_induction_generalization_positive_score`: `0`

## RECOMMENDATION
KEEP

## experiment_6969_error_structured_prompt_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made; therefore, there is no headline claim to falsify.

## WAS THAT CHECKED
No. The experiment was blocked before execution at the conductor pre-gate, so no method, comparator, or outcome was evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_reason": "actual=0 == expected=1"`; `"failed_field": "gguf_runtime_ready_score"`; `"failed_observed": 0`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
