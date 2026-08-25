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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_6583_gemma4_26b_a4b_flagship_source_shard.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6585_v573_terminal_recovery_and_execution_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no scientific, comparative, generalization, or verifier-value claim. Its operational receipt would be contradicted by an unreplayed terminal state, fewer than three distinct hard-limit attempts, an unbounded V573 task, or a created science verdict.

## WAS THAT CHECKED
Yes for the operational receipt: terminal-state, hard-limit-attempt, execution-budget, roadmap-gate, adversarial, and protected-file checks could fail and were recorded. No scientific refutation was applicable because no science claim was made.

## EVIDENCE
`honest_verdict`: `complete: all six V572 terminal states and three Exp6584 hard-limit attempts replay; V573 model tasks are bounded; no science verdict was created`; `verdict_class`: `null`; `science_disposition`: `infrastructure_contract_no_science`; `science_disposition`: `method_contract_no_model_outcome`; `science_disposition`: `runtime_evidence_no_quality_verdict`; `scientific_verdict_created`: `false`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6586_isolated_full_suite_truth_baseline.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6587_v573_constraint_first_method_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No outcome can refute a comparative method claim because none is made; such a claim would require routed CFR to tie or lose to the direct arm on exact success, safety, or total cost.

## WAS THAT CHECKED
No. This is a preregistration/readiness receipt produced before model inference, so it contains no arm outcomes.

## EVIDENCE
`model_inference_invoked`: `false`; `model_outcomes_available`: `false`; `llm_calls_issued`: `0`; `verdict_class`: `null`; `ready_contract_verdict_class`: `null`; `complete: source, fixture, router, metric, and exact-authority contracts are ready before any V573 model outcome`

## RECOMMENDATION
KEEP

## experiment_6588_v574_bounded_cfr_launch_root.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports launch-readiness infrastructure and explicitly disclaims any science or comparative result.

## WAS THAT CHECKED
No; no method-value, generalization, or comparative hypothesis was tested.

## EVIDENCE
`"no science result was created"`; `"v573_terminal_and_contract_replay_no_llm"`; `"llm_calls_issued": 0`; `"model_loads_issued": 0`; `"science_result_created": false`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_6589_isolated_pytest_receipt_remediation.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6590_qwen36_constraint_first_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or CFR-benefit claim to falsify; the narrower completeness assertion would be refuted by any frozen unit, arm, raw stage, exact check, checkpoint, cost, failure, model, or GPU receipt being absent or incomplete.

## WAS THAT CHECKED
Yes. Completeness is represented across `per_unit_rows`, `exact_checker_receipts`, `checkpoint_receipts`, `failure_rows`, `raw_stage_receipts`, and `gpu_process_receipts`; the mechanical pre-pass also reports every required category complete.

## EVIDENCE
`honest_verdict`; `complete: every frozen Qwen CFR unit, arm, raw stage, exact check, checkpoint, cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made`; `A complete source stream is null evidence infrastructure.`; `Exact checks define row validity, so later exact wins are circular-positive.`

## RECOMMENDATION
KEEP

## experiment_6591_gemma4_31b_constraint_first_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative CFR claim exists to falsify; an incomplete unit, arm, raw stage, check, checkpoint, cost, failure, model, or GPU receipt would instead refute only the artifact’s completeness assertion.

## WAS THAT CHECKED
No comparative benefit was checked or claimed; the artifact checks execution and receipt completeness.

## EVIDENCE
`honest_verdict`: `complete: every frozen Gemma CFR unit, arm, raw stage, exact check, checkpoint, cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made`

## RECOMMENDATION
KEEP
