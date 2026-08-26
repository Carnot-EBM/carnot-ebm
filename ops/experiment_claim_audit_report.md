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

## experiment_6615_v576_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The available evidence supports blocked decoding, null live projection, software-only partial sampler evidence, conformant lifecycle behavior without utility, and zero prospective self-learning benefit.

## WHAT WOULD REFUTE IT
A positive `held_future_benefit_over_static` for governed online memory on the prospective held-future pairs would refute the headline’s zero-benefit claim.

## WAS THAT CHECKED
Yes. `continuous_learning_replay` compares four arms across 36 held-future pairs, including the serious `static_projector` and `no_learning` baselines; it reports exactly zero benefit. Predictions preceded observations, and the disqualified source verdict was retained rather than promoted. The oracle was used for adjudication, but no positive verifier-value claim was made.

## EVIDENCE
`held_future_benefit_over_static` `0.0`; `held_future_benefit_over_shuffled` `0.0`; `held_future_pairs` `36`; `governed_online_memory` `88`; `no_learning` `88`; `static_projector` `88`; `all_predictions_before_observations` `true`; `scientific_verdict_class` `null`; `selected_minus_no_projection_mean_error` `0.0`; `selected_minus_random_mean_error` `0.0`; `oracle_defined_win_promoted_to_positive` `false`; `verifier_is_oracle` `true`; `verdict_class` `partial`

## RECOMMENDATION
KEEP

## experiment_6616_v577_execution_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is made; treating the blocked contract status as the operational assertion, it would be refuted if all 13 expected YAML task contracts were present and the contract checks passed.

## WAS THAT CHECKED
Yes. The exact task-set and model-policy checks could have passed, but recorded missing tasks and failed checks.

## EVIDENCE
`honest_verdict` `blocked_roadmap_contract_incomplete: expected Exp6616-Exp6628 YAML contracts are not all present` `execution_contract_ready_score` `0.0` `all_passed` `false` `expected_task_count` `13` `yaml_task_count` `3` `verdict_class` `blocked` `verifier_is_oracle` `true` `The verdict reports contract readiness without claiming scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_6617_gpu_lease_phase_receipts.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact is a blocked-gate receipt and makes no substantive or comparative claim about GPU lease or task-phase performance.

## WAS THAT CHECKED
No. The experiment did not run; it stopped at `conductor_pre_gate` after its sole readiness gate failed.

## EVIDENCE
`"schema"`: `"blocked_gate_check_v1"`; `"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"duration_s"`: `0.0`; `"gate_check_summary"`: `"1 of 1 gate(s) failed; first failure: exp6616-v577-execution-contract.execution_contract_ready_score (actual=0.0 == expected=1.0)"`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6619_v578_activation_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the operational blocked status, both YAML sources containing the complete Exp6619–Exp6632 contract with matching tasks, gates, models, hardware, and deliverables would refute it.

## WAS THAT CHECKED
Yes. The document-to-YAML checks explicitly compared the expected contract against both active and pre-staged sources and found four missing tasks.

## EVIDENCE
`activation_contract_ready_score`: `0.0`; `all_passed`: `false`; `active_task_count`: `10`; `expected_task_count`: `14`; `missing_task_ids`: `["exp6629-live-memory-actionability", "exp6630-error-independent-memory-patch-gate", "exp6631-prospective-support-preserving-csl", "exp6632-v578-independent-capstone"]`; `honest_verdict`: `blocked_v578_activation_contract_incomplete: document promises Exp6619-Exp6632 but both YAML sources contain only Exp6619-Exp6628`; `The verdict reports contract readiness without claiming scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_6620_gpu_lease_phase_receipts.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifier applies because this is a blocked-gate receipt, not a result claiming implementation success or comparative value.

## WAS THAT CHECKED
No; the method was never evaluated because execution stopped at the upstream gate. The prerequisite itself was checked and failed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_observed`: `0.0`; `failed_expected`: `1.0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6622_qwen36_direct_headroom.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing can falsify a performance or direct-headroom claim because no result was produced or asserted.

## WAS THAT CHECKED
No; execution stopped at the pre-gate because the required upstream artifact was unavailable.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"actual": null`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6624_delayed_two_level_decoding.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports a blocked gate check and makes no empirical or comparative claim.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate`, before any method outputs, comparator results, or scored rows were produced.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"duration_s": 0.0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6627_spectral_integrity_repair.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because this is a blocked-gate receipt and reports no result about sampler integrity or repair.

## WAS THAT CHECKED
No; execution stopped at the upstream gate before the experiment ran.

## EVIDENCE
`status` is `blocked`; `honest_verdict` is `blocked_gate_check_failed`; `passed` is `false`; `blocked_at_layer` is `conductor_pre_gate`.

## RECOMMENDATION
KEEP
