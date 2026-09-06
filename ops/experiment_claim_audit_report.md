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

## experiment_7064_v619_exact_entrance_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. The narrower fixture-readiness assertion would be refuted by an omitted legal entrance, an invalid witness, model-visible label leakage, a post-label split, or any failed readiness gate.

## WAS THAT CHECKED
Yes. The artifact checks enumeration completeness, witness replay, leakage, split sealing, hashes, and readiness gates. It does not test the verifier’s added value, but it does not claim to do so.

## EVIDENCE
The artifact identifies itself as `circular_positive`, records `verifier_is_oracle` as `true`, uses the `deterministic_verifier`, reports `model_invocation_count` as `0`, and states `all_legal_entrances_labeled` as `true`, `failed_checks` as `[]`, and `passed` as `true`.

## RECOMMENDATION
KEEP

## experiment_7065_v619_three_family_entrance_bank.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify; treating the blocked status as an operational assertion, it would be contradicted by successful GPU-lease preconditions followed by completed model inference and populated proposal results.

## WAS THAT CHECKED
Yes. The precondition gate explicitly failed at GPU-lease availability, and the downstream inference/result collections are empty.

## EVIDENCE
`"honest_verdict": "blocked_v619_three_family_entrance_bank_precondition_failed"`; `"failed_check": "owned_gpu_leases_available"`; `"passed": false`; `"all_passed": false`; `"entrance_proposal_bank_complete_score": 0`; `"models_used": []`; `"proposal_rows": []`; `"forced_prefix_rows": []`; `"per_game_results": []`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_7066_entrance_bank_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and reports no experimental or comparative result.

## WAS THAT CHECKED
No; the experiment did not run because the sole upstream gate failed.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"actual": 0`, `"expected": 1`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7069_v619_context_authorization_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No utility claim is made; a future utility claim would be refuted by the authorization method tying or losing to a serious context-matched baseline on sealed outcomes.

## WAS THAT CHECKED
No. The artifact checks contract readiness, leakage controls, mutations, transitions, and rollback behavior—not comparative future utility.

## EVIDENCE
`honest_verdict` = `complete_null_context_authorization_contract_ready_no_future_utility_claim`; `verdict_class` = `null`; `context_authorization_contract_ready_score` = `1`; `verifier_is_oracle` = `false`; `failed_checks` = `[]`.

## RECOMMENDATION
KEEP

## experiment_7070_v619_bcit_self_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no affirmative value claim to falsify. The blocked conclusion would be refuted by a sufficient frozen stream—at least 120 chronological events across at least 12 source groups—followed by completed comparative-arm results.

## WAS THAT CHECKED
Yes. The precondition gate checked event and source-group counts, failed both thresholds, and stopped before constructing arms or comparative rows.

## EVIDENCE
`honest_verdict`: `blocked_insufficient_frozen_bcit_stream`; `verdict_class`: `blocked`; `failed_check`: `minimum_chronological_event_count`; `expected_value`: `>=120`; `observed_value`: `8`; `failed_check`: `minimum_source_group_count`; `expected_value`: `>=12`; `observed_value`: `5`; `arm_definitions`: `[]`; `chronological_event_rows`: `[]`; `rows`: `[]`; `bcit_comparison_complete_score`: `0`; `bcit_self_learning_value_score`: `0`

## RECOMMENDATION
KEEP

## experiment_7071_bcit_drift_rollback_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and makes no substantive drift, rollback, or comparative claim.

## WAS THAT CHECKED
No; execution stopped at the upstream gate before the audit ran.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "bcit_comparison_complete_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7072_v619_live_arc_compaction_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No claim exists to falsify; a future compaction-value claim would be refuted by qualified paired results showing treatment ties or worsens context usage, quality, or operational performance versus control.

## WAS THAT CHECKED
No. The experiment stopped before eligible units, arm execution, or paired outcome measurement; no comparative data was produced.

## EVIDENCE
`"rows": []`, `"per_game_results": []`, `"qwen_pair_count": 0`, `"gemma_pair_count": 0`, `"compaction_value_ready_score": 0`, `"retirement_decision": "no_decision_precondition_blocked"`, `"failed_check": "eligible_hidden_or_rotation_units"`, `"observed_value": 0`, `"verdict_class": "blocked"`, `"honest_verdict": "blocked_live_arc_compaction_ab:eligible_hidden_or_rotation_units"`

## RECOMMENDATION
KEEP

## experiment_7075_v619_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V619 capstone reached a null result because all four branches terminated as resource-blocked rather than producing claim-grade positive evidence.

## WHAT WOULD REFUTE IT
A valid required upstream task passing its gate while the corresponding branch remained classified as resource-blocked would falsify the terminal-disposition claim.

## WAS THAT CHECKED
Yes. The artifact records discovery, validity, gate recomputation, and branch decisions; gates could pass or fail, and the displayed failed gates and missing or blocked tasks support the null disposition.

## EVIDENCE
`"honest_verdict": "complete_null_v619_terminal_branch_dispositions"`; `"verdict_class": "null"`; `"entrance_branch_decision": "blocked_resource"`; `"self_learning_branch_decision": "blocked_resource"`; `"arc_compaction_branch_decision": "blocked_resource"`; `"ising_branch_decision": "blocked_resource"`; `"observed_value": 0`; `"passed": false`; `"observed_value": null`; `"A required task is missing or resource-blocked."`; `"default_off": true`

## RECOMMENDATION
KEEP
