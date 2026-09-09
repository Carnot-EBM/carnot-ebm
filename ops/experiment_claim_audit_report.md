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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7147_v627_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V627 evidence matrix is structurally complete, but scientific completion is blocked by unavailable upstream terminal evidence.

## WHAT WOULD REFUTE IT
Either incomplete matrix-slot coverage would refute structural completeness, or complete upstream availability—with no missing artifacts or external-state blocks—would refute the blocked status.

## WAS THAT CHECKED
Yes. Matrix coverage was reported directly, while the upstream-availability gate compared expected and observed missing artifacts, unreadable artifacts, and external-state blocks.

## EVIDENCE
`"matrix_slot_coverage_rate": 1.0`; `"present_artifact_rate": 0.8181818181818182`; `"scientific_branch_promotion_rate": 0.0`; `"failed_check": "upstream_terminal_availability"`; `"passed": false`; `"missing_artifacts": [7140, 7143]`; `"external_state_blocks": ["gatemate_operator_receipt"]`; `"honest_verdict": "blocked_upstream_terminal_availability_v627_matrix_complete"`; `"v627_capstone_complete_score": 1`; `"status": "blocked"`; `"One means the evidence matrix is structurally complete, not scientifically promoted."`

## RECOMMENDATION
KEEP

## experiment_7148_v628_contract_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim exists to falsify. Treating the blocking status as an operational assertion, a readable, nonempty YAML mapping at the required path would refute it.

## WAS THAT CHECKED
Yes. The prerequisite check directly attempted to read the YAML source and recorded failure; no substantive contract run occurred.

## EVIDENCE
`honest_verdict`: `blocked_v628_contract_preflight_prerequisite_missing`; `inference_substrate_class`: `blocked_no_run`; `failed_check`: `v628_yaml_readable`; `passed`: `false`; `available`: `false`; `research-roadmap-next.yaml`: `null`; `rows`: `[]`

## RECOMMENDATION
KEEP

## experiment_7149_v628_source_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked no-run rather than a comparative or value claim.

## WAS THAT CHECKED
No; no experiment ran and no rows were produced.

## EVIDENCE
`"inference_substrate_class": "blocked_no_run"`, `"duration_s": 0.0`, `"rows": []`, `"passed": false`, `"failed_check": "preconditions_not_checked"`, `"verdict_class": "blocked"`, `"honest_verdict": "blocked_v628_source_delta_precondition"`

## RECOMMENDATION
KEEP

## experiment_7150_v628_grounding_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable. A future value claim for the `relational_sql` method would be refuted if the cheaper `direct` or `self_check` arm tied or beat it on valid, sealed-label rows.

## WAS THAT CHECKED
No. The arms were scheduled, but no comparative results were produced.

## EVIDENCE
`per_game_results`: `[]`; `grounding_preflight_ready_score`: `0`; `honest_verdict`: `blocked_real_qwen_canary`; `inference_substrate_class`: `blocked_no_run`; `verdict_class`: `blocked`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7151_v629_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7152_v629_source_delta.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed V629 source-and-cache delta found no decision-changing method or execution delta beyond the existing planner.

## WHAT WOULD REFUTE IT
A decision-changing mapped method with `already_in_v629_planner` equal to false or `execution_delta_new` equal to true, or a failed completeness gate, would refute the claim.

## WAS THAT CHECKED
Yes. The task-method map records whether each decision-changing method was already represented and whether it introduced a new execution delta; discovery counts and the final completeness gate also could have reported a non-null result or failure.

## EVIDENCE
`decision_changing`: `true`; `already_in_v629_planner`: `true`; `execution_delta_new`: `false`; `decision_changing_method_count`: `0`; `decision_changing_repository_count`: `0`; `v629_source_delta_complete_score`: `1`; `failed_check`: `null`; `passed`: `true`; `verifier_is_oracle`: `false`; `verdict_class`: `null`; `honest_verdict`: `null_v629_source_delta_complete_no_post_planner_change`

## RECOMMENDATION
KEEP

## experiment_7153_v629_grounding_runtime.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7154_v629_qwen_dual_side_grounding.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports no completed comparative or value result to falsify.

## WAS THAT CHECKED
No. The experiment did not run to scored rows, so no method-versus-rival comparison was possible.

## EVIDENCE
`verdict_class`: `partial`; `status`: `preconditions_passed`; `inference_substrate_class`: `blocked_no_run`; `qwen_dual_side_pilot_complete_score`: `0`; `duration_s`: `0.0`; `rows`: `[]`; `arm_metric_rows`: `[]`; `paired_comparison_rows`: `[]`

## RECOMMENDATION
KEEP
