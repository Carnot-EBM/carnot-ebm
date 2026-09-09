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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_7150_v628_grounding_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A future claim that `relational_sql` adds detection value would be refuted if the `direct` or `self_check` arm tied or beat it on valid scored rows.

## WAS THAT CHECKED
No. The artifact freezes comparison opportunities but contains no per-game outcomes or comparative scores.

## EVIDENCE
`"honest_verdict"`: `"blocked_real_qwen_canary"`; `"grounding_preflight_ready_score"`: `0`; `"inference_substrate_class"`: `"blocked_no_run"`; `"per_game_results"`: `[]`; `"One reports execution readiness and no detection value."`; `"The empty list states that this factual preflight has no game result."`

## RECOMMENDATION
KEEP

## experiment_7151_v629_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7152_v629_source_delta.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V629 source-and-cache refresh completed and found no decision-changing method absent from the existing planner.

## WHAT WOULD REFUTE IT
A mapped decision-changing method with `already_in_v629_planner` false or `execution_delta_new` true, or a failed source/cache/map/append gate, would refute the claim.

## WAS THAT CHECKED
Yes. The decision-changing mappings were checked in `task_method_map_rows`, and overall consistency was checked in `gate_check_summary`. The unavailable source requests limit discovery coverage but are explicitly recorded and do not contradict the bounded null claim.

## EVIDENCE
`decision_changing`: `true`; `already_in_v629_planner`: `true`; `execution_delta_new`: `false`; `decision_changing_method_count`: `0`; `failed_check`: `null`; `expected_value`: `1`; `observed_value`: `1`; `passed`: `true`; `verifier_is_oracle`: `false`; `verdict_class`: `null`; `honest_verdict`: `null_v629_source_delta_complete_no_post_planner_change`

## RECOMMENDATION
KEEP

## experiment_7153_v629_grounding_runtime.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7154_v629_qwen_dual_side_grounding.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7156_v630_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V630 task contract is disqualified because the independently parsed Markdown and YAML contracts do not match.

## WHAT WOULD REFUTE IT
Both sources yielding the same 13 tasks in the required order, with every per-task contract comparison passing and a nonzero conformance score.

## WAS THAT CHECKED
Yes. Independent Markdown and YAML parses were compared by count, order, identity, and contract fields; mismatches occurred.

## EVIDENCE
`honest_verdict`: `complete_disqualified_v630_markdown_yaml_contract_mismatch`; `verdict_class`: `disqualified`; `failed_check`: `markdown_task_count`; `expected_value`: `13`; `observed_value`: `14`; `passed`: `false`; `v630_task_contract_conforms_score`: `0`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_7157_v630_qwen38_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact makes no quality or comparative claim; its operational blocked-state premise would be refuted by observing at least one idle RTX 3090.

## WAS THAT CHECKED
Yes. The `idle_rtx_3090` gate checked for a minimum count of one and observed zero.

## EVIDENCE
`honest_verdict`: `blocked_idle_rtx_3090`; `status`: `blocked`; `verdict_class`: `blocked`; `inference_substrate`: `no_inference`; `inference_substrate_class`: `blocked_no_run`; `generation_receipts`: `[]`; `model_load_receipts`: `[]`; `expected_value`: `{"minimum_count": 1}`; `observed_value`: `{"count": 0, "indices": []}`; `passed`: `false`

## RECOMMENDATION
KEEP

## experiment_7158_v630_entity_evidence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The counterfactual fixture is ready; no verifier-value or performance claim is made.

## WHAT WOULD REFUTE IT
A failed fixture-integrity gate, incorrect term mutation, evaluation-truth leakage, or incomplete build would refute fixture readiness.

## WAS THAT CHECKED
Yes. The artifact reports the aggregate integrity gate, term-level mutation checks, sealed-field checks, split freezing, and evaluation-truth-access status. The displayed checks pass, although many rows are elided.

## EVIDENCE
`counterfactual_fixture_ready_score`: `1`; `honest_verdict`: `complete_positive_counterfactual_fixture_ready_no_verifier_value_claim`; `expected_value`: `all_fixture_integrity_checks_pass`; `observed_value`: `all_fixture_integrity_checks_pass`; `passed`: `true`; `evaluation_truth_accessed`: `false`; `frozen_before_threshold_fitting`: `true`; `status`: `complete`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
