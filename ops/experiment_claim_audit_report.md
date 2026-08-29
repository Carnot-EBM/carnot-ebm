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
| NO_CLAIM | 8 |

## experiment_6565_v569_evidence_and_retirement_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a contract/receipt audit, not a comparative value or generalization claim.

## WAS THAT CHECKED
No—there is no method-value claim to falsify; the artifact checks only contract consistency, provenance, gates, and retirement boundaries.

## EVIDENCE
`verdict_class`: `null`; `verifier_is_oracle`: `true`; `inference_substrate`: `immutable_v568_artifact_gate_failure_and_retirement_audit_no_llm`; `receipt_only_until_new_prospective_rows`: `true`; `no_game_or_level_solve_claim`: `true`; `no_llm_load_performed`: `true`

## RECOMMENDATION
KEEP

## experiment_6571_v570_evidence_gate_and_retirement_root.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive method or value claim is made; the limited receipt claim would be falsified by failed gate checks, incorrect eligibility classifications, changed protected files, or any assertion that extraction, generation, or graph-Potts utility ran.

## WAS THAT CHECKED
No substantive refutation was applicable because those methods were not run; only the administrative gate and receipt conditions were checked through per-unit rows and explicit failure fields.

## EVIDENCE
`verdict_class`: `null`; `source_extraction_claim_count`: `0`; `graph_potts_utility_claim_count`: `0`; `graph_potts_utility_claimed_by_exp6571`: `false`; `generation_ran`: `false`; `llm_load_performed`: `false`; `hardware_command_performed`: `false`; `inference_substrate`: `immutable_v569_artifact_gate_failure_and_retirement_audit_no_llm`

## RECOMMENDATION
KEEP

## experiment_6716_object_table_fetch_on_demand_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a precondition receipt, not a comparative result. Evidence that inference ran or that either A/B arm produced a score would contradict its factual blocked-run status.

## WAS THAT CHECKED
Yes. The run-status, inference-substrate, honest-verdict, and arm-result fields all record that inference never began and no comparison was produced.

## EVIDENCE
`"run_completed": false`, `"inference_substrate": "precondition_only_no_inference"`, `"verdict_class": "blocked"`, `"blocked_context_window: CARNOT_ARC_INDUCE_N_CTX is unset; the paired A/B did not start"`, `"table_on": null`, `"fetch_on_demand": null`, `"paired_delta_fetch_minus_on": null`

## RECOMMENDATION
KEEP

## experiment_6729_v587_activation_evidence_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim is made; the operational blocked status would be refuted if every required contract check passed and the activation-readiness field were true.

## WAS THAT CHECKED
Yes, in `gate_check_summary`, `model_policy_rows`, and `task_contract_rows`; failed checks produced a false readiness result and blocked verdict.

## EVIDENCE
`inference_substrate` is `source_receipts_and_local_method_preregistration_no_llm`; `v587_contract_ready` is `false`; `verdict_class` is `blocked`; `status` is `blocked_v587_activation_contract`.

## RECOMMENDATION
KEEP

## experiment_6730_arc_context_tool_preflight.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive result claim to falsify; the artifact is only a blocked-preflight receipt. A successful gate observation would contradict its procedural report that the preflight was blocked.

## WAS THAT CHECKED
Yes, for the procedural gate status only: the sole entry in `gates_evaluated` records the expected and observed Boolean values and marks the gate as failed. No method, outcome, or comparator was evaluated.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"failed_field"`: `"v587_contract_ready"`; `"failed_expected"`: `true`; `"failed_observed"`: `false`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6732_object_table_ab_cold_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports a blocked prerequisite gate and no experimental outcome or comparative claim.

## WAS THAT CHECKED
No. The experiment did not proceed beyond the prerequisite gate; no method, rival, success criterion, or result rows were evaluated.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `passed`: `false`; `actual`: `null`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6733_hardness_controlled_certificate_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative claim to falsify. The artifact’s gate-failure assertion would be refuted by the gate having passed or the observed value being true.

## WAS THAT CHECKED
Yes. The sole evaluated gate records an observed false value, a failed comparison against expected true, and a false pass status.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_observed": false`, `"failed_expected": true`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6735_oracle_distinct_diagnostic_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact asserts no experimental outcome or comparative result to falsify.

## WAS THAT CHECKED
No. The experiment stopped at the prerequisite gate before producing scored rows or results.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_observed": null`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
