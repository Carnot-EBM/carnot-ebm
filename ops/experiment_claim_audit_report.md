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
| NO_CLAIM | 5 |
| CANNOT_DETERMINE | 1 |

## experiment_6553_prospective_sota_continuous_self_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact’s operational blocked status would be refuted by passing the GPU VRAM precondition, loading all mandated models, performing fresh inference, and producing the planned evaluation rows; there is no comparative learning claim to falsify.

## WAS THAT CHECKED
Yes. The precondition and execution receipts checked those conditions, but execution stopped before any efficacy comparison occurred.

## EVIDENCE
`"honest_verdict": "blocked: current, retention, and future-support not evaluated; safety not promoted; receipts failed gpu_vram_contract"`, `"status": "blocked_prospective_csl_preconditions"`, `"gpu_vram_contract": false`, `"all_mandated_models_loaded": false`, `"fresh_local_inference_performed": false`, `"generated_token_invocation_count": 0`, `"per_unit_rows": []`, `"observed_rows": 0`

## RECOMMENDATION
KEEP

## experiment_6554_continuous_self_learning_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific observation can refute a claim the artifact expressly declines to make; the blocked disposition would instead be contradicted by complete live evidence and the expected 756 unit rows.

## WAS THAT CHECKED
No scientific comparison was checked because the required evidence was absent; the artifact checked those prerequisites and stopped blocked.

## EVIDENCE
`"continuous_self_learning_audited_ready_score": 0.0`, `"scientific_disposition_from_rows": "missing_live_evidence"`, `"verdict_class_from_rows": "blocked"`, `"row_count": 0`, `"expected_row_count": 756`, `"missing_input_block": true`, `"scientific_null_claim_allowed": false`, `"terminal_disposition": "blocked"`, `"per_unit_rows": []`

## RECOMMENDATION
KEEP

## experiment_6555_proof_preserving_constraint_saturation_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A failed proof, semantic drift in an equivalent variant, undeclared narrowing in a hardened variant, split leakage, checker round-trip failure, or fixture hash/count mismatch would refute the fixture-readiness receipt.

## WAS THAT CHECKED
Yes. The proof ledger, attack matrix, split contract, exact-checker contract, and aggregate recomputation report those construction checks; no model, method-value, generalization, or comparator claim is made.

## EVIDENCE
`"status": "complete_proof_preserving_constraint_saturation_fixture"`; `"verifier_is_oracle": true`; `"inference_substrate": "primary_source_sota_ingestion_and_proof_preserving_z3_fixture_no_llm"`; `"equivalence_or_hardening_proofs_pass": true`; `"exact_checkers_roundtrip": true`; `"attack_matrix_passed": true`; `"split_floors_hold": true`; `"failure_rows": []`; `"verdict_class": null`; `"A later Exp6556 row must show the all-clause phase curve by count without aggregate-only success."`

## RECOMMENDATION
KEEP

## experiment_6556_sota_constraint_saturation_intervention_ab.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['. Both ', ' have ', '. The ', '. Displayed phase-curve rows give both arms an ', ', while displayed ledger rows mark '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The bounded constraint-saturation intervention provides exact-joint benefit beyond longer-flat generation without harmful regressions after charging costs.

## WHAT WOULD REFUTE IT
The unconditional exact-tool control tying the combined bounded route would refute any claim that saturation-aware routing or decomposition adds value; it would show only that exact solving beats model-only generation.

## WAS THAT CHECKED
No—not as a falsification criterion. The exact-tool arm was present, but the adoption rule allowed either it or the combined route to trigger a positive verdict and never required the combined route to beat that control. Moreover, routing begins at constraint load one, leaving no below-threshold cases where the combined route could distinguish itself.

## EVIDENCE
The `adoption_rule` is `combined_bounded_route_or_exact_tool_cost_guard must beat flat and longer_flat on exact joint success or charged cost with no invalid release, harmful regression, timeout increase, or unsupported route`. Both `combined_bounded_route` and `exact_tool_cost_guard` have `max_new_tokens` of `0` and `solver_call_budget` of `1`. The `router_threshold_k` is `1`. Displayed phase-curve rows give both arms an `exact_joint_success_rate` of `1.0`, while displayed ledger rows mark `recovery_vs_longer_flat` as `true` for both.

## RECOMMENDATION
NARROW_CLAIM

## experiment_6557_constraint_saturation_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, not a result claim. A completed audit containing rows that contradict a stated comparative conclusion would refute such a conclusion.

## WAS THAT CHECKED
No. Execution stopped at the conductor pre-gate, so no experimental rows, comparator arms, or success criteria were evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_observed`: `null`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6558_arc_live_redirect_ledger_reachability.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The live receipt path was reachable, firings were inspected, and the evidence supported no selection-policy change.

## WHAT WOULD REFUTE IT
A supported replay in which an adequately sampled lower-priority arm outperformed the current ordering and produced a different recommended order—or live receipts proved unreachable or outcome-bearing data were missing—would refute the claim.

## WAS THAT CHECKED
Yes. Six live runs and seven firings were inspected; both lower-priority arms reached the three-firing support floor and produced no helped outcomes. The design demonstrably could block on missing outcome-bearing receipts, as its recorded prior failure did.

## EVIDENCE
`"honest_verdict": "complete: live receipt path reachable, firings inspected, no supported policy change"`; `"accepted_live_run_count": 6`; `"fired_total": 7`; `"allow_reinduction": 3`; `"force_exploration_diversity": 3`; `"helped_outcomes": 0`; `"support_floor_met": true`; `"support_disposition": "supported_no_help_lower_candidate"`; `"disposition": "unchanged"`; `"policy_changed": false`; `"reason": "supported replay does not improve the current curated order"`; `"exp6524_status": "blocked_missing_outcome_bearing_live_receipts"`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6559_gatemate_changed_state_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the artifact’s administrative findings, a valid operator-authored physical-state receipt newer than Exp6525, any executed hardware command, or failure to preserve the Exp3866 exclusion would contradict the recorded closure.

## WAS THAT CHECKED
Yes. The artifact audits 24 receipt candidates, recomputes command counts from action rows, and checks preservation of the Exp3866 exclusion; however, it explicitly makes no comparative, performance, quality, or availability claim.

## EVIDENCE
`"scope": "continuity receipt only; no performance, quality, or general availability claim"`; `"performance_claim_made": false`; `"valid_receipt_count": 0`; `"hardware_command_count_recomputed": 0`; `"hardware_action_rows": []`; `"preserved": true`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_6560_v567_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone reached an honest null disposition: production routing remains default-off, experimental mechanisms were not adopted, CSL and GateMate remain blocked, ARC selection did not change, and V567 did not close the publication requirement.

## WHAT WOULD REFUTE IT
Any eligible row showing production routing enabled by default, CSL or GateMate unblocked, ARC selection changed, experimental mechanisms adopted, or V567 credited with closing the independent-reproducer requirement would falsify the headline.

## WAS THAT CHECKED
Yes. The artifact checks these dispositions in the claim-and-adoption matrix, independent recomputation rows, ARC and hardware dispositions, publication-gate record, and artifact-eligibility rows. The checked outcomes consistently preserve the null or blocked states.

## EVIDENCE
`honest_verdict`: `complete_v567_independent_capstone_null: production adapter and Rust/PyO3 remain default-off; reversible memory and constraint saturation stay experiment-only; CSL and GateMate are blocked; ARC selection is unchanged; publication gate is reported without claiming V567 closed it`

`closed_class_counts`: `{"blocked": 4, "null": 5, "positive": 3}`

`state`: `default-off`

`state`: `experiment-only`

`state`: `blocked`

`policy_changed`: `false`

`selection_policy_disposition`: `unchanged`

`hardware_advanced`: `false`

`v567_integration_closes_independent_reproducer_requirement`: `false`

`verifier_is_oracle`: `false`

`quarantined_or_critical_artifacts`: `[]`

## RECOMMENDATION
KEEP
