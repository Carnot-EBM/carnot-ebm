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
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6768_targetable_proof_panel_expansion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The operational readiness statement would fail if any of the 126 fixtures cold-replayed differently, passed exact validation, duplicated another row, or failed to cover its declared error class; there is no model-quality or comparative claim to refute.

## WAS THAT CHECKED
Yes. The cold-replay receipt checks all 126 rows for mismatches, while the panel summaries check exact validity, duplication, and coverage across error classes and proof families.

## EVIDENCE
`honest_verdict`: `complete: targetable exact-invalid fixture ready with 126 cold-replayed rows; this is not a model-quality claim`; `inference_substrate`: `deterministic_local_certificate_mutation_no_llm`; `replayed_row_count`: `126`; `mismatches`: `[]`; `all_passed`: `true`; `exact_valid_mutations`: `0`; `duplicate_rows`: `0`; `targetable_panel_ready`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6769_environment_indexed_proof_grammar_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The environment-indexed grammar kept valid SAT and UNSAT witnesses reachable across the tested size bins while emitting zero out-of-environment symbols.

## WHAT WOULD REFUTE IT
An environment-indexed support row with an exact-valid SAT or UNSAT witness that was not candidate- or terminal-reachable, or any nonzero ghost-symbol violation, would refute the claim.

## WAS THAT CHECKED
Yes. The per-mode support receipts checked SAT and UNSAT witnesses in each reported size bin, and the artifact separately counted ghost violations. The static CFG ties on preserved support, so the artifact demonstrates reachability and exclusion—not comparative added value, which it explicitly does not claim.

## EVIDENCE
`"honest_verdict": "complete: environment-indexed SAT and UNSAT support remained reachable with zero ghost violations; no live SOTA comparison was run"`; `"valid_sat_reachable": true`; `"valid_unsat_reachable": true`; `"no_ghost_violations": 0`; `"support_preserved": true`; `"candidate_reachable": true`; `"terminal_reachable": true`; `"exact_valid": true`; `"verifier_is_oracle": false`; `"inference_substrate": "deterministic_automaton_no_llm"`; `"scope": "one exact-valid witness; no proof-language enumeration claimed"`.

## RECOMMENDATION
KEEP

## experiment_6770_dccd_environment_grammar_ab_v2.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the intended comparative claim, attributable paired rows showing `dccd_environment` tying or losing to `static_grammar` or `repaired_direct` on exact validity or semantic correctness would refute added value.

## WAS THAT CHECKED
No. Execution stopped at a failed precondition before inference, producing no rows; the reported zero deltas therefore provide no comparative test.

## EVIDENCE
`"verdict_class"`: `"blocked"`; `"live_model_invoked"`: `false`; `"proof_transport_ab_completed"`: `false`; `"rows"`: `[]`; `"models_used"`: `[]`; `"failed_check"`: `"one_model_vram"`; `"rows_complete"`: `false`

## RECOMMENDATION
KEEP

## experiment_6771_proof_transport_localization_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive experimental claim is made. The administrative blocked status would be refuted if the upstream gate had observed `proof_transport_ab_completed` as true or recorded the gate as passed.

## WAS THAT CHECKED
Yes. The sole prerequisite was checked in `gates_evaluated`; it failed before the experiment ran, so no comparative or value claim was tested.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "proof_transport_ab_completed"`; `"failed_expected": true`; `"failed_observed": false`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6773_csl_owned_lease_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an admission receipt, not a comparative or value claim. Its reported blocked state would be contradicted by an eligible selected GPU or actual live-model execution.

## WAS THAT CHECKED
Yes. The device-selection receipt, gate summary, invocation flag, and model-use receipts check those possibilities.

## EVIDENCE
`"verdict_class": "blocked"`, `"status": "blocked"`, `"eligible_device_count": 0`, `"selected_device": null`, `"live_model_invoked": false`, `"models_used": []`, `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6776_arc_shadow_supervisor_accrual.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; a supervisor-benefit claim would be refuted by valid live rows showing no advantage or harm versus a serious no-supervisor baseline.

## WAS THAT CHECKED
No. Preconditions blocked inference, so no live actions, invariance pairs, supervisor effects, or comparative outcomes were observed.

## EVIDENCE
`"status": "complete_blocked_shadow_supervisor_accrual"`; `"models_used": []`; `"live_model_invoked": false`; `"actions_observed": 0`; `"method": "not run because preconditions failed"`; `"passed": false`; `"status": "insufficient_evidence"`; `"shadow_supervisor_transport_ready": false`; `"solve_claim": false`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_6777_arc_tool_gap_transport.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked gate receipt, not a substantive comparative or value claim.

## WAS THAT CHECKED
No; the experiment was blocked before execution at the upstream gate.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `title`: `Live selfparse tool-gap transport receipt`; `honest_verdict`: `blocked_gate_check_failed`; `expected`: `true`; `actual`: `false`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6780_v590_branch_disposition.json

**SKIPPED_ALREADY_FLAGGED**
