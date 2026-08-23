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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 4 |
| CANNOT_DETERMINE | 1 |

## experiment_6522_chronological_conflict_self_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Chronological exact-conflict memory produced positive charged held-future benefit over scratch and frozen-empty controls while preserving exact answers, prefix support, and safe operation.

## WHAT WOULD REFUTE IT
Both valid-reuse arms tying or exceeding the scratch and frozen-empty held-future charged cost—making their charged benefit nonpositive—would refute the claimed benefit.

## WAS THAT CHECKED
Yes. The held-future support rows directly compare both valid-reuse arms against scratch and frozen-empty controls under matched dose; both reuse arms had substantially lower charged cost and positive benefit. Sequential access, exact equality, retention, and unsafe operations were also checked.

## EVIDENCE
`benefit_beyond_scratch_and_frozen_controls`: `true`; `matched_dose`: `true`; `valid_unbounded_reuse`; `charged_cost`: `40`; `charged_benefit_vs_scratch`: `84`; `charged_benefit_vs_frozen_empty`: `84`; `valid_bounded_reuse`; `charged_cost`: `44`; `charged_benefit_vs_scratch`: `80`; `charged_benefit_vs_frozen_empty`: `80`; `scratch`; `charged_cost`: `124`; `frozen_empty_memory`; `charged_cost`: `124`; `held_future_unread_until_boundary`: `true`; `decisions_use_only_prior_store_hash`: `true`; `verifier_is_oracle`: `false`; `exact_answer_equality`: `true`; `unsafe_use_count`: `0`; `unsafe_write_count`: `0`; `prefix_retention_within_margin`: `true`

## RECOMMENDATION
KEEP

## experiment_6523_adaptive_validation_csl_audit.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['; under ', ', both have ', '. Nevertheless, ', ' is reported as ', ' has ', ', versus ', ', with '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
The independent full audit supports continuous self-learning, while adaptive validation saves checks without changing the full-set decision.

## WHAT WOULD REFUTE IT
A serious comparator matching the claimed learning method’s held-future benefit and charged cost would refute added value; the restart arm does exactly that, tying valid unbounded reuse.

## WAS THAT CHECKED
Yes. The final full held-set audit evaluated every candidate, and the held-future support audit reports both the method and restart in the winner tie set. Adaptive validation itself was given a real chance to fail through decision disagreement or failure to save checks, but neither occurred.

## EVIDENCE
The `winner_tie_set` contains `valid_unbounded_reuse` and `restart`. Under `benefit_vs_scratch`, both have `84`; under `charged_cost_totals`, both have `40.0`. Nevertheless, `full_audit_winner` is reported as `valid_unbounded_reuse`. For validation cost, `variance_weighted_adaptive` has `charged_checks` of `108`, versus `132` for `fixed_subset` and `180` for `full_set`, with `decision_agreement_with_full` equal to `true`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_6524_arc_supervisor_redirect_generalization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An accepted live receipt marked outcome-bearing, with redirect or arm-outcome data, would refute the artifact’s stated reason for being blocked; a supervisor-value or generalization claim would additionally require outcome-bearing comparisons against a serious baseline.

## WAS THAT CHECKED
Yes. The accepted live receipts were individually classified, the outcome-bearing count was recomputed, and the redirect-outcome table was inspected; all three receipts were disabled and non-outcome-bearing. Thus the blocked status had a real chance to be contradicted, but no comparative supervisor claim was tested.

## EVIDENCE
`honest_verdict`: `blocked: missing outcome-bearing live trajectory-supervisor receipts`; `verdict_class`: `blocked`; `outcome_bearing_receipt_count`: `0`; `live_receipt_count`: `3`; `disabled_receipt_count`: `3`; `outcome_bearing`: `false`; `arm_outcomes_present`: `false`; `redirect_outcome_rows`: `[]`; `solve_credit_claimed`: `false`; `arc_generalization_slot_complete_score`: `0.0`

## RECOMMENDATION
KEEP

## experiment_6525_gatemate_changed_state_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No qualifying post-Exp6325 physical-state receipt existed, so no hardware command ran and Exp3866 remained excluded.

## WHAT WOULD REFUTE IT
A candidate row marked valid for a post-Exp6325 operator-authored material physical-state change, any executed hardware-command row, or evidence that Exp3866 was not excluded.

## WAS THAT CHECKED
Yes. The receipt candidates were individually evaluated, command rows and counts were recorded, and the Exp3866 exclusion state was checked.

## EVIDENCE
`"receipt_candidate_count": 24`, `"valid_receipt_count": 0`, `"new_post_exp6325_physical_receipt_found": false`, `"command_rows": []`, `"hardware_command_count": 0`, `"exp3866_clean_terminal_evidence_excluded": true`, `"verdict_class": "blocked"`, `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6526_v564_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Held, oracle-distinct rows support structural headroom, learned-router, continuous-self-learning, and adaptive-validation benefits, while ARC generalization and GateMate continuity remain blocked.

## WHAT WOULD REFUTE IT
A held comparison showing the learned router tied or lost to the best structural arm after charged costs—for example, zero held-benefit units—would refute the router component of the headline claim.

## WAS THAT CHECKED
Yes. The learned-router comparison used the best structural arm and reports 28 held-benefit units, candidate preservation, exact-solver release authority, and held-contamination checks. The artifact also demonstrates that adverse outcomes could survive aggregation by retaining the ARC and GateMate lineages as blocked.

## EVIDENCE
`verdict_class`: `partial`; `authority`: `oracle_distinct_row_reduction`; `held_benefit_beyond_best_structural_units`: `28`; `charged_cost_accounting_passed`: `true`; `candidate_preservation_passed`: `true`; `held_contamination_free`: `true`; `adaptive_charged_checks`: `108`; `full_set_charged_checks`: `180`; `adaptive_decision_agreement`: `true`; `oracle_distinct_held_future_benefit`: `true`; `blocked_lineage_count`: `2`; `outcome_bearing_receipt_count`: `0`; `hardware_command_count`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6527_v565_evidence_eligibility_corrigendum.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a governance/eligibility receipt, not a comparative value claim. A scientific claim would require outcome rows showing the adopted method ties or loses to a serious baseline.

## WAS THAT CHECKED
No; the artifact replays eligibility, hashes, rows, and command receipts but does not test a new comparative claim.

## EVIDENCE
`complete_v565_evidence_root_eligible`, `verdict_class`, `null`, `verifier_is_oracle`, `true`, `immutable_v564_row_replay_and_live_validation_no_llm`, `verifier_is_oracle is only for hash, row, and command receipts`, `verdict_class_null_governance_root`

## RECOMMENDATION
KEEP

## experiment_6528_v565_source_model_method_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify. Treating the blocked status as an operational assertion, verification of every required primary source—including the OpenReview sources with matching dates—would refute it.

## WAS THAT CHECKED
Yes. `gate_check_summary` checks required primary-source verification and records the two OpenReview failures; no experimental method comparison was run.

## EVIDENCE
`blocked_v565_source_model_method_contract`, `partial`, `required_primary_sources_verified`, `false`, `all_gates_passed`, `false`, `v565_method_contract_ready_score`, `0.0`, `model_loaded_or_run`, `false`, `low_concurrency_primary_source_and_cache_preflight_no_experimental_llm`, `No upstream aggregate or run database result transfers.`

## RECOMMENDATION
KEEP

## experiment_6530_external_constraint_corpus_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; treating the blocked status as an operational claim, it would be refuted if the required corpus inputs existed and all audit gates passed.

## WAS THAT CHECKED
Yes. The existence receipts, gate summary, empty audit rows, and recomputed readiness explicitly check those conditions and report failure rather than success.

## EVIDENCE
`status` `blocked_external_constraint_corpus_audit` `verdict_class` `blocked` `all_gates_passed` `false` `fixture_row_count` `0` `blocked_preconditions` `true` `external_constraint_corpus_audited_ready_score` `0.0` `verifier_is_oracle` `true`

## RECOMMENDATION
KEEP
