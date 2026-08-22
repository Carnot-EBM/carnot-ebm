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
| CLAIM_SUPPORTED | 5 |
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 2 |

## experiment_6478_identifiable_held_exact_energy_selection.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Exact-energy selection improves held exact success over the first-candidate and shuffled controls, so it is a positive result (verdict token `complete_positive`).

## WHAT WOULD REFUTE IT
A weight-free trivial comparator — just counting violated constraints, no energy weights at all — matching exact-energy's held success. If unweighted violation-count does as well, then the weighted "energy" selection adds nothing, and a `complete_positive` value result is unsupported. The only serious rival here is the `violation_count` arm; the other three beaten arms (`first_candidate`, `shuffled_energy`, `deterministic_random`) are sanity-check arms (arbitrary-first / shuffled / random).

## WAS THAT CHECKED
Yes — the artifact ran the `violation_count` arm and it ties `exact_energy` exactly: same 1.0 held success, paired gain 0.0, 24 ties / 0 wins / 0 losses, CI includes zero. The two formulas differ only by whether the same violated-constraint set is weighted or counted. So by the artifact's own data the weighting buys nothing; exact-energy wins only over sanity-check arms. The headline sentence is scoped to name only first/shuffled and dodge the tie, but the consumed verdict token is `complete_positive`, which reads as an added-value win. Compounding: `verifier_is_oracle` is true — the selector is derived from the same exact record that defines correctness, so the value claim is also circular, and beating `first_candidate` (rank-0 is always a perturbed/broken candidate) is near true-by-construction.

## EVIDENCE
- `"complete_positive: exact-energy selection improves held exact success over first and shuffled controls; exact backend remains the oracle"`
- `"arm": "violation_count"` with `"exact_success_rate": 1.0`
- `"arm": "exact_energy"` with `"exact_success_rate": 1.0`
- `"left_arm": "exact_energy"`, `"right_arm": "violation_count"`, `"paired_gain": 0.0`, `"tie_count": 24`, `"win_count": 0`, `"interval_excludes_zero": false`
- `"exact_energy_formula": "sum(weight_i for each violated Exp6477 source constraint)"`
- `"unweighted_violation_count_formula": "count(violated Exp6477 source constraints)"`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6503_v561_source_delta_method_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
No comparative claim — artifact is a source-receipt + frozen-method-contract preregistration ("source receipts pinned, method contract frozen, paper/product claims remain bounded context").

## WHAT WOULD REFUTE IT
Nothing empirical to refute. This is scaffolding: pins 11 source receipts, freezes a `method_contract`, records a readiness gate. Only "claims" are gate booleans (`method_contract_ready_score=1.0`, `all_gates_passed`), each recomputed from its own row set — pass/fail is a real check, not a comparative result. Any method-value claim is explicitly deferred: every promoted method carries `paper_claim_is_local_evidence: false` and points at a FUTURE local test (`exp6504`, `exp6506`, `exp6507`, `exp6509`).

## WAS THAT CHECKED
Not applicable — no comparative claim made. `verdict_class = "null"`, `verifier_is_oracle = false`, substrate `no_llm`. Blocked source (`openreview_clause_predictions`, `http_403`) is honestly recorded as `blocked_challenge`, not silently counted as success — receipt discipline holds.

## EVIDENCE
- `"verdict_class": "null"`
- `"honest_verdict": "complete_v561_source_delta_method_contract: source receipts are pinned, the method contract is frozen, and paper/product claims remain bounded context"`
- `"paper_claim_is_local_evidence": false`
- `"hardware_claim_allowed": false`
- `"retrieval_state": "blocked_challenge"`, `"http_state": "http_403"` (honestly flagged, not counted as win)
- `"all_gates_passed": true` with `"failed_checks": []`

## RECOMMENDATION
KEEP

## experiment_6502_v560_retirement_v561_lineage_lock.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V560 learned-energy/factor/ARC-policy/hardware scopes are retired (null result), and only the fresh exact SAT/CSP structural branch may seed V561 — a governance ledger, not a positive capability claim.

## WHAT WOULD REFUTE IT
- A retired scope whose own rows show a real positive held-out effect (e.g. learned head beats every shortcut control on held data) — would refute "retire".
- A forbidden-reuse attack that slips into V561 (`fail_closed: false`, `allowed_into_v561: true`) — would refute "locked".
- A V561 task gated on a retired upstream experiment ID — would refute the clean-lineage claim.

## WAS THAT CHECKED
Yes.
- Retirements derive from rows, not assertion: learned energy `trajectory_signal_ready_score_from_rows` = 0.0 because best shortcut control (`checkpoint`, 0.936975) beats best learned head (`mlp`, 0.880952) + 6 harmful flips. ARC alignment: held R² 0.004 < shuffled 0.017. These are the retirement being EARNED by rows that could have shown otherwise.
- All 8 forbidden-reuse attacks `fail_closed: true`, `allowed_into_v561: false`.
- `no_v561_task_depends_on_retired_upstream_experiment_id: true`.
- `verifier_is_oracle: true` but claim is null/governance, not a verifier value claim — no circularity harm.

## EVIDENCE
- `"verdict_class": "null"`
- `"best_learned_head_id": "mlp"`, `"best_learned_balanced_accuracy": 0.880952`, `"best_shortcut_control_id": "checkpoint"`, `"best_shortcut_balanced_accuracy": 0.936975`, `"harmful_flip_count": 6`
- `"held_incremental_r2": 0.004101`, `"shuffled_incremental_r2": 0.016746`, `"energy_alignment_positive_from_rows": false`
- `"all_forbidden_reuse_attacks_fail_closed": true`
- `"no_v561_task_depends_on_retired_upstream_experiment_id": true`
- `"every_decision_row_recomputed": true`

## RECOMMENDATION
KEEP

## experiment_6501_v560_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V560 capstone classified every upstream outcome and declares all four value claims (trajectory energy, continuous learning, ARC policy, hardware) INELIGIBLE.

## WHAT WOULD REFUTE IT
Headline promotes a value claim the rows do not support — e.g. marks `trajectory_energy_claim_eligible=true` while a shortcut survived, or reads a validity-flagged/disqualified row as a win. Refutation = any eligibility flag flipped true over a row showing shortcut survival, harmful flips, closed gate, or `flagged_adversarial`.

## WAS THAT CHECKED
Yes. Every value claim resolves to `eligible: false` with named reasons drawn from the disqualifying rows. Verifier is oracle (`verifier_is_oracle: true`) — but capstone makes NO verifier value claim; it declares all such claims ineligible. Disqualified/flagged rows are honored, not laundered. This is an honest-null bookkeeping artifact, not a promotion.

## EVIDENCE
- `"honest_verdict": "complete: V560 capstone classified every upstream outcome; trajectory_energy_claim_eligible=false; continuous_learning_claim_eligible=false; arc_policy_claim_eligible=false; hardware_claim_eligible=false"`
- `"all_shortcuts_rejected": false` with `"surviving_shortcut_ids": ["checkpoint"]`
- `"best_shortcut_balanced_accuracy": 0.936975` > `"best_learned_balanced_accuracy": 0.880952` (shortcut beats learned head — signal disqualified, honored)
- `"trajectory_signal_ready_score": 0.0`, `"harmful_flip_count": 6`
- trajectory_energy `"eligible": false` reasons `["trajectory_signal_not_ready","checkpoint_shortcut","harmful_flip_count_6",...]`
- `"headline_mismatch_count": 0`, `"all_expected_outcomes_classified": true`
- `"missing_unexplained_experiments": []`

## RECOMMENDATION
KEEP

## experiment_6500_gated_default_off_live_arc_policy_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
No claim. Artifact block record. Gate failed upstream, experiment never ran.

## WHAT WOULD REFUTE IT
Nothing to refute. No comparative claim, no A/B result, no verifier value claim. Just receipt: upstream gate `arc_energy_alignment_ready_score == 1.0` not met (`0.0`), so live ARC policy A/B skipped.

## WAS THAT CHECKED
Not applicable. Design never reached measurement. Block correct, fail-closed at `conductor_pre_gate`.

## EVIDENCE
- `"honest_verdict": "blocked_gate_check_failed"`
- `"status": "blocked"`
- `"blocked_reason": "actual=0.0 == expected=1.0"`
- `"duration_s": 0.0`
- `"gate_check_summary": "1 of 1 gate(s) failed; first failure: exp6499-arc-energy-progress-alignment.arc_energy_alignment_ready_score (actual=0.0 == expected=1.0)"`
- `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6499_arc_energy_progress_alignment.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Conservative prefix energy adds NO positive incremental alignment to live-agent progress over simple controls; energy-alignment gate NOT ready (`complete_null`).

## WHAT WOULD REFUTE IT
Energy showing positive held incremental R² beyond controls AND stable positive direction leave-one-game-out. That would flip verdict to `complete_positive`. Null claim refutable by own rows — not true-by-construction.

## WAS THAT CHECKED
Yes. Real rival present: shuffled_energy control. Shuffled energy scores HIGHER incremental R² (`0.016746`) than real energy (`0.004101`) — real signal loses to its own shuffle, strong null evidence. Energy coefficient negative (`-0.069817`), CI straddles zero. Leave-one-game-out direction never positive.

## EVIDENCE
- `"energy_alignment_positive_from_rows": false`
- `"held_incremental_r2": 0.004101` vs `"shuffled_incremental_r2": 0.016746`
- `"energy_signal_coefficient": -0.069817`
- ci95 `[-0.262655, 0.045856]` for energy_signal_coefficient (straddles 0)
- `"failed_gates": ["energy_has_positive_incremental_alignment", "leave_one_game_out_direction_stable"]`
- `"adequate_coverage_and_headroom": true` — headroom present, so null not a no-headroom artifact
- `"leave_one_game_out_direction_stable_from_rows": false`

## RECOMMENDATION
KEEP

## experiment_6498_csl_independent_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent row-replay of exp6496/exp6497 reproduces every upstream aggregate with zero discrepancies, and the continuous-learning value claim is withheld (ineligible) because held-future benefit is zero.

## WHAT WOULD REFUTE IT
Two refutation paths: (1) replay-validity half — a non-empty discrepancy row, a `source_matches_reported: false`, or `csl_audit_ready_score < 1.0` would show the recompute did not match source rows; (2) the honest-null half is refuted only if the artifact quietly asserted continuous-learning worked while rows show zero benefit — i.e. if any arm's `held_future_utility > 0` yet `continuous_learning_claim_eligible` were true, or vice versa (claim asserted on zero-benefit rows).

## WAS THAT CHECKED
Yes. `discrepancy_rows` is empty; recompute scores are 1.0; every future-metric arm shows `held_future_utility: 0.0`, and the artifact sets `continuous_learning_claim_eligible: false` matching that zero. The value claim is declined, not overstated. `verifier_is_oracle: true` is correct and harmless here — the audit IS the deterministic recompute (a reproduction check), and it makes no added-value claim that circularity could inflate.

## EVIDENCE
- `"honest_verdict": "complete_independent_audit: row replay validated; continuous-learning claim remains ineligible because held_future_benefit failed"`
- `"continuous_learning_claim_eligible": false`
- `"discrepancy_rows": []`
- `"held_future_benefit": false`
- `"continuous_self_learning_ready_score_from_rows": 0.0`
- `"csl_audit_ready_score": 1.0`
- future_metric_rows all arms: `"held_future_utility": 0.0`, `"metric_matches_source": true` (frozen_no_update, always_update, fixed_threshold)
- `"critical_discrepancy_count": 0`, `"false_accept_count": 0`, `"all_critical_fail_closed": true`

## RECOMMENDATION
KEEP

## experiment_6497_factor_pool_support_stress.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
`medium bounded capacity preserved support and exact safety under stress` — verdict token `complete_positive`, recommends `medium_bounded`.

## WHAT WOULD REFUTE IT
Frozen do-nothing baseline (`zero_frozen`, zero durable writes) matching `medium_bounded` on the two properties the sentence names — support preserved AND exact safety. If the no-write baseline ties on both stated dims, medium adds nothing on what the headline actually says; the real distinguisher (held future utility) is absent from the sentence and from the consumed token.

## WAS THAT CHECKED
Checked, and refutation OCCURRED in the artifact's own rows. `zero_frozen` and `small_bounded` both tie `medium_bounded` on `support_preserved` and `exact_safety`. Every arm is exact-safe (`exact_safety_all_capacities: true`). The only field that separates medium — `total_held_future_utility` (7.0 vs 2.5 vs 0.0) — is NOT in the headline sentence. Downstream reads `complete_positive` + "preserved support and exact safety," which the frozen no-write baseline achieves equally. Also: `verifier_is_oracle` true, so "exact safety" is oracle-defined and non-discriminating.

## EVIDENCE
- `"complete_positive: medium bounded capacity preserved support and exact safety under stress"`
- `"zero_frozen": {"exact_safety": true, "support_preserved": true, "total_held_future_utility": 0.0}`
- `"small_bounded": {"exact_safety": true, "support_preserved": true, "total_held_future_utility": 2.5}`
- medium: `"support_preserved": true`, `"total_held_future_utility": 7.0`
- `"exact_safety_all_capacities": true`
- `"reason": "No durable writes. This is the exact frozen baseline."` (zero_frozen)

## RECOMMENDATION
NARROW_CLAIM — headline + verdict token must cite the actual discriminator (`total_held_future_utility` 7.0, non-monotonic: overlarge 3.0 < medium despite bigger capacity), not "preserved support and exact safety," which the frozen no-write baseline and `small_bounded` both tie. Drop "exact safety" as load-bearing — oracle-defined and true for all arms.

## experiment_6496_continuous_factor_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Chronological replay ran row-complete across all 4 arms, but held-future learning readiness did not open (null).

## WHAT WOULD REFUTE IT
Any arm showing `held_future_benefit: true` / nonzero `held_future_utility` on the held_future horizon — readiness "opening." The verdict claims it did NOT open, so a single opening arm refutes it.

## WAS THAT CHECKED
Yes. All 4 arms recomputed from rows: every `future_evaluation` row `held_future_utility: 0`, `held_future_benefit: false`, readiness gate `held_future_benefit` in `readiness_failed_gates`. Refutation checked, absent. Verdict reports the null faithfully — makes no positive or comparative claim, no value claim (verifier_is_oracle true but no added-value assertion). Note: corpus is degenerate — every event `compile_reason: "empty_response"`, `durable_write_count: 0`, so no arm COULD admit a factor and all tie at 0. This makes the null true-by-construction, but the verdict does not overclaim method failure; it only claims readiness did not open, which is exactly what the rows show.

## EVIDENCE
- `"complete_null: chronological replay is row-complete, but held-future learning readiness did not open"`
- `"held_future_benefit": false`
- `"continuous_self_learning_ready_score": 0.0`
- `"csl_execution_complete_score": 1.0`
- `"durable_write_count": 0`
- `"compile_reason": "empty_response"`
- `"admitted_event_count": 0`
- `"readiness_failed_gates": ["held_future_benefit"]`
- exp6495 receipt: `"mechanism validated; no learning-benefit claim is made"`

## RECOMMENDATION
KEEP
