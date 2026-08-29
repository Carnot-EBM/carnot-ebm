# The 43 held-back fabrication stamps: per-artifact decisions

Date: 2026-08-29. Read-only measurement; nothing under `results/` was written to produce this.

Two commits on 2026-08-28 retracted 147 of 158 eligible stale stamps (`e0d69d1e30`, `60b9dcc246`).
43 were deliberately held back: 19 whose original flag included IMPLAUSIBLE_PERFECT, 9 carrying no
recorded reason at all, and 15 of mixed kind. The operator's instruction was explicit -- refactor
these per artifact, do not sweep them.

## Why a sweep would be wrong here, concretely

Re-judged with the live gate, **all 43 draw no CRITICAL flag today**. That is exactly the argument
a sweep would rest on, and it is not sufficient. Three of the nine 'no reason' stamps have
substantive reasons that live in COMMIT MESSAGES rather than in the artifact:

- **exp3929** -- `action_efficiency_ratio=1.959` is circular. `task_potential()` reads oracle goal
  fields the synthetic env exposes by construction, then plants a fixed string (`"...actually
  inconsistent."`); the energy verifier then detects the contradiction it was handed. A bare
  `if 'inconsistent' in step_text` reproduces the identical 1.96x.
- **exp3959** -- claimed 24.662x with the verifier NEVER IN THE LOOP: the two arms are
  `simulate_geometric()` draws from assumed probabilities, object-vs-pixel search geometry, despite
  `inference_substrate` claiming 'real games'.
- **exp3405** -- carries an adversarial-verify corrigendum recorded at commit time.

Clearing those three would have re-admitted results a human adversarial review deliberately
quarantined, on the strength of a gate that no longer emits the flag. The gate changing its mind
is not the same as the finding being wrong.

## The defect this exposes

A determination is stored as a boolean with no reason beside it. The reason, when one exists, is in
a commit message -- which no consumer of the gate reads. `scripts/conductor_gates.py` keys off the
field being present, so the artifact must carry its own justification or the next reader faces the
same undecidable choice. **The refactor is to move the reason INTO the artifact and keep the stamp**,
not to clear it.

## Per-artifact decisions

| action | n | meaning |
|---|---|---|
| KEEP+RECORD | 3 | stamp stands; write the commit-message reason into the artifact |
| KEEP | 5 | the declared verdict is itself the reason (disqualified, simulated, degenerate metric) |
| RETRACT-defensible | 26 | terminal honest result or mechanical transition, no live critical flag |
| RETRACT-low | 8 | `blocked_*`, never headline-eligible, so clearing changes nothing downstream |
| INVESTIGATE | 1 | no `honest_verdict` at all |

| experiment | action | reason | declared verdict |
|---|---|---|---|
| 833 | KEEP | verdict itself is the reason (simulated / missing write path / degenerate metric) | `write_path_missing` |
| 1736 | KEEP | verdict itself is the reason (simulated / missing write path / degenerate metric) | `vivado_simulated_success` |
| 2295 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: partial_fix — pypi_escalation ImportError re` |
| 2444 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: AUROC=0.500` |
| 2516 | KEEP | verdict itself is the reason (simulated / missing write path / degenerate metric) | `complete: best_242_auroc=0.0000; phase4_validated_any=` |
| 2983 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: trace_to_skill_memory_ready` |
| 3168 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_flagged_verifier: gated_skip=true: exp3165 pre` |
| 3183 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: counterexample_certificate_expansion_v3_read` |
| 3185 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_repair_gate_precondition: repair gate not unbl` |
| 3312 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: promotion_safe=true; garak_gate_passed=true;` |
| 3361 | RETRACT-defensible | mechanical milestone transition, nonterminal-by-design | `archive complete` |
| 3377 | RETRACT-defensible | mechanical milestone transition, nonterminal-by-design | `archive complete` |
| 3392 | RETRACT-defensible | mechanical milestone transition, nonterminal-by-design | `archive complete` |
| 3405 | KEEP+RECORD | substantive reason exists only in the commit message | `None` |
| 3844 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_fover_balanced_corpus_not_available` |
| 3886 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_graph_verifier_not_invoked` |
| 3928 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_all_gguf_inference_failed` |
| 3929 | KEEP+RECORD | substantive reason exists only in the commit message | `complete: arc_agi3_verifier_router_HELPS_ratio1.959_ci` |
| 3959 | KEEP+RECORD | substantive reason exists only in the commit message | `complete: m3_efficiency_real_games_pruner_helps` |
| 4212 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: certified_arc_corpus_absent_lift_ci_touches_` |
| 4513 | RETRACT-defensible | terminal honest result, no live critical flag | `success: adaptive_budget_median_actions_2984_below_776` |
| 4533 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: reinduction_no_deeper_level_barrier_refined_` |
| 4556 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: verifier_router_no_value_added_honest_null_g` |
| 4582 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: feature_router_no_value_honest_null_transfer` |
| 4583 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: diversity_floor_no_transfer_honest_null_gap_` |
| 4617 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: spatial_value_head_graduated_no_live_value_h` |
| 4726 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: online_action_learning_no_first_win_lift_res` |
| 5089 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_pbit_guided_cdcl_distribution_sensitive_no_wi` |
| 5119 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_sota_endpoint_rootcause_adversarial_flag` |
| 5134 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_archive_470_closed_471_active_roadmap_ready` |
| 5156 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_archive_472_closed_473_active_runtime_clean` |
| 5212 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_gap4_scale_validation_v477_n0_missing_protoco` |
| 5225 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: clean GAP-4 validation null decision with n=` |
| 5236 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: GAP-4 is still blocked after QA calibration ` |
| 5632 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: no_new_arc_level_banked_lf52_L7_bounded_live` |
| 5643 | RETRACT-defensible | terminal honest result, no live critical flag | `complete: no_new_arc_level_banked_lf52_L8_bounded_live` |
| 6228 | INVESTIGATE | no honest_verdict at all | `None` |
| 6275 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_partial: test_exit_codes` |
| 6487 | KEEP | artifact declares itself disqualified | `disqualified: candidate_identifier_length,candidate_id` |
| 6490 | KEEP | artifact declares itself disqualified | `disqualified: shortcut control survived held trajector` |
| 6586 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_isolated_environment: pytest_receipt` |
| 6589 | RETRACT-low | blocked_* was never headline-eligible; clearing changes nothing downstream | `blocked_receipt_validation_block: terminal_report_vali` |
| 6687 | RETRACT-defensible | terminal honest result, no live critical flag | `complete_null: V582 has a null execution-integrity rec` |

## Execution rules

Retraction follows the route `scripts/determination_preservation_lint.py` documents: keep the field,
set it false, add `flagged_adversarial_cleared_note` beside it. Never delete the field -- that lint
exists because a corrigendum was once lost exactly that way. Every artifact is re-verified
INDIVIDUALLY at write time rather than trusting this table, and `corrigendum_pending` /
`corrigendum_note` are preserved on all of them.

`results/arc_e3`, `results/arc_logo_snapshot` and `results/arc_e3_origin_fixtures` are evidence and
are not touched by any of this.
