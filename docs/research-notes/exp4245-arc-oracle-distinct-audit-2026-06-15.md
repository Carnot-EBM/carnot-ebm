# Audit: the .393 ARC "+44pp oracle-distinct win" (exp4245)

**Outer-loop audit, 2026-06-15, operator-requested.** Question: is the headline
`arc_oracle_distinct_set_encoder_beats_vote` (set_encoder@1 − vote@1 = **+0.442**,
CI95 [0.308, 0.596], `verifier_is_oracle=false`) a real oracle-distinct win, or an
artifact of leakage / task selection? It is **real and leak-free, but thin and
regime-specific** — it motivates the DiffusionGemma bet but does not yet robustly
establish "the learned verifier beats vote on ARC."

## What was checked

| Check | Result | Verdict |
|---|---|---|
| **Leakage (OOF integrity)** | exp4244 = genuine 5-fold task-level CV, 28443 candidate-rows. train_task_ids: 41–43 tasks/fold; **0 rows with empty train sets**; **0 rows where a task is in its own training set**. The gate (`load_heldout_pool`) only counts tasks where ALL candidates are train-excluded. | **CLEAN** — no leakage. |
| **Task selection** | The grown pool (exp4243) has `task_n = 52` total; the held-out is all 52. So it is the **entire available GAP-4 ARC pool**, NOT a wrong-majority-selected subset. vote@1=0.25 is the genuine vote rate on that pool. | **NOT cherry-picked.** |
| **Where the delta comes from** | set_encoder 36/52 vs vote 13/52. The held-out is 30/52 (58%) wrong-majority (oracle-present, vote-wrong). The **entire** +44pp delta = the set-encoder recovering **23/30 (77%)** wrong-majority basins; on the other 22 tasks SE and vote tie. | Real wrong-majority-basin recovery. |
| **Adversarial-verify** | clean, 0 flags; `circular_moat_overclaim_clean=true`. | Clean. |

## Verdict: REAL but THIN and GAP-4-regime-specific

The win is genuine — a learned, no-execution (`verifier_is_oracle=false`) DeepSets
set-encoder recovers 77% of the wrong-majority basins that majority vote misses on the
GAP-4 ARC candidate pool, leak-free, CI excludes 0. This is a legitimate FIRST ARC
oracle-distinct signal and closes "GAP-3-ties-vote" *on this pool*.

But three caveats temper the headline magnitude:

1. **n = 52** — the entire available GAP-4 ARC pool. Wide CI [0.31, 0.60]. A small-sample
   result; cannot grow without a bigger pool.
2. **GAP-4-generation-specific.** The pool is 58% wrong-majority because GAP-4 candidate
   generation emits many wrong candidates (hence vote@1=0.25). On a stronger candidate
   generator (where vote is better), the delta would shrink. The claim is "recovers
   wrong-majority basins on GAP-4 ARC candidates," not "beats vote 44pp on ARC broadly."
3. **Tiny task diversity.** Trained + tested OOF over only 52 ARC tasks; no truly
   out-of-distribution ARC test set. Generalization to NEW ARC tasks is unestablished.

## Recommendation

- **Honest headline:** "On the GAP-4 ARC pool (n=52), the oracle-distinct set-encoder
  recovers 77% of wrong-majority basins vote misses (+44pp, CI95 [0.31,0.60], leak-free,
  verifier_is_oracle=false)." Cite the n=52 / GAP-4-regime / wrong-majority context.
- **Before this robustly flips the DiffusionGemma gate** (vs merely motivating it): grow
  the ARC pool well beyond 52 tasks, add a fresh out-of-distribution ARC task set, and
  replicate across seeds — the analogue of the conductor's own A4 code-win replication,
  which currently replicates only the CODE win, not this ARC one.
- The CODE oracle-distinct win (exp4233, +3.1pp, n=160, balanced) remains the stronger-
  powered leg; ARC is the north-star but this single 52-task result is the thin part.

## Provenance
- exp4245 `results/experiment_4245_arc_set_encoder_beats_vote.json`
- exp4244 model `results/experiment_4244_arc_set_encoder_aggregator_model.json` (5-fold OOF)
- exp4243 pool `results/experiment_4243_arc_candidate_pool_grow_pool.json.gz` (task_n=52)
- module `python/carnot/reporting/arc_set_encoder_beats_vote_4245.py`
