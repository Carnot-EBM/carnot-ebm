# Multi-positive win-set L2 goal induction (lever #2): built, and the re-reaching mechanism is structurally dead — ARC wins are canonical (outer-loop, 2026-06-24)

Operator-authorized multi-hour build of lever #2 (the adversarial design panel's sharpest attack on the
`.430 "single-exemplar goal induction insufficient" residual). Built fully, verified the plumbing end-to-end,
and a cheap CPU probe delivered a **decisive structural result before spending headline LLM time.**

## The build (committed `6c3e78d64`, branch `outer-loop/multiwin-goal`)

Extended the induce path (`arc_executable_world_model.py`: `_transitions_block` → new `_win_state_set_block`,
`induce_prompt`, both `induce` methods) to accept `win_state_set` (K `(start, win, delta)` positives) +
`win_invariant_hint` — additive, default `None` = current behavior (parity-safe). New prototype
`scripts/experiments/proto_multiwin_goal.py`:
1. **GATHER** K L1 positives by re-reaching L1 with `CARNOT_ARC_EXPLORE_DIVERSITY=1` + per-run `random.Random`
   seeds injected into `policy.explorer._div_rng`.
2. **INVARIANT** — precompute cross-positive structural features (True in every win, False in every start).
3. **INDUCE** is_level_complete + engine via the real Qwen, with the WIN-STATE SET block.
4. **DISCRIMINATIVE GATE** — accept only if True on all K wins, False on all K starts + distractors, AND robust
   to a non-delta-cell perturbation (the leave-one-out anti-memorization test the `.430 gate lacked).
5. **PLAN** — `plan_in_model` → reaches_goal? (the field that was 0/False in `.430).

**Plumbing verified end-to-end** (mock proposer): `win_state_set` threads through; the gate correctly
**rejects** a literal-grid-match predicate via the perturbation test (`robust_to_nondelta_perturb=False`).

## DECISIVE FINDING: ARC-AGI-3 wins are CANONICAL → re-reaching cannot yield distinct positives

A CPU probe (no LLM) re-reached L1 across diversity-on, per-seed-distinct exploration paths on
**lp85, sc25, tu93, vc33**. Every game: **`distinct_wins = 1`** — every seed converges to the *identical* win
grid (`win_changed_cells` lp85=1487, sc25=359, tu93=711, vc33=2364, byte-identical across seeds).
Level-completion converges to ONE solved configuration, so **re-exploring the same level always produces the
same win grid.** The judge's failure-mode-1 ("K positives nearly identical") is therefore **universal, not
occasional** — the "re-reach K times" gather mechanism is structurally dead. K>1 degenerates to K=1; the
precomputed invariant degenerates to "the win is this grid" (memorization).

## The K=1 richer-encoding fallback: inconclusive (Qwen server failures)

What survives re-reaching's death is the **richer single-win encoding** (start + delta + invariant +
discriminative directive vs `.430's win-grid-only). On lp85+sc25 K=1 the real Qwen induce **failed at the
server** (`HTTPError 400` on lp85, `TimeoutError` on sc25; prompt ≈ 2.8k tokens, well under the 16k context —
NOT an overflow). The `.430 reinduction uses the same proposer config successfully, so these are
proposer/server-tier flakiness, not a goal-quality result — the K=1 fallback is **inconclusive**, and not
worth more cycles given the decisive finding above.

## Where this redirects: lever #4 (mechanic-template)

The canonical-win structure that **kills** multi-positive is exactly what **favors** lever #4: each game's win
is a *clean canonical mechanic* (lp85 align-pieces, sc25 toggle-then-reach-exit, tu93 reach-color14-goal, tn36
match-5-attributes — a small enumerable set). A **deterministic goal-predicate template keyed on the detected
mechanic** fits a canonical win far better than fuzzy multi-example induction. The extension (`win_state_set`
plumbing) + the discriminative gate built here are reusable for #4.

**Salvage paths for multi-positive (if pursued, both another build):**
- **Cross-game same-mechanic positives** — wins from DIFFERENT games sharing a mechanic class. Limited: per-game
  mechanics differ, so few same-mechanic games exist.
- **Augmentation** — synthetic transforms (translate/reflect) of the one canonical win, forcing a
  *transform-invariant* (structural) goal that still holds on the original engine output. Speculative; needs
  consistency with the engine's output space.

## Honest bottom line

The multi-hour build did its job: it **falsified the re-reaching multi-positive mechanism cheaply** (canonical
wins, CPU probe, no headline LLM burn) and sharpened the path. Multi-positive-via-re-reaching is dead; the
remaining live, well-motivated lever is **#4 mechanic-template**, which the canonical-win finding supports.
Cross-refs: `arc-l2-goal-induction-levers` workflow (the ranked panel), `multi-level-deepening-diagnostic-2026-06-23.md`
(the diagnosis), branch `outer-loop/multiwin-goal` (the build), `results/proto_multiwin_goal.json`.
