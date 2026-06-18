# Research Scope: Config/Toggle Target-Pattern Induction for ARC-AGI-3 (2026-06-18)

Outer-loop research scope. This document scopes the highest-leverage AND hardest remaining paradigm
in the first-contact ARC-AGI-3 solver (see `arc-first-contact-solver-2026-06-18.md`): inducing the
**target** of configuration/toggle games so they can be solved without a banked solution. It is a
research plan with falsifiable gates, not an implementation.

## 1. Problem statement

After the multi-paradigm solver reached **40% first-contact (10/25)** with navigation + click-
interaction tiers, **9 of the 15 remaining failures are config/toggle games**: bp35, dc22, g50t,
ka59, lf52, s5i5, tn36, sc25, tr87. (ka59, long mis-treated as a push game, is config — 65 recolor /
0 move events.) In these games:

- Actions (mostly clicks) **recolor/cycle cells in place** (no movement). The editable region is
  40-210 cells.
- The win is reached when the editable region matches a **specific target configuration** — a
  pattern, a rule-derived result (tr87 is a glyph-rewrite), or a multi-path condition (sc25).

The solver's back end (toggle-to-match search) is not the blocker. The blocker is **knowing the
target** so the search has a goal to descend toward.

## 2. Measured baseline (the easy path is killed)

Three generic, target-free approaches were built and measured; all fail (`arc3_config_paradigm_probe.json`):

1. **Stumble** (click-config BFS): intractable — the config space is colors^(40..210); a win is
   never stumbled.
2. **Shape-matched static target**: fails 4/5 probed — no static region matches the editable
   region's cell-shape (lf52/s5i5/sc25/g50t; only dc22, likely spurious). Config games do not have a
   generic "edit this to match that same-shape reference" layout.
3. **Systematic editable-cell sweep**: fails all 5 — 200-964 clicks cycling colors never wins; there
   is no monotone progress signal.

Conclusion: target induction needs the game's **specific** target, which is not recoverable from
generic geometric/progress heuristics. This is scene-understanding, not a heuristic tier.

## 3. Target taxonomy (research Stage 1 — measure before building)

Before building an inducer, characterize what KINDS of targets the 9 games actually have. For each
game, using its banked solve (which reveals the winning configuration) as ground truth, classify the
target by structure:

- **(a) Visible same-modality reference** — a static region (possibly DIFFERENT shape/scale than the
  editable) that the editable must reproduce (e.g. a goal pattern shown elsewhere). The shape-match
  probe failed because it required SAME shape; a different-shape/scaled reference is the likely real
  case.
- **(b) Legend/rule region** — a small visible key (symbol→colour) or a rewrite-rule grid (tr87)
  from which the target is COMPUTED, not copied.
- **(c) Constraint/relational** — the target is defined by a relation the cells must satisfy
  (sc25-style multi-path; "no two adjacent same colour"; counts), with no single reference region.

Deliverable: `arc3_config_target_taxonomy.json` — each of the 9 games tagged (a)/(b)/(c) with the
ground-truth target region (from the banked solve) and whether a visible reference/legend exists.
**Gate:** at least 4 of 9 are class (a) or (b) (a findable/derivable target) — else config is
dominated by (c) relational wins and the plan re-scopes to constraint-induction.

## 4. Proposed approaches (ranked by leverage × tractability)

### 4.1 LLM-as-scene-reader (primary; SOTA-aligned)

The frontier ARC-AGI-3 agents use an LLM to read a puzzle's implicit instructions (cf.
`reference_arc_agi3_sota_and_plan` — Family-B executable world models; the coding-agent reads the
scene). Apply the SAME idea narrowly: give a **local open-weight GGUF** (offline-legal per
Decentralization Rule 1) the rendered config scene (ASCII at logical resolution) + the editable
region location, and ask it to output the **target configuration** (or the reference region's
coordinates / the rewrite rule). The LLM does the perception/instruction-reading; the existing search
does the execution. Decompose so the LLM output is VERIFIED (the target it proposes is checkable: run
the toggle-to-match search; the real env confirms the win) — the Carnot pattern (LLM proposes,
verifier grounds). Falsifiable: the LLM-induced target lets the search win where the generic baseline
did not.

### 4.2 Reference-region detection, shape-agnostic (cheap heuristic, class (a))

Generalize the failed shape-match: find a static region whose COLOUR HISTOGRAM and structure match
the editable region's, allowing different shape/scale (template matching with scaling, or
colour-multiset correspondence). Cheap; catches class (a) games the same-shape probe missed.

### 4.3 Rule induction for class (b) (rewrite games, tr87-like)

Induce the visible rewrite rule (the rule grid → a program) and compute the target = rule applied to
the input. This is program synthesis from the rule region; tr87's GameAdapter did it by hand. Reuse
the M2-v2 / executable-world-model induction machinery. Narrow to the rewrite sub-class.

### 4.4 Learned target predictor (if (a)/(b) heuristics plateau)

Train on the config games whose target is known (banked solves) to predict the target region/pattern
from the scene; test leave-one-game-out. Only pursue if the LOGO transfer is above chance (prior
LOGO probes on value heads / engines came up chance — apply the same skeptical gate).

## 5. The executor (reuse — not the research)

Once a target T is induced, solving is the existing machinery: `best_first_search` (already reused
from `arc_heuristic_search_over_verified_wm.py`) over deep-copied real envs, **heuristic =
cell-mismatch(editable, T)**, action = click an editable cell (recolor), confirmed by the real-env
win. This is the same goal-directed-search back end the navigation paradigm uses; only the heuristic
target changes. Build it once, behind whichever inducer (4.1-4.4) produces T.

## 6. Staged plan with gates

- **Stage 1 — Taxonomy** (Section 3). Gate: >=4/9 class (a)/(b). Kill: dominated by (c) -> re-scope.
- **Stage 2 — Executor** (Section 5) validated with the GROUND-TRUTH target (from banked solve) as a
  positive control. Gate: toggle-to-match search solves >=3 config games when HANDED the true target.
  This isolates "can we execute given a target" from "can we induce the target" — and proves the back
  end before investing in induction. Kill: even with the true target the search cannot solve (the
  toggle mechanic is more complex than recolor) -> the executor itself needs research first.
- **Stage 3 — Induction** (Section 4.1 primary + 4.2). Gate: the induced target (no banked solve)
  lets the executor solve >=2 config games first-contact, beating the generic baseline (0). Kill:
  LLM-scene-reader + reference detection both fail to induce a usable target on any game -> config is
  a per-game / learned-model problem, documented as such.
- **Stage 4 — Generalize + scorecard.** Re-run `arc3_full_pass_scorecard.py`; report the new solve
  rate and which target-class each newly-solved game belonged to.

## 7. Success and kill criteria

- **Success:** config first-contact solves go from 0 to >=3 of 9, via induced (not hand-coded)
  targets, with the new rate on the re-runnable scorecard. A clean partial (e.g. class-(a) games
  solved, class-(c) deferred) is success with a sharpened residual.
- **Kill:** Stage 2 fails (executor can't solve even given the true target) OR Stage 3 induction
  yields no usable target on any game. Either kill re-scopes config to a learned-model / per-game
  research track and the generic-solver effort stops (per the no-doomed-rerun discipline).

## 8. Reuse and cross-references (no parallel edifice)

- Executor: `python/carnot/agentic/arc_heuristic_search_over_verified_wm.py` (`best_first_search`).
- Toggle/recolor model: `python/carnot/agentic/arc_world_model_dsl.py` (M2-v2 `recolor_clicked`).
- Pipeline + scorecard: `scripts/experiments/arc3_test_time_goal_induction.py`,
  `scripts/experiments/arc3_full_pass_scorecard.py`.
- LLM proposer (offline GGUF): `arc_executable_world_model.LocalGGUFProposer` (the same machinery the
  E3 world-model induction uses; run on the AMD iGPU per the 2026-06-17 directive, not the 3090s).
- Banked targets (Stage 1/2 ground truth): `ops/arc_solve_registry.yaml`, the metaharness.
- Measured baseline: `results/arc3_config_paradigm_probe.json`.
- Prior art: tr87 GameAdapter (rule-rewrite, the class-(b) exemplar);
  `reference_arc_agi3_sota_and_plan` (LLM-reads-the-puzzle is the SOTA approach).

## 9. Effort and sequencing note

Stages 1-2 are cheap (taxonomy + executor-with-ground-truth, no LLM, CPU). Stage 3 is the real
research (LLM-scene-reader). Do Stages 1-2 first: they are decision-grade and either prove the back
end (de-risking Stage 3) or kill the whole direction early. This respects the phase-prototype +
empirical-validation discipline: prove the executable prototype at small scale (handed the target)
before investing in the hard induction.
