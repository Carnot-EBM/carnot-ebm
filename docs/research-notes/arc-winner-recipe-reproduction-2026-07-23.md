# Reproducing the ARC-AGI-3 leaderboard-winner recipe (2026-07-23) — it learns dynamics but induces the wrong goal

**Status:** RESEARCH RESULT. Operator directive: "the leaderboard top projects are using the 31B and 27B
models. we should switch to match them." Chose "match model + architecture" with gemma-4-31B, then built
the winners' recipe faithfully in three stages and tested it against the REAL metric (live discovery,
adapter-free) on the three worst stalled games. All artifacts adversarial-verify clean.

**Reads as required input (per CLAUDE.md):** "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent",
"ARC Live-Path Reachability Discipline". Companion to `arc-top-project-search-architecture-audit-
2026-07-20.md` (the winner audit) and `arc-lever-triangulation-2026-07-23.md` (the binding-constraint
triangulation this result confirms).

## What was built (the four winner mechanisms, faithfully)

`python/carnot/agentic/arc_greedy_direct_agent.py` (REQ-ARC-WMTE-5829) — a Duck-Harness-style
greedy-direct agent: the LLM picks each action directly and commits it to the real env, NO search, NO
induced world model (the winners' actual architecture; NOT our induce-then-plan, which already nulled
with a 31B inducer in `experiment_5722`). Staged, each committed with tests:

1. **v1 — gemma-4-31B + greedy-direct + raw-grid perception.** Model + architecture matched.
2. **v2 — + object segmentation** (the audit's #1 "highest-value steal"; the model reasons over discrete
   objects, `arc_color_blob_salience`). Plus a correct logical→raw click-coordinate mapping (fixing a
   latent bug in the earlier tool-loop).
3. **v3 — + persistent reflection memory** (Reki/forge's mechanism: the model periodically rewrites an NL
   notes doc — RULES/GOAL/PROGRESS/AVOID — from accumulated play, re-injected into every decision).

Completion path: raw `/completion` + fence-priming (smoke-verified for gemma-4-31B; raw-unprimed derails
into gemma's reasoning channel, the chat-template endpoint returned empty).

## Results — all null on discovery, but v3 reveals WHY

`results/outer_loop_arc_winner_greedy_direct_ab_{20260723 (v1 grid), v2objects_, v3reflect_}.json`,
bp35/lf52/sc25, budget 120, adapter-free E3-substitute:

| version | mechanisms | discovered levels | behavior |
|---|---|---|---|
| v1 | 31B + greedy-direct + raw grid | 0 / 0 / 0 | fixates on clicking one coordinate cluster |
| v2 | + object segmentation | 0 / 0 / 0 | varies clicks across objects (lf52) but no discovery |
| v3 | + reflection memory | 0 / 0 / 0 | **learns real dynamics, pursues systematic goal hypotheses** |

**v3's reflection memory WORKED — a qualitative jump.** The model went from blind fixation to genuinely
reverse-engineering mechanics and forming+pursuing goal hypotheses, exactly as the winners do. Its actual
learned notes:
- **bp35:** "RULES: Action 6 sets cell (63,y) to 15. GOAL: Fill column 63 with 15s. PROGRESS: Column 63
  indices 0-62 are filled with 15s." — learned a real rule, formed a goal, and EXECUTED it to completion.
- **lf52:** "RULES: Action 6 sets cell (0,z) to state 1. GOAL: Set all grid cells to state 1. PROGRESS:
  Cells (0,0)-(0,59) are state 1."
- **sc25:** detailed reverse-engineered toggle mechanics across specific 2x2 blocks.

**It still nulls because the GOAL HYPOTHESES ARE WRONG.** gemma-31B induces plausible-but-incorrect win
conditions ("fill column 63", "set all cells to state 1"), pursues them systematically, and completes
them — but they are not the actual level-up condition, so no level fires.

## The conclusion (why this matters)

The winners' **full visible recipe** — the 27-31B model, the greedy-direct architecture, object-
segmentation perception, and reflection memory — reproduced faithfully with a 31B, hits the **exact same
wall** the 2026-07-23 lever triangulation pinned as the binding constraint: **GOAL INDUCTION** (part of
world-model-induction-grade reasoning). Model size + greedy architecture + segmentation + reflection do
NOT clear it on these games. This is consistent, not contradictory, evidence:
- `experiment_5722`: a 31B AS AN INDUCER in our architecture → 0 levels (dynamics induction too weak).
- This result: a 31B in the WINNERS' architecture → learns dynamics fine, but induces the WRONG GOAL → 0.

Both point at the same root: the agent cannot reliably induce **what winning looks like**. Perception,
search, action-generation, model size, and reflection are all necessary-not-sufficient; the goal predicate
is the wall.

**This points squarely at Carnot's own thesis** (`ops/verifier_gaps.md` `GAP-ARCH-GOAL-NOT-VERIFIED`): a
goal VERIFIER that checks a hypothesized goal against observed transitions and REJECTS a wrong one — e.g.
"does 'fill column 63' actually correlate with a level-up? No → discard that goal hypothesis and keep
exploring." The winner recipe supplies a fluent goal-HYPOTHESIZER (the reflection memory); what it lacks,
and what Carnot's verification-first thesis is built to add, is the goal-VERIFIER that keeps the
hypothesizer honest. This is the evidence-backed next lever — and it is oracle-distinct, sovereignty-safe,
and directly on Carnot's core value proposition.

## Honest caveats (do not overclaim)

- **Budget.** 64-120 actions/game (bp35/lf52 hit `game_over` at ~64). The winners run much longer per game
  (12h across all games). A larger budget was not tested here; a bigger budget lets the WRONG-goal pursuit
  run longer, so it is not obviously the fix, but it is untested.
- **These 3 games may be hard.** The winners score on the PUBLIC leaderboard, likely concentrated on games
  with more inferable goals; and even the winners are <0.4% on the HIDDEN set. bp35/lf52/sc25 have
  non-obvious win conditions gemma mis-induces. A broader game sweep was not run (deliberately — the goal-
  induction diagnosis is the same regardless of which specific game it manifests on).
- **Remaining faithfulness gap.** Duck's full sandboxed Python REPL (vs our constrained tool API) is the
  one winner mechanism not reproduced. It gives the #1 winner more flexible in-loop computation, but it
  does not supply a goal VERIFIER either — it is a richer hypothesizer, not a corrector, so it is unlikely
  to change the goal-induction conclusion.

## Cross-references

- REQ-ARC-WMTE-5829 (`openspec/capabilities/arc-world-model-trust-energy/spec.md`) — the agent spec.
- `python/carnot/agentic/arc_greedy_direct_agent.py` + `scripts/arc_winner_greedy_direct_ab.py` +
  `tests/python/test_arc_greedy_direct_agent.py` (22 tests).
- `results/outer_loop_arc_winner_greedy_direct_ab_{20260723,v2objects_20260723,v3reflect_20260723}.json`.
- `docs/research-notes/arc-lever-triangulation-2026-07-23.md` (the binding-constraint diagnosis this
  confirms); `arc-top-project-search-architecture-audit-2026-07-20.md` (the winner audit).
- `ops/verifier_gaps.md` `GAP-ARCH-GOAL-NOT-VERIFIED` (the goal-verifier lever this points to);
  `experiment_5722` (31B-as-inducer null).

## Decisive re-diagnosis (REQ-ARC-WMTE-5831): the wall is UPSTREAM of goal induction

After the goal verifier (REQ-ARC-WMTE-5830) + the full winner recipe still nulled on an 8-game sweep
(bp35/lf52/sc25/ls20/tu93/cn04/r11l/ft09, adapter-free, budget 120 —
`results/outer_loop_arc_winner_greedy_direct_ab_v5sweep8_20260723.json`, 0 discovery on all 8), I ran the
decisive test: 8 independent analyst agents each READ the game source (offline dev analysis, authorized per
the public-games source-reading discipline; never used in the hidden submission) to extract the TRUE win
condition, then compared it against the exact goal-hypothesis trajectory gemma-4-31B produced.

**Question:** did gemma-4-31B ever hypothesize the true win condition for any game?
**Answer: 8/8 NO** (0 yes, 0 partial). `results/outer_loop_arc_goal_hypothesis_coverage_diagnosis_20260723.json`.

| game | true win condition (from source) | model's fixated goal | root cause | budget helps? |
|------|----------------------------------|----------------------|-----------|---------------|
| bp35 | land the player on the gem (gravity platformer) | "fill column 63 with 15s" (= the move-budget LOSE bar) | dynamics-not-learned | no (64 acts → lose) |
| lf52 | reduce colored blocks to 1 via select-then-aim merge | "trigger changes outside row 0" (= the HUD bar) | perception | no |
| sc25 | navigate the wizard onto the exit tile | "clear values at col 62/63" (= the mana meter) | perception | no |
| ls20 | morph avatar's (shape,color,rotation) to match a goal pad, then stand on it | "Undetermined" | perception | no |
| tu93 | navigate the token onto the exit block | "clear all value-6 cells" (= the step-counter, the LOSE cond) | perception | no |
| cn04 | spatial endpoint-pairing after move/rotate | "clear cells of color 4/12/14" (decoy/background colors) | perception | no |
| r11l | slide each object onto its matching pad | "paint cells to value 5" | dynamics-not-learned | no |
| ft09 | hidden adjacency-color CSP in per-tile 3×3 pixels | "Unknown" (abandoned its only lever as "ineffective") | dynamics-not-learned | no |

**The pivotal implication.** The goal-hypothesis generator is NOT the first failure. Several true goals are
canonical, LLM-plausible goals — "navigate the avatar into the exit" (sc25, tu93), "slide each object onto
its matching pad" (r11l, ls20) — that a text LLM reasoning over a proper object list WOULD plausibly
hypothesize. It never did, because its perception fixated on the HUD/budget-bar and never *represented* the
player avatar, the exit tile, or the target blocks. 5/8 = perception; 3/8 = dynamics-not-learned; 0/8 helped
by budget. In bp35 and tu93 the budget bar it dutifully fills IS the lose timer, so its fixated goal is
literally the anti-goal.

**What this means for the goal verifier.** REQ-ARC-WMTE-5830 is correct and necessary (it demonstrably
falsifies the exact wrong goals the winner recipe fixated on), but it cannot manufacture a right goal when
the right goal is structurally unreachable given broken perception — falsifying wrong goals just walks the
model across a hypothesis space that does not contain the answer.

**Next ARC lever (evidence-backed, corrected).** NOT more goal machinery, more budget, or the verifier. It
is: **(1) HUD/budget-bar vs game-board perceptual disambiguation** (mask the every-action monotone register;
it is maximally salient and least relevant), and **(2) action-effect dynamics induction** (find the connected
component that translates under directional actions — that is the player token — and register it as a
first-class entity the goal-hypothesizer must reference). Filed as `GAP-ARC-PERCEPTION-HUD-VS-BOARD` in
`ops/verifier_gaps.md`. This independently corroborates `project_arc_live_agent_learning_gaps` (perception is
the binding constraint) — now confirmed to still hold at 31B + the full leaderboard-winner recipe.
