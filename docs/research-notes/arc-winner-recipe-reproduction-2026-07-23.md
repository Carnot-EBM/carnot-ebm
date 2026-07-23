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

## Counterfactual: oracle perception flips goal induction from 0/8 to 7/8 (REQ-ARC-WMTE-5832)

Before building perception detectors, I tested whether the named lever pays off. The oracle-perception
ablation hands the same gemma-4-31B the *correct entities* (what a perfect detector would output — presence
and role only, never the goal) and measures whether its goal hypothesis flips. Same model, same learned
RULES per game (held constant — testing perception, not rule-learning), three conditions: A = naive (only
the HUD/decoy the segmentation actually surfaced), B = +true entities (HUD not labeled), C = oracle (entities
+ HUD explicitly identified as a non-goal counter). Result
(`results/outer_loop_arc_oracle_perception_goal_ablation_20260723.json`):

| condition | correct | partial | wrong |
|-----------|---------|---------|-------|
| A naive   | 0/8     | 0       | 8/8   |
| B +entities | 6/8   | 1 (lf52)| 1 (bp35) |
| C oracle  | **7/8** | 1 (lf52)| 0     |

Examples (A → C): tu93 "consume all value-6 cells" (the LOSE condition) → **"move the player token to the
exit block"** (exact); ls20 "recolor floor 3/5→12" → **"move the avatar to a goal-pad while matching its
required shape, color, and rotation"**; bp35 "fill row 63 with color-15" (the anti-goal) → **"reach the
gem"**; r11l/ft09 (the dynamics / hypothesis-coverage contrast cases) also flipped to their true goals.

**Conclusion: the goal-hypothesis generator is not the bottleneck.** A fluent 31B produces the true win
condition the instant perception delivers the right entities. The entire wall is upstream perception —
exactly as the source-grounded diagnosis said.

**Both halves of the fix are load-bearing.** bp35 flipped only under C, not B: adding the player/gem entities
was *not* enough; you also had to identify the row-63 bar as a lose-timer. So the detector work is two
pieces: (a) a **mover/entity detector** (surface the player token — the component that translates under
directional actions — and the distinct target/exit tiles), and (b) a **HUD-register detector** (mask the
monotone every-action band so it stops dominating perception).

**Honest scope.** This measures goal-hypothesis correctness, not level discovery. The right goal is necessary
but not sufficient — the agent still has to *execute* it, which needs the dynamics/action model (bp35 gravity,
r11l two-click select-then-move remain execution gaps even with the correct goal). Single temperature (0.3),
2 seeds, goals highly stable; small n (8 games). Oracle facts were hand-authored from source (offline dev,
authorized; never in the hidden submission) — a real detector must produce them autonomously from frames,
which is the build this ablation justifies.

**Next build (evidence-backed):** the mover/entity detector + HUD-register detector on the live E3 path,
gated by re-running this goal-hypothesis measurement under *detector-produced* perception (not hand-authored)
to confirm it recovers the ~7/8 goal-correctness.

## Detectors built (5833) and the end-to-end gate (5834): an honest negative that sharpens the fix

**Detectors (REQ-ARC-WMTE-5833).** Built two detectors that run off the agent's own transitions: a
HUD-register detector (edge band changing on near-every action + position-independent = a counter) and a
mover detector (the color whose centroid translates with directional actions = the player). Real-frame gate
(offline arcade, no LLM): HUD recovered on 3/3 confuser games (bp35 row63, lf52 row0, tu93 row63 -- the exact
bars the model mistook for the goal), mover recovered on the navigate avatars (sc25/ls20 color-9). A key fix
surfaced only by real frames: the first HUD statistic (monotone background-count) missed every real counter
because ARC counters change cell VALUES, not fg/bg count -- replaced with change-fraction +
position-independence.

**End-to-end gate (REQ-ARC-WMTE-5834) -- honest NEGATIVE.** Fed the DETECTOR-produced perception to gemma's
goal prompt (same rules as the 5832 oracle ablation; only the perception source changed). It did NOT recover
the oracle's 7/8: **0 correct, 2 partial, 6 wrong.** The two partials (ls20, cn04) are the games where a
mover was detected, and both produced the right SHAPE -- "move the player to a target" -- which is the
encouraging signal. But the six wrong show why detector perception alone is insufficient:

1. **The model's own WRONG learned rules dominate.** bp35's rule "action 6 changes (63,x) to 15" and lf52's
   "actions change row 0" are exactly the HUD-fixation, and gemma follows the rule even when the detector
   correctly flags that band as a counter to ignore. An "ignore this band" note is weaker than a rule the
   model already believes.
2. **The detector names the player but not the target.** It lists candidate objects; it cannot say which is
   the exit. So even mover-games stay vague.
3. **Mover recall gap** -- missed on 4/8 (tu93/bp35/lf52/r11l), leaving no player concept.

**The lesson.** The oracle got 7/8 because it REPLACED the framing ("the bar is a lose-timer, NOT the
objective" + named the exit). The detector ADDS entities alongside the wrong rules instead of correcting
them. So the perception fix must RE-AUTHOR, not augment: (a) retract any learned rule that references a
detected HUD band; (b) propose the mover's nearest distinct object as the candidate target; (c) improve mover
recall. That is the concrete next iteration -- and it is a sharper, better-grounded target than "add a
perception detector," which this negative result earned.

## Re-authoring fix (5834): detector perception, corrected framing -> 0/8 to 4/8, zero HUD fixation

The negative said the fix must REPLACE the framing, not augment it. `reauthor_framing` does exactly that:
retract any learned rule referencing a detected HUD band, name the mover's nearest object as the candidate
target, and override the counter-fixation. Re-run the end-to-end gate with everything else identical:

| condition | correct | partial | wrong |
|-----------|---------|---------|-------|
| plain detector perception (5834) | 0 | 2 | 6 |
| **re-authored (5834 fix)** | **4** | 2 | 2 |
| hand-authored oracle (5832) | 7 | 1 | 0 |

Every game flipped from "fill/change the counter" to "move the player to a target." Correct on the 4
navigate/move-to-target games (sc25 "move PLAYER to the TARGET at (20,14)"; tu93; bp35 "reach the gem";
r11l "move the piece to a target pad") -- **including the 3 where no mover was detected, because retracting
the wrong HUD rule alone was enough to stop the fixation.** ZERO games still fixate on the HUD (the plain
gate had 6). Two partials (ls20 missing the morph, cn04 vague pairing).

**Residual + next step.** It OVERCORRECTS the 2 non-navigate games (lf52 block-merge, ft09 hidden-CSP) into a
navigate frame, because the block asserts "move player to a target" even when no mover was detected. The fix:
gate the player-target assertion on mover-detected; for no-mover games use a neutral "arrange the game
objects; the counters are not the goal" frame. Then wire `reauthor_framing` into the live `E3AgentPolicy`
perception + the greedy-direct reflection, and add the execution/dynamics half.

**The complete chain.** diagnosis (perception is the wall, 8/8 never hypothesized) -> oracle proof (correct
entities flip 0->7/8) -> detectors (recover HUD/mover from frames) -> re-authoring (0->4/8 correct from
frames alone, no HUD fixation). Each step a real measurement that earned the next.
