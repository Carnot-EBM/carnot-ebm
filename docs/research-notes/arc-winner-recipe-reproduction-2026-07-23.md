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

## Discovery A/B (5834): the perception fix flips the goal but not discovery -- execution is the next wall

Wired `reauthor_framing` into the live greedy-direct agent and ran the full winner recipe reauthor OFF vs ON,
same budget(120)/seed, all 8 games (`results/outer_loop_arc_reauthor_discovery_sweep_20260723.json`).

**Result: 0 levels discovered, both arms, all 8 games.** The perception fix flips the goal HYPOTHESIS
(0->4/8 offline) but does NOT convert to actual DISCOVERY.

This is the honestly-predicted separation, now measured: perception/goal-induction was necessary (and is
fixed), but it is NOT sufficient -- **execution/dynamics is the next binding constraint.** The action counts
show why: bp35 loses at exactly 64 actions (the move-budget cap) on both arms; the greedy-direct architecture
has no search/planning, so even with the right goal "move the player to the exit," the LLM has to blind-guess
the action sequence over a hex grid -- which it cannot do within budget.

**The earned strategic redirect.** The leaderboard-winner architecture we were asked to match (greedy-direct,
LLM picks each action, no search) is EXECUTION-LIMITED even with a correct goal. Carnot's OWN live
architecture -- `E3AgentPolicy`'s verifier-routed best-first search + `plan_in_model` -- is better suited for
execution precisely because it can SEARCH toward a named target. So the next step is not to keep polishing the
greedy-direct winner recipe; it is to feed the perception fix (retracted HUD rules + named target + player)
into Carnot's verifier-routed search as the goal/heuristic, where the search can navigate to the target and
the goal verifier confirms progress via the level counter. The winner-matching investigation ends by pointing
back at Carnot's search-based strength -- with a validated perception front-end to give that search a correct
target.

### The complete measured chain (winner-matching investigation, 2026-07-23)

1. Winner recipe (31B + greedy-direct + segmentation + reflection) -> **0 discovery**
2. Diagnosis -> **perception is the wall** (8/8 never hypothesized the true goal)
3. Oracle ablation -> fixing perception flips goals **0 -> 7/8**
4. Detectors (HUD + mover, from frames) -> recover the confuser-HUD + navigate avatars
5. End-to-end plain -> **0/8** (detector output alongside the wrong rules doesn't flip it)
6. Re-authoring -> goal gate **0 -> 4/8**, zero HUD fixation; wired into the live agent
7. Discovery A/B -> **0/8 both arms** -> perception necessary, not sufficient; **execution/dynamics is next**;
   feed the perception fix into Carnot's verifier-routed search (not the greedy-direct winner recipe)

## E3 search A/B (5835): routing the perception target into Carnot's search is also a null

Executed the redirect the discovery null pointed to: fed a perception-derived target (nearest non-HUD object
to the detected player, from a perceive_entities recon) into Carnot's LLM-free tier-1 explorer as a
player->target Manhattan value_head, routed (value_weight=1, best_first, navigation_cost_tiebreak=False), A/B
vs the default explorer, 8 games, scored by run_game.

**Result: 0 discovery, both arms, all 8 -- robust across search mode** (depth_first_ride vw3 also 0). The
value_head engages (instrumented ~41 calls/run, sane distances) but the player->target distance hovers at ~9
and never decreases: the search does not navigate the player to the target.

**What this bounds.** Combined with the greedy-direct null (5834): the perception fix is validated
(goal-framing 0->4/8 offline, 0->7/8 oracle) but NEITHER greedy-direct NOR value-head-routed search converts
it to a solved level out of the box. Execution/navigation is a genuinely hard, separate problem -- not cracked
by routing toward a Manhattan target. Likely unaddressed: the recon target may not be the true exit
(nearest-object heuristic); the search may not find the navigation path within budget even with a correct
target; the perception arm only applied to 3/8 games. The honest end state: the perception-fix program moved
the wall from perception/goal-induction to execution/navigation, and the execution wall stands.

## Full isolation (5836): the execution wall is perception-grounded verifier-routed pathfinding

Source extraction resolved the E3 null. Of the 4 games probed, only tu93 is navigation_only (player color-9
onto exit color-14); sc25 needs spell-casting, ls20 attribute-morphing, cn04 endpoint-pairing. So the E3
perception-target A/B ran on 3 NON-navigation games -- no player->exit heuristic can solve them -- and skipped
the one navigation game because detect_mover's fixed max_shift=4.0 rejected tu93's player (it jumps ~6 logical
cells/action). Fixed: max_shift is now relative (0.25*min(H,W)); tu93 detects a mover again.

On tu93 with source-confirmed colors: the E3 value-head-routed explorer moves the player 60->12 (real
navigation progress) but discovers 0 levels; a direct greedy-toward-exit navigator goes 60->42 then hits
game-over at move 26. tu93 is a MAZE (color-2 rails) with a move budget -- neither the novelty-explorer nor
greedy does obstacle-aware pathfinding within budget.

**The key.** Carnot's OfflineSolver (verifier-routed best-first search over replay-from-reset) ALREADY solves
tu93 to L5 (registry) using a hand-built GameAdapter's player->goal Manhattan verifier. The perception fix
provides AUTONOMOUSLY exactly what that adapter provides MANUALLY. So the execution wall is precisely located
and the next capability is scoped: feed the perception-detected player->target as the OfflineSolver verifier
(the mechanism that already solves navigation games), not the weaker E3 novelty-explorer.

### Complete measured chain (winner-matching -> execution isolation)

1. Winner recipe -> 0 discovery. 2. Perception is the wall (8/8). 3. Oracle ablation: fix flips goals 0->7/8.
4. Detectors recover HUD+mover from frames. 5. Re-authoring: goal gate 0->4/8, zero HUD fixation; wired live.
6. Discovery A/B: 0/8 -- execution is the next wall. 7. E3 routing A/B: 0/8. 8. FULL ISOLATION: 3/4 tested
games aren't navigation games; the one that is (tu93) is a maze+budget needing obstacle-aware pathfinding;
Carnot's OfflineSolver already solves it with a hand-built verifier -- the scoped fix is an AUTONOMOUS
perception-grounded verifier feeding that existing search.

## Chain PROVEN end-to-end (5837): autonomous perception -> verifier-routed search -> reproduced discovery

Built detect_static_target + derive_navigation_pair (player = mover; goal = small static marker). On tu93 the
autonomous perception layer derives (player=9, goal=14) FROM MOTION ALONE -- identical to the adapter's
hand-RE'd pair, never hardcoded, never read from source. Fed that perception-derived verifier into Carnot's
OfflineSolver verifier-routed search (reusing the adapter's action/apply/state machinery + the real
reproduction gate):

- hand verifier:       reached L3, reproduction gate PASSED (reproduced=True), 47-move path
- **perception verifier: reached L3, reproduction gate PASSED (reproduced=True), 47-move path -- IDENTICAL**

**This is the first actual DISCOVERY of the whole thread.** The perception fix that nulled on greedy-direct
and on the E3 novelty-explorer WORKS the instant it feeds the proper verifier-routed search that already
solves navigation games. Method validation (not a new solve claim -- tu93 L3 is already in the registry;
solve_provenance=development_proxy). The complete, closed chain:

1. Winner recipe -> 0 discovery. 2. Perception is the wall (8/8). 3. Oracle: fix flips goals 0->7/8.
4. Detectors recover HUD+mover from frames. 5. Re-authoring: goal 0->4/8, wired live. 6. Discovery A/B: 0/8
(execution wall). 7. E3 routing A/B: 0/8. 8. Full isolation: Carnot's OfflineSolver already solves tu93 with
a hand verifier. 9. **Perception-grounded solve: the AUTONOMOUS perception verifier reproduces tu93 L3 ==
the hand verifier.** The perception front-end now provably feeds Carnot's search a correct, autonomous target
that closes to a reproduced solve.

**Next (scoped):** wire derive_navigation_pair into the LIVE E3 path as the navigation-game verifier
(replacing the per-game hand-built GameAdapter verifier), and generalize player/target derivation beyond the
clean 2-color case.

## Autonomous generic-adapter solve (5838): tu93 L3 with ZERO per-game hand-RE

Closed the last gap. Solved tu93 with a FULLY GENERIC navigation adapter -- no per-game code (generic
directional-move actions, env.step, full-grid state key) + the perception-derived verifier (colors (9,14)
from motion). Result:

- fresh_env: **reached L3, reproduction gate PASSED (L3), 47-move path -- identical to the hand-built adapter**
- replay: reached L2 in search, gate correctly REJECTED (tu93's non-idempotent reset -> no false claim)

The only remaining per-game input is branch_mode (fresh_env vs replay), an auto-detectable reset-idempotency
property. **This is the live self-discovery capability the whole thread targeted: Carnot solves a navigation
game from its own frames with no per-game reverse-engineering.**

Remaining (scoped): auto-detect branch_mode; generalize player/target derivation beyond the clean 2-color
case; and the SCORED-path wiring -- give E3AgentPolicy the OfflineSolver-style verifier-routed search for
navigation games (its novelty-explorer only navigates partway, so the fix is the search structure + the
autonomous verifier together).

### The whole thread, one line

We matched the leaderboard leaders (0 discovery), proved perception is the wall, fixed it from frames
(0->7/8 goals), found the execution wall, and closed it: an AUTONOMOUS perception verifier + Carnot's
verifier-routed search reproduces tu93 L3 with no per-game hand-RE -- the sovereignty-aligned, oracle-distinct
path, not the winner recipe.

## Scored-path integration COMPLETE (5839): the scored agent self-discovers tu93 L3

Built `arc_perception_navigation.py`: the full autonomous pipeline (recon -> derive (player,goal) from motion
-> SELF-CORRECTING branch_mode -> generic verifier-routed OfflineSolver -> gate) + `PerceptionNavigationPolicy`
that carries the self-discovered solve behind the scored run_game interface (plan-then-replay, self-discovered
not hand-banked).

- solve_navigation(tu93): auto_branch_mode guessed 'replay'; the gate showed reproduced < searched, so it
  self-corrected to 'fresh_env' -> **reached L3, reproduced=True, 47-move path. Fully autonomous, ZERO per-game input.**
- scored run_game: default explorer reached 0; **PerceptionNavigationPolicy reached L3 (levels=3).**

The SELF-CORRECTION is the key robustness idea: don't predict reset-idempotency perfectly (a short no-win
probe can't see tu93's win-contingent parity); VERIFY -- the reproduction gate is ground truth, so try a mode
and keep whichever reproduces more.

Remaining (architectural): the search runs on the offline arcade in the policy __init__ then replays; 'search
live on the scored env during play' is the next step. Method validation (tu93 L3 registered).

### The complete thread (winner-matching -> scored self-discovery)

10 measured steps, closed: matched the leaders (0 discovery) -> perception is the wall (8/8) -> oracle proof
(0->7/8) -> detectors -> re-authoring (goal 0->4/8) -> discovery A/B 0/8 (execution wall) -> E3 routing A/B
0/8 -> full isolation (Carnot has the engine) -> perception-grounded solve (autonomous verifier reproduces
tu93 L3) -> generic-adapter solve (no hand-RE) -> **scored-path integration (PerceptionNavigationPolicy
reaches L3 via run_game, fully autonomous).** The sovereignty-aligned, oracle-distinct path -- not the winner
recipe -- self-discovers and reproduces a navigation solve on the scored interface.

## Live-search boundary (5840): reaches tu93 L1 live; deep live solving is an open problem

The plan-then-replay policy (5839) searched a SEPARATE offline arcade twin -- unavailable for a hidden game.
The true live step is to search the ONE env run_game gives (StepwiseExplorer via CarnotAgentPolicy, no
separate arcade). Measured with the perception verifier:

- tu93 (budget 2000): baseline explorer AND perception-routed both reach **L1**; NEITHER reaches L2/L3.
- g50t: both 0 (not pure navigation).

**Honest boundary.** Live single-env search reaches tu93 L1 -- limited live self-discovery on the scored env
IS achievable. But deep solving (L2/L3) is blocked because the offline L3 solve relied on
branch_mode='fresh_env' (a brand-new env per search node), which a single live scored env cannot supply
(tu93's reset is non-idempotent -- a parity toggle). The perception verifier adds no advantage over the
baseline explorer online (both L1); its proven value is in the OfflineSolver BATCH search (5837/5838/5839),
not the online explorer. And the public set has no clean idempotent-reset navigation game for a positive
live-search-beats-baseline demo.

**So:** the autonomous perception->verifier->search chain fully closes on the OFFLINE/dev-twin path (tu93 L3,
no hand-RE, what arc_loop_solve uses), but deep TRUE-LIVE scored solving of a non-idempotent-reset maze is a
genuine OPEN research problem -- a parity-aware live search, or a world-model rollout that branches without
resetting the real env -- not a wiring gap. That is the honest boundary of the whole perception-fix program.

## Dig into induction quality (5841): a plan_in_model regression + a structured-nav-model live solve

Digging into the 2026-07-20 induction-quality diagnosis's open question #6 ("why does even a PERFECT induction
execute to 0 real level-up with plan_len=1?"), the outer loop found the concrete cause: a REGRESSION.

**The bug.** `arc_graph_explore._components_detailed` was widened from a 4-tuple (cy,cx,area,color) to a
5-tuple (+is_grid_fallback) in commit 2f0760307 (GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS). That fix
updated the arc_graph_explore consumer defensively but MISSED plan_in_model's `_model_candidates`, whose rigid
`for cy,cx,_a,_c in comps` unpack then raised ValueError on ANY grid with components (tu93 has 65). The
live/harness call sites catch the exception, so it SILENTLY disabled the entire plan_in_model world-model
planning tier for every object-bearing game. (The diagnosis, dated 07-20, predates the regression.)

**The fix.** Defensive `for cy, cx, _a, _c, *_ in comps` unpack + a corrected docstring + a regression test.

**The downstream win.** With the fix AND a STRUCTURED nav world model (`InducedNavWorldModel` -- correct by
construction for the 4-direction navigation family, fitting per-action displacement + avatar + goal from the
agent's own transitions; the "mechanic-class prior" the diagnosis §6 flagged as highest-leverage, NOT the
near-universally-wrong LLM induction), `plan_in_model` finds an **18-action navigation plan that reaches a
REAL tu93 level-up (hv 60->6), reproducible 3/3.**

**Why this matters for live search.** plan_in_model plans IN IMAGINATION then executes ONCE from reset -- no
per-node resets -- so it SIDESTEPS tu93's non-idempotent-reset blocker that defeated the
OfflineSolver/StepwiseExplorer live search (5840). This is a **live-compatible** navigation solve, and it
uses the structured inducer the diagnosis pointed at.

**The concrete next unblock.** `InducedNavWorldModel` is currently ORPHANED from the live E3 path (imported
only by scripts/tests). Route navigation games (detected via `derive_navigation_pair`) to it instead of the
LLM induction, and feed its engine+is_level_complete to the live plan_in_model tier -- giving E3 a correct
model for nav games. That is the highest-leverage lever the 2026-07-20 diagnosis identified, now with a
working end-to-end proof and a same-day planner regression removed from its path.
