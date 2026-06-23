# Rapid SOTA-extraction levers for the 1-week ARC submission push (outer-loop, 2026-06-23)

Operator asked: with the top leaderboard projects cloned locally, what can we research *rapidly* in the
outer loop for a marked live-agent improvement before the 2026-06-30 deadline? 17-agent workflow
(`arc-sota-rapid-accel`) mined the two cloned SOTA solvers for un-embraced, **first-win-targeting**,
rapidly-testable techniques, with a ruthless "already-embraced / already-dead-this-week" filter. Shim
feasibility independently verified against the code.

## The wall, and what's left
The binding wall is **exploration-to-first-win on a hidden game** (reaching the first state-change /
level-up). We have already absorbed the *surface* of both SOTA agents: frame-hash GameGraph,
salience-ordered click candidates, the frame-change CNN expansion-prior (`.427`/`.428` graft),
shortest-path frontier navigation, an online on-path discriminator. Four levers already died this week
on the actual code: macro-vocab depth, click-heatmap pixel-precision (falsified on human replays —
0.9% off-object), trust-gate flip (provable no-op), TTT-route (blocked by this same wall). The `.428`
goal-energy generation lever (exp4640/exp4644) also landed as honest nulls today. So the question is
narrow: what *mechanism detail* inside the embraced surfaces is genuinely un-embraced, targets
first-win, and is offline-testable in days.

## #1 — RUN FIRST (decisive, gating, ~1.5 days): head-to-head levels-reached table
Run just-explore's `HeuristicAgent` (pure numpy, no GPU) and StochasticGoose's CNN agent (torch/CUDA)
against Carnot's `StepwiseExplorer` on our 25 offline games at a fixed budget → a per-game
levels-reached gap table. **A measurement, not a mechanism — and the most decisive thing we can produce
rapidly.** Feasibility verified: just-explore's `Agent(ABC)` (`agents/agent.py`) executes actions at one
HTTP seam (`_session.post`, line 146); shim = subclass, override that seam to call our offline
`env.step(GameAction, data={x,y})`, and **map our `levels_completed → their FrameData.score`** (our
frame has `levels_completed`, not `score`). Their explorer core runs unchanged (~150 LOC shim).
- **Blocking preflights:** (a) shim-validity — run the shimmed agent on a game we bank (lp85 L1), assert
  `.score` increments at level-up, else `blocked_shim_score_unmapped`; (b) best-of-3 seeds/game
  (just-explore's published 17-median is a 7.9h multi-restart swarm; a single 1000-action run *deflates*
  it → frame the gap as a LOWER BOUND, don't hand Carnot a noise win).
- **Falsifiable gate:** just-explore reaches a strictly deeper level than Carnot on ≥3 of 25 games
  (Σ deltas > 0) → the un-embraced exploration techniques are validated, extract them. Ties/loses on all
  25 → the exploration schedule is NOT the wall (it's induction/goal-guidance) → **stop mining**, conserve
  the week. *This single measurement collapses the week's risk.*

## #2 — cheap, but EXPECT NULL (~0.5 day): multi-pixel interior click
Emit an on-object interior pixel (grid color == component color) instead of the bare centroid, for the
hollow/ring-object case where the centroid pixel is background (guaranteed no-op). Source:
`heuristic_agent.py:367-371`. Correction: `object_centric_digest` computes `cells`
(`arc_solver_kit.py:4085-4099`) but doesn't return the mask — expose it first (~16 LOC),
`CARNOT_ARC_CLICK_INTERIOR=1`. **Honesty flag:** the centroid-coverage premise was *falsified this week*
(`click-heatmap-generator-falsified-2026-06-23.md`, 99.1% covered) — this survives ONLY as the hollow-
object sub-case the falsifier didn't isolate. 0.5 days to confirm it's (probably) dead; do not over-invest.

## #3 — if time (<1 day): tier-deferred candidate exhaustion
Bucket candidates into tiers (T0 = salient color ∈ {6..15} AND medium-width 2≤w,h≤32) and emit
tier-ascending instead of one flat `area*(1+1/color_rarity)` sort (which up-ranks the giant background
blob). Source: `heuristic_agent.py:866-898`. v1 = intra-node reorder + medium-width band only
(`CARNOT_ARC_TIER_SCHEDULE=1`), skip the cross-node bookkeeping. Test on the 4 unsolved first-contact
games (re86/sb26/bp35/lf52); gate = fewer actions-to-first-win on ≥2.

## #4 — highest upside, likely null (~2 days): online CNN + reset-on-level-up (the LEADER's *actual* edge)
We grafted StochasticGoose's CNN *architecture* (`.427`) but loaded it **FROZEN** — never its *learning
loop*. Online-fine-tune the `SmallFrameChangeCNN` on the game's own (state, action, frame_changed) stream
every 5 actions over a 200k hash-deduped buffer, and **reset weights+buffer on level-up** (source:
`action.py:149/246/255/334-353`). **Cheaper first probe (<1 day):** the existing online discriminator
already learns on-path; just clear its state on level-up (`arc_competition_agent.py ~1500`, ~15-30 LOC,
no GPU) — a 2-arm A/B (online-no-reset vs online-with-reset) isolates whether *reset* is the lift before
building any CNN buffer. (StochasticGoose's own source flags reset as a "TODO: try NOT resetting" knob —
measure it.) If neither online arm beats frozen, KILL and bank the negative; do NOT re-attempt with
goal-based reward (the dead TTT wall).

## Honest bottom line
**Do not bet the week on SOTA mining yielding a marked jump.** The SOTA *surface* is largely absorbed;
the residue is narrow exploration-schedule details that move first-win mostly on click/interactive games
and don't touch induction. The decisive move is **#1 (head-to-head, 1.5 days)** — it either green-lights
extracting #2/#3/#4 or proves the wall is generation-guidance and we stop. **Best case: +1–3 banked
levels behind a flag by 2026-06-30. Realistic case: a clean gap table that redirects the week.** If #1
ties, the week's real lever is the standing `arc_loop_solve` loop banking more reproduced levels
(currently 57) + operator-side `GameAdapter` RE on the 4 unsolved games — not further extraction.

Cloned refs: `/home/ianblenke/arc-sota-refs/{ARC3-solution, arc-agi-3-just-explore}` (arXiv:2512.24156).
