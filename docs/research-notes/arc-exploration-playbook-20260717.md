# ARC-AGI-3 Exploration Playbook (mined 2026-07-17)

## What this is and why it exists

This document distills the **game-agnostic exploration methodology** hidden inside
`ops/arc_solve_registry.yaml` — tens of thousands of words of `gotchas:` and
`dead_ends:` narrative accumulated across ~25 public games and 100+ solving rounds.
Those narratives are full of hard-won lessons, but each is buried next to per-game
trivia (a specific color, a specific coordinate, a specific mechanic). The scored
ARC-AGI-3 agent faces **hidden games it has never seen** (see CLAUDE.md, "ARC-AGI-3 IS
a Live Hidden-Game Discovery Agent"), so a per-game fact ("in bp35 the pillar eye is
color-15 B0B") transfers to zero hidden games and is actively misleading. What DOES
transfer is the recurring **move** — the verification habit, the search discipline,
the interaction heuristic, the way-out-of-stuck — that worked (or the recurring
mistake that wasted rounds) independent of any game's specific mechanics.

Every pattern below is stated generically first, then cited to 2-3 concrete corpus
rounds (game + round + one-line) for traceability. Nothing here names a color/coord/
mechanic as load-bearing; where a game is named it is only as evidence that the
*general* move recurred.

This is the shared foundation for two deliverables built on it:
`arc_solver_kit.py` utilities (Phase 2) and the live-agent stuck-detection exemplar
injection (Phase 3, `CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED`).

---

## Theme 1 — Verification habits (ground your beliefs, don't assume)

### 1.1 Re-verify action semantics empirically at every level; never assume carryover
Action IDs, movement quanta, control mappings, and even which button does what
routinely **change between levels of the same game** — assuming a prior level's
semantics is a top cause of wasted rounds and instant deaths.
- **bp35 L9 round 13**: "Confirmed L9 action semantics empirically rather than assuming
  carryover from L6: ACTION1/2/3 all move LEFT, ACTION4 moves RIGHT, ~5-6 cells per call
  (not a single step)."
- **m0r0 L5**: "Horizontal controls are REVERSED vs the L3/L4 description at L5 (ACTION3
  converges, ACTION4 diverges)." Same game, opposite mapping one level later.
- **sc25 gotcha**: "tank-controls: facing is load-bearing" and "first step after reset
  consumed" — controls are state-dependent, not fixed.
- **bp35 L7 round 11**: "L7 DOUBLES the per-level hard action-count budget from L6's 64
  to 128 — do not assume the timer stays at 64; verify it live each time."

### 1.2 The level counter is the only ground truth; visual cues are not
A glyph changing color, a piece visually overlapping a target, or an "all-green" board
does **not** mean you won. Only `frame.levels_completed` advancing (and, for a full
clear, `state == WIN`) proves success.
- **sp80 L6 round 5**: "a glyph-color change does not mean the stream actually entered
  through the open side" — the socket recolored but the win didn't fire until the elbow
  moved one 3-pixel cell.
- **wa30 L2**: "a temporary overlap is not a placement, only frame.levels_completed
  advancing is ground truth."
- **cn04 L4-L5**: "a transient ALL-GREEN state did not fire the win once ... if
  everything looks paired but nothing fires, try a rebuild before assuming the model is
  wrong." (Green = the game's own pairing feedback, yet still not a win signal.)

### 1.3 Measure positions/geometry programmatically from the frame; don't eyeball
Off-by-one estimation from a rendered image burns scarce commit/action budget. Extract
exact coordinates from the frame array and reason in lattice units.
- **sp80 L3-L4**: "measure piece and target positions from the frame programmatically,
  off-by-one eyeballing wastes commits"; the L6 win was a **single 3-pixel cell** of
  elbow offset that eyeballing missed for two rounds.
- **wa30 L2 v5**: switched "from improvised play to explicit computation — extracted
  exact grid coordinates ... then exhaustively searched" — the turning point toward
  progress.
- **cd82 gotcha**: "Palette click coordinates must be discovered from live sprite
  centers; hardcoded coordinates drift."

### 1.4 The win-state frame is often the *next* level's frame; search frame-after-next
Many config/placement games call the level-transition on the exact action that completes
the arrangement, so the solved board is never externally rendered. Don't hunt for a
"solved" frame; ground the predicate on execution state immediately before the
transition and count only the reproduced level advance.
- **lp85 L6-L7**: "the action that wins a level immediately renders the NEXT level, so
  the solved board state is never externally visible — do not search for a 'solved'
  frame, search for the frame-after-next."
- **s5i5 gotcha**: "next_level() fires in the same action that creates final marker
  coverage, so the L1 pre-win grid is not returned; ground on execution-state marker
  coords and count only reproduce() level advance."
- **cd82 L3-L6**: "Level transitions lag one frame (the winning stamp does not
  immediately show the new board)."

---

## Theme 2 — Perception & object interrogation (see what's actually there)

### 2.1 Click/interrogate every visually-unexplained object before committing to a route
Hazards and utility objects are frequently camouflaged as background texture or
decoration. An object you can't explain is a hypothesis to test, not scenery to route
around — and interrogating it has repeatedly been *the* unlock.
- **wa30 L2 v9 (the headline example)**: a "solid color-12 4x4 tile near the bay" that
  eight prior probe attempts ignored turned out to be a passive HELPER ROBOT — a whole
  second worker that made the "proven infeasible" level trivially solvable. "an
  'exhaustive' mechanic sweep is only as exhaustive as the hypotheses considered."
- **bp35 L3-L4**: "GRAVITY SWITCHES (color-8 framed blocks, often buried inside solid
  wall — clicking any unexplained glyph paid off again here)."
- **bp35 L9 round 15**: clicking the unexplained pillar *grew* it into a platform — "a
  previously-undocumented interaction" that (round 16) made an otherwise-fatal fall
  survivable and was the key to the eventual full clear.

### 2.2 Naive color/shape detectors are fooled by contact-highlights and selection state
Objects recolor on contact or selection. A detector keyed on a single "resting" color
will miss the same object once it is adjacent to the player, selected, or carrying
something.
- **wa30 L2**: "Contact-highlight recolors a block outline (3=player-adjacent,
  5=robot-adjacent), which can fool a naive frame-color block detector"; the L6 thief
  "contact-highlights with a color-11 outline, exactly like a block does — a naive
  color-15-only detector will miss it."
- **r11l L3**: "the black diamond shown on a handle is just its SELECTION highlight, not
  a special/locked handle" — a selection-state render mistaken for a distinct object.
- **sp80 L6 round 3**: "The selected piece renders as color 9 (unselected splitters are
  color 8, unselected elbows color 15)" — selection state *is* usable signal once you
  know it exists, but only if you don't confuse it for a different object.

### 2.3 Read the multi-layer animation array (`frame.frame` / `frame._frame`), not just the settled grid
The frame is a stack of animation sub-frames `(N, 64, 64)`; the last layer is the
settled grid, but the intermediate layers captured during one action (with a fixed
camera) reveal information the camera-relative settled frame hides — most importantly
absolute vertical trajectory. Also: some frames stack two real sub-frames — parse the
correct (usually last) one.
- **bp35 L9 round 19**: "the env f.frame is a multi-LAYER animation array (e.g. 57
  layers on the fatal fall) captured with a FIXED camera during a single action, so it
  reveals the player's ABSOLUTE vertical trajectory (up vs down) directly — a real
  advance over the camera-relative settled frame (which re-centers on the player every
  step and hides absolute motion)." This tool directly disambiguated the death cause and
  led to the full clear.
- **wa30 L2**: "frame.frame can contain 2 animation sub-frames — parse the LAST one."
- **cd82 L3-L6**: "the terminal WIN frame has ZERO grid layers — naive frame[-1]
  indexing crashes on it" — layer count itself is signal, and layer indexing must be
  defensive.

### 2.4 Camera is view-relative and re-centers/scrolls; re-verify coordinates after motion
Any game with a scrolling or re-centering camera makes click coordinates view-relative.
The same grid cell maps to different screen coordinates after the camera moves, and
absolute motion is hidden by the recentering. Confusing camera-relative with absolute is
a recurring failure mode.
- **lf52 L3**: "the camera follows and scrolls — all click coordinates are VIEW-relative,
  re-verify positions after any scroll."
- **bp35 gotcha**: "Camera-relative click coordinates repeat across different grid cells
  after falls; the replayed sequence is display-coordinate grounded, not grid-label
  grounded."
- **bp35 L9 rounds 22-23**: an entire "camera hard-clamp" theory was built and then
  retracted once the recenter formula (`rest_row - 38.5`) was measured — camera behavior
  must be measured, not assumed.

### 2.5 The game often enumerates its own legal moves/targets — use it as a free oracle
Many games render their own move affordances (highlighted legal landing squares, a
preview of the required color). Reading that is a free, exact move-enumerator; but a
*preview* of a target is not proof the target is satisfied.
- **lf52 L3**: "ACTION6 on a peg paints ... color-2 markers on every LEGAL landing square
  ... the game enumerates its own moves; use this as a free move-enumerator before
  planning."
- **su15 L3 / cn04 L4**: clicking a piece "reveals its markers" / legal interactions;
  "survey by clicking each sprite before planning."
- **re86 L5**: "Ring centers are pre-colored with their required color as a preview, but
  that alone is not proof of coverage — check stroke-adjacent pixels too." (The oracle
  shows *intent*, not *satisfaction*.)

### 2.6 Distinguish the interactive target from the goal *display*
Games render a picture/legend of the goal that is inert; the real delivery target is a
different on-field object. Clicking the display wastes actions and misleads.
- **su15 L3**: "DELIVER TO THE BLOBS, NOT THE TOP BOXES: the top hollow boxes are goal
  displays only; the actual delivery targets are the big round blobs on the field."
- **cd82**: "Clicking canvas cells directly, or clicking the target picture, both do
  nothing (confirmed inert)."
- **sb26 L3-L8**: solid tray tiles are consumable, hollow tiles are branch markers, and
  the legend row is a *sequence to satisfy*, not a thing to click — the display encodes
  the rule, not the interaction.

---

## Theme 3 — Search & reachability technique (know what you've actually proven)

### 3.1 "Unreachable" means PROVEN-exhaustive, not SEARCH-CAPPED — never conflate them
A search that returns no solution has two very different meanings: the frontier emptied
(a real proof of unreachability under the model) versus a node/time budget hit (says
nothing). Treating a capped search as a proof shuts down a live line prematurely; the
corpus is full of "settled dead ends" later overturned.
- **wa30 v8 -> v9**: v5-v8 concluded L2 was a "FULLY SETTLED dead end" via a real Manhattan
  lower-bound proof — then v9 overturned it because the *model* (single worker) was
  incomplete, not because the search was wrong. "the 'settled dead end' conclusion was
  correct GIVEN the single-worker model tested, but that model was incomplete."
- **bp35 L9 round 20**: "a full naive reachability BFS is INTRACTABLE to exhaust ... a
  RESTRICTED BFS explored 118 unique reachable states ... (partial, NOT exhaustive —
  corroborates but does not prove unreachability)." Explicitly labels the difference.
- **lf52 round 13**: "THE WIN IS STRONGLY-INDICATED-UNREACHABLE from the canonical start
  (NOT exhaustively proven)" — and round 14 then found the win, because a real dead-end
  claim requires exhaustion the round-13 search hadn't achieved.

### 3.2 When state explodes, hash on the SEMANTICALLY-RELEVANT subset, ignore cosmetics
Naive full-grid-hash deduplication explodes when the game has cosmetic variation
(animation phase, decorative growth, camera pixels). Dedup on the load-bearing state
only (position + facing + the flags that matter) and explicitly exclude cosmetic
variation — but be sure "cosmetic" is genuinely cosmetic.
- **lp85 gotcha**: "dedup by goal-relevant key (goal positions), not full grid hash, or
  the search explodes (26k states -> fails)."
- **lp85 L6-L7**: "The left-edge vertical strip records clicks and changes the frame even
  when the puzzle arrangement cycles back ... exclude it when comparing visible puzzle
  states or dedup will undercount."
- **bp35 L9 round 20 vs round 21**: round 20 deduped ignoring "pillar-growth cosmetics";
  round 21 CORRECTED that — "pillar growth is NOT cosmetic" (it carried the player
  vertically). The lesson cuts both ways: hash the semantic subset, but verify what is
  truly cosmetic before discarding it.

### 3.3 When navigation is exhausted and still fails, question the WIN-CONDITION model itself
If every reachable route and interaction has been tried and nothing wins, the bug may be
your model of what "winning" means, not your search over how to get there.
- **bp35 L9 round 20**: after 7 rounds of failed navigation, "tested whether L9's win is
  NON-SPATIAL (a score/collection/activation condition rather than diamond-touch)" —
  systematically ruling out an alternate win-condition class.
- **sp80 L5**: "both tower vines were routed into all three arch holes with zero strays
  ... and the commit STILL failed — there is a genuinely undiscovered extra win
  condition." The routing model was complete; the win model was not.
- **vc33 L5**: proved via unit-conservation that "BOTH nominal bar==door targets are
  unreachable as stated ... the true L5 win condition remains undiscovered" — the stated
  win model was provably wrong.

### 3.4 Source-grounded search (public games only) can break walls blind interaction can't — but validate against the real engine and enumerate every piece type
For the 25 PUBLIC development games, reading the game source and building an offline
simulator is permitted (CLAUDE.md: "Source-reading is a PUBLIC-games dev tool ONLY —
NEVER in the hidden live submission"). It breaks walls pure interaction cannot — but any
source-derived simulator must be stress-tested move-for-move against real `env.step`
before it is trusted, and any confinement/impossibility proof must enumerate EVERY
selectable piece type.
- **sk48 L4-L8 / lf52 L7**: solved "via an exact source-grounded abstract transition
  search ... the deep-copy substrate was verified move-for-move identical to live
  env.step before searching."
- **lf52 round 14**: round 13's source-grounded "confinement proof" had a load-bearing
  omission — it "generated jump successors ONLY for [one piece type], deleting every
  [other-type] jump." "confinement searches must enumerate EVERY selectable peg type even
  when only one type can satisfy the removal win condition."
- **lf52 round 13**: even a source-grounded conclusion "must still be validated against
  the real live engine before being trusted, and a proof/simulator built from source
  should be stress-tested against real historical transitions before being relied on."

### 3.5 A verifier/energy heuristic routes best-first search; the executable gate is the only authority
Learned or computed verifiers make search efficient (best-first ordering, pruning) but
are never the win oracle — only the offline reproduction gate counts a level. This keeps
the search fast without letting a mis-calibrated heuristic fabricate a win.
- Recurring across ls20, lf52, sk48, ar25, dc22, sb26, re86, ft09, vc33: "The learned
  verifier checkpoint routes/ranks future search only; the executable offline
  reproduction gate remains the oracle-distinct authority."
- **tr87 gotcha**: "Blind config-BFS is intractable ... The verifier is load-bearing. Use
  SUMMED CYCLIC-DISTANCE (not bare mismatch count): mismatch-count gives no gradient ...
  cyclic-distance drops 1 per step-toward-target and routes L4 in 319 states." (A
  *dense* heuristic beats a sparse one for routing.)

---

## Theme 4 — Hazard, budget & reset discipline (survive to keep searching)

### 4.1 Use fresh-environment branches to test risky hypotheses safely
Test a dangerous idea on a throwaway branch, not on a real in-progress attempt. This is
also the *correct* branching mode for games whose `reset()` is not idempotent (see 4.2).
- **tu93 gotcha #7**: `branch_mode='fresh_env'` (a brand-new env per candidate eval) is
  the fix for non-idempotent reset — "each evaluation sees the same pristine parity-0 the
  gate uses." (Encoded in `OfflineSolver.branch_mode`.)
- **sp80 gotcha**: "Use fresh-env branching for this adapter; replaying L1 on a reused
  advanced env does not reconstruct the L2 start state that the reproduction gate sees."
- **cn04 / g50t gotchas**: "Use fresh-env branching; reuse-one-env replay can stop at the
  L1 seed even though the full sequence reproduces on a fresh offline arcade."

### 4.2 `env.reset()` is often NOT a clean reset — build a fresh env instead
After a game-over or mid-game, `env.reset()` frequently leaves poisoned hidden state
(parity flags, leaked ghosts, depleted timers) that silently corrupts the next attempt.
A genuinely clean instance needs a brand-new `arc.make()` env. Some games also expose an
in-game RESET action distinct from `env.reset()`.
- **tu93 L6**: "env.reset() is parity-poisoned; after any death, rebuild with a NEW env
  via arc.make() and replay the prefix, never chain resets."
- **g50t gotcha**: "env.reset() on an existing offline-arcade env does NOT fully reset ...
  a committed ghost and the depleted action timer can leak into the 'fresh' game. A
  genuinely clean instance requires a brand-new env via arc.make()."
- **r11l L3 / sc25 L6**: "GameAction.RESET (not env.reset()) is the safe recovery from
  BOTH mid-level trouble and GAME_OVER ... env.reset() after a GAME_OVER leaves a broken
  state." Reset *scope* also varies (see 4.4).

### 4.3 Count every action against real budgets — failed/blocked/no-op actions cost too
Games meter actions (movement timers, click budgets, per-level action ceilings). Blocked
moves, failed clicks, selects, no-ops, and even undo often consume the same budget, and
failed clicks sometimes cost *more* (escalating lockouts). Plan routes as speedruns when
a hard ceiling exists.
- **su15 L3**: "A RESOURCE METER ... costs ~1-2 per successful drag but 3-13 (escalating)
  per FAILED click ... Repeated failures ... trigger an escalating lockout; never
  immediately retry a failed click."
- **bp35 L6 round 8**: "bp35 L6 has a HARD, INVISIBLE ACTION-COUNT TIMER that ends the
  game at EXACTLY total action 228 ... every route must be planned as a speedrun (no-ops,
  clicks, and ACTION7 undos all burn the same budget)." This retroactively invalidated a
  "lethal step" conclusion — the deaths were the clock, not the step.
- **r11l L3**: "A col-0 stripe is a 64-click per-level budget (selects, rejected clicks,
  and no-op moves all count) that causes GAME_OVER at zero."

### 4.4 Reset/timer SCOPE varies (current-level vs whole-game) and can flip on timing
Whether a reset restarts the current level or wipes the whole game — and whether the
level timer is separate from your session budget — must be checked, not assumed. The same
action can even have different scope depending on when it fires.
- **r11l L6**: "ACTION0 during active L6 restarts ONLY L6 and preserves levels_completed=5,
  but ACTION0 immediately after the L6 WIN resets the ENTIRE game to level 0 — a genuinely
  different reset scope depending on when it fires."
- **su15 L9 / vc33 L4**: "a final-level reset performs a FULL-GAME reset (not a
  current-level restart, unlike earlier levels)"; vc33's wasted-click budget exhaustion is
  "GAME OVER with levels_completed RESET TO 0, not a level restart."
- **ka59 / g50t / sp80**: level-internal action counters are distinct from the session
  budget; several games expose a per-level "keyhole" reset that leaves banked prefixes
  intact.

### 4.5 Bisect an action sequence to isolate the minimal death-causing prefix
When a sequence ends in death, don't re-derive by hand — binary-search the prefix to find
the exact action that kills, and separate the real hazard from an unrelated
budget/timer/harness cause.
- **bp35 L6 round 8**: the "lethal right step out of the notch" was disproven by isolating
  that death timing came from the action-228 clock, not the step — "this session stepped
  right out of the notch TWICE and stayed alive."
- **bp35 L9 round 23**: "Two-point proof: the ungrown gap float ... dies at exactly world
  row 0 ... while ... door-3 float died at exactly world -48 ... same screen row, 48 world
  rows apart" — isolating the true kill predicate by comparing two matched flights.
- **re86 L8 round 6-7**: isolated that per-fixture paint steps were "load-bearing, not
  incidental overhead" by testing the route with and without them (a controlled ablation).

### 4.6 Keep candidates honest under budget pressure; the outer loop runs the gate
When budget runs out before a full reproduction gate, hand off an explicitly unverified
candidate with the evidence you *do* have — never claim an unobserved win. This
agent/outer-loop division of labor recurs as the cleanest workflow.
- **wa30 v11**: "the agent stayed honest about an unverified candidate under budget
  pressure instead of either forcing a gate run that would overshoot budget or claiming
  success without one."
- **sk48 / ka59 / cn04 / vc33 (many rounds)**: "The source agent honestly flagged the win
  as reproduction_verified=false ... the outer loop independently verified: 5x fresh
  env.reset() + replay, all 5 confirmed."

---

## Theme 5 — When stuck: reframe, don't grind

### 5.1 Distinguish a game-over/degenerate/overview render from a real state or location-specific reveal
An empty or degenerate frame, or a wide "game-over overview" render, is frequently a
harness/engine artifact, not a real signal — and a death-overview shows the whole level
regardless of where death occurred, so it is not a location-specific reveal.
- **g50t L7 rounds 11-12**: an empty terminal frame (`levels_completed=0, is_empty=True`)
  was mistaken for a broken candidate; root cause was a settling loop that "re-submits the
  SAME label ... does not check for GameState.WIN, so a step submitted AFTER a genuine win
  fires one more env.step() [returning] the empty/degenerate terminal sentinel." Real win,
  harness artifact.
- **bp35 L9 round 14**: "the GAME_OVER frame was shown to be a FIXED, deterministic
  level-overview render (identical object bounding boxes) regardless of where the death
  actually occurred" — its contents are real level info but NOT tied to the death spot.
- **cd82 L3-L6**: "the terminal WIN frame has ZERO grid layers — naive frame[-1] indexing
  crashes on it."

### 5.2 A second look with a mandate to find the unexplained beats grinding the same hypotheses
Fresh eyes — or the same agent explicitly told to look for what prior attempts missed
rather than re-testing refuted hypotheses — repeatedly cracked "settled" walls. The
converse failure is re-running the same model expecting a different result.
- **wa30 v9**: run "as a deliberate model-comparison probe after v1-v8's dead end ...
  asked to look for anything the other model's 8 attempts might have missed rather than
  re-testing already-refuted hypotheses" — found the helper robot in one session.
- **cn04 L4-L5**: "if everything looks paired but nothing fires, try a rebuild before
  assuming the model is wrong" — a cheap reframe over deeper analysis.
- **sk48 L3 rounds 3-7**: many rounds of "superficially analogous plans left the anchored
  column unchanged" until round 7 correctly re-read the target sequence order — grinding
  the same construction never worked; re-reading the win condition did.

### 5.3 Refine a failing plan by naming the ONE missing step, not by starting over
Progress across rounds came from precisely diagnosing the single missing action/ordering
and inserting it, rather than re-deriving the whole route. A failing candidate is data:
its exact failure point localizes the fix.
- **ls20 L5 rounds 3-7**: round 3 diagnosed "insert an upper black-dot tile visit before
  the lower-pad orientation sequence"; round 6 found the exact one-action shortfall;
  round 7 closed it by financing one resource via a newly-found ring — each round added
  one named step.
- **s5i5 L4 rounds 2-3**: round 2 hypothesized "the two inner shelves likely must EXCHANGE
  vertical order"; round 3 confirmed the spirit and found the precise fix (retract one
  shelf downward while narrow).
- **sp80 L6 rounds 3-5**: narrowed from "some routing or hidden condition unresolved" to
  the exact "one 3-pixel cell" elbow offset — a geometric diagnosis, not a new mechanic.

### 5.4 Minimize tooling ceremony; reason from raw frame arrays and budget offline->live translation
A recurring self-inflicted failure is burning a whole session building visualization
harnesses or over-investing in offline simulation, then timing out before any live
verification. Reason from raw grid arrays, handle multi-layer frames defensively, and if
you do an offline search, hard-budget translating the plan into live actions and
executing it *before* the session ends.
- **lf52 round 3 (infra note)**: "three consecutive codex sessions ... all TIMED OUT ...
  every session spent its budget building an interactive PNG-rendering driver harness ...
  Next attempt should skip the PNG-rendering step entirely and reason from raw grid
  arrays." Round 5 did exactly that and solved the level with "zero rendering tooling
  built."
- **ka59 round 5 / ls20 round 4**: sessions crashed on their own frame-printing helpers
  (a single-64x64-grid assumption on a two-layer frame; a PNG-render IndexError) —
  "agent-tooling bug, not a game finding."
- **ka59 round 7 / dc22 round 6**: "the same tooling-churn failure mode ... rooted in
  over-investing in offline simulation instead of live verification" / "spent nearly all
  of its remaining wall-clock repeatedly rewriting its own scratch REPL driver script
  rather than driving exploration actions."

### 5.5 Beware ordering-contamination and prefix/animation contamination
Reaching the right *positions* is not enough when order matters: an intended sequence can
be contaminated by a nearer element seen first, by leftover prefix actions bleeding into
a level entry, or by residual animation state. Make candidate scripts phase-robust.
- **sk48 L6-L7**: "moving the [blocks] to the NEARER position ... fails, because the chain
  then sees the crossing block first, producing [wrong order] — ordering contamination,
  not just reachability, determines the correct target position."
- **wa30 L6 / r11l L6**: "the L1-L5 prefix can bleed leftover tail actions into the L6
  entry state ... any candidate script must be phase-robust to this kind of prefix
  contamination"; r11l L6 needed an explicit reset action to restore the clean layout.
- **tr87 gotcha #7 / cn04**: win-triggered fade animations leave residual state that a
  reuse-one-env search treats as a spurious win; "use a full-grid state_key so the
  animation frames stay distinct" + fresh-env branching.

---

## How the live hidden-game agent should use this

The scored agent (`arc_competition_agent.py`'s `E3AgentPolicy`) never sees these public
games. What it inherits from this playbook is **the method, applied to whatever it is
handed**:

1. On first contact, **survey before committing**: click/interrogate every unexplained
   object (2.1), read the game's own move affordances (2.5), and empirically confirm what
   each action does *this level* (1.1).
2. Treat only the level counter as truth (1.2), measure geometry from the frame (1.3),
   and read the multi-layer frame for absolute motion when a camera scrolls (2.3, 2.4).
3. Route best-first with a verifier heuristic but gate every claim (3.5); when a search
   fails, know whether you *proved* unreachability or merely capped it (3.1), and if
   navigation is exhausted, question the win model (3.3).
4. Survive to keep searching: branch on fresh envs (4.1-4.2), count every action against
   real budgets (4.3-4.4), bisect deaths (4.5), and stay honest under budget pressure
   (4.6).
5. When stuck, reframe rather than grind (5.1-5.5): rule out harness artifacts, look for
   the unexplained, name the one missing step, keep tooling minimal, and watch for
   ordering/prefix contamination.

Phase 2 mechanizes the most concrete of these (2.1, 2.3, 3.1/3.2, 4.5, plus an
action-semantics prober for 1.1) as reusable `arc_solver_kit` primitives. Phase 3 injects
the *pattern statements* above as few-shot methodology exemplars into the live agent's
stuck-detection re-induction path, dev-gated behind
`CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED`.
