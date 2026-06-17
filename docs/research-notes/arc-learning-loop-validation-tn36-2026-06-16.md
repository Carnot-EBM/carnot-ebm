# ARC learning-loop validation on a new game (tn36) — what transfers, what doesn't

**Outer-loop, 2026-06-16.** Operator-agreed test: validate the just-built ARC
learning loop on one new easy game to see if it generalizes off lp85. Honest,
time-boxed result.

## What transferred (the loop works as scaffolding)
- **Transfer routing is correct.** `recommend_approach("tn36")` ranked the proven
  recipes by similarity: r11l (7.0), lp85 (6.0), sc25 (5.0) — i.e. "this is a
  click game, start from r11l/lp85." Right call.
- **The general harness transfers with zero re-derivation.** offline arcade,
  frame-based level, replay-from-reset BFS, the verifier-routed `OfflineSolver`,
  and the reproduction gate all plugged into tn36 in minutes. None of the ~10
  general gotchas (warm-up, never-hardcode-coords, deepcopy-unreliable, etc.) had
  to be rediscovered. This is the real win: the months-of-pain general layer is
  now a library.

## What did NOT transfer (the per-game delta is irreducible)
- **The game-specific mechanic must still be reverse-engineered each time.**
  tn36 is NOT a simple click-to-template like r11l. It is a **program-matching
  state machine** (`Tn36`/`ytkjoffamq`): the player sprite (`mvqheosngn`) and a
  target (`bzirenxmrg`); clicking a button (`miytdaqzei[i]`) applies that button's
  "program" (position/rotation/scale/color); win = `bzirenxmrg.vklyonlcrw` (player
  matches target). lp85's `discover_click_buttons` returns **0** on tn36 — its
  buttons are a different mechanism. So the recipe transfer is *partial*: routing
  says "click game," but the action-model + win + state-extraction are bespoke.
- Two concrete delta-RE snags hit in the time-box: the buttons aren't populated on
  the object at `reset()` (lazily built), and player/target state needs
  game-specific accessors (not the generic grid hash).

## The honest conclusion (calibrates strategy)
The learning loop makes a new game **cheaper, not free**: reuse the whole harness
+ search engine + routing (skip ~10 general layers), but each game's unique
win/action/state mechanic is irreducible RE. And the **easy games are exhausted** —
of 25 games, the *only* unsolved click-only non-spatial one (tn36) is medium-hard
with a custom mechanic; the rest are harder (keyboard/spatial/hidden-state). So
growing the reproducible count beyond the current 6 means real per-game RE each
time, which the loop accelerates but does not eliminate.

**Recommendation:** because the loop is validated as scaffolding but per-game RE is
the irreducible cost, the scalable path is to **wire the loop into the conductor's
standing ARC task** so the per-game RE + verifier-routed solve happens
autonomously over many milestones (24/7), rather than solved by hand one game at a
time. The loop is the right machine; it now needs to run continuously.

## Provenance
- routing: `python/carnot/agentic/arc_solve_learning.py` (tn36 -> r11l/lp85)
- harness reused: `arc_solver_kit.OfflineSolver` plugged into tn36 directly
- tn36 mechanics: `environment_files/tn36/.../tn36.py` (`Tn36`, `ytkjoffamq`)
