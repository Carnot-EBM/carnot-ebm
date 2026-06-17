# ARC-AGI-3 live generalization gap: per-game RE does not transfer (2026-06-17)

Operator question (2026-06-17): "Are we properly learning these levels and generating new solvers
for new mechanics, and how are we ensuring we automatically learn these for our BFS/DFS/routing/TRM
so they can be applied during the LIVE submission process, which requires adapting on the fly to
games we haven't seen before?"

Honest answer: **No, not yet.** The tn36 climb (L1-L6 solved, 6 mechanics cracked) is per-game
MEMORIZATION via INTERNAL-STATE reading and MANUAL RE. It does not build toward live generalization.
This note states the gap precisely and the architecture that would close it, so the next build is
scoped and we stop mistaking per-game wins for transfer.

## The three concrete gaps (grounded in the code)

1. **Internal-state reading, not frame-only.** `scripts/arc3_tn36_offline_solver.py` (and
   `arc3_sc25_offline_solver.py`) read privileged Python internals: `env._game.fdksqlmpki.htntnzkbzu.x`,
   `okllwtboml`, `pfyayhyovw`, `bizgpiltwm`, `wgzwawbgew`, `ekdwmirldx`. The OFFLINE arcade exposes
   `env._game`; a REAL ARC-AGI-3 submission exposes ONLY rendered 64x64 frames. So:
   - A banked TRAJECTORY (list of clicks) replays frame-only (good — reproduced levels are real).
   - But COMPUTING a new solve on an unseen game cannot use internal state (there is none live).
   The deep-RE computation is therefore non-transferable to the live setting.

2. **Manual RE, not automated discovery.** Each mechanic was cracked by an LLM reading the
   obfuscated game source + probing internals. There is no automated process that, given a new
   game and only frames, INDUCES its mechanic. That process is exactly the live requirement.

3. **Per-game memorization, disconnected from the learned stack.** The trained router
   (`arc_router`) routes only `{bfs, cell_count, region_count}` goal-distance heuristics for
   graph-explore. It has no concept of the mechanic CLASSES discovered (program-editor,
   checkpoint-multi-run, timed-trap). The TRM is sudoku-trained, not ARC. Nothing about tn36
   transfers to an unseen game.

## What the tn36 climb WAS worth: the mechanic taxonomy

The value is not the per-game solvers — it is the discovered TAXONOMY of ARC-AGI-3 mechanics
(the input to a generalization design):
- **translation + obstacles** (single-run path-routing through gaps; L3)
- **program/transform editor** (a multi-slot move-program with a bit-button code editor; L1+)
- **scale/rotation/property transforms** with collision interaction (L4, L5)
- **checkpoint multi-run** (base advances at a checkpoint; path exceeds one run; L6)
- **timed/blinking traps** (obstacles toggle on a move-count parity; lethal on contact; L7)

## The architecture that would close the gap (frame-only, learned, dynamic)

1. **Frame-only mechanic INDUCTION (the non-negotiable foundation).** On a new game, PROBE it
   (take actions, observe frame->frame transitions) and INDUCE a transition/world model + win
   predicate FROM FRAMES — never internals. `python/carnot/agentic/arc_executable_world_model.py`
   (E3) already does frame-only induction; it is the right substrate. The deep mechanics must be
   re-derived this way: e.g. detect the program-editor by observing that clicking a small cell
   toggles a glyph and that a separate "run" click animates the whole board; detect blinking
   traps by observing cells that change on a fixed move cadence.

2. **A STRATEGY/solver-class LIBRARY (reusable, frame-operating).** Capture each taxonomy class as
   a reusable strategy that operates on the induced model (NOT a per-game solver): `path_route`,
   `program_editor`, `checkpoint_multirun`, `timed_trap_aware`. Each declares the observable
   FEATURES that signal it applies (e.g. "a discrete control palette of small repeated glyphs" ->
   program_editor; "obstacle cells with periodic visibility" -> timed_trap_aware).

3. **Extend the trained router from HEURISTIC-selection to STRATEGY-selection.** The router already
   learns "which heuristic for which game" from a ledger with leave-one-out validation. Extend its
   feature set with mechanic-signal features (control-type, periodic-obstacles, transform-controls)
   and its label set to the strategy classes, trained on the solved-game outcomes. A new game's
   induced features then ROUTE to a strategy class dynamically.

4. **Close the loop into BFS/DFS/TRM.** BFS/DFS are the search substrates a strategy runs over the
   INDUCED model (frame-only). The TRM (once ARC-trained on the captured frame trajectories) is the
   generator/refiner a strategy can call. The router selects {strategy x search-engine x heuristic}.

## Honest consequence

The per-game internal-state solvers should be treated as MECHANIC-DISCOVERY SCAFFOLDING (they
found the taxonomy + the banked, frame-replayable trajectories), NOT as the live solver. The live
solver is: frame-only induction -> mechanic-feature extraction -> learned strategy router ->
strategy over the induced model. Building that is the genuine next investment; continuing to crack
tn36 levels by reading internals adds reproduced offline levels but ZERO live-transfer capability.

Cross-refs: `docs/research-notes/arc-trained-router-2026-06-17.md` (the heuristic router to extend);
`python/carnot/agentic/arc_executable_world_model.py` (frame-only E3 induction substrate);
`ops/arc_solve_registry.yaml` (the mechanic taxonomy, per-game); `ops/verifier_gaps.md`
(GAP-ARC-LIVE-FRAME-ONLY-INDUCTION, GAP-ARC-STRATEGY-ROUTER).

## PROOF RESULT: frame-only mechanic detection works (2026-06-17)

Built `scripts/arc3_frame_induction.py` — a probe that uses ONLY `grid_of(frame)` +
`_levels_completed(frame)` (zero `env._game` access) to induce a game's control mechanic, and ran
it on tn36. Result:

- **DETECTED MECHANIC: program_editor** — classified correctly from frames alone.
- HUD counter at (61,1) detected as action-invariant (changed by ~every click) and MASKED.
- Edit-button palette located: 150 local-toggle cells in a dense block bbox (19,41)-(43,46),
  density 1.0 — the program-editor signature (a dense grid of small toggle-buttons), distinct from
  a direct-control game (where a click moves the avatar = a LARGE board change).
- Play-area sprites (object + target candidates) located above the editor.
- VALIDATION (separate, NOT used by the detector): "program-editor detected AND bbox covers the
  true internal slots (x 19..39): True." The frame-only detection matches the internal ground truth
  it never read.

This validates the load-bearing assumption: **mechanic-CLASS detection is achievable frame-only**,
so an unseen game can be probed and routed to a strategy without internal-state RE. Unit-tested
(`tests/python/test_arc_frame_induction.py`): editor vs direct-control vs empty.

### Honest limits found (the next induction steps)

1. **Run-button is frame-invisible with a losing program.** Clicking 'run' on a non-winning program
   resets the object (net frame change = HUD only) — no visual feedback. So the run trigger is NOT
   directly detectable by a single probe; it must be found via a level-advance with a winning
   program, or by observing the object animate (which needs multi-step run frames). This is a real
   frame-only limit, not a detector bug.
2. **Slot/bit decomposition is not gap-separable at 64x64 resolution.** The editor is a CONTIGUOUS
   block (any click in it edits the nearest slot), so the discrete 5-slots x 6-bits structure can't
   be read by gap-clustering. Using the editor needs a click->(slot,bit) mapping by position, or the
   toggle-glyph-location signal (which glyph each button flips) — the next refinement.
3. **Object-vs-target requires motion.** Both are play-area sprites; distinguishing the MOVABLE
   object from the FIXED target needs observing which one moves under a run — again gated on the
   run-frame visibility.

Net: the FIRST link (frame-only mechanic-class detection) is proven. The remaining induction
(run-trigger, decomposition, object/target-by-motion, then code semantics) is the scoped next build
toward a live solver — all still frame-only, no internal state.

## NEXT LINK DONE: the editor is now USABLE frame-only (2026-06-17)

Extended `scripts/arc3_frame_induction.py` from DETECTION to USE -- the run-trigger + the
click->(slot,bit) mapping, both derived frame-only, demonstrated by an end-to-end FRAME-ONLY solve
of tn36 L1 (zero `env._game`):

- **click->(slot,bit) mapping** (`induce_editor_layout`): each edit click toggles a GLYPH at its
  slot's display; clustering edit cells by glyph-toggle-x separates the 5 SLOTS (x 21,26,31,36,41),
  the edit-cell y's give the 2 BIT-rows (y 42,45). (Resolved a gotcha: the y=40 per-slot header
  differs between same-code slots; the code glyph is y41..47 -- compare that, not the header.)
- **run-trigger** (`find_run_button`): with the program set to a winning config, the RUN button is
  the non-editor cell whose click advances the level -> found (32,51), matching the internal
  sxhtkytekm exactly.
- **end-to-end frame-only solve** (`frame_only_solve`): set the program via the derived controls
  (toggle each editable slot's glyph to match the pre-set slots) + click the derived run -> LEVEL
  ADVANCES TO 1. Zero internal state on the solve path.

So both links the operator asked for are done: the editor is DETECTED and USABLE from frames alone.
The frame-only solve here uses a heuristic ("make editable slots match the pre-set ones") that wins
L1 specifically; the GENERAL winner-discovery (program-space search using the frame-only win signal,
or inducing the code semantics + object/target by motion) is the next link -- still frame-only.

## NEXT LINK DONE: general winner-discovery is BLIND search (no graded feedback) (2026-06-17)

Replaced the L1-specific "match the pre-set" heuristic with a GENERAL frame-only winner-search
(`frame_only_winner_search` in `scripts/arc3_frame_induction.py`) and, in the course of building it,
established the load-bearing LIMIT of frame-only solvability for the program-editor class. Three
probes (each strictly `grid_of(frame)` + `_levels_completed(frame)`, zero `env._game`):

1. **The run is ATOMIC -> object motion is frame-invisible.** Running the program advances the level
   ON win, but a single `env.step` executes the WHOLE move-program internally and resets the object;
   advancing the frame with inert clicks after a run leaves the object's salient region unchanged
   (`(31,21,2268)` for all 8 follow-up steps, level 0). So the per-move object trajectory -- the only
   thing from which the code semantics (what code 1/3/8/14 DOES) could be induced -- never appears in
   any frame. **Code-semantics-by-observation is impossible frame-only for this game.**

2. **No correctness/closeness gradient.** Toggling a slot's bit-row changes ~4 cells (the local glyph
   echo) IDENTICALLY whether the edit moves a slot TOWARD the winning code or AWAY from it
   (correct-slot-made-wrong: HUD_delta 1, full_delta 4 == need-fix-slot toggled: HUD_delta 1,
   full_delta 4). The top-band "HUD" delta is just the echo of my own click, not a distance signal.

3. **A losing run renders NO partial-match.** With a FIXED losing program, the run itself changes
   exactly ONE cell (an attempt counter at y0), IDENTICAL for k=0/1/2 slots-correct, no GAME_OVER on a
   single attempt. The win predicate is "object matches target on 5 attributes" but the
   attributes-matched count is NOT frame-visible. Only a FULL win re-renders the board (2607-cell
   delta, the binary `levels_completed` advance).

**Conclusion: the program-editor mechanic emits ONLY a binary win bit frame-only -- no gradient to
hill-climb, no pruning.** So general winner-discovery is BLIND program-space search. The implemented
search proves it works (L1 solved in **4 runs**, zero internal state, no pre-set-match assumption) via
a UNIFORM-code structural prior (try every observable code as a repeated program) backed by a bounded
PRODUCT fallback -- but the full reachable space is already **1024** with just the 2 frame-located
bit-rows (5 slots x 4 reachable glyphs), and the true 6-bit editor alphabet is far larger. With no
graded feedback the worst case is exponential in program length. **Blind frame-only search does not
scale.**

### What this means for the live-generalization plan (important)

The earlier links proved frame-only mechanic DETECTION + control USE generalize. This link proves the
PLANNING step does NOT, for the program-editor class, via online frame-only induction alone: you
cannot induce the atomic-run dynamics from frames, and there is no gradient for search. The live
solver therefore needs, for this class, an **offline-trained per-class dynamics/verifier model** (learn
"editor code -> object transform" once, offline, from the many program-editor games / our banked
internal-state RE) that the live agent then applies by reading the editor glyphs + target attributes
from frames and PLANNING with the learned model -- rather than blind online search. This is exactly
the "strategy/solver-class library + router" architecture (section above): the program-editor strategy
carries a learned transition model; it is NOT re-induced online. Logged as
`GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK` in `ops/verifier_gaps.md`. (Mechanic classes that DO expose
per-move motion -- direct-control / path-routing games -- remain online-inducible; the no-graded-
feedback limit is specific to atomic-run program-editor games.)
