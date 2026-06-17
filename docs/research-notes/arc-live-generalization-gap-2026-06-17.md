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
