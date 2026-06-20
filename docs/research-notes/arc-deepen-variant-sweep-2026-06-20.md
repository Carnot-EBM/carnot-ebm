# ARC deepen + variant sweep — 2026-06-20 (outer-loop, 50-agent parallel workflow)

Operator: loop through unsolved game LEVELS with the new solves + try variants. Run wf_508b7a10-9ef.

## Result headline

- **NEW reproduced levels banked: 0.** Every deepen attempt (all 25 games, +1 target) returned
  needs_per_game_RE or re-confirmed the existing level. Generic search re-confirms but does NOT
  deepen — confirming (yet again) that deepening L->L+1 is bottlenecked on per-game RE, not search.
- **Variant generalization: 7/25 games** re-derive the solve on a HELD-OUT reflected (position-moved)
  layout, independently verified: cd82, lp85, ls20, sp80, su15, tu93, vc33. (Up from 3/16 in the
  earlier color-only probe.) These 7 are the solvers that GENERALIZE; the other 18 replay
  layout-specific solutions and would not transfer to an unseen layout.

## Variant-generalization table (reflect = position-moved, the strong signal)

| Generalizes (reflect-solved) | Layout-specific (no transfer) |
|---|---|
| cd82, lp85, ls20, sp80, su15, tu93, vc33 | ar25, bp35, cn04, dc22, ft09, g50t, ka59, lf52, m0r0, r11l, re86, s5i5, sb26, sc25, sk48, tn36, tr87, wa30 |

## Per-game L2 RE-delta map (the actionable byproduct)

Each deepen agent, on no-advance, reported the specific reverse-engineering delta needed to reach the
next level. This is a ready per-game backlog for the conductor's deepening tasks:

- **ar25** (L1, status=needs_per_game_RE): ar25 L2 RE delta: model the hidden ACTION7 undo-stack (a registry-flagged missing-world-model-rule verifier gap — ACTION7 pops state not encoded in the visible 64x64 display grid) which the L1 [ACTION3 x5, ACTION2 x10] object-move plan ignores.
- **bp35** (L1, status=needs_per_game_RE): L2 RE delta (mechanic_class goal_directed_navigation_local_obstacle_clear): RE the L2 avatar-shape→movement-affordance mapping + the L2 removable-blocker clear coords from the offline env, since exp4480 only banked L1's hardcoded goal/blocker layout and no L2 entry/dead_end exists.
- **cd82** (L1 (reflect-generalizes), status=reproduced): cd82 has empty gotchas/dead_ends; only an adapter-free L1 keyboard-directional+confirm (ACTION1-4 dir, ACTION5 commit) solve is registered. To reach L2: register a cd82 GameAdapter from the L1 trajectory seed capturing the directional+confirm action-model and the L2 win-condition delta, so verifier-routed search can deepen past graph-explore's L1 re-confirm.
- **cn04** (L1, status=needs_per_game_RE): No L2 banked: loop ran routing-only. cn04 is graph-explore/unit-positioning; L1 needed E1 salience-ordering to surface low-raster-order winning actions. L2 delta = RE the L2 win/action DELTA and register a cn04 GameAdapter from the routed lf52 unit-positioning recipe; generic salience search only found the 13-action L1 sequence.
- **dc22** (L1, status=no_advance): L2 needs per-game RE of dc22's config_toggle_navigation: the banked L1 is a fixed 20-label buezna-toggle trajectory (clicks at (48,36)/(48,19)), not a general toggle-rule solver — must RE the L2 same-letter blocker layout + goknoi goal coord to advance jfva.
- **ft09** (L1, status=reproduced): L2 RE delta (local_constraint_color_cycle): the L1 solver only encodes a hardcoded 4-click plan [(36,36),(36,44),(52,44),(36,52)] re-gated via reproduce; L2 needs RE of the L2 frame's own zero/non-zero-neighbor color-cycle constraint cells and their display-pixel {x,y} click coords, gated by the real offline frame level counter (not visible color-cycle satisfaction).
- **g50t** (L1, status=needs_per_game_RE): L2 needs RE of the config_toggle_target_offset delta past the L1 player==target+1 predicate; no g50t GameAdapter exists to drive a verifier-routed deepening (loop emitted routing-only mode), so register an adapter grounding L2's visible toggles before re-running.
- **ka59** (L1, status=needs_per_game_RE): L2 needs ka59's E3 executable-world-model (results/arc_e3/ka59/world_model.py) extended to the L2 block/target layout: derive push-collision clicks from the live offline camera/grid offset (never hardcode), and the hidden bottom-row StepCounter HUD tick remains a verifier residual that must be grounded before the L2 win-check.
- **lf52** (L1, status=reproduced): L2 needs a lf52 GameAdapter encoding the unit-positioning placement rule (ACTION6 click coords) so verifier-routed search can deepen — adapter-free graph-explore only re-confirms the L1 8-click sequence; registry gotchas/dead_ends are empty.
- **lp85** (L5 (reflect-generalizes), status=no_advance): L6 needs a deeper per-piece (+1,+1) goal-alignment button-rotation plan extending the Exp4372 L5 adapter, with goal-relevant-key dedup (full-grid hashing explodes search to ~26k states and times out, per registry gotchas); run timed out (exit 124) without writing a fresh artifact — stale file shows only L4.
- **ls20** (L1 (reflect-generalizes), status=no_advance): ls20 registry has no mechanic_class/action_model and empty gotchas (only a frozen L1 solve_trace); per-game RE delta for L2 = reverse-engineer the L2 win-condition/action-model and register a GameAdapter, applying the tu93~wa30/ls20 family's known deeper-level gotcha (NON-IDEMPOTENT env.reset parity state -> set branch_mode='fresh_env' so fresh-env reproduction gate passes).
- **m0r0** (L1, status=needs_per_game_RE): m0r0 L1 is only a 15-action graph_explore_solve_v2+salience/mask explore trajectory (gotchas: [], no adapter/world-model); reaching L2 needs per-game RE of the L2 win predicate + a GameAdapter (no L2 mechanic captured) — generic A* routing-only re-confirms L1 but cannot deepen.
- **r11l** (L1, status=reproduced): L2 RE delta: r11l L1 is only a frozen live-recorded solve_trace (exp4296), no GameAdapter; must reverse-engineer the click-to-template piece->template (+1,+1) matching into a derivable win-predicate adapter so the solver computes L2 placements instead of replaying the fixed L1 trajectory.
- **re86** (L1, status=needs_per_game_RE): re86 mechanic_class=pattern_match_sprite_resize: L2 RE delta = extend the L1 sprite_overlay_resize_verifier to ground the explicit RESIZE/transformation variant (resize selector tag) of the overlay match, not just L1's static-position pattern match.
- **s5i5** (L1, status=needs_per_game_RE): L2 RE delta (config_toggle_marker_coverage): discover the L2 target/initial marker coords from EXECUTION STATE after L1's next_level (not L1's [(9,51),(51,9)]/[(9,33),(30,9)]), then re-derive the h_extend(47,21,+3x)/v_extend(22,47,+3y) click counts for L2's marker geometry — grounding on execution-state coords since next_level() fires in the same action and the pre-win grid is never returned.
- **sb26** (L1, status=needs_per_game_RE): sb26 has no GameAdapter; per-game RE must derive the ordered color_match_slot_sequence win predicate (route from s5i5/ft09, sim 6.0), discover display-pixel click coords from the offline env (never hardcode), and ground on execution-state immediately before the same-action next_level() fire, then register the adapter and re-run the loop to bank any level incl. L2.
- **sc25** (L5, status=needs_per_game_RE): RE the L6 two_phase_cast_grid_then_tank_exit delta: derive the L6 cast-grid toggle pattern + tank-control exit route as a new SC25_PLANS_BY_LEVEL[6] cumulative plan, using replay-from-reset (deepcopy broken), warm-up first step, facing in state-key, offline coords (24+5c,49+5r), and resolve the fireball animation before win-check.
- **sk48** (L1, status=needs_per_game_RE): sk48 mechanic = graph_explore_solve_v2 + salience/mask (14-action click placement, like cn04); to reach L2, RE the salience-ordered click-placement sequence that advances frame.levels_completed 1->2 — registry gotchas/dead_ends are empty, so the L2 win-condition delta must be reverse-engineered fresh from the offline env.
- **sp80** (L1 (reflect-generalizes), status=no_advance): sp80 is still adapter-free (graph_explore, gotchas:[], no mechanic_class/GameAdapter); per-game RE delta for L2 = build a sp80 GameAdapter extending the [ACTION4 x3, ACTION5 commit] keyboard model into multi-direction nav, then run verifier-routed OfflineSolver (the tu93 path that took it L1->L5).
- **su15** (L1 (reflect-generalizes), status=no_advance): su15 has no registered GameAdapter (gotchas empty, adapter-free L1 only); RE delta = register a GameAdapter from the click-to-connect ACTION6 model (diagonal coords ~(10,53)->(44,19), +6/-6 steps) so OfflineSolver runs verifier-routed best-first search past L1.
- **tn36** (L7, status=needs_per_game_RE): L8 needs extending L6's multi-run CHECKPOINT-maze (wgzwawbgew advances base mnvoffrbex across runs) to longer BFS-per-leg paths AND cracking the still-uncracked program-editor EDIT->EXECUTE semantics, then registering a tn36 GameAdapter (none exists, so the loop only routed).
- **tr87** (L6, status=reproduced): L7 needs per-game RE of a new mechanic beyond L6's all-three-flags (alter_rules+tree+double_translation inverse 2-pass A->B->C chain); registry documents no L7 win-condition and configs are path-dependent, so replay the banked L6 path first to inspect L7's setup before building its verifier.
- **tu93** (L5 (reflect-generalizes), status=no_advance): tu93 is graph_explore (goal-distance-routed best-first keyboard nav to colour-14 goal); generic cold-start verifier only re-reached L3, so reaching L6 needs the per-game RE delta of characterizing the L5->L6 goal/layout transition under gotcha #7 (non-idempotent env.reset + broken deepcopy) forcing branch_mode='fresh_env' per-candidate eval.
- **vc33** (L1 (reflect-generalizes), status=no_advance): RE the L2 config_support_clearance instance (new support/goal-color pairing + lower_click coord/count) and register a vc33 GameAdapter encoding the lower_click-shifts-paired-supports action model so verifier-routed search can derive L2 (adapter-free graph-explore only re-confirms L1).
- **wa30** (L1, status=needs_per_game_RE): wa30 has no GameAdapter (L1 is only a replayed live-recorded trace, gotchas:[]); to reach L2, RE the win/action/state delta and register an adapter reusing top transfer match tu93 (same wa30/ls20 family): goal-distance-routed best-first nav with branch_mode='fresh_env' (non-idempotent-reset gotcha #7).

## Takeaway

No new levels from generic deepening (expected — it needs per-game RE). The two real outputs: (1) a
verified 7/25 generalization baseline (the games whose solvers transfer to unseen layouts — the
capability the hidden eval rewards), and (2) the per-game L2 RE-delta backlog above. Recurring RE
pattern: most L1 solves are FROZEN trajectories / hardcoded coords with no GameAdapter; deepening
needs (a) a GameAdapter that DERIVES the win-predicate from the offline env (never hardcode coords),
and (b) for the hidden-state games (ar25 undo-stack, ka59 step-counter HUD) the GAP-ARCH-GRID-ONLY-STATE
register modeling. Cross-ref: arc-energy-augmented-strategy.md, arc-frame-change-predictor-spec.md.
