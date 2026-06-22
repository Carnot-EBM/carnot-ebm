"""Divergence-TOLERANT test-time-learning solve loop (coordinated-redesign piece 2, 2026-06-21).

The exact-match `plan_and_execute` HALTS the instant `pred != obs` exactly -- so a useful-but-imperfect
learned world model (the TTT CNN: ~0.55 changed-cell recall, exact-grid ~0) can NEVER drive a solve, even
once the cell-recall gate (piece 1) lets it through. This loop is piece 2: instead of halting on
divergence, it OBSERVES the real transition, LEARNS from the surprise (refit), and RE-PLANS from the actual
state. Each divergence is a correction signal, not a dead end -- the model improves as it acts. It composes:

  piece 1 (cell-recall gate)  -> the TTT engine is trusted enough to plan with
  piece 2 (this loop)         -> replan-on-divergence lets the imperfect model drive live execution
  piece 3 (directed explore)  -> reach the FIRST win so there is a goal to plan toward (the binding
                                 constraint; here only a NAIVE salient-cycle placeholder -- the honest
                                 measurement will show whether first-win is exploration-bound)

HONEST SCOPE: plan_in_model needs an OBSERVED win-state (is_level_complete is populated only by a real
level-up). On games where exploration never triggers a first level-up, there is no goal and the loop falls
back to exploration -- so this piece moves DEEPENING/efficiency on games a win is reachable on, and exposes
the exploration-to-first-win (piece 3) wall on the rest. No trained weights beyond the cross-game prior;
pure reusable PROCESS. Offline, zero quota.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def ttt_solve(
    game: str,
    *,
    prior_path: str = "models/arc_dynamics_prior.pt",
    budget: int = 2000,
    refit_every: int = 16,
    min_transitions: int = 16,
    cnn_epochs: int = 40,
    warmup: bool = False,
    max_plan_nodes: int = 3000,
    max_plan_depth: int = 20,
) -> dict:
    """Run the divergence-tolerant TTT solve loop on the OFFLINE arcade. Returns an outcome dict with the
    levels reached, actions used, and diagnostics (replans, divergence-learns, plan attempts)."""
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical, plan_in_model
    from carnot.agentic.arc_live_ttt import LiveTTTWorldModel, _load_prior

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    cell = detect_cell(grid_of(f))
    start_level = _levels_completed(f)
    levels = start_level

    model = LiveTTTWorldModel(game, dynamics_backend="cnn", prior_state=_load_prior(prior_path),
                              refit_every=refit_every, min_transitions=min_transitions, cnn_epochs=cnn_epochs)

    gp = to_logical(grid_of(f), cell)
    actions = 0
    replans = 0
    divergence_learns = 0
    plan_attempts = 0
    plans_found = 0
    explore_steps = 0
    first_levelup_actions: Optional[int] = None

    def _step(action_id: int, data: Any):
        nonlocal f
        nf = env.step(_game_action(GameAction, int(action_id)), data=data)
        f = nf if nf is not None else f
        return nf

    while actions < budget:
        have_goal = bool(getattr(model, "_win_states", None))
        plan = None
        if have_goal:
            plan_attempts += 1
            plan = plan_in_model(model.engine, model.is_level_complete, gp,
                                 max_nodes=max_plan_nodes, max_depth=max_plan_depth)
        if plan:
            plans_found += 1
            # execute with REPLAN-on-divergence (learn from each surprise instead of halting)
            for step in plan:
                if actions >= budget:
                    break
                pred = np.asarray(model.engine(gp.copy(), step["action"], step["data"]))
                nf = _step(step["action"], step["data"])
                actions += 1
                if nf is None:
                    break
                obs = to_logical(grid_of(nf), cell)
                lvl = _levels_completed(nf)
                model.observe(gp, step["action"], step["data"], obs, levels, lvl)
                if lvl > levels:                                  # LEVEL UP -> bank, replan for next level
                    if first_levelup_actions is None:
                        first_levelup_actions = actions
                    levels = lvl
                    gp = obs
                    break
                if pred.shape != obs.shape or not np.array_equal(pred, obs):   # DIVERGENCE
                    divergence_learns += 1
                    model.fit_now()                               # incorporate the surprise, then replan
                    gp = obs
                    replans += 1
                    break
                gp = obs
        else:
            # no goal yet (or planning found nothing): EXPLORE to gather transitions + maybe trigger a win.
            # NAIVE salient-cycle placeholder (piece 3 = directed exploration is the real fix here).
            cands = rich_action_candidates(f)
            if not cands:
                break
            c = cands[explore_steps % len(cands)]
            explore_steps += 1
            nf = _step(int(c.action_id), c.data)
            actions += 1
            if nf is None:
                break
            obs = to_logical(grid_of(nf), cell)
            lvl = _levels_completed(nf)
            model.observe(gp, int(c.action_id), c.data, obs, levels, lvl)
            if lvl > levels:
                if first_levelup_actions is None:
                    first_levelup_actions = actions
                levels = lvl
            if actions % refit_every == 0 and len(getattr(model, "_dsl_transitions", [])) >= min_transitions:
                model.fit_now()
            gp = obs

    return {
        "game": game,
        "levels_reached": int(levels - start_level),
        "first_levelup_actions": first_levelup_actions,
        "actions": actions,
        "plan_attempts": plan_attempts,
        "plans_found": plans_found,
        "replans_on_divergence": replans,
        "divergence_learns": divergence_learns,
        "explore_steps": explore_steps,
        "n_win_states": len(getattr(model, "_win_states", []) or []),
        "executor": "divergence_tolerant_replan",
    }
