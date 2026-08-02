#!/usr/bin/env python3
"""One (game, arm, seed) bounded run of the SCORED ARC policy, with per-action accounting.

REQ-ARC-WMTE-6071. Worker for ``scripts/arc_inert_label_defer_ab.py``; one killable subprocess
per cell, because the policy imports and can execute LLM-authored engine code (not on this
configuration -- induction is disabled -- but the isolation is not conditional on that).

WHAT IT DRIVES. ``E3AgentPolicy`` -- one of the two live entrypoints named by CLAUDE.md's ARC
Live-Path Reachability Discipline -- from a FRESH RESET on the offline arcade with a bounded
action budget. NOT a solve-conditioned window: every window built through
``arc_actions_to_progress.build_progress_window`` needs the game already solved to L1 through a
registered adapter and then shows a prefix of a banked WINNING route, so a search-efficiency
measurement built on one would inherit its own answer.

WHAT IT MEASURES, and why more than one thing. The trap in any "spend fewer actions" lever is
that an agent which simply EXPLORES LESS also spends fewer actions, so raw action count is not
a metric. Every cell therefore reports, in decreasing decisiveness:

  * ``levels_gained`` / ``actions_to_first_levelup`` -- the RHAE-relevant oracle (the level
    counter), and the exact quantity ARC-AGI-3 scores: ``min((baseline/agent)**2, 115)``.
  * ``hv_progress_best_level`` -- the per-level hand-verifier distance-closed proxy, reusing
    the SHIPPED ``arc_actions_to_progress`` implementation (including its measurability guard,
    so an immovable verifier reports None rather than a 0.0 that a mean would swallow).
  * ``states_discovered`` -- distinct graph nodes reached. Dense, defined on every cell, and
    the search's own currency. Reported in BOTH directions: an arm that spends fewer actions
    and discovers less is a regression, and this is the number that says so.
  * the per-class action census (inert / new-state / known-state / navigation), so the lever's
    own target class can be checked for movement rather than assumed.

Never plays a scored or online game (``offline_arcade`` runs ``OperationMode.OFFLINE`` over the
local ``environment_files/`` tree: no API key, no network, no scorecard submission). Never
starts a generator and never touches a GPU.

EVIDENCE GUARD. ``results/arc_e3/<game>/world_model.py`` is TRACKED, READ-ONLY evidence, and the
module guard that would catch a write to it is pytest-scoped -- so a measurement driver is
precisely the caller nothing protects. This process refuses to start unless ``CARNOT_ARC_E3_DIR``
has already been redirected to scratch, checked AFTER import because ``E3_DIR`` is resolved at
import time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

PROBE_BRANCHES = {"depth_ride.pop_untested", "frontier.pop_untested"}


class _NoGeneratorStandIn:
    """Absorbs the config attribute writes a proposer would receive; constructs nothing."""

    include_playbook_exemplars = False
    no_think_prefix = ""
    max_tokens = 0
    tries = 0


def run_cell(
    game: str, *, seed: int, budget: int, wall_s: float, min_obs: Optional[int] = None
) -> dict[str, Any]:
    import random

    import numpy as np
    from arcengine import GameAction

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_actions_to_progress import (
        _hand_verifier_fn,
        hv_progress_measurable_from_stats,
        per_level_hv_progress,
    )
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_competition_agent import (
        E3AgentPolicy,
        _level_of,
        _recommend_live_approach,
        _route_explore_budget,
    )
    import carnot.agentic.arc_strategy_router as arc_strategy_router

    random.seed(seed)
    np.random.seed(seed)

    rec = _recommend_live_approach(game)
    strategy = dict(rec.get("strategy") or arc_strategy_router.route_for_game(game))
    explore_budget = int(_route_explore_budget(strategy))

    t0 = time.time()
    # `min_obs` is the evidence-floor SENSITIVITY knob, and it is passed as a PRE-BUILT
    # instance rather than an env var on purpose: `StepwiseExplorer` treats an injected
    # instance as an explicit enable, so a `--min-obs` arm activates the lever through the
    # same constructor path the flag does, with only the floor differing. When it is None the
    # arm is decided entirely by `CARNOT_ARC_INERT_LABEL_DEFER`, so the control and defer arms
    # are unaffected by this parameter existing.
    memory_kwargs: dict[str, Any] = {}
    if min_obs is not None:
        from carnot.agentic.arc_inert_label_memory import InertLabelMemory

        memory_kwargs["inert_label_memory"] = InertLabelMemory(min_observations=int(min_obs))
    pol = E3AgentPolicy(
        game, proposer=_NoGeneratorStandIn(), explore_budget=explore_budget, **memory_kwargs
    )
    explorer = pol.explorer
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    hv_fn = _hand_verifier_fn(game)

    frames: list[Any] = []
    latest: Any = None
    trace: list[str] = []
    class_counts: dict[str, int] = {}
    start_level: Optional[int] = None
    best_level = 0
    actions_to_first_levelup: Optional[int] = None
    prev_raw: Optional[str] = None
    timed_out = False
    hv_per_level: dict[int, dict[str, float]] = {}
    hv_level_seen: list[int] = []
    levels_entered: list[int] = []
    levels_entered_set: set[int] = set()
    n_actions = 0

    def bump(name: str) -> None:
        class_counts[name] = class_counts.get(name, 0) + 1

    for i in range(budget):
        if time.time() - t0 > wall_s:
            timed_out = True
            break
        if pol.is_done(frames, latest):
            break
        kind, data = pol.next_move(frames, latest)
        if kind is None:
            break
        branch = getattr(explorer, "_prov_branch", None)
        serve_kind = getattr(explorer, "_prov_serve_kind", None)

        if kind == "RESET":
            latest = env.reset()
            trace.append("RESET")
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            trace.append(
                json.dumps({"action": int(kind), "data": data}, sort_keys=True)
                if data
                else json.dumps({"action": int(kind)}, sort_keys=True)
            )
        frames.append(latest)
        n_actions += 1

        try:
            raw = frame_hash(grid_of(latest)) if latest is not None else None
        except Exception:
            raw = None
        # RAW identity, computed WITHOUT the explorer's HUD mask: the masked hash answers "is
        # this new to the search", this answers the strictly cheaper "did any pixel change".
        unchanged = bool(prev_raw is not None and raw is not None and raw == prev_raw)
        prev_raw = raw if raw is not None else prev_raw

        lvl = _level_of(latest)
        if start_level is None:
            start_level = lvl
        if lvl > best_level and start_level is not None and lvl > start_level:
            if actions_to_first_levelup is None:
                actions_to_first_levelup = n_actions
        best_level = max(best_level, lvl)
        if lvl not in levels_entered_set:
            levels_entered_set.add(lvl)
            levels_entered.append(lvl)

        if hv_fn is not None and latest is not None:
            hv = hv_fn(getattr(env, "_game", None), latest)
            if hv is not None:
                if lvl not in hv_per_level:
                    hv_per_level[lvl] = {"start": float(hv), "best": float(hv)}
                    hv_level_seen.append(lvl)
                elif hv < hv_per_level[lvl]["best"]:
                    hv_per_level[lvl]["best"] = float(hv)

        # Per-action class. Same vocabulary as the 2026-08-02 census, minus the sub-classes of
        # navigation it does not need: what matters here is inert-vs-productive-vs-navigation.
        if branch == "bootstrap_reset":
            bump("bootstrap.reset")
        elif (getattr(pol, "_prov_top", None) or "").startswith("execute."):
            bump("plan.execute_step")
        elif serve_kind == "reset" or kind == "RESET":
            bump("renavigation.reset_action")
        elif serve_kind == "probe" or branch in PROBE_BRANCHES:
            bump(
                "expansion.probe_was_inert_frame_unchanged"
                if unchanged
                else "expansion.probe_moved_the_board"
            )
        elif serve_kind == "navigation":
            bump("renavigation.walk_or_replay")
        else:
            bump("other")

    hv_stats = getattr(hv_fn, "stats", None) if hv_fn is not None else None
    hv_measurable = hv_progress_measurable_from_stats(hv_stats)
    per_level, hv_best = per_level_hv_progress(
        hv_per_level, hv_level_seen, hv_measurable, levels_entered=levels_entered
    )

    try:
        defer_diag = explorer.inert_label_defer_diagnostics()
    except Exception as exc:  # pragma: no cover - diagnostics must never fail a cell
        defer_diag = {"error": f"{type(exc).__name__}: {exc}"}

    import hashlib

    return {
        "game": game,
        "seed": seed,
        "budget": budget,
        "explore_budget": explore_budget,
        "actions_spent": n_actions,
        "start_level": start_level,
        "best_level": best_level,
        "levels_gained": int(best_level - (start_level or 0)),
        "actions_to_first_levelup": actions_to_first_levelup,
        "states_discovered": len(explorer.graph or {}),
        "class_counts": class_counts,
        "inert_actions": class_counts.get("expansion.probe_was_inert_frame_unchanged", 0),
        "navigation_actions": (
            class_counts.get("renavigation.walk_or_replay", 0)
            + class_counts.get("renavigation.reset_action", 0)
        ),
        "hv_progress_per_level": per_level,
        "hv_progress_best_level": hv_best,
        "hv_progress_measurable": hv_measurable,
        "inert_label_defer_diagnostics": defer_diag,
        "trace_sha256": hashlib.sha256("\n".join(trace).encode()).hexdigest(),
        "trace_len": len(trace),
        "timed_out": timed_out,
        "wall_s": round(time.time() - t0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--wall-s", type=float, default=1200.0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--min-obs", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from carnot.agentic.arc_executable_world_model import E3_DIR, _TRACKED_E3_EVIDENCE_DIR

    if E3_DIR.resolve() == _TRACKED_E3_EVIDENCE_DIR.resolve():
        print(
            "[worker] REFUSING: E3_DIR resolves to the tracked evidence store "
            f"({_TRACKED_E3_EVIDENCE_DIR}). Set CARNOT_ARC_E3_DIR to scratch BEFORE the "
            "interpreter starts.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") != "1":
        print("[worker] REFUSING: CARNOT_ARC_DISABLE_INDUCTION must be 1.", file=sys.stderr)
        return 2

    out = run_cell(
        args.game,
        seed=args.seed,
        budget=args.budget,
        wall_s=args.wall_s,
        min_obs=args.min_obs,
    )
    out["min_obs_arg"] = args.min_obs
    out["arm"] = args.arm
    out["flag_env"] = os.environ.get("CARNOT_ARC_INERT_LABEL_DEFER")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(
        f"[worker] {args.arm}/{args.game}/{args.seed}: {out['actions_spent']} actions, "
        f"lvl+{out['levels_gained']}, inert={out['inert_actions']}, "
        f"states={out['states_discovered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
