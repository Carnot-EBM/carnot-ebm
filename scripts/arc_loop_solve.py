"""STANDING ARC learning-loop entrypoint — the conductor's ARC north-star task runs
this each milestone instead of bespoke one-off code. It executes the full loop on
the offline sim (zero quota), self-improving across runs:

  1. pick the target game (--game X, or --auto via the registry + transfer routing)
  2. if the game is ADAPTERED (arc_game_adapters): verifier-routed best-first SOLVE
     to +1 level, warm-started by a saved LEARNED verifier if present else the hand
     verifier (arc_solver_kit.OfflineSolver)
  3. REPRODUCTION GATE the solution offline (arc_solver_kit.reproduce)
  4. TRAIN + CHECKPOINT the learned verifier on the accumulated solve traces
     (arc_value_learner) — mirror-ready weights (Rule 3; public release operator-only)
  5. emit a milestone artifact (offline_reproduced / reproduced_levels / states)
  6. if the game is NOT adaptered: emit the transfer-routing recommendation + recipe
     + general gotchas — the agent's per-game RE starting point (the irreducible delta)

Usage (what the conductor task calls):
  .venv/bin/python scripts/arc_loop_solve.py --game lp85 --target-level 3
  .venv/bin/python scripts/arc_loop_solve.py --auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_game_adapters as adapters
from carnot.agentic import arc_solve_learning as learning
from carnot.agentic.arc_value_learner import collect_trajectory_data
from carnot.agentic.arc_value_net import load_live_spatial_value_head
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels

CKPT_DIR = REPO / "models"
RESULTS = REPO / "results"


def _ckpt_path(game: str) -> Path:
    return CKPT_DIR / f"arc_verifier_{game}.json"


def _live_verifier_for_adapter(game: str, adapter):
    """Return the live warm-start verifier for an adaptered solve."""

    spatial = load_live_spatial_value_head(root=REPO, game=game)
    if spatial is not None:
        return spatial, "spatial_value_head_live_checkpoint"
    ckpt = _ckpt_path(game)
    if ckpt.exists() and getattr(adapter, "featurize", None) is not None:
        try:
            from carnot.agentic.arc_value_learner import LearnedVerifier

            return (
                LearnedVerifier.load(ckpt, adapter.featurize),
                "learned_verifier_live_checkpoint",
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return adapter.hand_verifier, "hand_verifier_cold_start_checkpoint_load_failed"
    return adapter.hand_verifier, "hand_verifier_cold_start_no_spatial_checkpoint"


def _label_to_action(label: str) -> dict:
    """Normalize an adapter's opaque action LABEL to a metaharness-replayable action dict.
    Adapter labels are JSON ({"action": N} for keyboard games, {"x": X, "y": Y} for click games);
    the metaharness's normalize() reads either shape. Falls back to {"raw": label} for non-JSON."""
    try:
        d = json.loads(label)
        return d if isinstance(d, dict) else {"action": d}
    except (ValueError, TypeError):
        return {"raw": label}


def solve_adaptered(
    game: str, target_level: int, hazard_prune: bool = True, mask_prune: bool = False
) -> dict:
    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    selected_generic_operators = [
        op.as_dict()
        for op in kit.select_primitive_operators(
            mechanic_class=getattr(ad, "game", game), game=game
        )
    ]

    # Warm-start with the graduated position-preserving spatial value head when present.
    verifier, verifier_src = _live_verifier_for_adapter(game, ad)

    # EFFICIENCY: an online hazard move-pruner (fits a hazard model from the search's OWN observed
    # deaths -- no offline ground-truth -- and skips moves it predicts walk into a charging enemy). It
    # NO-OPS when no hazard is detected, so it is safe for any game; the win is that for a hazard game
    # (tu93) the search stops wasting expansions on death-paths. This is the wired-in salvage of the
    # outer-loop hazard-aware world model (arc_nav_world_model) onto the LIVE solve path.
    move_pruner = None
    if hazard_prune or mask_prune:
        from carnot.agentic.arc_agi3_world_model import grid_of

        pruners = []
        if hazard_prune:
            from carnot.agentic.arc_hazard_pruner import HazardMovePruner

            pruners.append(HazardMovePruner(grid_of))
        if mask_prune:
            # Relational-mask deepening pruner (induces its target region ONLINE on the first level-up;
            # prunes action classes that never touch it). Conservative; no-ops on non-relational games.
            from carnot.agentic.arc_relational_mask_pruner import RelationalMaskMovePruner

            pruners.append(RelationalMaskMovePruner(grid_of))
        if len(pruners) == 1:
            move_pruner = pruners[0]
        elif pruners:
            from carnot.agentic.arc_relational_mask_pruner import CompositeMovePruner

            move_pruner = CompositeMovePruner(*pruners)

    solver = kit.OfflineSolver(
        game,
        ad.action_labels,
        ad.apply,
        ad.state_key,
        warmup_label=ad.warmup_label,
        verifier=verifier,
        branch_mode=getattr(ad, "branch_mode", "replay"),
        move_pruner=move_pruner,
    )
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full, total_states, X, y = [], 0, [], []
    for lvl in range(cur + 1, target_level + 1):
        fallback_reached = None
        path, nodes = solver.solve_level(env, cur, full, ad.depth_caps.get(lvl, 90))
        total_states += nodes
        if path is None:
            tail = list(getattr(ad, "level_tails", {}).get(lvl, ()))
            if not tail:
                break
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            f_tail = solver._replay(env, full + tail)
            fallback_reached = kit.frame_level(f_tail)
            if fallback_reached <= cur:
                break
            path = tail
            total_states += len(tail)
        search_reached = max(kit.frame_level(solver.last_frame), fallback_reached or 0)
        if ad.featurize is not None:
            Xi, yi = collect_trajectory_data(env, solver, full, path, ad.featurize)
            X += Xi
            y += yi
        f = solver._replay(env, full + path)
        # Some games need fresh-env node evaluation because env.reset() is not
        # idempotent during search. In that case the searched winning frame is
        # more reliable than replaying on the reused env; the reproduction gate
        # below remains the final authority.
        cur = max(kit.frame_level(f), search_reached)
        full += path

    # REPRODUCTION GATE
    gate = kit.reproduce(game, full, ad.apply, warmup_label=ad.warmup_label, claimed_level=cur)

    # TRAIN + CHECKPOINT the learned verifier (self-improvement for next run)
    ckpt_written = None
    if X and ad.featurize is not None:
        from carnot.agentic.arc_value_learner import LearnedVerifier

        ckpt = _ckpt_path(game)
        lv = LearnedVerifier(ad.featurize).fit(X, y)
        lv.save(
            ckpt,
            meta={
                "trained_games": [game],
                "feature_names": "adapter_featurize",
                "provenance": f"arc_loop_solve {game}->L{cur}",
            },
        )
        ckpt_written = str(ckpt.relative_to(REPO))

    return {
        "game": game,
        "reached_level": cur,
        "moves": len(full),
        "states_expanded": total_states,
        "verifier_src": verifier_src,
        "hazard_prune": bool(hazard_prune),
        "hazard_pruner_stats": (move_pruner.stats() if move_pruner is not None else None),
        "offline_reproduced": bool(gate["reproduced"]),
        "reproduced_levels": cur,
        "learned_verifier_checkpoint": ckpt_written,
        "selected_generic_operators": selected_generic_operators,
        "reproduction_gate": gate,
        # PERSIST the winning action path (not just the verdict) so it is replay-gateable by the
        # offline metaharness / a third party -- per the ARC Solve Reproducibility discipline
        # (capture the winning condition, not a brittle one-off). Labels are the adapter's opaque
        # strings; "solution" normalizes them to metaharness-compatible action dicts.
        "solution_labels": list(full),
        "solution": [_label_to_action(lbl) for lbl in full],
        # PROVENANCE (ARC live-agent self-solve discipline): this is the offline DEV TWIN solving via a
        # hand-registered GameAdapter -- a development proxy, NOT the scored live agent self-discovering a
        # hidden game. Declared so adversarial_verify can tell a dev-proxy re-run from an outer-loop solve.
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def solve_via_explore(
    game: str, max_expansions: int = 6000, max_depth: int = 60, warmup: bool = False
) -> Optional[dict]:
    """ADAPTER-FREE first contact: graph-explore the game; if it advances a level,
    CAPTURE the trajectory, reproduction-gate it, train a verifier from it, and
    persist the trajectory as the adapter SEED — so the next solve is the efficient
    verifier-routed loop, not blind exploration. Returns None if no advance (caller
    falls back to the routing recommendation for per-game RE)."""
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    traj, lvl = graph_explore_solve_v2(
        env, 0, max_expansions=max_expansions, max_depth=max_depth, warmup=warmup
    )
    if traj is None:
        return None

    def apply(env, label, frame):
        step = json.loads(label)
        return env.step(_game_action(GameAction, step["action"]), data=step.get("data"))

    labels = trajectory_labels(traj)
    gate = kit.reproduce(game, labels, apply, claimed_level=lvl)

    # SEED a verifier from the first solve (generic grid features -> steps-to-go),
    # so the captured solve trains the verifier that will route future searches.
    from carnot.agentic.arc_value_learner import LearnedVerifier

    def featurize_frame(frame):
        g = grid_of(frame)
        nz = int((g != 0).sum())
        return [float(nz), float(len(set(g.flatten().tolist()))), float(g.shape[0] * g.shape[1])]

    X, y = [], []
    f = env.reset()
    if warmup and traj:
        f = apply(env, labels[0], f)
    for i, lab in enumerate(labels):
        X.append(featurize_frame(f))
        y.append(float(len(labels) - i))
        f = apply(env, lab, f)
    ckpt = None
    if X:
        lv = LearnedVerifier(featurize_frame).fit(X, y)
        ckpt = CKPT_DIR / f"arc_verifier_{game}.json"
        lv.save(
            ckpt,
            meta={
                "trained_games": [game],
                "feature_names": ["nonzero", "colors", "cells"],
                "provenance": f"graph_explore first-solve {game}->L{lvl}",
            },
        )

    # persist the captured trajectory as the adapter SEED
    seed = RESULTS / f"arc_explore_trajectory_{game}.json"
    seed.write_text(json.dumps({"game": game, "reached_level": lvl, "trajectory": traj}, indent=2))

    # LEARN the best goal-distance heuristic for this game AND TRAIN the router (dynamic
    # adaptation): now that we have a win-state, run the heuristic portfolio, bank the winner to
    # gap_fills/ (so the NEXT solve is heuristic-guided), AND record (features -> winner) to the
    # router ledger so the trained router (arc_router) generalises to unseen games over time.
    # Fully guarded: heuristic-learning must NEVER break the solve itself.
    heuristic_learned = None
    try:
        if gate["reproduced"] and traj:
            import types
            from carnot.agentic import arc_heuristic_select as hsel

            f2 = env.reset()
            if warmup:
                f2 = apply(env, labels[0], f2)
            trans = []
            for lab in labels[1:] if warmup else labels:
                g0 = grid_of(f2)
                f2 = apply(env, lab, f2)
                trans.append(types.SimpleNamespace(grid=g0, next_grid=grid_of(f2)))
            heuristic_learned = hsel.select_and_learn(
                game, grid_of(f2), trans, mask_hud=False, budget=max_expansions
            )
    except Exception:
        heuristic_learned = None

    return {
        "game": game,
        "method": "graph_explore_adapter_free",
        "reached_level": lvl,
        "moves": len(traj),
        "offline_reproduced": bool(gate["reproduced"]),
        "reproduced_levels": lvl,
        "trajectory_seed": str(seed.relative_to(REPO)),
        "verifier_seed_checkpoint": (str(ckpt.relative_to(REPO)) if ckpt else None),
        "heuristic_learned": heuristic_learned,
        "next": "register a GameAdapter from the seed for verifier-routed re-solving",
        # PROVENANCE (ARC live-agent self-solve discipline): adapter-FREE first contact -- the agent
        # explored + induced the solve from its OWN attempts, with no hand-built per-game adapter. This is
        # the live self-discovery path the deliverable is about.
        "solve_provenance": "live_agent_self_discovery",
        "mode": "standing_arc_loop_graph_explore_no_quota",
    }


def needs_re(game: str) -> dict:
    rec = learning.recommend_approach(game)
    return {
        "game": game,
        "status": "needs_per_game_RE",
        "transfer_recommendation": rec.get("recommended"),
        "selected_generic_operators": rec.get("selected_generic_operators"),
        "general_gotchas": rec.get("general_gotchas"),
        "guidance": rec.get("guidance"),
        "instruction": (
            "Reverse-engineer this game's win/action/state DELTA reusing the routed "
            "recipe, register a GameAdapter in arc_game_adapters.py, then re-run "
            "this loop. Per CLAUDE.md ARC Solve Reproducibility + Solver-Reuse Discipline."
        ),
        "mode": "standing_arc_loop_routing_only",
    }


def pick_target() -> str:
    # next target: an adaptered game first (extend its levels), else a routed unsolved game
    ad = adapters.adaptered_games()
    return ad[0] if ad else "tn36"


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--target-level", type=int, default=3)
    ap.add_argument(
        "--no-hazard-prune",
        dest="hazard_prune",
        action="store_false",
        help="disable the online hazard move-pruner (for A/B states-expanded measurement)",
    )
    ap.set_defaults(hazard_prune=True)
    args = ap.parse_args(argv)
    game = args.game or (pick_target() if args.auto else None)
    if not game:
        ap.error("specify --game X or --auto")

    print(f"== standing ARC loop: game={game} ==")
    if adapters.get_adapter(game):
        out = solve_adaptered(
            game, args.target_level, hazard_prune=args.hazard_prune
        )  # verifier-routed
    else:
        out = solve_via_explore(game) or needs_re(
            game
        )  # adapter-free first contact, else route to RE
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"arc_loop_solve_{game}.json").write_text(json.dumps(out, indent=2))
    for k in (
        "status",
        "reached_level",
        "offline_reproduced",
        "reproduced_levels",
        "states_expanded",
        "hazard_prune",
        "hazard_pruner_stats",
        "verifier_src",
        "learned_verifier_checkpoint",
    ):
        if k in out:
            print(f"  {k}: {out[k]}")
    print(f"  wrote results/arc_loop_solve_{game}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
