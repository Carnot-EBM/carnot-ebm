"""ARC-AGI-3 replay meta-harness — assemble our recorded per-game wins into ONE
offline EnvironmentScorecard. ZERO quota (OperationMode.OFFLINE + local
environment_files); no re-solving — deterministically replays the winning
`solve_trace.actions` banked in each game's best artifact.

This produces the "what we'd publish" scorecard locally so the operator can review
it BEFORE any operator-gated live scored run mirrors it. No network, no submission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_over

# game -> best banked artifact carrying solve_trace.actions (highest real-env level)
GAME_ARTIFACTS = {
    "r11l": "results/experiment_4296_arc_incremental_progress_new_game.json",  # L1
    # sc25 reproduces L1 via the E3 cast-grid plan (executed_steps with corrected coords) + a warmup
    # step (the first env.step after reset is consumed on sc25 -- see WARMUP_GAMES).
    "sc25": "results/experiment_4341_e3_sc25_reproduction.json",                # L1
    "lp85": "results/experiment_4190_arc_incremental_progress.json",            # L3
    "ls20": "results/experiment_4285_arc_incremental_progress_new_game.json",   # L1
    "wa30": "results/experiment_4275_arc_incremental_progress_new_game.json",   # L1
    # 2026-06-17 adapter-free sweep solves (graph_explore_solve_v2); trajectory format
    "cd82": "results/arc_explore_trajectory_cd82.json",                          # L1
    "sp80": "results/arc_explore_trajectory_sp80.json",                          # L1
    "su15": "results/arc_explore_trajectory_su15.json",                          # L1
    "tu93": "results/arc_explore_trajectory_tu93.json",                          # L1
    # E1-enabled sweep solves (salience + HUD-mask explorer upgrade)
    "cn04": "results/arc_explore_trajectory_cn04.json",                          # L1
    "m0r0": "results/arc_explore_trajectory_m0r0.json",                          # L1
    "sk48": "results/arc_explore_trajectory_sk48.json",                          # L1
    # PROGRAM-EDITOR class solved FRAME-ONLY (the frame approach: detect->route->perceive->plan->
    # solve; full frame-only maze solve L6/L7, 2026-06-17). tn36 is the deepest single-game solve.
    "tn36": "results/arc_explore_trajectory_tn36.json",                          # L7 (frame approach)
    # E3 explore-verify-plan world-model solves (executed_steps under plan_executed_detail)
    "ar25": "results/experiment_4339_e3_explore_verify_plan_ar25.json",          # L1
    "ka59": "results/experiment_4350_e3_explore_verify_plan_ka59.json",          # L1
}

# from-scratch OFFLINE re-solves (these replay deterministically on the offline
# layout; take precedence over the non-reproducible live-recorded banked traces)
RESOLVED_ARTIFACTS = {
    # L4 re-derived offline (solve_adaptered + learned verifier), reproduction-gated reached_level=4
    "lp85": "results/arc_explore_trajectory_lp85_l4.json",  # L4
    # L3 re-derived offline (GameAdapter + OfflineSolver branch_mode='fresh_env' for tu93's
    # non-idempotent reset, gotcha #7), reproduction-gated reached_level=3, 47-action path
    "tu93": "results/arc_loop_solve_tu93.json",             # L3 (was L2 graph_explore)
}


def load_actions(path: str) -> list[dict]:
    d = json.load(open(REPO / path))
    for keys in (("solution",), ("trajectory",), ("solve_trace", "actions"), ("solver_trace", "actions"),
                 ("plan_executed_detail", "plan_result", "executed_steps")):
        cur = d
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, list) and cur:
            return cur
    return []


def normalize(a: dict) -> tuple[int | None, dict | None]:
    aid = a.get("action")
    if "data" in a:                          # arc_explore_trajectory format: {"action", "data": {x,y}|null}
        return (int(aid) if aid is not None else None), a.get("data")
    x = a.get("x", a.get("world_x"))
    y = a.get("y", a.get("world_y"))
    has_xy = x is not None and y is not None
    if aid is None:
        aid = 6 if has_xy else None      # ACTION6 = click
    data = {"x": int(x), "y": int(y)} if has_xy else None
    return (int(aid) if aid is not None else None), data


# games whose FIRST env.step after reset is consumed (a no-op) -> prepend a throwaway warmup step
# so the real action sequence lands (the first-step-after-reset-consumed gotcha; sc25).
WARMUP_GAMES = {"sc25"}


def replay_game(arcade, game: str, scorecard_id: str, actions: list[dict]) -> dict:
    env = arcade.make(game, scorecard_id=scorecard_id)
    frame = env.reset()
    start = _levels_completed(frame)
    applied = 0
    if game in WARMUP_GAMES and actions:           # consume the swallowed first step
        aid, data = normalize(actions[0])
        if aid is not None:
            frame = env.step(getattr(GameAction, f"ACTION{aid}"), data=data,
                             reasoning={"policy": "warmup"})
    for a in actions:
        aid, data = normalize(a)
        if aid is None:
            continue
        ge = getattr(GameAction, f"ACTION{aid}")
        frame = env.step(ge, data=data, reasoning={"policy": "banked_win_replay"})
        applied += 1
        if frame is None:
            break
    reached = _levels_completed(frame) if frame is not None else -1
    return {"game": game, "actions_replayed": applied, "levels_completed": reached, "start": start}


def main() -> int:
    print("== ARC-AGI-3 REPLAY meta-harness -> aggregate offline scorecard (no quota) ==")
    arcade = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(REPO / "environment_files"))
    scorecard_id = arcade.open_scorecard()
    print(f"scorecard opened (offline): {scorecard_id}\n")

    rows, total = [], 0
    for game, art in GAME_ARTIFACTS.items():
        src = RESOLVED_ARTIFACTS.get(game, art)  # prefer the offline re-solve
        actions = load_actions(src)
        if not actions:
            print(f"  {game}: NO banked trajectory in {art}")
            continue
        r = replay_game(arcade, game, scorecard_id, actions)
        rows.append(r)
        total += max(0, r["levels_completed"])
        print(f"  {game:5} replayed {r['actions_replayed']:>3} actions -> levels_completed={r['levels_completed']}")

    card = arcade.get_scorecard(scorecard_id)
    closed = arcade.close_scorecard(scorecard_id)
    card = closed if closed is not None else card

    print("\n== AGGREGATE SCORECARD (the 'what we'd publish' card) ==")
    print(f"  games: {len(rows)}   total levels_completed: {total}")
    for r in rows:
        print(f"    {r['game']:5} L{r['levels_completed']}")
    print(f"  scorecard object: {type(card).__name__}  valid: {card is not None}")
    print(f"  OFFLINE — zero network/quota; nothing submitted.")
    out = REPO / "results" / "arc3_replay_aggregate_scorecard.json"
    out.write_text(json.dumps({"scorecard_id": scorecard_id, "total_levels": total,
                               "per_game": rows, "mode": "offline_replay_no_quota"}, indent=2))
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
