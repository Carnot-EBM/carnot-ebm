"""Step-1 local validation: confirm the ARC-AGI-3 scored pipeline produces a
valid scorecard ENTIRELY OFFLINE (OperationMode.OFFLINE + local environment_files),
spending ZERO arcprize.org rate-limit quota.

This proves the harness we would reuse for an eventual (operator-gated) live
scored run works end-to-end: Arcade.make -> reset -> step loop -> scorecard.
It uses the bounded exploration play-loop (mirrors arc_agi3_live_adapter); it is
NOT our best per-game solver, so levels_completed here reflects the pipeline, not
our headline progress. No network, no submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from carnot.agentic.arc_agi3_world_model import (
    GameGraph,
    compute_grid_delta,
    frame_hash,
    grid_of,
)
from carnot.agentic.arc_agi3_live_adapter import (
    _action_candidates,
    _game_action,
    _game_over,
    _levels_completed,
)

GAME = "r11l"
ACTION_BUDGET = 60


def main() -> int:
    print("== ARC-AGI-3 OFFLINE scorecard validation (no network, no quota) ==")
    arcade = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    envs = arcade.get_environments()
    game_ids = [getattr(e, "game_id", getattr(e, "id", str(e))) for e in envs]
    print(f"offline environments loaded: {len(game_ids)} -> {game_ids[:8]}{'...' if len(game_ids) > 8 else ''}")
    assert GAME in " ".join(map(str, game_ids)) or any(GAME in str(g) for g in game_ids), f"{GAME} not in offline set"

    scorecard_id = arcade.open_scorecard()
    print(f"scorecard opened (offline): {scorecard_id}")

    env = arcade.make(GAME, scorecard_id=scorecard_id)
    frame = env.reset()
    assert frame is not None, "reset returned no frame"
    graph = GameGraph(GAME)
    graph.see_node(frame_hash(grid_of(frame)), frame)

    used: set = set()
    actions_taken = 0
    start_levels = _levels_completed(frame)
    for i in range(1, ACTION_BUDGET + 1):
        prev = grid_of(frame)
        cur_hash = frame_hash(prev)
        cands = _action_candidates(frame)
        if not cands:
            print(f"  step {i}: no available actions; stop")
            break
        by_key = {c.key: c for c in cands}
        fresh = [c for c in cands if c.key not in used]
        sel = fresh[0] if fresh else (by_key[graph.untested(cur_hash, list(by_key))[0]]
                                      if graph.untested(cur_hash, list(by_key)) else cands[0])
        used.add(sel.key)
        nxt = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                       reasoning={"policy": "offline_validation_exploration"})
        actions_taken += 1
        if nxt is None:
            print(f"  step {i}: step returned no frame; stop")
            break
        delta = compute_grid_delta(prev, grid_of(nxt))
        lvl_delta = _levels_completed(nxt) - _levels_completed(frame)
        graph.record(cur_hash, sel.key, frame_hash(grid_of(nxt)), delta, lvl_delta, _game_over(nxt))
        graph.see_node(frame_hash(grid_of(nxt)), nxt)
        frame = nxt
        if lvl_delta > 0:
            print(f"  step {i}: LEVEL UP -> {_levels_completed(frame)}")
            break
        if _game_over(frame):
            print(f"  step {i}: game over; stop")
            break

    levels = _levels_completed(frame)
    card_open = arcade.get_scorecard(scorecard_id)   # read WHILE open
    card_closed = arcade.close_scorecard(scorecard_id)  # finalize; returns the card
    card = card_closed if card_closed is not None else card_open
    print("\n== RESULT ==")
    print(f"  game: {GAME}  actions_taken: {actions_taken}  levels_completed: {levels} (start {start_levels})")
    print(f"  scorecard (open read): {type(card_open).__name__}")
    print(f"  scorecard (close return): {type(card_closed).__name__}")
    if card is not None:
        for attr in ("cards", "scores", "guid", "scorecard_id", "won", "played"):
            if hasattr(card, attr):
                print(f"    scorecard.{attr}: {getattr(card, attr)!r}"[:160])
    valid = card is not None
    print(f"  scorecard valid: {valid}")
    print(f"  PIPELINE OFFLINE-VALID: {valid and frame is not None} (zero network calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
