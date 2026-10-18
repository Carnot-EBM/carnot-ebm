#!/usr/bin/env python3
"""Translate a source-planner route into real public lf52 engine actions.

Every claimed transition is made with ``env.step``.  The symbolic model is used
only to map source-grid coordinates through lf52's camera-follow offsets; no
private runtime object or level teleport is accessed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arcengine import GameAction  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action  # noqa: E402
from lf52_l8plus_source_planner import (  # noqa: E402
    CARRIER_ACTIONS,
    State,
    carrier_move,
    legal_jumps,
    parse_layout,
)


def action_label(action: int, data: dict[str, int] | None = None) -> str:
    row: dict[str, Any] = {"action": int(action)}
    if data is not None:
        row["data"] = {"x": int(data["x"]), "y": int(data["y"])}
    return json.dumps(row, separators=(",", ":"))


def click_for(point: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    return (point[0] * 6 + offset[0] + 1, point[1] * 6 + offset[1] + 1)


def visible_highlight_contains(frame: Any, click: tuple[int, int]) -> bool | None:
    x, y = click
    if x < 0 or y < 1 or x >= 64 or y >= 64:
        return None
    grid = np.asarray(frame.frame[-1])
    region = grid[max(1, y - 1) : min(64, y + 5), max(0, x - 1) : min(64, x + 5)]
    return bool(np.any(region == 2))


def execute(prefix_path: Path, plan_path: Path, level: int) -> dict[str, Any]:
    prefix_artifact = json.loads(prefix_path.read_text())
    prefix: list[str] = prefix_artifact["action_sequence"]
    plan_artifact = json.loads(plan_path.read_text())
    if not plan_artifact.get("found"):
        raise RuntimeError("planner artifact has no winning route")
    plan = plan_artifact["plan"]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lf52-plan-executor")
    arcade = kit.offline_arcade()
    env = arcade.make("lf52", scorecard_id=arcade.open_scorecard())
    frame = env.reset()
    env_steps = 0
    for encoded in prefix:
        row = json.loads(encoded)
        frame = env.step(_game_action(GameAction, int(row["action"])), data=row.get("data"))
        env_steps += 1
    if int(frame.levels_completed) != level - 1:
        raise RuntimeError(
            f"prefix reached levels_completed={frame.levels_completed}, expected {level - 1}"
        )

    layout = parse_layout(level)
    symbolic: State = layout.initial
    offset = [5, 3 if level == 10 else 5]
    suffix: list[str] = []
    trace: list[dict[str, Any]] = []
    delta_by_action = {action: delta for action, delta in CARRIER_ACTIONS}

    for plan_index, raw_action in enumerate(plan, 1):
        kind = raw_action[0]
        before_level = int(frame.levels_completed)
        if kind == "C":
            action = int(raw_action[1])
            delta = delta_by_action[action]
            successor = carrier_move(layout, symbolic, delta)
            if successor == symbolic:
                raise RuntimeError(f"planner carrier no-op at plan step {plan_index}")
            old_plain_carriers = {
                (x, y) for x, y, payload in symbolic.carriers if payload == "P"
            }
            new_plain_carriers = {
                (x, y) for x, y, payload in successor.carriers if payload == "P"
            }
            plain_carrier_moved = old_plain_carriers != new_plain_carriers
            frame = env.step(_game_action(GameAction, action))
            env_steps += 1
            suffix.append(action_label(action))
            if plain_carrier_moved:
                if level == 8:
                    offset[1] -= delta[1] * 6
                elif level == 9:
                    offset[0] -= delta[0] * 6
                    offset[1] -= delta[1] * 6
                # L10 intentionally has no camera-follow offset in tmhxwcojkh.
            symbolic = successor
            trace.append(
                {
                    "plan_step": plan_index,
                    "kind": "carrier",
                    "action": action,
                    "delta": delta,
                    "plain_carrier_moved": plain_carrier_moved,
                    "offset": offset.copy(),
                    "levels_completed": int(frame.levels_completed),
                    "state": str(frame.state),
                }
            )
        elif kind == "J":
            source = tuple(int(v) for v in raw_action[1][0])
            destination = tuple(int(v) for v in raw_action[1][1])
            matches = [
                successor
                for jump_source, jump_destination, successor in legal_jumps(layout, symbolic)
                if jump_source == source and jump_destination == destination
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one symbolic jump {source}->{destination} at step {plan_index}, "
                    f"found {len(matches)}"
                )
            source_click = click_for(source, tuple(offset))
            destination_click = click_for(destination, tuple(offset))
            source_data = {"x": source_click[0], "y": source_click[1]}
            destination_data = {"x": destination_click[0], "y": destination_click[1]}
            frame = env.step(_game_action(GameAction, 6), data=source_data)
            env_steps += 1
            suffix.append(action_label(6, source_data))
            highlight = visible_highlight_contains(frame, destination_click)
            if highlight is False:
                raise RuntimeError(
                    f"engine oracle did not highlight {destination_click} for symbolic "
                    f"jump {source}->{destination} at plan step {plan_index}"
                )
            frame = env.step(_game_action(GameAction, 6), data=destination_data)
            env_steps += 1
            suffix.append(action_label(6, destination_data))
            symbolic = matches[0]
            if level == 9 and destination == (6, 5):
                # L9 deliberately reveals the east board when any jumper lands
                # on the entrance carrier at source-grid (6,5).
                offset[0] -= 20
            trace.append(
                {
                    "plan_step": plan_index,
                    "kind": "jump",
                    "source": source,
                    "destination": destination,
                    "source_click": source_click,
                    "destination_click": destination_click,
                    "destination_highlight_visible": highlight,
                    "offset": offset.copy(),
                    "levels_completed": int(frame.levels_completed),
                    "state": str(frame.state),
                }
            )
        else:
            raise RuntimeError(f"unknown plan action {raw_action!r}")

        after_level = int(frame.levels_completed)
        print(
            "engine_step",
            json.dumps(trace[-1], sort_keys=True),
            flush=True,
        )
        if after_level > before_level and plan_index != len(plan):
            raise RuntimeError(f"level advanced early at plan step {plan_index}")

    full_sequence = prefix + suffix
    return {
        "game": "lf52",
        "level_attempted": level,
        "starting_levels_completed": level - 1,
        "final_levels_completed": int(frame.levels_completed),
        "final_state": str(frame.state),
        "plan_steps": len(plan),
        "suffix_action_count": len(suffix),
        "full_action_count": len(full_sequence),
        "env_steps": env_steps,
        "final_offset": offset,
        "suffix": suffix,
        "action_sequence": full_sequence,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(8, 9, 10), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.prefix, args.plan, args.level)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESULT",
        json.dumps({key: value for key, value in result.items() if key not in {"trace", "suffix", "action_sequence"}}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
