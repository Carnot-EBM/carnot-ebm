"""Public-frame-only bp35 level-9 characterization for outer-loop round 26.

This script intentionally knows nothing about the game implementation.  It replays
the banked public action prefix, issues only ``env.step`` actions, and derives its
measurements solely from returned 64x64 grids, state, and levels-completed fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action  # noqa: E402


PREFIX_PATH = REPO / "results/outer_loop_round12_bp35_probe_20260712.json"
SWITCH_WORLD_Y = (-81, -57, -45, -21, 27, 39)


def click(x: int, y: int) -> dict[str, Any]:
    return {"action": 6, "data": {"x": x, "y": y}}


def move(action: int) -> dict[str, Any]:
    return {"action": action}


COMMON = [
    click(39, 15),
    move(4),
    move(4),
    click(33, 33),
    click(33, 33),
    click(33, 33),
    click(27, 39),
    move(1),
    click(27, 33),
    click(27, 33),
    click(21, 39),
    move(1),
    click(15, 39),
    move(1),
    click(3, 3),
]

HIGH_UP = {
    27: COMMON[:10],
    21: COMMON[:12],
    15: COMMON[:14],
}


def top_rail_tail() -> list[dict[str, Any]]:
    """Public action route to the alive x15.5/y-75.5 upper rail rest."""
    rows = COMMON[:6] + [
        click(27, 21),
        click(27, 39),
        move(1),
    ]
    rows += [click(27, 33)] * 8
    rows += [
        click(33, 39),
        move(4),
        click(39, 39),
        move(4),
        click(45, 39),
        move(4),
        click(3, 27),
        click(45, 33),
        click(45, 33),
        click(51, 29),
        move(4),
        click(57, 29),
        move(4),
        click(3, 17),
    ]
    rows += [click(57, 33)] * 7
    for column in (51, 45, 39, 33, 27, 21, 15):
        rows += [click(column, 39), move(1)]
    return rows


def winning_l9_tail() -> list[dict[str, Any]]:
    """The fixed 68-action public L9 clear discovered in round 26."""
    rows = top_rail_tail()[:-10] + [click(45, 33), click(45, 33)]
    for column in (39, 33, 27, 21, 15):
        rows += [click(column, 39), move(1)]
    rows += [click(9, 39), move(1), move(1)]
    rows += [
        click(9, 3),
        click(3, 35),
        click(15, 3),
        click(3, 35),
        click(21, 3),
        click(3, 35),
        click(27, 3),
        click(3, 35),
        click(33, 3),
        move(4),
        move(4),
    ]
    assert len(rows) == 68
    return rows


def public_summary(run: PublicRun) -> dict[str, Any]:
    player = player_screen_center(run.settled)
    return {
        "offline_steps": run.offline_steps,
        "state": str(run.frame.state),
        "levels_completed": int(run.frame.levels_completed),
        "camera_y": run.camera_y,
        "player_screen": list(player) if player is not None else None,
        "player_world": ([player[0], player[1] + run.camera_y] if player is not None else None),
        "sha256": raw_hash(run.settled),
        "colors": {
            str(int(color)): int(count)
            for color, count in zip(*np.unique(run.settled, return_counts=True), strict=True)
        },
    }


def components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen: set[tuple[int, int]] = set()
    found: list[list[tuple[int, int]]] = []
    for y_raw, x_raw in zip(*np.where(mask), strict=True):
        start = (int(y_raw), int(x_raw))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (y + dy, x + dx)
                if 0 <= nxt[0] < 64 and 0 <= nxt[1] < 64 and nxt not in seen and bool(mask[nxt]):
                    seen.add(nxt)
                    stack.append(nxt)
        found.append(comp)
    return found


def player_screen_center(grid: np.ndarray) -> tuple[float, float] | None:
    ys_raw, xs_raw = np.where(grid == 9)
    if len(xs_raw) == 0:
        return None
    # The six-pixel sprite is intentionally asymmetric, so its pixel centroid is
    # displaced by 1/6 row or column.  The public 4x2/2x4 bounding-box midpoint is
    # the stable geometric center, and accepting partial edge clips lets us record
    # the exact first layer on which an offscreen sprite is reacquired.
    return (
        (float(np.min(xs_raw)) + float(np.max(xs_raw))) / 2.0,
        (float(np.min(ys_raw)) + float(np.max(ys_raw))) / 2.0,
    )


def switch_screen_centers(grid: np.ndarray) -> list[float]:
    centers: list[float] = []
    for comp in components(grid, 8):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        if len(comp) == 21 and max(xs) - min(xs) == 4 and max(ys) - min(ys) == 4:
            centers.append(float(np.mean(ys)))
    return sorted(centers)


def camera_from_switches(grid: np.ndarray, previous: int) -> int | None:
    screens = switch_screen_centers(grid)
    if not screens:
        return None
    candidates = {int(round(world - screen)) for world in SWITCH_WORLD_Y for screen in screens}
    valid: list[int] = []
    for camera in candidates:
        projected = sorted(
            float(world - camera) for world in SWITCH_WORLD_Y if -2 <= world - camera <= 65
        )
        if all(any(abs(screen - value) < 0.1 for value in projected) for screen in screens):
            valid.append(camera)
    if not valid:
        return None
    return min(valid, key=lambda value: (abs(value - previous), abs(value)))


def overlap_for_camera_delta(
    previous: np.ndarray, current: np.ndarray, delta: int
) -> tuple[np.ndarray, np.ndarray]:
    if delta >= 0:
        return previous[delta:, :], current[: 64 - delta, :]
    return previous[: 64 + delta, :], current[-delta:, :]


def camera_from_registration(
    previous_grid: np.ndarray, current_grid: np.ndarray, previous_camera: int
) -> int:
    scored: list[tuple[int, int, int]] = []
    for delta in range(-3, 4):
        old, new = overlap_for_camera_delta(previous_grid, current_grid, delta)
        static = ~np.isin(old, (8, 9, 11, 14, 15)) & ~np.isin(new, (8, 9, 11, 14, 15))
        informative = static & ((old != 0) | (new != 0))
        score = int(np.sum((old == new) & informative))
        scored.append((score, -abs(delta), delta))
    delta = max(scored)[2]
    return previous_camera + delta


def infer_camera(
    previous_grid: np.ndarray,
    grid: np.ndarray,
    previous_camera: int,
) -> int:
    by_switch = camera_from_switches(grid, previous_camera)
    if by_switch is not None and abs(by_switch - previous_camera) <= 3:
        return by_switch
    return camera_from_registration(previous_grid, grid, previous_camera)


def raw_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(grid, dtype=np.int8).tobytes()).hexdigest()


@dataclass
class PublicRun:
    env: Any
    frame: Any
    settled: np.ndarray
    camera_y: int
    offline_steps: int


def load_prefix() -> list[dict[str, Any]]:
    labels = json.loads(PREFIX_PATH.read_text())["action_sequence"]
    assert len(labels) == 342
    return [json.loads(label) for label in labels]


def public_step(env: Any, row: dict[str, Any]) -> Any:
    return env.step(_game_action(GameAction, int(row["action"])), data=row.get("data"))


def fresh_l9() -> PublicRun:
    arc = kit.offline_arcade()
    env = arc.make("bp35", scorecard_id=arc.open_scorecard())
    frame = env.reset()
    for row in load_prefix():
        frame = public_step(env, row)
    settled = np.asarray(frame.frame[-1], dtype=np.int8)
    assert int(frame.levels_completed) == 8
    assert "NOT_FINISHED" in str(frame.state)
    assert raw_hash(settled) == ("375b7eddf99ae0a2ef41439fe15d92d2c8925b57720f3c6c833b9bb64711d2c0")
    camera = camera_from_switches(settled, 0)
    assert camera == 0
    return PublicRun(env, frame, settled, camera, 342)


def step_settled(run: PublicRun, row: dict[str, Any]) -> dict[str, Any]:
    before_grid = run.settled
    before_camera = run.camera_y
    frame = public_step(run.env, row)
    if not frame.frame:
        run.frame = frame
        run.offline_steps += 1
        endpoint_player = player_screen_center(run.settled)
        return {
            "action": row,
            "layers": 0,
            "state": str(frame.state),
            "levels_completed": int(frame.levels_completed),
            "start_camera_y": before_camera,
            "end_camera_y": before_camera,
            "endpoint_player_screen": (
                list(endpoint_player) if endpoint_player is not None else None
            ),
            "endpoint_player_world": (
                [endpoint_player[0], endpoint_player[1] + before_camera]
                if endpoint_player is not None
                else None
            ),
            "endpoint_sha256": raw_hash(run.settled),
            "timeline": [],
        }
    camera = before_camera
    previous = before_grid
    layer_rows: list[dict[str, Any]] = []
    for index, layer_raw in enumerate(frame.frame, start=1):
        layer = np.asarray(layer_raw, dtype=np.int8)
        camera = infer_camera(previous, layer, camera)
        player = player_screen_center(layer)
        layer_rows.append(
            {
                "layer": index,
                "camera_y": camera,
                "player_screen": list(player) if player is not None else None,
                "player_world": ([player[0], player[1] + camera] if player is not None else None),
            }
        )
        previous = layer
    run.frame = frame
    run.settled = np.asarray(frame.frame[-1], dtype=np.int8)
    run.camera_y = camera
    run.offline_steps += 1
    endpoint_player = player_screen_center(run.settled)
    return {
        "action": row,
        "layers": len(frame.frame),
        "state": str(frame.state),
        "levels_completed": int(frame.levels_completed),
        "start_camera_y": before_camera,
        "end_camera_y": camera,
        "endpoint_player_screen": (list(endpoint_player) if endpoint_player is not None else None),
        "endpoint_player_world": (
            [endpoint_player[0], endpoint_player[1] + camera]
            if endpoint_player is not None
            else None
        ),
        "endpoint_sha256": raw_hash(run.settled),
        "timeline": layer_rows,
    }


def advance(run: PublicRun, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = []
    for row in rows:
        traces.append(step_settled(run, row))
        if "GAME_OVER" in str(run.frame.state) or "WIN" in str(run.frame.state):
            break
    return traces


def compress_timeline(trace: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = trace["timeline"]
    keep: list[dict[str, Any]] = []
    prior: tuple[Any, ...] | None = None
    for row in timeline:
        player = row["player_screen"]
        signature = (
            row["camera_y"],
            player is None,
            None if player is None else player[1] in (0.0, 63.0),
        )
        if signature != prior:
            keep.append(row)
        prior = signature
    if timeline and (not keep or keep[-1] != timeline[-1]):
        keep.append(timeline[-1])
    return keep


def result_row(name: str, run: PublicRun, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": name,
        "offline_steps": run.offline_steps,
        **{key: value for key, value in trace.items() if key != "timeline"},
        "timeline_keyframes": compress_timeline(trace),
        "blank_player_layers": [
            row["layer"] for row in trace["timeline"] if row["player_screen"] is None
        ],
        "camera_pan_first_layer": next(
            (
                row["layer"]
                for row in trace["timeline"]
                if row["camera_y"] != trace["start_camera_y"]
            ),
            None,
        ),
    }


def run_case(name: str) -> dict[str, Any]:
    run = fresh_l9()
    if name == "long_down_x32":
        advance(run, COMMON + [move(4), move(4)])
        trace = step_settled(run, move(4))
    elif name == "long_up_x32":
        advance(run, COMMON + [move(4), move(4), move(4)])
        trace = step_settled(run, click(3, 5))
    elif name == "ungrown_x32":
        advance(run, [move(4)])
        trace = step_settled(run, move(4))
    elif name == "grown_x32":
        advance(run, [click(39, 15), move(4)])
        trace = step_settled(run, move(4))
    elif name.startswith("door_high_"):
        column = int(name.rsplit("_", 1)[1])
        advance(run, HIGH_UP[column])
        if column == 27:
            # A lateral body at x21 keeps the x27 rider at -9.5 when gravity
            # reverses; without it, x27 drops twelve rows before the door click.
            advance(run, [click(21, 39)])
        advance(run, [click(3, 3), click(column, 23)])
        trace = step_settled(run, click(3, 17))
    elif name.startswith("closed_high_"):
        column = int(name.rsplit("_", 1)[1])
        advance(run, HIGH_UP[column])
        advance(run, [click(3, 3)])
        trace = step_settled(run, click(3, 17))
    elif name == "door_low_x15":
        advance(
            run,
            COMMON + [click(15, 23), move(4), move(4), move(4), move(1), move(1), move(1)],
        )
        trace = step_settled(run, click(3, 5))
    elif name.startswith("door_open_trigger_"):
        column = int(name.rsplit("_", 1)[1])
        advance(run, HIGH_UP[column])
        # Under UP gravity, opening the door itself is the launch action.
        trace = step_settled(run, click(column, 33))
    elif name.startswith("top_"):
        top_rows = top_rail_tail()
        if name.startswith("top_27_"):
            # Stop two lateral body hops earlier, directly below the only
            # public-grid structure visible above the toggle ceiling.
            top_rows = top_rows[:-4]
        elif name.startswith("top_45_"):
            top_rows = top_rows[:-10]
        advance(run, top_rows)
        player = player_screen_center(run.settled)
        assert player is not None
        expected_x = (
            27.5 if name.startswith("top_27_") else 45.5 if name.startswith("top_45_") else 15.5
        )
        assert [player[0], player[1] + run.camera_y] == [expected_x, -75.5]
        if name == "top_closed_carry":
            trace = step_settled(run, click(15, 39))
        elif name == "top_x9_left":
            advance(run, [click(9, 39)])
            trace = step_settled(run, move(1))
        elif name == "top_toggle15_only":
            trace = step_settled(run, click(15, 33))
        elif name == "top_toggle15_carry":
            advance(run, [click(15, 33)])
            trace = step_settled(run, click(15, 39))
        elif name == "top_toggle15_down":
            advance(run, [click(15, 33)])
            trace = step_settled(run, click(3, 33))
        elif name == "top_switch_down":
            trace = step_settled(run, click(3, 33))
        elif name == "top_27_toggle":
            trace = step_settled(run, click(27, 33))
        elif name == "top_27_door_open":
            advance(run, [click(27, 33)])
            trace = step_settled(run, click(27, 33))
        elif name == "top_45_toggle":
            trace = step_settled(run, click(45, 33))
        else:
            raise ValueError(f"unknown top case: {name}")
    else:
        raise ValueError(f"unknown case: {name}")
    return result_row(name, run, trace)


DEFAULT_CASES = [
    "long_down_x32",
    "long_up_x32",
    "ungrown_x32",
    "grown_x32",
    "closed_high_15",
    "door_high_15",
    "closed_high_21",
    "door_high_21",
    "closed_high_27",
    "door_high_27",
    "door_low_x15",
    "door_open_trigger_15",
    "door_open_trigger_21",
    "door_open_trigger_27",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    args = parser.parse_args()
    results = []
    for name in args.cases:
        row = run_case(name)
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "summary": {
                    "cases": len(results),
                    "offline_steps_including_prefixes": sum(
                        row["offline_steps"] for row in results
                    ),
                    "max_level_reached": max(row["levels_completed"] for row in results),
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
