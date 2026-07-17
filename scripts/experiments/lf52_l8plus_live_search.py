#!/usr/bin/env python3
"""Frame-only, public-action search probe for lf52 levels 8+.

This experiment deliberately treats the game as hidden.  It replays a banked
prefix, discovers legal peg jumps from ACTION6's rendered landing highlights,
and traverses reversible transitions with ACTION7.  It never imports or reads
the lf52 implementation and never accesses private runtime state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action  # noqa: E402


def label(action: int, data: dict[str, int] | None = None) -> str:
    row: dict[str, Any] = {"action": int(action)}
    if data is not None:
        row["data"] = {"x": int(data["x"]), "y": int(data["y"])}
    return json.dumps(row, separators=(",", ":"))


def connected_components(mask: np.ndarray) -> Iterable[list[tuple[int, int]]]:
    """Yield four-connected ``(y, x)`` components from a small Boolean frame."""

    remaining = {tuple(int(v) for v in point) for point in np.argwhere(mask)}
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            y, x = stack.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    component.append(point)
        yield component


class LiveSearch:
    def __init__(
        self,
        prefix: list[str],
        *,
        depth_limit: int,
        state_limit: int,
        time_limit_s: float,
        allow_carriers: bool,
        candidate_mode: str,
    ) -> None:
        self.prefix = prefix
        self.depth_limit = depth_limit
        self.state_limit = state_limit
        self.time_limit_s = time_limit_s
        self.allow_carriers = allow_carriers
        self.candidate_mode = candidate_mode
        self.started_at = time.monotonic()
        self.env_steps = 0
        self.expanded = 0
        self.seen: set[bytes] = set()
        self.solution: list[str] | None = None
        self.solution_level: int | None = None

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lf52-live-search")
        arcade = kit.offline_arcade()
        self.env = arcade.make("lf52", scorecard_id=arcade.open_scorecard())
        self.frame = self.env.reset()
        for action_label in prefix:
            row = json.loads(action_label)
            self.frame = self.step(int(row["action"]), row.get("data"))
        self.start_level = int(self.frame.levels_completed)
        self.root_grid = self.grid().copy()

    def step(self, action: int, data: dict[str, int] | None = None):
        self.env_steps += 1
        return self.env.step(_game_action(GameAction, int(action)), data=data)

    def grid(self) -> np.ndarray:
        return np.asarray(self.frame.frame[-1], dtype=np.int8)

    def state_key(self) -> bytes:
        # Row zero is the action indicator rather than physical game state.
        return self.grid()[1:].tobytes()

    def candidates(self, grid: np.ndarray) -> list[tuple[int, int]]:
        anchors = [(x, y) for y in range(6, 61, 6) for x in range(6, 61, 6)]
        if self.candidate_mode == "all":
            return anchors
        result: list[tuple[int, int]] = []
        for x, y in anchors:
            tile = grid[y : min(y + 4, 64), x : min(x + 4, 64)]
            # L8's four click-responsive peg sprites use rendered colors 9/14.
            # Rail pixels can cause harmless false positives but never false moves.
            if np.any((tile == 9) | (tile == 14)):
                result.append((x, y))
        return result

    def restore_assert(self, expected: np.ndarray, context: str) -> None:
        actual = self.grid()
        if not np.array_equal(expected[1:], actual[1:]):
            diff = int(np.count_nonzero(expected[1:] != actual[1:]))
            raise RuntimeError(f"ACTION7 restore failed after {context}: {diff} pixels")

    def legal_jumps(self, base: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        jumps: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for source in self.candidates(base):
            self.frame = self.step(6, {"x": source[0], "y": source[1]})
            selected = self.grid()
            destination_mask = (selected == 2) & (base != 2)
            for component in connected_components(destination_mask):
                ys = [point[0] for point in component]
                xs = [point[1] for point in component]
                # Landing glyphs occupy the four-by-four tile whose top-left is
                # the ACTION6 coordinate (multiples of six in this rendering).
                destination = ((min(xs) // 6) * 6, (min(ys) // 6) * 6)
                move = (source, destination)
                if move not in jumps:
                    jumps.append(move)
            self.frame = self.step(7)
            self.restore_assert(base, f"oracle click {source}")
        return jumps

    def expired(self) -> bool:
        return (
            self.expanded >= self.state_limit
            or time.monotonic() - self.started_at >= self.time_limit_s
        )

    def dfs(self, path: list[str], depth: int) -> bool:
        if self.expired():
            return False
        key = self.state_key()
        if key in self.seen:
            return False
        self.seen.add(key)
        self.expanded += 1
        if self.expanded == 1 or self.expanded % 25 == 0:
            digest = hashlib.sha256(key).hexdigest()[:12]
            print(
                "progress",
                json.dumps(
                    {
                        "expanded": self.expanded,
                        "seen": len(self.seen),
                        "depth": depth,
                        "path_actions": len(path),
                        "env_steps": self.env_steps,
                        "elapsed_s": round(time.monotonic() - self.started_at, 3),
                        "state": digest,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if depth >= self.depth_limit:
            return False

        base = self.grid().copy()
        jumps = self.legal_jumps(base)
        for source, destination in jumps:
            self.frame = self.step(6, {"x": source[0], "y": source[1]})
            self.frame = self.step(6, {"x": destination[0], "y": destination[1]})
            move_labels = [
                label(6, {"x": source[0], "y": source[1]}),
                label(6, {"x": destination[0], "y": destination[1]}),
            ]
            next_path = path + move_labels
            level = int(self.frame.levels_completed)
            if level > self.start_level:
                self.solution = next_path
                self.solution_level = level
                print(
                    "FOUND",
                    json.dumps(
                        {
                            "level": level,
                            "depth": depth + 1,
                            "suffix_actions": len(next_path),
                            "expanded": self.expanded,
                            "env_steps": self.env_steps,
                            "state": str(self.frame.state),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return True
            if self.dfs(next_path, depth + 1):
                return True
            self.frame = self.step(7)
            self.frame = self.step(7)
            self.restore_assert(base, f"jump {source}->{destination}")

        if self.allow_carriers:
            for action in (3, 4, 1, 2):
                self.frame = self.step(action)
                changed = not np.array_equal(base[1:], self.grid()[1:])
                if changed:
                    if self.dfs(path + [label(action)], depth + 1):
                        return True
                self.frame = self.step(7)
                self.restore_assert(base, f"carrier ACTION{action}")
        return False

    def run(self) -> dict[str, Any]:
        found = self.dfs([], 0)
        elapsed = time.monotonic() - self.started_at
        return {
            "found": found,
            "start_level": self.start_level,
            "solution_level": self.solution_level,
            "suffix": self.solution,
            "suffix_actions": len(self.solution or []),
            "expanded": self.expanded,
            "seen": len(self.seen),
            "env_steps": self.env_steps,
            "elapsed_s": round(elapsed, 3),
            "depth_limit": self.depth_limit,
            "state_limit": self.state_limit,
            "time_limit_s": self.time_limit_s,
            "allow_carriers": self.allow_carriers,
            "candidate_mode": self.candidate_mode,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=Path,
        default=REPO / "results/outer_loop_round14_lf52_probe_20260716.json",
    )
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--states", type=int, default=5_000)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--no-carriers", action="store_true")
    parser.add_argument("--candidate-mode", choices=("all", "peg-colors"), default="peg-colors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.prefix.read_text())
    prefix = artifact["action_sequence"]
    search = LiveSearch(
        prefix,
        depth_limit=args.depth,
        state_limit=args.states,
        time_limit_s=args.seconds,
        allow_carriers=not args.no_carriers,
        candidate_mode=args.candidate_mode,
    )
    result = search.run()
    print("RESULT", json.dumps(result, sort_keys=True), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
