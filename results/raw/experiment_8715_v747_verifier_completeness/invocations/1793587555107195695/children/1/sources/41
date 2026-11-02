"""cd82 adapter mechanics for ARC offline deepening.

Spec refs: REQ-ARC-WMTE-4496, SCENARIO-ARC-WMTE-4495.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from arcengine import GameAction

from carnot.agentic.arc_agi3_live_adapter import _game_action
from carnot.agentic.arc_exp4024_fifth_game_explore_first import (
    apply_cd82_region_fill,
    basket_navigation_actions,
    cd82_goal_mask,
)


ANIMATION_SETTLE_LIMIT = 64


def _sprites_named(game: Any, name: str) -> list[Any]:
    return list(game.current_level.get_sprites_by_name(name))


def _all_sprites(game: Any) -> list[Any]:
    return list(game.current_level.get_sprites())


def _canvas_sprite(game: Any) -> Any:
    return _sprites_named(game, "xytrjjbyib")[0]


def _target_sprite(game: Any) -> Any:
    for sprite in _all_sprites(game):
        if str(getattr(sprite, "name", "")).startswith("eoqnvkspoa-"):
            return sprite
    raise ValueError("cd82 target sprite not found")


def _canvas(game: Any) -> np.ndarray:
    return np.asarray(_canvas_sprite(game).pixels, dtype=np.int16)


def _target(game: Any) -> np.ndarray:
    return np.asarray(_target_sprite(game).pixels, dtype=np.int16)


def _camera_scale_offset(game: Any) -> tuple[float, float, float]:
    camera = getattr(game, "camera", None)
    if camera is None or not hasattr(camera, "_calculate_scale_and_offset"):
        return 1.0, 0.0, 0.0
    scale, x_offset, y_offset = camera._calculate_scale_and_offset()
    return float(scale), float(x_offset), float(y_offset)


def _sprite_center_in_action_space(game: Any, sprite: Any) -> tuple[float, float]:
    scale, x_offset, y_offset = _camera_scale_offset(game)
    return (
        (float(getattr(sprite, "x", 0)) + 2.0) * scale + x_offset,
        (float(getattr(sprite, "y", 0)) + 2.0) * scale + y_offset,
    )


def _label(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def _mismatch(canvas: np.ndarray, target: np.ndarray) -> int:
    mask = cd82_goal_mask()
    return int(np.count_nonzero(np.asarray(canvas)[mask] != np.asarray(target)[mask]))


def cd82_palette_actions(game: Any) -> dict[int, str]:
    """Return palette click labels keyed by color, using live sprite positions."""

    labels: dict[int, str] = {}
    for sprite in sorted(
        _sprites_named(game, "pqkenviek"),
        key=lambda item: (float(getattr(item, "y", 0)), float(getattr(item, "x", 0))),
    ):
        color = int(np.asarray(sprite.pixels)[2, 2])
        x_coord, y_coord = _sprite_center_in_action_space(game, sprite)
        labels[color] = _label(
            {
                "action": 6,
                "color": color,
                "role": "palette",
                "x": x_coord,
                "y": y_coord,
            }
        )
    return labels


def cd82_remaining_fills(game: Any, *, max_fills: int = 8) -> list[tuple[int, int]]:
    """Greedily derive the remaining region/color fills from canvas and target."""

    current = np.array(_canvas(game), dtype=np.int16, copy=True)
    target = _target(game)
    mask = cd82_goal_mask()
    colors = sorted({int(value) for value in target[mask].ravel()})
    plan: list[tuple[int, int]] = []
    for _ in range(max_fills):
        base = _mismatch(current, target)
        if base == 0:
            return plan
        best: tuple[int, int, int, np.ndarray] | None = None
        for region_index in range(8):
            for fill_color in colors:
                predicted = apply_cd82_region_fill(current, region_index, fill_color)
                distance = _mismatch(predicted, target)
                if best is None or (distance, region_index, fill_color) < (
                    best[0],
                    best[1],
                    best[2],
                ):
                    best = (distance, region_index, fill_color, predicted)
        if best is None or best[0] >= base:
            return []
        _, region_index, fill_color, predicted = best
        plan.append((region_index, fill_color))
        current = predicted
    return plan if _mismatch(current, target) == 0 else []


def cd82_action_labels(env: Any, frame: Any = None, path: tuple[str, ...] = ()) -> list[str]:
    """Return the next derived cd82 action labels for the current env state."""

    del frame, path
    game = env._game
    fills = cd82_remaining_fills(game)
    if not fills:
        return []
    target_region, target_color = fills[0]
    if int(getattr(game, "knqmgavuh", -1)) != target_color:
        return [cd82_palette_actions(game)[target_color]]
    route = basket_navigation_actions(int(getattr(game, "xwmfgtlso", 0)), target_region)
    if route:
        return [_label({"action": int(route[0])})]
    return [_label({"action": 5})]


def cd82_apply(env: Any, label: str, frame: Any) -> Any:
    """Apply one cd82 label and settle the basket/palette animation."""

    del frame
    row = json.loads(label)
    action = int(row["action"])
    data = None
    if action == 6:
        data = {"x": float(row["x"]), "y": float(row["y"])}
    stepped = env.step(_game_action(GameAction, action), data=data)
    for _ in range(ANIMATION_SETTLE_LIMIT):
        game = env._game
        if not bool(getattr(game, "edjesyzxk", False) or getattr(game, "yfobpcuef", False)):
            return stepped
        stepped = env.step(_game_action(GameAction, 1))
    raise RuntimeError("cd82 animation did not settle")


def cd82_state_key(game: Any, frame: Any = None) -> tuple[Any, ...]:
    """Deduplicate by level, selected fill state, canvas, target, and animation flags."""

    level = int(getattr(frame, "levels_completed", 0) or 0) if frame is not None else 0
    canvas_digest = hashlib.sha256(np.asarray(_canvas(game)).tobytes()).hexdigest()
    target_digest = hashlib.sha256(np.asarray(_target(game)).tobytes()).hexdigest()
    return (
        level,
        int(getattr(game, "xwmfgtlso", -1)),
        int(getattr(game, "knqmgavuh", -1)),
        bool(getattr(game, "edjesyzxk", False)),
        bool(getattr(game, "yfobpcuef", False)),
        tuple(int(value) for value in np.asarray(_canvas(game)).shape),
        canvas_digest,
        target_digest,
    )


def cd82_hand_verifier(game: Any, frame: Any = None) -> float:
    """Lower-is-better goal distance under cd82's offline target mask."""

    del frame
    return float(_mismatch(_canvas(game), _target(game)))
