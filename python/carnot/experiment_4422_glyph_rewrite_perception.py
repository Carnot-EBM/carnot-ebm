"""Exp 4422: from-pixels glyph-rewrite verifier for tr87.

Spec refs: REQ-REPORT-4422, SCENARIO-REPORT-4422.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4422_glyph_rewrite_perception.json"
BANKED_SOLUTION_RELATIVE_PATH = "results/arc_loop_solve_tr87.json"
LEGACY_GLYPH_SCRIPT_RELATIVE_PATH = "scripts/experiments/arc3_config_layerb_glyph_tr87.py"
TARGET_GAME = "tr87"
CLAIMED_LEVEL = 1
RANDOM_SEED = 4422
ON = 5
BACKGROUND_LIKE = {0, 1, 2, 3, 4, ON}
GLYPH_SIZE = 5
HAMMING_TOLERANCE = 2

Token = tuple[int, int]
Rule = tuple[tuple[Token, ...], tuple[Token, ...]]

REQUIRED_ARTIFACT_FIELDS = (
    "offline_reproduced",
    "reproduced_levels",
    "verifier_is_oracle",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "offline_reproduced": "Bare bool: only arc_solver_kit.reproduce-confirmed offline levels count.",
    "reproduced_levels": "Bare int: reproduced offline levels reached by the replay gate.",
    "verifier_is_oracle": "Bare bool=true: execution-grounded glyph check; ARC progress, not a moat headline.",
    "honest_verdict": "Terminal-prefixed final state.",
}


def _load_segment_glyphs() -> Callable[..., list[tuple[int, int, np.ndarray, int]]]:
    script = REPO_ROOT / LEGACY_GLYPH_SCRIPT_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("arc3_config_layerb_glyph_tr87", script)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive guard
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.segment_glyphs


segment_glyphs = _load_segment_glyphs()


@dataclass(frozen=True)
class TileObservation:
    col_start: int
    col_end: int
    frame_color: int
    pattern: np.ndarray


@dataclass(frozen=True)
class RegionLayout:
    rule_bands: list[tuple[int, int]]
    target_band: tuple[int, int] | None
    editable_band: tuple[int, int] | None


@dataclass(frozen=True)
class GroundingFrames:
    win_grid: np.ndarray
    nonwin_grids: list[np.ndarray]
    preconditions_checked: dict[str, Any]


@dataclass
class GlyphClassifier:
    tolerance: int = HAMMING_TOLERANCE
    codebooks: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def classify(self, frame_color: int, pattern: np.ndarray) -> Token:
        codebook = self.codebooks.setdefault(int(frame_color), [])
        if codebook:
            distances = [_rotation_hamming(pattern, prototype) for prototype in codebook]
            best = min(range(len(distances)), key=lambda index: distances[index])
            if distances[best] <= self.tolerance:
                return (int(frame_color), int(best))
        codebook.append(np.asarray(pattern, dtype=int))
        return (int(frame_color), len(codebook) - 1)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _on_pattern(tile: np.ndarray) -> np.ndarray:
    source = (np.asarray(tile) == ON).astype(int)
    out = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=int)
    rows = min(GLYPH_SIZE, source.shape[0])
    cols = min(GLYPH_SIZE, source.shape[1])
    out[:rows, :cols] = source[:rows, :cols]
    return out


def _rotation_hamming(pattern: np.ndarray, prototype: np.ndarray) -> int:
    a = np.asarray(pattern, dtype=int)
    b = np.asarray(prototype, dtype=int)
    return min(int((np.rot90(a, turns) != b).sum()) for turns in range(4))


def _frame_color(tile: np.ndarray) -> int:
    values = [int(v) for v in np.asarray(tile).ravel() if int(v) not in BACKGROUND_LIKE]
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def _content_bands(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid)
    rows = [int(row) for row in range(g.shape[0]) if (g[row, :] == ON).any()]
    bands: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for row in rows + [None]:
        if row is not None and start is None:
            start = previous = row
        elif row is not None and previous is not None and row == previous + 1:
            previous = row
        else:
            if start is not None and previous is not None:
                bands.append((start, previous))
            start = previous = row
    return bands


def localize_regions(grid: np.ndarray) -> RegionLayout:
    bands = _content_bands(np.asarray(grid))
    if len(bands) < 3:
        return RegionLayout(rule_bands=[], target_band=None, editable_band=None)
    return RegionLayout(rule_bands=bands[:-2], target_band=bands[-2], editable_band=bands[-1])


def _observe_band(grid: np.ndarray, band: tuple[int, int]) -> list[TileObservation]:
    g = np.asarray(grid)
    raw_tiles = segment_glyphs(g, list(range(band[0], band[1] + 1)), 0, g.shape[1] - 1)
    return [
        TileObservation(
            col_start=int(col_start),
            col_end=int(col_end),
            frame_color=_frame_color(tile),
            pattern=_on_pattern(tile),
        )
        for col_start, col_end, tile, _legacy_frame in raw_tiles
    ]


def _side_runs(tiles: Sequence[TileObservation]) -> list[list[TileObservation]]:
    runs: list[list[TileObservation]] = []
    for tile in tiles:
        if not runs or runs[-1][-1].frame_color != tile.frame_color:
            runs.append([tile])
        else:
            runs[-1].append(tile)
    return runs


def _tokens_for_tiles(tiles: Sequence[TileObservation], classifier: GlyphClassifier) -> tuple[Token, ...]:
    return tuple(classifier.classify(tile.frame_color, tile.pattern) for tile in tiles)


def _rules_from_layout(
    grid: np.ndarray,
    layout: RegionLayout,
    classifier: GlyphClassifier,
) -> list[Rule]:
    rules: list[Rule] = []
    for band in layout.rule_bands:
        runs = _side_runs(_observe_band(grid, band))
        for index in range(0, len(runs) - 1, 2):
            lhs = _tokens_for_tiles(runs[index], classifier)
            rhs = _tokens_for_tiles(runs[index + 1], classifier)
            if lhs and rhs:
                rules.append((lhs, rhs))
    return rules


def greedy_rewrite(sequence: Sequence[Token], rules: Sequence[Rule]) -> list[Token] | None:
    output: list[Token] = []
    position = 0
    source = list(sequence)
    while position < len(source):
        for lhs, rhs in rules:
            lhs_list = list(lhs)
            if lhs_list and source[position : position + len(lhs_list)] == lhs_list:
                output.extend(rhs)
                position += len(lhs_list)
                break
        else:
            return None
    return output


def sequence_rewrite_matches(
    target: Sequence[Token],
    editable: Sequence[Token],
    rules: Sequence[Rule],
    *,
    max_passes: int = 3,
) -> tuple[bool, list[list[Token]]]:
    sequence = list(target)
    passes: list[list[Token]] = []
    for _ in range(max_passes):
        rewritten = greedy_rewrite(sequence, rules)
        if rewritten is None:
            return False, passes
        passes.append(rewritten)
        if rewritten == list(editable):
            return True, passes
        sequence = rewritten
    return False, passes


def _serial_tokens(tokens: Sequence[Token]) -> list[list[int]]:
    return [[int(frame), int(value)] for frame, value in tokens]


def _serial_rules(rules: Sequence[Rule]) -> list[dict[str, list[list[int]]]]:
    return [{"lhs": _serial_tokens(lhs), "rhs": _serial_tokens(rhs)} for lhs, rhs in rules]


def decode_and_check_grid(grid: np.ndarray, *, max_passes: int = 3) -> tuple[bool, dict[str, Any]]:
    g = np.asarray(grid)
    layout = localize_regions(g)
    debug: dict[str, Any] = {
        "localized": {
            "rule_bands": list(layout.rule_bands),
            "target_band": layout.target_band,
            "editable_band": layout.editable_band,
        }
    }
    if layout.target_band is None or layout.editable_band is None:
        debug.update({"rules": 0, "target_len": 0, "editable_len": 0, "rewrite_passes": 0})
        return False, debug

    classifier = GlyphClassifier()
    rules = _rules_from_layout(g, layout, classifier)
    target = _tokens_for_tiles(_observe_band(g, layout.target_band), classifier)
    editable = _tokens_for_tiles(_observe_band(g, layout.editable_band), classifier)
    ok, passes = sequence_rewrite_matches(target, editable, rules, max_passes=max_passes)
    expected = passes[-1] if passes else []
    debug.update(
        {
            "rules": len(rules),
            "rules_decoded": _serial_rules(rules),
            "target_len": len(target),
            "editable_len": len(editable),
            "target_sequence": _serial_tokens(target),
            "editable_sequence": _serial_tokens(editable),
            "expected_sequence": _serial_tokens(expected),
            "rewrite_passes": len(passes),
            "codebook_sizes": {str(frame): len(patterns) for frame, patterns in classifier.codebooks.items()},
        }
    )
    return bool(ok), debug


def evaluate_grounding(win_grid: np.ndarray, nonwin_grids: Sequence[np.ndarray]) -> dict[str, Any]:
    fires_on_win, win_debug = decode_and_check_grid(win_grid)
    nonwin_results = []
    false_positives = 0
    for grid in nonwin_grids:
        fires, debug = decode_and_check_grid(grid)
        nonwin_results.append({"fires": bool(fires), "debug": debug})
        if fires:
            false_positives += 1
    n_nonwins = len(nonwin_results)
    return {
        "fires_on_win": bool(fires_on_win),
        "false_positives": int(false_positives),
        "false_positive_rate": round(false_positives / n_nonwins, 6) if n_nonwins else 0.0,
        "n_nonwins": n_nonwins,
        "grounded": bool(fires_on_win) and false_positives == 0,
        "win_debug": win_debug,
        "nonwin_results": nonwin_results,
    }


def load_banked_solution_labels(root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover - local artifact boundary
    data = json.loads((root / BANKED_SOLUTION_RELATIVE_PATH).read_text(encoding="utf-8"))
    labels = data.get("solution_labels") or []
    return [str(label) for label in labels]


def _action_input_from_label(label: str) -> Any:  # pragma: no cover - ARC SDK boundary
    from arcengine import ActionInput, GameAction

    payload = json.loads(label)
    action = getattr(GameAction, f"ACTION{int(payload['action'])}")
    return ActionInput(id=action, data=payload.get("data") or {})


def collect_tr87_grounding_frames(root: Path = REPO_ROOT) -> GroundingFrames:  # pragma: no cover - live ARC boundary
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_agi3_world_model import grid_of

    labels = load_banked_solution_labels(root)
    arc = kit.offline_arcade()
    env = arc.make(TARGET_GAME, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    nonwins: list[np.ndarray] = [np.asarray(grid_of(frame)).copy()]
    current_level = int(frame.levels_completed or 0)
    win_grid: np.ndarray | None = None

    for label in labels:
        game_before = copy.deepcopy(env._game)
        frame_before = frame
        payload = json.loads(label)
        frame = env.step(_game_action(GameAction, int(payload["action"])), data=payload.get("data"))
        next_level = int(frame.levels_completed or 0)
        if next_level > current_level:
            game_before._set_action(_action_input_from_label(label))
            game_before.step()
            win_grid = game_before.camera.render(game_before.current_level.get_sprites())
            break
        nonwins.append(np.asarray(grid_of(frame_before)).copy())
        current_level = next_level

    if win_grid is None:
        raise RuntimeError("banked tr87 solution did not expose a first-level win frame")

    return GroundingFrames(
        win_grid=np.asarray(win_grid),
        nonwin_grids=nonwins[-6:],
        preconditions_checked={
            "offline_env_loads": {TARGET_GAME: True},
            "banked_solution_labels": len(labels),
            "segment_glyphs_primitive": LEGACY_GLYPH_SCRIPT_RELATIVE_PATH,
        },
    )


def reproduce_solution(labels: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - live ARC boundary
    from carnot.agentic import arc_game_adapters, arc_solver_kit

    adapter = arc_game_adapters.get_adapter(TARGET_GAME)
    if adapter is None:
        return {"game": TARGET_GAME, "reached_level": 0, "claimed_level": CLAIMED_LEVEL, "reproduced": False}
    return dict(
        arc_solver_kit.reproduce(
            TARGET_GAME,
            labels,
            adapter.apply,
            warmup_label=adapter.warmup_label,
            claimed_level=CLAIMED_LEVEL,
        )
    )


def _verdict(grounded: bool, offline_reproduced: bool, reproduced_levels: int) -> str:
    if grounded and offline_reproduced and reproduced_levels >= 1:
        return "success_glyph_rewrite_perception_tr87_grounded_reproduced"
    if not grounded:
        return "complete_glyph_rewrite_perception_not_grounded"
    return "blocked_offline_reproduction_failed"


def build_artifact(
    *,
    root: Path,
    grounding: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    solution_labels: Sequence[str],
    reproduction_result: Mapping[str, Any],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    offline_reproduced = bool(reproduction_result.get("reproduced"))
    reproduced_levels = int(reproduction_result.get("reached_level") or 0) if offline_reproduced else 0
    grounded = bool(grounding.get("grounded"))
    checksum_payload = {
        "game": TARGET_GAME,
        "grounding": grounding,
        "solution_labels": list(solution_labels),
        "reproduction_result": dict(reproduction_result),
        "random_seed": RANDOM_SEED,
    }
    artifact = {
        "experiment": "experiment_4422_glyph_rewrite_perception",
        "schema": "carnot.exp4422.glyph_rewrite_perception.v1",
        "target_game": TARGET_GAME,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "verifier_is_oracle": True,
        "honest_verdict": _verdict(grounded, offline_reproduced, reproduced_levels),
        "fires_on_win": bool(grounding.get("fires_on_win")),
        "false_positives": int(grounding.get("false_positives") or 0),
        "false_positive_rate": float(grounding.get("false_positive_rate") or 0.0),
        "n_nonwins": int(grounding.get("n_nonwins") or 0),
        "grounded": grounded,
        "solution_label_count": len(solution_labels),
        "preconditions_checked": dict(preconditions_checked),
        "reproduction_result": dict(reproduction_result),
        "grounding_debug": dict(grounding),
        "random_seed": RANDOM_SEED,
        "inference_substrate": "offline_arc_agi3_glyph_rewrite_pixel_decode_cpu_no_llm",
        "submitted_to_leaderboard": False,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-REPORT-4422", "SCENARIO-REPORT-4422"],
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "result_path": RESULT_RELATIVE_PATH,
        "reproducibility_checksum": _sha256(checksum_payload),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing {field_name}")
    if not isinstance(artifact.get("offline_reproduced"), bool):
        errors.append("offline_reproduced must be bare bool")
    if not isinstance(artifact.get("reproduced_levels"), int):
        errors.append("reproduced_levels must be bare int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not isinstance(artifact.get("fires_on_win"), bool):
        errors.append("fires_on_win must be bare bool")
    if not isinstance(artifact.get("false_positives"), int):
        errors.append("false_positives must be bare int")
    if not isinstance(artifact.get("grounded"), bool):
        errors.append("grounded must be bare bool")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("success_", "blocked_", "complete_")):
        errors.append("honest_verdict must start with terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("success_"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("offline_reproduced must be true for success verdicts")
        if artifact.get("fires_on_win") is not True:
            errors.append("success verdict requires fires_on_win true")
        if artifact.get("false_positives") != 0:
            errors.append("success verdict requires zero false positives")
        if artifact.get("grounded") is not True:
            errors.append("success verdict requires grounded true")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    grounding_frames_fn: Callable[[Path], GroundingFrames] = collect_tr87_grounding_frames,
    solution_labels_fn: Callable[[Path], Sequence[str]] = load_banked_solution_labels,
    reproduce_fn: Callable[[Sequence[str]], Mapping[str, Any]] = reproduce_solution,
    now: Callable[[], float] = time.perf_counter,
) -> Path:
    started = now()
    frames = grounding_frames_fn(root)
    grounding = evaluate_grounding(frames.win_grid, frames.nonwin_grids)
    solution_labels = list(solution_labels_fn(root))
    reproduction_result = dict(reproduce_fn(solution_labels))
    artifact = build_artifact(
        root=root,
        grounding=grounding,
        preconditions_checked=frames.preconditions_checked,
        solution_labels=solution_labels,
        reproduction_result=reproduction_result,
        started_at=started,
        ended_at=now(),
    )
    return write_artifact(root, artifact)


def main() -> int:  # pragma: no cover - CLI wrapper
    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
