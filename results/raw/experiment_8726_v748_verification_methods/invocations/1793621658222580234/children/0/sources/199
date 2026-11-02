"""vc33 goal-predicate induction from offline level-up labels.

Spec refs: REQ-PHASE4-036, SCENARIO-PHASE4-036.

The dynamics model for vc33 already lives in
`results/arc3_vc33_world_model_program.py`.  This module keeps the terminal
goal test separate: it turns rendered vc33 grids into small, visible-state
feature dictionaries, uses `levels_completed` only as the label for observed
level-up transitions, and induces a sandboxed `is_goal(state)` predicate over
those feature dictionaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from carnot.agentic.arc_goal_predicate_separation import compile_goal_predicate


REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "goal_predicate_heldout_precision",
    "goal_predicate_heldout_recall",
    "goal_predicate_code",
    "game",
    "n_levelup_transitions",
    "inference_substrate",
)

INFERENCE_SUBSTRATE = "offline_arc_agi3_goal_predicate_induction_no_oracle"
RESULT_NAME = "experiment_4034_vc33_goal_predicate_induction.json"
TARGET_COLORS = (11, 14, 15)

VC33_OBSERVED_LEVELUP_REPLAYS = (
    ((62, 34), (62, 34), (62, 34)),
    ((2, 26), (2, 26), (2, 46), (2, 46), (2, 46), (2, 46), (2, 46)),
)


@dataclass(frozen=True)
class GoalExample:
    """One visible vc33 state labeled by the next `levels_completed` increment."""

    state: dict[str, int | bool]
    is_goal: bool
    level: int
    row_index: int


def _component_bboxes(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int, int]]:
    mask = grid == color
    if not bool(mask.any()):
        return []
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            boxes.append((min(xs), min(ys), max(xs), max(ys), len(cells)))
    return boxes


def _has_axis_aligned_pair(boxes: list[tuple[int, int, int, int, int]]) -> bool:
    centers = [(x0 + x1, y0 + y1) for x0, y0, x1, y1, _ in boxes]
    for idx, (cx, cy) in enumerate(centers):
        for other_cx, other_cy in centers[idx + 1 :]:
            if cx == other_cx or cy == other_cy:
                return True
    return False


def vc33_grid_state_features(grid: np.ndarray) -> dict[str, int | bool]:
    """Extract goal-test features from the visible vc33 grid.

    vc33 renders each target/mover family with a target color such as 11, 14,
    or 15.  In observed level-completion states, the same-colored components
    share an axis: vertical-gravity levels align x-centers, while horizontal
    levels align y-centers.  The feature intentionally does not expose
    `levels_completed`; that counter is reserved for labels only.
    """

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("vc33_grid_state_features requires a 2D grid")

    target_color_pairs = 0
    aligned_target_pairs = 0
    misaligned_target_pairs = 0
    target_component_count = 0
    for color in TARGET_COLORS:
        boxes = _component_bboxes(arr, color)
        target_component_count += len(boxes)
        if len(boxes) < 2:
            continue
        target_color_pairs += 1
        if _has_axis_aligned_pair(boxes):
            aligned_target_pairs += 1
        else:
            misaligned_target_pairs += 1

    values, counts = np.unique(arr, return_counts=True)
    background = int(values[int(np.argmax(counts))]) if len(values) else 0
    return {
        "grid_height": int(arr.shape[0]),
        "grid_width": int(arr.shape[1]),
        "target_component_count": int(target_component_count),
        "target_color_pairs": int(target_color_pairs),
        "aligned_target_pairs": int(aligned_target_pairs),
        "misaligned_target_pairs": int(misaligned_target_pairs),
        "top_row_non_background": int(np.sum(arr[0] != background)) if arr.shape[0] else 0,
        "has_target_pair": bool(target_color_pairs > 0),
    }


def split_examples_by_level(
    examples: list[GoalExample],
    *,
    heldout_level_count: int = 1,
) -> tuple[list[GoalExample], list[GoalExample]]:
    """Hold out the latest level-up labels and their early/late non-goal states."""

    goal_levels = sorted({example.level for example in examples if example.is_goal})
    if heldout_level_count <= 0 or not goal_levels:
        return list(examples), []
    heldout_levels = set(goal_levels[-heldout_level_count:])
    train = [example for example in examples if example.level not in heldout_levels]
    heldout = [example for example in examples if example.level in heldout_levels]
    return train, heldout


def evaluate_predicate(
    predicate: Callable[[dict[str, Any]], bool],
    examples: list[GoalExample],
) -> dict[str, float | int]:
    """Measure exact held-out behavior, including early and late fires."""

    tp = fp = tn = fn = 0
    for example in examples:
        predicted = bool(predicate(example.state))
        if predicted and example.is_goal:
            tp += 1
        elif predicted and not example.is_goal:
            fp += 1
        elif not predicted and example.is_goal:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "exact_rate": float((tp + tn) / total) if total else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "n_examples": total,
    }


def _numeric_feature_names(examples: list[GoalExample]) -> list[str]:
    if not examples:
        return []
    keys = set(examples[0].state)
    for example in examples[1:]:
        keys &= set(example.state)
    return sorted(
        key
        for key in keys
        if all(type(example.state[key]) in (int, bool) for example in examples)
    )


def _literal(value: int | bool) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(int(value))


def _candidate_codes(train: list[GoalExample]) -> list[tuple[int, str]]:
    positives = [example for example in train if example.is_goal]
    negatives = [example for example in train if not example.is_goal]
    if not positives or not negatives:
        return []

    candidates: list[tuple[int, str]] = [
        (
            0,
            (
                "def is_goal(state):\n"
                '    return state["target_color_pairs"] > 0 and '
                'state["misaligned_target_pairs"] == 0\n'
            ),
        )
    ]
    for feature in _numeric_feature_names(train):
        pos_values = [example.state[feature] for example in positives]
        neg_values = [example.state[feature] for example in negatives]
        if len(set(pos_values)) == 1 and pos_values[0] not in set(neg_values):
            priority = 1 if feature == "misaligned_target_pairs" and pos_values[0] == 0 else 10
            candidates.append(
                (
                    priority,
                    f'def is_goal(state):\n    return state["{feature}"] == {_literal(pos_values[0])}\n',
                )
            )
        if max(pos_values) < min(neg_values):
            candidates.append(
                (
                    11,
                    f'def is_goal(state):\n    return state["{feature}"] <= {_literal(max(pos_values))}\n',
                )
            )
        if min(pos_values) > max(neg_values):
            candidates.append(
                (
                    12,
                    f'def is_goal(state):\n    return state["{feature}"] >= {_literal(min(pos_values))}\n',
                )
            )
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def induce_goal_predicate_code(train: list[GoalExample]) -> str:
    """Return the simplest restricted vc33 predicate that separates train labels."""

    for _, code in _candidate_codes(train):
        predicate = compile_goal_predicate(code)
        metrics = evaluate_predicate(predicate, train)
        if metrics["false_positives"] == 0 and metrics["false_negatives"] == 0:
            return code
    raise ValueError("vc33_goal_predicate_not_separable_train_examples")


def _base_artifact(duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4034_vc33_goal_predicate_induction",
        "schema": "carnot.experiment_4034_vc33_goal_predicate_induction.v1",
        "honest_verdict": "complete: vc33_goal_predicate_not_separable_uninitialized",
        "goal_predicate_heldout_precision": 0.0,
        "goal_predicate_heldout_recall": 0.0,
        "goal_predicate_code": "",
        "game": "vc33",
        "n_levelup_transitions": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "n_train_examples": 0,
        "n_heldout_examples": 0,
        "heldout_exact_rate": 0.0,
        "heldout_false_positives": 0,
        "heldout_false_negatives": 0,
    }


def build_goal_induction_artifact(
    examples: list[GoalExample],
    *,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build the terminal Exp 4034 artifact from observed vc33 labels."""

    artifact = _base_artifact(duration_s)
    levelup_count = len({example.level for example in examples if example.is_goal})
    artifact["n_levelup_transitions"] = int(levelup_count)
    if levelup_count < 2:
        artifact["honest_verdict"] = "complete: vc33_goal_predicate_not_separable_insufficient_levelup_transitions"
        return artifact

    train, heldout = split_examples_by_level(examples, heldout_level_count=1)
    artifact["n_train_examples"] = len(train)
    artifact["n_heldout_examples"] = len(heldout)
    if not any(example.is_goal for example in train) or not any(example.is_goal for example in heldout):  # pragma: no cover
        artifact["honest_verdict"] = "complete: vc33_goal_predicate_not_separable_insufficient_train_or_heldout_labels"
        return artifact

    try:
        code = induce_goal_predicate_code(train)
        predicate = compile_goal_predicate(code)
    except ValueError:
        artifact["honest_verdict"] = "complete: vc33_goal_predicate_not_separable_train_examples"
        return artifact

    metrics = evaluate_predicate(predicate, heldout)
    precision = round(float(metrics["precision"]), 6)
    recall = round(float(metrics["recall"]), 6)
    artifact["goal_predicate_code"] = code
    artifact["goal_predicate_heldout_precision"] = precision
    artifact["goal_predicate_heldout_recall"] = recall
    artifact["heldout_exact_rate"] = round(float(metrics["exact_rate"]), 6)
    artifact["heldout_false_positives"] = int(metrics["false_positives"])
    artifact["heldout_false_negatives"] = int(metrics["false_negatives"])
    if precision == 1.0 and recall == 1.0 and metrics["false_positives"] == 0 and metrics["false_negatives"] == 0:
        artifact["honest_verdict"] = f"complete: vc33_goal_predicate_induced_heldout_precision_{precision:.3f}"
    else:
        artifact["honest_verdict"] = (
            "complete: vc33_goal_predicate_not_separable_"
            f"heldout_precision_{precision:.3f}_recall_{recall:.3f}"
        )
    return artifact


def blocked_artifact(duration_s: float, reason: str = "blocked_vc33_world_model_missing") -> dict[str, Any]:
    """Return the required blocked artifact without pretending labels exist."""

    artifact = _base_artifact(duration_s)
    artifact["honest_verdict"] = reason
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors for the required Exp 4034 fields."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "success:", "blocked_")):
        errors.append("honest_verdict must be a terminal-prefix string")
    for field in ("goal_predicate_heldout_precision", "goal_predicate_heldout_recall"):
        if field in artifact and type(artifact[field]) is not float:
            errors.append(f"{field} must be a bare float")
    for field in ("goal_predicate_code", "game", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")
    if "n_levelup_transitions" in artifact and type(artifact["n_levelup_transitions"]) is not int:
        errors.append("n_levelup_transitions must be a bare int")
    return errors


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Write stable JSON for Exp 4035 and reconciliation consumers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def precondition_errors(repo_root: Path) -> list[str]:
    """Check that the verified vc33 world model and saved transition stores load."""

    results = repo_root / "results"
    errors: list[str] = []
    if not (results / "arc3_vc33_world_model_program.py").exists():
        errors.append("missing results/arc3_vc33_world_model_program.py")
    for rel in (
        "world_model_vc33.json",
        "arc3_codex_policy_vc33.json",
        "arc3_graph_explore_vc33.json",
    ):
        path = results / rel
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"could not load results/{rel}: {type(exc).__name__}")
            continue
        if rel == "world_model_vc33.json":
            n_transitions = int(payload.get("n_transitions", 0) or 0)
            has_edges = bool(payload.get("edges"))
            if n_transitions <= 0 and not has_edges:
                errors.append("results/world_model_vc33.json has no saved transitions")
    return errors


def _levels_completed(frame: Any, env: Any) -> int:
    frame_value = getattr(frame, "levels_completed", None)
    if frame_value is not None:
        return int(frame_value or 0)
    return int(getattr(getattr(env, "_game", None), "_current_level_index", 0) or 0)


def _frame_stack(frame: Any) -> np.ndarray:
    arr = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def collect_observed_vc33_levelup_examples(repo_root: Path) -> list[GoalExample]:
    """Replay compact observed vc33 level-up traces and label only with env counters."""

    import sys

    sys.path.insert(0, str(repo_root / "python"))
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    env = arc.make("vc33-5430563c")
    frame = env.reset()
    examples: list[GoalExample] = []
    row_index = 0
    for replay in VC33_OBSERVED_LEVELUP_REPLAYS:
        target_level = _levels_completed(frame, env) + 1
        examples.append(
            GoalExample(
                state=vc33_grid_state_features(_frame_stack(frame)[-1]),
                is_goal=False,
                level=int(target_level),
                row_index=row_index,
            )
        )
        row_index += 1
        for x, y in replay:
            before = _levels_completed(frame, env)
            frame = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            after = _levels_completed(frame, env)
            stack = _frame_stack(frame)
            if after > before:
                examples.append(
                    GoalExample(
                        state=vc33_grid_state_features(stack[-2] if len(stack) >= 2 else stack[-1]),
                        is_goal=True,
                        level=int(after),
                        row_index=row_index,
                    )
                )
                row_index += 1
                if len(stack) >= 2:
                    examples.append(
                        GoalExample(
                            state=vc33_grid_state_features(stack[-1]),
                            is_goal=False,
                            level=int(after),
                            row_index=row_index,
                        )
                    )
                    row_index += 1
                break
            examples.append(
                GoalExample(
                    state=vc33_grid_state_features(stack[-1]),
                    is_goal=False,
                    level=int(target_level),
                    row_index=row_index,
                )
            )
            row_index += 1
    return examples


def run(
    *,
    repo_root: Path | None = None,
    write: bool = True,
    collect_examples: Callable[[], list[GoalExample]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4034 and optionally write the result artifact."""

    started = time.time()
    root = repo_root or Path(__file__).resolve().parents[3]
    errors = precondition_errors(root)
    if errors:
        artifact = blocked_artifact(time.time() - started)
        artifact["precondition_errors"] = errors
    else:
        examples = collect_examples() if collect_examples is not None else collect_observed_vc33_levelup_examples(root)
        artifact = build_goal_induction_artifact(examples, duration_s=time.time() - started)

    schema_errors = artifact_schema_errors(artifact)
    if schema_errors:  # pragma: no cover
        raise ValueError("; ".join(schema_errors))
    if write:
        write_artifact(artifact, root / "results" / RESULT_NAME)
    return artifact
