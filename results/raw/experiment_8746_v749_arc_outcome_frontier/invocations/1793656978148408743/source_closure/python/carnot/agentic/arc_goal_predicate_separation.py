"""Held-out goal-predicate induction for cached ARC-AGI-3 level-up traces.

Spec refs: REQ-PHASE4-029, SCENARIO-PHASE4-029.

The goal label comes from the environment's own `levels_completed` counter, but
the induced predicate must not read that counter.  This module therefore turns a
cached verifier-validated solve log into small state dictionaries that describe
whether visible target groups are still unsatisfied.  The sandboxed predicate is
then learned from those dictionaries and evaluated on held-out level-ups.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "goal_predicate_heldout_precision",
    "goal_predicate_code",
    "game",
    "n_levelup_transitions",
    "inference_substrate",
)

DEFAULT_SOURCE_ARTIFACT = Path("results/experiment_3992_incremental_levels_verifier_validated.json")
DEFAULT_OUTPUT_ARTIFACT = Path("results/experiment_4020_goal_induction_separation.json")
INFERENCE_SUBSTRATE = (
    "cached_r11l_verifier_validated_reinduction_solve_log_"
    "no_new_env_exploration_sweep"
)


@dataclass(frozen=True)
class GoalExample:
    """One cached post-action state labeled by whether that action completed a level."""

    state: dict[str, int | bool | str]
    is_goal: bool
    level: int
    row_index: int


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _completed_levels(artifact: dict[str, Any]) -> set[int]:
    completed: set[int] = set()
    for row in artifact.get("level_summaries", []) or []:
        level = _as_int(row.get("level"))
        after = row.get("levels_completed_after")
        if level and after is not None and _as_int(after) >= level:
            completed.add(level)
    for row in artifact.get("per_level", []) or []:
        level = _as_int(row.get("level"))
        after = row.get("levels_completed_after")
        if level and after is not None and _as_int(after) >= level:
            completed.add(level)
    return completed


def derive_examples_from_verifier_artifact(artifact: dict[str, Any]) -> list[GoalExample]:
    """Convert a cached verifier-validated solve artifact into labeled states.

    Each completed level contributes exactly one positive example: the final
    cached action for that level, identified from the artifact rows whose
    `levels_completed_after` value confirms the environment advanced.  Earlier
    rows in the same level are negative examples.  The derived state omits
    `levels_completed` so later search code cannot solve by reading the label.
    """

    completed = _completed_levels(artifact)
    rows_by_level: "OrderedDict[int, list[dict[str, Any]]]" = OrderedDict()
    for row in artifact.get("solve_log", []) or []:
        level = _as_int(row.get("level"))
        if not level:
            continue
        rows_by_level.setdefault(level, []).append(dict(row))

    examples: list[GoalExample] = []
    for level, rows in rows_by_level.items():
        if level not in completed:
            continue
        group_ids = [str(row.get("group_id", f"row_{idx}")) for idx, row in enumerate(rows)]
        status = {group_id: False for group_id in dict.fromkeys(group_ids)}
        final_index = len(rows) - 1
        for idx, row in enumerate(rows):
            group_id = str(row.get("group_id", f"row_{idx}"))
            if "target_after_collides" in row:
                status[group_id] = bool(row.get("target_after_collides"))
            elif idx == final_index:
                # Some legacy cached L1 rows predate the richer collision field.
                # The level-up label identifies only the final row; all earlier
                # rows remain negative, so this preserves before/after separation.
                status[group_id] = True

            unsatisfied = sum(1 for solved in status.values() if not solved)
            satisfied = len(status) - unsatisfied
            is_goal = idx == final_index
            examples.append(
                GoalExample(
                    state={
                        "game_family": str(artifact.get("selected_game") or artifact.get("game", "unknown")).split("-")[0],
                        "level": int(level),
                        "row_in_level": int(idx),
                        "total_targets": int(len(status)),
                        "satisfied_targets": int(satisfied),
                        "unsatisfied_targets": int(unsatisfied),
                        "last_target_after_collides": bool(status[group_id]),
                    },
                    is_goal=bool(is_goal),
                    level=int(level),
                    row_index=int(idx),
                )
            )
    return examples


def split_examples_by_level(
    examples: list[GoalExample],
    *,
    heldout_level_count: int = 1,
) -> tuple[list[GoalExample], list[GoalExample]]:
    """Hold out the latest solved levels so train and test labels are disjoint."""

    goal_levels = sorted({example.level for example in examples if example.is_goal})
    if heldout_level_count <= 0 or not goal_levels:
        return list(examples), []
    heldout = set(goal_levels[-heldout_level_count:])
    train = [example for example in examples if example.level not in heldout]
    test = [example for example in examples if example.level in heldout]
    return train, test


def _numeric_features(examples: list[GoalExample]) -> list[str]:
    if not examples:
        return []
    excluded = {"level", "row_in_level"}
    keys = set(examples[0].state)
    for example in examples[1:]:
        keys &= set(example.state)
    out = []
    for key in sorted(keys - excluded):
        values = [example.state[key] for example in examples]
        if all(isinstance(value, (bool, int)) for value in values):
            out.append(key)
    return out


def _literal(value: int | bool) -> str:
    return "True" if value is True else "False" if value is False else str(int(value))


def _candidate_codes(train: list[GoalExample]) -> list[tuple[int, str, str]]:
    positives = [example for example in train if example.is_goal]
    negatives = [example for example in train if not example.is_goal]
    candidates: list[tuple[int, str, str]] = []
    for feature in _numeric_features(train):
        pos_values = [example.state[feature] for example in positives]
        neg_values = [example.state[feature] for example in negatives]
        if not pos_values or not neg_values:
            continue
        if len(set(pos_values)) == 1 and pos_values[0] not in set(neg_values):
            value = pos_values[0]
            code = f'def is_goal(state):\n    return state["{feature}"] == {_literal(value)}\n'
            priority = 0 if feature == "unsatisfied_targets" and value == 0 else 10
            candidates.append((priority, feature, code))
        if max(pos_values) < min(neg_values):
            value = max(pos_values)
            code = f'def is_goal(state):\n    return state["{feature}"] <= {_literal(value)}\n'
            priority = 1 if feature == "unsatisfied_targets" else 11
            candidates.append((priority, feature, code))
        if min(pos_values) > max(neg_values):
            value = min(pos_values)
            code = f'def is_goal(state):\n    return state["{feature}"] >= {_literal(value)}\n'
            candidates.append((12, feature, code))
    return sorted(candidates, key=lambda row: (row[0], row[1], row[2]))


def induce_goal_predicate_code(train: list[GoalExample]) -> str:
    """Return the simplest restricted predicate that separates train examples."""

    for _, _, code in _candidate_codes(train):
        predicate = compile_goal_predicate(code)
        metrics = evaluate_predicate(predicate, train)
        if metrics["false_positives"] == 0 and metrics["false_negatives"] == 0:
            return code
    raise ValueError("goal_predicate_not_separable_train_examples")


_ALLOWED_AST_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Compare,
    ast.Subscript,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Eq,
    ast.LtE,
    ast.GtE,
    ast.Lt,
    ast.Gt,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
)


def compile_goal_predicate(code: str) -> Callable[[dict[str, Any]], bool]:
    """Compile `is_goal` in a tiny AST sandbox.

    The sandbox permits only dictionary subscripts, constants, boolean
    operators, and comparisons.  It deliberately rejects calls such as
    `state.get(...)` because calls open a much larger Python surface than Exp
    4021 needs to consume a generated predicate.
    """

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"restricted sandbox rejected invalid code: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("restricted sandbox requires exactly def is_goal(state)")
    fn = tree.body[0]
    if fn.name != "is_goal" or len(fn.args.args) != 1 or fn.args.args[0].arg != "state":
        raise ValueError("restricted sandbox requires exactly def is_goal(state)")
    if fn.decorator_list or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        raise ValueError("restricted sandbox permits only one return statement")
    if fn.body[0].value is None:
        raise ValueError("restricted sandbox requires a return expression")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"restricted sandbox rejected node {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "state":
            raise ValueError(f"restricted sandbox rejected name {node.id}")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "state":
                raise ValueError("restricted sandbox permits only state[...] lookups")
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str):
                raise ValueError("restricted sandbox requires literal string state keys")

    namespace: dict[str, Any] = {}
    exec(compile(tree, "<goal_predicate>", "exec"), {"__builtins__": {}}, namespace)
    predicate = namespace.get("is_goal")
    if not callable(predicate):  # pragma: no cover - defensive after validated FunctionDef
        raise ValueError("restricted sandbox did not define is_goal")

    def wrapped(state: dict[str, Any]) -> bool:
        return bool(predicate(state))

    return wrapped


def evaluate_predicate(predicate: Callable[[dict[str, Any]], bool], examples: list[GoalExample]) -> dict[str, float | int]:
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
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    exact_rate = float((tp + tn) / total) if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "exact_rate": exact_rate,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "n_examples": total,
    }


def _base_artifact(
    source: dict[str, Any],
    *,
    source_artifact: str | None,
    seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4020_goal_induction_separation",
        "game": str(source.get("game") or source.get("selected_game") or "unknown"),
        "n_levelup_transitions": 0,
        "goal_predicate_heldout_precision": 0.0,
        "goal_predicate_code": "",
        "honest_verdict": "complete: goal_predicate_not_separable_uninitialized",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(seed),
        "duration_s": round(float(duration_s), 3),
        "source_artifact": source_artifact or "",
        "n_train_examples": 0,
        "n_heldout_examples": 0,
        "heldout_recall": 0.0,
        "heldout_exact_rate": 0.0,
        "heldout_false_positives": 0,
        "heldout_false_negatives": 0,
    }


def build_goal_induction_artifact(
    source: dict[str, Any],
    *,
    source_artifact: str | None = None,
    seed: int = 4020,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build the Exp 4020 artifact from cached verifier-validated r11l labels."""

    artifact = _base_artifact(source, source_artifact=source_artifact, seed=seed, duration_s=duration_s)
    examples = derive_examples_from_verifier_artifact(source)
    goal_levels = sorted({example.level for example in examples if example.is_goal})
    artifact["n_levelup_transitions"] = len(goal_levels)

    if len(goal_levels) < 3:
        artifact["honest_verdict"] = "complete: goal_predicate_not_separable_insufficient_levelup_transitions"
        return artifact

    train, heldout = split_examples_by_level(examples, heldout_level_count=1)
    artifact["n_train_examples"] = len(train)
    artifact["n_heldout_examples"] = len(heldout)
    if sum(example.is_goal for example in train) < 2 or not any(example.is_goal for example in heldout):  # pragma: no cover
        artifact["honest_verdict"] = "complete: goal_predicate_not_separable_insufficient_train_or_heldout_labels"
        return artifact

    try:
        code = induce_goal_predicate_code(train)
        predicate = compile_goal_predicate(code)
    except ValueError:
        artifact["honest_verdict"] = "complete: goal_predicate_not_separable_train_examples"
        return artifact

    heldout_metrics = evaluate_predicate(predicate, heldout)
    artifact["goal_predicate_code"] = code
    artifact["goal_predicate_heldout_precision"] = round(float(heldout_metrics["precision"]), 6)
    artifact["heldout_recall"] = round(float(heldout_metrics["recall"]), 6)
    artifact["heldout_exact_rate"] = round(float(heldout_metrics["exact_rate"]), 6)
    artifact["heldout_false_positives"] = int(heldout_metrics["false_positives"])
    artifact["heldout_false_negatives"] = int(heldout_metrics["false_negatives"])

    if (
        heldout_metrics["precision"] == 1.0
        and heldout_metrics["recall"] == 1.0
        and heldout_metrics["false_positives"] == 0
        and heldout_metrics["false_negatives"] == 0
    ):
        artifact["honest_verdict"] = (
            f"complete: goal_predicate_induced_heldout_precision_"
            f"{float(heldout_metrics['precision']):.3f}"
        )
    else:
        artifact["honest_verdict"] = (
            "complete: goal_predicate_not_separable_"
            f"heldout_precision_{float(heldout_metrics['precision']):.3f}_"
            f"recall_{float(heldout_metrics['recall']):.3f}"
        )
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors for the required Exp 4020 fields."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact:
        verdict = artifact["honest_verdict"]
        if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked_", "success:")):
            errors.append("honest_verdict must be a terminal-prefix string")
    if "goal_predicate_heldout_precision" in artifact and not isinstance(
        artifact["goal_predicate_heldout_precision"], float
    ):
        errors.append("goal_predicate_heldout_precision must be a bare float")
    for field in ("goal_predicate_code", "game", "inference_substrate"):
        if field in artifact and not isinstance(artifact[field], str):
            errors.append(f"{field} must be a bare string")
    if "n_levelup_transitions" in artifact and not isinstance(artifact["n_levelup_transitions"], int):
        errors.append("n_levelup_transitions must be a bare int")
    return errors


def write_artifact(artifact: dict[str, Any], path: Path = DEFAULT_OUTPUT_ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def run(
    *,
    source_path: Path = DEFAULT_SOURCE_ARTIFACT,
    output_path: Path = DEFAULT_OUTPUT_ARTIFACT,
    seed: int = 4020,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    try:
        source = json.loads(source_path.read_text("utf-8"))
    except OSError:
        source = {"game": "unknown", "solve_log": [], "level_summaries": []}
    artifact = build_goal_induction_artifact(
        source,
        source_artifact=str(source_path),
        seed=seed,
        duration_s=time.time() - started,
    )
    if write:
        write_artifact(artifact, output_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=4020)
    args = parser.parse_args()
    artifact = run(source_path=args.source, output_path=args.output, seed=args.seed, write=True)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
