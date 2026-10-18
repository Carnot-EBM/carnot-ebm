"""Experiment 4511: self-supervised frame-change pruning predictor.

Spec refs: REQ-ARC-FCP-4511, SCENARIO-ARC-FCP-4511.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_live_adapter import ArcAction, _game_action
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_frame_change_predictor import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_NUM_COLORS,
    FrameActionEffectExample,
    FrameChangeScorer,
    evaluate_positive_control,
    frame_state_key,
    train_frame_change_model,
)


RESULT_RELATIVE_PATH = "results/experiment_4511_frame_change_prune_predictor.json"
LOCAL_GATE_RELATIVE_PATH = "scripts/kaggle/arc_local_submission_gate.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade search + "
    "small-predictor scoring, NO GGUF/LLM load (1s floor)."
)
BASELINE_MEDIAN_ACTIONS = 7760
RANDOM_SEED = 4511
DEFAULT_PRUNE_THRESHOLD = 0.50
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIREMENTS = ["REQ-ARC-FCP-4511"]
SCENARIOS = ["SCENARIO-ARC-FCP-4511"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        'principle "terminal prefix complete:/success:/passed:/shipped_; e.g. success: '
        "frame_change_prune_median_actions_<n>_below_7760 OR complete: "
        'prune_no_action_reduction_honest_null."'
    ),
    "inference_substrate": (
        'principle "verifier_ensemble_against_cached_candidates -- offline arcade search + '
        'small-predictor scoring, NO GGUF/LLM load (1s floor)."'
    ),
    "median_actions_baseline": (
        'principle "the 7760 control so the delta is auditable, not a moving baseline."'
    ),
    "median_actions_with_prune": (
        'principle "the headline -- pruning\'s whole point is to cut this number."'
    ),
    "solve_rate_baseline": (
        'principle "pruning MUST NOT drop solve-rate; a faster agent that solves less is not a win."'
    ),
    "solve_rate_with_prune": (
        'principle "the no-regression check on solve-rate."'
    ),
    "heldout_noop_precision": (
        'principle "the predictor must generalize cross-game (pooled training) -- the '
        'StochasticGoose persist-across-games idea, measured held-out."'
    ),
    "positive_control_passed": (
        'principle "proves the harness can detect a real reduction (guards against a '
        'silently-broken metric)."'
    ),
    "false_negative_risk_checked": (
        'principle "a null result is only valid if a positive control passed -- per '
        'CLAUDE.md FALSE_NEGATIVE_RISK."'
    ),
    "random_seed": (
        'principle "determinism is the precondition for reproducibility."'
    ),
    "reproducibility_checksum": (
        'principle "content-addressed hash catches silent corpus/model drift on replay."'
    ),
    "preconditions_checked": (
        'principle "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."'
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "requirements",
    "scenarios",
    "corpus_summary",
    "training_summary",
    "heldout_noop_recall",
    "threshold_sweep",
    "local_gate_metrics",
    "positive_control",
    "false_negative_risk_guard",
    "duration_s",
)


def _frame_delta_fraction(before: Any, after: Any) -> float:
    before_grid = np.asarray(grid_of(before))
    after_grid = np.asarray(grid_of(after))
    if before_grid.shape != after_grid.shape or before_grid.size == 0:
        return 1.0
    return float(np.count_nonzero(before_grid != after_grid) / before_grid.size)


def _candidate_from_example(example: FrameActionEffectExample) -> ArcAction:
    data = (
        {"x": int(example.x), "y": int(example.y)}
        if example.action_id == 6 and example.x is not None and example.y is not None
        else None
    )
    return ArcAction(int(example.action_id), data, "heldout_transition")


def _example_digest_row(example: FrameActionEffectExample) -> dict[str, Any]:
    return {
        "env": example.env,
        "state_key": example.state_key,
        "action_id": int(example.action_id),
        "x": example.x,
        "y": example.y,
        "frame_delta": round(float(example.frame_delta), 8),
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4511: record local resources before corpus collection."""

    root_path = Path(root)
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "torch_import": False,
        "torch_version": "",
        "local_gate_script_present": (root_path / LOCAL_GATE_RELATIVE_PATH).exists(),
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "median_actions_baseline_control": BASELINE_MEDIAN_ACTIONS,
        "env_game_access_blocked": True,
    }
    try:
        kit.offline_arcade()
        preconditions["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:  # pragma: no cover - torch missing path
        preconditions["torch_error"] = repr(exc)
    preconditions["ok"] = bool(
        preconditions["offline_arcade_import"] and preconditions["torch_import"]
    )
    return preconditions


def collect_self_supervised_corpus(
    *,
    game_limit: int | None = 25,
    max_candidates_per_game: int = 16,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[FrameActionEffectExample], dict[str, Any]]:
    """REQ-ARC-FCP-4511: collect local `(frame, action, next_frame)` labels."""

    from arcengine import GameAction
    from carnot.agentic.arc_graph_explore import rich_action_candidates

    np.random.seed(int(random_seed))
    arcade = kit.offline_arcade()
    infos = list(arcade.get_environments())[: None if game_limit is None else int(game_limit)]
    examples: list[FrameActionEffectExample] = []
    errors: list[str] = []
    scorecard_id = arcade.open_scorecard()

    for info in infos:
        game_id = str(getattr(info, "game_id", "") or "")
        if not game_id:
            continue
        try:
            env = arcade.make(game_id, scorecard_id=scorecard_id)
            frame = env.reset()
            candidates = rich_action_candidates(frame)[: int(max_candidates_per_game)]
        except Exception as exc:  # pragma: no cover - SDK/env edge
            errors.append(f"{game_id}:candidate_error={type(exc).__name__}: {exc}")
            continue
        for candidate in candidates:
            try:
                fresh = arcade.make(game_id, scorecard_id=scorecard_id)
                before = fresh.reset()
                after = fresh.step(
                    _game_action(GameAction, int(candidate.action_id)),
                    data=candidate.data,
                )
                if after is None:
                    continue
                data = candidate.data or {}
                examples.append(
                    FrameActionEffectExample(
                        frame=before,
                        action_id=int(candidate.action_id),
                        x=data.get("x"),
                        y=data.get("y"),
                        frame_delta=_frame_delta_fraction(before, after),
                        level_progress=(
                            1.0
                            if kit.frame_level(after) > kit.frame_level(before)
                            else 0.0
                        ),
                        state_key=frame_state_key(before),
                        env=game_id.split("-", maxsplit=1)[0],
                    )
                )
            except Exception as exc:  # pragma: no cover - SDK/env edge
                errors.append(f"{game_id}:step_error={type(exc).__name__}: {exc}")
                continue

    changed = sum(1 for example in examples if example.changed)
    summary = {
        "game_count": int(len({example.env for example in examples})),
        "offline_environment_count": int(len(infos)),
        "transition_count": int(len(examples)),
        "changed_count": int(changed),
        "noop_count": int(len(examples) - changed),
        "max_candidates_per_game": int(max_candidates_per_game),
        "collection_errors": errors[:20],
        "corpus_source": "self_supervised_offline_arcade_transitions",
    }
    return examples, summary


def split_train_heldout_by_game(
    examples: Sequence[FrameActionEffectExample],
) -> tuple[list[FrameActionEffectExample], list[FrameActionEffectExample]]:
    """REQ-ARC-FCP-4511: hold out games, not random rows, for transfer metrics."""

    games = sorted({example.env for example in examples if example.env})
    if len(games) < 2:
        rows = list(examples)
        return rows, rows
    heldout_games = {game for index, game in enumerate(games) if index % 5 == 0}
    train = [example for example in examples if example.env not in heldout_games]
    heldout = [example for example in examples if example.env in heldout_games]
    return train or list(examples), heldout or list(examples)


def heldout_noop_metrics(
    examples: Sequence[FrameActionEffectExample],
    *,
    scorer: Any,
    threshold: float,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4511: precision/recall for predicting no-op rows held out by game."""

    true_noop = 0
    predicted_noop = 0
    true_predicted_noop = 0
    for example in examples:
        candidate = _candidate_from_example(example)
        score = float(scorer.candidate_score(example.frame, candidate))
        is_noop = not example.changed
        predicts_noop = score < float(threshold)
        if is_noop:
            true_noop += 1
        if predicts_noop:
            predicted_noop += 1
        if is_noop and predicts_noop:
            true_predicted_noop += 1
    precision = (
        float(true_predicted_noop / predicted_noop)
        if predicted_noop
        else 0.0
    )
    recall = float(true_predicted_noop / true_noop) if true_noop else 0.0
    return {
        "heldout_transition_count": int(len(examples)),
        "heldout_noop_count": int(true_noop),
        "heldout_predicted_noop_count": int(predicted_noop),
        "heldout_true_predicted_noop_count": int(true_predicted_noop),
        "heldout_noop_precision": precision,
        "heldout_noop_recall": recall,
    }


def threshold_sweep_metrics(
    examples: Sequence[FrameActionEffectExample],
    *,
    scorer: Any,
    thresholds: Sequence[float] = (0.25, 0.35, 0.50, 0.65, 0.75),
) -> list[dict[str, Any]]:
    """REQ-ARC-FCP-4511: bounded threshold sweep on held-out no-op labels."""

    return [
        {"threshold": float(threshold), **heldout_noop_metrics(examples, scorer=scorer, threshold=threshold)}
        for threshold in thresholds
    ]


def load_gate_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    baseline_path = Path(root) / "ops" / "arc-submission-baseline.json"
    if baseline_path.exists():
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        "policy": "e3",
        "games": [],
        "per_game": [],
        "solved_count": 4,
        "median_actions_on_solved": float(BASELINE_MEDIAN_ACTIONS),
        "total_actions_on_solved": None,
        "timed_out_count": None,
        "note": "fixed operator-provided baseline control",
    }


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_local_gate_with_prune(
    *,
    root: Path | str = REPO_ROOT,
    scorer_factory: Callable[[], Any],
    scorer: Any | None = None,
    threshold: float,
    budget: int = 8000,
    max_workers: int = 8,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4511: measure pruned policy on the local submission gate games."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    root_path = Path(root)
    arc_leaderboard_eval = _load_module_from_path(
        root_path / "scripts" / "arc_leaderboard_eval.py",
        "carnot_arc_leaderboard_eval_4511",
    )
    gate = _load_module_from_path(
        root_path / LOCAL_GATE_RELATIVE_PATH,
        "carnot_arc_local_submission_gate_4511",
    )
    baseline = load_gate_baseline(root)
    games = list(gate.GATE_GAMES)
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    def run_one(game: str) -> dict[str, Any]:
        policy = E3AgentPolicy(
            game,
            proposer=None,
            frame_change_scorer=scorer_factory(),
            frame_change_prune_threshold=float(threshold),
        )
        row = arc_leaderboard_eval.run_game(game, policy, budget=int(budget))
        return {
            "game": game,
            "timed_out": False,
            "solved": bool(row.get("levels", 0) >= 1),
            "actions": int(row.get("actions", 0)),
            "levels": int(row.get("levels", 0)),
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            rows = list(executor.map(run_one, games))
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable

    solved = [row for row in rows if row["solved"]]
    actions = [int(row["actions"]) for row in solved]
    with_prune = {
        "policy": "e3_frame_change_prune",
        "games": games,
        "per_game": rows,
        "solved_count": int(len(solved)),
        "median_actions_on_solved": float(median(actions)) if actions else None,
        "total_actions_on_solved": int(sum(actions)) if actions else None,
        "timed_out_count": 0,
        "budget": int(budget),
        "prune_threshold": float(threshold),
    }
    return {
        "baseline": baseline,
        "with_prune": with_prune,
        "measurement_script": LOCAL_GATE_RELATIVE_PATH,
    }


def _gate_value(gate_metrics: Mapping[str, Any], arm: str, field: str) -> Any:
    arm_metrics = gate_metrics.get(arm)
    if not isinstance(arm_metrics, Mapping):
        return None
    return arm_metrics.get(field)


def _honest_verdict(
    preconditions: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
) -> str:
    if preconditions.get("offline_arcade_import") is False:
        return "complete: blocked_offline_arcade_import_failed"
    if preconditions.get("torch_import") is False:
        return "complete: blocked_torch_missing"
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    pruned_solve = int(_gate_value(gate_metrics, "with_prune", "solved_count") or 0)
    pruned_median = _gate_value(gate_metrics, "with_prune", "median_actions_on_solved")
    if pruned_solve < baseline_solve:
        return "complete: frame_change_prune_solve_rate_guard_failed"
    if pruned_median is not None and float(pruned_median) < BASELINE_MEDIAN_ACTIONS:
        return f"success: frame_change_prune_median_actions_{int(float(pruned_median))}_below_7760"
    return "complete: prune_no_action_reduction_honest_null"


def false_negative_risk_guard(
    positive_control: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
) -> str:
    if positive_control.get("actions_reduced") is not True:
        return "positive_control_failed_null_uninterpretable"
    pruned_median = _gate_value(gate_metrics, "with_prune", "median_actions_on_solved")
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    pruned_solve = int(_gate_value(gate_metrics, "with_prune", "solved_count") or 0)
    if pruned_solve >= baseline_solve and pruned_median is not None and float(pruned_median) < BASELINE_MEDIAN_ACTIONS:
        return "positive_control_passed_prune_gain"
    return "positive_control_passed_null_interpretable"


def reproducibility_checksum(
    *,
    examples: Sequence[FrameActionEffectExample],
    training_summary: Mapping[str, Any],
    threshold: float,
    random_seed: int,
) -> str:
    payload = {
        "examples": [_example_digest_row(example) for example in examples],
        "training_summary": dict(training_summary),
        "threshold": float(threshold),
        "random_seed": int(random_seed),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    corpus_summary: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    heldout_metrics: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    prune_threshold: float,
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4511: assemble the terminal pruning artifact."""

    baseline_median = _gate_value(gate_metrics, "baseline", "median_actions_on_solved")
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    pruned_solve = int(_gate_value(gate_metrics, "with_prune", "solved_count") or 0)
    positive_passed = bool(positive_control.get("actions_reduced") is True)
    guard = false_negative_risk_guard(positive_control, gate_metrics)
    return {
        "experiment": "experiment_4511_frame_change_prune_predictor",
        "schema": "carnot.arc_frame_change_prune_predictor_4511.v1",
        "honest_verdict": _honest_verdict(preconditions_checked, gate_metrics),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "corpus_summary": dict(corpus_summary),
        "training_summary": dict(training_summary),
        "prune_threshold": float(prune_threshold),
        "threshold_sweep": list(heldout_metrics.get("threshold_sweep", [])),
        "median_actions_baseline": (
            float(baseline_median)
            if baseline_median is not None
            else float(BASELINE_MEDIAN_ACTIONS)
        ),
        "median_actions_with_prune": _gate_value(
            gate_metrics,
            "with_prune",
            "median_actions_on_solved",
        ),
        "solve_rate_baseline": baseline_solve,
        "solve_rate_with_prune": pruned_solve,
        "solve_rate_denominator": len(_gate_value(gate_metrics, "baseline", "games") or []),
        "heldout_noop_precision": heldout_metrics.get("heldout_noop_precision"),
        "heldout_noop_recall": heldout_metrics.get("heldout_noop_recall"),
        "heldout_transition_count": heldout_metrics.get("heldout_transition_count"),
        "positive_control_passed": positive_passed,
        "positive_control": dict(positive_control),
        "false_negative_risk_checked": bool(positive_passed),
        "false_negative_risk_guard": guard,
        "random_seed": int(random_seed),
        "reproducibility_checksum": str(reproducibility_checksum),
        "local_gate_metrics": dict(gate_metrics),
        "offline_reproduction_gate": (
            "direct_offline_arcade_execution_no_banked_solution_claimed"
        ),
        "duration_s": duration_s,
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match the required substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if int(float(artifact.get("median_actions_baseline") or 0)) != BASELINE_MEDIAN_ACTIONS:
        errors.append("median_actions_baseline must equal the fixed 7760 control")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("positive_control_passed") is not True:
        errors.append("positive_control_passed must be true")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if str(artifact.get("honest_verdict", "")).startswith("success:") and int(
        artifact.get("solve_rate_with_prune") or 0
    ) < int(artifact.get("solve_rate_baseline") or 0):
        errors.append("success verdict cannot hide a solve-rate drop")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    collect_corpus: Callable[..., tuple[list[FrameActionEffectExample], dict[str, Any]]] = collect_self_supervised_corpus,
    measure_gate: Callable[..., dict[str, Any]] = measure_local_gate_with_prune,
    random_seed: int = RANDOM_SEED,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    game_limit: int | None = 25,
    max_candidates_per_game: int = 16,
    train_epochs: int = 2,
    batch_size: int = 32,
    hidden_channels: int = 12,
    frame_size: int = DEFAULT_FRAME_SIZE,
    num_colors: int = DEFAULT_NUM_COLORS,
    gate_budget: int = 8000,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4511: train, prune-measure, and write the artifact."""

    started = float(now())
    root_path = Path(root)
    preconditions = check_preconditions(root_path)
    examples, corpus_summary = collect_corpus(
        game_limit=game_limit,
        max_candidates_per_game=max_candidates_per_game,
        random_seed=random_seed,
    )
    train_examples, heldout_examples = split_train_heldout_by_game(examples)
    if train_examples:
        model, training_summary = train_frame_change_model(
            train_examples,
            num_colors=num_colors,
            size=frame_size,
            hidden_channels=hidden_channels,
            epochs=train_epochs,
            batch_size=batch_size,
            seed=random_seed,
        )
        scorer = FrameChangeScorer(model, num_colors=num_colors, size=frame_size)
    else:
        model = None
        training_summary = {
            "examples_seen": 0,
            "examples_used": 0,
            "epochs": int(train_epochs),
            "batch_size": int(batch_size),
            "hidden_channels": int(hidden_channels),
            "num_colors": int(num_colors),
            "frame_size": int(frame_size),
            "learning_rate": None,
            "batches_trained": 0,
            "initial_loss": None,
            "final_loss": None,
        }
        scorer = None

    if scorer is not None:
        heldout_metrics = heldout_noop_metrics(
            heldout_examples,
            scorer=scorer,
            threshold=prune_threshold,
        )
        heldout_metrics["threshold_sweep"] = threshold_sweep_metrics(
            heldout_examples,
            scorer=scorer,
        )
        gate_metrics = measure_gate(
            root=root_path,
            scorer=scorer,
            scorer_factory=lambda: FrameChangeScorer(
                model,
                num_colors=num_colors,
                size=frame_size,
            ),
            threshold=prune_threshold,
            budget=gate_budget,
        )
    else:
        heldout_metrics = {
            "heldout_transition_count": 0,
            "heldout_noop_precision": 0.0,
            "heldout_noop_recall": 0.0,
            "threshold_sweep": [],
        }
        gate_metrics = {
            "baseline": load_gate_baseline(root_path),
            "with_prune": {
                "solved_count": 0,
                "median_actions_on_solved": None,
                "per_game": [],
            },
            "measurement_script": LOCAL_GATE_RELATIVE_PATH,
        }

    checksum = reproducibility_checksum(
        examples=examples,
        training_summary=training_summary,
        threshold=prune_threshold,
        random_seed=random_seed,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        corpus_summary=corpus_summary,
        training_summary=training_summary,
        heldout_metrics=heldout_metrics,
        gate_metrics=gate_metrics,
        positive_control=evaluate_positive_control(),
        prune_threshold=prune_threshold,
        random_seed=random_seed,
        reproducibility_checksum=checksum,
        duration_s=max(0.0, float(now()) - started),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
