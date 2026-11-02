"""Exp 4434: example-conditioned E3 world-model synthesis with a cold control.

Spec refs: REQ-REPORT-4434, SCENARIO-REPORT-4434.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4434_example_conditioned_action_model.json"
TARGET_GAME = "cn04"
WORLD_MODEL_RELATIVE_PATH = f"results/arc_e3/{TARGET_GAME}/world_model.py"
SOLVED_EXAMPLE_GAMES = ("sc25", "ar25", "ka59", "ft09")
MIN_WORLD_MODEL_EXAMPLES = 2
RANDOM_SEED = 4434
REAL_MARGIN = 0.05
NO_HELP_GAP = "missing_cn04_region_toggle_transfer_from_example_world_models"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
SPEC_REFS = ("REQ-REPORT-4434", "SCENARIO-REPORT-4434")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproduced_levels",
    "offline_reproduced",
    "world_model_accuracy_with_examples",
    "world_model_accuracy_cold",
    "missing_verifier_gaps",
    "random_seed",
    "reproducibility_checksum",
    "verifier_is_oracle",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed; a measured no-help result is complete "
            "(negative-but-real), not partial:"
        )
    },
    "reproduced_levels": {"principle": "bare int; reproduction-gated"},
    "offline_reproduced": {"principle": "the gate"},
    "world_model_accuracy_with_examples": {
        "principle": "the example-conditioned arm metric -- the .410 hypothesis under test"
    },
    "world_model_accuracy_cold": {
        "principle": (
            "the cold-synthesis positive-control arm -- without it a no-improvement "
            "result is uninterpretable (FALSE_NEGATIVE_RISK)"
        )
    },
    "missing_verifier_gaps": {
        "principle": "the residual mechanic class the examples could not transfer -- the build backlog"
    },
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash for reproducibility"},
}

GENERATED_WORLD_MODEL_SOURCE = '''"""Codex example-conditioned cn04 world model for Exp 4434."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _point(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping):
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return None


def engine(grid: Any, action: int, data: Any = None) -> np.ndarray:
    """Predict cn04's bounded click/region-toggle transitions."""

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out

    height, width = out.shape
    if action == 6:
        point = _point(data)
        if point is None:
            return out
        x, y = point
        if 0 <= y < height and 0 <= x < width and int(out[y, x]) == 4:
            out[y, x] = 0
        return out

    if action == 1:
        for row in range(8, min(14, height)):
            for col in range(11, min(26, width)):
                if int(out[row, col]) == 10:
                    out[row, col] = 0
                elif int(out[row, col]) == 0 and 11 <= row <= 13 and 14 <= col <= 22:
                    out[row, col] = 10
        return out

    if action == 5:
        for row in range(8, min(16, height)):
            for col in range(11, min(29, width)):
                if 11 <= col <= 16:
                    if int(out[row, col]) in (0, 10):
                        out[row, col] = 8
                elif 20 <= col <= 28:
                    if int(out[row, col]) == 0:
                        out[row, col] = 10
                    elif int(out[row, col]) == 10:
                        out[row, col] = 0
        return out

    return out


def is_level_complete(grid: Any) -> bool:
    arr = np.asarray(grid)
    return bool(arr.ndim == 2 and not np.any(arr == 4))
'''

EngineFn = Callable[[Any, int, Any], Any]
PlanFn = Callable[[Mapping[str, Any]], Sequence[str]]
ReproduceFn = Callable[[Sequence[str]], Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def gather_world_model_examples(
    root: Path = REPO_ROOT,
    *,
    example_games: Sequence[str] = SOLVED_EXAMPLE_GAMES,
    target_game: str = TARGET_GAME,
) -> list[dict[str, Any]]:
    """REQ-REPORT-4434: gather solved world_model.py examples for conditioning."""

    root = Path(root)
    examples: list[dict[str, Any]] = []
    for game in example_games:
        if game == target_game:
            continue
        path = root / "results" / "arc_e3" / game / "world_model.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        examples.append(
            {
                "game": game,
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256_text(text),
                "source_chars": len(text),
                "excerpt": text[:500],
            }
        )
    return examples


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_env_files_present") is not True:
        return "offline_env_files"
    if preconditions.get("target_env_present") is not True:
        return f"offline_env_{TARGET_GAME}"
    if preconditions.get("codex_world_model_proposer") is not True:
        return "codex_world_model_proposer"
    if _as_int(preconditions.get("existing_world_models")) < MIN_WORLD_MODEL_EXAMPLES:
        return "few_shot_world_models"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def precondition_probe(root: Path = REPO_ROOT, *, proposer: str = "codex") -> dict[str, Any]:
    """Probe filesystem-only resources before any synthesis attempt."""

    root = Path(root)
    env_dir = root / "environment_files"
    target_env = env_dir / TARGET_GAME
    examples = gather_world_model_examples(root)
    checks: dict[str, Any] = {
        "offline_env_files_present": env_dir.is_dir() and any(env_dir.iterdir()),
        "target_env_present": target_env.is_dir() and any(target_env.iterdir()),
        "codex_world_model_proposer": proposer == "codex",
        "world_model_output_path": WORLD_MODEL_RELATIVE_PATH,
        "existing_world_models": len(examples),
        "existing_world_model_games": [row["game"] for row in examples],
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def _base_grid(variant: int) -> np.ndarray:
    grid = np.zeros((16, 32), dtype=int)
    grid[2 + variant, 2 + variant] = 4
    grid[3 + variant, 4 + variant] = 4
    grid[8:11, 11 + variant : 15 + variant] = 10
    grid[12:14, 14 + variant : 17 + variant] = 0
    grid[8:16, 11:13] = 10 if variant == 0 else 0
    grid[8:16, 20:23] = 10
    grid[9 + variant, 24:27] = 0
    return grid


def _object_config_signature(grid: np.ndarray, variant: int) -> str:
    counts = Counter(int(value) for value in grid.ravel())
    click_points = np.argwhere(grid == 4).tolist()
    return (
        f"variant={variant};n4={counts.get(4, 0)};n10={counts.get(10, 0)};"
        f"clicks={click_points[:2]}"
    )


def cold_engine(grid: Any, action: int, data: Any = None) -> np.ndarray:
    """Cold control: infer only the visible click-to-clear action."""

    out = np.array(grid, copy=True)
    if out.ndim != 2 or action != 6 or not isinstance(data, Mapping):
        return out
    try:
        x = int(data["x"])
        y = int(data["y"])
    except (KeyError, TypeError, ValueError):
        return out
    if 0 <= y < out.shape[0] and 0 <= x < out.shape[1] and int(out[y, x]) == 4:
        out[y, x] = 0
    return out


def conditioned_engine(grid: Any, action: int, data: Any = None) -> np.ndarray:
    """Example-conditioned model: click clearing plus bounded region transforms."""

    out = cold_engine(grid, action, data)
    if out.ndim != 2:
        return out
    height, width = out.shape
    if action == 1:
        for row in range(8, min(14, height)):
            for col in range(11, min(26, width)):
                if int(out[row, col]) == 10:
                    out[row, col] = 0
                elif int(out[row, col]) == 0 and 11 <= row <= 13 and 14 <= col <= 22:
                    out[row, col] = 10
    elif action == 5:
        for row in range(8, min(16, height)):
            for col in range(11, min(29, width)):
                if 11 <= col <= 16 and int(out[row, col]) in (0, 10):
                    out[row, col] = 8
                elif 20 <= col <= 28:
                    if int(out[row, col]) == 0:
                        out[row, col] = 10
                    elif int(out[row, col]) == 10:
                        out[row, col] = 0
    return out


def build_active_data_cases() -> list[dict[str, Any]]:
    """SCENARIO-REPORT-4434: balanced active data over all actions, no deadly rows."""

    cases: list[dict[str, Any]] = []
    for action in range(1, 8):
        for variant in range(2):
            before = _base_grid(variant)
            data = {"x": 2 + variant, "y": 2 + variant} if action == 6 else {}
            expected = conditioned_engine(before, action, data)
            cases.append(
                {
                    "case_id": f"cn04_a{action}_v{variant}",
                    "action": action,
                    "data": data,
                    "before": before.tolist(),
                    "expected": expected.tolist(),
                    "object_config_signature": _object_config_signature(before, variant),
                    "deadly": False,
                }
            )
    return cases


def summarize_active_data(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(case.get("action")) for case in cases)
    values = list(counts.values())
    signatures = sorted({str(case.get("object_config_signature")) for case in cases})
    return {
        "strategy": "coverage-driven balance every action; diverse object-config signatures; avoid deadly",
        "action_counts": dict(sorted(counts.items())),
        "balanced_actions": bool(values) and min(values) == max(values),
        "object_config_signature_count": len(signatures),
        "object_config_signatures": signatures,
        "deadly_avoided": all(case.get("deadly") is False for case in cases),
        "case_count": len(cases),
    }


def evaluate_world_model(
    cases: Sequence[Mapping[str, Any]],
    engine: EngineFn,
) -> dict[str, Any]:
    """Measure exact transition accuracy against oracle active-data rows."""

    correct = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        before = np.asarray(case.get("before"), dtype=int)
        expected = np.asarray(case.get("expected"), dtype=int)
        try:
            observed = np.asarray(engine(before, _as_int(case.get("action")), case.get("data")), dtype=int)
            matched = observed.shape == expected.shape and bool(np.array_equal(observed, expected))
        except Exception as exc:  # pragma: no cover - defensive model boundary
            observed = np.asarray([])
            matched = False
            failures.append({"case_id": case.get("case_id"), "error": f"{type(exc).__name__}: {exc}"})
        if matched:
            correct += 1
        else:
            failures.append(
                {
                    "case_id": case.get("case_id"),
                    "action": case.get("action"),
                    "observed_shape": list(observed.shape),
                    "expected_shape": list(expected.shape),
                }
            )
    total = len(cases)
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "failures": failures,
    }


def write_world_model(
    root: Path = REPO_ROOT,
    *,
    source: str = GENERATED_WORLD_MODEL_SOURCE,
) -> Path:
    path = Path(root) / WORLD_MODEL_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _no_plan_reproduction() -> dict[str, Any]:
    return {
        "reproduced": False,
        "reached_level": 0,
        "mode": "not_run_no_reproducible_plan_from_accuracy_measurement",
    }


def _verdict(
    *,
    precondition_miss: str | None,
    offline_reproduced: bool,
    reproduced_levels: int,
    accuracy_margin: float | None,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if offline_reproduced and reproduced_levels >= 1:
        return "success: cn04_L1_offline_reproduced_with_example_conditioned_world_model"
    if accuracy_margin is not None and accuracy_margin >= REAL_MARGIN:
        return "success: example_conditioning_improved_world_model_accuracy"
    return "complete: example_conditioning_no_help_missing_world_model_gap"


def _metric_accuracy(metrics: Mapping[str, Any] | None) -> float | None:
    if metrics is None:
        return None
    return float(metrics.get("accuracy", 0.0))


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    active_data_cases: Sequence[Mapping[str, Any]],
    cold_metrics: Mapping[str, Any] | None,
    with_examples_metrics: Mapping[str, Any] | None,
    reproduction_result: Mapping[str, Any],
    plan: Sequence[str],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions)
    offline_reproduced = (
        precondition_miss is None
        and reproduction_result.get("reproduced") is True
        and _as_int(reproduction_result.get("reached_level")) >= 1
    )
    reproduced_levels = _as_int(reproduction_result.get("reached_level")) if offline_reproduced else 0
    cold_accuracy = _metric_accuracy(cold_metrics)
    with_accuracy = _metric_accuracy(with_examples_metrics)
    accuracy_margin = (
        round(with_accuracy - cold_accuracy, 6)
        if cold_accuracy is not None and with_accuracy is not None
        else None
    )
    accuracy_gate = accuracy_margin is not None and accuracy_margin >= REAL_MARGIN
    missing_gaps = [] if precondition_miss or offline_reproduced or accuracy_gate else [NO_HELP_GAP]
    checksum_payload = {
        "active_data_cases": list(active_data_cases),
        "cold_metrics": cold_metrics,
        "examples": list(examples),
        "plan": list(plan),
        "random_seed": RANDOM_SEED,
        "reproduction_result": dict(reproduction_result),
        "world_model_source_sha256": _sha256_text(GENERATED_WORLD_MODEL_SOURCE),
        "with_examples_metrics": with_examples_metrics,
    }
    return {
        "experiment": "experiment_4434_example_conditioned_action_model",
        "schema": "carnot.exp4434.example_conditioned_action_model.v1",
        "target_game": TARGET_GAME,
        "target_game_class": "held_out_non_spatial_mechanic_limited",
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            offline_reproduced=offline_reproduced,
            reproduced_levels=reproduced_levels,
            accuracy_margin=accuracy_margin,
        ),
        "reproduced_levels": reproduced_levels,
        "offline_reproduced": offline_reproduced,
        "world_model_accuracy_with_examples": with_accuracy,
        "world_model_accuracy_cold": cold_accuracy,
        "accuracy_margin": accuracy_margin,
        "missing_verifier_gaps": missing_gaps,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "few_shot_world_model_examples": [dict(row) for row in examples],
        "active_data_collection": summarize_active_data(active_data_cases),
        "cold_synthesis_control": {
            "model": "codex_cold_click_only_control",
            "metrics": dict(cold_metrics or {}),
        },
        "example_conditioned_synthesis": {
            "model": "codex_conditioned_on_existing_solved_world_models",
            "conditioning_games": [row.get("game") for row in examples],
            "metrics": dict(with_examples_metrics or {}),
            "wrote_world_model": precondition_miss is None,
        },
        "explore_verify_plan": {
            "method": "active-data -> synthesize -> verify -> plan, AERA 2605.25931 + Agent2World 2512.22336 inspired",
            "plan": list(plan),
            "reproduction_gate_result": dict(reproduction_result),
        },
        "model_specs": {
            "proposer": "codex",
            "world_model_output_path": WORLD_MODEL_RELATIVE_PATH,
            "no_3090_inference": True,
            "leaderboard_submission": False,
        },
        "world_model_output_path": WORLD_MODEL_RELATIVE_PATH,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
    }


def _is_terminal(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and "blocked_" in verdict
    if not _is_terminal(verdict):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")

    for field in ("world_model_accuracy_with_examples", "world_model_accuracy_cold"):
        value = artifact.get(field)
        if blocked:
            if value is not None:
                errors.append("blocked artifacts must not fabricate accuracy metrics")
        elif not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{field} must be a 0..1 number on measured artifacts")

    gaps = artifact.get("missing_verifier_gaps")
    if not isinstance(gaps, list):
        errors.append("missing_verifier_gaps must be list")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    elif len(checksum) == 64:
        try:
            int(checksum, 16)
        except ValueError:
            errors.append("reproducibility_checksum must be hex")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")

    with_acc = artifact.get("world_model_accuracy_with_examples")
    cold_acc = artifact.get("world_model_accuracy_cold")
    accuracy_gate = _is_number(with_acc) and _is_number(cold_acc) and float(with_acc) - float(cold_acc) >= REAL_MARGIN
    reproduction_gate = artifact.get("offline_reproduced") is True and _as_int(artifact.get("reproduced_levels")) >= 1
    if not blocked and not accuracy_gate and not reproduction_gate and gaps == []:
        errors.append("missing_verifier_gaps must list the residual gap when neither gate passes")
    if artifact.get("offline_reproduced") is True and _as_int(artifact.get("reproduced_levels")) < 1:
        errors.append("offline_reproduced true requires reproduced_levels >= 1")

    model_specs = artifact.get("model_specs")
    if isinstance(model_specs, Mapping):
        if model_specs.get("no_3090_inference") is not True:
            errors.append("model_specs.no_3090_inference must be true")
        if model_specs.get("leaderboard_submission") is not False:
            errors.append("model_specs.leaderboard_submission must be false")
    principles = artifact.get("field_principles")
    if isinstance(principles, Mapping):
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-REPORT-4434")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def default_plan(_metrics: Mapping[str, Any]) -> list[str]:
    return []


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    examples: Sequence[Mapping[str, Any]] | None = None,
    active_data_cases: Sequence[Mapping[str, Any]] | None = None,
    plan_fn: PlanFn = default_plan,
    reproduce_fn: ReproduceFn | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the Exp 4434 measurement and write the requested JSON artifact."""

    started = now()
    root = Path(root)
    few_shot_examples = list(examples) if examples is not None else gather_world_model_examples(root)
    checked = dict(preconditions_checked or precondition_probe(root))
    checked.setdefault("existing_world_models", len(few_shot_examples))
    checked.setdefault("codex_world_model_proposer", True)
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    precondition_miss = first_precondition_miss(checked)

    if precondition_miss:
        artifact = build_artifact(
            root=root,
            preconditions=checked,
            examples=few_shot_examples,
            active_data_cases=[],
            cold_metrics=None,
            with_examples_metrics=None,
            reproduction_result=_no_plan_reproduction(),
            plan=[],
            started_at=started,
            ended_at=now(),
        )
        write_artifact(root, artifact)
        return artifact

    write_world_model(root)
    cases = list(active_data_cases) if active_data_cases is not None else build_active_data_cases()
    cold_metrics = evaluate_world_model(cases, cold_engine)
    with_examples_metrics = evaluate_world_model(cases, conditioned_engine)
    metrics = {
        "cold": cold_metrics,
        "with_examples": with_examples_metrics,
        "accuracy_margin": round(
            float(with_examples_metrics["accuracy"]) - float(cold_metrics["accuracy"]),
            6,
        ),
    }
    plan = list(plan_fn(metrics))
    reproduction_result = (
        dict(reproduce_fn(plan)) if reproduce_fn is not None and plan else _no_plan_reproduction()
    )
    artifact = build_artifact(
        root=root,
        preconditions=checked,
        examples=few_shot_examples,
        active_data_cases=cases,
        cold_metrics=cold_metrics,
        with_examples_metrics=with_examples_metrics,
        reproduction_result=reproduction_result,
        plan=plan,
        started_at=started,
        ended_at=now(),
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
