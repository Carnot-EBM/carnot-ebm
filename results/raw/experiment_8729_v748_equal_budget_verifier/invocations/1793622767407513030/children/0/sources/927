"""Experiment 4527: nav metric local submission gate hardening.

Spec refs: REQ-ARC-FCP-4527, SCENARIO-ARC-FCP-4527.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4527_nav_metric_harness.json"
GATE_RELATIVE_PATH = "scripts/kaggle/arc_local_submission_gate.py"
BASELINE_RELATIVE_PATH = "ops/arc-submission-baseline.json"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_arc_submission_gate_verdict.py "
    "tests/python/test_experiment_4527_nav_metric_harness.py -q --no-cov"
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
)
CORE_EFFICIENCY_BASELINE = 2.0074
REQUIREMENTS = ("REQ-ARC-FCP-4527",)
SCENARIOS = ("SCENARIO-ARC-FCP-4527",)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; shipped: nav_metric_first_class_ci_guarded OR "
        "complete: nav_metric_partial_<reason>."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
    ),
    "nav_metric_added": (
        "names the per-game nav fields the gate now tracks -- the nav-regression early warning."
    ),
    "tests_added_pass": "Tests Must Run and Assert.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "gate_result",
    "baseline_guard",
    "canonical_game_set",
    "canonical_core_games",
    "core_efficiency_baseline",
    "score_metric_primary",
    "nav_metric_secondary",
    "per_game_deepest_level_reached",
    "per_game_per_level_efficiency",
    "per_game_nav_diagnostics",
    "reproducibility_checksum",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


def _load_gate_module() -> Any:
    gate_path = REPO_ROOT / GATE_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("arc_local_submission_gate_4527", gate_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib failure boundary.
        raise RuntimeError(f"cannot load {gate_path}")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


GATE = _load_gate_module()
CANONICAL_GAME_SET = tuple(GATE.CANONICAL_GAME_SET)
CANONICAL_CORE_GAMES = tuple(GATE.CANONICAL_CORE_GAMES)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _current_measurement(gate_result: Mapping[str, Any]) -> Mapping[str, Any]:
    current = gate_result.get("current")
    return current if isinstance(current, Mapping) else {}


def _map_int(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(game): int(level or 0) for game, level in value.items()}


def _deepest_level_by_game(current: Mapping[str, Any]) -> dict[str, int]:
    explicit = _map_int(current.get("deepest_level_by_game"))
    if explicit:
        return explicit
    out: dict[str, int] = {}
    for row in current.get("per_game", []) or []:
        if not isinstance(row, Mapping) or row.get("game") is None:
            continue
        out[str(row["game"])] = int(row.get("deepest_level_reached", row.get("reached", row.get("levels", 0))) or 0)
    return out


def _per_level_efficiency_by_game(current: Mapping[str, Any]) -> dict[str, float]:
    for key in ("per_level_efficiency_by_game", "efficiency_by_game"):
        value = current.get(key)
        if isinstance(value, Mapping) and value:
            return {str(game): float(score or 0.0) for game, score in value.items()}
    out: dict[str, float] = {}
    for row in current.get("per_game", []) or []:
        if not isinstance(row, Mapping) or row.get("game") is None:
            continue
        score = row.get("per_level_efficiency", row.get("efficiency"))
        if score is not None:
            out[str(row["game"])] = float(score)
    return out


def _nav_by_game(current: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    explicit = current.get("navigation_by_game")
    if isinstance(explicit, Mapping) and explicit:
        return {
            str(game): {
                "reset_replay_steps": int((value or {}).get("reset_replay_steps") or 0),
                "forward_walk_hit_rate": float((value or {}).get("forward_walk_hit_rate") or 0.0),
            }
            for game, value in explicit.items()
            if isinstance(value, Mapping)
        }
    out: dict[str, dict[str, float]] = {}
    for row in current.get("per_game", []) or []:
        if not isinstance(row, Mapping) or row.get("game") is None:
            continue
        diagnostics = row.get("navigation_diagnostics")
        if not isinstance(diagnostics, Mapping):
            diagnostics = row
        out[str(row["game"])] = {
            "reset_replay_steps": int(diagnostics.get("reset_replay_steps") or 0),
            "forward_walk_hit_rate": float(diagnostics.get("forward_walk_hit_rate") or 0.0),
        }
    return out


def _extract_json_object(text: str) -> dict[str, Any]:  # pragma: no cover - subprocess boundary.
    start = text.find("{")
    if start < 0:
        return {}
    return json.loads(text[start:])


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - local resource boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "gate_help_precondition": False,
        "baseline_file_present": (root_path / BASELINE_RELATIVE_PATH).exists(),
        "spec_has_req_4527": spec_path.exists() and "REQ-ARC-FCP-4527" in spec_path.read_text(encoding="utf-8"),
    }
    cmd = [str(root_path / ".venv" / "bin" / "python"), str(root_path / GATE_RELATIVE_PATH), "--help"]
    try:
        completed = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=30)
        checks["gate_help_precondition"] = completed.returncode == 0
        if completed.returncode != 0:
            checks["gate_help_error"] = completed.stderr[-500:]
    except Exception as exc:
        checks["gate_help_error"] = repr(exc)
    checks["ok"] = bool(
        checks["agents_md_read"]
        and checks["codex_md_read"]
        and checks["gate_help_precondition"]
        and checks["baseline_file_present"]
        and checks["spec_has_req_4527"]
    )
    return checks


def run_gate(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - offline gate boundary.
    root_path = Path(root)
    cmd = [
        str(root_path / ".venv" / "bin" / "python"),
        str(root_path / GATE_RELATIVE_PATH),
        "--check",
        "--json",
        "--lever",
        "submitted_default",
    ]
    completed = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=240)
    parsed = _extract_json_object(completed.stdout)
    parsed["gate_command"] = " ".join(cmd)
    parsed["gate_returncode"] = completed.returncode
    if completed.stderr:
        parsed["gate_stderr"] = completed.stderr[-2000:]
    if not parsed:
        return {
            "pass": False,
            "verdict": "gate_json_parse_failed",
            "gate_command": " ".join(cmd),
            "gate_returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    return parsed


def run_focused_tests(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - subprocess boundary.
    root_path = Path(root)
    completed = subprocess.run(
        FOCUSED_TEST_COMMAND.split(),
        cwd=root_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "command": FOCUSED_TEST_COMMAND,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _honest_verdict(
    *,
    preconditions_checked: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    tests_added_pass: Mapping[str, Any],
    deepest: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    nav: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("ok") is not True:
        return "blocked_nav_metric_preconditions"
    guard = gate_result.get("baseline_guard")
    if isinstance(guard, Mapping) and guard.get("ok") is not True:
        return "complete: nav_metric_partial_baseline_guard"
    if tests_added_pass.get("passed") is not True:
        return "complete: nav_metric_partial_tests_not_green"
    if not deepest or not efficiency or not nav:
        return "complete: nav_metric_partial_missing_per_game_fields"
    if gate_result.get("pass") is not True:
        return "complete: nav_metric_partial_gate_regression"
    return "shipped: nav_metric_first_class_ci_guarded"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    tests_added_pass: Mapping[str, Any],
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4527: assemble the terminal nav metric harness artifact."""

    current = _current_measurement(gate_result)
    deepest = _deepest_level_by_game(current)
    efficiency = _per_level_efficiency_by_game(current)
    nav = _nav_by_game(current)
    dashboard = gate_result.get("lever_dashboard_row")
    warning = ""
    if isinstance(dashboard, Mapping):
        warning = str(dashboard.get("nav_regression_warning") or "")
    artifact = {
        "experiment": "experiment_4527_nav_metric_harness",
        "schema": "carnot.arc_nav_metric_harness_4527.v1",
        "honest_verdict": _honest_verdict(
            preconditions_checked=preconditions_checked,
            gate_result=gate_result,
            tests_added_pass=tests_added_pass,
            deepest=deepest,
            efficiency=efficiency,
            nav=nav,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "nav_metric_added": {
            "per_game_fields": [
                "deepest_level_reached",
                "per_level_efficiency",
                "reset_replay_steps",
                "forward_walk_hit_rate",
            ],
            "score_fields": ["deepest_level_reached", "per_level_efficiency"],
            "nav_fields": ["reset_replay_steps", "forward_walk_hit_rate"],
            "secondary_warning": warning,
        },
        "tests_added_pass": dict(tests_added_pass),
        "gate_result": dict(gate_result),
        "baseline_guard": dict(gate_result.get("baseline_guard") or {}),
        "canonical_game_set": list(CANONICAL_GAME_SET),
        "canonical_core_games": list(CANONICAL_CORE_GAMES),
        "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
        "score_metric_primary": "per_level_efficiency_and_deepest_level_reached",
        "nav_metric_secondary": "reset_replay_steps_and_forward_walk_hit_rate_warn_only",
        "per_game_deepest_level_reached": deepest,
        "per_game_per_level_efficiency": efficiency,
        "per_game_nav_diagnostics": nav,
        "reproducibility_checksum": "",
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-FCP-4527")
    if artifact.get("canonical_game_set") != list(CANONICAL_GAME_SET):
        errors.append("canonical_game_set must match")
    if artifact.get("canonical_core_games") != list(CANONICAL_CORE_GAMES):
        errors.append("canonical_core_games must match")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if str(verdict).startswith("shipped:") and artifact.get("tests_added_pass", {}).get("passed") is not True:
        errors.append("shipped artifact requires tests_added_pass.passed=true")
    nav_metric = artifact.get("nav_metric_added")
    if not isinstance(nav_metric, Mapping):
        errors.append("nav_metric_added must be a mapping")
    elif nav_metric.get("per_game_fields") != [
        "deepest_level_reached",
        "per_level_efficiency",
        "reset_replay_steps",
        "forward_walk_hit_rate",
    ]:
        errors.append("nav_metric_added must name the four per-game fields")
    for field in (
        "per_game_deepest_level_reached",
        "per_game_per_level_efficiency",
        "per_game_nav_diagnostics",
    ):
        if str(verdict).startswith("shipped:") and not isinstance(artifact.get(field), Mapping):
            errors.append(f"{field} must be a mapping")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    gate_runner: Callable[[Path], dict[str, Any]] = run_gate,
    tests_runner: Callable[[Path], dict[str, Any]] = run_focused_tests,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4527: run the gate hardening checks and write the JSON artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    gate_result = gate_runner(root_path) if preconditions.get("ok") is True else {
        "pass": False,
        "verdict": "blocked_nav_metric_preconditions",
        "baseline_guard": {},
        "current": {},
    }
    tests_result = tests_runner(root_path) if preconditions.get("ok") is True else {
        "command": FOCUSED_TEST_COMMAND,
        "passed": False,
        "blocked": True,
    }
    artifact = build_artifact(
        preconditions_checked=preconditions,
        gate_result=gate_result,
        tests_added_pass=tests_result,
        duration_s=max(0.0, float(now()) - started),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
