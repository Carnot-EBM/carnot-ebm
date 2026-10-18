"""Experiment 4518: canonical ARC local submission metric harness.

Spec refs: REQ-ARC-FCP-4518, SCENARIO-ARC-FCP-4518.
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
RESULT_RELATIVE_PATH = "results/experiment_4518_metric_harness_canonical.json"
GATE_RELATIVE_PATH = "scripts/kaggle/arc_local_submission_gate.py"
BASELINE_RELATIVE_PATH = "ops/arc-submission-baseline.json"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_arc_submission_gate_verdict.py "
    "tests/python/test_experiment_4518_metric_harness_canonical.py -q --no-cov"
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
)
REQUIREMENTS = ("REQ-ARC-FCP-4518",)
SCENARIOS = ("SCENARIO-ARC-FCP-4518",)
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


def _load_gate_module() -> Any:
    gate_path = REPO_ROOT / GATE_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("arc_local_submission_gate_4518", gate_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib failure boundary.
        raise RuntimeError(f"cannot load {gate_path}")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


GATE = _load_gate_module()
CANONICAL_GAME_SET = tuple(GATE.CANONICAL_GAME_SET)
CANONICAL_CORE_GAMES = tuple(GATE.CANONICAL_CORE_GAMES)
CANONICAL_BASELINE_MEDIAN_ACTIONS = float(GATE.CANONICAL_BASELINE_MEDIAN_ACTIONS)
CANONICAL_ACTION_FIELD = str(GATE.CANONICAL_ACTION_FIELD)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        'principle "terminal prefix; e.g. shipped: metric_harness_canonical_ci_guarded '
        'OR complete: metric_harness_partial_<reason>."'
    ),
    "inference_substrate": (
        'principle "verifier_ensemble_against_cached_candidates -- runs the offline gate, '
        'no LLM load (1s floor)."'
    ),
    "canonical_game_set": (
        'principle "the fixed 8 games -- pins the metric so no A-task can cherry-pick an easier subset."'
    ),
    "canonical_baseline": (
        'principle "7760, guarded against silent movement (raising-a-cap-to-hide-drift is forbidden)."'
    ),
    "positive_control_passed": (
        'principle "proves the harness can detect a real reduction (guards a silently-broken metric)."'
    ),
    "tests_added_pass": 'principle "Tests Must Run and Assert."',
    "preconditions_checked": (
        'principle "records resources verified; pre-empts missing-resource fabrication."'
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "requirements",
    "scenarios",
    "field_principles",
    "canonical_default_budget",
    "headroom_budget_measurement",
    "sample_dashboard_row",
    "reproducibility_checksum",
    "leaderboard_submission",
    "result_path",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - local resource boundary.
    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_precondition": False,
        "gate_help_precondition": False,
        "baseline_file_present": (root_path / BASELINE_RELATIVE_PATH).exists(),
    }
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        checks["offline_arcade_precondition"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)

    cmd = [str(root_path / ".venv" / "bin" / "python"), str(root_path / GATE_RELATIVE_PATH), "--help"]
    try:
        completed = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=30)
        checks["gate_help_precondition"] = completed.returncode == 0
        if completed.returncode != 0:
            checks["gate_help_error"] = completed.stderr[-500:]
    except Exception as exc:
        checks["gate_help_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_precondition"] and checks["gate_help_precondition"])
    return checks


def load_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    return json.loads((Path(root) / BASELINE_RELATIVE_PATH).read_text(encoding="utf-8"))


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


def measure_headroom(gate: Any = GATE, *, policy: str = "e3", cap: int = 115) -> dict[str, Any]:  # pragma: no cover - offline gate boundary.
    try:
        return gate.measure_headroom_budget(policy, cap)
    except Exception as exc:
        return {
            "selected_default_budget": int(gate.DEFAULT_BUDGET),
            "measured": False,
            "rows": [],
            "error": repr(exc),
        }


def _sample_treatment() -> dict[str, Any]:
    actions = dict(GATE.CANONICAL_BASELINE_ACTIONS_BY_GAME)
    actions["ft09"] = 7600
    return {
        "policy": "sample",
        "games": list(CANONICAL_GAME_SET),
        "action_metric": {
            "field": CANONICAL_ACTION_FIELD,
            "definition": "total_actions_on_solved_games",
        },
        "solved_count": len(actions),
        "solved_games": sorted(actions),
        "actions_by_game": actions,
        "median_actions_on_solved": 7724,
    }


def _honest_verdict(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline_guard: Mapping[str, Any],
    headroom: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    tests_added_pass: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("offline_arcade_precondition") is not True:
        return "blocked_offline_arcade_precondition"
    if preconditions_checked.get("gate_help_precondition") is not True:
        return "blocked_gate_help_precondition"
    if baseline_guard.get("ok") is not True:
        return "complete: metric_harness_partial_baseline_guard"
    if headroom.get("measured") is not True:
        return "complete: metric_harness_partial_headroom_budget"
    if positive_control.get("passed") is not True:
        return "complete: metric_harness_partial_positive_control"
    if tests_added_pass.get("passed") is not True:
        return "complete: metric_harness_partial_tests_not_green"
    return "shipped: metric_harness_canonical_ci_guarded"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_guard: Mapping[str, Any],
    headroom: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    sample_dashboard_row: Mapping[str, Any],
    tests_added_pass: Mapping[str, Any],
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4518: assemble the canonical metric harness artifact."""

    canonical_baseline = {
        "median_actions_on_core": CANONICAL_BASELINE_MEDIAN_ACTIONS,
        "action_metric_field": CANONICAL_ACTION_FIELD,
        "core_games": list(CANONICAL_CORE_GAMES),
        "source": BASELINE_RELATIVE_PATH,
        "guard": dict(baseline_guard),
    }
    checksum_payload = {
        "baseline": baseline,
        "baseline_guard": baseline_guard,
        "headroom": headroom,
        "positive_control": positive_control,
        "sample_dashboard_row": sample_dashboard_row,
        "tests_added_pass": tests_added_pass,
    }
    return {
        "experiment": "experiment_4518_metric_harness_canonical",
        "schema": "carnot.arc_metric_harness_canonical_4518.v1",
        "honest_verdict": _honest_verdict(
            preconditions_checked=preconditions_checked,
            baseline_guard=baseline_guard,
            headroom=headroom,
            positive_control=positive_control,
            tests_added_pass=tests_added_pass,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "canonical_game_set": list(CANONICAL_GAME_SET),
        "canonical_baseline": canonical_baseline,
        "canonical_default_budget": int(headroom.get("selected_default_budget") or GATE.DEFAULT_BUDGET),
        "headroom_budget_measurement": dict(headroom),
        "positive_control_passed": bool(positive_control.get("passed")),
        "positive_control": dict(positive_control),
        "tests_added_pass": dict(tests_added_pass),
        "sample_dashboard_row": dict(sample_dashboard_row),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
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
        errors.append("field_principles must match required principles")
    if artifact.get("canonical_game_set") != list(CANONICAL_GAME_SET):
        errors.append("canonical_game_set must be the fixed 8 games")
    baseline = artifact.get("canonical_baseline")
    if not isinstance(baseline, Mapping):
        errors.append("canonical_baseline must be a mapping")
    elif float(baseline.get("median_actions_on_core") or 0.0) != CANONICAL_BASELINE_MEDIAN_ACTIONS:
        errors.append("canonical_baseline must pin 7760")
    if artifact.get("positive_control_passed") is not True and str(verdict).startswith("shipped:"):
        errors.append("shipped artifact requires positive_control_passed=true")
    tests = artifact.get("tests_added_pass")
    if not isinstance(tests, Mapping):
        errors.append("tests_added_pass must be a mapping")
    elif tests.get("passed") is not True and str(verdict).startswith("shipped:"):
        errors.append("shipped artifact requires tests_added_pass.passed=true")
    row = artifact.get("sample_dashboard_row")
    if not isinstance(row, Mapping):
        errors.append("sample_dashboard_row must be a mapping")
    else:
        for field in ("median_actions_on_core", "core_solves_preserved", "bonus_solves"):
            if field not in row:
                errors.append(f"sample_dashboard_row missing {field}")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    gate: Any = GATE,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    measure_headroom: Callable[..., dict[str, Any]] = measure_headroom,
    tests_added_pass: Mapping[str, Any] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4518: run the canonical harness checks and write the JSON artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    baseline = load_baseline(root_path)
    baseline_guard = gate.validate_canonical_baseline(baseline)
    headroom = measure_headroom(gate)
    positive = gate.positive_control(baseline)
    sample_row = gate.dashboard_row(_sample_treatment(), baseline, lever="sample_dashboard")
    tests_result = dict(tests_added_pass) if tests_added_pass is not None else run_focused_tests(root_path)
    artifact = build_artifact(
        preconditions_checked=preconditions,
        baseline=baseline,
        baseline_guard=baseline_guard,
        headroom=headroom,
        positive_control=positive,
        sample_dashboard_row=sample_row,
        tests_added_pass=tests_result,
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


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    main()
