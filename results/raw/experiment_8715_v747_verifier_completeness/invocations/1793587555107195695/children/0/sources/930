"""Experiment 4528: .417 B-track infra carryforward audit.

Spec refs: REQ-ARC-FCP-4528, SCENARIO-ARC-FCP-4528.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4528_infra_carryforward.json"
UPSTREAM_RELATIVE_PATH = "results/experiment_4518_metric_harness_canonical.json"
GATE_RELATIVE_PATH = "scripts/kaggle/arc_local_submission_gate.py"
FIXTURE_TEST_RELATIVE_PATH = "tests/python/test_arc_submission_gate_verdict.py"
FOCUSED_TEST_COMMAND = ".venv/bin/pytest tests/python/test_experiment_4528_infra_carryforward.py -q --no-cov"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts (reads the .417 B2 artifact)"
REQUIREMENTS = ("REQ-ARC-FCP-4528",)
SCENARIOS = ("SCENARIO-ARC-FCP-4528",)
CANONICAL_GAME_SET = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CANONICAL_CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
CANONICAL_ACTION_FIELD = "actions"
DEFAULT_BUDGET = 8000
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
CORE_FIXTURE_TESTS = (
    "test_a1_frame_change_prune_fails_lost_core_m0r0",
    "test_a2_imitation_prior_fails_core_traded_for_fringe",
    "test_positive_core_faster_passes_improved",
    "test_neutral_core_same_passes_non_inferior",
    "test_bonus_solve_reported_but_core_required",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "principle \"terminal prefix; shipped: infra_carryforward_complete OR "
        "complete: infra_audit_already_done.\""
    ),
    "inference_substrate": (
        "principle \"aggregation_from_upstream_artifacts (reads the .417 B2 artifact) "
        "unless it measures B* live (then verifier_ensemble_against_cached_candidates).\""
    ),
    "b_track_status": (
        "principle \"what landed in .417 B2 vs what this task completed -- the audit trail.\""
    ),
    "cited_upstream_artifacts": (
        "principle \"traceability of the .417 B2 numbers this reconciles.\""
    ),
    "preconditions_checked": (
        "principle \"records resources verified; pre-empts missing-resource fabrication.\""
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "requirements",
    "scenarios",
    "field_principles",
    "tests_added_pass",
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


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def load_upstream(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    return json.loads((Path(root) / UPSTREAM_RELATIVE_PATH).read_text(encoding="utf-8"))


def read_fixture_test_source(root: Path | str = REPO_ROOT) -> str:
    return (Path(root) / FIXTURE_TEST_RELATIVE_PATH).read_text(encoding="utf-8")


def _fixture_guard(upstream: Mapping[str, Any], fixture_test_source: str) -> dict[str, Any]:
    tests = _as_mapping(upstream.get("tests_added_pass"))
    command = str(tests.get("command", ""))
    present = [name for name in CORE_FIXTURE_TESTS if f"def {name}" in fixture_test_source]
    return {
        "landed_in_4518": bool(tests.get("passed") is True),
        "ci_guarded": bool(
            tests.get("passed") is True
            and FIXTURE_TEST_RELATIVE_PATH in command
            and len(present) >= 4
        ),
        "fixture_count": len(present),
        "fixture_tests_present": present,
        "ci_command": command,
    }


def _canonical_gate(upstream: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _as_mapping(upstream.get("canonical_baseline"))
    guard = _as_mapping(baseline.get("guard"))
    game_set = list(upstream.get("canonical_game_set") or [])
    core_games = list(baseline.get("core_games") or [])
    action_field = str(baseline.get("action_metric_field", ""))
    canonical = bool(
        str(upstream.get("honest_verdict", "")).startswith("shipped:")
        and guard.get("ok") is True
        and game_set == list(CANONICAL_GAME_SET)
        and core_games == list(CANONICAL_CORE_GAMES)
        and action_field == CANONICAL_ACTION_FIELD
    )
    return {
        "landed_in_4518": canonical,
        "core_containment_gate_canonical": canonical,
        "canonical_game_set": game_set,
        "core_games": core_games,
        "action_metric_field": action_field,
        "baseline_guard_ok": guard.get("ok") is True,
        "upstream_honest_verdict": str(upstream.get("honest_verdict", "")),
    }


def _headroom_status(upstream: Mapping[str, Any]) -> dict[str, Any]:
    headroom = _as_mapping(upstream.get("headroom_budget_measurement"))
    rows = _as_rows(headroom.get("rows"))
    stable_rows = [row for row in rows if row.get("stable_vs_1_5x") is True]
    selected_b_star = _int_or_none(stable_rows[0].get("budget")) if stable_rows else None
    upstream_default = _int_or_none(upstream.get("canonical_default_budget"))
    if upstream_default is None:
        upstream_default = _int_or_none(headroom.get("selected_default_budget"))
    if upstream_default is None:
        upstream_default = DEFAULT_BUDGET
    if selected_b_star is None:
        no_blind_budget_raise = upstream_default == DEFAULT_BUDGET
    else:
        no_blind_budget_raise = upstream_default == selected_b_star
    return {
        "landed_in_4518": headroom.get("measured") is True,
        "measured": headroom.get("measured") is True,
        "candidate_rows": [dict(row) for row in rows],
        "stable_rows": [dict(row) for row in stable_rows],
        "b_star_measured": selected_b_star is not None,
        "selected_b_star": selected_b_star,
        "upstream_default_budget": upstream_default,
        "no_stable_candidate": selected_b_star is None and bool(rows),
        "no_blind_budget_raise": no_blind_budget_raise,
    }


def audit_upstream(upstream: Mapping[str, Any], *, fixture_test_source: str) -> dict[str, Any]:
    """REQ-ARC-FCP-4528: reconcile the .417 B2 artifact into a B-track audit."""

    canonical_gate = _canonical_gate(upstream)
    fixture_guard = _fixture_guard(upstream, fixture_test_source)
    headroom = _headroom_status(upstream)
    gaps: list[str] = []
    if not canonical_gate["core_containment_gate_canonical"]:
        gaps.append("missing_canonical_gate_evidence")
    if not fixture_guard["ci_guarded"]:
        gaps.append("missing_fixture_ci_guard")
    if not headroom["measured"]:
        gaps.append("missing_headroom_measurement")
    return {
        "upstream_417_b2_landed": {
            "canonical_gate": canonical_gate,
            "fixture_ci_guard": fixture_guard,
            "headroom_budget": headroom,
        },
        "this_task_completed": {
            "audit_only": True,
            "completed_missing_piece": "none",
            "no_blind_budget_raise": headroom["no_blind_budget_raise"],
            "result_path": RESULT_RELATIVE_PATH,
        },
        "gaps": gaps,
    }


def _cited_upstream_artifacts(upstream: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": UPSTREAM_RELATIVE_PATH,
            "sha256": _stable_checksum(upstream),
            "fields_imported": [
                "honest_verdict",
                "canonical_game_set",
                "canonical_baseline",
                "tests_added_pass",
                "headroom_budget_measurement",
                "canonical_default_budget",
            ],
        }
    ]


def _honest_verdict(
    *,
    preconditions_checked: Mapping[str, Any],
    b_track_status: Mapping[str, Any],
    tests_added_pass: Mapping[str, Any],
) -> str:
    upstream = _as_mapping(b_track_status.get("upstream_417_b2_landed"))
    canonical_gate = _as_mapping(upstream.get("canonical_gate"))
    fixture_guard = _as_mapping(upstream.get("fixture_ci_guard"))
    headroom = _as_mapping(upstream.get("headroom_budget"))
    if preconditions_checked.get("ok") is not True:
        return "blocked_infra_carryforward_preconditions"
    if canonical_gate.get("core_containment_gate_canonical") is not True:
        return "complete: infra_carryforward_missing_canonical_gate_evidence"
    if fixture_guard.get("ci_guarded") is not True:
        return "complete: infra_carryforward_missing_fixture_ci_guard"
    if headroom.get("measured") is not True:
        return "complete: infra_carryforward_missing_headroom_measurement"
    if tests_added_pass.get("passed") is not True:
        return "complete: infra_carryforward_tests_not_green"
    return "complete: infra_audit_already_done"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    upstream_artifact: Mapping[str, Any],
    fixture_test_source: str,
    tests_added_pass: Mapping[str, Any],
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4528: assemble the carryforward audit artifact."""

    b_track_status = audit_upstream(upstream_artifact, fixture_test_source=fixture_test_source)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4528_infra_carryforward",
        "schema": "carnot.arc_infra_carryforward_4528.v1",
        "honest_verdict": _honest_verdict(
            preconditions_checked=preconditions_checked,
            b_track_status=b_track_status,
            tests_added_pass=tests_added_pass,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "b_track_status": b_track_status,
        "cited_upstream_artifacts": _cited_upstream_artifacts(upstream_artifact),
        "tests_added_pass": dict(tests_added_pass),
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
        errors.append("inference_substrate must match the required substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required principles")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    b_status = artifact.get("b_track_status")
    if not isinstance(b_status, Mapping):
        errors.append("b_track_status must be a mapping")
    else:
        task = _as_mapping(b_status.get("this_task_completed"))
        if task.get("no_blind_budget_raise") is not True and str(verdict).startswith(("complete:", "shipped:")):
            errors.append("no_blind_budget_raise must be true for terminal audits")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        errors.append("cited_upstream_artifacts must be a list")
    tests = artifact.get("tests_added_pass")
    if not isinstance(tests, Mapping):
        errors.append("tests_added_pass must be a mapping")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match payload")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path must point to the 4528 artifact")
    return errors


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - local resource boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "gate_help_precondition": False,
        "upstream_artifact_present": (root_path / UPSTREAM_RELATIVE_PATH).exists(),
        "fixture_test_file_present": (root_path / FIXTURE_TEST_RELATIVE_PATH).exists(),
        "spec_has_req_4528": spec_path.exists() and "REQ-ARC-FCP-4528" in spec_path.read_text(encoding="utf-8"),
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
        and checks["upstream_artifact_present"]
        and checks["fixture_test_file_present"]
        and checks["spec_has_req_4528"]
    )
    return checks


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


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    fixture_test_source: str | None = None,
    tests_added_pass: Mapping[str, Any] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4528: write the audit artifact from upstream B2 evidence."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    upstream = load_upstream(root_path)
    source = fixture_test_source if fixture_test_source is not None else read_fixture_test_source(root_path)
    tests_result = dict(tests_added_pass) if tests_added_pass is not None else run_focused_tests(root_path)
    artifact = build_artifact(
        preconditions_checked=preconditions,
        upstream_artifact=upstream,
        fixture_test_source=source,
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
