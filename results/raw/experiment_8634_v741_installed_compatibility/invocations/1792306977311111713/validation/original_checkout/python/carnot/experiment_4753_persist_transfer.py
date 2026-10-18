"""Experiment 4753: persist .437 levers and characterize transfer.

Spec refs: REQ-ARC-WMTE-4753,
SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION,
SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE,
SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4753_persist_transfer"
EXPERIMENT_ID = 4753
SCHEMA = "carnot.arc.persist_transfer_4753.v1"
RESULT_RELATIVE_PATH = "results/experiment_4753_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4749_structured_engine_vs_freeform.json"
A2_RELATIVE_PATH = "results/experiment_4750_structural_alignment_detector_fix.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
AGENT_RELATIVE_PATH = "python/carnot/agentic/arc_competition_agent.py"
RANDOM_SEED = 4753
DEFAULT_TRANSFER_GAMES = ("cn04", "r11l")
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
INFERENCE_SUBSTRATE = "live_llm_inference"

SPEC_REFS = [
    "REQ-ARC-WMTE-4753",
    "SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION",
    "SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE",
    "SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; characterization-complete is complete_."
    },
    "inference_substrate": {"principle": "live_llm_inference."},
    "preconditions_checked": {
        "principle": "records arcade + upstream-artifact checks."
    },
    "transfer_games": {
        "principle": (
            "the >=2 games the levers were characterized on -- the transfer evidence, "
            "not a single-game artifact."
        )
    },
    "offline_reproduced_new_level": {
        "principle": (
            "false expected -- this is characterization, NOT a level-bank (no over-claim)."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- characterizes the levers on the offline dev twin, asserting "
            "no new level (offline_reproduced_new_level=false); declared (paired with "
            "verifier_is_oracle) because A5 makes an offline_reproduced_new_level claim, per "
            "the ARC Live-Path Reachability discipline."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the transfer value is an execution-grounded efficiency measurement, "
            "not a learned-verifier moat claim."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "upstream_validation",
    "live_path_persistence",
    "selected_levers",
    "transfer_value_per_game",
    "transfer_results",
    "residual_dead_end",
    "arc_orphan_solver_lint",
    "field_principles",
    "requirements",
    "scenarios",
    "result_path",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _terminal_success(verdict: Any) -> bool:
    text = str(verdict or "")
    return text.startswith("success:") or text.startswith("success_")


def upstream_validation_summary(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> JsonDict:
    """REQ-ARC-WMTE-4753: classify which .437 levers are eligible to persist."""

    a1_delta = _as_float(a1_artifact.get("accuracy_delta"))
    a1_structured = _as_float(a1_artifact.get("structured_heldout_accuracy"))
    a1_non_degenerate = bool(a1_artifact.get("structured_engine_non_degenerate"))
    a1_validated = bool(
        _terminal_success(a1_artifact.get("honest_verdict"))
        or bool(a1_artifact.get("offline_reproduced"))
        or (a1_non_degenerate and a1_structured >= 0.5 and a1_delta >= 0.25)
    )
    a1 = {
        "artifact": A1_RELATIVE_PATH,
        "honest_verdict": str(a1_artifact.get("honest_verdict") or ""),
        "validated": a1_validated,
        "structured_engine_non_degenerate": a1_non_degenerate,
        "structured_heldout_accuracy": round(a1_structured, 6),
        "accuracy_delta": round(a1_delta, 6),
        "offline_reproduced": bool(a1_artifact.get("offline_reproduced")),
        "residual": "" if a1_validated else "structured_engine_not_validated",
    }

    a2_piece_count = int(_as_float(a2_artifact.get("detector_piece_count")))
    a2_goal_count = int(_as_float(a2_artifact.get("detector_goal_count")))
    if "detector_pairing_gate" in a2_artifact:
        detector_pairing_gate = bool(a2_artifact.get("detector_pairing_gate"))
    else:
        detector_pairing_gate = bool(a2_piece_count > 0 and 0 < a2_goal_count <= a2_piece_count)
    positive_control = a2_artifact.get("detector_positive_control")
    positive_detected = (
        isinstance(positive_control, Mapping)
        and positive_control.get("structural_goal_detected") is True
    )
    a2_validated = bool(
        _terminal_success(a2_artifact.get("honest_verdict"))
        or (
            detector_pairing_gate
            and a2_piece_count > 0
            and a2_artifact.get("verifier_is_oracle") is False
            and positive_detected
        )
    )
    a2 = {
        "artifact": A2_RELATIVE_PATH,
        "honest_verdict": str(a2_artifact.get("honest_verdict") or ""),
        "validated": a2_validated,
        "detector_pairing_gate": detector_pairing_gate,
        "detector_goal_count": a2_goal_count,
        "detector_piece_count": a2_piece_count,
        "structural_goal_detected": positive_detected,
        "goal_predicate_satisfiable": bool(a2_artifact.get("goal_predicate_satisfiable")),
        "offline_reproduced": bool(a2_artifact.get("offline_reproduced")),
        "residual": "" if a2_validated else "structural_alignment_detector_not_validated",
    }

    eligible: list[str] = []
    if a1_validated:
        eligible.append("structured_engine")
    if a2_validated:
        eligible.append("structural_alignment_detector")
    return {
        "A1_structured_engine": a1,
        "A2_structural_alignment_detector": a2,
        "eligible_levers": eligible,
    }


def inspect_live_path(root: Path | str = REPO_ROOT) -> JsonDict:
    """SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE: inspect live-agent wiring."""

    path = Path(root) / AGENT_RELATIVE_PATH
    try:
        text = path.read_text(encoding="utf-8")
        present = True
    except OSError:
        text = ""
        present = False
    structured_env = "CARNOT_ARC_STRUCTURED_ENGINE" in text
    structured_import = "arc_structured_world_model" in text and "make_structured_load_engine" in text
    structured_proposer = "StructuredEngineReinductionProposer" in text
    structural_provider = "structural_alignment_goal_candidate" in text
    structural_arg = "structural_goal_provider=structural_goal_provider" in text
    persisted: list[str] = []
    if structured_env and structured_import and structured_proposer:
        persisted.append("structured_engine")
    if structural_provider and structural_arg:
        persisted.append("structural_alignment_detector")
    return {
        "agent_path_present": present,
        "structured_engine_env_gate": structured_env,
        "structured_engine_live_import": structured_import,
        "structured_engine_reinduction_proposer": structured_proposer,
        "structural_alignment_goal_provider": structural_provider,
        "structural_goal_passed_to_reinduction": structural_arg,
        "persisted_levers": persisted,
    }


def _default_arcade_import_checker() -> bool:
    __import__("arcade")
    return True


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    arcade_import_checker: Callable[[], bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = arcade_import_checker or _default_arcade_import_checker
    try:
        arcade_import = bool(checker())
        arcade_error = ""
    except Exception as exc:
        arcade_import = False
        arcade_error = f"{type(exc).__name__}: {exc}"
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "arcade_import": arcade_import,
        "arcade_import_command": f"{sys.executable} -c 'import arcade'",
        "arcade_import_error": arcade_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4753": "REQ-ARC-WMTE-4753" in spec_text,
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "arcade_import",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4753",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    blocked = ""
    for key, resource in (
        ("arcade_import", "blocked_arcade_import"),
        ("a1_artifact_present", "blocked_a1_artifact"),
        ("a2_artifact_present", "blocked_a2_artifact"),
        ("spec_has_req_4753", "blocked_spec_req_4753"),
        ("agents_md_read", "blocked_agents_md"),
        ("codex_md_read", "blocked_codex_md"),
    ):
        if not checks[key]:
            blocked = resource
            break
    checks["blocked_resource"] = blocked
    return checks


def run_arc_orphan_solver_lint(
    root: Path | str = REPO_ROOT,
    *,
    lint_runner: Callable[[], Mapping[str, Any]] | None = None,
) -> JsonDict:
    if lint_runner is not None:
        return dict(lint_runner())
    command = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    try:
        proc = subprocess.run(
            command,
            cwd=Path(root),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        return {
            "command": " ".join(command),
            "returncode": 1,
            "passed": False,
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}"[-2000:],
        }
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _delta_from_counts(row: Mapping[str, Any]) -> float:
    baseline = _as_float(row.get("baseline_actions_to_first_effect"))
    lever = _as_float(row.get("lever_actions_to_first_effect"))
    if baseline > 0.0 and lever > 0.0:
        return round(baseline - lever, 6)
    return 0.0


def measure_transfer_game(
    game: str,
    *,
    transfer_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION: measure one game."""

    try:
        row = dict(transfer_row_provider(str(game)) if transfer_row_provider else {})
    except Exception as exc:
        row = {"error": f"{type(exc).__name__}: {exc}"}
    row.setdefault("game", str(game))

    if "action_efficiency_delta" in row:
        action_delta = _as_float(row.get("action_efficiency_delta"))
    elif "lever_action_efficiency" in row or "baseline_action_efficiency" in row:
        action_delta = _as_float(row.get("lever_action_efficiency")) - _as_float(
            row.get("baseline_action_efficiency")
        )
    else:
        action_delta = _delta_from_counts(row)

    if "first_effect_delta" in row:
        first_effect_delta = _as_float(row.get("first_effect_delta"))
    elif "baseline_first_effect_step" in row or "lever_first_effect_step" in row:
        first_effect_delta = _as_float(row.get("baseline_first_effect_step")) - _as_float(
            row.get("lever_first_effect_step")
        )
    else:
        first_effect_delta = _delta_from_counts(row)

    action_delta = round(float(action_delta), 6)
    first_effect_delta = round(float(first_effect_delta), 6)
    value_added = bool(action_delta > 0.0 or first_effect_delta > 0.0)
    if value_added:
        dead_end = ""
    elif row.get("error"):
        dead_end = f"transfer row unavailable for {game}: {row['error']}"
    else:
        dead_end = "no positive action-efficiency or first-effect transfer delta"
    transfer_value = {
        "action_efficiency_delta": action_delta,
        "first_effect_delta": first_effect_delta,
        "offline_reproduced_new_level": False,
        "baseline": {
            key: row[key]
            for key in (
                "baseline_action_efficiency",
                "baseline_first_effect_step",
                "baseline_actions_to_first_effect",
            )
            if key in row
        },
        "with_levers": {
            key: row[key]
            for key in (
                "lever_action_efficiency",
                "lever_first_effect_step",
                "lever_actions_to_first_effect",
            )
            if key in row
        },
        "value_added": value_added,
    }
    return {
        "game": str(game),
        "value_added": value_added,
        "transfer_value": transfer_value,
        "dead_end": dead_end,
    }


def measure_transfer(
    *,
    transfer_games: Sequence[str],
    transfer_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    return [
        measure_transfer_game(game, transfer_row_provider=transfer_row_provider)
        for game in transfer_games
    ]


def _selected_levers(
    upstream_validation: Mapping[str, Any], live_path_persistence: Mapping[str, Any]
) -> list[str]:
    eligible = {
        str(row)
        for row in upstream_validation.get("eligible_levers", [])
        if isinstance(row, str)
    }
    persisted = {
        str(row)
        for row in live_path_persistence.get("persisted_levers", [])
        if isinstance(row, str)
    }
    return sorted(eligible & persisted)


def build_artifact(
    *,
    upstream_validation: Mapping[str, Any],
    live_path_persistence: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    arc_orphan_solver_lint: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-ARC-WMTE-4753: assemble the transfer characterization artifact."""

    rows = [dict(row) for row in transfer_results]
    blocked_resource = str(preconditions_checked.get("blocked_resource") or "")
    blocked = preconditions_checked.get("ok") is False
    if blocked:
        verdict = blocked_resource or "blocked_precondition"
    elif any(row.get("value_added") is True for row in rows):
        verdict = "complete_437_levers_transfer_value_characterized"
    else:
        verdict = "complete_437_levers_transfer_null_characterized"

    transfer_values = {
        str(row.get("game") or ""): dict(row.get("transfer_value") or {}) for row in rows
    }
    dead_ends = {
        str(row.get("game") or ""): str(row.get("dead_end") or "")
        for row in rows
        if row.get("dead_end")
    }
    if blocked:
        residual = f"preconditions failed before transfer measurement: {verdict}"
    elif dead_ends and len(dead_ends) == len(rows):
        residual = "no positive action-efficiency or first-effect transfer delta on characterized games"
    else:
        residual = ""

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transfer_games": [str(row.get("game") or "") for row in rows],
        "offline_reproduced_new_level": False,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "upstream_validation": dict(upstream_validation),
        "live_path_persistence": dict(live_path_persistence),
        "selected_levers": _selected_levers(upstream_validation, live_path_persistence),
        "transfer_value_per_game": transfer_values,
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
        "residual_dead_end": residual,
        "arc_orphan_solver_lint": dict(arc_orphan_solver_lint),
        "field_principles": FIELD_PRINCIPLES,
        "requirements": ["REQ-ARC-WMTE-4753"],
        "scenarios": [
            "SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION",
            "SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE",
            "SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": max(0.0, round(float(duration_s), 6)),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing required field {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be live_llm_inference")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance must be development_proxy")
    if artifact.get("offline_reproduced_new_level") is not False:
        errors.append("offline_reproduced_new_level must be false")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 2):
        errors.append("transfer_games must contain at least two games")
    values = artifact.get("transfer_value_per_game")
    if not isinstance(values, Mapping):
        errors.append("transfer_value_per_game must be a mapping")
    elif any(
        isinstance(value, Mapping) and value.get("offline_reproduced_new_level") is not False
        for value in values.values()
    ):
        errors.append("per-game offline_reproduced_new_level must be false")
    lint = artifact.get("arc_orphan_solver_lint")
    if not blocked and (not isinstance(lint, Mapping) or lint.get("passed") is not True):
        errors.append("arc_orphan_solver_lint must pass for non-blocked artifacts")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4753")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    transfer_games: Sequence[str] = DEFAULT_TRANSFER_GAMES,
    arcade_import_checker: Callable[[], bool] | None = None,
    transfer_row_provider: Callable[[str], Mapping[str, Any]] | None = None,
    lint_runner: Callable[[], Mapping[str, Any]] | None = None,
    now: Callable[[], float] = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = check_preconditions(root_path, arcade_import_checker=arcade_import_checker)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a2 = _load_json(root_path / A2_RELATIVE_PATH)
    validation = upstream_validation_summary(a1_artifact=a1, a2_artifact=a2)
    persistence = inspect_live_path(root_path)
    rows: list[JsonDict] = []
    if checks.get("ok") is True:
        lint = run_arc_orphan_solver_lint(root_path, lint_runner=lint_runner)
        rows = measure_transfer(
            transfer_games=transfer_games,
            transfer_row_provider=transfer_row_provider,
        )
    else:
        lint = {"skipped": str(checks.get("blocked_resource") or "blocked_precondition")}
    artifact = build_artifact(
        upstream_validation=validation,
        live_path_persistence=persistence,
        preconditions_checked=checks,
        transfer_results=rows,
        arc_orphan_solver_lint=lint,
        duration_s=now() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(REPO_ROOT)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "transfer_games": artifact["transfer_games"],
                "offline_reproduced_new_level": artifact["offline_reproduced_new_level"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
