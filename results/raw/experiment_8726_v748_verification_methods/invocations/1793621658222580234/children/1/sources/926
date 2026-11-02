"""Experiment 4526: submitted deeper-level integration gate.

Spec refs: REQ-ARC-WMTE-4526, SCENARIO-ARC-WMTE-4526.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4526_integration_8game_gate.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the "
    "per-level gate (no headline GGUF load unless the induction tier is invoked)."
)
CORE_EFFICIENCY_BASELINE = 2.0074
BASELINE_MEDIAN_ACTIONS = 7760.0
RANDOM_SEED = 4526
DEFAULT_GATE_BUDGET = 8000
DEFAULT_GATE_CAP_SECONDS = 115
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
REQUIREMENTS = ("REQ-ARC-WMTE-4526",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4526",)
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
UPSTREAM_ARTIFACTS = {
    "a1_forward_walk": "results/experiment_4523_forward_walk_navigation.json",
    "a2_reach_deeper_levels": "results/experiment_4524_reach_deeper_levels.json",
    "a2_stop_after_levelup": "results/experiment_4524_stop_after_levelup.json",
}
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR "
        "complete: no_lever_raises_core_efficiency_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the "
        "per-level gate (no headline GGUF load unless the induction tier is invoked)."
    ),
    "core_efficiency_baseline": (
        "2.0074 -- the REAL per-level metric control (NOT median actions, retired as a score lever)."
    ),
    "core_efficiency_integrated": (
        "the HEADLINE -- the SUBMITTED-config per-level efficiency after wiring the winners "
        "(did it solve MORE levels)."
    ),
    "core_solves_preserved": "integration must preserve every CORE solve (set-containment).",
    "levers_integrated": (
        "names which of A1/A2 were wired -- traceable to their measured deltas; [] is an honest null."
    ),
    "additivity_checked": (
        "integrated CORE median vs the naive sum of isolated A1+A2 deltas -- surfaces a destructive "
        "batch-expand x stop-after-levelup interaction instead of burying it."
    ),
    "heldout_solve_rate": "the real transfer signal; integration should not regress it.",
    "nav_diagnostics": (
        "reset_replay_steps end-to-end -- did the integrated config actually cut the replay tax."
    ),
    "ready_for_operator_submit": (
        "True if the integrated config is a CORE-preserved improvement worth a 1/day submission slot; "
        "the task NEVER submits (operator-only)."
    ),
    "false_negative_risk_checked": (
        "an honest null only valid with the 7760 baseline measured the same way."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "gate_games",
    "core_games",
    "submitted_agent_config",
    "upstream_decision",
    "gate_result",
    "per_game_deepest_level_reached",
    "local_gate_budget",
    "operator_submission_performed",
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


def _kit() -> Any:  # pragma: no cover - ARC SDK boundary.
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    root_path = Path(root)
    return {
        name: _read_json(root_path / relative_path)
        for name, relative_path in UPSTREAM_ARTIFACTS.items()
    }


def load_gate_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    baseline = Path(root) / "ops" / "arc-submission-baseline.json"
    if baseline.exists():
        return json.loads(baseline.read_text(encoding="utf-8"))
    return {
        "games": list(GATE_GAMES),
        "solved_games": list(CORE_GAMES),
        "core_efficiency": CORE_EFFICIENCY_BASELINE,
        "median_actions_on_solved": BASELINE_MEDIAN_ACTIONS,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "a1_artifact_present": (root_path / UPSTREAM_ARTIFACTS["a1_forward_walk"]).exists(),
        "a2_reach_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a2_reach_deeper_levels"]
        ).exists(),
        "spec_has_req_4526": spec_path.exists()
        and "REQ-ARC-WMTE-4526" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def _levels_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(game): int(level or 0) for game, level in value.items()}


def _core_regressions(control: Mapping[str, int], treatment: Mapping[str, int]) -> list[str]:
    return [
        game
        for game in CORE_GAMES
        if int(treatment.get(game, 0)) < int(control.get(game, 0))
    ]


def _a2_control_levels(a2_reach_artifact: Mapping[str, Any]) -> dict[str, int]:
    deepest = a2_reach_artifact.get("deepest_level_reached_per_core_game")
    if isinstance(deepest, Mapping):
        for key in ("control_max_depth_45", "control", "submitted_control"):
            levels = _levels_from_mapping(deepest.get(key))
            if levels:
                return levels
    for row in a2_reach_artifact.get("levers_tried", []) or []:
        if isinstance(row, Mapping) and str(row.get("lever", "")).startswith("control"):
            levels = _levels_from_mapping(row.get("deepest_level_by_game"))
            if levels:
                return levels
    return {game: 0 for game in CORE_GAMES}


def _nav_diagnostics_from_a1(a1_artifact: Mapping[str, Any]) -> dict[str, Any]:
    before_after = a1_artifact.get("nav_diagnostics_before_after")
    before: Mapping[str, Any] = {}
    after: Mapping[str, Any] = {}
    if isinstance(before_after, Mapping):
        before_value = before_after.get("before")
        after_value = before_after.get("after")
        before = before_value if isinstance(before_value, Mapping) else {}
        after = after_value if isinstance(after_value, Mapping) else {}
    reset_before = int(before.get("reset_replay_steps") or 0)
    reset_after = int(after.get("reset_replay_steps") or 0)
    return {
        "source": "results/experiment_4523_forward_walk_navigation.json",
        "integrated_config": str(a1_artifact.get("chosen_submitted_config") or "unchanged"),
        "reset_replay_steps_integrated": reset_before,
        "reset_replay_steps_candidate_after": reset_after,
        "reset_replay_steps_delta_candidate": reset_after - reset_before,
        "forward_walk_hit_rate_integrated": float(before.get("forward_walk_hit_rate") or 0.0),
        "forward_walk_hit_rate_candidate_after": float(after.get("forward_walk_hit_rate") or 0.0),
    }


def _a1_summary(a1_artifact: Mapping[str, Any]) -> dict[str, Any]:
    control = a1_artifact.get("median_actions_on_core_control")
    best = a1_artifact.get("median_actions_on_core_best")
    delta = None
    if control is not None and best is not None:
        delta = float(best) - float(control)
    return {
        "name": "A1_forward_walk_navigation",
        "honest_verdict": a1_artifact.get("honest_verdict"),
        "flagged_adversarial": a1_artifact.get("flagged_adversarial") is True,
        "chosen_submitted_config": a1_artifact.get("chosen_submitted_config"),
        "median_actions_delta": delta,
        "reason": "retired_metric_no_core_efficiency_gain",
    }


def select_integrated_levers(
    *,
    a1_artifact: Mapping[str, Any],
    a2_reach_artifact: Mapping[str, Any],
    stop_after_levelup_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4526: select only A2 levers that improve per-level CORE depth."""

    accepted: list[str] = []
    rejected: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {"A1_forward_walk_navigation": _a1_summary(a1_artifact)}
    if summaries["A1_forward_walk_navigation"]["flagged_adversarial"]:
        rejected["A1_forward_walk_navigation"] = {
            **summaries["A1_forward_walk_navigation"],
            "reason": "flagged_adversarial",
        }
    else:
        rejected["A1_forward_walk_navigation"] = summaries["A1_forward_walk_navigation"]

    control_levels = _a2_control_levels(a2_reach_artifact)
    best_a2_delta = round(
        float(a2_reach_artifact.get("core_efficiency_best") or CORE_EFFICIENCY_BASELINE)
        - CORE_EFFICIENCY_BASELINE,
        4,
    )
    offline_reproduced = a2_reach_artifact.get("offline_reproduced") is True
    for row in a2_reach_artifact.get("levers_tried", []) or []:
        if not isinstance(row, Mapping):
            continue
        lever = str(row.get("lever") or "unknown")
        if lever.startswith("control"):
            continue
        key = f"A2_reach_deeper_levels:{lever}"
        levels = _levels_from_mapping(row.get("deepest_level_by_game"))
        regressions = _core_regressions(control_levels, levels)
        efficiency = round(float(row.get("core_efficiency") or 0.0), 4)
        deeper = any(
            int(levels.get(game, 0)) >= 2
            and int(levels.get(game, 0)) > int(control_levels.get(game, 0))
            for game in CORE_GAMES
        )
        summary = {
            "lever": lever,
            "core_efficiency": efficiency,
            "delta_vs_baseline": round(efficiency - CORE_EFFICIENCY_BASELINE, 4),
            "deepest_level_by_game": {game: int(levels.get(game, 0)) for game in CORE_GAMES},
            "lost_core_level_games": regressions,
            "offline_reproduced": offline_reproduced,
            "flagged_adversarial": (
                a2_reach_artifact.get("flagged_adversarial") is True
                or row.get("flagged_adversarial") is True
            ),
        }
        summaries[key] = summary
        if summary["flagged_adversarial"]:
            rejected[key] = {**summary, "reason": "flagged_adversarial"}
        elif regressions:
            rejected[key] = {**summary, "reason": "core_level_regression"}
        elif efficiency <= CORE_EFFICIENCY_BASELINE:
            rejected[key] = {**summary, "reason": "no_core_efficiency_gain"}
        elif not deeper:
            rejected[key] = {**summary, "reason": "no_deeper_core_level"}
        elif not offline_reproduced:
            rejected[key] = {**summary, "reason": "offline_reproduction_missing"}
        else:
            accepted.append(key)

    if stop_after_levelup_artifact is not None:
        rejected["A2_stop_after_levelup"] = {
            "honest_verdict": stop_after_levelup_artifact.get("honest_verdict"),
            "flagged_adversarial": stop_after_levelup_artifact.get("flagged_adversarial") is True,
            "reason": "action_trimming_retired",
        }

    return {
        "accepted_levers": accepted,
        "rejected_levers": rejected,
        "upstream_summaries": summaries,
        "a2_control_levels": {game: int(control_levels.get(game, 0)) for game in CORE_GAMES},
        "a2_best_delta_core_efficiency": best_a2_delta,
        "nav_diagnostics": _nav_diagnostics_from_a1(a1_artifact),
    }


def _parse_gate_stdout(stdout: str) -> dict[str, Any]:  # pragma: no cover - subprocess boundary.
    start = stdout.find("{")
    if start < 0:
        return {"parse_error": "json_object_not_found", "stdout": stdout}
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        return {"parse_error": repr(exc), "stdout": stdout}


def run_local_submission_gate(
    *,
    root: Path | str = REPO_ROOT,
    budget: int = DEFAULT_GATE_BUDGET,
    cap: int = DEFAULT_GATE_CAP_SECONDS,
) -> dict[str, Any]:  # pragma: no cover - slow end-to-end boundary.
    root_path = Path(root)
    cmd = [
        str(root_path / ".venv" / "bin" / "python"),
        str(root_path / "scripts" / "kaggle" / "arc_local_submission_gate.py"),
        "--check",
        "--policy",
        "e3",
        "--budget",
        str(int(budget)),
        "--cap",
        str(int(cap)),
        "--lever",
        "experiment_4526_integration",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=cap + 60)
    parsed = _parse_gate_stdout(proc.stdout)
    parsed["gate_command"] = cmd
    parsed["gate_returncode"] = proc.returncode
    if proc.stderr:
        parsed["gate_stderr"] = proc.stderr
    return parsed


def _current_measurement(gate_result: Mapping[str, Any]) -> Mapping[str, Any]:
    current = gate_result.get("current")
    return current if isinstance(current, Mapping) else gate_result


def _efficiency_by_game(measurement: Mapping[str, Any]) -> dict[str, float]:
    explicit = measurement.get("efficiency_by_game")
    if isinstance(explicit, Mapping):
        return {str(game): float(value or 0.0) for game, value in explicit.items()}
    out: dict[str, float] = {}
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, Mapping) and row.get("game") is not None and row.get("efficiency") is not None:
            out[str(row["game"])] = float(row["efficiency"] or 0.0)
    return out


def _core_efficiency(measurement: Mapping[str, Any]) -> float:
    if measurement.get("core_efficiency") is not None:
        return round(float(measurement["core_efficiency"]), 4)
    eff = _efficiency_by_game(measurement)
    return round(sum(eff.get(game, 0.0) for game in CORE_GAMES), 4)


def _solved_games(measurement: Mapping[str, Any]) -> set[str]:
    solved = measurement.get("solved_games")
    if isinstance(solved, Sequence) and not isinstance(solved, (str, bytes)):
        return {str(game) for game in solved}
    return {
        str(row["game"])
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping) and row.get("solved") is True
    }


def _baseline_core(baseline: Mapping[str, Any]) -> set[str]:
    solved = baseline.get("solved_games")
    if isinstance(solved, Sequence) and not isinstance(solved, (str, bytes)) and solved:
        return {str(game) for game in solved}
    return set(CORE_GAMES)


def _per_game_deepest_level(measurement: Mapping[str, Any]) -> dict[str, int]:
    rows = measurement.get("per_game", []) or []
    out = {game: 0 for game in GATE_GAMES}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("game") is None:
            continue
        game = str(row["game"])
        level = row.get("best_level", row.get("levels"))
        if level is None:
            level = 1 if row.get("solved") is True else 0
        out[game] = int(level or 0)
    return out


def _heldout_solve_rate(measurement: Mapping[str, Any]) -> float:
    heldout = [game for game in GATE_GAMES if game not in CORE_GAMES]
    solved = _solved_games(measurement)
    return round(sum(1 for game in heldout if game in solved) / float(len(heldout)), 10)


def _false_negative_risk_checked(
    *,
    baseline: Mapping[str, Any],
    gate_result: Mapping[str, Any],
) -> bool:
    guard = gate_result.get("baseline_guard")
    guard_ok = not isinstance(guard, Mapping) or guard.get("ok") is True
    baseline_efficiency = round(float(baseline.get("core_efficiency") or 0.0), 4)
    return bool(guard_ok and baseline_efficiency == CORE_EFFICIENCY_BASELINE)


def _additivity_checked(
    *,
    upstream_decision: Mapping[str, Any],
    integrated_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    a1_delta = 0.0
    a2_delta = float(upstream_decision.get("a2_best_delta_core_efficiency") or 0.0)
    integrated_delta = round(_core_efficiency(integrated_measurement) - CORE_EFFICIENCY_BASELINE, 4)
    return {
        "metric": "core_efficiency",
        "a1_delta_core_efficiency": a1_delta,
        "a2_best_delta_core_efficiency": round(a2_delta, 4),
        "naive_sum_delta": round(a1_delta + a2_delta, 4),
        "integrated_delta": integrated_delta,
        "interaction_delta": round(integrated_delta - (a1_delta + a2_delta), 4),
        "action_trimming_retired": True,
    }


def _honest_verdict(
    *,
    core_efficiency_integrated: float,
    core_solves_preserved: bool,
    levers_integrated: Sequence[str],
) -> str:
    if (
        core_efficiency_integrated > CORE_EFFICIENCY_BASELINE
        and core_solves_preserved
        and bool(levers_integrated)
    ):
        return (
            f"success: integrated_core_efficiency_{core_efficiency_integrated:.4f}_"
            f"above_{CORE_EFFICIENCY_BASELINE:.4f}"
        )
    return "complete: no_lever_raises_core_efficiency_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    upstream_decision: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-4526: assemble the terminal integration artifact."""

    current = _current_measurement(gate_result)
    core_efficiency_integrated = _core_efficiency(current)
    solved = _solved_games(current)
    baseline_core = _baseline_core(baseline)
    core_solves_preserved = baseline_core.issubset(solved)
    levers_integrated = list(upstream_decision.get("accepted_levers") or [])
    ready = bool(
        levers_integrated
        and core_efficiency_integrated > CORE_EFFICIENCY_BASELINE
        and core_solves_preserved
    )
    artifact = {
        "experiment": "experiment_4526_integration_8game_gate",
        "schema": "carnot.arc_integration_8game_gate_4526.v1",
        "honest_verdict": _honest_verdict(
            core_efficiency_integrated=core_efficiency_integrated,
            core_solves_preserved=core_solves_preserved,
            levers_integrated=levers_integrated,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
        "core_efficiency_integrated": core_efficiency_integrated,
        "core_solves_preserved": bool(core_solves_preserved),
        "levers_integrated": levers_integrated,
        "additivity_checked": _additivity_checked(
            upstream_decision=upstream_decision,
            integrated_measurement=current,
        ),
        "heldout_solve_rate": _heldout_solve_rate(current),
        "nav_diagnostics": dict(upstream_decision.get("nav_diagnostics") or {}),
        "ready_for_operator_submit": ready,
        "false_negative_risk_checked": _false_negative_risk_checked(
            baseline=baseline,
            gate_result=gate_result,
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "gate_games": list(GATE_GAMES),
        "core_games": list(CORE_GAMES),
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "upstream_decision": dict(upstream_decision),
        "gate_result": dict(gate_result),
        "per_game_deepest_level_reached": _per_game_deepest_level(current),
        "local_gate_budget": int(DEFAULT_GATE_BUDGET),
        "operator_submission_performed": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    baseline: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    missing = [
        key
        for key in ("offline_arcade_import",)
        if preconditions_checked.get(key) is not True
    ]
    reason = "_".join(missing) if missing else "unknown_resource"
    decision = {
        "accepted_levers": [],
        "rejected_levers": {},
        "upstream_summaries": {},
        "a2_control_levels": {game: 0 for game in CORE_GAMES},
        "a2_best_delta_core_efficiency": 0.0,
        "nav_diagnostics": {"reset_replay_steps_integrated": 0},
    }
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        baseline=baseline,
        upstream_decision=decision,
        gate_result={
            "current": {
                "games": list(GATE_GAMES),
                "per_game": [],
                "solved_games": [],
                "core_efficiency": CORE_EFFICIENCY_BASELINE,
            },
            "baseline_guard": {"ok": False},
        },
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{reason}"
    artifact["false_negative_risk_checked"] = False
    artifact["ready_for_operator_submit"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    success = isinstance(verdict, str) and verdict.startswith("success:")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4526")
    if float(artifact.get("core_efficiency_baseline") or 0.0) != CORE_EFFICIENCY_BASELINE:
        errors.append("core_efficiency_baseline must equal 2.0074")
    if not isinstance(artifact.get("core_efficiency_integrated"), (int, float)):
        errors.append("core_efficiency_integrated must be numeric")
    if not isinstance(artifact.get("core_solves_preserved"), bool):
        errors.append("core_solves_preserved must be bool")
    if not isinstance(artifact.get("levers_integrated"), list):
        errors.append("levers_integrated must be a list")
    if not isinstance(artifact.get("additivity_checked"), Mapping):
        errors.append("additivity_checked must be a mapping")
    if not isinstance(artifact.get("heldout_solve_rate"), (int, float)):
        errors.append("heldout_solve_rate must be numeric")
    nav = artifact.get("nav_diagnostics")
    if not isinstance(nav, Mapping) or "reset_replay_steps_integrated" not in nav:
        errors.append("nav_diagnostics must include reset_replay_steps_integrated")
    if artifact.get("ready_for_operator_submit") is True and not success:
        errors.append("ready_for_operator_submit cannot be true without success")
    if success:
        if float(artifact.get("core_efficiency_integrated") or 0.0) <= CORE_EFFICIENCY_BASELINE:
            errors.append("success requires core_efficiency_integrated above baseline")
        if artifact.get("core_solves_preserved") is not True:
            errors.append("success requires core_solves_preserved=true")
        if not artifact.get("levers_integrated"):
            errors.append("success requires an integrated lever")
    if not blocked and artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true for complete/success artifacts")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("operator_submission_performed") is not False:
        errors.append("operator_submission_performed must be false")
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
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
    load_upstream_artifacts: Callable[[Path], dict[str, dict[str, Any]]] = load_upstream_artifacts,
    gate_runner: Callable[..., dict[str, Any]] = run_local_submission_gate,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4526: run the integration gate and write its artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    gate_baseline = dict(baseline) if baseline is not None else load_gate_baseline(root_path)
    duration = lambda: max(0.0, float(now()) - started)
    if preconditions.get("offline_arcade_import") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            random_seed=random_seed,
            duration_s=duration(),
        )
    else:
        upstream = load_upstream_artifacts(root_path)
        decision = select_integrated_levers(
            a1_artifact=upstream.get("a1_forward_walk", {}),
            a2_reach_artifact=upstream.get("a2_reach_deeper_levels", {}),
            stop_after_levelup_artifact=upstream.get("a2_stop_after_levelup"),
        )
        gate_result = gate_runner(
            root=root_path,
            budget=DEFAULT_GATE_BUDGET,
            cap=DEFAULT_GATE_CAP_SECONDS,
        )
        artifact = build_artifact(
            preconditions_checked=preconditions,
            baseline=gate_baseline,
            upstream_decision=decision,
            gate_result=gate_result,
            random_seed=random_seed,
            duration_s=duration(),
        )
        errors = artifact_schema_errors(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    main()
