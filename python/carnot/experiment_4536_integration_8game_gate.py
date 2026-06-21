"""Experiment 4536: submitted A1/A2 integration gate refresh.

Spec refs: REQ-ARC-WMTE-4536, SCENARIO-ARC-WMTE-4536.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG
from carnot.experiment_4526_integration_8game_gate import (
    _baseline_core,
    _core_efficiency,
    _current_measurement,
    _heldout_solve_rate,
    _parse_gate_stdout,
    _per_game_deepest_level,
    _solved_games,
    payload_checksum,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4536_integration_8game_gate.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the "
    "per-level gate."
)
CORE_EFFICIENCY_BASELINE = 2.0074
RANDOM_SEED = 4536
DEFAULT_GATE_BUDGET = 8000
DEFAULT_GATE_CAP_SECONDS = 115
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
REQUIREMENTS = ("REQ-ARC-WMTE-4536",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4536",)
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
    "a1_per_level_goal_reinduction": "results/experiment_4533_per_level_goal_reinduction.json",
    "a2_energy_trust_next_level_routing": "results/experiment_4534_energy_trust_next_level_routing.json",
}
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR "
        "complete: no_lever_raises_core_efficiency_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the per-level gate."
    ),
    "core_efficiency_baseline": (
        "2.0074 -- the REAL per-level metric control (NOT median actions, retired)."
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
        "integrated CORE core_efficiency vs the naive sum of isolated A1+A2 deltas -- surfaces a "
        "destructive re-induction x energy-routing interaction instead of burying it."
    ),
    "heldout_solve_rate": "the real transfer signal; integration should not regress it.",
    "ready_for_operator_submit": (
        "True if the integrated config is a CORE-preserved core_efficiency improvement worth a 1/day "
        "submission slot; the task NEVER submits (operator-only)."
    ),
    "false_negative_risk_checked": (
        "an honest null only valid with the 2.0074 baseline measured the same way."
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
        "median_actions_on_solved": 7760.0,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "a1_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a1_per_level_goal_reinduction"]
        ).exists(),
        "a2_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a2_energy_trust_next_level_routing"]
        ).exists(),
        "spec_has_req_4536": spec_path.exists()
        and "REQ-ARC-WMTE-4536" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        _kit().offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive for malformed artifacts.
        return None


def _round_delta(value: float | None) -> float:
    return round(float(value or 0.0), 4)


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


def _deeper_core_level(control: Mapping[str, int], treatment: Mapping[str, int]) -> bool:
    return any(
        int(treatment.get(game, 0)) >= 2
        and int(treatment.get(game, 0)) > int(control.get(game, 0))
        for game in CORE_GAMES
    )


def _permitted_flagged_null(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("flagged_adversarial") is not True:
        return False
    baseline = _float_or_none(artifact.get("core_efficiency_baseline"))
    best = _float_or_none(artifact.get("core_efficiency_best"))
    delta = _float_or_none(artifact.get("efficiency_delta"))
    note = str(artifact.get("null_delta_methodology_note") or "")
    return bool(
        baseline == CORE_EFFICIENCY_BASELINE
        and best == CORE_EFFICIENCY_BASELINE
        and delta == 0.0
        and note
        and "baseline==best" in note
    )


def _flag_status(artifact: Mapping[str, Any]) -> str:
    if artifact.get("flagged_adversarial") is not True:
        return "clean"
    if _permitted_flagged_null(artifact):
        return "permitted_flagged_null"
    return "rejected_flagged_adversarial"


def _a1_control_levels(a1_artifact: Mapping[str, Any]) -> dict[str, int]:
    for row in a1_artifact.get("target_levels_sweep", []) or []:
        if isinstance(row, Mapping) and int(row.get("target_levels") or 0) == 1:
            levels = _levels_from_mapping(row.get("deepest_level_by_game"))
            if levels:
                return levels
    deepest = a1_artifact.get("deepest_level_reached_per_core_game")
    if isinstance(deepest, Mapping):
        levels = _levels_from_mapping(deepest.get("1"))
        if levels:
            return levels
    return {game: 0 for game in CORE_GAMES}


def _a1_delta(a1_artifact: Mapping[str, Any]) -> float:
    explicit = _float_or_none(a1_artifact.get("efficiency_delta"))
    if explicit is not None:
        return _round_delta(explicit)
    best = _float_or_none(a1_artifact.get("core_efficiency_best"))
    if best is None:
        return 0.0
    return _round_delta(best - CORE_EFFICIENCY_BASELINE)


def _a2_control_levels(a2_artifact: Mapping[str, Any]) -> dict[str, int]:
    control = a2_artifact.get("no_energy_control")
    if isinstance(control, Mapping):
        levels = _levels_from_mapping(control.get("deepest_level_by_game"))
        if levels:
            return levels
    deepest = a2_artifact.get("deepest_level_reached_per_core_game")
    if isinstance(deepest, Mapping):
        levels = _levels_from_mapping(deepest.get("no_energy_control"))
        if levels:
            return levels
    return {game: 0 for game in CORE_GAMES}


def _a2_energy_measurement(a2_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    measurement = a2_artifact.get("energy_routed_measurement")
    return measurement if isinstance(measurement, Mapping) else {}


def _a2_delta(a2_artifact: Mapping[str, Any]) -> float:
    measurement = _a2_energy_measurement(a2_artifact)
    efficiency = _float_or_none(measurement.get("core_efficiency"))
    if efficiency is None:
        efficiency = _float_or_none(a2_artifact.get("core_efficiency_energy_routed"))
    if efficiency is None:
        return 0.0
    return _round_delta(efficiency - CORE_EFFICIENCY_BASELINE)


def _summary_for_a1(a1_artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "honest_verdict": a1_artifact.get("honest_verdict"),
        "flag_status": _flag_status(a1_artifact),
        "efficiency_delta": _a1_delta(a1_artifact),
        "core_efficiency_best": _float_or_none(a1_artifact.get("core_efficiency_best")),
        "null_delta_methodology_note": a1_artifact.get("null_delta_methodology_note"),
    }


def _summary_for_a2(a2_artifact: Mapping[str, Any]) -> dict[str, Any]:
    measurement = _a2_energy_measurement(a2_artifact)
    return {
        "honest_verdict": a2_artifact.get("honest_verdict"),
        "flag_status": _flag_status(a2_artifact),
        "efficiency_delta": _a2_delta(a2_artifact),
        "core_efficiency_energy_routed": _float_or_none(
            measurement.get("core_efficiency")
            if measurement
            else a2_artifact.get("core_efficiency_energy_routed")
        ),
        "positive_control_passed": a2_artifact.get("positive_control_passed"),
        "false_negative_risk_checked": a2_artifact.get("false_negative_risk_checked"),
    }


def _reject(rejected: dict[str, dict[str, Any]], key: str, summary: Mapping[str, Any], reason: str) -> None:
    rejected[key] = {**dict(summary), "reason": reason}


def select_integrated_levers(
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4536: select only deeper-level CORE efficiency winners."""

    accepted: list[str] = []
    rejected: dict[str, dict[str, Any]] = {}
    a1_summary = _summary_for_a1(a1_artifact)
    a2_summary = _summary_for_a2(a2_artifact)
    summaries = {
        "A1_per_level_goal_reinduction": a1_summary,
        "A2_energy_trust_next_level_routing": a2_summary,
    }

    a1_control = _a1_control_levels(a1_artifact)
    if a1_summary["flag_status"] == "rejected_flagged_adversarial":
        _reject(rejected, "A1_per_level_goal_reinduction", a1_summary, "flagged_adversarial")
    else:
        for row in a1_artifact.get("target_levels_sweep", []) or []:
            if not isinstance(row, Mapping):
                continue
            target = int(row.get("target_levels") or 0)
            if target <= 1:
                continue
            key = f"A1_per_level_goal_reinduction:target_levels_{target}"
            levels = _levels_from_mapping(row.get("deepest_level_by_game"))
            regressions = _core_regressions(a1_control, levels)
            efficiency = _round_delta(_float_or_none(row.get("core_efficiency")))
            row_summary = {
                "target_levels": target,
                "core_efficiency": efficiency,
                "delta_vs_baseline": _round_delta(efficiency - CORE_EFFICIENCY_BASELINE),
                "deepest_level_by_game": {game: int(levels.get(game, 0)) for game in CORE_GAMES},
                "lost_core_level_games": regressions,
                "flag_status": a1_summary["flag_status"],
            }
            if regressions:
                _reject(rejected, key, row_summary, "core_level_regression")
            elif row.get("core_solves_preserved") is not True:
                _reject(rejected, key, row_summary, "core_solves_not_preserved")
            elif efficiency <= CORE_EFFICIENCY_BASELINE:
                _reject(rejected, key, row_summary, "no_core_efficiency_gain")
            elif not _deeper_core_level(a1_control, levels):
                _reject(rejected, key, row_summary, "no_deeper_core_level")
            elif a1_artifact.get("offline_reproduced") is not True:
                _reject(rejected, key, row_summary, "offline_reproduction_missing")
            else:
                accepted.append(key)

    a2_control = _a2_control_levels(a2_artifact)
    a2_key = "A2_energy_trust_next_level_routing:energy_routed"
    a2_measurement = _a2_energy_measurement(a2_artifact)
    a2_levels = _levels_from_mapping(a2_measurement.get("deepest_level_by_game"))
    a2_efficiency = _round_delta(_float_or_none(a2_measurement.get("core_efficiency")))
    a2_regressions = _core_regressions(a2_control, a2_levels)
    a2_row_summary = {
        "core_efficiency": a2_efficiency,
        "delta_vs_baseline": _round_delta(a2_efficiency - CORE_EFFICIENCY_BASELINE),
        "deepest_level_by_game": {game: int(a2_levels.get(game, 0)) for game in CORE_GAMES},
        "lost_core_level_games": a2_regressions,
        "flag_status": a2_summary["flag_status"],
    }
    if a2_summary["flag_status"] == "rejected_flagged_adversarial":
        _reject(rejected, a2_key, a2_row_summary, "flagged_adversarial")
    elif a2_regressions:
        _reject(rejected, a2_key, a2_row_summary, "core_level_regression")
    elif a2_artifact.get("core_solves_preserved") is not True:
        _reject(rejected, a2_key, a2_row_summary, "core_solves_not_preserved")
    elif a2_efficiency <= CORE_EFFICIENCY_BASELINE:
        _reject(rejected, a2_key, a2_row_summary, "no_core_efficiency_gain")
    elif not _deeper_core_level(a2_control, a2_levels):
        _reject(rejected, a2_key, a2_row_summary, "no_deeper_core_level")
    else:
        accepted.append(a2_key)

    isolated_deltas = {
        "A1_per_level_goal_reinduction": _a1_delta(a1_artifact),
        "A2_energy_trust_next_level_routing": _a2_delta(a2_artifact),
    }
    return {
        "accepted_levers": accepted,
        "rejected_levers": rejected,
        "upstream_summaries": summaries,
        "isolated_deltas": isolated_deltas,
        "naive_isolated_delta": _round_delta(sum(isolated_deltas.values())),
        "a1_control_levels": {game: int(a1_control.get(game, 0)) for game in CORE_GAMES},
        "a2_control_levels": {game: int(a2_control.get(game, 0)) for game in CORE_GAMES},
    }


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
        "experiment_4536_integration",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=cap + 60)
    parsed = _parse_gate_stdout(proc.stdout)
    parsed["gate_command"] = cmd
    parsed["gate_returncode"] = proc.returncode
    if proc.stderr:
        parsed["gate_stderr"] = proc.stderr
    return parsed


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
    deltas = upstream_decision.get("isolated_deltas")
    isolated = dict(deltas) if isinstance(deltas, Mapping) else {}
    a1_delta = float(isolated.get("A1_per_level_goal_reinduction") or 0.0)
    a2_delta = float(isolated.get("A2_energy_trust_next_level_routing") or 0.0)
    naive_sum_delta = _round_delta(a1_delta + a2_delta)
    integrated_delta = _round_delta(_core_efficiency(integrated_measurement) - CORE_EFFICIENCY_BASELINE)
    return {
        "metric": "core_efficiency",
        "a1_delta_core_efficiency": _round_delta(a1_delta),
        "a2_delta_core_efficiency": _round_delta(a2_delta),
        "naive_sum_delta": naive_sum_delta,
        "integrated_delta": integrated_delta,
        "interaction_delta": _round_delta(integrated_delta - naive_sum_delta),
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
    """SCENARIO-ARC-WMTE-4536: assemble the terminal integration artifact."""

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
        "experiment": "experiment_4536_integration_8game_gate",
        "schema": "carnot.arc_integration_8game_gate_4536.v1",
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
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        baseline=baseline,
        upstream_decision={
            "accepted_levers": [],
            "rejected_levers": {},
            "upstream_summaries": {},
            "isolated_deltas": {
                "A1_per_level_goal_reinduction": 0.0,
                "A2_energy_trust_next_level_routing": 0.0,
            },
            "naive_isolated_delta": 0.0,
        },
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
        errors.append("field_principles must match REQ-ARC-WMTE-4536")
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
    """REQ-ARC-WMTE-4536: run the integration gate and write its artifact."""

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
            upstream.get("a1_per_level_goal_reinduction", {}),
            upstream.get("a2_energy_trust_next_level_routing", {}),
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
