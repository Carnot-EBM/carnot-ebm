"""Experiment 4548: submitted A1/A4 integration gate.

Spec refs: REQ-ARC-WMTE-4548, SCENARIO-ARC-WMTE-4548.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.submitted_agent_config_ast import parse_submitted_agent_config


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4548_integration_8game_gate.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the "
    "per-level gate (if the integrated config invokes the LLM proposer in the measured "
    "run, declare live_llm_inference + add the model precondition)."
)
CORE_EFFICIENCY_BASELINE = 2.0074
RANDOM_SEED = 4548
DEFAULT_GATE_BUDGET = 8000
DEFAULT_GATE_CAP_SECONDS = 115
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
REQUIREMENTS = ("REQ-ARC-WMTE-4548",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4548",)
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
    "a1_llm_proposer_reinduction": "results/experiment_4544_llm_proposer_reinduction.json",
    "a4_frame_change_cnn_ranker": "results/experiment_4547_frame_change_predictor.json",
}
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR "
        "complete: no_lever_raises_core_efficiency_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the "
        "per-level gate (if the integrated config invokes the LLM proposer in the measured run, "
        "declare live_llm_inference + add the model precondition)."
    ),
    "core_efficiency_baseline": (
        "2.0074 -- the REAL per-level metric control (NOT median actions, retired)."
    ),
    "core_efficiency_integrated": (
        "the HEADLINE -- the SUBMITTED-config per-level efficiency after wiring the winners "
        "(did it solve MORE/DEEPER levels)."
    ),
    "core_solves_preserved": "integration must preserve every CORE solve (set-containment).",
    "levers_integrated": (
        "names which of A1/A4 were wired -- traceable to their measured deltas; [] is an honest null."
    ),
    "additivity_checked": (
        "integrated CORE core_efficiency vs the naive sum of isolated A1+A4 deltas -- surfaces a "
        "destructive LLM-proposer x CNN-ranker interaction instead of burying it."
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


def _submitted_agent_config() -> dict[str, Any]:
    """Read the submitted agent's config from source, without importing the agent.

    The parsing itself lives in ``carnot.submitted_agent_config_ast`` -- shared with experiment
    4560, which needs the identical read. It used to be copy-pasted into both, and both copies
    carried the same defect: a constant defined by *reference* rather than by literal was silently
    dropped by ``literal_eval``'s except-continue, then resurfaced much later as a bare
    ``KeyError`` from the lookup that needed it. Sharing one implementation is what stops the next
    fix from landing in only one of them.
    """
    return parse_submitted_agent_config(
        REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
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


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "a1_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a1_llm_proposer_reinduction"]
        ).exists(),
        "a4_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a4_frame_change_cnn_ranker"]
        ).exists(),
        "spec_has_req_4548": spec_path.exists()
        and "REQ-ARC-WMTE-4548" in spec_path.read_text(encoding="utf-8"),
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
    except (TypeError, ValueError):  # pragma: no cover - malformed upstream defense.
        return None


def _round_delta(value: float | None) -> float:
    return round(float(value or 0.0), 4)


def _levels_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(game): int(level or 0) for game, level in value.items()}


def _core_levels_or_zero(levels: Mapping[str, int]) -> dict[str, int]:
    return {game: int(levels.get(game, 0)) for game in CORE_GAMES}


def _levels_from_nested(
    artifact: Mapping[str, Any],
    *,
    outer_keys: Sequence[str],
    direct_keys: Sequence[str],
) -> dict[str, int]:
    for key in direct_keys:
        value = artifact.get(key)
        if isinstance(value, Mapping):
            levels = _levels_from_mapping(value.get("deepest_level_by_game"))
            if levels:
                return levels
    deepest = artifact.get("deepest_level_reached_per_core_game")
    if isinstance(deepest, Mapping):
        for key in outer_keys:
            levels = _levels_from_mapping(deepest.get(key))
            if levels:
                return levels
    return {game: 0 for game in CORE_GAMES}


def _core_regressions(control: Mapping[str, int], treatment: Mapping[str, int]) -> list[str]:
    return [game for game in CORE_GAMES if int(treatment.get(game, 0)) < int(control.get(game, 0))]


def _core_progress_gain(control: Mapping[str, int], treatment: Mapping[str, int]) -> bool:
    return any(
        int(treatment.get(game, 0)) > int(control.get(game, 0)) and int(treatment.get(game, 0)) > 0
        for game in CORE_GAMES
    )


def _corrigendum_kinds(artifact: Mapping[str, Any]) -> list[str]:
    rows = artifact.get("corrigendum_pending")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    kinds: list[str] = []
    for row in rows:
        if isinstance(row, Mapping) and row.get("kind") is not None:
            kinds.append(str(row["kind"]))
    return kinds


def _permitted_flagged_null(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("flagged_adversarial") is not True:
        return False
    baseline = _float_or_none(artifact.get("core_efficiency_baseline"))
    best = _float_or_none(artifact.get("core_efficiency_best"))
    delta = _float_or_none(artifact.get("efficiency_delta"))
    note = str(artifact.get("null_delta_methodology_note") or "")
    return bool(
        _corrigendum_kinds(artifact) == ["TAUTOLOGY"]
        and baseline == CORE_EFFICIENCY_BASELINE
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
    return _core_levels_or_zero(
        _levels_from_nested(
            a1_artifact,
            outer_keys=("offline_dsl_baseline", "baseline", "control"),
            direct_keys=("offline_dsl_baseline", "control"),
        )
    )


def _a1_treatment_levels(a1_artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            a1_artifact,
            outer_keys=("llm_proposer", "best", "treatment"),
            direct_keys=("llm_proposer_measurement", "best_measurement"),
        )
    )


def _a4_control_levels(a4_artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            a4_artifact,
            outer_keys=("blind_bfs", "blind", "bare_explorer", "control"),
            direct_keys=("blind_bfs_control", "blind_control", "control"),
        )
    )


def _a4_treatment_levels(a4_artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            a4_artifact,
            outer_keys=("cnn_ranker", "cnn", "frame_change_ranker", "treatment"),
            direct_keys=("frame_change_ranker_measurement", "cnn_measurement", "best_measurement"),
        )
    )


def _a1_delta(a1_artifact: Mapping[str, Any]) -> float:
    explicit = _float_or_none(a1_artifact.get("efficiency_delta"))
    if explicit is not None:
        return _round_delta(explicit)
    best = _float_or_none(a1_artifact.get("core_efficiency_best"))
    if best is None:
        return 0.0
    return _round_delta(best - CORE_EFFICIENCY_BASELINE)


def _a4_efficiency(a4_artifact: Mapping[str, Any]) -> float | None:
    for key in ("frame_change_ranker_measurement", "cnn_measurement", "best_measurement"):
        value = a4_artifact.get(key)
        if isinstance(value, Mapping):
            efficiency = _float_or_none(value.get("core_efficiency"))
            if efficiency is not None:
                return efficiency
    for key in ("core_efficiency_cnn", "core_efficiency_best", "core_efficiency_ranked"):
        efficiency = _float_or_none(a4_artifact.get(key))
        if efficiency is not None:
            return efficiency
    return None


def _a4_delta(a4_artifact: Mapping[str, Any]) -> float:
    efficiency = _a4_efficiency(a4_artifact)
    if efficiency is None:
        return 0.0
    return _round_delta(efficiency - CORE_EFFICIENCY_BASELINE)


def _summary_for_a1(a1_artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "honest_verdict": a1_artifact.get("honest_verdict"),
        "flag_status": _flag_status(a1_artifact),
        "raw_efficiency_delta": _a1_delta(a1_artifact),
        "core_efficiency_best": _float_or_none(a1_artifact.get("core_efficiency_best")),
        "corrigendum_kinds": _corrigendum_kinds(a1_artifact),
        "null_delta_methodology_note": a1_artifact.get("null_delta_methodology_note"),
    }


def _summary_for_a4(a4_artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "honest_verdict": a4_artifact.get("honest_verdict"),
        "core_efficiency_cnn": _a4_efficiency(a4_artifact),
        "efficiency_delta": _a4_delta(a4_artifact),
        "median_actions_to_first_levelup_blind": a4_artifact.get(
            "median_actions_to_first_levelup_blind"
        ),
        "median_actions_to_first_levelup_cnn": a4_artifact.get(
            "median_actions_to_first_levelup_cnn"
        ),
        "solve_rate_preserved": a4_artifact.get("solve_rate_preserved"),
        "positive_control_passed": a4_artifact.get("positive_control_passed"),
        "false_negative_risk_checked": a4_artifact.get("false_negative_risk_checked"),
    }


def _reject(
    rejected: dict[str, dict[str, Any]], key: str, summary: Mapping[str, Any], reason: str
) -> None:
    rejected[key] = {**dict(summary), "reason": reason}


def select_integrated_levers(
    a1_artifact: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4548: select only CORE-efficiency A1/A4 winners."""

    accepted: list[str] = []
    rejected: dict[str, dict[str, Any]] = {}
    a1_summary = _summary_for_a1(a1_artifact)
    a4_summary = _summary_for_a4(a4_artifact)
    summaries = {
        "A1_llm_proposer_reinduction": a1_summary,
        "A4_frame_change_cnn_ranker": a4_summary,
    }

    a1_control = _a1_control_levels(a1_artifact)
    a1_treatment = _a1_treatment_levels(a1_artifact)
    a1_efficiency = _float_or_none(a1_artifact.get("core_efficiency_best"))
    a1_efficiency_rounded = _round_delta(a1_efficiency)
    a1_regressions = _core_regressions(a1_control, a1_treatment)
    a1_key = "A1_llm_proposer_reinduction:llm_proposer"
    a1_row_summary = {
        "core_efficiency": a1_efficiency_rounded,
        "delta_vs_baseline": _round_delta((a1_efficiency or 0.0) - CORE_EFFICIENCY_BASELINE),
        "deepest_level_by_game": dict(a1_treatment),
        "lost_core_level_games": a1_regressions,
        "flag_status": a1_summary["flag_status"],
    }
    if a1_summary["flag_status"] == "rejected_flagged_adversarial":
        _reject(rejected, "A1_llm_proposer_reinduction", a1_summary, "flagged_adversarial")
    elif a1_regressions:
        _reject(rejected, a1_key, a1_row_summary, "core_level_regression")
    elif a1_artifact.get("core_solves_preserved") is not True:
        _reject(rejected, a1_key, a1_row_summary, "core_solves_not_preserved")
    elif a1_efficiency is None or a1_efficiency <= CORE_EFFICIENCY_BASELINE:
        _reject(rejected, a1_key, a1_row_summary, "no_core_efficiency_gain")
    elif not _core_progress_gain(a1_control, a1_treatment):
        _reject(rejected, a1_key, a1_row_summary, "no_deeper_core_level")
    elif a1_artifact.get("offline_reproduced") is not True:
        _reject(rejected, a1_key, a1_row_summary, "offline_reproduction_missing")
    elif a1_artifact.get("positive_control_passed") is not True:
        _reject(rejected, a1_key, a1_row_summary, "positive_control_failed")
    else:
        accepted.append(a1_key)

    a4_control = _a4_control_levels(a4_artifact)
    a4_treatment = _a4_treatment_levels(a4_artifact)
    a4_efficiency = _a4_efficiency(a4_artifact)
    a4_regressions = _core_regressions(a4_control, a4_treatment)
    a4_key = "A4_frame_change_cnn_ranker:cnn_ranker"
    a4_row_summary = {
        "core_efficiency": None if a4_efficiency is None else _round_delta(a4_efficiency),
        "delta_vs_baseline": None
        if a4_efficiency is None
        else _round_delta(a4_efficiency - CORE_EFFICIENCY_BASELINE),
        "deepest_level_by_game": dict(a4_treatment),
        "lost_core_level_games": a4_regressions,
        "median_actions_retired": True,
    }
    if a4_efficiency is None:
        _reject(rejected, a4_key, a4_row_summary, "no_core_efficiency_evidence")
    elif a4_regressions:
        _reject(rejected, a4_key, a4_row_summary, "core_level_regression")
    elif a4_artifact.get("core_solves_preserved") is not True:
        _reject(rejected, a4_key, a4_row_summary, "core_solves_not_preserved")
    elif a4_efficiency <= CORE_EFFICIENCY_BASELINE:
        _reject(rejected, a4_key, a4_row_summary, "no_core_efficiency_gain")
    elif not _core_progress_gain(a4_control, a4_treatment):
        _reject(rejected, a4_key, a4_row_summary, "no_deeper_core_level")
    elif a4_artifact.get("solve_rate_preserved") is False:
        _reject(rejected, a4_key, a4_row_summary, "solve_rate_not_preserved")
    else:
        accepted.append(a4_key)

    a1_isolated_delta = (
        0.0
        if a1_summary["flag_status"] == "rejected_flagged_adversarial"
        else _a1_delta(a1_artifact)
    )
    a4_isolated_delta = _a4_delta(a4_artifact)
    isolated_deltas = {
        "A1_llm_proposer_reinduction": _round_delta(a1_isolated_delta),
        "A4_frame_change_cnn_ranker": _round_delta(a4_isolated_delta),
    }
    return {
        "accepted_levers": accepted,
        "rejected_levers": rejected,
        "upstream_summaries": summaries,
        "isolated_deltas": isolated_deltas,
        "naive_isolated_delta": _round_delta(sum(isolated_deltas.values())),
        "a1_control_levels": dict(a1_control),
        "a4_control_levels": dict(a4_control),
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
        "experiment_4548_integration",
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
        if (
            isinstance(row, Mapping)
            and row.get("game") is not None
            and row.get("efficiency") is not None
        ):
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
    deltas = upstream_decision.get("isolated_deltas")
    isolated = dict(deltas) if isinstance(deltas, Mapping) else {}
    a1_delta = float(isolated.get("A1_llm_proposer_reinduction") or 0.0)
    a4_delta = float(isolated.get("A4_frame_change_cnn_ranker") or 0.0)
    naive_sum_delta = _round_delta(a1_delta + a4_delta)
    integrated_delta = _round_delta(
        _core_efficiency(integrated_measurement) - CORE_EFFICIENCY_BASELINE
    )
    return {
        "metric": "core_efficiency",
        "a1_delta_core_efficiency": _round_delta(a1_delta),
        "a4_delta_core_efficiency": _round_delta(a4_delta),
        "naive_sum_delta": naive_sum_delta,
        "integrated_delta": integrated_delta,
        "interaction_delta": _round_delta(integrated_delta - naive_sum_delta),
        "median_actions_retired": True,
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
    """SCENARIO-ARC-WMTE-4548: assemble the terminal integration artifact."""

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
        "experiment": "experiment_4548_integration_8game_gate",
        "schema": "carnot.arc_integration_8game_gate_4548.v1",
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
        "submitted_agent_config": _submitted_agent_config(),
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
        key for key in ("offline_arcade_import",) if preconditions_checked.get(key) is not True
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
                "A1_llm_proposer_reinduction": 0.0,
                "A4_frame_change_cnn_ranker": 0.0,
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
        errors.append("field_principles must match REQ-ARC-WMTE-4548")
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
    """REQ-ARC-WMTE-4548: run the integration gate and write its artifact."""

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
            upstream.get("a1_llm_proposer_reinduction", {}),
            upstream.get("a4_frame_change_cnn_ranker", {}),
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
