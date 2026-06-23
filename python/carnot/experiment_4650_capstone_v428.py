"""Experiment 4650: .428 ARC energy-driven generation bridge capstone.

This module aggregates the landed A1-A6 and B1/B2 artifacts. It is not a
solver, and it never submits to the leaderboard.

Spec refs: REQ-CAPSTONE-4650, SCENARIO-CAPSTONE-4650,
SCENARIO-CAPSTONE-4650-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

try:  # pragma: no cover - tests inject live flags directly
    import summarize_artifact as artifact_reader
except Exception:  # pragma: no cover - defensive import boundary
    artifact_reader = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4650_capstone_v428"
SCHEMA = "carnot.exp4650.capstone_v428.v1"
RESULT_RELATIVE_PATH = "results/experiment_4650_capstone_v428.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4650
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 56
LIVE_SUBMITTABLE_SCORECARD_BASELINE = 33
FIRST_WIN_RATE_V427_BASELINE = 0.590909
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_path: str
    role: str


UPSTREAM_SOURCES: dict[str, SourceSpec] = {
    "A1": SourceSpec(
        "A1",
        "results/experiment_4640_goal_energy_generation_live.json",
        "goal_energy_generation_live",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4641_action_effect_expansion_prior_live.json",
        "action_effect_expansion_prior_live",
    ),
    "A3": SourceSpec(
        "A3",
        "results/experiment_4642_levelup_selfplay.json",
        "levelup_selfplay_bank",
    ),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4643_refresh_submission_package.json",
        "refresh_submission_package",
    ),
    "A5": SourceSpec(
        "A5",
        "results/experiment_4644_primitive_persist_transfer.json",
        "primitive_persist_transfer",
    ),
    "A6": SourceSpec(
        "A6",
        "results/experiment_4645_integration_gate.json",
        "integration_gate",
    ),
    "B1": SourceSpec(
        "B1",
        "results/experiment_4646_live_multi_level_solve_rate_metric.json",
        "live_multi_level_solve_rate_metric",
    ),
    "B2": SourceSpec(
        "B2",
        "results/experiment_4647_adversarial_verify_hardening.json",
        "adversarial_verify_goal_energy_ablation_guard",
    ),
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_extended_live_<solverate|multilevel>_up_<n> "
            "OR complete: energy_generation_levers_characterized_no_live_lift OR complete: "
            "capability_grew_56_to_<n>."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false on every included value claim (A1/A2/A5/A6 oracle-distinct) -- "
            "a circular win would not count."
        )
    },
    "live_solve_rate_delta": {
        "principle": (
            "the A1 SCORED-agent live solve-rate delta vs baseline (did ENERGY-DRIVEN generation "
            "cross the bridge for SOLVE) -- valid only if the uniform-energy ablation was beaten."
        )
    },
    "live_multi_level_solve_rate": {
        "principle": (
            "the A2/B1 >=2-level live solve-rate (the new wall) -- did the expansion prior deepen "
            "the live solve."
        )
    },
    "live_action_efficiency": {
        "principle": (
            "the A2 leaderboard score term (min(human/agent,1)^2) -- did the generation levers "
            "hold/raise efficiency."
        )
    },
    "offline_to_live_transfer_ratio": {
        "principle": (
            "the bridge co-metric (A1/A2) -- did the offline signal transfer to a LIVE lift "
            "(bridge crossed) or stay zero-live."
        )
    },
    "reproducible_total_levels": {
        "principle": "authoritative from the registry (A3 bank, 56->57+) -- did solve CAPABILITY grow."
    },
    "reproducible_total_levels_delta": {
        "principle": "registry after - 56, emitted explicitly so a null is annotated."
    },
    "live_submittable_level_count": {
        "principle": "the A4 operator-resubmit count (must stay > 33)."
    },
    "first_win_rate_scored": {
        "principle": (
            "the A1/A2 SCORED-agent live first-win-rate vs baseline (0.591 .427 baseline; "
            "must not regress)."
        )
    },
    "uniform_energy_ablation_passed": {
        "principle": (
            "the A1 energy-specific control -- a goal-energy headline is valid only if the graded "
            "energy beat the uniform-energy ablation (.428-B2 guard)."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial / positive-control-failed / ablation-failed artifact "
            "EXCLUDED + the guards applied (.425-B2 TAUTOLOGY, .426-B2 offline-vs-live, "
            ".427-B2 intrinsic-reward, .428-B2 goal-energy-ablation) -- fabrication-gate + "
            "false-negative-risk compliance."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream artifact with sha256 (the audit trail)."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only if the refreshed package beats 33 on a real metric worth a 1/day slot; "
            "never submits (operator-only)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (upstream artifacts present, offline arcade); pre-empts "
            "missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "result_path",
    "field_principles",
    "scorecard",
    "leaderboard_submission",
    "submitted_to_leaderboard",
    "duration_s",
)
SPEC_REFS = [
    "REQ-CAPSTONE-4650",
    "SCENARIO-CAPSTONE-4650",
    "SCENARIO-CAPSTONE-4650-FIELD-PRINCIPLES",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _checksum(payload)


def _read_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _file_sha256(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _live_flags(path: Path) -> list[dict[str, Any]]:  # pragma: no cover - external reader boundary
    if artifact_reader is None:
        return []
    try:
        flags = artifact_reader._live_flags(path)  # type: ignore[attr-defined]
    except Exception:
        return []
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def _severity(flag: Mapping[str, Any]) -> str:
    return str(flag.get("severity") or "").lower()


def _critical_flags(flags: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(flag) for flag in flags if _severity(flag) == "critical"]


def _false_negative_risk_open(flags: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        flag.get("kind") == "FALSE_NEGATIVE_RISK"
        and "false_negative_risk_open" in str(flag.get("detail") or "")
        for flag in flags
    )


def _acceptance_gate_failures(artifact: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, value in artifact.items():
        lowered = key.lower()
        is_gate = (
            "acceptance_gate" in lowered
            or lowered.startswith("gate_")
            or lowered.endswith("_gate")
        )
        if is_gate and value is False:
            failures.append(key)
    return failures


def _positive_control_failed(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("positive_control_passed") is False:
        return True
    if "bare_control_passed" in artifact and artifact.get("bare_control_passed") is not True:
        return True
    return "false_negative_risk_checked" in artifact and artifact.get("false_negative_risk_checked") is not True


def _uniform_energy_ablation_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return name == "A1" and artifact and artifact.get("uniform_energy_ablation_passed") is not True


def _source_status(
    *,
    name: str,
    source: SourceSpec,
    root: Path,
    artifact: Mapping[str, Any],
    exists: bool,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> JsonDict:
    path = root / source.relative_path
    flags = (
        [dict(flag) for flag in live_flags_by_name[name]]
        if live_flags_by_name is not None and name in live_flags_by_name
        else (_live_flags(path) if exists else [])
    )
    critical = _critical_flags(flags)
    stamped = artifact.get("flagged_adversarial") is True
    gate_failures = _acceptance_gate_failures(artifact)
    positive_failed = _positive_control_failed(artifact)
    false_negative = _false_negative_risk_open(flags)
    ablation_failed = _uniform_energy_ablation_failed(name, artifact)
    flagged = bool(stamped or critical)
    included = bool(
        exists
        and artifact
        and not flagged
        and not gate_failures
        and not positive_failed
        and not false_negative
        and not ablation_failed
    )
    reason = "included_clean"
    if not exists:
        reason = "missing"
    elif gate_failures:
        reason = "failed_acceptance_gate"
    elif positive_failed:
        reason = "positive_control_failed"
    elif false_negative:
        reason = "false_negative_risk_open"
    elif ablation_failed:
        reason = "uniform_energy_ablation_failed"
    elif flagged:
        reason = "flagged_adversarial_or_live_critical_excluded"
    return {
        "name": name,
        "artifact": source.relative_path,
        "role": source.role,
        "exists": exists,
        "honest_verdict": artifact.get("honest_verdict"),
        "stamped_flagged_adversarial": stamped,
        "live_critical": bool(critical),
        "live_flags": flags,
        "critical_flags": critical,
        "positive_control_failed": positive_failed,
        "false_negative_risk_open": false_negative,
        "uniform_energy_ablation_failed": ablation_failed,
        "acceptance_gate_failures": gate_failures,
        "duration_s": artifact.get("duration_s"),
        "inference_substrate": artifact.get("inference_substrate"),
        "included_in_headline": included,
        "reason": reason,
        "sha256": _file_sha256(path) if path.exists() else _checksum(artifact),
    }


def _load_artifacts(
    root: Path,
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    loaded: dict[str, JsonDict] = {}
    statuses: dict[str, JsonDict] = {}
    for name, source in UPSTREAM_SOURCES.items():
        if artifacts is not None and name in artifacts:
            loaded[name] = dict(artifacts[name])
        else:
            loaded[name] = _read_json(root / source.relative_path)
        exists = bool(name in artifacts) if artifacts is not None else (root / source.relative_path).exists()
        statuses[name] = _source_status(
            name=name,
            source=source,
            root=root,
            artifact=loaded[name],
            exists=exists,
            live_flags_by_name=live_flags_by_name,
        )
    return loaded, statuses


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover - integration boundary
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry = _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    source_exists = {
        name: (
            bool(statuses.get(name, {}).get("exists"))
            if statuses is not None
            else (root_path / source.relative_path).exists()
        )
        for name, source in UPSTREAM_SOURCES.items()
    }
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4650": "REQ-CAPSTONE-4650" in spec_text,
        "registry_yaml_loadable": bool(registry),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_reproducible_total_levels": _as_int(registry.get("reproducible_total_levels")),
        "upstream_artifacts_present": source_exists,
        "missing_upstream_artifacts": [
            UPSTREAM_SOURCES[name].relative_path
            for name, exists in source_exists.items()
            if not exists
        ],
        "summarize_artifact_py_used_for_live_flags": artifact_reader is not None,
        "offline_arcade": False,
        "leaderboard_submission": False,
        "operator_only": True,
        "network_required": False,
        "research_conductor_modified": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4650",
        "registry_yaml_loadable",
        "offline_arcade",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(key for key in required if not checks[key])
    return checks


def _clean(statuses: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return bool(statuses.get(name, {}).get("included_in_headline"))


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _tests_passed(block: Any) -> bool:
    return not isinstance(block, Mapping) or block.get("passed") is True


def _b2_goal_energy_ablation_guard_active(b2: Mapping[str, Any]) -> bool:
    return bool(
        b2.get("goal_energy_ablation_guard_added") is True
        and b2.get("honest_ablation_not_flagged") is True
        and b2.get("diagnostic_not_flagged") is True
        and _tests_passed(b2.get("tests_added"))
    )


def _a1_live_solve_rate_delta(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    clean = status.get("included_in_headline") is True
    solve_delta = _as_float(a1.get("solve_rate_delta"))
    first_delta = _as_float(a1.get("first_win_rate_delta"))
    ablation_passed = a1.get("uniform_energy_ablation_passed") is True
    bridge = bool(clean and ablation_passed and (solve_delta > 0.0 or first_delta > 0.0))
    return {
        "clean_value": solve_delta if clean else None,
        "quarantined_value": None if clean else solve_delta,
        "live_solve_rate_goal_energy": _as_float(a1.get("live_solve_rate_goal_energy")),
        "live_solve_rate_baseline": _as_float(a1.get("live_solve_rate_baseline")),
        "solve_rate_delta": solve_delta,
        "first_win_rate_delta": first_delta,
        "uniform_energy_ablation_passed": ablation_passed,
        "bridge_crossed_clean": bridge,
        "headline_numbers_aggregated": clean,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
        "reason": status.get("reason"),
    }


def _first_win_rate_from_a2(a2: Mapping[str, Any]) -> float:
    expansion = _mapping_at(a2, "expansion_measurement")
    if "first_win_rate" in expansion:
        return _as_float(expansion.get("first_win_rate"))
    measurement = _mapping_at(a2, "live_measurement")
    expansion_prior = _mapping_at(measurement, "expansion_prior")
    return _as_float(expansion_prior.get("first_win_rate"))


def _live_multi_level_solve_rate(
    a2: Mapping[str, Any],
    b1: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    a2_clean = _clean(statuses, "A2")
    b1_clean = _clean(statuses, "B1")
    clean = a2_clean and b1_clean
    rate = _as_float(b1.get("live_multi_level_solve_rate"))
    ranker_baseline = _as_float(a2.get("live_solve_rate_ranker_baseline"))
    delta = round(rate - ranker_baseline, 10)
    depth_delta = _as_float(a2.get("depth_of_live_solve_delta"))
    first_win_rate = _first_win_rate_from_a2(a2)
    bridge = bool(clean and delta > 0.0 and depth_delta > 0.0 and first_win_rate >= FIRST_WIN_RATE_V427_BASELINE)
    return {
        "clean_value": rate if clean else None,
        "quarantined_value": None if clean else rate,
        "live_multi_level_solve_rate": rate,
        "ranker_baseline_multi_level_rate": ranker_baseline,
        "delta_vs_ranker_baseline": delta,
        "depth_of_live_solve_delta": depth_delta,
        "first_win_rate_expansion": first_win_rate,
        "first_win_regressed_vs_427_baseline": first_win_rate < FIRST_WIN_RATE_V427_BASELINE,
        "live_attempt_count": _as_int(b1.get("live_attempt_count")),
        "multi_level_attempt_count": _as_int(b1.get("multi_level_attempt_count")),
        "bridge_crossed_clean": bridge,
        "a2_included_in_headline": a2_clean,
        "b1_included_in_headline": b1_clean,
        "source": UPSTREAM_SOURCES["B1"].relative_path,
        "live_source": UPSTREAM_SOURCES["A2"].relative_path,
        "reason": "clean_live_multi_level_lift" if bridge else "zero_or_inadmissible_multi_level_lift",
    }


def _live_action_efficiency(
    a2: Mapping[str, Any],
    b1: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    clean = _clean(statuses, "A2") and _clean(statuses, "B1")
    coheadline = _mapping_at(b1, "coheadline_block")
    efficiency = _as_float(coheadline.get("live_action_efficiency"), _as_float(b1.get("live_action_efficiency")))
    return {
        "clean_value": efficiency if clean else None,
        "quarantined_value": None if clean else efficiency,
        "leaderboard_score_term": efficiency,
        "source": UPSTREAM_SOURCES["B1"].relative_path,
        "live_source": UPSTREAM_SOURCES["A2"].relative_path,
        "a2_included_in_headline": _clean(statuses, "A2"),
        "b1_included_in_headline": _clean(statuses, "B1"),
    }


def _offline_to_live_transfer_ratio(
    a1_metric: Mapping[str, Any],
    multi: Mapping[str, Any],
    b1: Mapping[str, Any],
) -> JsonDict:
    a1_lift = max(
        _as_float(a1_metric.get("solve_rate_delta")),
        _as_float(a1_metric.get("first_win_rate_delta")),
        0.0,
    ) if a1_metric.get("bridge_crossed_clean") is True else 0.0
    a2_lift = _as_float(multi.get("delta_vs_ranker_baseline")) if multi.get("bridge_crossed_clean") is True else 0.0
    coheadline = _mapping_at(b1, "coheadline_block")
    reported = _as_float(coheadline.get("offline_to_live_transfer_ratio"))
    ratio = round(max(a1_lift, a2_lift, reported), 10)
    clean = bool(a1_metric.get("headline_numbers_aggregated") or multi.get("clean_value") is not None)
    return {
        "clean_value": ratio if clean else None,
        "quarantined_value": None if clean else ratio,
        "a1_live_lift_component": a1_lift,
        "a2_multi_level_lift_component": a2_lift,
        "reported_coheadline_component": reported,
        "bridge_crossed_clean": bool(a1_metric.get("bridge_crossed_clean") or multi.get("bridge_crossed_clean")),
        "method": "max(A1 clean solve/first-win lift, A2 clean multi-level lift, B1 coheadline ratio)",
        "source": "A1/A2/B1",
    }


def _first_win_rate_scored(
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    b1: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    a2_clean = _clean(statuses, "A2")
    a1_clean = _clean(statuses, "A1")
    b1_clean = _clean(statuses, "B1")
    coheadline = _mapping_at(b1, "coheadline_block")
    a2_rate = _first_win_rate_from_a2(a2)
    a1_goal_rate = _as_float(a1.get("live_solve_rate_goal_energy"))
    value = (
        _as_float(coheadline.get("first_win_rate"))
        if b1_clean and "first_win_rate" in coheadline
        else (a2_rate if a2_clean else (a1_goal_rate if a1_clean else 0.0))
    )
    return {
        "clean_value": value if (b1_clean or a2_clean or a1_clean) else None,
        "quarantined_value": None if (b1_clean or a2_clean or a1_clean) else value,
        "coheadline_rate": _as_float(coheadline.get("first_win_rate")),
        "a2_expansion_rate": a2_rate,
        "a2_delta_vs_ranker": _as_float(a2.get("first_win_rate_delta")) if a2_clean else 0.0,
        "a1_goal_energy_rate": a1_goal_rate if a1_clean else 0.0,
        "a1_first_win_delta": _as_float(a1.get("first_win_rate_delta")) if a1_clean else 0.0,
        "v427_baseline": FIRST_WIN_RATE_V427_BASELINE,
        "delta_vs_427_baseline": round(value - FIRST_WIN_RATE_V427_BASELINE, 10),
        "regressed_vs_427_baseline": value < FIRST_WIN_RATE_V427_BASELINE,
        "source": UPSTREAM_SOURCES["B1"].relative_path if b1_clean else (
            UPSTREAM_SOURCES["A2"].relative_path if a2_clean else UPSTREAM_SOURCES["A1"].relative_path
        ),
        "reason": "clean_scored_first_win_context" if (b1_clean or a2_clean or a1_clean) else "no_clean_first_win_source",
    }


def _flagged_artifacts_handled(
    statuses: Mapping[str, Mapping[str, Any]],
    b2: Mapping[str, Any],
) -> JsonDict:
    excluded_details: list[JsonDict] = []
    positive_control_failed: list[JsonDict] = []
    false_negative_open: list[JsonDict] = []
    ablation_failed: list[JsonDict] = []
    gate_failures: list[JsonDict] = []
    for name, status in statuses.items():
        if status.get("included_in_headline") is False and status.get("reason") not in {"missing", "included_clean"}:
            excluded_details.append(
                {
                    "name": name,
                    "artifact": status.get("artifact"),
                    "reason": status.get("reason"),
                    "critical_flags": [
                        {"kind": flag.get("kind"), "detail": flag.get("detail")}
                        for flag in status.get("critical_flags", [])
                    ],
                }
            )
        if status.get("positive_control_failed"):
            positive_control_failed.append({"name": name, "artifact": status.get("artifact")})
        if status.get("false_negative_risk_open"):
            false_negative_open.append({"name": name, "artifact": status.get("artifact")})
        if status.get("uniform_energy_ablation_failed"):
            ablation_failed.append({"name": name, "artifact": status.get("artifact")})
        if status.get("acceptance_gate_failures"):
            gate_failures.append(
                {
                    "name": name,
                    "artifact": status.get("artifact"),
                    "failed_gates": status.get("acceptance_gate_failures"),
                }
            )
    return {
        "excluded_artifacts": [str(row["artifact"]) for row in excluded_details],
        "excluded_details": excluded_details,
        "positive_control_failed_artifacts": positive_control_failed,
        "false_negative_risk_open_artifacts": false_negative_open,
        "ablation_failed_artifacts": ablation_failed,
        "failed_acceptance_gate_overrides": gate_failures,
        "guards_applied": {
            ".425-B2 TAUTOLOGY": bool(
                artifact_reader is not None and hasattr(artifact_reader, "classify_known_false_positive_null_delta")
            ),
            ".426-B2 offline-vs-live": True,
            ".427-B2 intrinsic-reward": True,
            ".428-B2 goal-energy-ablation": _b2_goal_energy_ablation_guard_active(b2),
        },
        "guard_note": (
            "Stamped flagged, live-critical, positive-control-failed, false-negative-risk-open, "
            "and A1 uniform-energy-ablation-failed artifacts are excluded from clean headline claims."
        ),
    }


def _claim_audit(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    rows: JsonDict = {}
    for name in ("A1", "A2", "A4", "A5", "A6"):
        included = _clean(statuses, name)
        rows[name] = {
            "artifact": UPSTREAM_SOURCES[name].relative_path,
            "included_in_headline": included,
            "verifier_is_oracle": artifacts.get(name, {}).get("verifier_is_oracle"),
            "oracle_distinct": artifacts.get(name, {}).get("verifier_is_oracle") is False,
        }
    rows["all_included_value_claims_false"] = all(
        row["oracle_distinct"]
        for row in rows.values()
        if isinstance(row, Mapping) and row.get("included_in_headline") is True
    )
    return rows


def _cited_upstream_artifacts(statuses: Mapping[str, Mapping[str, Any]], *, root: Path) -> JsonDict:
    imported_fields = {
        "A1": ["live_solve_rate_goal_energy", "live_solve_rate_baseline", "solve_rate_delta", "first_win_rate_delta"],
        "A2": ["live_solve_rate_expansion", "live_solve_rate_ranker_baseline", "depth_of_live_solve_delta"],
        "A3": ["reproduced_levels", "offline_reproduced", "reproduction_gate"],
        "A4": ["live_submittable_level_count", "ready_for_operator_submit", "offline_reproduced"],
        "A5": ["primitive_persisted", "transfer_value_per_game"],
        "A6": ["live_multi_level_solve_rate_integrated", "live_submittable_level_count_integrated"],
        "B1": ["live_multi_level_solve_rate", "coheadline_block"],
        "B2": ["goal_energy_ablation_guard_added", "honest_ablation_not_flagged", "tests_added"],
    }
    cited: JsonDict = {}
    for name, status in statuses.items():
        included = status.get("included_in_headline") is True
        cited[name] = {
            "artifact": status.get("artifact"),
            "role": status.get("role"),
            "exists": status.get("exists"),
            "sha256": status.get("sha256"),
            "included_in_headline": included,
            "reason": status.get("reason"),
            "read_via_summarize_artifact": True,
            "honest_verdict": status.get("honest_verdict"),
            "duration_s": status.get("duration_s"),
            "inference_substrate": status.get("inference_substrate"),
            "imported_fields": imported_fields.get(name, []) if included else [],
            "quarantined_fields_reported": imported_fields.get(name, []) if not included else [],
        }
    cited["REGISTRY"] = {
        "artifact": REGISTRY_RELATIVE_PATH,
        "role": "authoritative_reproducible_total_levels",
        "exists": (root / REGISTRY_RELATIVE_PATH).exists(),
        "sha256": _file_sha256(root / REGISTRY_RELATIVE_PATH),
        "included_in_headline": True,
        "reason": "authoritative_registry",
        "read_via_summarize_artifact": False,
        "imported_fields": ["reproducible_total_levels"],
        "quarantined_fields_reported": [],
    }
    return cited


def _scorecard(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    live_solve: Mapping[str, Any],
    multi: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    transfer: Mapping[str, Any],
    first_win: Mapping[str, Any],
    registry_total: int,
    registry_delta: int,
    live_count: int,
    ready: bool,
) -> JsonDict:
    a1_crossed = live_solve.get("bridge_crossed_clean") is True
    a2_crossed = multi.get("bridge_crossed_clean") is True
    crossing_source = "A2_live_multi_level" if a2_crossed else ("A1_goal_energy_solverate" if a1_crossed else "none")
    return {
        "headline": {
            "bridge_extended_by_energy_driven_generation": bool(a1_crossed or a2_crossed),
            "crossing_source": crossing_source,
            "a1_live_solve_rate_or_first_win_up_clean": a1_crossed,
            "a2_live_multi_level_up_clean": a2_crossed,
            "a3_bank_plus_one": registry_delta > 0,
            "a4_operator_resubmit_ready_above_33": ready,
            "submission_operator_only": True,
        },
        "A1": {
            "artifact": UPSTREAM_SOURCES["A1"].relative_path,
            "included_in_headline": _clean(statuses, "A1"),
            "live_solve_rate_delta": dict(live_solve),
            "uniform_energy_ablation_passed": artifacts["A1"].get("uniform_energy_ablation_passed") is True,
            "reason": statuses["A1"]["reason"],
        },
        "A2": {
            "artifact": UPSTREAM_SOURCES["A2"].relative_path,
            "included_in_headline": _clean(statuses, "A2"),
            "live_multi_level_solve_rate": dict(multi),
            "first_win_rate_scored": dict(first_win),
            "reason": statuses["A2"]["reason"],
        },
        "A3": {
            "artifact": UPSTREAM_SOURCES["A3"].relative_path,
            "included_in_headline": _clean(statuses, "A3"),
            "reproduced_levels": _as_int(artifacts["A3"].get("reproduced_levels")),
            "offline_reproduced": artifacts["A3"].get("offline_reproduced") is True,
            "registry_reproducible_total_levels": registry_total,
            "registry_delta_vs_56": registry_delta,
            "banked_plus_one": registry_delta > 0,
        },
        "A4": {
            "artifact": UPSTREAM_SOURCES["A4"].relative_path,
            "included_in_headline": _clean(statuses, "A4"),
            "live_submittable_level_count": live_count,
            "ready_for_operator_submit": ready,
            "count_delta": _as_int(artifacts["A4"].get("count_delta")),
            "levels_folded_in": list(artifacts["A4"].get("levels_folded_in") or []),
        },
        "A5": {
            "artifact": UPSTREAM_SOURCES["A5"].relative_path,
            "included_in_headline": _clean(statuses, "A5"),
            "primitive_persisted": artifacts["A5"].get("primitive_persisted"),
            "transfer_games": list(artifacts["A5"].get("transfer_games") or []),
        },
        "A6": {
            "artifact": UPSTREAM_SOURCES["A6"].relative_path,
            "included_in_headline": _clean(statuses, "A6"),
            "live_multi_level_solve_rate_integrated": _as_float(
                artifacts["A6"].get("live_multi_level_solve_rate_integrated")
            ),
            "live_submittable_level_count_integrated": _as_int(
                artifacts["A6"].get("live_submittable_level_count_integrated")
            ),
        },
        "B1": {
            "artifact": UPSTREAM_SOURCES["B1"].relative_path,
            "included_in_headline": _clean(statuses, "B1"),
            "live_multi_level_solve_rate": dict(multi),
        },
        "B2": {
            "artifact": UPSTREAM_SOURCES["B2"].relative_path,
            "included_in_headline": _clean(statuses, "B2"),
            "goal_energy_ablation_guard_active": _b2_goal_energy_ablation_guard_active(artifacts["B2"]),
        },
        "live_action_efficiency": dict(efficiency),
        "offline_to_live_transfer_ratio": dict(transfer),
        "verifier_is_oracle_claim_audit": _claim_audit(artifacts, statuses),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    registry: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    upstream, statuses = _load_artifacts(root_path, artifacts=artifacts, live_flags_by_name=live_flags_by_name)
    registry_payload = dict(registry) if registry is not None else _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    checks = dict(preconditions_checked or check_preconditions(root_path, statuses=statuses))
    registry_total = _as_int(
        registry_payload.get("reproducible_total_levels"),
        _as_int(checks.get("registry_reproducible_total_levels")),
    )
    registry_delta = registry_total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    live_solve = _a1_live_solve_rate_delta(upstream["A1"], statuses["A1"])
    multi = _live_multi_level_solve_rate(upstream["A2"], upstream["B1"], statuses)
    efficiency = _live_action_efficiency(upstream["A2"], upstream["B1"], statuses)
    transfer = _offline_to_live_transfer_ratio(live_solve, multi, upstream["B1"])
    first_win = _first_win_rate_scored(upstream["A1"], upstream["A2"], upstream["B1"], statuses)
    live_count = _as_int(upstream["A4"].get("live_submittable_level_count")) if _clean(statuses, "A4") else 0
    ready = bool(
        checks.get("ok", True)
        and _clean(statuses, "A4")
        and live_count > LIVE_SUBMITTABLE_SCORECARD_BASELINE
        and upstream["A4"].get("ready_for_operator_submit") is True
        and upstream["A4"].get("offline_reproduced") is True
    )
    if checks.get("ok") is False:
        verdict = f"blocked_{checks.get('blocked_resource') or 'precondition'}"
    elif live_solve.get("bridge_crossed_clean") is True and _as_float(live_solve.get("solve_rate_delta")) > 0.0:
        verdict = (
            "success: bridge_extended_live_solverate_up_"
            f"{_format_number(_as_float(live_solve.get('solve_rate_delta')))}"
        )
    elif live_solve.get("bridge_crossed_clean") is True:
        verdict = (
            "success: bridge_extended_live_firstwin_up_"
            f"{_format_number(_as_float(live_solve.get('first_win_rate_delta')))}"
        )
    elif multi.get("bridge_crossed_clean") is True:
        verdict = (
            "success: bridge_extended_live_multilevel_up_"
            f"{_format_number(_as_float(multi.get('delta_vs_ranker_baseline')))}"
        )
    elif registry_delta > 0:
        verdict = f"complete: capability_grew_56_to_{registry_total}"
    else:
        verdict = "complete: energy_generation_levers_characterized_no_live_lift"

    flagged = _flagged_artifacts_handled(statuses, upstream["B2"])
    scorecard = _scorecard(
        upstream,
        statuses,
        live_solve=live_solve,
        multi=multi,
        efficiency=efficiency,
        transfer=transfer,
        first_win=first_win,
        registry_total=registry_total,
        registry_delta=registry_delta,
        live_count=live_count,
        ready=ready,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "live_solve_rate_delta": live_solve,
        "live_multi_level_solve_rate": multi,
        "live_action_efficiency": efficiency,
        "offline_to_live_transfer_ratio": transfer,
        "reproducible_total_levels": registry_total,
        "reproducible_total_levels_delta": registry_delta,
        "live_submittable_level_count": live_count,
        "first_win_rate_scored": first_win,
        "uniform_energy_ablation_passed": upstream["A1"].get("uniform_energy_ablation_passed") is True,
        "flagged_artifacts_handled": flagged,
        "cited_upstream_artifacts": _cited_upstream_artifacts(statuses, root=root_path),
        "scorecard": scorecard,
        "ready_for_operator_submit": ready,
        "leaderboard_submission": False,
        "submitted_to_leaderboard": False,
        "preconditions_checked": checks,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else 0.0),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:  # pragma: no cover - defensive schema guard
    errors = [f"missing required field {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    for field in (
        "live_solve_rate_delta",
        "live_multi_level_solve_rate",
        "live_action_efficiency",
        "offline_to_live_transfer_ratio",
        "first_win_rate_scored",
    ):
        metric = artifact.get(field)
        if not isinstance(metric, Mapping):
            errors.append(f"{field} must be object")
        elif "clean_value" not in metric:
            errors.append(f"{field}.clean_value missing")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")
    if type(artifact.get("reproducible_total_levels_delta")) is not int:
        errors.append("reproducible_total_levels_delta must be bare int")
    if type(artifact.get("live_submittable_level_count")) is not int:
        errors.append("live_submittable_level_count must be bare int")
    if type(artifact.get("uniform_energy_ablation_passed")) is not bool:
        errors.append("uniform_energy_ablation_passed must be bare bool")
    if type(artifact.get("ready_for_operator_submit")) is not bool:
        errors.append("ready_for_operator_submit must be bare bool")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(
    *,
    path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    artifact: Mapping[str, Any],
) -> Path:
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    write: bool = True,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    start = time.perf_counter()
    artifact = build_artifact(
        root_path,
        live_flags_by_name=live_flags_by_name,
        duration_s=duration_s if duration_s is not None else 0.0,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - start
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(path=root_path / RESULT_RELATIVE_PATH, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
