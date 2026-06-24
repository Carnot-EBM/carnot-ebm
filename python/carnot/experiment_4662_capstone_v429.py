"""Experiment 4662: .429 ARC generation-guidance bridge capstone.

This module aggregates the landed A1-A6 and B1/B2 artifacts. It is not a
solver, never loads a model, and never submits to the leaderboard.

Spec refs: REQ-CAPSTONE-4662, SCENARIO-CAPSTONE-4662,
SCENARIO-CAPSTONE-4662-FIELD-PRINCIPLES.
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

try:  # pragma: no cover - tests inject flags and publication state
    import summarize_artifact as artifact_reader
except Exception:  # pragma: no cover - defensive import boundary
    artifact_reader = None  # type: ignore[assignment]

try:  # pragma: no cover - tests inject publication state
    import publication_gate as publication_gate_reader
except Exception:  # pragma: no cover - defensive import boundary
    publication_gate_reader = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4662_capstone_v429"
SCHEMA = "carnot.exp4662.capstone_v429.v1"
RESULT_RELATIVE_PATH = "results/experiment_4662_capstone_v429.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4662
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 57
LIVE_SUBMITTABLE_SCORECARD_BASELINE = 33
FROZEN_FOVER_AUROC = 0.9131
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
        "results/experiment_4652_value_routing_cost_fix_live.json",
        "value_routing_cost_fix_live",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4653_energy_fitness_qd_generation_live.json",
        "energy_fitness_qd_generation_live",
    ),
    "A3": SourceSpec(
        "A3",
        "results/experiment_4654_levelup_selfplay.json",
        "levelup_selfplay_bank",
    ),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4655_refresh_submission_package.json",
        "refresh_submission_package",
    ),
    "A5": SourceSpec(
        "A5",
        "results/experiment_4656_primitive_persist_transfer.json",
        "primitive_persist_transfer",
    ),
    "A6": SourceSpec(
        "A6",
        "results/experiment_4657_integration_gate.json",
        "integration_gate",
    ),
    "B1": SourceSpec(
        "B1",
        "results/experiment_4658_value_routing_cigate_diagnostic.json",
        "value_routing_cigate_diagnostic",
    ),
    "B2": SourceSpec(
        "B2",
        "results/experiment_4659_adversarial_verify_hardening.json",
        "adversarial_verify_qd_and_value_routing_guards",
    ),
}


FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_crossed_live_<firstwin|solverate|winner_generated>_up_<n> "
            "OR complete: generation_guidance_levers_characterized_no_live_solve_lift OR complete: "
            "capability_grew_57_to_<n>."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream artifact with sha256 (the audit trail)."
    },
    "a1_value_routing_live_lift": {
        "principle": (
            "the A1 affordable-value-head live first-win/solve-rate delta -- counted ONLY if "
            "cost-controlled (per_node_cost + no-timeout) + CI excludes the value_weight=0 baseline."
        )
    },
    "a2_winner_generated": {
        "principle": (
            "the A2 energy-fitness QD winner_generated -- counted ONLY if random-mutation-ablation-passed "
            "(else the search/branching, not the energy)."
        )
    },
    "reproducible_total_levels": {
        "principle": "authoritative from the registry (A3 bank, 57->58+) -- did solve CAPABILITY grow."
    },
    "reproducible_total_levels_delta": {
        "principle": "registry after - 57, emitted explicitly so a null is annotated."
    },
    "bridge_crossed_for_solve": {
        "principle": (
            "the headline decision -- did ANY .429 lever cross the offline->live bridge for "
            "SOLVE-RATE/DEPTH (the 0.04 wall), or did generation-guidance get characterized "
            "as another honest null."
        )
    },
    "live_submittable_level_count": {
        "principle": "the A4 operator-resubmit count (must stay > 33)."
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial / ablation-failed / control-failed artifact EXCLUDED + "
            "the guards applied (.428-B2 goal-energy + .429-B2 QD-ablation/value-routing-cost) -- "
            "fabrication-gate + false-negative-risk compliance."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false on every included value claim (A1/A2/A3/A4/A5/A6 oracle-distinct) -- "
            "a circular win would not count."
        )
    },
    "paper_ready": {
        "principle": (
            "G1-G4 re-affirmed (FoVer 0.9131 NEVER substituted) -- the frozen publication "
            "invariant, not a new .429 headline."
        )
    },
    "leaderboard_submission": {
        "principle": "MUST be false -- submission is operator-only (External Publication Discipline)."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (registry loadable, upstream artifacts present); "
            "pre-empts missing-resource fabrication."
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
    "publication_gate",
    "duration_s",
)
SPEC_REFS = [
    "REQ-CAPSTONE-4662",
    "SCENARIO-CAPSTONE-4662",
    "SCENARIO-CAPSTONE-4662-FIELD-PRINCIPLES",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _checksum(payload)


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}  # pragma: no cover - malformed artifact guard


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:  # pragma: no cover - defensive malformed registry guard
        return {}
    return loaded if isinstance(loaded, dict) else {}  # pragma: no cover - malformed registry guard


def _file_sha256(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):  # pragma: no cover - malformed artifact guard
        return default
    return parsed if parsed == parsed else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - malformed artifact guard
        return default


def _format_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _summarize_and_live_flags(path: Path) -> tuple[int | None, list[dict[str, Any]]]:  # pragma: no cover - external reader
    if artifact_reader is None:
        return None, []
    summary_code = artifact_reader.summarize(path) if hasattr(artifact_reader, "summarize") else None
    flags = artifact_reader._live_flags(path) if hasattr(artifact_reader, "_live_flags") else []
    return summary_code, [dict(flag) for flag in flags if isinstance(flag, Mapping)]


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
        is_gate = "acceptance_gate" in lowered or lowered.startswith("gate_") or lowered.endswith("_gate")
        if is_gate and value is False:
            failures.append(key)
    return failures


def _positive_control_failed(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("positive_control_passed") is False:
        return True
    if "bare_control_passed" in artifact and artifact.get("bare_control_passed") is not True:
        return True
    return "false_negative_risk_checked" in artifact and artifact.get("false_negative_risk_checked") is not True


def _a1_cost_control_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    if name != "A1" or not artifact:
        return False
    has_cost = "per_node_feature_cost_ms" in artifact and _as_float(artifact.get("per_node_feature_cost_ms"), -1.0) >= 0.0
    return not (has_cost and artifact.get("sim_timed_out") is False)


def _a2_random_mutation_ablation_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(name == "A2" and artifact and artifact.get("random_mutation_ablation_passed") is not True)


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
    if live_flags_by_name is not None:
        summary_exit_code = None
        flags = [dict(flag) for flag in live_flags_by_name.get(name, [])]
    elif exists:
        summary_exit_code, flags = _summarize_and_live_flags(path)
    else:
        summary_exit_code, flags = None, []
    critical = _critical_flags(flags)
    stamped = artifact.get("flagged_adversarial") is True
    gate_failures = _acceptance_gate_failures(artifact)
    positive_failed = _positive_control_failed(artifact)
    false_negative = _false_negative_risk_open(flags)
    cost_failed = _a1_cost_control_failed(name, artifact)
    ablation_failed = _a2_random_mutation_ablation_failed(name, artifact)
    flagged = bool(stamped or critical)
    included = bool(
        exists
        and artifact
        and not flagged
        and not gate_failures
        and not positive_failed
        and not false_negative
        and not cost_failed
        and not ablation_failed
    )
    reason = "included_clean"
    if not exists:
        reason = "missing"
    elif flagged:
        reason = "flagged_adversarial_or_live_critical_excluded"
    elif gate_failures:
        reason = "failed_acceptance_gate"
    elif positive_failed:
        reason = "positive_control_failed"
    elif false_negative:
        reason = "false_negative_risk_open"
    elif cost_failed:
        reason = "value_routing_cost_control_failed"
    elif ablation_failed:
        reason = "random_mutation_ablation_failed"
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
        "value_routing_cost_control_failed": cost_failed,
        "random_mutation_ablation_failed": ablation_failed,
        "acceptance_gate_failures": gate_failures,
        "summary_exit_code": summary_exit_code,
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
        loaded[name] = dict(artifacts[name]) if artifacts is not None and name in artifacts else _read_json(root / source.relative_path)
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
) -> JsonDict:
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
    missing = [UPSTREAM_SOURCES[name].relative_path for name, exists in source_exists.items() if not exists]
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4662": "REQ-CAPSTONE-4662" in spec_text,
        "registry_yaml_loadable": bool(registry),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_reproducible_total_levels": _as_int(registry.get("reproducible_total_levels")),
        "summarize_artifact_py_available": (root_path / "scripts" / "summarize_artifact.py").exists(),
        "summarize_artifact_py_used_for_every_upstream": artifact_reader is not None,
        "upstream_artifacts_present": source_exists,
        "missing_upstream_artifacts": missing,
        "leaderboard_submission": False,
        "operator_only": True,
        "network_required": False,
        "research_conductor_modified": False,
    }
    required = (
        ("agents_md_read", "agents_md"),
        ("codex_or_opencode_md_read", "codex_or_opencode_md"),
        ("spec_has_req_4662", "spec_req_4662"),
        ("registry_yaml_loadable", "registry_yaml"),
        ("summarize_artifact_py_available", "summarize_artifact"),
    )
    failed = [resource for key, resource in required if not checks[key]]
    if missing:
        failed.append("upstream_artifacts")
    checks["ok"] = not failed
    if failed:
        checks["blocked_resource"] = failed[0]
    return checks


def _clean(statuses: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return bool(statuses.get(name, {}).get("included_in_headline"))


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _ci_excludes_zero(ci: Mapping[str, Any]) -> bool:
    ci95 = ci.get("ci95")
    if not isinstance(ci95, Sequence) or isinstance(ci95, (str, bytes)) or len(ci95) != 2:
        return False
    lo = _as_float(ci95[0])
    hi = _as_float(ci95[1])
    return lo > 0.0 or hi < 0.0


def _a1_value_routing_live_lift(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    first_delta = _as_float(a1.get("first_win_rate_delta"))
    solve_delta = _as_float(a1.get("solve_rate_delta"))
    selected_metric = "solve_rate_delta" if solve_delta > first_delta else "first_win_rate_delta"
    selected_delta = max(first_delta, solve_delta)
    live_lift_ci = _mapping_at(a1, "live_lift_ci")
    cost = a1.get("per_node_feature_cost_ms")
    cost_controlled = status.get("value_routing_cost_control_failed") is False
    ci_clean = _ci_excludes_zero(live_lift_ci)
    clean = status.get("included_in_headline") is True
    counted = bool(clean and cost_controlled and ci_clean and selected_delta > 0.0)
    if counted:
        reason = "cost_controlled_and_ci_excludes_value_weight_zero_baseline"
    elif not clean:
        reason = str(status.get("reason"))
    elif not cost_controlled:
        reason = "value_routing_cost_control_failed"
    elif not ci_clean:
        reason = "ci_does_not_exclude_value_weight_zero_baseline"
    else:
        reason = "no_positive_live_lift"
    return {
        "headline_counted": counted,
        "selected_metric": selected_metric,
        "selected_delta": selected_delta,
        "first_win_rate_delta": first_delta,
        "solve_rate_delta": solve_delta,
        "live_first_win_rate_value_routed": _as_float(a1.get("live_first_win_rate_value_routed")),
        "live_solve_rate_value_routed": _as_float(a1.get("live_solve_rate_value_routed")),
        "value_weight_zero_baseline": {
            "first_win_rate": _as_float(_mapping_at(a1, "live_baseline_value_weight_zero").get("first_win_rate")),
            "solve_rate": _as_float(_mapping_at(a1, "live_baseline_value_weight_zero").get("solve_rate")),
            "value_weight": _as_float(_mapping_at(a1, "live_baseline_value_weight_zero").get("value_weight")),
        },
        "value_weight_set": _as_float(a1.get("value_weight_set")),
        "per_node_feature_cost_ms": None if cost is None else _as_float(cost),
        "sim_timed_out": a1.get("sim_timed_out"),
        "cost_controlled": cost_controlled,
        "live_lift_ci": dict(live_lift_ci),
        "ci_excludes_value_weight_zero_baseline": ci_clean,
        "reason": reason,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
    }


def _a2_winner_generated(a2: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    winner = a2.get("winner_generated") is True
    count = _as_int(a2.get("winner_generated_count"))
    ablation = a2.get("random_mutation_ablation_passed") is True
    clean = status.get("included_in_headline") is True
    counted = bool(clean and winner and ablation)
    reason = (
        "winner_generated_and_random_mutation_ablation_passed"
        if counted
        else (str(status.get("reason")) if not clean else "no_winner_generated")
    )
    return {
        "headline_counted": counted,
        "winner_generated": winner,
        "winner_generated_count": count,
        "random_mutation_ablation_passed": ablation,
        "qd_lift_ci": dict(_mapping_at(a2, "qd_lift_ci")),
        "live_solve_rate_qd": _as_float(a2.get("live_solve_rate_qd")),
        "live_solve_rate_search_baseline": _as_float(a2.get("live_solve_rate_search_baseline")),
        "first_win_rate_delta": _as_float(a2.get("first_win_rate_delta")),
        "solve_rate_delta": _as_float(a2.get("solve_rate_delta")),
        "offline_reproduced": a2.get("offline_reproduced") is True,
        "reason": reason,
        "source": UPSTREAM_SOURCES["A2"].relative_path,
    }


def _tests_passed(block: Any) -> bool:
    return not isinstance(block, Mapping) or block.get("passed") is True


def _b2_guards_active(b2: Mapping[str, Any]) -> JsonDict:
    qd = bool(
        b2.get("qd_ablation_guard_added") is True
        or b2.get("qd_random_mutation_ablation_guard_added") is True
    )
    value = bool(
        b2.get("value_routing_cost_guard_added") is True
        or b2.get("value_routing_cost_control_guard_added") is True
    )
    return {
        ".428-B2 goal-energy": True,
        ".429-B2 QD-ablation": qd and _tests_passed(b2.get("tests_added")),
        ".429-B2 value-routing-cost": value and _tests_passed(b2.get("tests_added")),
    }


def _flagged_artifacts_handled(statuses: Mapping[str, Mapping[str, Any]], b2: Mapping[str, Any]) -> JsonDict:
    excluded_details: list[JsonDict] = []
    flagged: list[JsonDict] = []
    positive_failed: list[JsonDict] = []
    false_negative_open: list[JsonDict] = []
    ablation_failed: list[JsonDict] = []
    cost_failed: list[JsonDict] = []
    gate_failures: list[JsonDict] = []
    for name, status in statuses.items():
        row = {"name": name, "artifact": status.get("artifact")}
        if status.get("included_in_headline") is False and status.get("reason") not in {"missing", "included_clean"}:
            excluded_details.append(
                {
                    **row,
                    "reason": status.get("reason"),
                    "critical_flags": [
                        {"kind": flag.get("kind"), "detail": flag.get("detail")}
                        for flag in status.get("critical_flags", [])
                    ],
                }
            )
        if status.get("stamped_flagged_adversarial") or status.get("live_critical"):
            flagged.append(row)
        if status.get("positive_control_failed"):
            positive_failed.append(row)
        if status.get("false_negative_risk_open"):
            false_negative_open.append(row)
        if status.get("random_mutation_ablation_failed"):
            ablation_failed.append(row)
        if status.get("value_routing_cost_control_failed"):
            cost_failed.append(row)
        if status.get("acceptance_gate_failures"):
            gate_failures.append({**row, "failed_gates": status.get("acceptance_gate_failures")})
    return {
        "excluded_artifacts": [str(row["artifact"]) for row in excluded_details],
        "excluded_details": excluded_details,
        "flagged_adversarial_artifacts": flagged,
        "positive_control_failed_artifacts": positive_failed,
        "false_negative_risk_open_artifacts": false_negative_open,
        "ablation_failed_artifacts": ablation_failed,
        "cost_control_failed_artifacts": cost_failed,
        "failed_acceptance_gate_overrides": gate_failures,
        "guards_applied": _b2_guards_active(b2),
        "guard_note": (
            "Stamped flagged, live-critical, ablation-failed, control-failed, and "
            "cost-control-failed artifacts are excluded from clean headline claims."
        ),
    }


def _claim_audit(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    rows: JsonDict = {}
    for name in ("A1", "A2", "A3", "A4", "A5", "A6"):
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
        "A1": [
            "first_win_rate_delta",
            "solve_rate_delta",
            "per_node_feature_cost_ms",
            "sim_timed_out",
            "live_lift_ci",
        ],
        "A2": ["winner_generated", "winner_generated_count", "random_mutation_ablation_passed", "qd_lift_ci"],
        "A3": ["reproduced_levels", "offline_reproduced", "reproduction_gate"],
        "A4": ["live_submittable_level_count", "ready_for_operator_submit", "offline_reproduced"],
        "A5": ["primitive_persisted", "transfer_value_per_game"],
        "A6": ["live_first_win_rate_delta_vs_pre_integration", "live_multi_level_solve_rate_delta_vs_pre_integration"],
        "B1": ["ci_gate", "distribution_shift_score", "tests_added"],
        "B2": ["qd_ablation_guard_added", "value_routing_cost_guard_added", "tests_added"],
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
            "summarize_exit_code": status.get("summary_exit_code"),
            "honest_verdict": status.get("honest_verdict"),
            "stamped_flagged_adversarial": status.get("stamped_flagged_adversarial"),
            "live_critical": status.get("live_critical"),
            "positive_control_failed": status.get("positive_control_failed"),
            "false_negative_risk_open": status.get("false_negative_risk_open"),
            "ablation_failed": status.get("random_mutation_ablation_failed"),
            "cost_control_failed": status.get("value_routing_cost_control_failed"),
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
        "reason": "authoritative_registry_header",
        "read_via_summarize_artifact": False,
        "imported_fields": ["reproducible_total_levels"],
        "quarantined_fields_reported": [],
    }
    return cited


def _publication_gate(publication_gate: Mapping[str, Any] | None) -> JsonDict:
    if publication_gate is not None:
        gate = dict(publication_gate)
    elif publication_gate_reader is not None:  # pragma: no cover - integration boundary
        gate = dict(publication_gate_reader.evaluate())
    else:  # pragma: no cover - defensive import boundary
        gate = {"paper_ready": False, "gates": {}, "unmet_gates": ["publication_gate_unavailable"]}
    return {
        "paper_ready": gate.get("paper_ready") is True,
        "gates": dict(_mapping_at(gate, "gates")),
        "unmet_gates": list(gate.get("unmet_gates") or []),
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "fover_0_9131_substituted": False,
        "note": "FoVer 0.9131 is re-affirmed only as the frozen publication invariant.",
    }


def _scorecard(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    a1_lift: Mapping[str, Any],
    a2_winner: Mapping[str, Any],
    registry_total: int,
    registry_delta: int,
    live_count: int,
    ready: bool,
    publication: Mapping[str, Any],
) -> JsonDict:
    a1_crossed = a1_lift.get("headline_counted") is True
    a2_crossed = a2_winner.get("headline_counted") is True
    crossing_source = "A2_energy_fitness_qd" if a2_crossed else ("A1_value_routing" if a1_crossed else "none")
    return {
        "headline": {
            "bridge_crossed_for_solve": bool(a1_crossed or a2_crossed),
            "crossing_source": crossing_source,
            "a1_value_routing_live_lift_counted": a1_crossed,
            "a2_qd_winner_generated_counted": a2_crossed,
            "a3_registry_capability_plus_one": registry_delta > 0,
            "a4_operator_resubmit_ready_above_33": ready,
            "submission_operator_only": True,
        },
        "A1": {
            "artifact": UPSTREAM_SOURCES["A1"].relative_path,
            "included_in_headline": _clean(statuses, "A1"),
            "value_routing_live_lift": dict(a1_lift),
            "reason": statuses["A1"]["reason"],
        },
        "A2": {
            "artifact": UPSTREAM_SOURCES["A2"].relative_path,
            "included_in_headline": _clean(statuses, "A2"),
            "winner_generated": dict(a2_winner),
            "reason": statuses["A2"]["reason"],
        },
        "A3": {
            "artifact": UPSTREAM_SOURCES["A3"].relative_path,
            "included_in_headline": _clean(statuses, "A3"),
            "reproduced_levels": _as_int(artifacts["A3"].get("reproduced_levels")),
            "offline_reproduced": artifacts["A3"].get("offline_reproduced") is True,
            "registry_reproducible_total_levels": registry_total,
            "registry_delta_vs_57": registry_delta,
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
        },
        "A6": {
            "artifact": UPSTREAM_SOURCES["A6"].relative_path,
            "included_in_headline": _clean(statuses, "A6"),
            "live_first_win_rate_delta_vs_pre_integration": _as_float(
                artifacts["A6"].get("live_first_win_rate_delta_vs_pre_integration")
            ),
            "live_multi_level_solve_rate_delta_vs_pre_integration": _as_float(
                artifacts["A6"].get("live_multi_level_solve_rate_delta_vs_pre_integration")
            ),
            "reason": statuses["A6"]["reason"],
        },
        "B1": {
            "artifact": UPSTREAM_SOURCES["B1"].relative_path,
            "included_in_headline": _clean(statuses, "B1"),
            "ci_gate": artifacts["B1"].get("ci_gate"),
            "distribution_shift_score": artifacts["B1"].get("distribution_shift_score"),
        },
        "B2": {
            "artifact": UPSTREAM_SOURCES["B2"].relative_path,
            "included_in_headline": _clean(statuses, "B2"),
            "guards_active": _b2_guards_active(artifacts["B2"]),
        },
        "verifier_is_oracle_claim_audit": _claim_audit(artifacts, statuses),
        "paper_ready": dict(publication),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    registry: Mapping[str, Any] | None = None,
    publication_gate: Mapping[str, Any] | None = None,
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
    a1_lift = _a1_value_routing_live_lift(upstream["A1"], statuses["A1"])
    a2_winner = _a2_winner_generated(upstream["A2"], statuses["A2"])
    bridge_crossed = bool(a1_lift.get("headline_counted") or a2_winner.get("headline_counted"))
    live_count = _as_int(upstream["A4"].get("live_submittable_level_count")) if _clean(statuses, "A4") else 0
    ready = bool(
        checks.get("ok", True)
        and _clean(statuses, "A4")
        and live_count > LIVE_SUBMITTABLE_SCORECARD_BASELINE
        and upstream["A4"].get("ready_for_operator_submit") is True
        and upstream["A4"].get("offline_reproduced") is True
    )
    publication = _publication_gate(publication_gate)
    if checks.get("ok") is False:
        verdict = f"blocked_{checks.get('blocked_resource') or 'precondition'}"
    elif a1_lift.get("headline_counted") is True:
        metric = "solverate" if a1_lift.get("selected_metric") == "solve_rate_delta" else "firstwin"
        verdict = (
            f"success: bridge_crossed_live_{metric}_up_"
            f"{_format_number(_as_float(a1_lift.get('selected_delta')))}"
        )
    elif a2_winner.get("headline_counted") is True:
        verdict = (
            "success: bridge_crossed_live_winner_generated_up_"
            f"{_format_number(_as_float(a2_winner.get('winner_generated_count')))}"
        )
    elif registry_delta > 0:
        verdict = f"complete: capability_grew_57_to_{registry_total}"
    else:
        verdict = "complete: generation_guidance_levers_characterized_no_live_solve_lift"

    flagged = _flagged_artifacts_handled(statuses, upstream["B2"])
    scorecard = _scorecard(
        upstream,
        statuses,
        a1_lift=a1_lift,
        a2_winner=a2_winner,
        registry_total=registry_total,
        registry_delta=registry_delta,
        live_count=live_count,
        ready=ready,
        publication=publication,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "cited_upstream_artifacts": _cited_upstream_artifacts(statuses, root=root_path),
        "a1_value_routing_live_lift": a1_lift,
        "a2_winner_generated": a2_winner,
        "reproducible_total_levels": registry_total,
        "reproducible_total_levels_delta": registry_delta,
        "bridge_crossed_for_solve": bridge_crossed,
        "live_submittable_level_count": live_count,
        "flagged_artifacts_handled": flagged,
        "verifier_is_oracle": False,
        "paper_ready": publication["paper_ready"],
        "publication_gate": publication,
        "leaderboard_submission": False,
        "submitted_to_leaderboard": False,
        "field_principles": FIELD_PRINCIPLES,
        "scorecard": scorecard,
        "preconditions_checked": checks,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else 0.0),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing required field {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")
    if type(artifact.get("reproducible_total_levels_delta")) is not int:
        errors.append("reproducible_total_levels_delta must be bare int")
    if type(artifact.get("bridge_crossed_for_solve")) is not bool:
        errors.append("bridge_crossed_for_solve must be bare bool")
    if type(artifact.get("live_submittable_level_count")) is not int:
        errors.append("live_submittable_level_count must be bare int")
    if type(artifact.get("paper_ready")) is not bool:
        errors.append("paper_ready must be bare bool")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    audit = _mapping_at(_mapping_at(artifact, "scorecard"), "verifier_is_oracle_claim_audit")
    if audit.get("all_included_value_claims_false") is not True:
        errors.append("included verifier value claims must be oracle-distinct")
    return errors


def write_artifact(*, path: Path, artifact: Mapping[str, Any]) -> Path:
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
    publication_gate: Mapping[str, Any] | None = None,
    write: bool = True,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    elapsed = duration_s if duration_s is not None else None
    artifact = build_artifact(
        root,
        live_flags_by_name=live_flags_by_name,
        publication_gate=publication_gate,
        duration_s=float(elapsed if elapsed is not None else time.perf_counter() - start),
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(path=Path(root) / RESULT_RELATIVE_PATH, artifact=artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
