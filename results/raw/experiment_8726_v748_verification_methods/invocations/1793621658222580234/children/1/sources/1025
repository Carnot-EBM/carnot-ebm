"""Experiment 4626: .426 ARC offline-to-live bridge capstone.

This module is intentionally an aggregator, not a solver. It reads the already
landed A1-A6/B1/B2 artifacts, applies the same fabrication and false-negative
guards used by the artifact reader, and writes the milestone scorecard without
submitting anything to the leaderboard.

Spec refs: REQ-CAPSTONE-4626, SCENARIO-CAPSTONE-4626,
SCENARIO-CAPSTONE-4626-FIELD-PRINCIPLES.
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
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - script helper guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

try:  # pragma: no cover - exercised through injected flags in unit tests
    import summarize_artifact as artifact_reader
except Exception:  # pragma: no cover - defensive import boundary
    artifact_reader = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4626_capstone_v426"
SCHEMA = "carnot.exp4626.capstone_v426.v1"
RESULT_RELATIVE_PATH = "results/experiment_4626_capstone_v426.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4626
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 55
LIVE_SUBMITTABLE_SCORECARD_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
ALLOWED_CAUSES = {"compute_cost", "distribution_shift", "calibration", "inseparable_multi_cause", "unknown"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_path: str
    role: str


UPSTREAM_SOURCES: dict[str, SourceSpec] = {
    "A1": SourceSpec(
        "A1",
        "results/experiment_4616_offline_live_bridge_disambiguation.json",
        "offline_live_bridge_disambiguation",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4617_graduate_spatial_value_head_live.json",
        "graduate_spatial_value_head_live",
    ),
    "A3": SourceSpec(
        "A3",
        "results/experiment_4618_levelup_selfplay.json",
        "levelup_selfplay_bank",
    ),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4619_refresh_submission_package.json",
        "refresh_submission_package",
    ),
    "A5": SourceSpec(
        "A5",
        "results/experiment_4620_primitive_persist_transfer.json",
        "primitive_persist_transfer",
    ),
    "A6": SourceSpec(
        "A6",
        "results/experiment_4621_integration_gate.json",
        "integration_gate",
    ),
    "B1": SourceSpec(
        "B1",
        "results/experiment_4622_offline_to_live_transfer_ratio_metric.json",
        "offline_to_live_transfer_ratio_metric",
    ),
    "B2": SourceSpec(
        "B2",
        "results/experiment_4623_adversarial_verify_hardening.json",
        "adversarial_verify_hardening",
    ),
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_crossed_live_first_win_up_<n>_cause_"
            "<compute|shift|calibration> OR complete: bridge_characterized_cause_isolated_no_live_lift "
            "OR complete: bridge_characterized_capability_grew_55_to_<n>."
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
    "offline_to_live_transfer_ratio": {
        "principle": (
            "the HEADLINE co-metric (A1/A2/B1) -- did the offline verifier signal (0.725) "
            "transfer to a LIVE lift (the bridge crossed) or stay zero-live (bridge not crossed)."
        )
    },
    "binding_bridge_cause": {
        "principle": (
            "which of {compute_cost, distribution_shift, calibration} A1 isolated (or inseparable) "
            "-- the diagnostic deliverable."
        )
    },
    "reproducible_total_levels": {
        "principle": "authoritative from the registry (A3 bank, 55->56+) -- did solve CAPABILITY grow."
    },
    "reproducible_total_levels_delta": {
        "principle": "registry after - 55, emitted explicitly so a null is annotated."
    },
    "live_submittable_level_count": {
        "principle": "the A4 operator-resubmit count (must stay > 33)."
    },
    "first_win_rate_scored": {
        "principle": (
            "the A2 SCORED-agent live first-win-rate vs the linear baseline (did graduating the "
            "value head earn its place on the leaderboard path)."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial / positive-control-failed artifact EXCLUDED + the guards "
            "applied (.425-B2 TAUTOLOGY carve-out, .426-B2 offline-vs-live) -- fabrication-gate + "
            "false-negative-risk compliance."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream artifact with sha256 (the audit trail)."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only if the refreshed package beats 33 on a real metric worth a 1/day slot; never "
            "submits (operator-only)."
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
    "REQ-CAPSTONE-4626",
    "SCENARIO-CAPSTONE-4626",
    "SCENARIO-CAPSTONE-4626-FIELD-PRINCIPLES",
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
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> JsonDict:
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


def _tests_passed(block: Any) -> bool:
    return not isinstance(block, Mapping) or block.get("passed") is True


def _b2_overclaim_guard_active(b2: Mapping[str, Any]) -> bool:
    return bool(
        b2.get("offline_live_overclaim_guard_added") is True
        and b2.get("honest_offline_result_not_flagged") is True
        and _tests_passed(b2.get("tests_added"))
    )


def _tautology_null_delta_carveout_available() -> bool:
    return bool(
        artifact_reader is not None
        and hasattr(artifact_reader, "classify_known_false_positive_null_delta")
    )


def _tautology_carveout_flag(flag: Mapping[str, Any], *, b2_active: bool) -> bool:
    detail = str(flag.get("detail") or "").lower()
    return (
        b2_active
        and flag.get("kind") == "TAUTOLOGY"
        and _severity(flag) == "critical"
        and (
            "shared-denominator" in detail
            or "k/n" in detail
            or "declared_null_delta" in detail
            or "null-delta" in detail
            or "control-vs-treatment" in detail
        )
    )


def _critical_flags(
    flags: Sequence[Mapping[str, Any]],
    *,
    b2_active: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    critical: list[dict[str, Any]] = []
    carveouts: list[dict[str, Any]] = []
    for flag in flags:
        if _severity(flag) != "critical":
            continue
        if _tautology_carveout_flag(flag, b2_active=b2_active):
            carveouts.append(dict(flag))
        else:
            critical.append(dict(flag))
    return critical, carveouts


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
    return (
        ("positive_control_passed" in artifact and artifact.get("positive_control_passed") is not True)
        or (
            "false_negative_risk_checked" in artifact
            and artifact.get("false_negative_risk_checked") is not True
        )
    )


def _source_status(
    *,
    name: str,
    source: SourceSpec,
    root: Path,
    artifact: Mapping[str, Any],
    exists: bool,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    b2_active: bool,
) -> JsonDict:
    path = root / source.relative_path
    flags = (
        [dict(flag) for flag in live_flags_by_name[name]]
        if live_flags_by_name is not None and name in live_flags_by_name
        else (_live_flags(path) if exists else [])
    )
    critical, tautology_carveouts = _critical_flags(flags, b2_active=b2_active)
    stamped = artifact.get("flagged_adversarial") is True
    gate_failures = _acceptance_gate_failures(artifact)
    positive_failed = _positive_control_failed(artifact)
    false_negative = _false_negative_risk_open(flags)
    flagged = bool(stamped or critical)
    included = bool(
        exists
        and artifact
        and not flagged
        and not gate_failures
        and not positive_failed
        and not false_negative
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
    elif flagged:
        reason = "flagged_adversarial_or_live_critical_excluded"
    elif tautology_carveouts:
        reason = "included_clean_with_tautology_carveout"
    return {
        "name": name,
        "artifact": source.relative_path,
        "role": source.role,
        "exists": exists,
        "stamped_flagged_adversarial": stamped,
        "live_critical": bool(critical),
        "live_flags": flags,
        "critical_flags": critical,
        "tautology_carveout_flags": tautology_carveouts,
        "positive_control_failed": positive_failed,
        "false_negative_risk_open": false_negative,
        "acceptance_gate_failures": gate_failures,
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
    for name, source in UPSTREAM_SOURCES.items():
        if artifacts is not None and name in artifacts:
            loaded[name] = dict(artifacts[name])
        else:
            loaded[name] = _read_json(root / source.relative_path)
    b2_active = _tautology_null_delta_carveout_available() or _b2_overclaim_guard_active(loaded.get("B2", {}))
    statuses: dict[str, JsonDict] = {}
    for name, source in UPSTREAM_SOURCES.items():
        exists = bool(name in artifacts) if artifacts is not None else (root / source.relative_path).exists()
        statuses[name] = _source_status(
            name=name,
            source=source,
            root=root,
            artifact=loaded[name],
            exists=exists,
            live_flags_by_name=live_flags_by_name,
            b2_active=b2_active,
        )
    return loaded, statuses


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover - integration boundary covered by run smoke tests
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
        "spec_has_req_4626": "REQ-CAPSTONE-4626" in spec_text,
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
        "spec_has_req_4626",
        "registry_yaml_loadable",
        "offline_arcade",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(key for key in required if not checks[key])
    return checks


def _clean(statuses: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return bool(statuses.get(name, {}).get("included_in_headline"))


def _binding_cause(a1: Mapping[str, Any], statuses: Mapping[str, Mapping[str, Any]]) -> str:
    if not _clean(statuses, "A1"):
        return "unknown"
    cause = str(a1.get("binding_bridge_cause") or "unknown")
    return cause if cause in ALLOWED_CAUSES else "unknown"


def _live_lift_supported(a2: Mapping[str, Any]) -> bool:
    return _as_float(a2.get("first_win_delta")) > 0.0 or _as_float(a2.get("actions_delta")) > 0.0


def _offline_to_live_metric(
    b1: Mapping[str, Any],
    a2: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    b1_clean = _clean(statuses, "B1")
    a2_clean = _clean(statuses, "A2")
    offline = _as_float(b1.get("offline_auroc_component"))
    live_lift = _as_float(b1.get("live_lift_component"))
    ratio = _as_float(
        b1.get("offline_to_live_transfer_ratio"),
        live_lift / offline if offline > 0.0 else 0.0,
    )
    a2_live_claim_admissible = bool(a2_clean and _live_lift_supported(a2))
    bridge_crossed_clean = bool(
        b1_clean
        and a2_live_claim_admissible
        and b1.get("bridge_crossed") is True
        and live_lift > 0.0
    )
    clean_value = ratio if b1_clean and (bridge_crossed_clean or live_lift == 0.0) else None
    return {
        "clean_value": clean_value,
        "quarantined_value": None if clean_value is not None else ratio,
        "offline_auroc_component": offline,
        "live_lift_component": live_lift,
        "first_win_lift_component": _as_float(b1.get("first_win_lift_component")),
        "action_efficiency_lift_component": _as_float(b1.get("action_efficiency_lift_component")),
        "bridge_crossed_reported_by_b1": b1.get("bridge_crossed") is True,
        "bridge_crossed_clean": bridge_crossed_clean,
        "a2_live_claim_admissible": a2_live_claim_admissible,
        "a2_included_in_headline": a2_clean,
        "source": UPSTREAM_SOURCES["B1"].relative_path,
        "live_source": UPSTREAM_SOURCES["A2"].relative_path,
        "reason": (
            "clean_live_lift_against_linear_baseline"
            if bridge_crossed_clean
            else "zero_or_inadmissible_live_lift_no_bridge_crossing"
        ),
    }


def _first_win_metric(a2: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    clean = status.get("included_in_headline") is True
    rate = _as_float(a2.get("first_win_rate_graduated"))
    linear = _as_float(a2.get("first_win_rate_linear_baseline"))
    bare = _as_float(a2.get("first_win_rate_bare"))
    graduated_actions = _as_float(a2.get("median_actions_to_first_levelup_graduated"))
    linear_actions = _as_float(a2.get("median_actions_to_first_levelup_linear_baseline"))
    return {
        "clean_value": rate if clean else None,
        "quarantined_value": None if clean else rate,
        "linear_baseline_rate": linear,
        "bare_rate": bare,
        "delta_vs_linear_baseline": _as_float(a2.get("first_win_delta"), rate - linear),
        "ci": a2.get("first_win_ci"),
        "median_actions_to_first_levelup_graduated": graduated_actions,
        "median_actions_to_first_levelup_linear_baseline": linear_actions,
        "actions_delta": _as_float(a2.get("actions_delta"), linear_actions - graduated_actions),
        "solve_rate_graduated": _as_float(a2.get("solve_rate_graduated")),
        "solve_rate_linear_baseline": _as_float(a2.get("solve_rate_linear_baseline")),
        "headline_numbers_aggregated": clean,
        "source": UPSTREAM_SOURCES["A2"].relative_path,
        "reason": status.get("reason"),
    }


def _a5_summary(a5: Mapping[str, Any], included: bool) -> JsonDict:
    values = a5.get("transfer_value_per_game")
    games = [
        str(game)
        for game, row in (values.items() if isinstance(values, Mapping) else [])
        if isinstance(row, Mapping) and row.get("value_added") is True
    ]
    return {
        "included_in_headline": included,
        "primitive_persisted": a5.get("primitive_persisted"),
        "transfer_games": list(a5.get("transfer_games") or []),
        "value_added_games": games,
        "new_levels_banked": _as_int(a5.get("new_levels_banked")),
        "verifier_is_oracle": a5.get("verifier_is_oracle"),
    }


def _a6_summary(a6: Mapping[str, Any], included: bool, status: Mapping[str, Any]) -> JsonDict:
    return {
        "included_in_headline": included,
        "submitted_config_raised_metric_clean": bool(
            included and a6.get("submitted_config_raised_metric_clean") is True
        ),
        "offline_to_live_transfer_ratio_integrated": _as_float(
            a6.get("offline_to_live_transfer_ratio_integrated")
        ),
        "first_win_rate_delta_vs_bare": _as_float(a6.get("first_win_rate_delta_vs_bare")),
        "actions_delta_vs_bare": _as_float(a6.get("actions_delta_vs_bare")),
        "parity_test_green": a6.get("parity_test_green") is True,
        "reason": status.get("reason"),
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


def _flagged_artifacts_handled(
    statuses: Mapping[str, Mapping[str, Any]],
    b2: Mapping[str, Any],
) -> JsonDict:
    excluded_details: list[JsonDict] = []
    positive_control_failed: list[JsonDict] = []
    false_negative_open: list[JsonDict] = []
    gate_failures: list[JsonDict] = []
    tautology_carveouts: list[JsonDict] = []
    for name, status in statuses.items():
        excluded_for_guard = status.get("included_in_headline") is False and (
            status.get("stamped_flagged_adversarial")
            or status.get("live_critical")
            or status.get("positive_control_failed")
            or status.get("false_negative_risk_open")
        )
        if excluded_for_guard:
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
        if status.get("acceptance_gate_failures"):
            gate_failures.append(
                {
                    "name": name,
                    "artifact": status.get("artifact"),
                    "failed_gates": status.get("acceptance_gate_failures"),
                }
            )
        if status.get("tautology_carveout_flags"):
            tautology_carveouts.append(
                {
                    "name": name,
                    "artifact": status.get("artifact"),
                    "flags": status.get("tautology_carveout_flags"),
                }
            )
    return {
        "excluded_artifacts": [str(row["artifact"]) for row in excluded_details],
        "excluded_details": excluded_details,
        "positive_control_failed_artifacts": positive_control_failed,
        "false_negative_risk_open_artifacts": false_negative_open,
        "failed_acceptance_gate_overrides": gate_failures,
        "tautology_carveouts": tautology_carveouts,
        "tautology_null_delta_carveout_applied": _tautology_null_delta_carveout_available(),
        "offline_vs_live_overclaim_guard_active": _b2_overclaim_guard_active(b2),
        "guard_note": (
            "Stamped flagged, live-critical, positive-control-failed, and false-negative-risk-open "
            "artifacts are excluded from clean headline claims. The .426 guard prevents an offline "
            "AUROC from being reported as a LIVE win without a LIVE metric."
        ),
    }


def _cited_upstream_artifacts(
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
) -> JsonDict:
    cited: JsonDict = {}
    imported_fields = {
        "A1": ["binding_bridge_cause", "positive_control_passed", "false_negative_risk_checked"],
        "A2": ["first_win_rate_graduated", "first_win_rate_linear_baseline", "first_win_delta", "actions_delta"],
        "A3": ["reproduced_levels", "offline_reproduced", "reproduction_gate"],
        "A4": ["live_submittable_level_count", "ready_for_operator_submit", "offline_reproduced"],
        "A5": ["primitive_persisted", "transfer_value_per_game", "new_levels_banked"],
        "A6": ["offline_to_live_transfer_ratio_integrated", "submitted_config_raised_metric_clean"],
        "B1": ["offline_to_live_transfer_ratio", "offline_auroc_component", "live_lift_component"],
        "B2": ["offline_live_overclaim_guard_added", "cheap_value_substrate_floor_added", "tests_added"],
    }
    for name, status in statuses.items():
        included = status.get("included_in_headline") is True
        cited[name] = {
            "artifact": status.get("artifact"),
            "role": status.get("role"),
            "exists": status.get("exists"),
            "sha256": status.get("sha256"),
            "included_in_headline": included,
            "reason": status.get("reason"),
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
        "imported_fields": ["reproducible_total_levels"],
        "quarantined_fields_reported": [],
    }
    return cited


def _scorecard(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    cause: str,
    bridge: Mapping[str, Any],
    first_win: Mapping[str, Any],
    registry_total: int,
    registry_delta: int,
    live_count: int,
    ready: bool,
) -> JsonDict:
    return {
        "headline": {
            "cause_isolated_clean": cause != "unknown",
            "bridge_crossed_clean": bridge.get("bridge_crossed_clean") is True,
            "scored_first_win_or_efficiency_up_clean": bool(
                first_win.get("clean_value") is not None
                and (
                    _as_float(first_win.get("delta_vs_linear_baseline")) > 0.0
                    or _as_float(first_win.get("actions_delta")) > 0.0
                )
            ),
            "a3_bank_plus_one": registry_delta > 0,
            "a4_operator_resubmit_ready_above_33": ready,
        },
        "A1": {
            "artifact": UPSTREAM_SOURCES["A1"].relative_path,
            "included_in_headline": _clean(statuses, "A1"),
            "binding_bridge_cause": cause,
            "indicated_fix": artifacts["A1"].get("indicated_fix"),
            "reason": statuses["A1"]["reason"],
        },
        "A2": {
            "artifact": UPSTREAM_SOURCES["A2"].relative_path,
            "included_in_headline": _clean(statuses, "A2"),
            "first_win_rate_scored": dict(first_win),
            "reason": statuses["A2"]["reason"],
        },
        "A3": {
            "artifact": UPSTREAM_SOURCES["A3"].relative_path,
            "included_in_headline": _clean(statuses, "A3"),
            "reproduced_levels": _as_int(artifacts["A3"].get("reproduced_levels")),
            "offline_reproduced": artifacts["A3"].get("offline_reproduced") is True,
            "registry_reproducible_total_levels": registry_total,
            "registry_delta_vs_55": registry_delta,
            "banked_plus_one": registry_delta > 0,
        },
        "A4": {
            "artifact": UPSTREAM_SOURCES["A4"].relative_path,
            "included_in_headline": _clean(statuses, "A4"),
            "live_submittable_level_count": live_count,
            "ready_for_operator_submit": ready,
            "count_delta": _as_int(artifacts["A4"].get("count_delta")),
        },
        "A5": _a5_summary(artifacts["A5"], _clean(statuses, "A5")),
        "A6": _a6_summary(artifacts["A6"], _clean(statuses, "A6"), statuses["A6"]),
        "B1": {
            "artifact": UPSTREAM_SOURCES["B1"].relative_path,
            "included_in_headline": _clean(statuses, "B1"),
            "offline_to_live_transfer_ratio": dict(bridge),
            "reason": statuses["B1"]["reason"],
        },
        "B2": {
            "artifact": UPSTREAM_SOURCES["B2"].relative_path,
            "included_in_headline": _clean(statuses, "B2"),
            "offline_vs_live_overclaim_guard_active": _b2_overclaim_guard_active(artifacts["B2"]),
            "tautology_null_delta_carveout_available": _tautology_null_delta_carveout_available(),
        },
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
    upstream, statuses = _load_artifacts(
        root_path,
        artifacts=artifacts,
        live_flags_by_name=live_flags_by_name,
    )
    registry_payload = dict(registry) if registry is not None else _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    checks = dict(preconditions_checked or check_preconditions(root_path, statuses=statuses))
    registry_total = _as_int(
        registry_payload.get("reproducible_total_levels"),
        _as_int(checks.get("registry_reproducible_total_levels")),
    )
    registry_delta = registry_total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    cause = _binding_cause(upstream["A1"], statuses)
    bridge = _offline_to_live_metric(upstream["B1"], upstream["A2"], statuses)
    first_win = _first_win_metric(upstream["A2"], statuses["A2"])
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
    elif cause != "unknown" and bridge.get("bridge_crossed_clean") is True:
        verdict = (
            "success: bridge_crossed_live_first_win_up_"
            f"{_format_number(_as_float(first_win.get('delta_vs_linear_baseline')))}_cause_{cause}"
        )
    elif cause != "unknown":
        verdict = "complete: bridge_characterized_cause_isolated_no_live_lift"
    else:
        verdict = f"complete: bridge_characterized_capability_grew_55_to_{registry_total}"

    flagged = _flagged_artifacts_handled(statuses, upstream["B2"])
    scorecard = _scorecard(
        upstream,
        statuses,
        cause=cause,
        bridge=bridge,
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
        "offline_to_live_transfer_ratio": bridge,
        "binding_bridge_cause": cause,
        "reproducible_total_levels": registry_total,
        "reproducible_total_levels_delta": registry_delta,
        "live_submittable_level_count": live_count,
        "first_win_rate_scored": first_win,
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


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _metric_errors(artifact: Mapping[str, Any], field: str) -> list[str]:
    metric = artifact.get(field)
    if not isinstance(metric, Mapping):
        return [f"{field} must be object"]
    errors: list[str] = []
    if "clean_value" not in metric:
        errors.append(f"{field}.clean_value missing")
    for key in ("clean_value", "quarantined_value"):
        if key in metric and metric.get(key) is not None and not isinstance(metric.get(key), float):
            errors.append(f"{field}.{key} must be float or null")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    errors.extend(_metric_errors(artifact, "offline_to_live_transfer_ratio"))
    errors.extend(_metric_errors(artifact, "first_win_rate_scored"))
    if artifact.get("binding_bridge_cause") not in ALLOWED_CAUSES:
        errors.append("binding_bridge_cause invalid")
    if not _is_bare_int(artifact.get("reproducible_total_levels")):
        errors.append("reproducible_total_levels must be bare int")
    if not _is_bare_int(artifact.get("reproducible_total_levels_delta")):
        errors.append("reproducible_total_levels_delta must be bare int")
    if not _is_bare_int(artifact.get("live_submittable_level_count")):
        errors.append("live_submittable_level_count must be bare int")
    if not _is_bare_bool(artifact.get("ready_for_operator_submit")):
        errors.append("ready_for_operator_submit must be bare bool")
    if artifact.get("ready_for_operator_submit") is True and _as_int(
        artifact.get("live_submittable_level_count")
    ) <= LIVE_SUBMITTABLE_SCORECARD_BASELINE:
        errors.append("ready_for_operator_submit requires count above 33")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles missing")
    else:
        for field in FIELD_PRINCIPLES:
            if field not in artifact.get("field_principles", {}):
                errors.append(f"missing field principle for {field}")
    if not isinstance(artifact.get("cited_upstream_artifacts"), Mapping):
        errors.append("cited_upstream_artifacts must be object")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be object")
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
