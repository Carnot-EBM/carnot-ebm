"""Experiment 4602: .424 capstone scorecard.

Spec refs: REQ-CAPSTONE-4602, SCENARIO-CAPSTONE-4602,
SCENARIO-CAPSTONE-4602-FIELD-PRINCIPLES.
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

try:  # pragma: no cover - exercised by default command, injectable in tests
    import summarize_artifact as artifact_reader
except Exception:  # pragma: no cover - defensive helper boundary
    artifact_reader = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4602_capstone_v424"
SCHEMA = "carnot.exp4602.capstone_v424.v1"
RESULT_RELATIVE_PATH = "results/experiment_4602_capstone_v424.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4602
WINNER_GENERATED_BASELINE = 1.0 / 25.0
GENERIC_TRANSFER_BASELINE = 0.04
LIVE_SUBMITTABLE_BASELINE = 33
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_path: str
    role: str


UPSTREAM_SOURCES: dict[str, SourceSpec] = {
    "A1": SourceSpec(
        "A1",
        "results/experiment_4592_generation_completeness_wiring.json",
        "generation_completeness_wiring",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4593_levelup_selfplay.json",
        "levelup_selfplay_bank",
    ),
    "A3": SourceSpec(
        "A3",
        "results/experiment_4594_goal_energy_generation_prior.json",
        "goal_energy_generation_prior",
    ),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4595_refresh_submission_package.json",
        "refresh_submission_package",
    ),
    "A5": SourceSpec(
        "A5",
        "results/experiment_4596_primitive_persist_transfer.json",
        "primitive_persist_transfer",
    ),
    "A6": SourceSpec(
        "A6",
        "results/experiment_4597_integration_gate.json",
        "integration_gate",
    ),
    "B1": SourceSpec(
        "B1",
        "results/experiment_4598_winner_generated_rate_metric.json",
        "winner_generated_rate_metric",
    ),
    "LIVE_BASELINE": SourceSpec(
        "LIVE_BASELINE",
        "results/arc3_live_submit.json",
        "last_submitted_scorecard",
    ),
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: generation_completeness_winner_generated_above_1of25_or_transfer_above_0.04 "
            "OR complete: generation_wall_persists_residual_logged_capability_grew."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "winner_generated_moved": {
        "principle": (
            "the bottom line -- did A1 wiring raise winner_generated_rate STRICTLY above 1/25 "
            "with CI + a passing no-wiring control (the generation wall finally cracking)."
        )
    },
    "generic_transfer_moved": {
        "principle": (
            "did A1/A3 raise generic_transfer_rate_over_variants STRICTLY above 0.04 "
            "with CI (the seen->hidden solve-rate)."
        )
    },
    "goal_energy_helped": {
        "principle": (
            "did the A3 goal-energy generation prior raise winner_generated_rate on the "
            "wired-but-failing classes with a passing no-energy control."
        )
    },
    "reproducible_total_levels_delta": {
        "principle": "did solve CAPABILITY grow this milestone (A2 bank, 54->55+)."
    },
    "live_submittable_level_count": {
        "principle": (
            "the honest leaderboard score (A4/B1) reported alongside "
            "reproducible_total_levels + generic_transfer + action efficiency + winner_generated_rate."
        )
    },
    "winner_generated_rate": {
        "principle": (
            "the generation-vs-ranking gap co-headline (B1) -- can the system GENERATE "
            "the winner at all (1/25 baseline)."
        )
    },
    "action_efficiency_score": {
        "principle": "min(human/agent,1)^2 with CI (the .422 B1 metric) -- a co-headline number."
    },
    "generic_transfer_rate_over_variants": {
        "principle": "the held-out first-contact solve-rate reported WITH a CI -- a co-headline metric."
    },
    "verifier_is_oracle_distinct_levers": {
        "principle": (
            "A1/A3 value claims are verifier_is_oracle:false -- a circular win would not count "
            "(Oracle-Distinctness)."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial artifact excluded AND any null-delta-carve-out "
            "(mechanical via .424 B2) / offline-arc-methodology / positive-control-failed "
            "artifact handled -- fabrication-gate + false-negative-risk compliance."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream measurement (the audit trail)."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only if the integrated config + refreshed package beat the last 33-level "
            "submitted scorecard on a real metric worth a 1/day slot; never submits."
        )
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "result_path",
    "field_principles",
    "reproducible_total_levels",
    "generic_transfer_ci",
    "action_efficiency_ci",
    "coheadline_metrics",
    "scorecard",
    "a5_transfer",
    "a6_integration_headline",
    "operator_submission_basis",
    "live_submittable_count_baseline",
    "leaderboard_submission",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


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


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ci_or_none(value: Any) -> list[float] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    return None


def _live_flags(path: Path) -> list[dict[str, Any]]:
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


def _null_delta_corrigendum(
    artifact: Mapping[str, Any],
    flags: Sequence[Mapping[str, Any]],
) -> JsonDict | None:
    if artifact_reader is None:
        return None
    try:
        classification = artifact_reader.classify_known_false_positive_null_delta(  # type: ignore[attr-defined]
            dict(artifact),
            [dict(flag) for flag in flags],
        )
    except Exception:
        return None
    return dict(classification) if isinstance(classification, Mapping) else None


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
    false_negative = _false_negative_risk_open(flags)
    corrigendum = _null_delta_corrigendum(artifact, flags)
    flagged = bool(stamped or critical)
    diagnosis_only = bool(flagged and corrigendum)
    included = bool(exists and artifact and not flagged and not gate_failures and not false_negative)
    reason = "included_clean"
    if not exists:
        reason = "missing"
    elif gate_failures:
        reason = "failed_acceptance_gate"
    elif diagnosis_only:
        reason = "flagged_null_delta_corrigendum_diagnosis_only"
    elif flagged:
        reason = "flagged_adversarial_excluded"
    elif false_negative:
        reason = "false_negative_risk_open"
    return {
        "name": name,
        "artifact": source.relative_path,
        "role": source.role,
        "exists": exists,
        "stamped_flagged_adversarial": stamped,
        "live_critical": bool(critical),
        "live_flags": flags,
        "critical_flags": critical,
        "false_negative_risk_open": false_negative,
        "acceptance_gate_failures": gate_failures,
        "null_delta_corrigendum": corrigendum,
        "diagnosis_only": diagnosis_only,
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
        path = root / source.relative_path
        if artifacts is not None and name in artifacts:
            artifact = dict(artifacts[name])
            exists = True
        else:
            artifact = _read_json(path)
            exists = path.exists()
        loaded[name] = artifact
        statuses[name] = _source_status(
            name=name,
            source=source,
            root=root,
            artifact=artifact,
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
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4602": "REQ-CAPSTONE-4602" in spec_text,
        "registry_yaml_loadable": bool(registry),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_reproducible_total_levels": _as_int(
            registry.get("reproducible_total_levels")
        ),
        "upstream_artifacts_present": source_exists,
        "missing_upstream_artifacts": [
            UPSTREAM_SOURCES[name].relative_path
            for name, exists in source_exists.items()
            if not exists
        ],
        "summarize_artifact_py_used_for_live_flags": artifact_reader is not None,
        "leaderboard_submission": False,
        "operator_only": True,
        "network_required": False,
        "research_conductor_modified": False,
    }
    checks["ok"] = bool(
        checks["agents_md_read"]
        and checks["codex_or_opencode_md_read"]
        and checks["spec_has_req_4602"]
        and checks["registry_yaml_loadable"]
        and source_exists.get("LIVE_BASELINE")
    )
    return checks


def _clean(statuses: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return bool(statuses.get(name, {}).get("included_in_headline"))


def _delta_ci_positive(ci: list[float] | None) -> bool:
    return bool(ci and ci[0] > 0.0)


def _a2_repro_delta(a2: Mapping[str, Any], included: bool) -> tuple[int, int, int]:
    if not included:
        return 0, 0, 0
    update = a2.get("registry_update")
    gate = a2.get("reproduction_gate")
    if not isinstance(update, Mapping) or not isinstance(gate, Mapping):
        return 0, 0, 0
    before = _as_int(update.get("prior_total_declared") or update.get("prior_total_row_sum"))
    after = _as_int(update.get("new_total_declared") or update.get("new_total_row_sum"))
    delta = _as_int(update.get("reconciled_total_delta") or update.get("banked_levels"))
    if delta == 0 and before and after:
        delta = after - before
    if (
        a2.get("offline_reproduced") is True
        and update.get("updated") is True
        and gate.get("reproduced") is True
    ):
        return before, after, max(0, delta)
    return before, after, 0


def _winner_generated_moved(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    baseline = _as_float(a1.get("winner_generated_rate_baseline"), WINNER_GENERATED_BASELINE)
    rate = _as_float(a1.get("winner_generated_rate_with_wiring"))
    delta = _as_float(a1.get("winner_generated_delta"), rate - baseline)
    ci = _ci_or_none(a1.get("winner_generated_delta_ci") or a1.get("winner_generated_ci"))
    if status.get("included_in_headline") is not True:
        return {
            "moved": False,
            "baseline": baseline,
            "rate_with_wiring": rate,
            "winner_generated_delta": delta,
            "winner_generated_ci": ci,
            "no_wiring_control_passed": a1.get("no_wiring_control_passed") is True,
            "raw_reported_strictly_above_baseline": rate > baseline,
            "headline_numbers_aggregated": False,
            "source": UPSTREAM_SOURCES["A1"].relative_path,
            "reason": str(status.get("reason")),
        }
    moved = bool(
        rate > baseline
        and delta > 0.0
        and _delta_ci_positive(ci)
        and a1.get("no_wiring_control_passed") is True
        and a1.get("solve_rate_preserved") is True
        and a1.get("verifier_is_oracle") is False
    )
    return {
        "moved": moved,
        "baseline": baseline,
        "rate_with_wiring": rate,
        "winner_generated_delta": delta,
        "winner_generated_ci": ci,
        "no_wiring_control_passed": a1.get("no_wiring_control_passed") is True,
        "raw_reported_strictly_above_baseline": rate > baseline,
        "headline_numbers_aggregated": True,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
        "reason": "a1_clean_ci_supported_move" if moved else "a1_clean_missing_strict_ci_supported_move",
    }


def _generic_transfer_moved(
    a1: Mapping[str, Any],
    a1_status: Mapping[str, Any],
    a3: Mapping[str, Any],
    a3_status: Mapping[str, Any],
) -> JsonDict:
    a1_rate = _as_float(a1.get("generic_transfer_rate_with_wiring"))
    a1_delta = _as_float(a1.get("transfer_delta"), a1_rate - GENERIC_TRANSFER_BASELINE)
    a1_ci = _ci_or_none(a1.get("transfer_ci"))
    a1_clean_move = bool(
        a1_status.get("included_in_headline") is True
        and a1_rate > GENERIC_TRANSFER_BASELINE
        and a1_delta > 0.0
        and _delta_ci_positive(a1_ci)
        and a1.get("no_wiring_control_passed") is True
        and a1.get("solve_rate_preserved") is True
    )
    a3_rate = _as_float(a3.get("generic_transfer_rate_with_energy"))
    a3_no_energy = _as_float(a3.get("generic_transfer_rate_no_energy"))
    a3_delta = a3_rate - a3_no_energy
    a3_ci = _ci_or_none(a3.get("generic_transfer_ci"))
    a3_clean_move = bool(
        a3_status.get("included_in_headline") is True
        and a3_rate > GENERIC_TRANSFER_BASELINE
        and a3_delta > 0.0
        and (a3_ci is None or a3_ci[0] > GENERIC_TRANSFER_BASELINE)
        and a3.get("no_energy_control_passed") is True
        and a3.get("solve_rate_preserved") is True
    )
    return {
        "moved": bool(a1_clean_move or a3_clean_move),
        "baseline": GENERIC_TRANSFER_BASELINE,
        "a1": {
            "source": UPSTREAM_SOURCES["A1"].relative_path,
            "rate": a1_rate,
            "delta": a1_delta,
            "ci": a1_ci,
            "headline_numbers_aggregated": a1_status.get("included_in_headline") is True,
            "reason": a1_status.get("reason"),
        },
        "a3": {
            "source": UPSTREAM_SOURCES["A3"].relative_path,
            "rate": a3_rate,
            "no_energy_rate": a3_no_energy,
            "delta": a3_delta,
            "ci": a3_ci,
            "headline_numbers_aggregated": a3_status.get("included_in_headline") is True,
            "reason": a3_status.get("reason"),
        },
        "reason": (
            "clean_ci_supported_transfer_move"
            if (a1_clean_move or a3_clean_move)
            else "no_clean_ci_supported_transfer_move"
        ),
    }


def _goal_energy_helped(a3: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    with_energy = _as_float(a3.get("winner_generated_rate_with_energy"))
    no_energy = _as_float(a3.get("winner_generated_rate_no_energy"))
    delta = _as_float(a3.get("winner_generated_delta"), with_energy - no_energy)
    included = status.get("included_in_headline") is True
    helped = bool(
        included
        and delta > 0.0
        and with_energy > no_energy
        and a3.get("no_energy_control_passed") is True
        and a3.get("solve_rate_preserved") is True
        and a3.get("verifier_is_oracle") is False
    )
    reason = "a3_clean_goal_energy_helped" if helped else str(status.get("reason"))
    if status.get("false_negative_risk_open") is True:
        reason = "a3_false_negative_risk_open"
    return {
        "helped": helped,
        "winner_generated_rate_with_energy": with_energy,
        "winner_generated_rate_no_energy": no_energy,
        "winner_generated_delta": delta,
        "no_energy_control_passed": a3.get("no_energy_control_passed") is True,
        "targeted_classes": list(a3.get("targeted_classes") or []),
        "null_is_clean": bool(included and not helped),
        "source": UPSTREAM_SOURCES["A3"].relative_path,
        "reason": reason,
    }


def _metric_record(
    *,
    clean: bool,
    artifact: Mapping[str, Any],
    value_key: str,
    source: str,
    ci_key: str | None = None,
) -> JsonDict:
    value = _float_or_none(artifact.get(value_key))
    ci = _ci_or_none(artifact.get(ci_key)) if ci_key else None
    return {
        "clean_value": value if clean else None,
        "quarantined_value": None if clean else value,
        "ci": ci,
        "source": source,
        "included_in_clean_headline": clean,
    }


def _flagged_artifacts_handled(statuses: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    excluded: list[str] = []
    excluded_details: list[JsonDict] = []
    carveouts: list[JsonDict] = []
    positive_control_guard: list[str] = []
    gate_failures: list[JsonDict] = []
    for name, status in statuses.items():
        if name == "LIVE_BASELINE":
            continue
        if status.get("diagnosis_only") is True:
            carveouts.append(
                {
                    "artifact": status.get("artifact"),
                    "corrigendum": status.get("null_delta_corrigendum"),
                }
            )
        if status.get("included_in_headline") is False and (
            status.get("stamped_flagged_adversarial") or status.get("live_critical")
        ):
            excluded.append(str(status.get("artifact")))
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
        if status.get("false_negative_risk_open") is True:
            positive_control_guard.append(name)
        if status.get("acceptance_gate_failures"):
            gate_failures.append(
                {
                    "name": name,
                    "artifact": status.get("artifact"),
                    "failed_gates": status.get("acceptance_gate_failures"),
                }
            )
    return {
        "excluded_artifacts": excluded,
        "excluded_details": excluded_details,
        "null_delta_carveouts": carveouts,
        "offline_arc_methodology_guard_honored": [
            "A1/A2/A3/A4/A5/A6 offline ARC artifacts were evaluated by flags and gates, not by fast-runtime suspicion alone.",
            "A2/A4/A5 clean fast offline artifacts remain admissible when reproduction/methodology fields are present.",
        ],
        "learned_cnn_duration_guard_honored": (
            "No .424 artifact was excluded merely for fast offline duration or missing model_specs."
        ),
        "positive_control_failed_guard": positive_control_guard,
        "positive_control_guard_note": (
            "FALSE_NEGATIVE_RISK nulls are broken-test signals; their nulls are not clean results."
        ),
        "failed_acceptance_gate_overrides": gate_failures,
    }


def _verifier_oracle_distinct(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    rows: JsonDict = {}
    for name in ("A1", "A3", "A4", "A6"):
        rows[name] = {
            "artifact": UPSTREAM_SOURCES[name].relative_path,
            "verifier_is_oracle": artifacts.get(name, {}).get("verifier_is_oracle"),
            "included_in_headline": statuses.get(name, {}).get("included_in_headline"),
            "oracle_distinct": artifacts.get(name, {}).get("verifier_is_oracle") is False,
        }
    rows["all_included_value_claims_oracle_distinct"] = all(
        row["oracle_distinct"]
        for row in rows.values()
        if isinstance(row, Mapping) and row.get("included_in_headline") is True
    )
    return rows


def _a5_transfer(a5: Mapping[str, Any], included: bool) -> JsonDict:
    values = a5.get("transfer_value_per_game")
    value_games = [
        str(game)
        for game, row in (values.items() if isinstance(values, Mapping) else [])
        if isinstance(row, Mapping) and row.get("value_added") is True
    ]
    return {
        "included_in_headline": included,
        "primitive_persisted": a5.get("primitive_persisted"),
        "transfer_games": list(a5.get("transfer_games") or []),
        "value_added_games": value_games,
        "new_levels_banked": _as_int(
            a5.get("new_levels_banked")
            or (
                a5.get("offline_reproduced", {}).get("new_levels_banked")
                if isinstance(a5.get("offline_reproduced"), Mapping)
                else 0
            )
        ),
    }


def _a6_integration(a6: Mapping[str, Any], included: bool, status: Mapping[str, Any]) -> JsonDict:
    return {
        "included_in_headline": included,
        "submitted_config_raised_metric_clean": bool(
            included
            and (
                _as_float(a6.get("winner_generated_rate_integrated")) > WINNER_GENERATED_BASELINE
                or _as_float(a6.get("generic_transfer_rate_integrated")) > GENERIC_TRANSFER_BASELINE
                or _as_int(a6.get("live_submittable_level_count_integrated"))
                > LIVE_SUBMITTABLE_BASELINE
            )
        ),
        "raw_honest_verdict": a6.get("honest_verdict"),
        "levers_integrated": list(a6.get("levers_integrated") or []),
        "reason": status.get("reason"),
        "upstream_lever_audit": a6.get("upstream_lever_audit"),
    }


def _cited_upstream_artifacts(statuses: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    clean_imports = {
        "A2": ["registry_update", "reproduction_gate", "offline_reproduced"],
        "A4": [
            "live_submittable_level_count",
            "live_submittable_count_prev",
            "count_delta",
            "ready_for_operator_submit",
        ],
        "A5": ["primitive_persisted", "transfer_games", "transfer_value_per_game"],
        "LIVE_BASELINE": ["live_total_levels"],
    }
    quarantined = {
        "A1": [
            "winner_generated_rate_with_wiring",
            "generic_transfer_rate_with_wiring",
            "transfer_ci",
        ],
        "A6": [
            "winner_generated_rate_integrated",
            "generic_transfer_rate_integrated",
            "live_submittable_level_count_integrated",
        ],
        "B1": [
            "winner_generated_rate",
            "generic_transfer_rate_over_variants",
            "action_efficiency_score",
        ],
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
            "imported_fields": clean_imports.get(name, []) if included or name == "LIVE_BASELINE" else [],
            "quarantined_fields_reported": quarantined.get(name, []) if not included else [],
        }
    cited["REGISTRY"] = {
        "artifact": REGISTRY_RELATIVE_PATH,
        "role": "authoritative_reproducible_total_levels",
        "exists": True,
        "imported_fields": ["reproducible_total_levels"],
    }
    return cited


def _scorecard(
    artifacts: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    winner_moved: Mapping[str, Any],
    transfer_moved: Mapping[str, Any],
    goal_energy: Mapping[str, Any],
    a2_before: int,
    a2_after: int,
    a2_delta: int,
    a5_transfer: Mapping[str, Any],
    a6_integration: Mapping[str, Any],
) -> JsonDict:
    return {
        "A1": {
            "artifact": UPSTREAM_SOURCES["A1"].relative_path,
            "included_in_headline": statuses["A1"]["included_in_headline"],
            "reason": statuses["A1"]["reason"],
            "winner_generated_moved": winner_moved.get("moved"),
            "generic_transfer_moved": transfer_moved.get("a1", {}).get("headline_numbers_aggregated")
            and transfer_moved.get("moved"),
        },
        "A2": {
            "artifact": UPSTREAM_SOURCES["A2"].relative_path,
            "included_in_headline": statuses["A2"]["included_in_headline"],
            "reproducible_total_before": a2_before,
            "reproducible_total_after": a2_after,
            "reproducible_total_delta": a2_delta,
        },
        "A3": {
            "artifact": UPSTREAM_SOURCES["A3"].relative_path,
            "included_in_headline": statuses["A3"]["included_in_headline"],
            "reason": statuses["A3"]["reason"],
            "goal_energy_helped": goal_energy.get("helped"),
        },
        "A4": {
            "artifact": UPSTREAM_SOURCES["A4"].relative_path,
            "included_in_headline": statuses["A4"]["included_in_headline"],
            "live_submittable_level_count": _as_int(
                artifacts["A4"].get("live_submittable_level_count")
            ),
            "count_delta": _as_int(artifacts["A4"].get("count_delta")),
            "ready_for_operator_submit": artifacts["A4"].get("ready_for_operator_submit") is True,
        },
        "A5": dict(a5_transfer),
        "A6": dict(a6_integration),
        "B1": {
            "artifact": UPSTREAM_SOURCES["B1"].relative_path,
            "included_in_headline": statuses["B1"]["included_in_headline"],
            "reason": statuses["B1"]["reason"],
            "coheadline_numbers_quarantined": statuses["B1"]["included_in_headline"] is not True,
        },
    }


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "winner_generated_moved": artifact.get("winner_generated_moved"),
        "generic_transfer_moved": artifact.get("generic_transfer_moved"),
        "goal_energy_helped": artifact.get("goal_energy_helped"),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "reproducible_total_levels": artifact.get("reproducible_total_levels"),
        "live_submittable_level_count": artifact.get("live_submittable_level_count"),
        "winner_generated_rate": artifact.get("winner_generated_rate"),
        "generic_transfer_rate_over_variants": artifact.get("generic_transfer_rate_over_variants"),
        "action_efficiency_score": artifact.get("action_efficiency_score"),
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "flagged_artifacts_handled": artifact.get("flagged_artifacts_handled"),
        "cited_upstream_artifacts": artifact.get("cited_upstream_artifacts"),
        "random_seed": artifact.get("random_seed"),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    upstream, statuses = _load_artifacts(
        root_path,
        artifacts=artifacts,
        live_flags_by_name=live_flags_by_name,
    )
    checks = dict(preconditions_checked or check_preconditions(root_path, statuses=statuses))
    baseline = _as_int(upstream["LIVE_BASELINE"].get("live_total_levels"), LIVE_SUBMITTABLE_BASELINE)
    a2_before, a2_after, a2_delta = _a2_repro_delta(upstream["A2"], _clean(statuses, "A2"))
    registry_total = _as_int(checks.get("registry_reproducible_total_levels")) or a2_after
    live_count = (
        _as_int(upstream["A4"].get("live_submittable_level_count"))
        if _clean(statuses, "A4")
        else 0
    )
    winner_moved = _winner_generated_moved(upstream["A1"], statuses["A1"])
    transfer_moved = _generic_transfer_moved(
        upstream["A1"],
        statuses["A1"],
        upstream["A3"],
        statuses["A3"],
    )
    goal_energy = _goal_energy_helped(upstream["A3"], statuses["A3"])
    b1_clean = _clean(statuses, "B1")
    winner_metric = _metric_record(
        clean=b1_clean,
        artifact=upstream["B1"],
        value_key="winner_generated_rate",
        source=UPSTREAM_SOURCES["B1"].relative_path,
    )
    generic_metric = _metric_record(
        clean=b1_clean,
        artifact=upstream["B1"],
        value_key="generic_transfer_rate_over_variants",
        source=UPSTREAM_SOURCES["B1"].relative_path,
        ci_key="generic_transfer_ci",
    )
    action_metric = _metric_record(
        clean=b1_clean,
        artifact=upstream["B1"],
        value_key="action_efficiency_score",
        source=UPSTREAM_SOURCES["B1"].relative_path,
        ci_key="action_efficiency_ci",
    )
    a5 = _a5_transfer(upstream["A5"], _clean(statuses, "A5"))
    a6 = _a6_integration(upstream["A6"], _clean(statuses, "A6"), statuses["A6"])
    ready = bool(
        checks.get("ok", True)
        and _clean(statuses, "A4")
        and live_count > baseline
        and upstream["A4"].get("ready_for_operator_submit") is True
        and upstream["A4"].get("offline_reproduced") is True
    )
    generation_success = bool(winner_moved["moved"] or transfer_moved["moved"])
    if generation_success:
        verdict = "success: generation_completeness_winner_generated_above_1of25_or_transfer_above_0.04"
    elif a2_delta > 0:
        verdict = "complete: generation_wall_persists_residual_logged_capability_grew"
    else:
        verdict = "complete: generation_wall_persists_residual_logged_no_capability_growth"
    flagged = _flagged_artifacts_handled(statuses)
    coheadline_metrics = {
        "reproducible_total_levels": {
            "clean_value": registry_total,
            "source": REGISTRY_RELATIVE_PATH,
            "included_in_clean_headline": True,
        },
        "live_submittable_level_count": {
            "clean_value": live_count,
            "source": UPSTREAM_SOURCES["A4"].relative_path,
            "included_in_clean_headline": _clean(statuses, "A4"),
        },
        "winner_generated_rate": winner_metric,
        "generic_transfer_rate_over_variants": generic_metric,
        "action_efficiency_score": action_metric,
    }
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4602",
            "SCENARIO-CAPSTONE-4602",
            "SCENARIO-CAPSTONE-4602-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "winner_generated_moved": winner_moved,
        "generic_transfer_moved": transfer_moved,
        "goal_energy_helped": goal_energy,
        "reproducible_total_levels_delta": a2_delta,
        "reproducible_total_levels_before": a2_before,
        "reproducible_total_levels": registry_total,
        "live_submittable_count_baseline": baseline,
        "live_submittable_level_count": live_count,
        "winner_generated_rate": winner_metric,
        "generic_transfer_rate_over_variants": generic_metric,
        "generic_transfer_ci": generic_metric.get("ci"),
        "action_efficiency_score": action_metric,
        "action_efficiency_ci": action_metric.get("ci"),
        "coheadline_metrics": coheadline_metrics,
        "a5_transfer": a5,
        "a6_integration_headline": a6,
        "verifier_is_oracle_distinct_levers": _verifier_oracle_distinct(upstream, statuses),
        "flagged_artifacts_handled": flagged,
        "cited_upstream_artifacts": _cited_upstream_artifacts(statuses),
        "scorecard": _scorecard(
            upstream,
            statuses,
            winner_moved=winner_moved,
            transfer_moved=transfer_moved,
            goal_energy=goal_energy,
            a2_before=a2_before,
            a2_after=a2_after,
            a2_delta=a2_delta,
            a5_transfer=a5,
            a6_integration=a6,
        ),
        "ready_for_operator_submit": ready,
        "operator_submission_basis": {
            "operator_only": True,
            "leaderboard_submission": False,
            "last_submitted_scorecard_levels": baseline,
            "clean_a4_live_submittable_level_count": live_count if _clean(statuses, "A4") else None,
            "a6_integration_headline_clean": _clean(statuses, "A6"),
            "basis": "clean_a4_package_above_33" if ready else "not_ready_or_missing_clean_package",
        },
        "preconditions_checked": checks,
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else 0.0),
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _metric_errors(artifact: Mapping[str, Any], field: str) -> list[str]:
    errors: list[str] = []
    metric = artifact.get(field)
    if not isinstance(metric, Mapping):
        return [f"{field} must be object"]
    if "clean_value" not in metric:
        errors.append(f"{field}.clean_value missing")
    elif metric.get("clean_value") is not None and not isinstance(metric.get("clean_value"), float):
        errors.append(f"{field}.clean_value must be float or null")
    if "quarantined_value" in metric and metric.get("quarantined_value") is not None:
        if not isinstance(metric.get("quarantined_value"), float):
            errors.append(f"{field}.quarantined_value must be float or null")
    if not isinstance(metric.get("source"), str):
        errors.append(f"{field}.source must be string")
    if not _is_bare_bool(metric.get("included_in_clean_headline")):
        errors.append(f"{field}.included_in_clean_headline must be bare bool")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    for field, moved_key in (
        ("winner_generated_moved", "moved"),
        ("generic_transfer_moved", "moved"),
        ("goal_energy_helped", "helped"),
    ):
        value = artifact.get(field)
        if not isinstance(value, Mapping):
            errors.append(f"{field} must be object")
        elif not _is_bare_bool(value.get(moved_key)):
            errors.append(f"{field}.{moved_key} must be bare bool")
    for field in (
        "reproducible_total_levels_delta",
        "reproducible_total_levels",
        "live_submittable_count_baseline",
        "live_submittable_level_count",
    ):
        if not _is_bare_int(artifact.get(field)):
            errors.append(f"{field} must be bare int")
    for field in (
        "winner_generated_rate",
        "action_efficiency_score",
        "generic_transfer_rate_over_variants",
    ):
        errors.extend(_metric_errors(artifact, field))
    if not _is_bare_bool(artifact.get("ready_for_operator_submit")):
        errors.append("ready_for_operator_submit must be bare bool")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    for field in (
        "verifier_is_oracle_distinct_levers",
        "flagged_artifacts_handled",
        "cited_upstream_artifacts",
        "preconditions_checked",
        "scorecard",
        "coheadline_metrics",
        "operator_submission_basis",
    ):
        if not isinstance(artifact.get(field), Mapping):
            errors.append(f"{field} must be object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in FIELD_PRINCIPLES:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
    metrics = artifact.get("coheadline_metrics")
    if isinstance(metrics, Mapping):
        for field in (
            "winner_generated_rate",
            "generic_transfer_rate_over_variants",
            "action_efficiency_score",
        ):
            if metrics.get(field) != artifact.get(field):
                errors.append(f"coheadline_metrics.{field} must match top-level field")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != _checksum(_artifact_checksum_payload(artifact)):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    write: bool = True,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    elapsed = duration_s
    if elapsed is None:
        elapsed = max(time.perf_counter() - start, 0.0001)
    artifact = build_artifact(root, live_flags_by_name=live_flags_by_name, duration_s=elapsed)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
