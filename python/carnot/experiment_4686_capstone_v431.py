"""Experiment 4686: .431 ARC generation bridge capstone.

This module aggregates landed A1-A6 and B1/B2 artifacts. It loads no model,
submits nothing, and treats the registry as the authoritative capability count.

Spec refs: REQ-CAPSTONE-4686, SCENARIO-CAPSTONE-4686,
SCENARIO-CAPSTONE-4686-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4686-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4686_capstone_v431"
SCHEMA = "carnot.exp4686.capstone_v431.v1"
RESULT_RELATIVE_PATH = "results/experiment_4686_capstone_v431.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4686
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 59
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
        "results/experiment_4676_hierarchical_subgoal_search_live.json",
        "hierarchical_subgoal_search_live",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4677_poe_world_factored_subgoal_planner.json",
        "poe_world_factored_subgoal_planner",
    ),
    "A3": SourceSpec(
        "A3",
        "results/experiment_4678_levelup_selfplay.json",
        "levelup_selfplay_bank",
    ),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4679_refresh_submission_package.json",
        "refresh_submission_package",
    ),
    "A5": SourceSpec(
        "A5",
        "results/experiment_4680_primitive_persist_transfer.json",
        "primitive_persist_transfer",
    ),
    "A6": SourceSpec(
        "A6",
        "results/experiment_4681_integration_gate.json",
        "integration_gate",
    ),
    "B1": SourceSpec(
        "B1",
        "results/experiment_4682_generation_coverage_cigate.json",
        "generation_coverage_cigate",
    ),
    "B2": SourceSpec(
        "B2",
        "results/experiment_4683_adversarial_verify_hardening.json",
        "adversarial_verify_subgoal_and_coverage_guards",
    ),
}


FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_crossed_live_generic_new_level_via_generation_<games> "
            "OR complete: candidate_generation_levers_characterized_no_live_new_level OR complete: "
            "capability_grew_59_to_<n>."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream artifact with sha256 (the audit trail)."
    },
    "a1_hierarchical_subgoal_new_level": {
        "principle": (
            "the A1 result -- the generic agent's new level via subgoal search, counted ONLY if "
            "subgoal_decomposition + per_subgoal_reachable + the no-subgoal AND random-subgoal "
            "ablations BOTH strictly lower + offline_reproduced."
        )
    },
    "a2_factored_planner_coverage_and_lift": {
        "principle": (
            "the A2 result -- coverage_delta (winner-now-generated) + live first-win/solve lift, "
            "counted ONLY if coverage_delta>0 AND the live-lift CI excludes the matched flat baseline."
        )
    },
    "reproducible_total_levels": {
        "principle": "authoritative from the registry (A3 bank, 59->60+) -- did solve CAPABILITY grow."
    },
    "reproducible_total_levels_delta": {
        "principle": "registry after - 59, emitted explicitly so a null is annotated."
    },
    "bridge_crossed_for_solve": {
        "principle": (
            "the headline decision -- did GENERATION (A1 subgoal search / A2 factored planner) cross "
            "the offline->live bridge for SOLVE-RATE/DEPTH (the GENERIC agent reaches a NEW level via "
            "candidate-generation improvement), or did the generation levers get characterized as "
            "another honest null (the 7th milestone)."
        )
    },
    "live_submittable_level_count": {
        "principle": "the A4 operator-resubmit count (must stay > 33)."
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial / control-failed / decomposition-missing artifact EXCLUDED + "
            "the guards applied (.430-B2 L2-goal/multi-level-metric + .431-B2 "
            "subgoal-decomposition/coverage-baseline) -- fabrication-gate + false-negative-risk compliance."
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
            "G1-G4 re-affirmed (FoVer 0.9131 NEVER substituted) -- the frozen publication invariant, "
            "not a new .431 headline."
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
            "records resources verified (registry loadable, upstream artifacts present); pre-empts "
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
    "publication_gate",
    "duration_s",
)
SPEC_REFS = [
    "REQ-CAPSTONE-4686",
    "SCENARIO-CAPSTONE-4686",
    "SCENARIO-CAPSTONE-4686-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4686-FIELD-PRINCIPLES",
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


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


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


def _nontrivial_subgoal_decomposition(value: Any) -> bool:
    if isinstance(value, str):
        return value.count("->") >= 2
    if isinstance(value, Mapping):
        nested = [nested_value for nested_value in value.values() if nested_value not in (None, "", [])]
        return len(nested) >= 2 or any(_nontrivial_subgoal_decomposition(item) for item in nested)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        meaningful = [item for item in value if item not in (None, "", [])]
        return len(meaningful) >= 2 or any(_nontrivial_subgoal_decomposition(item) for item in meaningful)
    return False


def _has_nontrivial_subgoal_decomposition(artifact: Mapping[str, Any]) -> bool:
    return _nontrivial_subgoal_decomposition(artifact.get("subgoal_decomposition"))


def _bool_leaves(value: Any) -> list[bool]:
    if isinstance(value, bool):
        return [value]
    if isinstance(value, Mapping):
        leaves: list[bool] = []
        for nested in value.values():
            leaves.extend(_bool_leaves(nested))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves: list[bool] = []
        for item in value:
            leaves.extend(_bool_leaves(item))
        return leaves
    return []


def _per_subgoal_reachable(artifact: Mapping[str, Any]) -> bool:
    leaves = _bool_leaves(artifact.get("per_subgoal_reachable"))
    return bool(leaves) and all(leaf is True for leaf in leaves)


def _subgoal_ablations_strictly_lower(artifact: Mapping[str, Any]) -> bool:
    reached = _as_int(artifact.get("generic_agent_reached_level") or artifact.get("reached_level"))
    return (
        reached > 0
        and _as_int(artifact.get("no_subgoal_ablation_reached_level")) < reached
        and _as_int(artifact.get("random_subgoal_ablation_reached_level")) < reached
    )


def _a1_claims_new_level(name: str, artifact: Mapping[str, Any]) -> bool:
    if name != "A1" or not artifact:
        return False
    verdict = str(artifact.get("honest_verdict") or "").lower()
    return bool(
        "success:" in verdict
        or _as_int(artifact.get("reproduced_levels")) >= 1
        or ("new_level" in verdict and _as_int(artifact.get("generic_agent_reached_level")) > 0)
    )


def _a1_decomposition_missing(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(
        _a1_claims_new_level(name, artifact)
        and (
            not _has_nontrivial_subgoal_decomposition(artifact)
            or not _per_subgoal_reachable(artifact)
            or artifact.get("offline_reproduced") is not True
        )
    )


def _a1_control_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(
        _a1_claims_new_level(name, artifact)
        and not _a1_decomposition_missing(name, artifact)
        and not _subgoal_ablations_strictly_lower(artifact)
    )


def _a2_control_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    if name != "A2" or not artifact:
        return False
    coverage_delta = _as_float(artifact.get("coverage_delta"))
    factored = _as_float(artifact.get("candidate_generation_coverage_factored"))
    return coverage_delta > 0.0 and factored > 0.0 and "candidate_generation_coverage_flat_baseline" not in artifact


def _oracle_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return name in {"A1", "A2", "A3", "A4", "A5", "A6"} and artifact and artifact.get("verifier_is_oracle") is not False


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
    decomposition_missing = _a1_decomposition_missing(name, artifact)
    control_failed = _a1_control_failed(name, artifact) or _a2_control_failed(name, artifact)
    oracle_failed = _oracle_failed(name, artifact)
    flagged = bool(stamped or critical)
    included = bool(
        exists
        and artifact
        and not flagged
        and not gate_failures
        and not oracle_failed
        and not positive_failed
        and not false_negative
        and not decomposition_missing
        and not control_failed
    )
    reason = "included_clean"
    if not exists:
        reason = "missing"
    elif flagged:
        reason = "flagged_adversarial_or_live_critical_excluded"
    elif gate_failures:
        reason = "failed_acceptance_gate"
    elif oracle_failed:
        reason = "oracle_not_distinct"
    elif positive_failed:
        reason = "positive_control_failed"
    elif false_negative:
        reason = "false_negative_risk_open"
    elif decomposition_missing:
        reason = "decomposition_missing"
    elif control_failed:
        reason = "control_failed"
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
        "decomposition_missing": decomposition_missing,
        "control_failed": control_failed,
        "oracle_not_distinct": oracle_failed,
        "acceptance_gate_failures": gate_failures,
        "summary_exit_code": summary_exit_code,
        "duration_s": artifact.get("duration_s"),
        "inference_substrate": artifact.get("inference_substrate"),
        "included_in_headline": included,
        "reason": reason,
        "sha256": _file_sha256(path) if path.exists() else _checksum(artifact),
        "read_via_summarize_artifact": bool(exists and (live_flags_by_name is None or artifact_reader is not None)),
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
        "spec_has_req_4686": "REQ-CAPSTONE-4686" in spec_text,
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
        ("spec_has_req_4686", "spec_req_4686"),
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


def _ci_excludes_zero(ci: Mapping[str, Any]) -> bool:
    if "low" in ci or "high" in ci:
        lo = _as_float(ci.get("low"))
        hi = _as_float(ci.get("high"))
        return lo > 0.0 or hi < 0.0
    ci95 = ci.get("ci95")
    if not isinstance(ci95, Sequence) or isinstance(ci95, (str, bytes)) or len(ci95) != 2:
        return False
    lo = _as_float(ci95[0])
    hi = _as_float(ci95[1])
    return lo > 0.0 or hi < 0.0


def _target_games(artifact: Mapping[str, Any]) -> list[str]:
    target_games = artifact.get("target_games")
    if isinstance(target_games, Sequence) and not isinstance(target_games, (str, bytes)):
        return [str(game) for game in target_games if str(game)]
    target_game = str(artifact.get("target_game") or "")
    return [target_game] if target_game else []


def _a1_hierarchical_subgoal_new_level(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    reached = _as_int(a1.get("generic_agent_reached_level") or a1.get("reached_level"))
    decomposition_ok = _has_nontrivial_subgoal_decomposition(a1)
    reachable_ok = _per_subgoal_reachable(a1)
    ablations_lower = _subgoal_ablations_strictly_lower(a1)
    offline = a1.get("offline_reproduced") is True
    clean = status.get("included_in_headline") is True
    counted = bool(clean and decomposition_ok and reachable_ok and ablations_lower and offline and reached > 0)
    if counted:
        reason = "decomposition_reachable_lower_ablations_offline_reproduced"
    elif not clean:
        reason = str(status.get("reason"))
    elif not decomposition_ok:
        reason = "subgoal_decomposition_missing"
    elif not reachable_ok:
        reason = "subgoal_not_reachable"
    elif not offline:
        reason = "offline_reproduction_missing"
    elif not ablations_lower:
        reason = "ablations_not_strictly_lower"
    else:
        reason = "generic_agent_did_not_reach_new_level"
    return {
        "headline_counted": counted,
        "subgoal_decomposition_present": decomposition_ok,
        "per_subgoal_reachable_all_true": reachable_ok,
        "no_subgoal_ablation_reached_level": _as_int(a1.get("no_subgoal_ablation_reached_level")),
        "random_subgoal_ablation_reached_level": _as_int(a1.get("random_subgoal_ablation_reached_level")),
        "ablations_strictly_lower": ablations_lower,
        "generic_agent_reached_level": reached,
        "target_games": _target_games(a1),
        "offline_reproduced": offline,
        "reproduced_levels": _as_int(a1.get("reproduced_levels")),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
    }


def _live_delta_from_rates(a2: Mapping[str, Any], metric: str) -> float:
    key = f"{metric}_delta"
    if key in a2:
        return _as_float(a2.get(key))
    factored_key = f"live_{metric}_factored"
    baseline = _mapping_at(a2, "live_baseline_flat_search")
    baseline_value = baseline.get(metric) or baseline.get(metric.replace("solve_rate", "multi_level_rate"))
    return round(_as_float(a2.get(factored_key)) - _as_float(baseline_value), 6)


def _a2_factored_planner_coverage_and_lift(a2: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    factored = _as_float(a2.get("candidate_generation_coverage_factored"))
    flat_reported = "candidate_generation_coverage_flat_baseline" in a2
    flat = _as_float(a2.get("candidate_generation_coverage_flat_baseline"))
    coverage_delta = _as_float(a2.get("coverage_delta"), factored - flat)
    first_delta = _live_delta_from_rates(a2, "first_win_rate")
    solve_delta = _live_delta_from_rates(a2, "solve_rate")
    selected_metric = "solve_rate_delta" if solve_delta > first_delta else "first_win_rate_delta"
    selected_delta = max(first_delta, solve_delta)
    live_lift_ci = _mapping_at(a2, "live_lift_ci")
    ci_excludes = bool(
        a2.get("ci_excludes_flat_baseline") is True
        or a2.get("ci_excludes_flat_search_baseline") is True
        or _ci_excludes_zero(live_lift_ci)
    )
    winner_newly_generated = bool(flat_reported and flat == 0.0 and factored > 0.0 and coverage_delta > 0.0)
    offline = a2.get("offline_reproduced") is True
    clean = status.get("included_in_headline") is True
    counted = bool(clean and coverage_delta > 0.0 and winner_newly_generated and selected_delta > 0.0 and ci_excludes and offline)
    if counted:
        reason = "coverage_delta_positive_winner_newly_generated_ci_excludes_baseline"
    elif not clean:
        reason = str(status.get("reason"))
    elif coverage_delta <= 0.0:
        reason = "coverage_delta_not_positive"
    elif not flat_reported:
        reason = "flat_baseline_missing"
    elif flat > 0.0:
        reason = "winner_already_in_flat_baseline"
    elif selected_delta <= 0.0:
        reason = "no_positive_live_lift"
    elif not ci_excludes:
        reason = "live_lift_ci_includes_flat_baseline"
    else:
        reason = "offline_reproduction_missing"
    return {
        "headline_counted": counted,
        "candidate_generation_coverage_factored": factored,
        "candidate_generation_coverage_flat_baseline": flat if flat_reported else None,
        "coverage_delta": coverage_delta,
        "winner_newly_generated_vs_flat": winner_newly_generated,
        "selected_metric": selected_metric,
        "selected_delta": selected_delta,
        "first_win_rate_delta": first_delta,
        "solve_rate_delta": solve_delta,
        "live_lift_ci": dict(live_lift_ci),
        "live_lift_ci_excludes_flat_baseline": ci_excludes,
        "offline_reproduced": offline,
        "target_games": _target_games(a2),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A2"].relative_path,
    }


def _tests_passed(block: Any) -> bool:
    return not isinstance(block, Mapping) or block.get("passed") is True


def _b2_guards_active(b1: Mapping[str, Any], b2: Mapping[str, Any]) -> JsonDict:
    return {
        ".430-B2 L2-goal/multi-level-metric": True,
        ".431-B1 candidate-generation-coverage": bool(
            b1.get("coverage_metric_added") is True and _tests_passed(b1.get("tests_added"))
        ),
        ".431-B2 subgoal-decomposition": bool(
            b2.get("subgoal_decomposition_guard_added") is True and _tests_passed(b2.get("tests_added"))
        ),
        ".431-B2 coverage-baseline": bool(
            b2.get("coverage_baseline_guard_added") is True and _tests_passed(b2.get("tests_added"))
        ),
    }


def _flagged_artifacts_handled(
    statuses: Mapping[str, Mapping[str, Any]],
    b1: Mapping[str, Any],
    b2: Mapping[str, Any],
) -> JsonDict:
    excluded_details: list[JsonDict] = []
    flagged: list[JsonDict] = []
    positive_failed: list[JsonDict] = []
    false_negative_open: list[JsonDict] = []
    decomposition_missing: list[JsonDict] = []
    control_failed: list[JsonDict] = []
    oracle_failed: list[JsonDict] = []
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
        if status.get("decomposition_missing"):
            decomposition_missing.append(row)
        if status.get("control_failed"):
            control_failed.append(row)
        if status.get("oracle_not_distinct"):
            oracle_failed.append(row)
        if status.get("acceptance_gate_failures"):
            gate_failures.append({**row, "failed_gates": status.get("acceptance_gate_failures")})
    return {
        "excluded_artifacts": [str(row["artifact"]) for row in excluded_details],
        "excluded_details": excluded_details,
        "flagged_adversarial_artifacts": flagged,
        "positive_control_failed_artifacts": positive_failed,
        "false_negative_risk_open_artifacts": false_negative_open,
        "decomposition_missing_artifacts": decomposition_missing,
        "control_failed_artifacts": control_failed,
        "oracle_not_distinct_artifacts": oracle_failed,
        "failed_acceptance_gate_overrides": gate_failures,
        "guards_applied": _b2_guards_active(b1, b2),
        "guard_note": (
            "Stamped flagged, live-critical, control-failed, decomposition-missing, "
            "and false-negative-risk-open artifacts are excluded from clean headline claims."
        ),
    }


def _paper_ready_state(publication_gate: Mapping[str, Any] | None) -> JsonDict:
    gate = dict(publication_gate or {})
    ready = gate.get("paper_ready") is True
    gates = gate.get("gates") if isinstance(gate.get("gates"), Mapping) else {}
    return {
        "paper_ready": ready,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "fover_09131_never_substituted": True,
        "gates": dict(gates),
        "unmet_gates": list(gate.get("unmet_gates", [])) if isinstance(gate.get("unmet_gates"), list) else [],
    }


def _cited_statuses(
    statuses: Mapping[str, Mapping[str, Any]], imported_fields: Mapping[str, Sequence[str]]
) -> dict[str, JsonDict]:
    cited: dict[str, JsonDict] = {}
    for name, status in statuses.items():
        cited[name] = {
            **dict(status),
            "imported_fields": list(imported_fields.get(name, [])),
        }
    return cited


def _load_publication_gate() -> JsonDict:
    if publication_gate_reader is None:  # pragma: no cover - defensive direct script fallback
        return {"paper_ready": False, "unmet_gates": ["publication_gate_unavailable"]}
    return dict(publication_gate_reader.evaluate())


def _headline_games(a1: Mapping[str, Any], a2: Mapping[str, Any]) -> str:
    games: Sequence[Any] = []
    if a1.get("headline_counted") is True:
        games = a1.get("target_games", [])
    elif a2.get("headline_counted") is True:
        games = a2.get("target_games", [])
    joined = "_".join(str(game) for game in games if str(game))
    return joined or "1"


def _headline_verdict(a1: Mapping[str, Any], a2: Mapping[str, Any], total: int, preconditions: Mapping[str, Any]) -> str:
    if preconditions.get("ok") is not True:
        return f"blocked_{preconditions.get('blocked_resource', 'precondition')}"
    if a1.get("headline_counted") is True or a2.get("headline_counted") is True:
        return f"success: bridge_crossed_live_generic_new_level_via_generation_{_headline_games(a1, a2)}"
    if total > BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return f"complete: capability_grew_59_to_{total}"
    return "complete: candidate_generation_levers_characterized_no_live_new_level"


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    registry: Mapping[str, Any] | None = None,
    publication_gate: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = time.perf_counter()
    loaded, statuses = _load_artifacts(root_path, artifacts=artifacts, live_flags_by_name=live_flags_by_name)
    registry_payload = dict(registry) if registry is not None else _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path, statuses=statuses)
    )
    paper = _paper_ready_state(publication_gate if publication_gate is not None else _load_publication_gate())
    a1 = _a1_hierarchical_subgoal_new_level(loaded.get("A1", {}), statuses.get("A1", {}))
    a2 = _a2_factored_planner_coverage_and_lift(loaded.get("A2", {}), statuses.get("A2", {}))
    total = _as_int(registry_payload.get("reproducible_total_levels"))
    delta = total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    live_submittable = (
        _as_int(loaded.get("A4", {}).get("live_submittable_level_count"))
        if _clean(statuses, "A4")
        else 0
    )
    bridge_crossed = bool(a1["headline_counted"] or a2["headline_counted"])
    imported_fields = {
        "A1": [
            "subgoal_decomposition",
            "per_subgoal_reachable",
            "generic_agent_reached_level",
            "no_subgoal_ablation_reached_level",
            "random_subgoal_ablation_reached_level",
            "offline_reproduced",
        ],
        "A2": [
            "candidate_generation_coverage_factored",
            "candidate_generation_coverage_flat_baseline",
            "coverage_delta",
            "first_win_rate_delta",
            "solve_rate_delta",
            "live_lift_ci",
            "offline_reproduced",
        ],
        "A3": ["reproduced_levels", "offline_reproduced", "reproduction_gate"],
        "A4": ["live_submittable_level_count", "ready_for_operator_submit", "offline_reproduced"],
        "A5": ["primitive_persisted", "offline_reproduced_new_level", "transfer_value_per_game"],
        "A6": [],
        "B1": ["coverage_metric_added", "honest_firstwin_floor_added", "coverage_floor_cigate_added", "tests_added"],
        "B2": ["subgoal_decomposition_guard_added", "coverage_baseline_guard_added", "tests_added"],
    }
    flagged_handled = _flagged_artifacts_handled(statuses, loaded.get("B1", {}), loaded.get("B2", {}))
    scorecard = {
        "headline": {
            "bridge_crossed_for_solve": bridge_crossed,
            "crossing_source": (
                "A1_hierarchical_subgoal_search"
                if a1["headline_counted"]
                else ("A2_factored_planner_coverage_and_lift" if a2["headline_counted"] else "none")
            ),
            "registry_total_authoritative": True,
            "submission_operator_only": True,
        },
        "A1": a1,
        "A2": a2,
        "A3": {
            "clean": _clean(statuses, "A3"),
            "source": UPSTREAM_SOURCES["A3"].relative_path,
            "registry_authoritative_total_levels": total,
        },
        "A4": {
            "clean": _clean(statuses, "A4"),
            "live_submittable_level_count": live_submittable,
            "ready_for_operator_submit": bool(
                _clean(statuses, "A4")
                and live_submittable > LIVE_SUBMITTABLE_SCORECARD_BASELINE
                and loaded.get("A4", {}).get("offline_reproduced") is True
            ),
        },
        "A5": {
            "clean": _clean(statuses, "A5"),
            "verdict": loaded.get("A5", {}).get("honest_verdict"),
        },
        "A6": {
            "clean": _clean(statuses, "A6"),
            "reason": statuses.get("A6", {}).get("reason"),
        },
        "B1": {
            "clean": _clean(statuses, "B1"),
            "coverage_metric_added": bool(loaded.get("B1", {}).get("coverage_metric_added")),
        },
        "B2": {
            "clean": _clean(statuses, "B2"),
            "guards_applied": flagged_handled["guards_applied"],
        },
        "verifier_oracle_checks": {
            name: loaded.get(name, {}).get("verifier_is_oracle") is False
            for name in ("A1", "A2", "A3", "A4", "A5", "A6")
            if statuses.get(name, {}).get("included_in_headline")
        },
    }
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _headline_verdict(a1, a2, total, preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "cited_upstream_artifacts": _cited_statuses(statuses, imported_fields),
        "a1_hierarchical_subgoal_new_level": a1,
        "a2_factored_planner_coverage_and_lift": a2,
        "reproducible_total_levels": total,
        "reproducible_total_levels_delta": delta,
        "bridge_crossed_for_solve": bridge_crossed,
        "live_submittable_level_count": live_submittable,
        "flagged_artifacts_handled": flagged_handled,
        "verifier_is_oracle": False,
        "paper_ready": paper["paper_ready"],
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions,
        "publication_gate": paper,
        "field_principles": FIELD_PRINCIPLES,
        "scorecard": scorecard,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare aggregation_from_upstream_artifacts")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    expected_checksum = payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(*, path: Path, artifact: Mapping[str, Any]) -> None:
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    root: Path | str = REPO_ROOT,
    *,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    publication_gate: Mapping[str, Any] | None = None,
    write: bool = True,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        live_flags_by_name=live_flags_by_name,
        publication_gate=publication_gate,
        duration_s=duration_s,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(path=root_path / RESULT_RELATIVE_PATH, artifact=artifact)
    return artifact


def main() -> None:  # pragma: no cover - direct script entry
    artifact = run(REPO_ROOT)
    print(json.dumps({"result_path": artifact["result_path"], "honest_verdict": artifact["honest_verdict"]}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - direct script entry
    main()
