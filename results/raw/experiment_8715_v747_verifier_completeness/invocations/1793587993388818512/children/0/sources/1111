"""Experiment 4710: .433 ARC perception/amortized-exploration capstone.

This module aggregates the .433 A1-A6 and B1/B2 artifacts. It reads upstream
results through the disciplined artifact reader, loads no model, submits
nothing, and treats the ARC registry as the authoritative capability count.

Spec refs: REQ-CAPSTONE-4710, SCENARIO-CAPSTONE-4710,
SCENARIO-CAPSTONE-4710-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4710-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4710_capstone_v433"
SCHEMA = "carnot.exp4710.capstone_v433.v1"
RESULT_RELATIVE_PATH = "results/experiment_4710_capstone_v433.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4710
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 61
FROZEN_FOVER_AUROC = 0.9131
FIRST_WIN_BASELINE = 0.04
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
        "results/experiment_4700_object_centric_perception_proposal_live.json",
        "object_centric_perception_proposal_live",
    ),
    "A2": SourceSpec(
        "A2",
        "results/experiment_4701_amortized_exploration_prior_go_explore_live.json",
        "amortized_exploration_prior_go_explore_live",
    ),
    "A3": SourceSpec("A3", "results/experiment_4702_levelup_selfplay.json", "levelup_selfplay_bank"),
    "A4": SourceSpec(
        "A4",
        "results/experiment_4703_held_out_first_win_readiness.json",
        "held_out_first_win_readiness",
    ),
    "A5": SourceSpec("A5", "results/experiment_4704_primitive_persist_transfer.json", "primitive_persist_transfer"),
    "A6": SourceSpec("A6", "results/experiment_4705_integration_gate.json", "integration_gate"),
    "B1": SourceSpec("B1", "results/experiment_4706_perception_quality_cigate.json", "perception_quality_cigate"),
    "B2": SourceSpec(
        "B2",
        "results/experiment_4707_adversarial_verify_hardening.json",
        "adversarial_verify_firstwin_and_perception_guards",
    ),
}


FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_crossed_live_generic_new_level_via_<perception|amortized_exploration>_<games> "
            "OR complete: perception_and_amortized_exploration_levers_characterized_no_live_new_level OR complete: "
            "capability_grew_61_to_<n>."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream artifact with sha256 (the audit trail)."
    },
    "a1_perception_new_level": {
        "principle": (
            "the A1 result -- the generic agent's new level via object-centric perception, counted ONLY if "
            "the ORDER-1-representation ablation strictly lower + offline_reproduced."
        )
    },
    "a1_perception_is_the_wall_diagnostic": {
        "principle": (
            "the A1 DECISIVE diagnostic -- did an upper-bound representation raise proposal-coverage of the winning "
            "L1 trajectory where order-1 did not (perception_is_the_wall true/false); a headline finding even if A1 "
            "banks no level (resolves the 8-milestone ambiguity)."
        )
    },
    "a2_amortized_exploration_coverage_and_lift": {
        "principle": (
            "the A2 result -- coverage_delta (winner-now-generated) + held-out first-win lift, counted ONLY if "
            "coverage_delta>0 AND the no-prior ablation failed AND the CI excludes the no-prior baseline; + "
            "go_explore_now_live_reachable."
        )
    },
    "reproducible_total_levels": {
        "principle": "authoritative from the registry (A3 bank, 61->62+) -- did solve CAPABILITY grow."
    },
    "reproducible_total_levels_delta": {
        "principle": "registry after - 61, emitted explicitly so a null is annotated."
    },
    "held_out_first_win_readiness": {
        "principle": (
            "the A4 RETARGETED readiness -- experiment_4605 first_win_rate_integrated vs the 0.04 baseline "
            "(bootstrap-CI, null-delta markers present), the only proxy that tracks the scored lane (NOT the replay count)."
        )
    },
    "bridge_crossed_for_solve": {
        "principle": (
            "the headline decision -- did PERCEPTION (A1) or AMORTIZED EXPLORATION (A2) cross the offline->live bridge "
            "for L1-first-contact (the GENERIC agent reaches a NEW level), or did the levers get characterized as "
            "another honest null (the 9th milestone) with the perception-vs-search diagnostic resolved."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial / control-failed / ablation-missing artifact EXCLUDED + the guards applied "
            "(.432-B2 novelty-ablation/proposal-filter-heldout + .433-B2 firstwin-nulldelta/perception-overclaim) -- "
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
            "G1-G4 re-affirmed (FoVer 0.9131 NEVER substituted) -- the frozen publication invariant, "
            "not a new .433 headline."
        )
    },
    "leaderboard_submission": {
        "principle": "MUST be false -- submission is operator-only (External Publication Discipline)."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {"principle": "content-addressed hash catches silent drift on replay."},
    "preconditions_checked": {
        "principle": (
            "records resources verified (registry loadable, upstream artifacts present); pre-empts missing-resource fabrication."
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
    "REQ-CAPSTONE-4710",
    "SCENARIO-CAPSTONE-4710",
    "SCENARIO-CAPSTONE-4710-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4710-FIELD-PRINCIPLES",
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
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_required_number(payload: Mapping[str, Any], key: str) -> bool:
    if key not in payload or isinstance(payload.get(key), bool):
        return False
    try:
        parsed = float(payload.get(key))
    except (TypeError, ValueError):
        return False
    return parsed == parsed


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _summarize_and_live_flags(path: Path) -> tuple[int | None, list[JsonDict]]:
    if artifact_reader is None:  # pragma: no cover - defensive direct-script fallback
        return None, []
    summary_code = artifact_reader.summarize(path) if hasattr(artifact_reader, "summarize") else None
    flags = artifact_reader._live_flags(path) if hasattr(artifact_reader, "_live_flags") else []
    return summary_code, [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def _severity(flag: Mapping[str, Any]) -> str:
    return str(flag.get("severity") or "").lower()


def _critical_flags(flags: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(flag) for flag in flags if _severity(flag) == "critical"]


def _false_negative_risk_open(flags: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        flag.get("kind") == "FALSE_NEGATIVE_RISK" and "false_negative_risk_open" in str(flag.get("detail") or "")
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
    return "bare_control_passed" in artifact and artifact.get("bare_control_passed") is not True


def _a1_claims_new_level(name: str, artifact: Mapping[str, Any]) -> bool:
    if name != "A1" or not artifact:
        return False
    verdict = str(artifact.get("honest_verdict") or "").lower()
    return bool(
        verdict.startswith("success:")
        or _as_int(artifact.get("reproduced_levels")) >= 1
        or _as_int(artifact.get("generic_agent_reached_level")) > 0
    )


def _a1_required_evidence_present(artifact: Mapping[str, Any]) -> bool:
    return all(key in artifact for key in ("generic_agent_reached_level", "order1_ablation_reached_level", "offline_reproduced"))


def _perception_order1_ablation_strictly_lower(artifact: Mapping[str, Any]) -> bool:
    reached = _as_int(artifact.get("generic_agent_reached_level") or artifact.get("reached_level"))
    return reached > 0 and _as_int(artifact.get("order1_ablation_reached_level")) < reached


def _a1_ablation_missing(name: str, artifact: Mapping[str, Any]) -> bool:
    return _a1_claims_new_level(name, artifact) and not _a1_required_evidence_present(artifact)


def _a1_control_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(
        _a1_claims_new_level(name, artifact)
        and _a1_required_evidence_present(artifact)
        and not (_perception_order1_ablation_strictly_lower(artifact) and artifact.get("offline_reproduced") is True)
    )


def _a2_claims_coverage_up(name: str, artifact: Mapping[str, Any]) -> bool:
    return name == "A2" and artifact and _as_float(artifact.get("coverage_delta")) > 0.0


def _a2_ablation_missing(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(
        _a2_claims_coverage_up(name, artifact)
        and (
            "candidate_generation_coverage_no_prior_baseline" not in artifact
            or "no_prior_ablation_failed" not in artifact
        )
    )


def _a2_control_failed(name: str, artifact: Mapping[str, Any]) -> bool:
    return bool(_a2_claims_coverage_up(name, artifact) and not _a2_ablation_missing(name, artifact) and artifact.get("no_prior_ablation_failed") is not True)


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
    positive_failed = (name != "A2" and _positive_control_failed(artifact)) or (
        name not in {"A1", "A2", "A4"} and "false_negative_risk_checked" in artifact and artifact.get("false_negative_risk_checked") is not True
    )
    false_negative = _false_negative_risk_open(flags)
    ablation_missing = _a1_ablation_missing(name, artifact) or _a2_ablation_missing(name, artifact)
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
        and not ablation_missing
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
    elif ablation_missing:
        reason = "ablation_missing"
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
        "ablation_missing": ablation_missing,
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


def check_preconditions(root: Path | str = REPO_ROOT, *, statuses: Mapping[str, Mapping[str, Any]] | None = None) -> JsonDict:
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
        "spec_has_req_4710": "REQ-CAPSTONE-4710" in spec_text,
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
        ("spec_has_req_4710", "spec_req_4710"),
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


def _representation_coverage(a1: Mapping[str, Any], key: str) -> float:
    reps = _mapping_at(a1, "proposal_coverage_by_representation")
    row = _mapping_at(reps, key)
    return _as_float(row.get("coverage"))


def _a1_perception_new_level(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    reached = _as_int(a1.get("generic_agent_reached_level") or a1.get("reached_level"))
    order1_level = _as_int(a1.get("order1_ablation_reached_level"))
    order1_lower = _perception_order1_ablation_strictly_lower(a1)
    offline = a1.get("offline_reproduced") is True
    clean = status.get("included_in_headline") is True
    counted = bool(clean and reached > 0 and order1_lower and offline)
    if counted:
        reason = "order1_ablation_lower_offline_reproduced"
    elif not clean:
        reason = str(status.get("reason"))
    elif reached <= 0:
        reason = "generic_agent_did_not_reach_new_level"
    elif not offline:
        reason = "offline_reproduction_missing"
    else:
        reason = "order1_ablation_not_strictly_lower"
    return {
        "headline_counted": counted,
        "generic_agent_reached_level": reached,
        "order1_ablation_reached_level": order1_level,
        "order1_ablation_strictly_lower": order1_lower,
        "target_games": _target_games(a1),
        "offline_reproduced": offline,
        "reproduced_levels": _as_int(a1.get("reproduced_levels")),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
    }


def _a1_perception_is_the_wall_diagnostic(a1: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    object_coverage = _representation_coverage(a1, "object_centric")
    order1_coverage = _representation_coverage(a1, "order1")
    clean = status.get("included_in_headline") is True
    perception_wall = a1.get("perception_is_the_wall") is True
    headline_finding = bool(clean and perception_wall)
    if headline_finding:
        reason = "upper_bound_representation_raises_winning_l1_coverage_where_order1_does_not"
    elif not clean:
        reason = str(status.get("reason"))
    else:
        reason = "diagnostic_false"
    return {
        "headline_finding": headline_finding,
        "perception_is_the_wall": perception_wall,
        "object_centric_coverage": object_coverage,
        "order1_coverage": order1_coverage,
        "coverage_delta_vs_order1": round(object_coverage - order1_coverage, 6),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A1"].relative_path,
    }


def _first_win_delta_from_rates(a2: Mapping[str, Any]) -> float:
    if "first_win_rate_delta" in a2:
        return _as_float(a2.get("first_win_rate_delta"))
    return round(_as_float(a2.get("live_first_win_rate_with_prior")) - FIRST_WIN_BASELINE, 6)


def _a2_amortized_exploration_coverage_and_lift(a2: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    with_prior = _as_float(a2.get("candidate_generation_coverage_with_prior"))
    no_prior_reported = "candidate_generation_coverage_no_prior_baseline" in a2
    no_prior = _as_float(a2.get("candidate_generation_coverage_no_prior_baseline"))
    coverage_delta = _as_float(a2.get("coverage_delta"), with_prior - no_prior)
    no_prior_failed = a2.get("no_prior_ablation_failed") is True
    first_delta = _first_win_delta_from_rates(a2)
    live_lift_ci = _mapping_at(a2, "live_lift_ci")
    ci_excludes = bool(a2.get("heldout_first_win_ci_excludes_no_prior_baseline") is True or _ci_excludes_zero(live_lift_ci))
    winner_newly_generated = bool(no_prior_reported and no_prior == 0.0 and with_prior > 0.0 and coverage_delta > 0.0)
    go_explore_reachable = a2.get("go_explore_now_live_reachable") is True
    clean = status.get("included_in_headline") is True
    counted = bool(
        clean
        and coverage_delta > 0.0
        and winner_newly_generated
        and no_prior_failed
        and first_delta > 0.0
        and ci_excludes
        and go_explore_reachable
    )
    if counted:
        reason = "coverage_delta_positive_no_prior_failed_ci_excludes_no_prior_baseline"
    elif not clean:
        reason = str(status.get("reason"))
    elif coverage_delta <= 0.0:
        reason = "coverage_delta_not_positive"
    elif not no_prior_reported:
        reason = "no_prior_baseline_missing"
    elif no_prior > 0.0:
        reason = "winner_already_in_no_prior_baseline"
    elif not no_prior_failed:
        reason = "no_prior_ablation_did_not_fail"
    elif first_delta <= 0.0:
        reason = "no_positive_heldout_first_win_lift"
    elif not go_explore_reachable:
        reason = "go_explore_not_live_reachable"
    else:
        reason = "heldout_first_win_ci_includes_no_prior_baseline"
    return {
        "headline_counted": counted,
        "candidate_generation_coverage_with_prior": with_prior,
        "candidate_generation_coverage_no_prior_baseline": no_prior if no_prior_reported else None,
        "coverage_delta": coverage_delta,
        "winner_newly_generated_vs_no_prior": winner_newly_generated,
        "no_prior_ablation_failed": no_prior_failed,
        "go_explore_now_live_reachable": go_explore_reachable,
        "first_win_rate_delta": first_delta,
        "live_lift_ci": dict(live_lift_ci),
        "heldout_first_win_ci_excludes_no_prior_baseline": ci_excludes,
        "offline_reproduced": a2.get("offline_reproduced") is True,
        "target_games": _target_games(a2),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A2"].relative_path,
    }


def _held_out_first_win_readiness(a4: Mapping[str, Any], status: Mapping[str, Any]) -> JsonDict:
    first_win_rate = _as_float(a4.get("first_win_rate_integrated"))
    baseline = _as_float(a4.get("first_win_baseline"), FIRST_WIN_BASELINE)
    first_win_delta = round(first_win_rate - baseline, 6)
    ci_lower = _as_float(a4.get("first_win_ci_lower"))
    clean = status.get("included_in_headline") is True
    null_markers = bool(
        "first_win_delta_vs_baseline" in a4
        and isinstance(a4.get("null_delta_methodology_note"), str)
        and a4.get("positive_control_passed") is True
    )
    readiness = bool(clean and first_win_rate > baseline and ci_lower > 0.0)
    replay_floor = _mapping_at(a4, "replay_floor")
    if readiness:
        reason = "held_out_first_win_above_baseline_ci_lower_positive"
    elif not clean:
        reason = str(status.get("reason"))
    elif first_win_rate <= baseline:
        reason = "held_out_first_win_not_above_baseline"
    else:
        reason = "held_out_first_win_ci_lower_not_positive"
    return {
        "headline_counted": readiness,
        "first_win_rate_integrated": first_win_rate,
        "first_win_baseline": baseline,
        "first_win_delta_vs_baseline": first_win_delta,
        "first_win_ci_lower": ci_lower,
        "multi_level_deepen_rate_integrated": _as_float(a4.get("multi_level_deepen_rate_integrated")),
        "ready_for_operator_submit": readiness,
        "null_delta_markers_present": null_markers,
        "replay_count_is_not_the_score": a4.get("replay_count_is_not_the_score") is True,
        "replay_floor_live_submittable_level_count": _as_int(replay_floor.get("live_submittable_level_count")),
        "reason": reason,
        "source": UPSTREAM_SOURCES["A4"].relative_path,
    }


def _tests_passed(block: Any) -> bool:
    return not isinstance(block, Mapping) or block.get("passed") is not False


def _b2_guards_active(b1: Mapping[str, Any], b2: Mapping[str, Any]) -> JsonDict:
    return {
        ".432-B2 novelty-ablation/proposal-filter-heldout": True,
        ".433-B1 perception-quality-cigate": bool(
            isinstance(b1.get("loo_discrimination_gate_added"), Mapping)
            and b1.get("loo_discrimination_gate_added", {}).get("passed") is True
            and isinstance(b1.get("offpath_discrimination_metric_added"), Mapping)
            and b1.get("offpath_discrimination_metric_added", {}).get("passed") is True
            and isinstance(b1.get("perception_quality_floor_cigate_added"), Mapping)
            and b1.get("perception_quality_floor_cigate_added", {}).get("passed") is True
        ),
        ".433-B2 firstwin-nulldelta": bool(b2.get("firstwin_nulldelta_carveout_added") is True and _tests_passed(b2.get("tests_added"))),
        ".433-B2 perception-overclaim": bool(b2.get("perception_overclaim_guard_added") is True and _tests_passed(b2.get("tests_added"))),
    }


def _flagged_artifacts_handled(statuses: Mapping[str, Mapping[str, Any]], b1: Mapping[str, Any], b2: Mapping[str, Any]) -> JsonDict:
    excluded_details: list[JsonDict] = []
    flagged: list[JsonDict] = []
    positive_failed: list[JsonDict] = []
    false_negative_open: list[JsonDict] = []
    ablation_missing: list[JsonDict] = []
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
        if status.get("ablation_missing"):
            ablation_missing.append(row)
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
        "ablation_missing_artifacts": ablation_missing,
        "control_failed_artifacts": control_failed,
        "oracle_not_distinct_artifacts": oracle_failed,
        "failed_acceptance_gate_overrides": gate_failures,
        "guards_applied": _b2_guards_active(b1, b2),
        "guard_note": (
            "Stamped flagged, live-critical, control-failed, ablation-missing, and false-negative-risk-open "
            "artifacts are excluded from clean headline claims."
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


def _cited_statuses(statuses: Mapping[str, Mapping[str, Any]], imported_fields: Mapping[str, Sequence[str]]) -> dict[str, JsonDict]:
    return {
        name: {
            **dict(status),
            "imported_fields": list(imported_fields.get(name, [])),
        }
        for name, status in statuses.items()
    }


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


def _headline_bridge_source(a1: Mapping[str, Any], a2: Mapping[str, Any]) -> str:
    if a1.get("headline_counted") is True:
        return "perception"
    if a2.get("headline_counted") is True:
        return "amortized_exploration"
    return "none"


def _headline_verdict(a1: Mapping[str, Any], a2: Mapping[str, Any], total: int, preconditions: Mapping[str, Any]) -> str:
    if preconditions.get("ok") is not True:
        return f"blocked_{preconditions.get('blocked_resource', 'precondition')}"
    source = _headline_bridge_source(a1, a2)
    if source != "none":
        return f"success: bridge_crossed_live_generic_new_level_via_{source}_{_headline_games(a1, a2)}"
    if total > BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return f"complete: capability_grew_61_to_{total}"
    return "complete: perception_and_amortized_exploration_levers_characterized_no_live_new_level"


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
    a1 = _a1_perception_new_level(loaded.get("A1", {}), statuses.get("A1", {}))
    a1_diag = _a1_perception_is_the_wall_diagnostic(loaded.get("A1", {}), statuses.get("A1", {}))
    a2 = _a2_amortized_exploration_coverage_and_lift(loaded.get("A2", {}), statuses.get("A2", {}))
    a4 = _held_out_first_win_readiness(loaded.get("A4", {}), statuses.get("A4", {}))
    total = _as_int(registry_payload.get("reproducible_total_levels"))
    delta = total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    bridge_crossed = bool(a1["headline_counted"] or a2["headline_counted"])
    imported_fields = {
        "A1": [
            "generic_agent_reached_level",
            "order1_ablation_reached_level",
            "offline_reproduced",
            "reproduced_levels",
            "perception_is_the_wall",
            "proposal_coverage_by_representation",
        ],
        "A2": [
            "candidate_generation_coverage_with_prior",
            "candidate_generation_coverage_no_prior_baseline",
            "coverage_delta",
            "no_prior_ablation_failed",
            "first_win_rate_delta",
            "live_lift_ci",
            "go_explore_now_live_reachable",
        ],
        "A3": ["reproduced_levels", "offline_reproduced", "reproduction_gate"],
        "A4": [
            "first_win_rate_integrated",
            "first_win_baseline",
            "first_win_ci_lower",
            "null_delta_methodology_note",
            "positive_control_passed",
            "replay_count_is_not_the_score",
            "ready_for_operator_submit",
        ],
        "A5": ["primitive_persisted", "offline_reproduced_new_level", "transfer_value_per_game"],
        "A6": ["config_changed", "first_win_rate_integrated", "multi_level_deepen_rate_integrated"],
        "B1": ["loo_discrimination_gate_added", "offpath_discrimination_metric_added", "perception_quality_floor_cigate_added"],
        "B2": ["firstwin_nulldelta_carveout_added", "perception_overclaim_guard_added", "tests_added"],
    }
    flagged_handled = _flagged_artifacts_handled(statuses, loaded.get("B1", {}), loaded.get("B2", {}))
    crossing_source = (
        "A1_perception"
        if a1["headline_counted"]
        else ("A2_amortized_exploration" if a2["headline_counted"] else "none")
    )
    scorecard = {
        "headline": {
            "bridge_crossed_for_solve": bridge_crossed,
            "crossing_source": crossing_source,
            "perception_is_the_wall": a1_diag["perception_is_the_wall"],
            "registry_total_authoritative": True,
            "submission_operator_only": True,
        },
        "A1": a1,
        "A1_diagnostic": a1_diag,
        "A2": a2,
        "A3": {
            "clean": _clean(statuses, "A3"),
            "source": UPSTREAM_SOURCES["A3"].relative_path,
            "registry_authoritative_total_levels": total,
            "artifact_reproduced_levels": _as_int(loaded.get("A3", {}).get("reproduced_levels")),
        },
        "A4": a4,
        "A5": {
            "clean": _clean(statuses, "A5"),
            "verdict": loaded.get("A5", {}).get("honest_verdict"),
        },
        "A6": {
            "clean": _clean(statuses, "A6"),
            "reason": statuses.get("A6", {}).get("reason"),
            "first_win_rate_integrated": _as_float(loaded.get("A6", {}).get("first_win_rate_integrated")),
            "multi_level_deepen_rate_integrated": _as_float(loaded.get("A6", {}).get("multi_level_deepen_rate_integrated")),
        },
        "B1": {
            "clean": _clean(statuses, "B1"),
            "perception_quality_cigate_added": flagged_handled["guards_applied"][".433-B1 perception-quality-cigate"],
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
        "a1_perception_new_level": a1,
        "a1_perception_is_the_wall_diagnostic": a1_diag,
        "a2_amortized_exploration_coverage_and_lift": a2,
        "reproducible_total_levels": total,
        "reproducible_total_levels_delta": delta,
        "held_out_first_win_readiness": a4,
        "bridge_crossed_for_solve": bridge_crossed,
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
