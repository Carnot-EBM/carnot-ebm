"""Experiment 4735: .435 ARC sprint capstone scorecard.

This module aggregates the .435 upstream artifacts only after the disciplined
artifact summary reader has run. It makes no new solve claim, treats the ARC
registry as authoritative for capability growth, and preserves the frozen FoVer
publication gate.

Spec refs: REQ-CAPSTONE-4735, SCENARIO-CAPSTONE-4735,
SCENARIO-CAPSTONE-4735-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4735-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import glob
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

try:  # pragma: no cover - tests inject summary state
    import summarize_artifact as artifact_reader
except Exception:  # pragma: no cover - defensive import boundary
    artifact_reader = None  # type: ignore[assignment]

try:  # pragma: no cover - tests inject publication state
    import publication_gate as publication_gate_reader
except Exception:  # pragma: no cover - defensive import boundary
    publication_gate_reader = None  # type: ignore[assignment]


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4735_capstone_v435"
SCHEMA = "carnot.exp4735.capstone_v435.v1"
RESULT_RELATIVE_PATH = "results/experiment_4735_capstone_v435.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4735
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 63
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
TERMINAL_PREFIXES = ("complete:", "blocked_", "success:", "passed:", "shipped:")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_pattern: str
    role: str


UPSTREAM_SOURCES: dict[str, SourceSpec] = {
    "PREVIOUS": SourceSpec("PREVIOUS", "results/experiment_4723_capstone_v434.json", "previous_v434_scorecard"),
    "B1": SourceSpec("B1", "results/experiment_4725_*.json", "silent_bug_audit"),
    "A1": SourceSpec("A1", "results/experiment_4726_*.json", "online_driver"),
    "A2": SourceSpec("A2", "results/experiment_4727_*.json", "active_probe"),
    "A3": SourceSpec("A3", "results/experiment_4728_*.json", "levelup_selfplay_bank"),
    "A4": SourceSpec("A4", "results/experiment_4729_*.json", "held_out_readiness"),
    "A5": SourceSpec("A5", "results/experiment_4730_*.json", "primitive_persist"),
    "A6": SourceSpec("A6", "results/experiment_4731_*.json", "integration"),
    "B2": SourceSpec("B2", "results/experiment_4732_*.json", "guard"),
    "C": SourceSpec("C", "results/experiment_4733_*.json", "kv260"),
    "D": SourceSpec("D", "results/experiment_4734_*.json", "sota_ingestion"),
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete: <capability_grew_63_to_64 | bridge_crossed_for_solve_<game>_L<n>> "
            "-- the honest one-line milestone outcome."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "bridge_crossed_for_solve": {
        "principle": (
            "the HEADLINE DECISION -- True only if a GENERIC agent banked a NEW level via self-discovery "
            "(A1 online-driver L2 | A2 active-probe new level), offline-reproduced, ablation-controlled; "
            "False if only capability/depth grew."
        )
    },
    "a1_online_driver_result": {
        "principle": (
            "did A1 prove arms_non_degenerate AND beat frozen by >=+0.05 / deepen to L2 -- or confirm "
            "online_driver_arms_degenerate (the .433/.434 nulls tested dead code)?"
        )
    },
    "a2_active_probe_new_level": {
        "principle": (
            "did A2 reach a NEW generic level via active probing with the no-probe ablation FAILING? -- "
            "the verifier earning its place as a probe-router."
        )
    },
    "b1_silent_bug_reopen_list": {
        "principle": (
            "the .428-.434 nulls B1 classified silent_bug_must_reopen (incl. the .434 A4 verdict) -- "
            "closed levers that must reopen (trust correction)."
        )
    },
    "reproducible_total_levels_delta": {
        "principle": (
            "the registry header delta this milestone (63->64 if A1/A2/A3 banked) -- "
            "the monotonic north-star metric."
        )
    },
    "publication_gate": {
        "principle": (
            "the G1-G4 booleans + paper_ready + unmet_gates from publication_gate.py --json; "
            "FoVer 0.9131 frozen, never substituted."
        )
    },
    "verifier_is_oracle_confirmed_false": {
        "principle": (
            "confirms every aggregated value claim carries verifier_is_oracle:false "
            "(no circular moat over-claim)."
        )
    },
    "skipped_artifacts": {
        "principle": (
            "the artifacts skipped for flagged_adversarial / control-failed / ablation-missing -- "
            "the fabrication-gate discipline (never aggregate a flagged number)."
        )
    },
    "next_milestone_fallback": {
        "principle": (
            "the .436 direction (D's flagged_for_v436 + B1's reopen list + the strongest open lever) -- "
            "the convergence hand-off."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {"principle": "content-addressed hash of the aggregated artifact set."},
    "preconditions_checked": {
        "principle": (
            "records resources verified (upstream artifacts present, summarize/publication tools runnable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "a3_levelup_banked",
    "headline_decision",
    "scorecard",
    "cited_upstream_artifacts",
    "missing_artifacts",
    "field_principles",
    "spec_refs",
    "schema",
    "experiment",
    "result_path",
    "duration_s",
    "leaderboard_submission",
    "reproducible_total_levels",
    "solve_provenance_confirmed",
)
SPEC_REFS = [
    "REQ-CAPSTONE-4735",
    "SCENARIO-CAPSTONE-4735",
    "SCENARIO-CAPSTONE-4735-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4735-FIELD-PRINCIPLES",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
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


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _resolve_source(root: Path, source: SourceSpec) -> Path | None:
    pattern = root / source.relative_pattern
    matches = sorted(Path(hit) for hit in glob.glob(str(pattern)))
    if matches:
        return matches[0]
    exact = root / source.relative_pattern
    return exact if exact.exists() else None


def _summarize_and_live_flags(path: Path) -> tuple[int | None, list[JsonDict]]:
    if artifact_reader is None:
        return None, []
    exit_code = artifact_reader.summarize(path) if hasattr(artifact_reader, "summarize") else None
    raw_flags = artifact_reader._live_flags(path) if hasattr(artifact_reader, "_live_flags") else []
    return exit_code, [dict(flag) for flag in raw_flags if isinstance(flag, Mapping)]


def _severity(flag: Mapping[str, Any]) -> str:
    return str(flag.get("severity") or "").lower()


def _acceptance_gate_failures(artifact: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, value in artifact.items():
        lower = key.lower()
        if ("acceptance_gate" in lower or lower.startswith("gate_") or lower.endswith("_gate")) and value is False:
            failures.append(key)
    return failures


def _explicit_control_failed(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("control_failed") is True or artifact.get("control_passed") is False


def _explicit_ablation_missing(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("ablation_missing") is True or artifact.get("ablation_control_present") is False


def _source_status(
    *,
    name: str,
    source: SourceSpec,
    root: Path,
    artifact: Mapping[str, Any],
    path: Path | None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> JsonDict:
    if path is not None and live_flags_by_name is None:
        summary_exit_code, flags = _summarize_and_live_flags(path)
        read_via_summary = artifact_reader is not None
    else:
        summary_exit_code = None
        flags = [dict(flag) for flag in (live_flags_by_name or {}).get(name, [])]
        read_via_summary = path is not None
    critical = [dict(flag) for flag in flags if _severity(flag) == "critical"]
    stamped = artifact.get("flagged_adversarial") is True
    failed_gates = _acceptance_gate_failures(artifact)
    control_failed = _explicit_control_failed(artifact)
    ablation_missing = _explicit_ablation_missing(artifact)
    if path is None:
        reason = "missing"
    elif stamped or critical:
        reason = "flagged_adversarial_or_live_critical"
    elif failed_gates:
        reason = "failed_gate"
    elif control_failed:
        reason = "control_failed"
    elif ablation_missing:
        reason = "ablation_missing"
    else:
        reason = "included_clean"
    return {
        "name": name,
        "artifact": source.relative_pattern,
        "resolved_path": str(path.relative_to(root)) if path is not None and path.is_relative_to(root) else None,
        "role": source.role,
        "exists": path is not None,
        "honest_verdict": artifact.get("honest_verdict"),
        "stamped_flagged_adversarial": stamped,
        "live_critical": bool(critical),
        "live_flags": flags,
        "critical_flags": critical,
        "acceptance_gate_failures": failed_gates,
        "control_failed": control_failed,
        "ablation_missing": ablation_missing,
        "included_in_headline": reason == "included_clean",
        "reason": reason,
        "summary_exit_code": summary_exit_code,
        "sha256": _file_sha256(path) if path is not None else _checksum(artifact),
        "read_via_summarize_artifact": read_via_summary,
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
        if artifacts is not None:
            artifact = dict(artifacts[name]) if name in artifacts else {}
            path = root / source.relative_pattern if name in artifacts else None
        else:
            path = _resolve_source(root, source)
            artifact = _read_json(path) if path is not None else {}
        loaded[name] = artifact
        statuses[name] = _source_status(
            name=name,
            source=source,
            root=root,
            artifact=artifact,
            path=path,
            live_flags_by_name=live_flags_by_name,
        )
    return loaded, statuses


def _publication_gate_state(publication_gate: Mapping[str, Any] | None) -> JsonDict:
    gate = dict(publication_gate or {})
    return {
        "paper_ready": gate.get("paper_ready") is True,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "fover_09131_frozen_never_substituted": True,
        "gates": dict(gate.get("gates")) if isinstance(gate.get("gates"), Mapping) else {},
        "unmet_gates": list(gate.get("unmet_gates", [])) if isinstance(gate.get("unmet_gates"), list) else [],
    }


def _load_publication_gate() -> JsonDict:
    if publication_gate_reader is None:
        return {"paper_ready": False, "unmet_gates": ["publication_gate_unavailable"]}
    return dict(publication_gate_reader.evaluate())


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    statuses: Mapping[str, Mapping[str, Any]],
    registry_payload: Mapping[str, Any],
    publication_gate_available: bool,
) -> JsonDict:
    root_path = Path(root)
    doc_root = root_path if (root_path / "AGENTS.md").exists() else REPO_ROOT
    spec_path = doc_root / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    missing = [str(status["artifact"]) for status in statuses.values() if status.get("exists") is not True]
    checks: JsonDict = {
        "agents_md_read": (doc_root / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (doc_root / "CODEX.md").exists() or (doc_root / "OPENCODE.md").exists(),
        "spec_has_req_4735": "REQ-CAPSTONE-4735" in spec_text,
        "registry_yaml_loadable": bool(registry_payload),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_reproducible_total_levels": _as_int(registry_payload.get("reproducible_total_levels")),
        "summarize_artifact_py_available": (doc_root / "scripts" / "summarize_artifact.py").exists(),
        "summarize_artifact_py_used_for_every_present_upstream": all(
            status.get("read_via_summarize_artifact") for status in statuses.values() if status.get("exists")
        ),
        "publication_gate_py_runnable": publication_gate_available or (doc_root / "scripts" / "publication_gate.py").exists(),
        "upstream_artifacts_present": {name: status.get("exists") is True for name, status in statuses.items()},
        "missing_upstream_artifacts": missing,
        "leaderboard_submission": False,
        "operator_only": True,
        "research_conductor_modified": False,
    }
    failed: list[str] = []
    for key, resource in (
        ("agents_md_read", "agents_md"),
        ("codex_or_opencode_md_read", "codex_or_opencode_md"),
        ("spec_has_req_4735", "spec_req_4735"),
        ("registry_yaml_loadable", "registry_yaml"),
        ("summarize_artifact_py_available", "summarize_artifact"),
        ("publication_gate_py_runnable", "publication_gate"),
    ):
        if checks[key] is not True:
            failed.append(resource)
    if missing:
        failed.append("upstream_artifacts")
    checks["ok"] = not failed
    if failed:
        checks["blocked_resource"] = failed[0]
    return checks


def _clean(statuses: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return statuses.get(name, {}).get("included_in_headline") is True


def _target_game(artifact: Mapping[str, Any], default: str) -> str:
    value = artifact.get("target_game")
    if isinstance(value, str) and value:
        return value
    games = artifact.get("target_games")
    if isinstance(games, Sequence) and not isinstance(games, (str, bytes)) and games:
        return str(games[0])
    return default


def _a1_online_driver_result(
    a1: Mapping[str, Any],
    clean: bool,
    status: Mapping[str, Any],
) -> JsonDict:
    arms = a1.get("arms_non_degenerate") is True
    delta = _as_float(a1.get("online_warm_vs_frozen_delta"))
    beat = bool(clean and arms and delta >= 0.05)
    l2 = bool(
        clean
        and arms
        and a1.get("goal_free_l2_reached") is True
        and a1.get("offline_reproduced") is True
        and _as_int(a1.get("reproduced_levels")) >= 2
        and a1.get("solve_provenance") == "live_agent_self_discovery"
        and a1.get("verifier_is_oracle") is False
    )
    if l2:
        reason = "online_driver_goal_free_l2_offline_reproduced"
    elif beat:
        reason = "online_driver_non_degenerate_beat_frozen_by_0_05"
    elif not clean:
        reason = str(status.get("reason") or "source_not_clean")
    elif not arms:
        reason = "online_driver_arms_degenerate"
    elif a1.get("goal_free_l2_reached") is True and a1.get("offline_reproduced") is not True:
        reason = "goal_free_l2_not_offline_reproduced"
    elif a1.get("goal_free_l2_reached") is True and a1.get("solve_provenance") != "live_agent_self_discovery":
        reason = "solve_provenance_missing"
    elif delta < 0.05:
        reason = "online_warm_delta_below_0_05"
    else:  # pragma: no cover - exhaustive predicate guard
        reason = "online_driver_no_bridge"
    return {
        "included_in_headline": clean,
        "arms_non_degenerate": arms if clean else False,
        "per_arm_action_distribution_distinct": a1.get("per_arm_action_distribution_distinct") is True,
        "online_train_steps_executed": _as_int(a1.get("online_train_steps_executed")),
        "beat_frozen_by_0_05": beat,
        "goal_free_l2_banked": l2,
        "crossed": l2,
        "banked": l2,
        "online_warm_vs_frozen_delta": delta,
        "online_warm_first_win": _as_float(a1.get("online_warm_first_win")),
        "frozen_first_win": _as_float(a1.get("frozen_first_win")),
        "offline_reproduced": a1.get("offline_reproduced") is True,
        "reproduced_levels": _as_int(a1.get("reproduced_levels")),
        "solve_provenance": a1.get("solve_provenance"),
        "reason": reason,
    }


def _a2_active_probe_new_level(a2: Mapping[str, Any], clean: bool) -> JsonDict:
    game = _target_game(a2, "unknown")
    reached = _as_int(a2.get("generic_agent_reached_level"))
    ablation = _as_int(a2.get("no_probe_ablation_reached_level"))
    probed = (
        a2.get("hypothesis_posterior_built") is True
        and _as_int(a2.get("probe_actions_taken")) > 0
        and _as_float(a2.get("posterior_entropy_reduction")) > 0.0
    )
    offline = a2.get("offline_reproduced") is True
    provenance = a2.get("solve_provenance") == "live_agent_self_discovery"
    oracle_false = a2.get("verifier_is_oracle") is False
    crossed = bool(
        clean
        and probed
        and reached > 0
        and ablation < reached
        and offline
        and _as_int(a2.get("reproduced_levels")) >= reached
        and provenance
        and oracle_false
    )
    if crossed:
        reason = "active_probe_new_level_with_no_probe_ablation_failing"
    elif not clean:
        reason = "source_not_clean"
    elif not probed:
        reason = "probe_mechanism_did_not_run"
    elif reached <= 0:
        reason = "generic_agent_no_new_level"
    elif ablation >= reached:
        reason = "no_probe_ablation_not_lower"
    elif not offline:
        reason = "offline_reproduction_missing"
    elif not provenance:
        reason = "solve_provenance_missing"
    elif not oracle_false:
        reason = "verifier_oracle_not_false"
    else:  # pragma: no cover - exhaustive predicate guard
        reason = "not_active_probe_counted"
    return {
        "crossed": crossed,
        "banked": crossed,
        "surfaced": crossed,
        "target_game": game,
        "hypothesis_posterior_built": a2.get("hypothesis_posterior_built") is True,
        "probe_actions_taken": _as_int(a2.get("probe_actions_taken")),
        "posterior_entropy_reduction": _as_float(a2.get("posterior_entropy_reduction")),
        "generic_agent_reached_level": reached,
        "no_probe_ablation_reached_level": ablation,
        "offline_reproduced": offline,
        "reproduced_levels": _as_int(a2.get("reproduced_levels")),
        "solve_provenance": a2.get("solve_provenance"),
        "reason": reason,
    }


def _a3_levelup_banked(a3: Mapping[str, Any], clean: bool, registry_total: int) -> JsonDict:
    new_levels = _as_int(a3.get("new_levels_banked"))
    banked = bool(clean and a3.get("offline_reproduced") is True and new_levels >= 1 and registry_total >= 64)
    return {
        "banked": banked,
        "crossed": False,
        "target_game": _target_game(a3, "ar25"),
        "new_levels_banked": new_levels,
        "reproduced_levels": _as_int(a3.get("reproduced_levels")),
        "reproducible_total_levels_before": _as_int(a3.get("reproducible_total_levels_before"), 63),
        "reproducible_total_levels_after": registry_total,
        "solve_provenance": a3.get("solve_provenance"),
        "reason": "offline_reproduced_registry_delta_63_to_64" if banked else "no_clean_registry_bank",
    }


def _extract_reopen_list(b1: Mapping[str, Any]) -> list[str]:
    for key in ("silent_bug_must_reopen", "silent_bug_nulls", "silent_bug_reopen_list", "b1_silent_bug_reopen_list", "must_reopen"):
        value = b1.get(key)
        if isinstance(value, str) and value:
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            reopens: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    reopens.append(
                        str(
                            item.get("artifact")
                            or item.get("artifact_path")
                            or item.get("null_id")
                            or item.get("lever")
                            or item.get("id")
                            or item
                        )
                    )
                else:
                    reopens.append(str(item))
            return [item for item in reopens if item]
    return []


def _b1_reopened_434_a4(b1: Mapping[str, Any], reopens: Sequence[str]) -> bool:
    text = " ".join([str(b1.get("a4_tautology_verdict") or ""), *[str(item) for item in reopens]]).lower()
    return "a4" in text or "4715" in text or "online_driver" in text or "online action" in text


def _flagged_for_v436(d: Mapping[str, Any]) -> list[str]:
    roadmap = d.get("flagged_for_next_roadmap")
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, (str, bytes)):
        return []
    return [str(item) for item in roadmap if "flagged_for_v436" in str(item)]


def _verifier_false_confirmed(
    loaded: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
) -> bool:
    return all(
        loaded.get(name, {}).get("verifier_is_oracle") is False
        for name, status in statuses.items()
        if status.get("included_in_headline") is True and name != "PREVIOUS"
    )


def _solve_provenance_confirmed(
    loaded: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
) -> bool:
    for name, artifact in loaded.items():
        if statuses.get(name, {}).get("included_in_headline") is not True:
            continue
        solve_like = (
            artifact.get("offline_reproduced") is True
            or _as_int(artifact.get("reproduced_levels")) > 0
            or _as_int(artifact.get("new_levels_banked")) > 0
        )
        if solve_like and not artifact.get("solve_provenance"):
            return False
    return True


def _headline_verdict(
    *,
    preconditions: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    total: int,
) -> str:
    if preconditions.get("ok") is not True:
        return f"blocked_{preconditions.get('blocked_resource', 'precondition')}"
    if a1.get("goal_free_l2_banked") is True:
        return "complete: bridge_crossed_for_solve_online_driver_L2"
    if a2.get("crossed") is True:
        return f"complete: bridge_crossed_for_solve_{a2['target_game']}_L{a2['generic_agent_reached_level']}"
    if total > BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return f"complete: capability_grew_63_to_{total}"
    return "complete: no_bridge_crossed_capability_unchanged"


def _source_set_checksum(artifact: Mapping[str, Any]) -> str:
    return _checksum(
        {
            "cited_upstream_artifacts": artifact.get("cited_upstream_artifacts"),
            "missing_artifacts": artifact.get("missing_artifacts"),
            "reproducible_total_levels": artifact.get("reproducible_total_levels"),
            "publication_gate": artifact.get("publication_gate"),
        }
    )


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
    publication_raw = dict(publication_gate) if publication_gate is not None else _load_publication_gate()
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(
            root_path,
            statuses=statuses,
            registry_payload=registry_payload,
            publication_gate_available=publication_gate is not None or publication_gate_reader is not None,
        )
    )
    publication = _publication_gate_state(publication_raw)
    total = _as_int(registry_payload.get("reproducible_total_levels"))
    a1 = _a1_online_driver_result(loaded["A1"], _clean(statuses, "A1"), statuses["A1"])
    a2 = _a2_active_probe_new_level(loaded["A2"], _clean(statuses, "A2"))
    a3 = _a3_levelup_banked(loaded["A3"], _clean(statuses, "A3"), total)
    b1_reopens = _extract_reopen_list(loaded["B1"]) if _clean(statuses, "B1") else []
    b1_reopened_a4 = _b1_reopened_434_a4(loaded["B1"], b1_reopens) if _clean(statuses, "B1") else False
    skipped = [
        str(status["artifact"])
        for status in statuses.values()
        if status.get("exists") is True and status.get("included_in_headline") is not True
    ]
    missing = [str(status["artifact"]) for status in statuses.values() if status.get("exists") is not True]
    bridge = bool(a1["goal_free_l2_banked"] or a2["crossed"])
    scorecard = {
        name: {
            "verdict": loaded.get(name, {}).get("honest_verdict"),
            "clean": _clean(statuses, name),
            "reason": statuses.get(name, {}).get("reason"),
            "crossed": False,
            "banked": False,
            "surfaced": False,
        }
        for name in UPSTREAM_SOURCES
        if name != "PREVIOUS"
    }
    scorecard["A1"].update({"crossed": a1["crossed"], "banked": a1["banked"], "details": a1})
    scorecard["A2"].update({"crossed": a2["crossed"], "banked": a2["banked"], "surfaced": a2["surfaced"], "details": a2})
    scorecard["A3"].update({"banked": a3["banked"], "details": a3})
    scorecard["B1"].update({"surfaced": bool(b1_reopens), "details": {"reopen_list": b1_reopens, "reopened_434_a4": b1_reopened_a4}})
    strongest_open = "none"
    if not a2["crossed"]:
        strongest_open = "A2_active_probe_new_level"
    elif not a1["goal_free_l2_banked"]:
        strongest_open = "A1_online_driver_L2"
    elif b1_reopens:
        strongest_open = "B1_silent_bug_reopen"
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bridge_crossed_for_solve": bridge if preconditions.get("ok") is True else False,
        "a1_online_driver_result": a1,
        "a2_active_probe_new_level": a2,
        "a3_levelup_banked": a3,
        "b1_silent_bug_reopen_list": b1_reopens,
        "headline_decision": {
            "bridge_crossed_for_solve": bridge if preconditions.get("ok") is True else False,
            "capability_delta": total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "b1_silent_bug_reopen_list": b1_reopens,
            "b1_reopened_434_a4": b1_reopened_a4,
            "a1_arms_non_degenerate": a1["arms_non_degenerate"],
            "a1_beat_frozen_by_0_05": a1["beat_frozen_by_0_05"],
            "a1_deepened_to_l2": a1["goal_free_l2_banked"],
            "a2_active_probe_new_level": a2["crossed"],
            "a3_banked_63_to_64": a3["banked"],
        },
        "reproducible_total_levels": total,
        "reproducible_total_levels_delta": total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "publication_gate": publication,
        "verifier_is_oracle_confirmed_false": _verifier_false_confirmed(loaded, statuses),
        "solve_provenance_confirmed": _solve_provenance_confirmed(loaded, statuses),
        "skipped_artifacts": skipped,
        "missing_artifacts": missing,
        "next_milestone_fallback": {
            "flagged_for_v436": _flagged_for_v436(loaded["D"]) if _clean(statuses, "D") else [],
            "b1_reopen_list": b1_reopens,
            "strongest_open_lever": strongest_open,
        },
        "cited_upstream_artifacts": {name: dict(status) for name, status in statuses.items()},
        "scorecard": scorecard,
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = _headline_verdict(
        preconditions=preconditions,
        a1=a1,
        a2=a2,
        total=total,
    )
    artifact["reproducibility_checksum"] = _source_set_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare aggregation_from_upstream_artifacts")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("verifier_is_oracle_confirmed_false") is not True:
        errors.append("verifier_is_oracle_confirmed_false must be true")
    if artifact.get("publication_gate", {}).get("frozen_fover_auroc") != FROZEN_FOVER_AUROC:
        errors.append("publication_gate must preserve frozen FoVer 0.9131")
    if artifact.get("reproducibility_checksum") != _source_set_checksum(artifact):
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
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    live_flags_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    registry: Mapping[str, Any] | None = None,
    publication_gate: Mapping[str, Any] | None = None,
    write: bool = True,
    duration_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        artifacts=artifacts,
        live_flags_by_name=live_flags_by_name,
        registry=registry,
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
