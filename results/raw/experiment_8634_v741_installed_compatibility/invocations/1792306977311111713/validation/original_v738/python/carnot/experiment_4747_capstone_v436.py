"""Experiment 4747: .436 ARC sprint capstone scorecard.

This module aggregates .436 upstream artifacts after the disciplined artifact
summary reader has run. It makes no new solve claim, treats the ARC registry as
authoritative for capability growth, and preserves the frozen FoVer publication
gate.

Spec refs: REQ-CAPSTONE-4747, SCENARIO-CAPSTONE-4747,
SCENARIO-CAPSTONE-4747-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4747-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4747_capstone_v436"
SCHEMA = "carnot.exp4747.capstone_v436.v1"
RESULT_RELATIVE_PATH = "results/experiment_4747_capstone_v436.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4747
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 64
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
TERMINAL_PREFIXES = ("complete:", "blocked_", "success:", "passed:", "shipped:")
DEFERRED_REOPENS = ("P1_go_explore", "P4_subgoal", "A2_active_probe")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_pattern: str
    role: str


UPSTREAM_SOURCES: dict[str, SourceSpec] = {
    "PREVIOUS": SourceSpec("PREVIOUS", "results/experiment_4735_capstone_v435.json", "previous_v435_scorecard"),
    "A1": SourceSpec("A1", "results/experiment_4737_*.json", "goal_energy_candidate_generation"),
    "A2": SourceSpec("A2", "results/experiment_4738_*.json", "energy_fitness_qd_generation"),
    "A3": SourceSpec("A3", "results/experiment_4739_*.json", "levelup_selfplay_bank"),
    "A4": SourceSpec("A4", "results/experiment_4740_*.json", "held_out_readiness"),
    "A5": SourceSpec("A5", "results/experiment_4741_*.json", "primitive_persist"),
    "A6": SourceSpec("A6", "results/experiment_4742_*.json", "integration"),
    "B1": SourceSpec("B1", "results/experiment_4743_*.json", "adversarial_verify_carveout"),
    "B2": SourceSpec("B2", "results/experiment_4744_*.json", "submission_package_readiness"),
    "C": SourceSpec("C", "results/experiment_4745_*.json", "kv260"),
    "D": SourceSpec("D", "results/experiment_4746_*.json", "sota_ingestion"),
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete: <capability_grew_64_to_65 | bridge_crossed_for_solve_<game>_L<n>> "
            "-- the honest one-line milestone outcome."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "bridge_crossed_for_solve": {
        "principle": (
            "the HEADLINE DECISION -- True only if a GENERIC agent banked a NEW level via self-discovery "
            "(A1 goal-energy L2 | A2 energy-QD L2), offline-reproduced, control-passed; False if only "
            "capability/depth grew."
        )
    },
    "a1_goal_energy_result": {
        "principle": (
            "did A1 prove arms_non_degenerate AND beat baseline by >=+0.05 / deepen to L2 -- or confirm "
            "goal_energy_generation_arms_degenerate (a harness bug) or a real non-degenerate null?"
        )
    },
    "a2_energy_qd_result": {
        "principle": (
            "did A2 generate the winner (a NEW level via energy-fitness QD where naive search misses it) -- "
            "or confirm a harness bug / a real null?"
        )
    },
    "a3_banked_level": {
        "principle": "did A3 bank +1 (64->65)? the Level-Up Guarantee + self-play-every-milestone outcome."
    },
    "b1_carveout_fix_confirmed": {
        "principle": (
            "did B1 (INFRA) stop the honest non-degenerate-zero-lift null from being quarantined "
            "(the .435-A1 escape) + flag the declared-but-unrun mechanism (the .435-A2 escape)?"
        )
    },
    "submission_package_ready": {
        "principle": (
            "B2's submission-readiness (5 days to deadline) -- is the frozen-stack package ready for the "
            "operator to submit?"
        )
    },
    "reproducible_total_levels_delta": {
        "principle": (
            "the registry header delta this milestone (64->65 if A1/A2/A3 banked) -- the monotonic "
            "north-star metric."
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
            "the .437 direction (D's flagged_for_v437 + the deferred reopens P1/P4/A2-active-probe + "
            "the strongest open lever) -- the convergence hand-off."
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
    "REQ-CAPSTONE-4747",
    "SCENARIO-CAPSTONE-4747",
    "SCENARIO-CAPSTONE-4747-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4747-FIELD-PRINCIPLES",
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
    matches = sorted(Path(hit) for hit in glob.glob(str(root / source.relative_pattern)))
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
        "spec_has_req_4747": "REQ-CAPSTONE-4747" in spec_text,
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
        ("spec_has_req_4747", "spec_req_4747"),
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


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _a1_goal_energy_result(a1: Mapping[str, Any], clean: bool, status: Mapping[str, Any]) -> JsonDict:
    arms = bool(
        a1.get("arms_non_degenerate") is True
        and a1.get("candidate_pool_differs_from_baseline") is True
        and _as_float(a1.get("goal_energy_score_variance")) > 0.0
    )
    delta = _as_float(a1.get("goal_energy_vs_baseline_delta"))
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
    if not clean:
        reason = str(status.get("reason") or "source_not_clean")
    elif not arms:
        reason = "goal_energy_generation_arms_degenerate"
    elif l2:
        reason = "goal_energy_l2_offline_reproduced"
    elif beat:
        reason = "goal_energy_non_degenerate_beat_baseline_by_0_05"
    elif a1.get("goal_free_l2_reached") is True and a1.get("offline_reproduced") is not True:
        reason = "goal_energy_l2_not_offline_reproduced"
    elif a1.get("goal_free_l2_reached") is True and a1.get("solve_provenance") != "live_agent_self_discovery":
        reason = "solve_provenance_missing"
    elif abs(delta) < 1e-9 and _non_empty(a1.get("null_delta_methodology_note")) and a1.get("positive_control_passed") is True:
        reason = "goal_energy_real_non_degenerate_zero_lift_null"
    else:
        reason = "goal_energy_delta_below_0_05"
    return {
        "included_in_headline": clean,
        "arms_non_degenerate": arms if clean else False,
        "beat_baseline_by_0_05": beat,
        "deepened_to_l2": l2,
        "crossed": l2,
        "banked": l2,
        "generated": bool(beat or l2),
        "goal_energy_vs_baseline_delta": delta,
        "goal_energy_first_win": _as_float(a1.get("goal_energy_first_win")),
        "baseline_first_win": _as_float(a1.get("baseline_first_win")),
        "offline_reproduced": a1.get("offline_reproduced") is True,
        "reproduced_levels": _as_int(a1.get("reproduced_levels")),
        "solve_provenance": a1.get("solve_provenance"),
        "reason": reason,
    }


def _a2_energy_qd_result(a2: Mapping[str, Any], clean: bool, status: Mapping[str, Any]) -> JsonDict:
    arms = a2.get("arms_non_degenerate") is True and _as_int(a2.get("novel_candidates_generated")) > 0
    delta = _as_float(a2.get("energy_qd_vs_naive_delta"))
    qd_first = _as_float(a2.get("energy_qd_first_win"))
    naive_first = _as_float(a2.get("naive_search_first_win"))
    naive_missed = a2.get("naive_search_generated_winner") is False or naive_first < qd_first
    generated_winner = bool(
        clean
        and arms
        and naive_missed
        and (a2.get("winner_generated_by_energy_qd") is True or delta >= 0.05)
    )
    l2 = bool(
        clean
        and arms
        and generated_winner
        and a2.get("goal_free_l2_reached") is True
        and a2.get("offline_reproduced") is True
        and _as_int(a2.get("reproduced_levels")) >= 2
        and a2.get("solve_provenance") == "live_agent_self_discovery"
        and a2.get("verifier_is_oracle") is False
    )
    if not clean:
        reason = str(status.get("reason") or "source_not_clean")
    elif not arms:
        reason = "energy_qd_generation_arms_degenerate"
    elif l2:
        reason = "energy_qd_generated_winner_l2_offline_reproduced"
    elif a2.get("goal_free_l2_reached") is True and a2.get("offline_reproduced") is not True:
        reason = "energy_qd_l2_not_offline_reproduced"
    elif a2.get("goal_free_l2_reached") is True and a2.get("solve_provenance") != "live_agent_self_discovery":
        reason = "solve_provenance_missing"
    elif generated_winner:
        reason = "energy_qd_generated_winner_but_no_l2_bank"
    elif abs(delta) < 1e-9 and _non_empty(a2.get("null_delta_methodology_note")) and a2.get("positive_control_passed") is True:
        reason = "energy_qd_real_non_degenerate_zero_lift_null"
    else:
        reason = "energy_qd_delta_below_0_05_or_naive_not_missed"
    return {
        "included_in_headline": clean,
        "arms_non_degenerate": arms if clean else False,
        "generated_winner_where_naive_missed": generated_winner,
        "deepened_to_l2": l2,
        "crossed": l2,
        "banked": l2,
        "generated": generated_winner,
        "target_game": _target_game(a2, "energy_qd"),
        "energy_qd_vs_naive_delta": delta,
        "energy_qd_first_win": qd_first,
        "naive_search_first_win": naive_first,
        "novel_candidates_generated": _as_int(a2.get("novel_candidates_generated")),
        "offline_reproduced": a2.get("offline_reproduced") is True,
        "reproduced_levels": _as_int(a2.get("reproduced_levels")),
        "solve_provenance": a2.get("solve_provenance"),
        "reason": reason,
    }


def _a3_banked_level(a3: Mapping[str, Any], clean: bool, registry_total: int) -> JsonDict:
    new_levels = _as_int(a3.get("new_levels_banked"))
    banked = bool(clean and a3.get("offline_reproduced") is True and new_levels >= 1 and registry_total >= 65)
    return {
        "banked": banked,
        "crossed": False,
        "generated": False,
        "target_game": _target_game(a3, "re86"),
        "new_levels_banked": new_levels,
        "reproduced_levels": _as_int(a3.get("reproduced_levels")),
        "reproducible_total_levels_before": _as_int(a3.get("reproducible_total_levels_before"), 64),
        "reproducible_total_levels_after": registry_total,
        "solve_provenance": a3.get("solve_provenance"),
        "reason": "offline_reproduced_registry_delta_64_to_65" if banked else "no_clean_registry_bank",
    }


def _b1_confirmed(b1: Mapping[str, Any], clean: bool) -> bool:
    return bool(
        clean
        and (b1.get("tautology_carveout_added") or {}).get("passed") is True
        and (b1.get("exercise_evidence_extension_added") or {}).get("passed") is True
        and (b1.get("a1_exemplar_downgraded_to_warn") or {}).get("passed") is True
        and (b1.get("a2_exemplar_flagged") or {}).get("passed") is True
        and (b1.get("positive_exercise_null_not_flagged") or {}).get("passed") is True
    )


def _submission_ready(b2: Mapping[str, Any], clean: bool) -> bool:
    return bool(clean and b2.get("submission_package_ready") is True)


def _flagged_for_v437(d: Mapping[str, Any]) -> list[str]:
    roadmap = d.get("flagged_for_next_roadmap")
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, (str, bytes)):
        return []
    return [str(item) for item in roadmap if "flagged_for_v437" in str(item)]


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
    if a1.get("deepened_to_l2") is True:
        return "complete: bridge_crossed_for_solve_goal_energy_L2"
    if a2.get("crossed") is True:
        return f"complete: bridge_crossed_for_solve_{a2['target_game']}_L{a2['reproduced_levels']}"
    if total > BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return f"complete: capability_grew_64_to_{total}"
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
    a1 = _a1_goal_energy_result(loaded["A1"], _clean(statuses, "A1"), statuses["A1"])
    a2 = _a2_energy_qd_result(loaded["A2"], _clean(statuses, "A2"), statuses["A2"])
    a3 = _a3_banked_level(loaded["A3"], _clean(statuses, "A3"), total)
    b1_confirmed = _b1_confirmed(loaded["B1"], _clean(statuses, "B1"))
    b2_ready = _submission_ready(loaded["B2"], _clean(statuses, "B2"))
    skipped = [
        str(status["artifact"])
        for status in statuses.values()
        if status.get("exists") is True and status.get("included_in_headline") is not True
    ]
    missing = [str(status["artifact"]) for status in statuses.values() if status.get("exists") is not True]
    bridge = bool(a1["crossed"] or a2["crossed"])
    scorecard = {
        name: {
            "verdict": loaded.get(name, {}).get("honest_verdict"),
            "clean": _clean(statuses, name),
            "reason": statuses.get(name, {}).get("reason"),
            "crossed": False,
            "banked": False,
            "generated": False,
        }
        for name in UPSTREAM_SOURCES
        if name != "PREVIOUS"
    }
    scorecard["A1"].update({"crossed": a1["crossed"], "banked": a1["banked"], "generated": a1["generated"], "details": a1})
    scorecard["A2"].update({"crossed": a2["crossed"], "banked": a2["banked"], "generated": a2["generated"], "details": a2})
    scorecard["A3"].update({"banked": a3["banked"], "details": a3})
    scorecard["B1"].update({"banked": False, "generated": False, "details": {"carveout_fix_confirmed": b1_confirmed}})
    scorecard["B2"].update({"banked": False, "generated": False, "details": {"submission_package_ready": b2_ready}})
    if not a2["generated_winner_where_naive_missed"]:
        strongest_open = "A2_energy_qd_generation"
    elif not a1["generated"]:
        strongest_open = "A1_goal_energy_generation"
    elif not a3["banked"]:
        strongest_open = "A3_levelup_bank"
    elif not b2_ready:
        strongest_open = "B2_submission_package"
    else:
        strongest_open = "D_flagged_for_v437"
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bridge_crossed_for_solve": bridge if preconditions.get("ok") is True else False,
        "a1_goal_energy_result": a1,
        "a2_energy_qd_result": a2,
        "a3_banked_level": a3,
        "b1_carveout_fix_confirmed": b1_confirmed,
        "submission_package_ready": b2_ready,
        "headline_decision": {
            "bridge_crossed_for_solve": bridge if preconditions.get("ok") is True else False,
            "capability_delta": total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "a1_arms_non_degenerate": a1["arms_non_degenerate"],
            "a1_beat_baseline_by_0_05": a1["beat_baseline_by_0_05"],
            "a1_deepened_to_l2": a1["deepened_to_l2"],
            "a2_generated_winner_where_naive_missed": a2["generated_winner_where_naive_missed"],
            "a2_deepened_to_l2": a2["deepened_to_l2"],
            "a3_banked_64_to_65": a3["banked"],
            "b1_carveout_fix_confirmed": b1_confirmed,
            "submission_package_ready": b2_ready,
        },
        "reproducible_total_levels": total,
        "reproducible_total_levels_delta": total - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "publication_gate": publication,
        "verifier_is_oracle_confirmed_false": _verifier_false_confirmed(loaded, statuses),
        "solve_provenance_confirmed": _solve_provenance_confirmed(loaded, statuses),
        "skipped_artifacts": skipped,
        "missing_artifacts": missing,
        "next_milestone_fallback": {
            "flagged_for_v437": _flagged_for_v437(loaded["D"]) if _clean(statuses, "D") else [],
            "deferred_reopens": list(DEFERRED_REOPENS),
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
    artifact["honest_verdict"] = _headline_verdict(preconditions=preconditions, a1=a1, a2=a2, total=total)
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
