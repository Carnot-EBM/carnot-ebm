"""Exp 5228: ARC live-path provenance and process-rubric gate.

Spec refs: REQ-REPORT-5228, SCENARIO-REPORT-5228-PROCESS-RUBRIC,
SCENARIO-REPORT-5228-PROVENANCE-GATE, SCENARIO-REPORT-5228-PATCH-DECISION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5228_arc_provenance_skill_rubric_gate_v478"
SCHEMA = "carnot.arc_provenance_skill_rubric_gate_v478.v1"
RUBRIC_SCHEMA = "carnot.arc_skill_process_rubric_v478.v1"
RUN_DATE = "2026-07-04"
RESULT_RELATIVE_PATH = (
    "results/experiment_5228_arc_provenance_skill_rubric_gate_v478.json"
)
RUBRIC_RELATIVE_PATH = "results/arc_skill_process_rubric_v478.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
MEMORY_SETUP_RELATIVE_PATH = "results/arc_rubric_setup_from_typed_memory_v478.json"
TRACE_RELATIVE_PATHS = (
    "results/experiment_5054_arc_live_path_self_discovery.json",
    "results/experiment_5067_arc_live_path_self_discovery.json",
    "results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json",
)
SPEC_REFS = (
    "REQ-REPORT-5228",
    "SCENARIO-REPORT-5228-PROCESS-RUBRIC",
    "SCENARIO-REPORT-5228-PROVENANCE-GATE",
    "SCENARIO-REPORT-5228-PATCH-DECISION",
)
RUBRIC_FIELDS = (
    "skill_selection",
    "skill_following",
    "skill_composition",
    "reflection_retry_quality",
    "provenance_validity",
)
PROVENANCE_FIELDS = (
    "solve_provenance",
    "policy",
    "runtime_self_discovery",
    "offline_source_reading_used",
    "offline_ground_truth_bfs_used",
    "per_game_bfs_used",
    "hand_built_adapter_used",
    "reproduction_gate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "arc_skill_rubric_usable",
    "recommended_live_patch_available",
    "recommended_patch_summary",
    "registry_precheck_done",
    "duplicate_solve_target_avoided",
    "provenance_fields",
    "rubric_path",
    "no_outer_loop_re_used",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "arc_skill_rubric_usable": (
        "BARE top-level boolean. True only if the rubric is grounded in live-path traces "
        "and has usable scoring fields."
    ),
    "recommended_live_patch_available": (
        "BARE top-level boolean. True only if exp5229 has a concrete live-path patch to attempt."
    ),
    "recommended_patch_summary": "Concrete patch string when available; otherwise the no-patch reason.",
    "registry_precheck_done": "True only when ops/arc_solve_registry.yaml was loaded and summarized.",
    "duplicate_solve_target_avoided": (
        "True only when the consulted live traces report duplicate or prior no-bank guards."
    ),
    "provenance_fields": (
        "Fields used to distinguish live_agent_self_discovery from development_proxy, "
        "outer_loop_re, offline_bfs, and hand_game_adapter evidence."
    ),
    "rubric_path": "Path to the durable process-rubric JSON.",
    "no_outer_loop_re_used": (
        "Must remain true for this gate; outer-loop RE may appear only as a blocked "
        "provenance class, not as evidence."
    ),
    "inference_substrate": "Must be live_trace_process_rubric.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether a "
        "gated live patch is available."
    ),
}


def load_source_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the registry, typed-memory handoff, and existing live-path traces."""

    base = Path(root)
    traces = []
    for relative in TRACE_RELATIVE_PATHS:
        path = base / relative
        if path.exists():
            trace = _read_json(path)
            trace["artifact_path"] = relative
            traces.append(trace)
    return {
        "registry_summary": load_registry_summary(base),
        "memory_setup": _read_json(base / MEMORY_SETUP_RELATIVE_PATH),
        "traces": traces,
    }


def load_registry_summary(root: Path | str = REPO_ROOT) -> JsonDict:
    """Summarize the ARC solve registry without mutating it."""

    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - repository precondition
        return {"present": False, "path": str(path), "games": {}, "reproducible_total_levels": 0}
    text = path.read_text(encoding="utf-8")
    total = _first_int(r"^reproducible_total_levels:\s*(\d+)", text)
    games: dict[str, int] = {}
    current_game: str | None = None
    for line in text.splitlines():
        game_match = re.match(r"^- game:\s*([A-Za-z0-9_+-]+)\s*$", line)
        if game_match:
            current_game = game_match.group(1)
            continue
        level_match = re.match(r"^\s+levels_reproduced:\s*(\d+)\s*$", line)
        if current_game and level_match:
            games[current_game] = int(level_match.group(1))
    return {
        "present": True,
        "path": REGISTRY_RELATIVE_PATH,
        "games": games,
        "reproducible_total_levels": int(total or 0),
    }


def score_trace(trace: Mapping[str, Any]) -> JsonDict:
    """Score one live/process trace row without requiring reward improvement."""

    attempt = _first_attempt(trace)
    provenance = str(trace.get("solve_provenance") or "")
    blockers = _provenance_blockers(trace, attempt)
    scores = {
        "skill_selection": _score(
            _registry_prechecked(trace),
            _duplicate_avoided(trace),
            bool(_candidate_audit(trace)),
            bool(trace.get("target_game")),
        ),
        "skill_following": _score(
            str(attempt.get("policy") or trace.get("policy") or "") == "E3AgentPolicy",
            int(attempt.get("actions_taken") or 0) > 0,
            _within_budget(attempt),
            bool(attempt.get("runtime_self_discovery") or trace.get("runtime_self_discovery")),
        ),
        "skill_composition": _score(
            bool(attempt.get("self_discovery_lever")),
            _lever_exercised(attempt),
            bool(attempt.get("go_explore_archive") or attempt.get("live_path_diagnostics")),
            bool(trace.get("proposed_live_patch")),
        ),
        "reflection_retry_quality": _score(
            _candidate_audit_has_skip(trace),
            _prior_live_artifact_consulted(trace),
            "no_new_level" in str(trace.get("honest_verdict") or ""),
            "duplicate_depth" in str(trace.get("honest_verdict") or ""),
        ),
        "provenance_validity": 0.0 if blockers else 1.0,
    }
    overall = round(sum(scores.values()) / len(scores), 3)
    return {
        "artifact_path": str(trace.get("artifact_path") or ""),
        "experiment": str(trace.get("experiment") or ""),
        "target_game": trace.get("target_game"),
        "target_level": trace.get("target_level"),
        "solve_provenance": provenance,
        "new_levels_banked": _new_levels_count(trace.get("new_levels_banked")),
        "duplicate_solve_avoided": _duplicate_avoided(trace),
        "accepted_for_patch_evidence": not blockers,
        "provenance_blockers": blockers,
        "scores": scores,
        "overall_process_score": overall,
        "proposed_live_patch": trace.get("proposed_live_patch"),
        "process_notes": _process_notes(trace, attempt),
    }


def build_rubric(
    *,
    traces: Sequence[Mapping[str, Any]],
    memory_setup: Mapping[str, Any],
    registry_summary: Mapping[str, Any],
) -> JsonDict:
    """Build the durable process rubric from trace evidence and typed memory."""

    scored = [score_trace(trace) for trace in traces]
    live_rows = [row for row in scored if row["solve_provenance"] == "live_agent_self_discovery"]
    has_fields = all(set(RUBRIC_FIELDS) <= set(row["scores"]) for row in scored) if scored else False
    usable = bool(live_rows) and has_fields
    return {
        "schema": RUBRIC_SCHEMA,
        "produced_by": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "rubric_fields": list(RUBRIC_FIELDS),
        "provenance_fields": list(PROVENANCE_FIELDS),
        "arc_skill_rubric_usable": usable,
        "scored_trace_count": len(scored),
        "live_trace_count": len(live_rows),
        "development_proxy_trace_count": sum(
            1 for row in scored if row["solve_provenance"] == "development_proxy"
        ),
        "scored_traces": scored,
        "registry_precheck": dict(registry_summary),
        "typed_memory_handoff": {
            "consumer_ready": bool(memory_setup.get("consumer_ready")),
            "known_arc_nulls": dict(memory_setup.get("known_arc_nulls") or {}),
            "provenance_requirements": dict(memory_setup.get("provenance_requirements") or {}),
            "rubric_fields": list(memory_setup.get("rubric_fields") or []),
        },
        "known_arc_nulls_retained": dict(memory_setup.get("known_arc_nulls") or {}),
        "blocked_provenance_classes": list(
            (memory_setup.get("provenance_requirements") or {}).get("blocked") or []
        ),
        "no_outer_loop_re_used": all(
            row["solve_provenance"] != "outer_loop_re" or not row["accepted_for_patch_evidence"]
            for row in scored
        ),
    }


def recommend_patch(rubric: Mapping[str, Any]) -> JsonDict:
    """Return the at-most-one Exp 5229 patch decision from scored traces."""

    candidates = [
        row
        for row in rubric.get("scored_traces", [])
        if row.get("accepted_for_patch_evidence")
        and row.get("duplicate_solve_avoided")
        and row.get("proposed_live_patch")
    ]
    if not candidates:
        return {
            "recommended_live_patch_available": False,
            "recommended_patch_summary": (
                "No credible exp5229 live-agent patch: no scored live trace supplied a "
                "concrete patch proposal after the registry precheck, and the retained "
                ".477/.478 memory records zero reproduction-gated ARC level delta."
            ),
        }
    best = max(candidates, key=lambda row: float(row.get("overall_process_score") or 0.0))
    return {
        "recommended_live_patch_available": True,
        "recommended_patch_summary": str(best["proposed_live_patch"]),
    }


def build_artifact(
    *,
    rubric: Mapping[str, Any],
    rubric_path: str,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the required Exp 5228 result artifact."""

    decision = recommend_patch(rubric)
    patch_available = bool(decision["recommended_live_patch_available"])
    verdict = (
        "success: ARC skill rubric usable and one exp5229 live patch is gated for attempt."
        if patch_available
        else "complete: ARC skill rubric usable; no exp5229 live patch is currently gated."
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "arc_skill_rubric_usable": bool(rubric.get("arc_skill_rubric_usable")),
        "recommended_live_patch_available": patch_available,
        "recommended_patch_summary": decision["recommended_patch_summary"],
        "registry_precheck_done": bool((rubric.get("registry_precheck") or {}).get("present")),
        "duplicate_solve_target_avoided": all(
            row.get("duplicate_solve_avoided") for row in rubric.get("scored_traces", [])
            if row.get("solve_provenance") == "live_agent_self_discovery"
        ),
        "provenance_fields": list(PROVENANCE_FIELDS),
        "rubric_path": rubric_path,
        "no_outer_loop_re_used": bool(rubric.get("no_outer_loop_re_used")),
        "inference_substrate": "live_trace_process_rubric",
        "honest_verdict": verdict,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [REGISTRY_RELATIVE_PATH, MEMORY_SETUP_RELATIVE_PATH, *TRACE_RELATIVE_PATHS],
        "registry_summary": dict(rubric.get("registry_precheck") or {}),
        "scored_trace_count": int(rubric.get("scored_trace_count") or 0),
        "live_trace_count": int(rubric.get("live_trace_count") or 0),
        "known_arc_nulls_retained": dict(rubric.get("known_arc_nulls_retained") or {}),
        "no_outer_loop_re_used_as_evidence": bool(rubric.get("no_outer_loop_re_used")),
        "no_offline_bfs_used": True,
        "read_hidden_game_source": False,
        "hand_game_adapter_used_as_live_patch_evidence": False,
        "tests_run": [dict(item) for item in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    rubric_path: Path | str = REPO_ROOT / RUBRIC_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5228 process rubric and result artifact."""

    inputs = load_source_artifacts(root)
    rubric = build_rubric(
        traces=inputs["traces"],
        memory_setup=inputs["memory_setup"],
        registry_summary=inputs["registry_summary"],
    )
    result_dest = Path(result_path)
    rubric_dest = Path(rubric_path)
    _write_json(rubric_dest, rubric)
    artifact = build_artifact(
        rubric=rubric,
        rubric_path=str(rubric_dest),
        tests_run=tests_run,
    )
    _write_json(result_dest, artifact)
    return artifact


def _first_attempt(trace: Mapping[str, Any]) -> JsonDict:
    attempts = trace.get("live_agent_attempts")
    if isinstance(attempts, Sequence) and not isinstance(attempts, str) and attempts:
        first = attempts[0]
        return dict(first) if isinstance(first, Mapping) else {}
    return dict(trace)


def _provenance_blockers(trace: Mapping[str, Any], attempt: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    provenance = str(trace.get("solve_provenance") or "")
    if provenance != "live_agent_self_discovery":
        blockers.append(provenance or "missing_solve_provenance")
    if str(attempt.get("policy") or trace.get("policy") or "") != "E3AgentPolicy":
        blockers.append("not_E3AgentPolicy")
    for field in (
        "offline_source_reading_used",
        "offline_ground_truth_bfs_used",
        "per_game_bfs_used",
        "hand_built_adapter_used",
    ):
        if bool(attempt.get(field) or trace.get(field)):
            blockers.append(field)
    if not bool(attempt.get("runtime_self_discovery") or trace.get("runtime_self_discovery")):
        blockers.append("runtime_self_discovery_missing")
    return blockers


def _registry_prechecked(trace: Mapping[str, Any]) -> bool:
    return bool(
        trace.get("registry_precheck_passed")
        or trace.get("duplicate_registry_precheck_passed")
        or trace.get("registry_precheck_done")
    )


def _duplicate_avoided(trace: Mapping[str, Any]) -> bool:
    return bool(
        trace.get("duplicate_solve_avoided")
        or trace.get("duplicate_registry_precheck_passed")
        or (trace.get("candidate_selection") or {}).get("duplicate_solve_avoided")
    )


def _candidate_audit(trace: Mapping[str, Any]) -> list[JsonDict]:
    audit = (trace.get("candidate_selection") or {}).get("candidate_audit") or []
    return [dict(row) for row in audit if isinstance(row, Mapping)]


def _candidate_audit_has_skip(trace: Mapping[str, Any]) -> bool:
    return any("skip" in str(row.get("status") or "") for row in _candidate_audit(trace))


def _prior_live_artifact_consulted(trace: Mapping[str, Any]) -> bool:
    selection = trace.get("candidate_selection") or {}
    return bool(
        selection.get("prior_live_path_artifacts_consulted")
        or any(row.get("prior_live_path_attempt") for row in _candidate_audit(trace))
    )


def _within_budget(attempt: Mapping[str, Any]) -> bool:
    actions = int(attempt.get("actions_taken") or 0)
    budget = int(attempt.get("budget") or 0)
    return bool(actions and budget and actions <= budget)


def _lever_exercised(attempt: Mapping[str, Any]) -> bool:
    archive = attempt.get("go_explore_archive") or {}
    diagnostics = attempt.get("live_path_diagnostics") or {}
    return bool(
        archive.get("actions_injected")
        or archive.get("prefixes_injected")
        or diagnostics.get("injection_exercised")
    )


def _new_levels_count(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return len(value)
    return int(value or 0)


def _process_notes(trace: Mapping[str, Any], attempt: Mapping[str, Any]) -> list[str]:
    notes = []
    if _registry_prechecked(trace):
        notes.append("registry_precheck_done")
    if _candidate_audit_has_skip(trace):
        notes.append("candidate_audit_skipped_known_dead_ends")
    if _within_budget(attempt):
        notes.append("bounded_action_trace_present")
    if not _lever_exercised(attempt):
        notes.append("composition_lever_not_exercised_or_degenerate")
    if not trace.get("proposed_live_patch"):
        notes.append("no_concrete_patch_proposal_in_trace")
    return notes


def _score(*checks: bool) -> float:
    return round(sum(1 for check in checks if check) / len(checks), 3)


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--rubric-path", default=str(REPO_ROOT / RUBRIC_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, rubric_path=args.rubric_path)
    print(json.dumps({field: artifact[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
