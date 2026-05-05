"""Build the Exp 1302 sandboxed skill-graph promotion/demotion artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1302_skill_graph_promotion_demotion_v2.json"

EXPERIMENT = "1302_skill_graph_promotion_demotion_v2"
SCHEMA = "skill_graph_promotion_demotion_v2"
RUN_DATE = "20260505"
SOURCE_FILE = "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
SOURCE_ARTIFACT = f"results/{SOURCE_FILE}"
PROMOTE_MIN_SUPPORT = 5
EXPIRE_MAX_SUPPORT = 1


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {
        "project_root": str(project_root),
        "run_date": run_date,
    }


def _empty_counts() -> dict[str, int]:
    return {
        "skill_graph_candidate_count": 0,
        "promoted_memory_count": 0,
        "demoted_memory_count": 0,
        "expired_memory_count": 0,
        "replay_evidence_count": 0,
    }


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1302-1: write a durable in-progress artifact before analysis."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifact": SOURCE_ARTIFACT,
            "candidate_artifact_path": f"results/{Path(out_path).name}",
            "status": "in_progress",
            **_empty_counts(),
            "memory_update_written": False,
            "honest_verdict": "in_progress",
        },
    )


def _routing_decision(record: Mapping[str, Any]) -> str:
    support = int(record.get("support") or 0)
    if support <= EXPIRE_MAX_SUPPORT:
        return "expire"
    if support >= PROMOTE_MIN_SUPPORT:
        return "promote"
    return "demote"


def _memory_tags(record: Mapping[str, Any]) -> list[str]:
    decision = str(record.get("selected_decision") or "")
    policy_tag = "repair_policy" if decision == "repair" else "accept_policy"
    memory_kind = "procedural" if decision == "repair" else "episodic"
    return [memory_kind, "verifier_feedback", policy_tag]


def _pattern_domain(pattern: str) -> str:
    domain, separator, _detail = pattern.partition(":")
    return domain if separator else pattern


def _candidate_entry(record: Mapping[str, Any], replay_count: int) -> dict[str, Any]:
    pattern = str(record.get("constraint_pattern") or "unknown")
    repair_hint = str(record.get("repair_hint") or "unknown")
    decision = str(record.get("selected_decision") or "unknown")
    verifier_result = str(record.get("verifier_result") or "unknown")
    support = int(record.get("support") or 0)
    routing_decision = _routing_decision(record)
    return {
        "skill_id": f"exp1302/{pattern.replace(':', '/')}/{decision}/{repair_hint}",
        "constraint_pattern": pattern,
        "constraint_domain": _pattern_domain(pattern),
        "verifier_result": verifier_result,
        "repair_hint": repair_hint,
        "selected_decision": decision,
        "memory_type_tags": _memory_tags(record),
        "memory_routing_decision": routing_decision,
        "replay_evidence": {
            "source_experiment": 1288,
            "support": support,
            "replay_slices_observed": replay_count,
            "verifier_backed": verifier_result in {"failed", "passed"},
        },
        "promotion_criteria": {
            "min_support": PROMOTE_MIN_SUPPORT,
            "requires_verifier_backing": True,
            "requires_memory_update_written": True,
        },
        "demotion_criteria": {
            "demote_when_support_below": PROMOTE_MIN_SUPPORT,
            "harmful_if_verifier_conflict": True,
            "count_separately_from_promotions": True,
        },
        "expiry_criteria": {
            "expire_when_support_at_or_below": EXPIRE_MAX_SUPPORT,
            "expire_when_stale_or_harmful": True,
        },
        "sandbox": {
            "production_skill_modified": False,
            "allowed_write_roots": ["results/"],
        },
    }


def build_artifact(
    exp1288_payload: Mapping[str, Any],
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-LEARN-1302-2/4: build the final sandboxed skill-graph artifact."""

    if exp1288_payload.get("memory_update_written") is not True:
        return {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifact": SOURCE_ARTIFACT,
            "candidate_artifact_path": f"results/{Path(out_path).name}",
            "status": "complete",
            **_empty_counts(),
            "memory_update_written": False,
            "skill_graph_candidates": [],
            "honest_verdict": "blocked_no_exp1288_memory_update",
        }

    records = list(exp1288_payload.get("clause_prediction_records") or [])
    replay_slices = list(exp1288_payload.get("replay_slices") or [])
    replay_count = len(replay_slices)
    candidates = [_candidate_entry(record, replay_count) for record in records]
    promoted = sum(entry["memory_routing_decision"] == "promote" for entry in candidates)
    demoted = sum(entry["memory_routing_decision"] == "demote" for entry in candidates)
    expired = sum(entry["memory_routing_decision"] == "expire" for entry in candidates)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifact": SOURCE_ARTIFACT,
        "candidate_artifact_path": f"results/{Path(out_path).name}",
        "status": "complete",
        "skill_graph_candidate_count": len(candidates),
        "promoted_memory_count": promoted,
        "demoted_memory_count": demoted,
        "expired_memory_count": expired,
        "replay_evidence_count": replay_count,
        "memory_update_written": bool(candidates),
        "skill_graph_candidates": candidates,
        "honest_verdict": "skill_graph_candidates_written_sandboxed",
    }


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1302-5: write a sandboxed results artifact, not live skills."""

    results_path = Path(results_dir)
    output_path = Path(out_path)
    write_in_progress_artifact(output_path, project_root=project_root, run_date=run_date)
    source_payload = json.loads((results_path / SOURCE_FILE).read_text(encoding="utf-8"))
    artifact = build_artifact(
        source_payload,
        project_root=project_root,
        run_date=run_date,
        out_path=output_path,
    )
    return _write_json(output_path, artifact)


__all__ = [
    "build_artifact",
    "run",
    "write_in_progress_artifact",
]
