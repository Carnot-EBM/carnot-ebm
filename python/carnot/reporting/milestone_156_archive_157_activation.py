"""Build the Exp 2008 `.156` archive and `.157` activation artifact.

Spec: REQ-REPORT-2008, SCENARIO-REPORT-2008.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260513"
PREDECESSOR_MILESTONE = "2026.05.156"
TARGET_MILESTONE = "2026.05.157"
EXPERIMENT = "2008_archive_156_activate_157"
SCHEMA = "carnot.milestone_156_archive_157_activation.v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_2008_archive_156_activate_157.json"

SOURCE_FILES = {
    "exp1996": "experiment_1996_nsvif_smt_extractor.json",
    "exp1997": "experiment_1997_llm_as_extractor.json",
    "exp1998": "experiment_1998_live_it_baselines_gsm8k.json",
    "exp1999": "experiment_1999_code_verification_humaneval.json",
    "exp2000": "experiment_2000_deep_sade_implementation.json",
    "exp2001": "experiment_2001_run_csp_message_passing.json",
    "exp2002": "experiment_2002_cold_decoding_integration.json",
    "exp2003": "experiment_2003_tier2_constraint_memory_fr11.json",
    "exp2004": "experiment_2004_ebm_transformer_reasoning_evaluation.json",
    "exp2005": "experiment_2005_adaptive_energy_landscapes_kan.json",
    "exp2006": "experiment_2006_milestone_156_pre_retro.json",
    "exp2007": "experiment_2007_milestone_156_retro.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "milestone",
    "predecessor_milestone",
    "success",
    "previous_milestone_artifacts_archived",
    "archive_move_required",
    "archive_artifacts",
    "missing_artifacts",
    "milestone_environment_ready",
    "roadmap_157_active",
    "conductor_activation_logged",
    "protected_files_unchanged",
    "handoff_requirements",
    "tests_run",
    "honest_verdict",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-2008: persist a started marker before source evidence reads."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "experiment_id": 2008,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "status": "in_progress",
        "success": False,
        "previous_milestone_artifacts_archived": False,
        "archive_move_required": False,
        "archive_artifacts": [],
        "missing_artifacts": [],
        "milestone_environment_ready": False,
        "roadmap_157_active": False,
        "conductor_activation_logged": False,
        "protected_files_unchanged": False,
        "handoff_requirements": {},
        "tests_run": [],
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _relative_path(path: Path) -> str:
    parts = path.parts
    for marker in ("results", "ops", "openspec"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) :]))
    return path.name


def _source_path(source_id: str) -> str:
    return f"results/{SOURCE_FILES[source_id]}"


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for source_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(f"results/{filename}")
        else:
            sources[source_id] = payload
    return sources, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _retro_complete(payload: Mapping[str, Any]) -> bool:
    return bool(
        _status(payload) in {"complete", "success"}
        and payload.get("milestone") == PREDECESSOR_MILESTONE
        and payload.get("retro_complete") is True
    )


def _research_complete_archives_156(text: str) -> bool:
    return f"id: {PREDECESSOR_MILESTONE}" in text and all(
        f"results/{filename}" in text for filename in SOURCE_FILES.values()
    )


def _roadmap_157_active(roadmap_text: str) -> bool:
    return TARGET_MILESTONE in roadmap_text and "exp2008-archive-156-activate-157" in roadmap_text


def _roadmap_doc_157_ready(roadmap_doc_text: str) -> bool:
    normalized = roadmap_doc_text.lower()
    return (
        TARGET_MILESTONE in roadmap_doc_text
        and "archive `.156`" in normalized
        and "initialize `.157`" in normalized
    )


def _conductor_activation_logged(conductor_log_text: str) -> bool:
    return f"Milestone {TARGET_MILESTONE} activated" in conductor_log_text


def _protected_files_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "research-roadmap.yaml", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _archive_artifacts(
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": source_id,
            "path": _source_path(source_id),
            "exists": source_id in sources,
            "status": sources.get(source_id, {}).get("status", "missing"),
            "honest_verdict": sources.get(source_id, {}).get("honest_verdict", "missing"),
        }
        for source_id in SOURCE_FILES
    ]


def _handoff_requirements(
    retro_payload: Mapping[str, Any],
    roadmap_text: str,
    roadmap_doc_text: str,
) -> dict[str, Any]:
    recommendations = [
        str(item) for item in retro_payload.get("recommendations", []) if item is not None
    ]
    combined = "\n".join(
        [
            *recommendations,
            str(retro_payload.get("gate_contract_gap_note") or ""),
            roadmap_text,
            roadmap_doc_text,
        ]
    ).lower()
    return {
        "prior_failures_required_for_reproposed_blocks": "prior_failures" in combined,
        "terminal_prefix_required": "terminal-prefix" in combined or "terminal prefix" in combined,
        "real_gguf_ebt_eval_required": "real gguf" in combined or "real inference" in combined,
        "blocked_experiments_to_preserve": list(retro_payload.get("blocked_experiments", [])),
        "recommendations": recommendations,
    }


def _predecessor_summary(retro_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "criteria_met": retro_payload.get("criteria_met"),
        "criteria_total": retro_payload.get("criteria_total"),
        "completed_task_count": retro_payload.get("completed_task_count"),
        "blocked_task_count": retro_payload.get("blocked_task_count"),
        "failed_task_count": retro_payload.get("failed_task_count"),
        "retro_complete": retro_payload.get("retro_complete"),
    }


def _source_inputs_read(
    *,
    research_complete_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    conductor_log_text: str,
) -> dict[str, dict[str, bool]]:
    inputs = {_source_path(source_id): {"exists": True} for source_id in SOURCE_FILES}
    inputs.update(
        {
            "research-complete.yaml": {"exists": bool(research_complete_text)},
            "research-roadmap.yaml": {"exists": bool(roadmap_text)},
            "openspec/change-proposals/research-roadmap-vNEXT.md": {
                "exists": bool(roadmap_doc_text)
            },
            "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        }
    )
    return inputs


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    research_complete_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    conductor_log_text: str,
    protected_files_unchanged: bool,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """SCENARIO-REPORT-2008: derive `.157` activation from `.156` evidence."""

    retro_payload = sources.get("exp2007", {})
    predecessor_retro_complete = _retro_complete(retro_payload)
    research_complete_archived = _research_complete_archives_156(research_complete_text)
    previous_milestone_artifacts_archived = bool(
        predecessor_retro_complete and not missing_source_paths and research_complete_archived
    )
    roadmap_active = _roadmap_157_active(roadmap_text)
    roadmap_doc_ready = _roadmap_doc_157_ready(roadmap_doc_text)
    activation_logged = _conductor_activation_logged(conductor_log_text)
    milestone_environment_ready = bool(
        roadmap_active and roadmap_doc_ready and activation_logged and protected_files_unchanged
    )
    success = bool(previous_milestone_artifacts_archived and milestone_environment_ready)

    blocked_reasons: list[str] = []
    if missing_source_paths:
        blocked_reasons.append("missing predecessor artifacts")
    if not predecessor_retro_complete:
        blocked_reasons.append("Exp 2007 does not report .156 retro completion")
    if not research_complete_archived:
        blocked_reasons.append("research-complete.yaml does not archive .156")
    if not roadmap_active:
        blocked_reasons.append("research-roadmap.yaml does not activate .157")
    if not roadmap_doc_ready:
        blocked_reasons.append("research-roadmap-vNEXT.md does not document .157 activation")
    if not activation_logged:
        blocked_reasons.append("conductor log does not record .157 activation")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed")

    status = "complete" if success else "blocked"
    honest_verdict = (
        "complete: milestone_156_archived_157_activation_ready"
        if success
        else "blocked: " + "; ".join(blocked_reasons)
    )

    return {
        "experiment": EXPERIMENT,
        "experiment_id": 2008,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": status,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "success": success,
        "previous_milestone_artifacts_archived": previous_milestone_artifacts_archived,
        "archive_move_required": not research_complete_archived,
        "archive_decision": (
            "no_directory_move_required_existing_research_complete_archive_keeps_results_canonical"
            if research_complete_archived
            else "archive_directory_move_or_research_complete_backfill_required"
        ),
        "archive_artifacts": _archive_artifacts(sources),
        "missing_artifacts": list(missing_source_paths),
        "milestone_environment_ready": milestone_environment_ready,
        "roadmap_157_active": roadmap_active,
        "roadmap_doc_157_ready": roadmap_doc_ready,
        "conductor_activation_logged": activation_logged,
        "protected_files_unchanged": protected_files_unchanged,
        "predecessor_summary": _predecessor_summary(retro_payload),
        "handoff_requirements": _handoff_requirements(
            retro_payload, roadmap_text, roadmap_doc_text
        ),
        "blocked_reasons": blocked_reasons,
        "source_inputs_read": _source_inputs_read(
            research_complete_text=research_complete_text,
            roadmap_text=roadmap_text,
            roadmap_doc_text=roadmap_doc_text,
            conductor_log_text=conductor_log_text,
        ),
        "tests_run": list(tests_run),
        "honest_verdict": honest_verdict,
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """REQ-REPORT-2008: write the `.156` archive and `.157` activation JSON."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_paths = _load_sources(root_path / "results")
    protected_clean = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    artifact = build_artifact(
        sources=sources,
        missing_source_paths=missing_source_paths,
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_doc_text=_read_text(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ),
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        protected_files_unchanged=protected_clean,
        tests_run=tests_run,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
