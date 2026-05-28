"""Build the Exp 3293 milestone .304 capstone artifact.

Spec refs: REQ-REPORT-3293, SCENARIO-REPORT-3293.

This module is a closeout ledger. It reads matrix v36 plus the prior .303
capstone and records which blockers moved. It does not rerun Garak, repair,
verifier scoring, KAN training, the conductor, or any next-milestone action.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.milestone_capstone.v304_matrix_v36_closeout.v1"
EXPERIMENT_ID = "exp3293"
TASK_ID = "exp3293-capstone-v304"
ARTIFACT = "experiment_3293_capstone_v304"
MILESTONE = "2026.05.304"
PRIOR_MILESTONE = "2026.05.303"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
OUTPUT_REL_PATH = Path("results/experiment_3293_capstone_v304.json")
RANDOM_SEED = 3293

MATRIX_V36_REL_PATH = Path("results/experiment_3292_evidence_matrix_v36.json")
CAPSTONE_V303_REL_PATH = Path("results/experiment_3280_capstone_v303.json")
PROTECTED_FILES = ("research-roadmap.yaml", "scripts/research_conductor.py")
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
ALLOWED_KAN_BOUNDARY_DECISIONS = {
    "retire_from_prompt_injection_headline",
    "bounded_sidecar_only",
    "retired_from_headline",
    "rebuild_required_before_headline",
}
REQUIRED_ARTIFACT_FIELDS = {
    "capstone_v304_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v303",
    "garak_unblocked",
    "clean_verifier_abstention_unblocked",
    "kan_boundary_resolved",
    "repair_gate_open",
    "repair_micro_panel_ready",
    "fr11_memory_replay_safe",
    "next_top_gap",
    "recommended_next_milestone_title",
    "protected_files_untouched",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, treating absent, malformed, or list payloads as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact source bytes so the capstone can be reproduced later."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3293: aggregate matrix v36 and the .303 capstone into closeout JSON."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V36_REL_PATH)
    prior_capstone = read_json_object(root_path / CAPSTONE_V303_REL_PATH)
    rows = [_as_mapping(row) for row in _as_list(matrix.get("rows"))]
    gate_summary = _as_mapping(matrix.get("gate_summary"))
    prior_count = _prior_publication_blocker_count(prior_capstone, matrix)
    publication_count = _publication_blocker_count(matrix, prior_count)
    paper_ready, paper_ready_blockers = _paper_ready_inputs(
        matrix,
        rows,
        gate_summary,
        publication_count,
    )
    top_gap = _next_top_gap(matrix)
    protected_status = _protected_file_status(root_path)
    protected_clean = all(not record["modified"] for record in protected_status.values())
    garak_unblocked = _garak_unblocked(gate_summary)
    clean_verifier_unblocked = _clean_verifier_abstention_unblocked(gate_summary)
    kan_decision = _kan_boundary_decision(gate_summary, rows)
    kan_boundary_resolved = kan_decision in ALLOWED_KAN_BOUNDARY_DECISIONS
    repair_open = _repair_gate_open(gate_summary)
    repair_panel_ready = _repair_micro_panel_ready(gate_summary)
    repair_panel_headline = _repair_micro_panel_headline_eligible(gate_summary)
    fr11_safe = _fr11_memory_replay_safe(gate_summary, rows)
    capstone_ready = (
        matrix.get("matrix_v36_ready") is True
        and prior_capstone.get("capstone_v303_ready") is True
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": _principle_annotations(),
        "capstone_v304_ready": capstone_ready,
        "paper_ready": paper_ready,
        "paper_ready_blockers": paper_ready_blockers,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_count,
        "blocker_delta_from_v303": publication_count - prior_count,
        "garak_unblocked": garak_unblocked,
        "garak_gate_passed": _garak_gate_passed(gate_summary),
        "clean_verifier_abstention_unblocked": clean_verifier_unblocked,
        "kan_boundary_resolved": kan_boundary_resolved,
        "kan_boundary_decision": kan_decision,
        "repair_gate_open": repair_open,
        "repair_micro_panel_ready": repair_panel_ready,
        "repair_micro_panel_headline_eligible": repair_panel_headline,
        "fr11_memory_replay_safe": fr11_safe,
        "next_top_gap": top_gap,
        "recommended_next_milestone_title": _recommended_next_milestone_title(top_gap),
        "gate_status_details": _gate_status_details(gate_summary),
        "matrix_v36_summary": _matrix_summary(matrix),
        "prior_capstone_summary": _prior_capstone_summary(prior_capstone),
        "source_artifacts": _source_artifacts(root_path, matrix, prior_capstone),
        "source_checksums": _source_checksums(root_path),
        "protected_files_untouched": protected_clean,
        "protected_file_status": protected_status,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_next_milestone_activation": True,
        "no_external_submission_or_publication": True,
        "no_push": True,
        "research_roadmap_modified_by_this_task": False,
        "scripts_research_conductor_modified_by_this_task": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3293 capstone JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject capstones that omit required fields or overclaim publication readiness."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3293")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3293-capstone-v304")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.304")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("publication_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while publication blockers remain")


def _source_artifacts(
    root: Path,
    matrix: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, MATRIX_V36_REL_PATH, "evidence_matrix_v36", matrix),
        _source_record(root, CAPSTONE_V303_REL_PATH, "prior_capstone_v303", prior_capstone),
    ]


def _source_record(root: Path, path: Path, role: str, payload: Mapping[str, Any]) -> JsonDict:
    full_path = root / path
    return {
        "role": role,
        "path": path.as_posix(),
        "present": full_path.is_file(),
        "readable_json_object": bool(payload),
        "reported_experiment_id": str(payload.get("experiment_id") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(full_path),
    }


def _source_checksums(root: Path) -> JsonDict:
    checksums = {}
    for path in (MATRIX_V36_REL_PATH, CAPSTONE_V303_REL_PATH):
        digest = sha256_file(root / path)
        if digest:
            checksums[path.as_posix()] = digest
    return checksums


def _prior_publication_blocker_count(
    prior_capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> int:
    direct = _int_value(prior_capstone.get("publication_blocker_count"))
    if direct:
        return direct
    prior_matrix = _as_mapping(matrix.get("prior_matrix"))
    matrix_estimate = _int_value(prior_matrix.get("publication_blocker_count_estimate"))
    if matrix_estimate:
        return matrix_estimate
    exp3281 = _row_by_id(_as_list(matrix.get("rows")), "exp3281")
    return _int_value(_as_mapping(exp3281.get("summary")).get("prior_publication_blocker_count"))


def _publication_blocker_count(matrix: Mapping[str, Any], prior_count: int) -> int:
    if "publication_blocker_count" in matrix:
        return _int_value(matrix.get("publication_blocker_count"))
    if "paper_blocker_count" in matrix:
        return _int_value(matrix.get("paper_blocker_count"))
    return prior_count


def _paper_ready_inputs(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    gate_summary: Mapping[str, Any],
    publication_count: int,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if matrix.get("matrix_v36_ready") is not True:
        blockers.append("matrix_v36_missing_or_not_ready")
    if matrix.get("paper_ready") is not True:
        blockers.append("matrix_paper_ready_false")
    if publication_count != 0:
        blockers.append("publication_blockers_present")
    if _as_list(matrix.get("carried_forward_blockers")):
        blockers.append("carried_forward_dot303_blockers_present")
    if any(_row_blocks_headline(row) for row in rows):
        blockers.append("blocked_or_flagged_or_sidecar_rows_present")
    if not _garak_gate_passed(gate_summary):
        blockers.append("garak_gate_not_passed")
    if not _clean_verifier_abstention_unblocked(gate_summary):
        blockers.append("clean_verifier_abstention_not_unblocked")
    if _kan_boundary_decision(gate_summary, rows) not in ALLOWED_KAN_BOUNDARY_DECISIONS:
        blockers.append("kan_boundary_not_resolved")
    if not _repair_gate_open(gate_summary):
        blockers.append("repair_gate_not_open")
    if not _repair_micro_panel_headline_eligible(gate_summary):
        blockers.append("repair_micro_panel_not_headline_eligible")
    if not _fr11_memory_replay_safe(gate_summary, rows):
        blockers.append("fr11_memory_replay_not_safe")
    return not blockers, blockers


def _row_blocks_headline(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("status") or "") != "clean"
        or row.get("paper_blocking") is True
        or bool(_as_list(row.get("quality_flags")))
    )


def _garak_unblocked(gate_summary: Mapping[str, Any]) -> bool:
    toolchain = _as_mapping(gate_summary.get("garak_toolchain"))
    redteam = _as_mapping(gate_summary.get("garak_redteam"))
    return (
        toolchain.get("garak_runner_ready") is True
        and toolchain.get("garak_available") is True
        and redteam.get("garak_redteam_eval_ready") is True
    )


def _garak_gate_passed(gate_summary: Mapping[str, Any]) -> bool:
    redteam = _as_mapping(gate_summary.get("garak_redteam"))
    return redteam.get("garak_gate_passed") is True and redteam.get("dataflip_gate_passed") is True


def _clean_verifier_abstention_unblocked(gate_summary: Mapping[str, Any]) -> bool:
    clean = _as_mapping(gate_summary.get("clean_verifier"))
    return (
        clean.get("clean_verifier_rerun_ready") is True
        and clean.get("repair_gate_input_clean_enough") is True
        and _number_value(clean.get("abstention_rate")) <= 0.05
    )


def _kan_boundary_decision(gate_summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    kan_gate = _as_mapping(gate_summary.get("kan_boundary"))
    decision = str(kan_gate.get("kan_boundary_decision") or "")
    if decision:
        return decision
    row = _row_by_id(rows, "exp3288")
    return str(_as_mapping(row.get("summary")).get("kan_boundary_decision") or "")


def _repair_gate_open(gate_summary: Mapping[str, Any]) -> bool:
    return _as_mapping(gate_summary.get("repair_gate")).get("repair_gate_open") is True


def _repair_micro_panel_ready(gate_summary: Mapping[str, Any]) -> bool:
    panel = _as_mapping(gate_summary.get("repair_panel"))
    return panel.get("ready") is True and panel.get("repair_panel_ran") is True


def _repair_micro_panel_headline_eligible(gate_summary: Mapping[str, Any]) -> bool:
    panel = _as_mapping(gate_summary.get("repair_panel"))
    return (
        _repair_micro_panel_ready(gate_summary)
        and panel.get("headline_claim_allowed") is True
        and panel.get("status") == "clean"
    )


def _fr11_memory_replay_safe(gate_summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> bool:
    fr11 = _as_mapping(gate_summary.get("fr11"))
    row = _row_by_id(rows, "exp3291")
    summary = _as_mapping(row.get("summary"))
    raw_preserved = summary.get("raw_episodes_preserved")
    return (
        fr11.get("ready") is True
        and fr11.get("controller_memory_only") is True
        and fr11.get("foundation_weight_updates_performed") is False
        and raw_preserved is not False
    )


def _next_top_gap(matrix: Mapping[str, Any]) -> str:
    if matrix.get("matrix_v36_ready") is not True:
        return "produce_ready_evidence_matrix_v36"
    first = _as_mapping(_as_list(matrix.get("top_gaps"))[0]) if _as_list(matrix.get("top_gaps")) else {}
    return str(first.get("gap") or "publication_blocker_retirement_review")


def _recommended_next_milestone_title(next_top_gap: str) -> str:
    if next_top_gap == "pass_garak_redteam_gate":
        return "Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence"
    if next_top_gap == "repair_panel_duration_and_scope_boundary":
        return "Repair Panel Methodology Hardening"
    if next_top_gap == "resolve_dot303_methodology_flags":
        return "Prompt-Injection Methodology Corrigendum Closure"
    if next_top_gap == "produce_ready_evidence_matrix_v36":
        return "Evidence Matrix V36 Repair Before Closeout"
    return "Publication Blocker Retirement Review"


def _gate_status_details(gate_summary: Mapping[str, Any]) -> JsonDict:
    keys = (
        "garak_toolchain",
        "garak_redteam",
        "clean_verifier",
        "kan_boundary",
        "repair_gate",
        "repair_panel",
        "fr11",
    )
    return {key: _as_mapping(gate_summary.get(key)) for key in keys}


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v36_ready": matrix.get("matrix_v36_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "paper_blocker_count": _int_value(matrix.get("paper_blocker_count")),
        "top_gap": _next_top_gap(matrix),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _prior_capstone_summary(prior_capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_v303_ready": prior_capstone.get("capstone_v303_ready") is True,
        "paper_ready": prior_capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(prior_capstone.get("publication_blocker_count")),
        "next_top_gap": str(prior_capstone.get("next_top_gap") or ""),
        "honest_verdict": str(prior_capstone.get("honest_verdict") or ""),
    }


def _protected_file_status(root: Path) -> JsonDict:
    status = {
        path: {"modified": False, "git_status_available": False, "status": ""}
        for path in PROTECTED_FILES
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--", *PROTECTED_FILES],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return status
    if result.returncode != 0:
        return status
    modified = {}
    for line in result.stdout.splitlines():
        if len(line) >= 4:
            modified[line[3:].strip()] = line[:2]
    for path, record in status.items():
        record["git_status_available"] = True
        record["status"] = modified.get(path, "")
        record["modified"] = path in modified
    return status


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Capstone v304 reads matrix v36 and the .303 capstone only.",
        "paper_ready": "Publication requires clean, unblocked, headline-eligible evidence, not just completed artifacts.",
        "blocker_delta": "Blocker movement is compared to the .303 capstone count.",
        "garak": "Garak availability and Garak gate success are separated so runnable does not imply safe.",
        "repair": "Repair gate reopening is separated from headline-eligible repair evidence.",
        "protected_files": "Roadmap and conductor files stay untouched during closeout.",
    }


def _row_by_id(rows: list[Any], experiment_id: str) -> JsonDict:
    return next(
        (_as_mapping(row) for row in rows if _as_mapping(row).get("experiment_id") == experiment_id),
        {},
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "capstone_v304_ready": artifact.get("capstone_v304_ready"),
        "paper_ready": artifact.get("paper_ready"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "blocker_delta_from_v303": artifact.get("blocker_delta_from_v303"),
        "garak_unblocked": artifact.get("garak_unblocked"),
        "garak_gate_passed": artifact.get("garak_gate_passed"),
        "clean_verifier_abstention_unblocked": artifact.get(
            "clean_verifier_abstention_unblocked"
        ),
        "kan_boundary_resolved": artifact.get("kan_boundary_resolved"),
        "repair_gate_open": artifact.get("repair_gate_open"),
        "repair_micro_panel_ready": artifact.get("repair_micro_panel_ready"),
        "fr11_memory_replay_safe": artifact.get("fr11_memory_replay_safe"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v304_ready="
        f"{str(artifact.get('capstone_v304_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v303={artifact.get('blocker_delta_from_v303')}; "
        f"garak_unblocked={str(artifact.get('garak_unblocked') is True).lower()}; "
        f"garak_gate_passed={str(artifact.get('garak_gate_passed') is True).lower()}; "
        "clean_verifier_abstention_unblocked="
        f"{str(artifact.get('clean_verifier_abstention_unblocked') is True).lower()}; "
        f"kan_boundary_resolved={str(artifact.get('kan_boundary_resolved') is True).lower()}; "
        f"repair_gate_open={str(artifact.get('repair_gate_open') is True).lower()}; "
        f"fr11_memory_replay_safe={str(artifact.get('fr11_memory_replay_safe') is True).lower()}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
