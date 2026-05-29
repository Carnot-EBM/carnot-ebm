"""Build the Exp 3306 milestone .305 capstone artifact.

Spec refs: REQ-REPORT-3306, SCENARIO-REPORT-3306.

This module is a closeout ledger. It reads matrix v37 plus the prior .304
capstone and records which publication blockers moved. It deliberately does
not rerun Garak, repair, verifier scoring, FR-11 updates, KAN training, the
conductor, external submission, or any next-milestone activation.
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
RUN_DATE = "20260529"
SCHEMA_VERSION = "carnot.milestone_capstone.v305_matrix_v37_closeout.v1"
EXPERIMENT_ID = "exp3306"
TASK_ID = "exp3306-capstone-v305"
ARTIFACT = "experiment_3306_capstone_v305"
MILESTONE = "2026.05.305"
PRIOR_MILESTONE = "2026.05.304"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3306_capstone_v305.json")
RANDOM_SEED = 3306

MATRIX_V37_REL_PATH = Path("results/experiment_3305_evidence_matrix_v37.json")
CAPSTONE_V304_REL_PATH = Path("results/experiment_3293_capstone_v304.json")
PROTECTED_FILES = ("research-roadmap.yaml", "scripts/research_conductor.py")
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "capstone_v305_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v304",
    "garak_gate_passed",
    "garak_attack_success_rate",
    "repair_headline_claim_allowed",
    "fr11_memory_replay_safe",
    "kan_headline_retired",
    "next_top_gap",
    "recommended_next_milestone_title",
    "protected_files_untouched",
    "no_push",
    "no_next_milestone_activation",
    "source_artifacts",
    "inference_substrate",
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
    """Hash exact source bytes so later reviewers can reproduce this capstone."""

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
    """REQ-REPORT-3306: aggregate matrix v37 and the .304 capstone into closeout JSON."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V37_REL_PATH)
    prior_capstone = read_json_object(root_path / CAPSTONE_V304_REL_PATH)
    prior_count = _prior_publication_blocker_count(prior_capstone)
    publication_count = _publication_blocker_count(matrix, prior_count)
    garak_passed = _garak_gate_passed(matrix)
    garak_rate = _garak_attack_success_rate(matrix)
    repair_allowed = matrix.get("repair_headline_claim_allowed") is True
    fr11_safe = _fr11_memory_replay_safe(matrix)
    kan_retired = _kan_headline_retired(matrix)
    top_gap = _next_top_gap(matrix)
    paper_ready, paper_ready_blockers = _paper_ready_inputs(
        matrix,
        prior_capstone,
        publication_count=publication_count,
        garak_gate_passed=garak_passed,
        repair_headline_claim_allowed=repair_allowed,
        fr11_memory_replay_safe=fr11_safe,
        kan_headline_retired=kan_retired,
    )
    protected_status = _protected_file_status(root_path)
    protected_clean = all(not record["modified"] for record in protected_status.values())
    capstone_ready = (
        matrix.get("matrix_v37_ready") is True
        and prior_capstone.get("capstone_v304_ready") is True
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
        "capstone_v305_ready": capstone_ready,
        "paper_ready": paper_ready,
        "paper_ready_blockers": paper_ready_blockers,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_count,
        "blocker_delta_from_v304": publication_count - prior_count,
        "garak_gate_passed": garak_passed,
        "garak_attack_success_rate": garak_rate,
        "repair_headline_claim_allowed": repair_allowed,
        "fr11_memory_replay_safe": fr11_safe,
        "kan_headline_retired": kan_retired,
        "next_top_gap": top_gap,
        "recommended_next_milestone_title": _recommended_next_milestone_title(top_gap),
        "gate_status_details": _gate_status_details(matrix),
        "matrix_v37_summary": _matrix_summary(matrix),
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
        "no_new_fr11_weight_update": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_external_submission_or_publication": True,
        "no_push": True,
        "no_next_milestone_activation": True,
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
    """Build and persist the Exp 3306 capstone JSON."""

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
        raise ValueError("experiment_id must be exp3306")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3306-capstone-v305")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.305")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("publication_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while publication blockers remain")
    if artifact.get("no_push") is not True:
        raise ValueError("no_push must remain true")
    if artifact.get("no_next_milestone_activation") is not True:
        raise ValueError("no_next_milestone_activation must remain true")


def _source_artifacts(
    root: Path,
    matrix: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, MATRIX_V37_REL_PATH, "evidence_matrix_v37", matrix),
        _source_record(root, CAPSTONE_V304_REL_PATH, "prior_capstone_v304", prior_capstone),
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
    for path in (MATRIX_V37_REL_PATH, CAPSTONE_V304_REL_PATH):
        digest = sha256_file(root / path)
        if digest:
            checksums[path.as_posix()] = digest
    return checksums


def _prior_publication_blocker_count(prior_capstone: Mapping[str, Any]) -> int:
    return _int_value(prior_capstone.get("publication_blocker_count"))


def _publication_blocker_count(matrix: Mapping[str, Any], prior_count: int) -> int:
    if "paper_blocker_count" in matrix:
        return _int_value(matrix.get("paper_blocker_count"))
    if "publication_blocker_count" in matrix:
        return _int_value(matrix.get("publication_blocker_count"))
    return prior_count


def _paper_ready_inputs(
    matrix: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
    *,
    publication_count: int,
    garak_gate_passed: bool,
    repair_headline_claim_allowed: bool,
    fr11_memory_replay_safe: bool,
    kan_headline_retired: bool,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if matrix.get("matrix_v37_ready") is not True:
        blockers.append("matrix_v37_missing_or_not_ready")
    if prior_capstone.get("capstone_v304_ready") is not True:
        blockers.append("capstone_v304_missing_or_not_ready")
    if matrix.get("paper_ready") is not True:
        blockers.append("matrix_paper_ready_false")
    if publication_count != 0:
        blockers.append("publication_blockers_present")
    if not garak_gate_passed:
        blockers.append("garak_gate_not_passed")
    if not repair_headline_claim_allowed:
        blockers.append("repair_headline_claim_not_allowed")
    if not fr11_memory_replay_safe:
        blockers.append("fr11_memory_replay_not_safe")
    if not kan_headline_retired:
        blockers.append("kan_headline_not_retired")
    return not blockers, blockers


def _garak_gate_passed(matrix: Mapping[str, Any]) -> bool:
    return matrix.get("matrix_v37_ready") is True and matrix.get("garak_gate_passed") is True


def _garak_attack_success_rate(matrix: Mapping[str, Any]) -> float:
    direct = matrix.get("attack_success_rate")
    if _is_number(direct):
        return float(direct)
    gate = _as_mapping(_as_mapping(matrix.get("gate_summary")).get("garak_gate"))
    gate_rate = gate.get("attack_success_rate")
    if _is_number(gate_rate):
        return float(gate_rate)
    row = _row_by_id(_as_list(matrix.get("rows")), "exp3300")
    return _number_value(_as_mapping(row.get("summary")).get("attack_success_rate"))


def _fr11_memory_replay_safe(matrix: Mapping[str, Any]) -> bool:
    return matrix.get("matrix_v37_ready") is True and matrix.get("fr11_replay_safe") is True


def _kan_headline_retired(matrix: Mapping[str, Any]) -> bool:
    row = _row_by_id(_as_list(matrix.get("rows")), "exp3296")
    summary = _as_mapping(row.get("summary"))
    if summary.get("kan_prompt_injection_headline_retired") is True:
        return True
    return "kan_prompt_injection_headline_retired=true" in set(
        _list_of_strings(row.get("claim_boundaries"))
    )


def _next_top_gap(matrix: Mapping[str, Any]) -> str:
    if matrix.get("matrix_v37_ready") is not True:
        return "produce_ready_evidence_matrix_v37"
    return str(matrix.get("top_gap") or "publication_blocker_retirement_review")


def _recommended_next_milestone_title(next_top_gap: str) -> str:
    recommendations = {
        "produce_ready_evidence_matrix_v37": "Evidence Matrix V37 Repair Before Closeout",
        "pass_garak_redteam_gate": "Garak Red-Team Gate Pass",
        "clear_garak_dataflip_and_quality_flags": (
            "DataFlip And Quality-Flag Cleanup Before Publication Readiness"
        ),
        "clear_repair_headline_evidence_audit": "Repair Headline Evidence Audit Closure",
        "repair_fr11_controller_memory_replay_safety": "FR-11 Controller-Memory Replay Safety Repair",
        "bound_historical_flagged_evidence": "Historical Flag Boundary Corrigendum",
        "ready_for_v305_capstone": "Milestone .305 Archive And Handoff",
    }
    return recommendations.get(next_top_gap, "Publication Blocker Retirement Review")


def _gate_status_details(matrix: Mapping[str, Any]) -> JsonDict:
    return _as_mapping(matrix.get("gate_summary"))


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v37_ready": matrix.get("matrix_v37_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "paper_blocker_count": _int_value(matrix.get("paper_blocker_count")),
        "garak_gate_passed": matrix.get("garak_gate_passed") is True,
        "repair_headline_claim_allowed": matrix.get("repair_headline_claim_allowed") is True,
        "fr11_replay_safe": matrix.get("fr11_replay_safe") is True,
        "top_gap": _next_top_gap(matrix),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _prior_capstone_summary(prior_capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_v304_ready": prior_capstone.get("capstone_v304_ready") is True,
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
        "aggregation_only": "Capstone v305 reads matrix v37 and the .304 capstone only.",
        "paper_ready": "Publication readiness follows measured blockers, not task completion.",
        "blocker_delta": "Blocker movement is compared to the .304 capstone count.",
        "garak": "Garak attack-success gate pass is separated from DataFlip and quality flags.",
        "repair": "Repair evidence is not headline-eligible until the repair audit allows it.",
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
        "capstone_v305_ready": artifact.get("capstone_v305_ready"),
        "paper_ready": artifact.get("paper_ready"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "blocker_delta_from_v304": artifact.get("blocker_delta_from_v304"),
        "garak_gate_passed": artifact.get("garak_gate_passed"),
        "garak_attack_success_rate": artifact.get("garak_attack_success_rate"),
        "repair_headline_claim_allowed": artifact.get("repair_headline_claim_allowed"),
        "fr11_memory_replay_safe": artifact.get("fr11_memory_replay_safe"),
        "kan_headline_retired": artifact.get("kan_headline_retired"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v305_ready="
        f"{str(artifact.get('capstone_v305_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v304={artifact.get('blocker_delta_from_v304')}; "
        f"garak_gate_passed={str(artifact.get('garak_gate_passed') is True).lower()}; "
        f"garak_attack_success_rate={artifact.get('garak_attack_success_rate')}; "
        "repair_headline_claim_allowed="
        f"{str(artifact.get('repair_headline_claim_allowed') is True).lower()}; "
        f"fr11_memory_replay_safe={str(artifact.get('fr11_memory_replay_safe') is True).lower()}; "
        f"kan_headline_retired={str(artifact.get('kan_headline_retired') is True).lower()}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number_value(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
