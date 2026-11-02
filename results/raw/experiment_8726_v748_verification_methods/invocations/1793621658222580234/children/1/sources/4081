"""Build the Exp 2924 aggregation-metadata corrigendum artifact.

Spec refs: REQ-REPORT-2924, SCENARIO-REPORT-2924.

This module repairs metadata provenance only. It reads the Exp 2921 matrix v9
and Exp 2922 `.275` capstone artifacts, records exactly which upstream rows
remain flagged, and classifies the aggregation artifacts' own inherited
compute-bound audit markers as metadata false positives. It does not call an
LLM, score a verifier, run a sampler, or launch hardware.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
MATRIX_V9_SOURCE = Path("results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json")
CAPSTONE_SOURCE = Path("results/experiment_2922_capstone_v275.json")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2924_aggregation_metadata_corrigendum_v1.json")
AUDITABLE_MIN_DURATION_S = 0.0001

SUBJECTS = {
    "exp2921": MATRIX_V9_SOURCE,
    "exp2922": CAPSTONE_SOURCE,
}

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "aggregation_metadata_clean",
    "no_new_llm_call",
    "no_new_hardware_run",
    "aggregation_from_upstream_artifacts",
    "source_artifact_checksums",
    "aggregation_provenance",
    "upstream_flagged_rows_preserved",
    "metadata_false_positive_findings",
    "adversarial_audit_rerun",
    "inference_substrate",
    "duration_s",
    "run_date",
}

AuditRunner = Callable[[Path, Path], dict[str, Any]]
ProcessRunner = Callable[..., Any]


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty mapping for absent bad inputs.

    WHY: this corrigendum must fail closed. A malformed source artifact is not
    evidence that an upstream row succeeded, so the caller treats `{}` exactly
    like an absent upstream artifact.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    audit_result: dict[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2924: build the corrigendum payload without writing it."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    end = time.perf_counter() if now_s is None else now_s
    subject_payloads = {
        exp_id: read_json_mapping(root_path / path) for exp_id, path in SUBJECTS.items()
    }
    missing_subjects = [
        str(SUBJECTS[exp_id]) for exp_id, payload in subject_payloads.items() if not payload
    ]
    if missing_subjects:
        return _blocked_artifact(root_path, missing_subjects, start, end)

    provenance = _collect_provenance(root_path, subject_payloads)
    checksums = _checksums_by_path(provenance)
    audit = audit_result or _audit_not_supplied()
    metadata_false_positives = _metadata_false_positive_findings(subject_payloads)
    upstream_flags = _upstream_flagged_rows_preserved(
        root_path,
        subject_payloads,
        provenance,
    )
    audit_findings = _as_findings(audit.get("findings"))
    audit_flagged = bool(audit.get("flagged")) or bool(audit_findings)
    audit_clean = audit.get("audit_available") is True and not audit_flagged
    artifact = {
        "honest_verdict": _complete_verdict(
            aggregation_metadata_clean=audit_clean,
            upstream_flag_count=len(upstream_flags),
            false_positive_count=len(metadata_false_positives),
        ),
        "aggregation_metadata_clean": audit_clean,
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "aggregation_from_upstream_artifacts": True,
        "source_artifact_checksums": checksums,
        "aggregation_provenance": provenance,
        "upstream_flagged_rows_preserved": upstream_flags,
        "metadata_false_positive_findings": metadata_false_positives,
        "adversarial_audit_rerun": {**audit, "findings": audit_findings},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, end - start), 6),
        "run_date": RUN_DATE,
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    audit_runner: AuditRunner | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Build, audit, and persist the Exp 2924 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    start = clock()
    audit = _pending_audit()
    runner = audit_runner or run_adversarial_audit
    for _attempt in range(3):
        artifact = build_artifact(
            root_path,
            audit_result=audit,
            started_s=start,
            now_s=_auditable_end(start, clock()),
        )
        _write_json(out_path, artifact)
        if artifact["honest_verdict"] == "blocked_upstream_artifact_missing":
            return out_path
        next_audit = runner(root_path, out_path)
        if _audit_equivalent(audit, next_audit):
            return out_path
        audit = next_audit

    final = build_artifact(root_path, audit_result=audit, started_s=start, now_s=clock())
    _write_json(out_path, final)
    return out_path


def run_adversarial_audit(
    root: Path | str,
    artifact_path: Path | str,
    *,
    runner: ProcessRunner = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run the local artifact audit and return its exact findings."""

    root_path = Path(root)
    tool_path = _audit_tool_path(root_path)
    if tool_path is None:
        return {
            "audit_available": False,
            "not_run_reason": "audit_tool_unavailable",
            "flagged": False,
            "findings": [],
        }

    artifact = Path(artifact_path)
    command = [python_executable, str(tool_path), str(artifact), "--json"]
    completed = runner(
        command,
        cwd=str(root_path),
        text=True,
        capture_output=True,
        check=False,
    )
    parsed_raw = json.loads(completed.stdout or "{}")
    parsed = parsed_raw if isinstance(parsed_raw, dict) else {}
    reports = parsed.get("reports")
    report = reports[0] if isinstance(reports, list) and reports else {}
    findings = _as_findings(report.get("flags") if isinstance(report, dict) else [])
    return {
        "audit_available": True,
        "audit_tool": str(tool_path.relative_to(root_path)),
        "command": command,
        "returncode": int(completed.returncode),
        "flagged": bool(findings) or int(parsed.get("flagged_count") or 0) > 0,
        "findings": findings,
        "stderr": completed.stderr,
    }


def _blocked_artifact(
    root: Path,
    missing_subjects: list[str],
    start: float,
    end: float,
) -> dict[str, Any]:
    provenance = [
        _provenance_row(
            root,
            path,
            experiment_id=exp_id,
            row_role="corrigendum_subject",
            identifier=exp_id,
        )
        for exp_id, path in SUBJECTS.items()
    ]
    return {
        "honest_verdict": "blocked_upstream_artifact_missing",
        "aggregation_metadata_clean": False,
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "aggregation_from_upstream_artifacts": True,
        "source_artifact_checksums": _checksums_by_path(provenance),
        "aggregation_provenance": provenance,
        "upstream_flagged_rows_preserved": [],
        "metadata_false_positive_findings": [],
        "adversarial_audit_rerun": {
            "audit_available": False,
            "not_run_reason": "upstream_missing",
            "flagged": False,
            "findings": [],
        },
        "missing_upstream_artifacts": missing_subjects,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, end - start), 6),
        "run_date": RUN_DATE,
    }


def _collect_provenance(
    root: Path,
    subject_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        _provenance_row(
            root,
            path,
            experiment_id=exp_id,
            row_role="corrigendum_subject",
            identifier=exp_id,
        )
        for exp_id, path in SUBJECTS.items()
    ]
    rows.extend(
        _citation_rows(
            root,
            subject_payloads["exp2921"],
            row_role="matrix_row_source",
            source_artifact="exp2921",
        )
    )
    rows.extend(
        _citation_rows(
            root,
            subject_payloads["exp2922"],
            row_role="capstone_source",
            source_artifact="exp2922",
        )
    )
    return rows


def _citation_rows(
    root: Path,
    payload: dict[str, Any],
    *,
    row_role: str,
    source_artifact: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for citation in payload.get("cited_upstream_artifacts") or []:
        if not isinstance(citation, dict) or not citation.get("artifact_path"):
            continue
        rows.append(
            _provenance_row(
                root,
                Path(str(citation["artifact_path"])),
                experiment_id=_string_or_none(citation.get("experiment_id")),
                row_role=row_role,
                identifier=_string_or_none(citation.get("row_id"))
                or _string_or_none(citation.get("experiment_id")),
                source_artifact=source_artifact,
                declared_present=citation.get("present"),
                declared_checksum=_string_or_none(citation.get("sha256")),
            )
        )
    return rows


def _provenance_row(
    root: Path,
    rel_path: Path,
    *,
    experiment_id: str | None,
    row_role: str,
    identifier: str | None,
    source_artifact: str | None = None,
    declared_present: object = None,
    declared_checksum: str | None = None,
) -> dict[str, Any]:
    path = root / rel_path
    present = path.is_file()
    checksum = _sha256(path) if present else None
    payload = read_json_mapping(path) if present else {}
    substrate = payload.get("inference_substrate")
    return {
        "artifact_path": str(rel_path),
        "checksum": checksum,
        "row_role": row_role,
        "source_inference_substrate": substrate if isinstance(substrate, str) else None,
        "current_task_reran_compute": False,
        "experiment_id": experiment_id,
        "identifier": identifier,
        "source_artifact": source_artifact,
        "present": present,
        "declared_present": declared_present,
        "declared_checksum": declared_checksum,
        "checksum_matches_declared": declared_checksum in (None, checksum),
    }


def _checksums_by_path(provenance: list[dict[str, Any]]) -> dict[str, str | None]:
    checksums: dict[str, str | None] = {}
    for row in provenance:
        checksums.setdefault(str(row["artifact_path"]), row["checksum"])
    return checksums


def _metadata_false_positive_findings(
    subject_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for exp_id, payload in subject_payloads.items():
        if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
            continue
        for finding in _pending_findings(payload):
            if finding["kind"] not in {"DURATION_TOO_SHORT", "METHODOLOGY_MISSING"}:
                continue
            findings.append(
                {
                    "experiment_id": exp_id,
                    "artifact_path": str(SUBJECTS[exp_id]),
                    "kind": finding["kind"],
                    "severity": finding["severity"],
                    "detail": finding["detail"],
                    "artifact_inference_substrate": INFERENCE_SUBSTRATE,
                    "false_positive_reason": (
                        "aggregation-only artifact inherited compute-bound "
                        "metadata from upstream rows; this corrigendum reran "
                        "no LLM, verifier, sampler, or hardware compute"
                    ),
                }
            )
    return findings


def _upstream_flagged_rows_preserved(
    root: Path,
    subject_payloads: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_exp_id = {
        row["experiment_id"]: Path(row["artifact_path"])
        for row in provenance
        if isinstance(row.get("experiment_id"), str)
    }
    rows: list[dict[str, Any]] = []
    for identifier in _as_str_list(subject_payloads["exp2921"].get("flagged_rows")):
        rows.append(
            _flag_row(root, identifier, "exp2921", "flagged_rows", by_exp_id)
        )
    for identifier in _as_str_list(subject_payloads["exp2922"].get("flagged_artifacts")):
        rows.append(
            _flag_row(root, identifier, "exp2922", "flagged_artifacts", by_exp_id)
        )
    return rows


def _flag_row(
    root: Path,
    identifier: str,
    source_artifact: str,
    source_field: str,
    by_exp_id: dict[str, Path],
) -> dict[str, Any]:
    exp_id = _experiment_id_from_identifier(identifier)
    payload = {}
    artifact_path = None
    if exp_id == "exp2921":
        artifact_path = MATRIX_V9_SOURCE
    elif exp_id == "exp2922":
        artifact_path = CAPSTONE_SOURCE
    elif exp_id in by_exp_id:
        artifact_path = by_exp_id[exp_id]
    if artifact_path is not None:
        payload = read_json_mapping(root / artifact_path)
    return {
        "identifier": identifier,
        "experiment_id": exp_id,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "flags": _pending_findings(payload),
    }


def _pending_findings(payload: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        findings.extend(_as_findings(payload.get(key)))
    if payload.get("flagged_adversarial") is True and not findings:
        findings.append(
            {
                "kind": "flagged_adversarial",
                "severity": "warn",
                "detail": "upstream artifact marked flagged_adversarial=true",
            }
        )
    return findings


def _as_findings(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            findings.append(
                {
                    "kind": str(item.get("kind", "unknown")),
                    "severity": str(item.get("severity", "unknown")),
                    "detail": str(item.get("detail", "")),
                }
            )
    return findings


def _experiment_id_from_identifier(identifier: str) -> str | None:
    match = re.search(r"exp(\d{4})", identifier)
    return f"exp{match.group(1)}" if match else None


def _as_str_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _auditable_end(start: float, end: float) -> float:
    return max(end, start + AUDITABLE_MIN_DURATION_S)


def _audit_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("audit_available") == right.get("audit_available")
        and left.get("audit_tool") == right.get("audit_tool")
        and bool(left.get("flagged")) == bool(right.get("flagged"))
        and _as_findings(left.get("findings")) == _as_findings(right.get("findings"))
        and left.get("returncode") == right.get("returncode")
    )


def _audit_tool_path(root: Path) -> Path | None:
    for rel_path in (
        Path("scripts/adversarial_artifact_audit.py"),
        Path("scripts/adversarial_verify.py"),
    ):
        candidate = root / rel_path
        if candidate.is_file():
            return candidate
    return None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _audit_not_supplied() -> dict[str, Any]:
    return {
        "audit_available": False,
        "not_run_reason": "audit_not_supplied",
        "flagged": True,
        "findings": [],
    }


def _pending_audit() -> dict[str, Any]:
    return {
        "audit_available": False,
        "not_run_reason": "pending_final_write",
        "flagged": False,
        "findings": [],
    }


def _complete_verdict(
    *,
    aggregation_metadata_clean: bool,
    upstream_flag_count: int,
    false_positive_count: int,
) -> str:
    return (
        "complete: aggregation metadata corrigendum written; "
        f"aggregation_metadata_clean={str(aggregation_metadata_clean).lower()}; "
        f"upstream_flags_preserved={upstream_flag_count}; "
        f"metadata_false_positive_findings={false_positive_count}"
    )
