"""Build the Exp 2935 cross-corpus matrix v10 paper-boundary artifact.

Spec refs: REQ-REPORT-2935, SCENARIO-REPORT-2935.

This module is an evidence-boundary aggregator. It reads prior JSON artifacts,
records checksums and flags, and writes the matrix-v10 claim boundary without
rerunning any model, verifier, sampler, or hardware command.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.cross_corpus_matrix.v10_paper_boundary_corrigendum.v1"
ARTIFACT = "experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path(
    "results/experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1.json"
)
AUDITABLE_MIN_DURATION_S = 0.0001

MATRIX_V9_SOURCE = Path("results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json")
EXP2924_SOURCE = Path("results/experiment_2924_aggregation_metadata_corrigendum_v1.json")
EXP2925_SOURCE = Path(
    "results/experiment_2925_code_hallucination_taxonomy_provenance_corrigendum_v2.json"
)
EXP2926_SOURCE = Path("results/experiment_2926_constraintbench_constrained_output_rerun_v2.json")
EXP2927_SOURCE = Path("results/experiment_2927_gatemate_himbaechel_constraints_preflight_v3.json")
EXP2928_SOURCE = Path("results/experiment_2928_gatemate_n16_himbaechel_bitstream_build_v3.json")
EXP2929_SOURCE = Path("results/experiment_2929_gatemate_flash_timing_boundary_v1.json")
EXP2930_SOURCE = Path("results/experiment_2930_kv260_pbit_ssqa_scaling_projection_v1.json")
EXP2931_SOURCE = Path("results/experiment_2931_llmeval_logic_z3_mini_v1.json")
EXP2932_FIXTURE_SOURCE = Path("results/experiment_2932_citation_fixture_v1.json")
EXP2932_SOURCE = Path("results/experiment_2932_citation_hallucination_field_verifier_v1.json")
EXP2933_SOURCE = Path("results/experiment_2933_kan_cl_per_knot_self_learning_v1.json")
EXP2934_SOURCE = Path("results/experiment_2934_aquaforte_beaver_reformulation_pipeline_v1.json")


@dataclass(frozen=True)
class Dot276Source:
    experiment_id: str
    row_id: str
    corpus_or_task: str
    path: Path
    verifier_type: str


DOT276_SOURCES: tuple[Dot276Source, ...] = (
    Dot276Source(
        "exp2924",
        "exp2924_aggregation_metadata_corrigendum",
        "Matrix v9 aggregation-metadata corrigendum",
        EXP2924_SOURCE,
        "aggregation provenance audit",
    ),
    Dot276Source(
        "exp2925",
        "exp2925_taxonomy_corrigendum",
        "Code hallucination taxonomy provenance corrigendum",
        EXP2925_SOURCE,
        "deterministic taxonomy/provenance verifier",
    ),
    Dot276Source(
        "exp2926",
        "exp2926_constraintbench_corrigendum",
        "ConstraintBench constrained-output rerun",
        EXP2926_SOURCE,
        "live LLM plus constrained-output verifier",
    ),
    Dot276Source(
        "exp2927",
        "exp2927_gatemate_himbaechel_preflight",
        "GateMate himbaechel constraints preflight",
        EXP2927_SOURCE,
        "hardware toolchain preflight",
    ),
    Dot276Source(
        "exp2928",
        "exp2928_gatemate_bitstream",
        "GateMate n=16 himbaechel bitstream build",
        EXP2928_SOURCE,
        "hardware bitstream build",
    ),
    Dot276Source(
        "exp2929",
        "exp2929_gatemate_flash_timing_boundary",
        "GateMate flash smoke timing boundary",
        EXP2929_SOURCE,
        "physical board smoke",
    ),
    Dot276Source(
        "exp2930",
        "exp2930_kv260_scaling_projection",
        "KV260 p-bit/SSQA scaling projection",
        EXP2930_SOURCE,
        "projection-only resource accounting",
    ),
    Dot276Source(
        "exp2931",
        "exp2931_llmeval_logic_z3_mini",
        "LLMEval-Logic-style Z3 mini benchmark",
        EXP2931_SOURCE,
        "live LLM plus Z3 verifier",
    ),
    Dot276Source(
        "exp2932",
        "exp2932_citation_field_verifier",
        "Citation hallucination field verifier",
        EXP2932_SOURCE,
        "live LLM plus deterministic citation-field verifier",
    ),
    Dot276Source(
        "exp2933",
        "exp2933_kan_cl_self_learning",
        "KAN per-knot continuous self-learning",
        EXP2933_SOURCE,
        "local training simulation with exact verifier rewards",
    ),
    Dot276Source(
        "exp2934",
        "exp2934_reformulation_pipeline",
        "AquaForte/BEAVER reformulation pipeline",
        EXP2934_SOURCE,
        "live LLM plus exact verifier",
    ),
)
DOT276_SOURCE_BY_EXP = {source.experiment_id: source for source in DOT276_SOURCES}

REQUIRED_GATES: tuple[tuple[str, Path, str], ...] = (
    ("exp2924", EXP2924_SOURCE, "aggregation_metadata_clean"),
    ("exp2925", EXP2925_SOURCE, "taxonomy_corrigendum_clean"),
    ("exp2926", EXP2926_SOURCE, "constraintbench_corrigendum_ready"),
    ("exp2933", EXP2933_SOURCE, "kan_cl_self_learning_ready"),
)

SOURCE_PATHS: tuple[Path, ...] = (
    MATRIX_V9_SOURCE,
    *(source.path for source in DOT276_SOURCES),
    EXP2932_FIXTURE_SOURCE,
)

DOT276_HEADLINE_ELIGIBLE = {
    "exp2926_constraintbench_corrigendum",
    "exp2933_kan_cl_self_learning",
}
DOT276_SUPPORTING_PAPER_ELIGIBLE = {
    "exp2925_taxonomy_corrigendum",
}

AuditRunner = Callable[[Path, Path], dict[str, Any]]
ProcessRunner = Callable[..., Any]


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail closed to an empty mapping."""

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
    """REQ-REPORT-2935: build matrix v10 without invoking new compute."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    end = time.perf_counter() if now_s is None else now_s
    duration_s = round(max(0.0, end - start), 6)
    checksums = _source_artifact_checksums(root_path)
    payloads = _load_dot276_payloads(root_path)

    gate_errors = _required_gate_errors(payloads)
    matrix_v9 = read_json_mapping(root_path / MATRIX_V9_SOURCE)
    if not matrix_v9:
        gate_errors.append(
            {
                "experiment_id": "exp2921",
                "artifact_path": str(MATRIX_V9_SOURCE),
                "required_field": "matrix_rows",
                "actual_value": None,
                "row_id": "exp2921_matrix_v9",
            }
        )
    if gate_errors:
        return _blocked_artifact(checksums, gate_errors, duration_s)

    rows = _build_rows(matrix_v9, payloads, checksums)
    buckets = _bucket_rows(rows)
    headline_rows = [
        row["row_id"]
        for row in rows
        if row["headline_eligible"] is True and row["row_class"] == "clean"
    ]
    audit = _normalize_audit(audit_result or _audit_not_supplied())
    boundary = _paper_claim_boundary(True, headline_rows, rows)
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": _complete_verdict(buckets, headline_rows),
        "matrix_v10_ready": True,
        "matrix_v10_paper_boundary_ready": True,
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "row_classification_counts": buckets["row_classification_counts"],
        "headline_eligible_rows": headline_rows,
        "clean_rows": buckets["clean_rows"],
        "flagged_rows": buckets["flagged_rows"],
        "blocked_rows": buckets["blocked_rows"],
        "missing_rows": buckets["missing_rows"],
        "projection_only_rows": buckets["projection_only_rows"],
        "pilot_only_rows": buckets["pilot_only_rows"],
        "diagnostic_only_rows": buckets["diagnostic_only_rows"],
        "paper_claim_boundary": boundary,
        "source_artifact_checksums": checksums,
        "upstream_flags_preserved": _upstream_flags_preserved(rows, payloads["exp2924"]),
        "matrix_rows": rows,
        "adversarial_audit_rerun": audit,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    audit_runner: AuditRunner | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Build, locally audit, and persist the Exp 2935 deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    start = clock()
    first = build_artifact(
        root_path,
        audit_result=_pending_audit(),
        started_s=start,
        now_s=_auditable_end(start, clock()),
    )
    _write_json(out_path, first)
    if first["matrix_v10_ready"] is not True:
        return out_path

    runner = audit_runner or run_adversarial_audit
    audit = runner(root_path, out_path)
    final = build_artifact(
        root_path,
        audit_result=audit,
        started_s=start,
        now_s=_auditable_end(start, clock()),
    )
    _write_json(out_path, final)
    return out_path


def run_adversarial_audit(
    root: Path | str,
    artifact_path: Path | str,
    *,
    runner: ProcessRunner = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run the available local artifact audit and return normalized findings."""

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
    findings_source = report.get("flags") if isinstance(report, dict) else parsed.get("flags")
    findings = _as_findings(findings_source)
    return {
        "audit_available": True,
        "audit_tool": str(tool_path.relative_to(root_path)),
        "command": command,
        "returncode": int(completed.returncode),
        "flagged": bool(findings) or int(parsed.get("flagged_count") or 0) > 0,
        "findings": findings,
        "stderr": completed.stderr,
    }


def _load_dot276_payloads(root: Path) -> dict[str, dict[str, Any]]:
    return {
        source.experiment_id: read_json_mapping(root / source.path) for source in DOT276_SOURCES
    }


def _source_artifact_checksums(root: Path) -> dict[str, str | None]:
    checksums: dict[str, str | None] = {}
    for rel_path in SOURCE_PATHS:
        path = root / rel_path
        checksums[str(rel_path)] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return checksums


def _required_gate_errors(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for exp_id, path, field in REQUIRED_GATES:
        payload = payloads.get(exp_id, {})
        actual = payload.get(field) if payload else None
        if actual is not True:
            source = DOT276_SOURCE_BY_EXP[exp_id]
            errors.append(
                {
                    "experiment_id": exp_id,
                    "artifact_path": str(path),
                    "required_field": field,
                    "actual_value": actual,
                    "row_id": source.row_id,
                }
            )
    return errors


def _blocked_artifact(
    checksums: dict[str, str | None],
    gate_errors: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    blocked_rows = [
        error["row_id"] for error in gate_errors if error.get("actual_value") is not None
    ]
    missing_rows = [error["row_id"] for error in gate_errors if error.get("actual_value") is None]
    counts = _empty_counts()
    counts["blocked"] = len(blocked_rows)
    counts["missing"] = len(missing_rows)
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": "blocked_required_corrigendum_missing",
        "matrix_v10_ready": False,
        "matrix_v10_paper_boundary_ready": False,
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "row_classification_counts": counts,
        "headline_eligible_rows": [],
        "clean_rows": [],
        "flagged_rows": [],
        "blocked_rows": blocked_rows,
        "missing_rows": missing_rows,
        "projection_only_rows": [],
        "pilot_only_rows": [],
        "diagnostic_only_rows": [],
        "paper_claim_boundary": _paper_claim_boundary(False, [], []),
        "source_artifact_checksums": checksums,
        "upstream_flags_preserved": [],
        "matrix_rows": [],
        "required_gate_errors": gate_errors,
        "adversarial_audit_rerun": {
            "audit_available": False,
            "not_run_reason": "required_gate_failed",
            "flagged": False,
            "findings": [],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
    }


def _build_rows(
    matrix_v9: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    checksums: dict[str, str | None],
) -> list[dict[str, Any]]:
    rows = [_v9_matrix_row(row, checksums) for row in _v9_rows(matrix_v9)]
    rows.extend(
        _dot276_row(source, payloads.get(source.experiment_id, {}), checksums)
        for source in DOT276_SOURCES
    )
    return rows


def _v9_rows(matrix_v9: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix_v9.get("matrix_rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _v9_matrix_row(row: dict[str, Any], checksums: dict[str, str | None]) -> dict[str, Any]:
    row_id = str(row.get("row_id") or "v9_row_missing_id")
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    row_class = _classify_v9_row(row)
    flag_reasons = _v9_flag_reasons(row)
    inference_substrate = _row_inference_substrate(row, summary)
    headline_eligible = bool(row.get("headline_eligible")) and row_class == "clean"
    return _matrix_row(
        row_id=row_id,
        corpus_or_task=str(row.get("row_label") or row_id),
        verifier_type=str(row.get("row_kind") or "v9 carry-forward row"),
        row_class=row_class,
        source_experiment_id=str(row.get("source_experiment_id") or "exp2921"),
        source_artifact=str(row.get("source_artifact") or MATRIX_V9_SOURCE),
        source_checksum=str(row.get("source_sha256") or checksums[str(MATRIX_V9_SOURCE)]),
        inference_substrate=inference_substrate,
        hardware_substrate=_hardware_substrate(row_id, summary, inference_substrate),
        model_specs_if_live_llm=_live_model_specs(summary, inference_substrate),
        headline_eligible=headline_eligible,
        paper_claim_eligible=headline_eligible,
        claim_boundary=str(row.get("claim_boundary") or ""),
        non_eligible_reason="" if headline_eligible else _non_eligible_reason(row_class, row_id),
        upstream_flags=flag_reasons,
        source_summary=summary,
    )


def _dot276_row(
    source: Dot276Source,
    payload: dict[str, Any],
    checksums: dict[str, str | None],
) -> dict[str, Any]:
    row_class = _classify_dot276_source(source, payload)
    flag_reasons = _row_flag_reasons(payload)
    inference_substrate = _payload_inference_substrate(payload)
    headline_eligible = row_class == "clean" and source.row_id in DOT276_HEADLINE_ELIGIBLE
    paper_claim_eligible = headline_eligible or (
        row_class == "clean" and source.row_id in DOT276_SUPPORTING_PAPER_ELIGIBLE
    )
    return _matrix_row(
        row_id=source.row_id,
        corpus_or_task=source.corpus_or_task,
        verifier_type=source.verifier_type,
        row_class=row_class,
        source_experiment_id=source.experiment_id,
        source_artifact=str(source.path),
        source_checksum=checksums[str(source.path)],
        inference_substrate=inference_substrate,
        hardware_substrate=_hardware_substrate(source.row_id, payload, inference_substrate),
        model_specs_if_live_llm=_live_model_specs(payload, inference_substrate),
        headline_eligible=headline_eligible,
        paper_claim_eligible=paper_claim_eligible,
        claim_boundary=_dot276_claim_boundary(source.row_id, payload, paper_claim_eligible),
        non_eligible_reason=""
        if paper_claim_eligible
        else _non_eligible_reason(row_class, source.row_id),
        upstream_flags=flag_reasons,
        source_summary=_dot276_summary(payload),
    )


def _matrix_row(
    *,
    row_id: str,
    corpus_or_task: str,
    verifier_type: str,
    row_class: str,
    source_experiment_id: str,
    source_artifact: str,
    source_checksum: str | None,
    inference_substrate: str,
    hardware_substrate: str,
    model_specs_if_live_llm: list[dict[str, Any]],
    headline_eligible: bool,
    paper_claim_eligible: bool,
    claim_boundary: str,
    non_eligible_reason: str,
    upstream_flags: list[str],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "corpus_or_task": corpus_or_task,
        "verifier_type": verifier_type,
        "row_class": row_class,
        "source_experiment_id": source_experiment_id,
        "source_artifact": source_artifact,
        "source_checksum": source_checksum,
        "inference_substrate": inference_substrate,
        "hardware_substrate": hardware_substrate,
        "model_specs_if_live_llm": model_specs_if_live_llm,
        "headline_eligible": headline_eligible,
        "paper_claim_eligible": paper_claim_eligible,
        "claim_boundary": claim_boundary,
        "non_eligible_reason": non_eligible_reason,
        "upstream_flags": upstream_flags,
        "source_summary": source_summary,
    }


def _classify_v9_row(row: dict[str, Any]) -> str:
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    original_status = str(row.get("row_status") or "")
    if summary.get("pilot_only") is True or original_status == "pilot_only":
        return "pilot_only"
    if _v9_flag_reasons(row) or original_status == "flagged" or "flagged" in original_status:
        return "flagged"
    if original_status == "projection_only":
        return "projection_only"
    if original_status == "diagnostic_only":
        return "diagnostic_only"
    if original_status == "missing":
        return "missing"
    if original_status == "blocked":
        return "blocked"
    return "clean" if original_status == "clean" else "blocked"


def _classify_dot276_source(source: Dot276Source, payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    exp_id = source.experiment_id
    if exp_id == "exp2924":
        return "clean" if payload.get("aggregation_metadata_clean") is True else "blocked"
    if exp_id == "exp2925":
        return "clean" if payload.get("taxonomy_corrigendum_clean") is True else "blocked"
    if exp_id == "exp2930" and payload.get("projection_only") is True:
        return "projection_only"
    if _has_unresolved_flags(payload):
        return "flagged"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if exp_id == "exp2926":
        return "clean" if payload.get("constraintbench_corrigendum_ready") is True else "blocked"
    if exp_id == "exp2933":
        return "clean" if payload.get("kan_cl_self_learning_ready") is True else "blocked"
    readiness_fields = (
        "gatemate_himbaechel_ready",
        "gatemate_bitstream_built",
        "gatemate_flash_smoke_ready",
        "logic_verifier_mini_ready",
        "citation_verifier_ready",
        "reformulation_pipeline_ready",
    )
    if any(payload.get(field) is True for field in readiness_fields):
        return "clean"
    return "clean" if str(payload.get("honest_verdict") or "").startswith("complete") else "blocked"


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _empty_counts()
    buckets = {
        "row_classification_counts": counts,
        "clean_rows": [],
        "flagged_rows": [],
        "blocked_rows": [],
        "missing_rows": [],
        "projection_only_rows": [],
        "pilot_only_rows": [],
        "diagnostic_only_rows": [],
    }
    for row in rows:
        row_id = row["row_id"]
        row_class = row["row_class"]
        counts[row_class] += 1
        bucket_name = f"{row_class}_rows"
        if bucket_name in buckets:
            buckets[bucket_name].append(row_id)
    return buckets


def _empty_counts() -> dict[str, int]:
    return {
        "clean": 0,
        "flagged": 0,
        "blocked": 0,
        "missing": 0,
        "projection_only": 0,
        "diagnostic_only": 0,
        "pilot_only": 0,
    }


def _paper_claim_boundary(
    ready: bool,
    headline_eligible_rows: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    headline_claims = {
        row["row_id"]: row["claim_boundary"]
        for row in rows
        if row["row_id"] in headline_eligible_rows
    }
    supporting = [
        row["row_id"]
        for row in rows
        if row["paper_claim_eligible"] and row["row_id"] not in headline_eligible_rows
    ]
    non_paper = {
        row["row_id"]: {
            "row_class": row["row_class"],
            "reason": row["non_eligible_reason"],
            "source_artifact": row["source_artifact"],
        }
        for row in rows
        if not row["paper_claim_eligible"]
    }
    return {
        "ready": ready,
        "headline_eligible_rows": headline_eligible_rows,
        "supporting_paper_claim_rows": supporting,
        "headline_claims": headline_claims,
        "non_paper_claim_rows": non_paper,
        "boundary_rules": [
            "Only clean rows with direct bounded support are headline eligible.",
            "Flagged, blocked, missing, diagnostic-only, and pilot-only rows remain non-headline.",
            "Projection-only rows are retained for planning context but are not paper-claim eligible.",
            "A clean corrigendum can add a replacement/supporting row without laundering the older flagged row.",
        ],
    }


def _upstream_flags_preserved(
    rows: list[dict[str, Any]],
    exp2924_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    preserved = [
        {
            "identifier": row["row_id"],
            "row_class": row["row_class"],
            "source_artifact": row["source_artifact"],
            "flags": row["upstream_flags"],
        }
        for row in rows
        if row["upstream_flags"]
    ]
    for item in exp2924_payload.get("upstream_flagged_rows_preserved") or []:
        if isinstance(item, dict):
            preserved.append(
                {
                    "identifier": str(item.get("identifier", "")),
                    "row_class": "preserved_upstream_flag",
                    "source_artifact": str(EXP2924_SOURCE),
                    "flags": _as_string_list(item.get("flags")),
                }
            )
    return preserved


def _v9_flag_reasons(row: dict[str, Any]) -> list[str]:
    reasons = _as_string_list(row.get("flag_reasons"))
    if row.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial=true")
    return reasons


def _row_flag_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial=true")
    if payload.get("adversarial_verify_passed") is False:
        reasons.append("adversarial_verify_passed=false")
    for finding in _as_findings(payload.get("corrigendum_pending")):
        reasons.append(f"{finding['kind']}:{finding['severity']}")
    for finding in _as_findings(payload.get("adversarial_verify_flags")):
        reasons.append(f"{finding['kind']}:{finding['severity']}")
    audit = payload.get("adversarial_audit_rerun")
    if isinstance(audit, dict) and audit.get("flagged") is True:
        reasons.append("adversarial_audit_rerun_flagged=true")
    return reasons


def _has_unresolved_flags(payload: dict[str, Any]) -> bool:
    return bool(_row_flag_reasons(payload))


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


def _as_string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _row_inference_substrate(row: dict[str, Any], summary: dict[str, Any]) -> str:
    for candidate in (summary.get("inference_substrate"), row.get("inference_substrate")):
        if isinstance(candidate, str) and candidate:
            return candidate
    return INFERENCE_SUBSTRATE


def _payload_inference_substrate(payload: dict[str, Any]) -> str:
    value = payload.get("inference_substrate")
    return value if isinstance(value, str) and value else "unknown"


def _live_model_specs(payload: dict[str, Any], inference_substrate: str) -> list[dict[str, Any]]:
    if "live_llm" not in inference_substrate:
        return []
    specs = payload.get("model_specs")
    return (
        [dict(item) for item in specs if isinstance(item, dict)] if isinstance(specs, list) else []
    )


def _hardware_substrate(row_id: str, payload: dict[str, Any], inference_substrate: str) -> str:
    text = f"{row_id} {json.dumps(payload, sort_keys=True)} {inference_substrate}".lower()
    if "gatemate" in text or "ccgm" in text:
        return "GateMate CCGM1A1"
    if "kv260" in text:
        return "KV260"
    if "hardware" in inference_substrate or "physical_board" in inference_substrate:
        return "hardware substrate declared by source"
    return "none"


def _dot276_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "honest_verdict",
        "aggregation_metadata_clean",
        "taxonomy_corrigendum_clean",
        "constraintbench_corrigendum_ready",
        "gatemate_himbaechel_ready",
        "gatemate_bitstream_built",
        "gatemate_flash_smoke_ready",
        "kv260_scaling_projection_ready",
        "projection_only",
        "logic_verifier_mini_ready",
        "citation_verifier_ready",
        "kan_cl_self_learning_ready",
        "reformulation_pipeline_ready",
        "syntax_valid_rate",
        "feasibility_rate_overall",
        "optimality_rate_given_feasible",
        "forgetting_rate",
        "utility_delta_vs_replay_only",
    )
    return {key: payload[key] for key in summary_keys if key in payload}


def _dot276_claim_boundary(
    row_id: str,
    payload: dict[str, Any],
    paper_claim_eligible: bool,
) -> str:
    if not paper_claim_eligible:
        return ""
    if row_id == "exp2925_taxonomy_corrigendum":
        return "Supporting paper claim: deterministic code-taxonomy provenance is clean; no code-corpus metric is promoted."
    if row_id == "exp2926_constraintbench_corrigendum":
        return (
            "Bounded ConstraintBench claim: live local GGUF constrained-output rerun "
            f"reports syntax_valid_rate={payload.get('syntax_valid_rate')}, "
            f"feasibility_rate_overall={payload.get('feasibility_rate_overall')}, and "
            f"optimality_rate_given_feasible={payload.get('optimality_rate_given_feasible')}."
        )
    if row_id == "exp2933_kan_cl_self_learning":
        return (
            "Bounded self-learning claim: KAN per-knot structural-memory update "
            f"reports utility_delta_vs_replay_only={payload.get('utility_delta_vs_replay_only')} "
            f"with forgetting_rate={payload.get('forgetting_rate')}."
        )
    return "Bounded clean supporting claim; no broader generalization is implied."


def _non_eligible_reason(row_class: str, row_id: str) -> str:
    reasons = {
        "flagged": "excluded because unresolved upstream/adversarial flags remain",
        "blocked": "excluded because the source artifact is blocked",
        "missing": "excluded because the expected source artifact is missing",
        "projection_only": "excluded because projection-only rows cannot support paper claims",
        "diagnostic_only": "excluded because diagnostic-only rows are context, not claims",
        "pilot_only": "excluded because pilot-only rows cannot support headline claims",
    }
    return reasons.get(row_class, f"{row_id} is clean support/context but not claim-eligible")


def _complete_verdict(buckets: dict[str, Any], headline_rows: list[str]) -> str:
    counts = buckets["row_classification_counts"]
    return (
        "complete: matrix_v10_ready=true; matrix_v10_paper_boundary_ready=true; "
        f"headline_eligible_rows={len(headline_rows)}; clean={counts['clean']}; "
        f"flagged={counts['flagged']}; blocked={counts['blocked']}; "
        f"missing={counts['missing']}; projection_only={counts['projection_only']}; "
        f"diagnostic_only={counts['diagnostic_only']}; pilot_only={counts['pilot_only']}"
    )


def _normalize_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {**audit, "findings": _as_findings(audit.get("findings"))}


def _audit_not_supplied() -> dict[str, Any]:
    return {
        "audit_available": False,
        "not_run_reason": "audit_not_supplied",
        "flagged": False,
        "findings": [],
    }


def _pending_audit() -> dict[str, Any]:
    return {
        "audit_available": False,
        "not_run_reason": "pending_final_write",
        "flagged": False,
        "findings": [],
    }


def _audit_tool_path(root: Path) -> Path | None:
    for rel_path in (
        Path("scripts/adversarial_artifact_audit.py"),
        Path("scripts/adversarial_verify.py"),
    ):
        candidate = root / rel_path
        if candidate.is_file():
            return candidate
    return None


def _auditable_end(start: float, end: float) -> float:
    return max(end, start + AUDITABLE_MIN_DURATION_S)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "AUDITABLE_MIN_DURATION_S",
    "DOT276_SOURCE_BY_EXP",
    "EXP2924_SOURCE",
    "EXP2925_SOURCE",
    "EXP2926_SOURCE",
    "EXP2927_SOURCE",
    "EXP2928_SOURCE",
    "EXP2929_SOURCE",
    "EXP2930_SOURCE",
    "EXP2931_SOURCE",
    "EXP2932_FIXTURE_SOURCE",
    "EXP2932_SOURCE",
    "EXP2933_SOURCE",
    "EXP2934_SOURCE",
    "INFERENCE_SUBSTRATE",
    "MATRIX_V9_SOURCE",
    "OUTPUT_REL_PATH",
    "build_artifact",
    "read_json_mapping",
    "run_adversarial_audit",
    "write_artifact",
]
