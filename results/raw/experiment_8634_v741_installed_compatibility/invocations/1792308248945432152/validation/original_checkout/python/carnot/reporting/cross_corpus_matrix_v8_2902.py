"""Build the Exp 2902 cross-corpus matrix v8 artifact.

Spec refs: REQ-REPORT-2902, SCENARIO-REPORT-2902.

This is a forward-only aggregation layer. It reads matrix v7 plus the clean
support artifacts requested for v8, records row status boundaries, and keeps
every row tied to the exact upstream artifact bytes used to construct it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.cross_corpus_matrix.v8"
ARTIFACT = "experiment_2902_cross_corpus_matrix_v8"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2902_cross_corpus_matrix_v8.json")

V7_REL_PATH = Path("results/experiment_2894_cross_corpus_matrix_v7.json")
EXP2890_REL_PATH = Path("results/experiment_2890_code_structural_dependency_verifier_v1.json")
EXP2891_REL_PATH = Path(
    "results/experiment_2891_cctu_executable_constraint_validator_pilot_v1.json"
)
EXP2892_REL_PATH = Path("results/experiment_2892_vericot_exact_frontier_expansion_v1.json")
EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)

UPSTREAM_ARTIFACTS: dict[str, Path] = {
    "exp2894": V7_REL_PATH,
    "exp2890": EXP2890_REL_PATH,
    "exp2891": EXP2891_REL_PATH,
    "exp2892": EXP2892_REL_PATH,
    "exp2898": EXP2898_REL_PATH,
}

FIELDS_IMPORTED: dict[str, list[str]] = {
    "exp2894": [
        "honest_verdict",
        "cross_corpus_matrix_built",
        "matrix_rows",
        "headline_eligible_rows",
        "pilot_only_rows",
        "taxonomy_only_rows",
        "blocked_rows",
        "missing_rows",
        "source_status_by_artifact",
    ],
    "exp2890": [
        "honest_verdict",
        "structural_dependency_verifier_ready",
        "headline_metric_claim_made",
        "n_contracts_built",
        "n_rows_verified",
        "generated_outputs_consumed",
        "violation_types",
    ],
    "exp2891": [
        "honest_verdict",
        "cctu_validator_ready",
        "headline_metric_claim_made",
        "executable_validation_used",
        "live_llm_called",
        "n_cases",
        "constraint_categories",
        "category_coverage",
        "unsupported_categories",
    ],
    "exp2892": [
        "honest_verdict",
        "vericot_frontier_ready",
        "n_candidate_rows",
        "n_vericot_supported_rows",
        "n_unsupported_rows",
        "unsupported_reasons",
        "solver_backend",
        "autoformalization_llm_called",
    ],
    "exp2898": [
        "honest_verdict",
        "inference_substrate",
        "kv260_overlay_loaded",
        "kv260_uio_devices_present",
        "bitstream_sha256",
        "board_transcript_path",
        "per_seed_results",
        "sample_count_sweep_results",
    ],
}

SUPPORT_ROW_IDS = {
    "exp2890": "exp2890_code_structural_dependency",
    "exp2891": "exp2891_cctu",
    "exp2892": "exp2892_vericot",
    "exp2898": "exp2898_kv260_hardware",
}

PROVENANCE_PRINCIPLE = (
    "Forward-only provenance discipline per the new Inference-Substrate Declaration rule. "
    "Lets a third party verify the aggregation is not synthesizing numbers from nothing."
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty object when unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2902: aggregate matrix v8 with per-row upstream SHA256."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    payloads = {
        exp_id: read_json(root_path / rel_path) for exp_id, rel_path in UPSTREAM_ARTIFACTS.items()
    }
    shas = {
        exp_id: _sha256_file(root_path / rel_path)
        for exp_id, rel_path in UPSTREAM_ARTIFACTS.items()
        if (root_path / rel_path).is_file()
    }
    end = time.perf_counter() if now_s is None else now_s

    if not (root_path / V7_REL_PATH).is_file():
        return _base_artifact(
            honest_verdict="blocked_v7_missing",
            rows=[],
            rows_clean=[],
            rows_flagged=[],
            rows_blocked=["exp2894"],
            rows_pilot_only=[],
            shas=shas,
            duration_s=end - started,
        )

    v7_payload = payloads["exp2894"]
    if _source_status("exp2894", v7_payload) != "clean":
        row = _blocked_source_row("exp2894", "blocked", v7_payload, shas)
        return _base_artifact(
            honest_verdict="blocked_v7_unclean",
            rows=[row],
            rows_clean=[],
            rows_flagged=[],
            rows_blocked=["exp2894"],
            rows_pilot_only=[],
            shas=shas,
            duration_s=end - started,
        )

    rows: list[dict[str, Any]] = []
    rows_clean: list[str] = []
    rows_flagged: list[str] = []
    rows_blocked: list[str] = []
    rows_pilot_only: list[str] = []

    for row in _v7_rows(v7_payload):
        built = _v7_matrix_row(row, v7_payload, shas["exp2894"])
        rows.append(built)
        row_id = built["row_id"]
        if built["row_status"] == "clean":
            rows_clean.append(row_id)
        if built["row_status"].startswith("pilot_only"):
            rows_pilot_only.append(row_id)
        if built["flag_reasons"]:
            rows_flagged.append(row_id)

    for exp_id in ("exp2890", "exp2891", "exp2892", "exp2898"):
        status = _source_status(exp_id, payloads[exp_id])
        row = _support_row(exp_id, status, payloads[exp_id], shas)
        rows.append(row)
        row_id = row["row_id"]
        if status == "clean":
            rows_clean.append(row_id)
        elif status == "pilot_only":
            rows_pilot_only.append(row_id)
        elif status == "flagged":
            rows_flagged.append(row_id)
        else:
            rows_blocked.append(row_id)

    return _base_artifact(
        honest_verdict=_honest_verdict(rows_clean, rows_flagged, rows_blocked, rows_pilot_only),
        rows=rows,
        rows_clean=rows_clean,
        rows_flagged=rows_flagged,
        rows_blocked=rows_blocked,
        rows_pilot_only=rows_pilot_only,
        shas=shas,
        duration_s=end - started,
    )


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2902 matrix v8 artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _base_artifact(
    *,
    honest_verdict: str,
    rows: list[dict[str, Any]],
    rows_clean: list[str],
    rows_flagged: list[str],
    rows_blocked: list[str],
    rows_pilot_only: list[str],
    shas: dict[str, str],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "rows_clean": rows_clean,
        "rows_flagged": rows_flagged,
        "rows_blocked": rows_blocked,
        "rows_pilot_only": rows_pilot_only,
        "matrix_rows": rows,
        "cited_upstream_artifacts": _cited_upstream_artifacts(shas),
        "synthetic_rows_created": False,
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, duration_s), 6),
    }


def _cited_upstream_artifacts(shas: dict[str, str]) -> dict[str, Any]:
    return {
        "principle": PROVENANCE_PRINCIPLE,
        "shape": "list of {experiment_id, fields_imported, sha256}",
        "artifacts": [
            {
                "experiment_id": exp_id,
                "artifact_path": str(UPSTREAM_ARTIFACTS[exp_id]),
                "fields_imported": list(FIELDS_IMPORTED[exp_id]),
                "sha256": shas[exp_id],
            }
            for exp_id in UPSTREAM_ARTIFACTS
            if exp_id in shas
        ],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_status(exp_id: str, payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if not _terminal_success(payload.get("honest_verdict")):
        return "unclean"
    if exp_id == "exp2894":
        return "clean" if payload.get("cross_corpus_matrix_built") is True else "unclean"
    if exp_id == "exp2890":
        return (
            "clean"
            if payload.get("structural_dependency_verifier_ready") is True
            and payload.get("headline_metric_claim_made") is False
            else "unclean"
        )
    if exp_id == "exp2891":
        pilot_ready = (
            payload.get("cctu_validator_ready") is True
            and payload.get("headline_metric_claim_made") is False
            and payload.get("executable_validation_used") is True
            and payload.get("live_llm_called") is False
        )
        return "pilot_only" if pilot_ready else "unclean"
    if exp_id == "exp2892":
        return (
            "clean"
            if payload.get("vericot_frontier_ready") is True
            and payload.get("autoformalization_llm_called") is False
            else "unclean"
        )
    if exp_id == "exp2898":
        clean_hardware_smoke = (
            payload.get("inference_substrate") == "hardware_smoke"
            and bool(payload.get("per_seed_results"))
            and _preconditions_available(payload)
            and not _has_speedup_key(payload)
        )
        return "clean" if clean_hardware_smoke else "unclean"
    return "unclean"


def _terminal_success(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("complete:", "success:"))


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("adversarial_verify_passed") is False:
        return True
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return True
    summary = payload.get("adversarial_verify_summary")
    return isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0


def _preconditions_available(payload: dict[str, Any]) -> bool:
    checked = payload.get("preconditions_checked")
    return isinstance(checked, list) and all(
        isinstance(item, dict) and item.get("available") is True for item in checked
    )


def _has_speedup_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "speedup" in str(key).lower() or _has_speedup_key(child) for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_speedup_key(child) for child in value)
    return False


def _v7_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("matrix_rows")
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _v7_matrix_row(
    source_row: dict[str, Any],
    v7_payload: dict[str, Any],
    v7_sha: str,
) -> dict[str, Any]:
    corpus = str(source_row.get("corpus", "unknown"))
    row_id = f"corpus:{_safe_row_token(corpus)}"
    flag_reasons = _v7_flag_reasons(corpus, source_row, v7_payload)
    pilot_only = (
        source_row.get("pilot_only") is True or source_row.get("row_status") == "pilot_only"
    )
    if pilot_only and flag_reasons:
        row_status = "pilot_only_flagged_support"
    elif pilot_only:
        row_status = "pilot_only"
    elif flag_reasons:
        row_status = "flagged"
    else:
        row_status = "clean"

    return {
        "row_id": row_id,
        "row_label": corpus,
        "row_kind": "v7_corpus_row",
        "row_status": row_status,
        "flag_reasons": flag_reasons,
        "summary": _selected_fields(source_row, _v7_row_fields()),
        "provenance": _provenance("exp2894", v7_sha),
    }


def _v7_row_fields() -> list[str]:
    return [
        "corpus",
        "row_status",
        "headline_eligible",
        "pilot_only",
        "taxonomy_only",
        "source_artifact",
        "source_honest_verdict",
        "label_evidence",
        "primary_metric",
        "generated_code_status",
        "structural_dependency_verification",
        "cctu_constraint_category_coverage",
        "vericot_exact_support",
        "kan_complexity",
        "residual_gap",
    ]


def _v7_flag_reasons(
    corpus: str,
    source_row: dict[str, Any],
    v7_payload: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    generated = source_row.get("generated_code_status")
    if (
        isinstance(generated, dict)
        and generated.get("status") == "blocked_unresolved_adversarial_flags"
    ):
        reasons.extend(str(reason) for reason in generated.get("flag_reasons", []))
    blocked_rows = v7_payload.get("blocked_rows")
    if isinstance(blocked_rows, dict):
        blocked = blocked_rows.get(corpus)
        if isinstance(blocked, dict):
            reasons.extend(str(reason) for reason in blocked.get("reasons", []))
    return list(dict.fromkeys(reasons))


def _support_row(
    exp_id: str,
    status: str,
    payload: dict[str, Any],
    shas: dict[str, str],
) -> dict[str, Any]:
    if status in {"clean", "pilot_only"}:
        row_status = status
        blocked_reason = None
    elif status == "flagged":
        row_status = "flagged"
        blocked_reason = None
    else:
        row_status = "blocked"
        blocked_reason = _blocked_reason(status, payload)

    row = {
        "row_id": SUPPORT_ROW_IDS[exp_id],
        "row_label": _support_label(exp_id),
        "row_kind": "support_artifact_row",
        "row_status": row_status,
        "summary": _selected_fields(payload, FIELDS_IMPORTED[exp_id]),
        "provenance": _provenance(exp_id, shas.get(exp_id, "")),
    }
    if blocked_reason is not None:
        row["blocked_reason"] = blocked_reason
    if row_status == "flagged":
        row["flag_reasons"] = _source_flag_reasons(payload)
    return row


def _blocked_source_row(
    exp_id: str,
    row_status: str,
    payload: dict[str, Any],
    shas: dict[str, str],
) -> dict[str, Any]:
    return {
        "row_id": exp_id,
        "row_label": "matrix v7",
        "row_kind": "required_source",
        "row_status": row_status,
        "blocked_reason": _blocked_reason(_source_status(exp_id, payload), payload),
        "summary": _selected_fields(payload, FIELDS_IMPORTED[exp_id]),
        "provenance": _provenance(exp_id, shas.get(exp_id, "")),
    }


def _selected_fields(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: payload[field] for field in fields if field in payload}


def _provenance(exp_id: str, sha256: str) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "artifact_path": str(UPSTREAM_ARTIFACTS[exp_id]),
        "fields_imported": list(FIELDS_IMPORTED[exp_id]),
        "sha256": sha256,
    }


def _safe_row_token(label: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in label)
    return "_".join(part for part in token.split("_") if part)


def _support_label(exp_id: str) -> str:
    labels = {
        "exp2890": "Code Structural Dependency",
        "exp2891": "CCTU executable constraint pilot",
        "exp2892": "VeriCoT exact frontier",
        "exp2898": "KV260 Ising hardware latency",
    }
    return labels[exp_id]


def _blocked_reason(status: str, payload: dict[str, Any]) -> str:
    verdict = payload.get("honest_verdict")
    if status == "missing":
        return "source_missing"
    if status == "blocked" and isinstance(verdict, str):
        return verdict
    return "source_not_clean"


def _source_flag_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial=true")
    if payload.get("adversarial_verify_passed") is False:
        reasons.append("adversarial_verify_passed=false")
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            reasons.append(f"{key}_present")
    summary = payload.get("adversarial_verify_summary")
    if isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0:
        reasons.append("adversarial_verify_summary_flag_count")
    return reasons


def _honest_verdict(
    rows_clean: list[str],
    rows_flagged: list[str],
    rows_blocked: list[str],
    rows_pilot_only: list[str],
) -> str:
    return (
        "complete: cross-corpus matrix v8 aggregated with forward-only provenance; "
        f"clean={len(rows_clean)}; flagged={len(rows_flagged)}; "
        f"blocked={len(rows_blocked)}; pilot_only={len(rows_pilot_only)}"
    )
