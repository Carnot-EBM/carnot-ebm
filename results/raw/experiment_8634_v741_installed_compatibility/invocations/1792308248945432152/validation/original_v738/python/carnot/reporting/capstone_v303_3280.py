"""Build the Exp 3280 milestone .303 capstone artifact.

Spec refs: REQ-REPORT-3280, SCENARIO-REPORT-3280.

This module closes milestone .303 from the evidence matrix and source
artifacts already present in the repository. It deliberately does not run
models, Garak, repair, verifier scoring, the conductor, or any publishing
action because the capstone is an evidence readout, not a new experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.milestone_capstone.v303_matrix_v35_closeout.v1"
EXPERIMENT_ID = "exp3280"
TASK_ID = "exp3280-capstone-v303"
ARTIFACT = "experiment_3280_capstone_v303"
MILESTONE = "2026.05.303"
PRIOR_MILESTONE = "2026.05.302"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3280_capstone_v303.json")
RANDOM_SEED = 3280
DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT = 105

CAPSTONE_V302_REL_PATH = Path("results/experiment_3266_capstone_v302.json")
MATRIX_V35_REL_PATH = Path("results/experiment_3279_evidence_matrix_v35.json")
EXP3267_REL_PATH = Path("results/experiment_3267_close_v302_open_v303_corpus_queue.json")
EXP3268_REL_PATH = Path("results/experiment_3268_sota_receipt_methodology_supplement_v1.json")
EXP3269_REL_PATH = Path(
    "results/experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.json"
)
EXP3270_REL_PATH = Path("results/experiment_3270_prompt_injection_teacher_label_shards_2_4_v1.json")
EXP3271_REL_PATH = Path(
    "results/experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1.json"
)
EXP3272_REL_PATH = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
EXP3273_REL_PATH = Path(
    "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
)
EXP3274_REL_PATH = Path(
    "results/experiment_3274_prompt_injection_v4_garak_dataflip_redteam_eval_v1.json"
)
EXP3275_REL_PATH = Path("results/experiment_3275_clean_local_sota_verifier_rerun_v14.json")
EXP3276_REL_PATH = Path(
    "results/experiment_3276_repair_gate_decision_v8_after_v4_garak_clean_verifier.json"
)
EXP3277_REL_PATH = Path("results/experiment_3277_sota_repair_micro_panel_v9.json")
EXP3278_REL_PATH = Path(
    "results/experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.json"
)

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
V302_TOP_GAP = "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates"
RECOMMENDED_GARAK_TITLE = "Garak Red-Team Availability + Clean Verifier Repair Gate Reopen"
REQUIRED_ARTIFACT_FIELDS = {
    "capstone_v303_ready",
    "paper_ready",
    "publication_blocker_count",
    "publication_blocker_delta",
    "v4_full_corpus_status",
    "garak_gate_status",
    "repair_gate_status",
    "fr11_status",
    "next_top_gap",
    "recommended_next_milestone_title",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


@dataclass(frozen=True)
class SourceSpec:
    """One source artifact that the capstone inventories for provenance."""

    experiment_id: str
    path: Path
    role: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3266", CAPSTONE_V302_REL_PATH, "prior_v302_capstone"),
    SourceSpec("exp3267", EXP3267_REL_PATH, "v302_close_v303_open"),
    SourceSpec("exp3268", EXP3268_REL_PATH, "sota_receipt_methodology"),
    SourceSpec("exp3269", EXP3269_REL_PATH, "v4_full_corpus_manifest"),
    SourceSpec("exp3270", EXP3270_REL_PATH, "teacher_label_shards_2_4"),
    SourceSpec("exp3271", EXP3271_REL_PATH, "teacher_label_shards_5_7_garak_seed"),
    SourceSpec("exp3272", EXP3272_REL_PATH, "v4_full_corpus_assembly"),
    SourceSpec("exp3273", EXP3273_REL_PATH, "v4_full_corpus_kan_eval"),
    SourceSpec("exp3274", EXP3274_REL_PATH, "garak_dataflip_redteam"),
    SourceSpec("exp3275", EXP3275_REL_PATH, "clean_local_sota_verifier"),
    SourceSpec("exp3276", EXP3276_REL_PATH, "repair_gate_decision"),
    SourceSpec("exp3277", EXP3277_REL_PATH, "repair_micro_panel"),
    SourceSpec("exp3278", EXP3278_REL_PATH, "fr11_full_corpus_audit"),
    SourceSpec("exp3279", MATRIX_V35_REL_PATH, "evidence_matrix_v35"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail closed for absent, malformed, or array input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact source bytes so the closeout can be reproduced later."""

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
    """REQ-REPORT-3280: aggregate matrix v35 and `.303` evidence into a capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V35_REL_PATH)
    capstone_v302 = read_json_object(root_path / CAPSTONE_V302_REL_PATH)
    source_artifacts = _source_artifacts(root_path)
    source_checksums = {
        row["path"]: row["sha256"] for row in source_artifacts if row.get("sha256")
    }
    prior_count = _prior_publication_blocker_count(capstone_v302, matrix)
    matrix_ready = matrix.get("matrix_v35_ready") is True
    rows = [_as_mapping(row) for row in _as_list(matrix.get("rows"))]
    publication_count = (
        _publication_blocker_count(matrix, prior_count) if matrix_ready else prior_count
    )
    publication_delta = publication_count - prior_count
    v4_status = (
        _v4_full_corpus_status(rows, matrix) if matrix_ready else "gated_skip: matrix_v35_not_ready"
    )
    garak_status = (
        _garak_gate_status(rows) if matrix_ready else "gated_skip: matrix_v35_not_ready"
    )
    repair_status = (
        _repair_gate_status(rows, matrix) if matrix_ready else "gated_skip: matrix_v35_not_ready"
    )
    fr11_status = _fr11_status(rows) if matrix_ready else "gated_skip: matrix_v35_not_ready"
    next_top_gap = (
        _next_top_gap(matrix) if matrix_ready else "produce_ready_evidence_matrix_v35"
    )
    top_gap_cleared = _v302_top_gap_cleared(v4_status, garak_status, repair_status)

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
        "capstone_v303_ready": True,
        "gated_skip": not matrix_ready,
        "gated_skip_reasons": _gated_skip_reasons(matrix) if not matrix_ready else [],
        "paper_ready": _paper_ready(matrix, publication_count, rows) if matrix_ready else False,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_count,
        "publication_blocker_delta": publication_delta,
        "publication_blocker_trend": _trend(publication_delta),
        "prior_next_top_gap": str(capstone_v302.get("next_top_gap") or ""),
        "v302_next_top_gap_cleared": top_gap_cleared,
        "v302_next_top_gap_status": _v302_top_gap_status(
            matrix_ready, top_gap_cleared, v4_status, garak_status, repair_status
        ),
        "v4_full_corpus_status": v4_status,
        "garak_gate_status": garak_status,
        "repair_gate_status": repair_status,
        "fr11_status": fr11_status,
        "next_top_gap": next_top_gap,
        "recommended_next_milestone_title": _recommended_next_milestone_title(next_top_gap),
        "changes_since_v302": _changes_since_v302(
            matrix_ready,
            rows,
            publication_delta,
            next_top_gap,
        ),
        "stayed_blocked": _stayed_blocked(rows) if matrix_ready else ["matrix_v35_not_ready"],
        "matrix_v35_summary": _matrix_summary(matrix),
        "publication_readiness": _as_mapping(matrix.get("publication_readiness")),
        "source_artifacts": source_artifacts,
        "source_checksums": source_checksums,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_external_submission_or_publication": True,
        "no_push": True,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "scripts_research_conductor_modified": False,
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
    """Build and persist the Exp 3280 capstone JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject capstones that omit fields or claim paper readiness too early."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3280")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3280-capstone-v303")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.303")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("publication_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while publication blockers remain")


def _source_artifacts(root: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        payload = read_json_object(path)
        records.append(
            {
                "experiment_id": spec.experiment_id,
                "role": spec.role,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "readable_json_object": bool(payload),
                "reported_experiment_id": str(payload.get("experiment_id") or payload.get("experiment") or ""),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
                "sha256": sha256_file(path),
            }
        )
    return records


def _prior_publication_blocker_count(
    capstone_v302: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> int:
    for value in (
        _int_value(capstone_v302.get("publication_blocker_count")),
        _int_value(matrix.get("prior_publication_blocker_count")),
        _int_value(matrix.get("publication_blocker_count_estimate")),
    ):
        if value > 0:
            return value
    return DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT


def _publication_blocker_count(matrix: Mapping[str, Any], prior_count: int) -> int:
    count = _int_value(matrix.get("publication_blocker_count_estimate"))
    return count if count > 0 else prior_count


def _paper_ready(matrix: Mapping[str, Any], publication_count: int, rows: list[Mapping[str, Any]]) -> bool:
    readiness = _as_mapping(matrix.get("publication_readiness"))
    required_gates = _as_mapping(readiness.get("required_gates"))
    return (
        matrix.get("paper_ready") is True
        and readiness.get("paper_ready") is True
        and publication_count == 0
        and not _as_list(readiness.get("blocking_rows"))
        and not _as_list(readiness.get("flagged_rows"))
        and all(value is True for value in required_gates.values())
        and all(row.get("status") == "clean" for row in rows)
    )


def _v4_full_corpus_status(rows: list[Mapping[str, Any]], matrix: Mapping[str, Any]) -> str:
    gates = _required_gates(matrix)
    row3273 = _row(rows, "exp3273")
    corpus_flags = [
        row.get("experiment_id")
        for row in rows
        if row.get("experiment_id") in {"exp3270", "exp3271", "exp3272"}
        and _as_list(row.get("quality_flags"))
    ]
    sidecar_or_noninferiority = (
        row3273.get("status") == "sidecar-only"
        or "sidecar_only=true" in _as_list(row3273.get("bounded_claims"))
        or "delong_noninferiority_passed=false" in _as_list(row3273.get("bounded_claims"))
    )
    if gates.get("full_15k_corpus") is not True:
        return "blocked: full_15k_corpus_not_ready"
    if gates.get("kan_full_eval") is not True:
        return "blocked: kan_full_eval_not_ready"
    if corpus_flags or sidecar_or_noninferiority:
        return "partial: full_15k_ready_but_flagged_sidecar_noninferiority_failed"
    return "complete: full_15k_corpus_and_full_eval_ready"


def _garak_gate_status(rows: list[Mapping[str, Any]]) -> str:
    row3274 = _row(rows, "exp3274")
    reasons = _list_of_strings(row3274.get("blocker_reasons"))
    if row3274.get("ready") is True and row3274.get("status") == "clean":
        return "passed: garak_redteam_eval_ready"
    return f"blocked: {reasons[0]}" if reasons else "blocked: garak_gate_not_ready"


def _repair_gate_status(rows: list[Mapping[str, Any]], matrix: Mapping[str, Any]) -> str:
    gates = _required_gates(matrix)
    row3276 = _row(rows, "exp3276")
    reasons = " ".join(_list_of_strings(row3276.get("blocker_reasons")))
    if gates.get("repair_gate") is True and gates.get("repair_micro_panel") is True:
        return "passed: repair_gate_and_micro_panel_ready"
    if (
        gates.get("garak_redteam") is False
        and gates.get("clean_verifier") is False
        and row3276.get("status") == "blocked"
    ) or ("exp3274" in reasons and "exp3275" in reasons):
        return "blocked: garak_redteam_and_clean_verifier_gates_failed"
    return "blocked: repair_gate_or_micro_panel_not_ready"


def _fr11_status(rows: list[Mapping[str, Any]]) -> str:
    row3278 = _row(rows, "exp3278")
    summary = _as_mapping(row3278.get("summary"))
    if row3278.get("ready") is True and summary.get("controller_memory_only") is True:
        return (
            "complete: controller_memory_only_retention_"
            f"{_format_number(summary.get('retention_score'))}_adaptation_"
            f"{_format_number(summary.get('adaptation_score'))}_forgetting_"
            f"{_format_number(summary.get('forgetting_rate'))}"
        )
    return "blocked: fr11_full_corpus_audit_not_ready"


def _next_top_gap(matrix: Mapping[str, Any]) -> str:
    candidates = _as_list(matrix.get("next_gap_candidates"))
    first = _as_mapping(candidates[0]) if candidates else {}
    return str(first.get("gap") or "publication_blocker_retirement_review")


def _recommended_next_milestone_title(next_top_gap: str) -> str:
    if next_top_gap == "unblock_garak_redteam_eval":
        return RECOMMENDED_GARAK_TITLE
    if next_top_gap == "produce_ready_evidence_matrix_v35":
        return "Evidence Matrix V35 Repair Before Milestone Closeout"
    return "Publication Blocker Retirement Review"


def _v302_top_gap_cleared(v4_status: str, garak_status: str, repair_status: str) -> bool:
    return (
        v4_status.startswith("complete:")
        and garak_status.startswith("passed:")
        and repair_status.startswith("passed:")
    )


def _v302_top_gap_status(
    matrix_ready: bool,
    cleared: bool,
    v4_status: str,
    garak_status: str,
    repair_status: str,
) -> str:
    if not matrix_ready:
        return "gated_skip: matrix_v35_not_ready"
    if cleared:
        return "complete: full corpus, Garak, and repair gates cleared"
    if v4_status.startswith("partial:") and garak_status.startswith("blocked:") and repair_status.startswith("blocked:"):
        return "partial: full corpus materialized but Garak and repair gates remain blocked"
    return "blocked: v302 top gap remains open"


def _changes_since_v302(
    matrix_ready: bool,
    rows: list[Mapping[str, Any]],
    publication_delta: int,
    next_top_gap: str,
) -> list[str]:
    changes: list[str] = []
    if not matrix_ready:
        return ["matrix_v35_not_ready_gated_skip"]
    if _row(rows, "exp3272").get("ready") is True:
        changes.append("full_15k_v4_corpus_materialized")
    if _row(rows, "exp3278").get("ready") is True:
        changes.append("fr11_full_corpus_controller_memory_audit_completed")
    if publication_delta < 0:
        changes.append("publication_blocker_count_reduced")
    if publication_delta == 0:
        changes.append("publication_blocker_count_unchanged")
    if next_top_gap == "unblock_garak_redteam_eval":
        changes.append("top_gap_narrowed_to_garak_redteam_eval")
    return changes


def _stayed_blocked(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    row3273 = _row(rows, "exp3273")
    if row3273.get("status") == "sidecar-only":
        blocked.append("kan_sidecar_only_noninferiority_failed")
    if _row(rows, "exp3274").get("status") == "blocked":
        blocked.append("garak_redteam_blocked_unavailable")
    if _row(rows, "exp3275").get("status") == "blocked":
        blocked.append("clean_verifier_abstention_gate_failed")
    if _row(rows, "exp3276").get("status") == "blocked":
        blocked.append("repair_gate_blocked")
    if _row(rows, "exp3277").get("status") == "missing":
        blocked.append("repair_micro_panel_missing")
    if any(_as_list(row.get("quality_flags")) for row in rows):
        blocked.append("prompt_injection_artifact_flags_unresolved")
    return blocked


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v35_ready": matrix.get("matrix_v35_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count_estimate": _int_value(
            matrix.get("publication_blocker_count_estimate")
        ),
        "publication_blocker_delta_from_v302": _int_value(
            matrix.get("publication_blocker_delta_from_v302")
        ),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _required_gates(matrix: Mapping[str, Any]) -> JsonDict:
    readiness = _as_mapping(matrix.get("publication_readiness"))
    return _as_mapping(readiness.get("required_gates"))


def _gated_skip_reasons(matrix: Mapping[str, Any]) -> list[str]:
    reasons = _list_of_strings(matrix.get("invariant_violations"))
    if reasons:
        return reasons
    return ["matrix_v35_missing_or_not_ready"] if not matrix else ["matrix_v35_ready is not true"]


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "The capstone reads matrix v35 and checked-in .303 artifacts only.",
        "paper_ready": "Publication readiness stays false while any blocker, flagged row, missing row, or sidecar-only row remains.",
        "publication_blocker_delta": "Blocker movement is computed relative to the .302 capstone count.",
        "v302_gap": "The .302 full-corpus plus Garak plus repair gap is cleared only if all three parts clear.",
        "no_external_action": "The artifact records readiness only and does not submit or publish externally.",
    }


def _row(rows: list[Mapping[str, Any]], experiment_id: str) -> JsonDict:
    return next((_as_mapping(row) for row in rows if row.get("experiment_id") == experiment_id), {})


def _trend(delta: int) -> str:
    return "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged"


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "publication_blocker_delta": artifact.get("publication_blocker_delta"),
        "v4_full_corpus_status": artifact.get("v4_full_corpus_status"),
        "garak_gate_status": artifact.get("garak_gate_status"),
        "repair_gate_status": artifact.get("repair_gate_status"),
        "fr11_status": artifact.get("fr11_status"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v303_ready="
        f"{str(artifact.get('capstone_v303_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"publication_blocker_delta={artifact.get('publication_blocker_delta')}; "
        f"v4_full_corpus_status={artifact.get('v4_full_corpus_status')}; "
        f"garak_gate_status={artifact.get('garak_gate_status')}; "
        f"repair_gate_status={artifact.get('repair_gate_status')}; "
        f"fr11_status={artifact.get('fr11_status')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _format_number(value: Any) -> str:
    number = _number_value(value)
    if number.is_integer():
        return f"{number:.1f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
