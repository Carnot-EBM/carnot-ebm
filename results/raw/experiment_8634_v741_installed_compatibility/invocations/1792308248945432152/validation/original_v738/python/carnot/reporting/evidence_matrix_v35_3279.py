"""Build the Exp 3279 evidence matrix v35 artifact.

Spec refs: REQ-REPORT-3279, SCENARIO-REPORT-3279.

This is a ledger, not an experiment runner. It reads the checked-in `.303`
artifacts, records which rows are clean, blocked, flagged, missing, pilot-only,
or sidecar-only, and keeps publication readiness conservative. That matters
because a missing repair artifact or a blocked Garak gate is evidence of a
remaining blocker, not permission to infer downstream success.
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
SCHEMA_VERSION = "carnot.evidence_matrix.v35_dot303_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3279"
TASK_ID = "exp3279-evidence-matrix-v35"
ARTIFACT = "experiment_3279_evidence_matrix_v35"
MILESTONE = "2026.05.303"
PRIOR_MILESTONE = "2026.05.302"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
OUTPUT_REL_PATH = Path("results/experiment_3279_evidence_matrix_v35.json")
RANDOM_SEED = 3279
DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT = 105

CAPSTONE_V302_REL_PATH = Path("results/experiment_3266_capstone_v302.json")
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

STATUSES = ("clean", "blocked", "flagged", "missing", "pilot-only", "sidecar-only")
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "matrix_v35_ready",
    "clean_row_count",
    "blocked_row_count",
    "flagged_row_count",
    "missing_row_count",
    "sidecar_only_row_count",
    "publication_blocker_count_estimate",
    "next_gap_candidates",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
SUMMARY_KEYS = (
    "v302_closed_v303_opened",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "sota_receipt_methodology_supplement_v1_ready",
    "clean_sota_receipt_eligible",
    "full_corpus_manifest_ready",
    "target_total_examples",
    "teacher_label_shards_2_4_ready",
    "teacher_label_shards_5_7_garak_seed_ready",
    "cumulative_label_count",
    "garak_seed_count",
    "full_15k_corpus_ready",
    "leakage_audit_passed",
    "assembled_example_count",
    "v4_full_eval_ready",
    "sidecar_only",
    "full_corpus_auroc",
    "full_corpus_auprc",
    "delong_noninferiority_passed",
    "garak_redteam_eval_ready",
    "garak_available",
    "garak_gate_passed",
    "dataflip_gate_passed",
    "clean_verifier_rerun_ready",
    "clean_rerun_allowed",
    "repair_gate_input_clean_enough",
    "status",
    "blocked_at_layer",
    "fr11_full_corpus_audit_ready",
    "controller_memory_only",
    "foundation_weight_updates_performed",
    "retention_score",
    "adaptation_score",
    "forgetting_rate",
    "negative_transfer_rate",
    "heldout_trace_count",
)


@dataclass(frozen=True)
class SourceSpec:
    """One planned `.303` source row that matrix v35 must account for."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str


EXPECTED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3267",
        "exp3267-close-v302-open-v303-corpus-queue",
        EXP3267_REL_PATH,
        "v302_close_v303_open",
        "v302_closed_v303_opened",
    ),
    SourceSpec(
        "exp3268",
        "exp3268-sota-receipt-methodology-supplement-v1",
        EXP3268_REL_PATH,
        "sota_receipt_methodology_supplement",
        "clean_sota_receipt_eligible",
    ),
    SourceSpec(
        "exp3269",
        "exp3269-prompt-injection-v4-full-corpus-split-manifest-v1",
        EXP3269_REL_PATH,
        "prompt_injection_full_corpus_manifest",
        "full_corpus_manifest_ready",
    ),
    SourceSpec(
        "exp3270",
        "exp3270-prompt-injection-teacher-label-shards-2-4-v1",
        EXP3270_REL_PATH,
        "prompt_injection_teacher_label_shards_2_4",
        "teacher_label_shards_2_4_ready",
    ),
    SourceSpec(
        "exp3271",
        "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1",
        EXP3271_REL_PATH,
        "prompt_injection_teacher_label_shards_5_7_garak_seed",
        "teacher_label_shards_5_7_garak_seed_ready",
    ),
    SourceSpec(
        "exp3272",
        "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
        EXP3272_REL_PATH,
        "prompt_injection_full_corpus_assembly_leakage_audit",
        "full_15k_corpus_ready",
    ),
    SourceSpec(
        "exp3273",
        "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
        EXP3273_REL_PATH,
        "prompt_injection_kan_full_corpus_delong_eval",
        "v4_full_eval_ready",
    ),
    SourceSpec(
        "exp3274",
        "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1",
        EXP3274_REL_PATH,
        "prompt_injection_garak_dataflip_redteam_eval",
        "garak_redteam_eval_ready",
    ),
    SourceSpec(
        "exp3275",
        "exp3275-clean-local-sota-verifier-rerun-v14",
        EXP3275_REL_PATH,
        "clean_local_sota_verifier_rerun",
        "clean_verifier_rerun_ready",
    ),
    SourceSpec(
        "exp3276",
        "exp3276-repair-gate-decision-v8-after-v4-garak-clean-verifier",
        EXP3276_REL_PATH,
        "repair_gate_decision_v8",
        "repair_gate_decision_v8_ready",
    ),
    SourceSpec(
        "exp3277",
        "exp3277-sota-repair-micro-panel-v9",
        EXP3277_REL_PATH,
        "sota_repair_micro_panel_v9",
        "repair_micro_panel_v9_ready",
    ),
    SourceSpec(
        "exp3278",
        "exp3278-fr11-full-corpus-continual-self-learning-audit-v1",
        EXP3278_REL_PATH,
        "fr11_full_corpus_continual_self_learning_audit",
        "fr11_full_corpus_audit_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absent, malformed, or array input as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact artifact bytes for reproducible matrix provenance."""

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
    """REQ-REPORT-3279: aggregate evidence matrix v35 from checked-in `.303` rows."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone_v302 = read_json_object(root_path / CAPSTONE_V302_REL_PATH)
    rows = [_source_row(root_path, spec) for spec in EXPECTED_SOURCES]
    payloads = {row["experiment_id"]: _as_mapping(row.get("payload")) for row in rows}
    public_rows = [_public_row(row) for row in rows]
    primary_counts = _status_counts(public_rows)
    flagged_count = sum(1 for row in public_rows if row["quality_flags"])
    prior_count = _prior_publication_blocker_count(payloads.get("exp3267", {}), capstone_v302)
    publication_count = prior_count
    publication_readiness = _publication_readiness(public_rows, publication_count)
    movement = _publication_movement(publication_count, prior_count)
    invariant_violations = _invariant_violations(payloads)

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
        "matrix_v35_ready": not invariant_violations,
        "rows": public_rows,
        "primary_status_counts": primary_counts,
        "clean_row_count": primary_counts["clean"],
        "blocked_row_count": primary_counts["blocked"],
        "flagged_row_count": flagged_count,
        "flagged_primary_row_count": primary_counts["flagged"],
        "missing_row_count": primary_counts["missing"],
        "pilot_only_row_count": primary_counts["pilot-only"],
        "sidecar_only_row_count": primary_counts["sidecar-only"],
        "artifacts_expected": [_expected_record(spec) for spec in EXPECTED_SOURCES],
        "artifacts_found": [row for row in public_rows if row["present"]],
        "artifacts_missing": [row for row in public_rows if not row["present"]],
        "source_checksums": {
            row["path"]: row["sha256"] for row in public_rows if row.get("sha256")
        },
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count_estimate": publication_count,
        "publication_blocker_delta_from_v302": publication_count - prior_count,
        "publication_blocker_movement": movement,
        "publication_readiness": publication_readiness,
        "paper_ready": publication_readiness["paper_ready"],
        "next_gap_candidates": _next_gap_candidates(public_rows),
        "loaded_artifact_paths": [row["path"] for row in public_rows if row["present"]],
        "protected_files_untouched": {"scripts/research_conductor.py": True},
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
        "invariant_violations": invariant_violations,
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
    """Build and persist the Exp 3279 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when v35 omits required fields or overclaims publication readiness."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3279")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3279-evidence-matrix-v35")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.303")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count_estimate")) < 0:
        raise ValueError("publication_blocker_count_estimate must be non-negative")
    if (
        artifact.get("paper_ready") is True
        and _int_value(artifact.get("publication_blocker_count_estimate")) != 0
    ):
        raise ValueError("paper_ready cannot be true while publication blockers remain")


def _source_row(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    present = path.is_file()
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
        "present": present,
        "payload": payload,
        "status": _status_for_source(spec, payload, present),
        "sha256": sha256_file(path),
    }


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(row.get("payload"))
    spec_ready_field = str(row.get("ready_field") or "")
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(payload.get("task_id") or row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "ready_field": spec_ready_field,
        "present": row.get("present") is True,
        "status": _normal_status(str(row.get("status") or "missing")),
        "ready": payload.get(spec_ready_field) is True,
        "reported_experiment_id": str(payload.get("experiment_id") or payload.get("experiment") or ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "blocker_reasons": _blocker_reasons(row, payload),
        "quality_flags": _quality_flags(payload),
        "bounded_claims": _bounded_claims(payload),
        "summary": _row_summary(payload),
        "sha256": row.get("sha256"),
    }


def _expected_record(spec: SourceSpec) -> JsonDict:
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
    }


def _status_for_source(spec: SourceSpec, payload: Mapping[str, Any], present: bool) -> str:
    if not present or not payload:
        return "missing"
    if _is_gate_blocked(payload) or _has_blockers(spec, payload):
        return "blocked"
    if payload.get("sidecar_only") is True:
        return "sidecar-only"
    if payload.get("pilot_only") is True:
        return "pilot-only"
    if _quality_flags(payload):
        return "flagged"
    return "clean"


def _is_gate_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or str(payload.get("honest_verdict") or "").startswith("blocked_gate_check")
    )


def _has_blockers(spec: SourceSpec, payload: Mapping[str, Any]) -> bool:
    return bool(_explicit_blockers(payload)) or payload.get(spec.ready_field) is False


def _explicit_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    blockers += _list_of_strings(payload.get("blocked_reasons"))
    blockers += _list_of_strings(payload.get("gate_reasons"))
    blocked_reason = str(payload.get("blocked_reason") or "").strip()
    gate_summary = str(payload.get("gate_check_summary") or "").strip()
    blockers += [blocked_reason] if blocked_reason else []
    blockers += [gate_summary] if gate_summary else []
    return blockers


def _blocker_reasons(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    if row.get("present") is not True:
        return [f"artifact_missing: {row.get('path')}"]
    reasons = _explicit_blockers(payload)
    failed_gate_reasons = [
        str(gate.get("reason"))
        for gate in _as_list(payload.get("gates_evaluated"))
        if _as_mapping(gate).get("passed") is False and gate.get("reason") is not None
    ]
    reasons += failed_gate_reasons
    ready_field = str(row.get("ready_field") or "")
    if not reasons and ready_field and payload.get(ready_field) is False:
        reasons.append(f"{ready_field}=false")
    return reasons


def _quality_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    flags = [_as_mapping(item) for item in _as_list(payload.get("corrigendum_pending"))]
    if payload.get("flagged_adversarial") is True and not flags:
        flags.append({"kind": "flagged_adversarial", "detail": "flagged_adversarial=true"})
    return [
        {
            "kind": str(flag.get("kind") or "flagged_adversarial"),
            "detail": str(flag.get("detail") or flag.get("severity") or ""),
        }
        for flag in flags
    ]


def _bounded_claims(payload: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    claims += ["sidecar_only=true"] if payload.get("sidecar_only") is True else []
    claims += ["pilot_only=true"] if payload.get("pilot_only") is True else []
    claims += (
        ["delong_noninferiority_passed=false"]
        if payload.get("delong_noninferiority_passed") is False
        else []
    )
    return claims


def _row_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {key: payload.get(key) for key in SUMMARY_KEYS if key in payload}


def _prior_publication_blocker_count(
    handoff: Mapping[str, Any], capstone_v302: Mapping[str, Any]
) -> int:
    candidates = (
        _int_value(handoff.get("prior_publication_blocker_count")),
        _int_value(capstone_v302.get("publication_blocker_count")),
        DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT,
    )
    return next(count for count in candidates if count > 0)


def _publication_readiness(rows: list[Mapping[str, Any]], publication_count: int) -> JsonDict:
    blocking_rows = [
        str(row.get("experiment_id"))
        for row in rows
        if row.get("status") in {"blocked", "missing", "pilot-only", "sidecar-only"}
    ]
    flagged_rows = [str(row.get("experiment_id")) for row in rows if row.get("quality_flags")]
    paper_ready = publication_count == 0 and not blocking_rows and not flagged_rows
    return {
        "paper_ready": paper_ready,
        "blocking_rows": blocking_rows,
        "flagged_rows": flagged_rows,
        "required_gates": {
            "full_15k_corpus": _row_ready(rows, "exp3272"),
            "kan_full_eval": _row_ready(rows, "exp3273"),
            "garak_redteam": _row_ready(rows, "exp3274"),
            "clean_verifier": _row_ready(rows, "exp3275"),
            "repair_gate": _row_ready(rows, "exp3276"),
            "repair_micro_panel": _row_ready(rows, "exp3277"),
            "fr11_full_corpus": _row_ready(rows, "exp3278"),
        },
    }


def _row_ready(rows: list[Mapping[str, Any]], experiment_id: str) -> bool:
    return any(row.get("experiment_id") == experiment_id and row.get("ready") is True for row in rows)


def _publication_movement(publication_count: int, prior_count: int) -> str:
    return "decreased" if publication_count < prior_count else "increased" if publication_count > prior_count else "unchanged"


def _next_gap_candidates(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    by_id = {str(row.get("experiment_id")): row for row in rows}
    candidates = [
        _gap("unblock_garak_redteam_eval", by_id.get("exp3274")),
        _gap("reduce_clean_verifier_abstention_rate", by_id.get("exp3275")),
        _gap("reopen_repair_gate_after_garak_and_clean_verifier", by_id.get("exp3276")),
        _gap("produce_or_gate_skip_repair_micro_panel_v9", by_id.get("exp3277")),
        _gap("resolve_prompt_injection_corrigendum_flags", _first_flagged(rows)),
        _gap("bound_kan_sidecar_or_improve_noninferiority", by_id.get("exp3273")),
    ]
    compact = [candidate for candidate in candidates if candidate["source_experiment_id"]]
    fallback = {
        "gap": "publication_blocker_retirement_review",
        "source_experiment_id": "",
        "reason": "all matrix rows are clean",
    }
    return [dict(candidate, rank=index) for index, candidate in enumerate(compact or [fallback], 1)]


def _gap(name: str, row: Mapping[str, Any] | None) -> JsonDict:
    row_map = _as_mapping(row)
    reasons = _list_of_strings(row_map.get("blocker_reasons"))
    reasons += [flag["kind"] for flag in _as_list(row_map.get("quality_flags"))]
    reasons += _list_of_strings(row_map.get("bounded_claims"))
    return {
        "gap": name if row_map.get("status") != "clean" else "",
        "source_experiment_id": str(row_map.get("experiment_id") or "")
        if row_map.get("status") != "clean"
        else "",
        "reason": reasons[0] if reasons else str(row_map.get("status") or ""),
    }


def _first_flagged(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("quality_flags")), {})


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _invariant_violations(payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    violations: list[str] = []
    if payloads.get("exp3267", {}).get("v302_closed_v303_opened") is not True:
        violations.append("exp3267 .303 handoff artifact is missing or not ready")
    return violations


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Matrix v35 reads checked-in .303 artifacts only.",
        "missing_is_not_success": "Absent artifacts stay missing and never imply a downstream pass.",
        "exact_blockers_preserved": "Blocked rows copy blocker strings and failed gate reasons.",
        "flags_visible": "Adversarial verifier and methodology flags are counted even on blocked rows.",
        "paper_ready_rule": "Publication readiness requires zero blockers and no blocked, flagged, missing, pilot-only, or sidecar-only required rows.",
    }


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("_", "-")
    return normalized if normalized in STATUSES else "missing"


def _bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "rows": artifact.get("rows"),
        "publication_blocker_count_estimate": artifact.get(
            "publication_blocker_count_estimate"
        ),
        "publication_blocker_delta_from_v302": artifact.get("publication_blocker_delta_from_v302"),
        "next_gap_candidates": artifact.get("next_gap_candidates"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v35_ready="
        f"{str(artifact.get('matrix_v35_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count_estimate={artifact.get('publication_blocker_count_estimate')}; "
        f"publication_blocker_delta_from_v302={artifact.get('publication_blocker_delta_from_v302')}; "
        f"clean={artifact.get('clean_row_count')}; "
        f"blocked={artifact.get('blocked_row_count')}; "
        f"flagged={artifact.get('flagged_row_count')}; "
        f"missing={artifact.get('missing_row_count')}; "
        f"sidecar_only={artifact.get('sidecar_only_row_count')}; "
        f"next_top_gap={_as_list(artifact.get('next_gap_candidates'))[0].get('gap')}"
    )
