"""Exp5527 exact-validated SOTA hard/soft panel v2.

Spec refs: REQ-VERIFY-5527, SCENARIO-VERIFY-5527.

This module is the narrow panel that comes after Exp5526 repaired the local
GGUF structured-row path. It does not introduce a new scorer. It takes bounded
candidate rows from the repaired local-SOTA evidence, parses them through the
Exp5512 schema, and recomputes every headline metric through the Exp5499 exact
hard/soft validators so stale row metadata cannot become a claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5526_sota_structured_repair_loop as repair_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5527_sota_hard_soft_panel_v2.json")
UPSTREAM_REPAIR_RELATIVE_PATH = repair_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5527.sota_hard_soft_panel_v2.v502"
EXPERIMENT = 5527
EXPERIMENT_ID = "exp5527-sota-hard-soft-panel-v2"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5527
INFERENCE_SUBSTRATE = "exact_validated_local_sota_gguf_panel"
SPEC_REFS = ("REQ-VERIFY-5527", "SCENARIO-VERIFY-5527")

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "models_attempted",
    "rows_requested",
    "rows_emitted",
    "schema_validity_rate",
    "missing_candidate_rows",
    "exact_validator_accuracy",
    "preference_optimality_rate",
    "abstention_rate",
    "confident_wrong_rate",
    "gpu_offload_evidence",
    "sota_structured_panel_ready",
    "sota_hard_soft_claim_allowed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5527_sota_hard_soft_panel_v2.py",
    "tests/python/test_experiment_5526_sota_structured_repair_loop.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
    "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Names all mandated local GGUF candidates so panel evidence cannot detach from the approved SOTA set.",
    "models_attempted": "Identifies the local SOTA GGUF model path that actually supplied upstream repaired-row evidence.",
    "rows_requested": "Bounds the hard/soft fixture denominator before candidate parsing.",
    "rows_emitted": "Separates model/repaired-row output volume from correctness.",
    "schema_validity_rate": "Gates validator handoff on schema-valid rows rather than parser optimism.",
    "missing_candidate_rows": "Keeps absent rows visible and prevents treating missing output as abstention.",
    "exact_validator_accuracy": "Reports only deterministic Exp5499 validator correctness for schema-valid rows.",
    "preference_optimality_rate": "Reports soft-preference optimality only after hard constraints pass.",
    "abstention_rate": "Counts explicit schema-valid abstentions without absorbing missing rows.",
    "confident_wrong_rate": "Surfaces high-confidence schema-valid failures separately from calibrated abstention.",
    "gpu_offload_evidence": "Preserves the runtime/offload receipt for the local GGUF substrate.",
    "sota_structured_panel_ready": "States whether the exact-validated structured panel gate is open.",
    "sota_hard_soft_claim_allowed": "Controls whether the bounded hard/soft claim may be cited.",
    "tests_added_or_reused": "Links the artifact to tests for parsing and exact validator handoff.",
    "field_principles": "Explains why each headline and gate field must stay present.",
    "inference_substrate": "Declares exact-validated local SOTA GGUF panel semantics and excludes external text scoring.",
    "honest_verdict": "Provides the terminal status without promoting missing or schema-invalid rows.",
    "upstream_repair_loop_ready": "Confirms Exp5526 opened the repaired-row gate before this panel runs.",
    "upstream_repair_artifact": "Pins the source artifact so repaired rows cannot be silently substituted.",
    "no_autotokenizer_on_gguf": "Guards against the known invalid transformers tokenizer path for GGUF repositories.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_upstream_repair_artifact(
    path: Path = REPO_ROOT / UPSTREAM_REPAIR_RELATIVE_PATH,
) -> JsonDict:
    """Load the Exp5526 repair-loop artifact used as the v2 gate."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_artifact(
    *,
    repair_path: Path = REPO_ROOT / UPSTREAM_REPAIR_RELATIVE_PATH,
    upstream_repair_artifact: Mapping[str, Any] | None = None,
    row_limit: int | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5527 exact-validated panel artifact."""

    started = time.monotonic()
    upstream_loadable = True
    upstream_load_error = None
    if upstream_repair_artifact is None:
        try:
            repair_artifact = load_upstream_repair_artifact(Path(repair_path))
        except Exception as exc:  # noqa: BLE001
            upstream_loadable = False
            upstream_load_error = f"{type(exc).__name__}: {exc}"
            repair_artifact = {}
    else:
        repair_artifact = dict(upstream_repair_artifact)

    fixture = positive.load_fixture_artifact()["fixture"]
    requested_payloads = positive.build_fixture_candidate_payloads(fixture)
    if row_limit is not None:
        requested_payloads = requested_payloads[: max(0, int(row_limit))]
    requested_ids = [str(row["instance_id"]) for row in requested_payloads]

    upstream_ready = _upstream_repair_loop_ready(repair_artifact)
    source_rows = _bounded_candidate_rows(
        repair_artifact.get("repaired_candidate_rows", []) if upstream_ready else [],
        requested_ids,
    )
    report = evaluate_candidate_rows(
        source_rows,
        fixture=fixture,
        requested_instance_ids=requested_ids,
    )
    models_attempted = _models_attempted(repair_artifact)
    gpu_offload_evidence = _gpu_offload_evidence(repair_artifact)
    blockers = _readiness_blockers(
        upstream_loadable=upstream_loadable,
        upstream_ready=upstream_ready,
        models_attempted=models_attempted,
        gpu_offload_evidence=gpu_offload_evidence,
        report=report,
    )
    ready = not blockers
    claim_allowed = ready
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "model_specs": _model_specs(repair_artifact),
        "models_attempted": models_attempted,
        "rows_requested": report["rows_requested"],
        "rows_emitted": report["rows_emitted"],
        "schema_validity_rate": report["schema_validity_rate"],
        "missing_candidate_rows": report["missing_candidate_rows"],
        "exact_validator_accuracy": report["exact_validator_accuracy"],
        "preference_optimality_rate": report["preference_optimality_rate"],
        "abstention_rate": report["abstention_rate"],
        "confident_wrong_rate": report["confident_wrong_rate"],
        "gpu_offload_evidence": gpu_offload_evidence,
        "sota_structured_panel_ready": ready,
        "sota_hard_soft_claim_allowed": claim_allowed,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(claim_allowed, blockers),
        "upstream_repair_artifact": UPSTREAM_REPAIR_RELATIVE_PATH.as_posix(),
        "upstream_repair_loadable": upstream_loadable,
        "upstream_repair_load_error": upstream_load_error,
        "upstream_repair_loop_ready": upstream_ready,
        "upstream_repair_honest_verdict": str(repair_artifact.get("honest_verdict", "")),
        "panel_row_source": "exp5526_repaired_candidate_rows",
        "panel_rows": report["panel_rows"],
        "extra_emitted_rows": report["extra_emitted_rows"],
        "missing_instance_ids": report["missing_instance_ids"],
        "exact_validator_rows_scored": report["exact_validator_rows_scored"],
        "schema_valid_rows": report["schema_valid_rows"],
        "assignment_rows_scored": report["assignment_rows_scored"],
        "confident_wrong_rows": report["confident_wrong_rows"],
        "readiness_blockers": blockers,
        "no_autotokenizer_on_gguf": True,
        "no_external_text_scorer_claim": True,
        "logits_energy_diagnostics_sidecar_only": True,
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "duration_s": round(time.monotonic() - started, 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def evaluate_candidate_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    fixture: Mapping[str, Any] | None = None,
    requested_instance_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Parse candidate rows and recompute exact-validator metrics."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    if requested_instance_ids is None:
        requested_instance_ids = [
            str(row["instance_id"]) for row in positive.build_fixture_candidate_payloads(fixture_payload)
        ]
    requested_ids = [str(row) for row in requested_instance_ids]
    slots: dict[str, JsonDict | None] = {instance_id: None for instance_id in requested_ids}
    extra_rows: list[JsonDict] = []

    for record in candidate_rows:
        row = _reclassify_candidate_record(record, fixture_payload)
        instance_id = str(row.get("instance_id") or "")
        if instance_id in slots and slots[instance_id] is None:
            slots[instance_id] = row
        else:
            extra_rows.append(row)

    panel_rows = [row for instance_id in requested_ids if (row := slots[instance_id]) is not None]
    missing_ids = [instance_id for instance_id in requested_ids if slots[instance_id] is None]
    schema_valid_rows = [row for row in panel_rows if row.get("schema_valid") is True]
    assignment_rows = [
        row for row in schema_valid_rows if row.get("parse_status") == "schema_valid_assignment"
    ]
    abstention_rows = [
        row for row in schema_valid_rows if row.get("parse_status") == "schema_valid_abstention"
    ]
    confident_wrong_rows = [row for row in schema_valid_rows if _confident_wrong(row)]
    rows_requested = len(requested_ids)
    return {
        "rows_requested": rows_requested,
        "rows_emitted": len(candidate_rows),
        "schema_validity_rate": _rate(len(schema_valid_rows), rows_requested),
        "missing_candidate_rows": len(missing_ids),
        "exact_validator_accuracy": _rate(
            sum(int(row.get("exact_validator_correct") is True) for row in schema_valid_rows),
            len(schema_valid_rows),
        ),
        "preference_optimality_rate": _rate(
            sum(int(row.get("soft_optimal") is True) for row in assignment_rows),
            len(assignment_rows),
        ),
        "abstention_rate": _rate(len(abstention_rows), rows_requested),
        "confident_wrong_rate": _rate(len(confident_wrong_rows), len(schema_valid_rows)),
        "panel_rows": panel_rows,
        "extra_emitted_rows": extra_rows,
        "missing_instance_ids": missing_ids,
        "exact_validator_rows_scored": len(schema_valid_rows),
        "schema_valid_rows": len(schema_valid_rows),
        "assignment_rows_scored": len(assignment_rows),
        "confident_wrong_rows": len(confident_wrong_rows),
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    repair_path: Path = REPO_ROOT / UPSTREAM_REPAIR_RELATIVE_PATH,
    row_limit: int | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5527 deliverable JSON."""

    artifact = build_artifact(
        repair_path=Path(repair_path),
        row_limit=row_limit,
        tests_run=tests_run,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate Exp5527 fields and fail closed on overclaiming."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")),
        "honest_verdict",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(_model_ids_match_mandated(artifact.get("model_specs", [])), "model_specs")
    _require(
        set(artifact.get("models_attempted", [])).issubset(positive.MANDATED_HEADLINE_MODEL_IDS),
        "models_attempted",
    )
    for field in (
        "schema_validity_rate",
        "exact_validator_accuracy",
        "preference_optimality_rate",
        "abstention_rate",
        "confident_wrong_rate",
    ):
        _require(0.0 <= float(artifact.get(field, -1.0)) <= 1.0, field)
    _require(int(artifact.get("rows_requested", -1)) >= 0, "rows_requested")
    _require(int(artifact.get("rows_emitted", -1)) >= 0, "rows_emitted")
    _require(int(artifact.get("missing_candidate_rows", -1)) >= 0, "missing_candidate_rows")
    _require(isinstance(artifact.get("gpu_offload_evidence"), Mapping), "gpu_offload_evidence")
    _require(isinstance(artifact.get("sota_structured_panel_ready"), bool), "ready")
    _require(isinstance(artifact.get("sota_hard_soft_claim_allowed"), bool), "claim_allowed")
    _require(
        artifact.get("sota_structured_panel_ready")
        is artifact.get("sota_hard_soft_claim_allowed"),
        "sota_hard_soft_claim_allowed",
    )
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("no_external_text_scorer_claim") is True, "text_scorer")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    if artifact.get("sota_hard_soft_claim_allowed") is True:
        _require(artifact.get("upstream_repair_loop_ready") is True, "upstream_repair_loop_ready")
        _require(bool(artifact.get("models_attempted")), "models_attempted")
        _require(
            artifact.get("gpu_offload_evidence", {}).get("gpu_offload_verified") is True,
            "gpu_offload_evidence",
        )
        _require(int(artifact.get("rows_requested", 0)) > 0, "rows_requested")
        _require(
            int(artifact.get("rows_emitted", 0)) >= int(artifact.get("rows_requested", 0)),
            "rows_emitted",
        )
        _require(artifact.get("schema_validity_rate") == 1.0, "schema_validity_rate")
        _require(artifact.get("missing_candidate_rows") == 0, "missing_candidate_rows")
        _require(artifact.get("exact_validator_accuracy") == 1.0, "exact_validator_accuracy")
        _require(artifact.get("preference_optimality_rate") == 1.0, "preference_optimality_rate")
        _require(artifact.get("confident_wrong_rate") == 0.0, "confident_wrong_rate")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(claim_allowed: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that names the exact hard/soft claim gate."""

    if claim_allowed:
        return "complete: sota_hard_soft_panel_v2_ready_bounded_exact_validated_claim_allowed"
    suffix = "_".join(blockers) if blockers else "insufficient_exact_validated_evidence"
    return f"blocked: sota_hard_soft_panel_v2_not_ready_{suffix}"


def _upstream_repair_loop_ready(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("repair_loop_ready") is True
        or artifact.get("sota_structured_repair_loop_ready") is True
    )


def _bounded_candidate_rows(
    rows: Any,
    requested_ids: Sequence[str],
) -> list[JsonDict]:
    requested = set(requested_ids)
    bounded = []
    for row in rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []:
        if not isinstance(row, Mapping):
            continue
        instance_id = _record_instance_id(row)
        if instance_id is None or instance_id in requested:
            bounded.append(dict(row))
    return bounded


def _record_instance_id(record: Mapping[str, Any]) -> str | None:
    payload = _payload_from_record(record)
    if isinstance(payload, Mapping) and payload.get("instance_id"):
        return str(payload["instance_id"])
    return None


def _reclassify_candidate_record(record: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    payload = _payload_from_record(record)
    row = positive.classify_candidate_payload(payload, fixture=fixture)
    _attach_payload_metadata(row, payload)
    row["source_exact_validator_correct"] = record.get("exact_validator_correct")
    row["source_exact_validator_verdict"] = record.get("exact_validator_verdict")
    row["source_soft_optimal"] = record.get("soft_optimal")
    row["source_reference_agreement"] = record.get("reference_agreement")
    row["source_repair_method"] = record.get("repair_method")
    row["model_hf_id"] = record.get("model_hf_id")
    return row


def _payload_from_record(record: Mapping[str, Any]) -> Any:
    payload = record.get("parsed_payload")
    if isinstance(payload, Mapping):
        return payload
    if record.get("candidate_schema_version") == positive.CANDIDATE_SCHEMA_VERSION:
        return record
    return record


def _attach_payload_metadata(row: JsonDict, payload: Mapping[str, Any]) -> None:
    conclusion = payload.get("conclusion")
    if isinstance(conclusion, Mapping):
        row["conclusion_status"] = str(conclusion.get("status", ""))
        try:
            row["candidate_confidence"] = float(conclusion.get("confidence", 0.0))
        except (TypeError, ValueError):
            row["candidate_confidence"] = 0.0
    else:
        row["conclusion_status"] = ""
        row["candidate_confidence"] = 0.0


def _models_attempted(artifact: Mapping[str, Any]) -> list[str]:
    attempted: list[str] = []
    for field in ("models_attempted", "smoke_models_used", "headline_models_used"):
        values = artifact.get(field, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            hf_id = str(value)
            if hf_id in positive.MANDATED_HEADLINE_MODEL_IDS and hf_id not in attempted:
                attempted.append(hf_id)
    return attempted


def _model_specs(artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows = artifact.get("model_specs", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        specs = [dict(row) for row in rows if isinstance(row, Mapping)]
        if _model_ids_match_mandated(specs):
            return specs
    return [dict(row) for row in positive.MODEL_SPECS]


def _model_ids_match_mandated(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    ids = [str(row.get("hf_id")) for row in rows if isinstance(row, Mapping)]
    return ids == list(positive.MANDATED_HEADLINE_MODEL_IDS)


def _gpu_offload_evidence(repair_artifact: Mapping[str, Any]) -> JsonDict:
    direct = repair_artifact.get("gpu_offload_evidence")
    if isinstance(direct, Mapping):
        evidence = dict(direct)
        evidence.setdefault("evidence_source", UPSTREAM_REPAIR_RELATIVE_PATH.as_posix())
        return evidence
    taxonomy_path = repair_artifact.get("upstream_taxonomy_path")
    if isinstance(taxonomy_path, str) and taxonomy_path:
        path = Path(taxonomy_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            taxonomy = json.loads(path.read_text(encoding="utf-8"))
            evidence = taxonomy.get("gpu_offload_evidence")
            if isinstance(evidence, Mapping):
                copied = dict(evidence)
                copied["evidence_source"] = taxonomy_path
                return copied
        except Exception as exc:  # noqa: BLE001
            return {
                "gpu_offload_verified": False,
                "llama_cpp_cuda_available": False,
                "gpu_memory_delta_mb": 0.0,
                "offload_diagnostics": [],
                "evidence_source": taxonomy_path,
                "load_error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "gpu_offload_verified": False,
        "llama_cpp_cuda_available": False,
        "gpu_memory_delta_mb": 0.0,
        "offload_diagnostics": [],
        "evidence_source": None,
    }


def _readiness_blockers(
    *,
    upstream_loadable: bool,
    upstream_ready: bool,
    models_attempted: Sequence[str],
    gpu_offload_evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[str]:
    blockers = []
    if not upstream_loadable:
        blockers.append("upstream_repair_artifact_not_loadable")
    if not upstream_ready:
        blockers.append("upstream_repair_loop_not_ready")
    if not models_attempted:
        blockers.append("no_mandated_sota_model_attempted")
    if gpu_offload_evidence.get("gpu_offload_verified") is not True:
        blockers.append("gpu_offload_evidence_absent_or_false")
    if int(report.get("rows_requested", 0)) <= 0:
        blockers.append("no_rows_requested")
    if int(report.get("missing_candidate_rows", 0)) > 0:
        blockers.append("missing_candidate_rows")
    if float(report.get("schema_validity_rate", 0.0)) < 1.0:
        blockers.append("schema_invalid_or_missing_rows")
    if float(report.get("exact_validator_accuracy", 0.0)) < 1.0:
        blockers.append("exact_validator_mismatch")
    if float(report.get("preference_optimality_rate", 0.0)) < 1.0:
        blockers.append("preference_suboptimal_or_unscored")
    if float(report.get("confident_wrong_rate", 0.0)) > 0.0:
        blockers.append("confident_wrong_rows")
    return sorted(set(blockers))


def _confident_wrong(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("conclusion_status") == "candidate"
        and row.get("exact_validator_correct") is False
        and float(row.get("candidate_confidence", 0.0)) >= 0.5
    )


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "sota_hard_soft_claim_allowed": artifact["sota_hard_soft_claim_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
