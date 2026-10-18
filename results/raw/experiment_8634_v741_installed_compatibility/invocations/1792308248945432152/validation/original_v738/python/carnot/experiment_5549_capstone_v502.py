"""Exp5549 capstone reconciliation for milestone 2026.07.502.

Spec refs: REQ-REPORT-5549, SCENARIO-REPORT-5549,
SCENARIO-REPORT-5549-MISSING-INPUT, SCENARIO-REPORT-5549-FIELD-PRINCIPLES.

This module is a synthesis-only claim ledger. It reads the `.502` experiment
artifacts that already landed, separates flags, blocked gates, clean nulls, and
clean evidence, then emits the smallest set of headline booleans that the
upstream evidence actually permits. The important constraint is that a capstone
cannot launder a flagged five-arm CSL result, a blocked cross-model transfer,
or an ARC no-bank attempt into a success by averaging them with clean receipts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5549_capstone_v502.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5549_capstone_v502"
EXPERIMENT_ID = "exp5549-v502-capstone-reconciliation"
MILESTONE = "2026.07.502"
TASK_RANGE = "exp5536-exp5549"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5549
SCHEMA = "carnot.experiment_5549.capstone_v502.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

QWEN_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_26_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA_31_HF_ID = "unsloth/gemma-4-31B-it-GGUF"
MANDATED_HF_IDS = (QWEN_HF_ID, GEMMA_26_HF_ID, GEMMA_31_HF_ID)

SPEC_REFS = (
    "REQ-REPORT-5549",
    "SCENARIO-REPORT-5549",
    "SCENARIO-REPORT-5549-MISSING-INPUT",
    "SCENARIO-REPORT-5549-FIELD-PRINCIPLES",
)

PRIMARY_ARTIFACT_PATHS = (
    Path("results/experiment_5536_transition_v502.json"),
    Path("results/experiment_5537_v502_source_delta_ingestion.json"),
    Path("results/experiment_5538_sota_panel_duration_substrate_corrigendum.json"),
    Path("results/experiment_5539_gram2token_grammar_table_preflight.json"),
    Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"),
    Path("results/experiment_5541_llm_fsm_exact_fixture.json"),
    Path("results/experiment_5542_csl_residue_metric_independence_corrigendum.json"),
    Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"),
    Path("results/experiment_5544_cross_model_sota_csl_transfer.json"),
    Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json"),
    Path("results/experiment_5546_hardware_receipt_substrate_corrigendum.json"),
    Path("results/experiment_5547_arc_no_llm_substrate_precheck.json"),
    Path("results/experiment_5548_arc_clean_live_levelup.json"),
)
AUXILIARY_ARTIFACT_PATHS = (Path("results/experiment_5548_arc_clean_live_levelup_trajectory.json"),)
EXPECTED_ARTIFACT_PATHS = (*PRIMARY_ARTIFACT_PATHS, *AUXILIARY_ARTIFACT_PATHS)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/conductor-log.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/e2e-test-plan.md"),
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the `.502` capstone.",
    "task_range": "Closed conductor boundary from transition through this capstone.",
    "artifacts_expected": (
        "Count of upstream `.502` artifacts Exp5549 expected to inspect before making claims."
    ),
    "artifacts_read": "Count of expected upstream artifacts actually parsed as JSON evidence.",
    "missing_artifacts": (
        "Absent or unreadable expected artifacts stay visible and never become successful evidence."
    ),
    "skipped_by_gates": "Blocked or gated upstreams are preserved as skipped evidence, not promoted.",
    "flagged_artifacts": (
        "Adversarial or methodology-flagged upstreams are excluded from headline claims."
    ),
    "honest_nulls": "Clean negative or no-bank results are recorded separately from failures.",
    "clean_artifacts": (
        "Readable, unflagged, unblocked, non-null artifacts safe for bounded aggregation."
    ),
    "structured_sota_claim_allowed": (
        "Bare boolean for complete schema-valid exact-validated SOTA structured rows."
    ),
    "sota_hard_soft_claim_allowed": (
        "Bare boolean imported only from clean Exp5540 hard/soft gate evidence."
    ),
    "continuous_self_learning_evidence": (
        "Bare boolean for clean residue independence evidence, separate from broad CSL claim eligibility."
    ),
    "csl_claim_allowed": (
        "Bare boolean blocked by flagged five-arm evidence or blocked cross-model transfer."
    ),
    "sparse_repair_claim_allowed": (
        "Bare boolean for exact-checked FSM repair evidence, not a speedup claim."
    ),
    "hardware_speedup_claim": "Must remain false without authenticated matched timing evidence.",
    "arc_registry_delta": (
        "Registry delta imported only from offline-reproduced live self-discovery ARC evidence."
    ),
    "reproduced_levels": (
        "Reproduced-level count imported only from the live attempt, never from capstone aggregation."
    ),
    "protected_files_unchanged": (
        "Protected-file map for `research-roadmap.yaml` and `scripts/research_conductor.py`."
    ),
    "docs_updated": (
        "Files intentionally updated by this workflow; ops/status, ops/changelog, and BMAD remain untouched when reconciler-owned."
    ),
    "checks_run": "Validation commands and protected-file checks actually run.",
    "field_principles": "One-line annotations for every headline and gate field.",
    "inference_substrate": (
        "Must equal `aggregation_from_upstream_artifacts` because Exp5549 is synthesis only."
    ),
    "honest_verdict": (
        "Terminal summary starting with `complete:` or `blocked:` that names the true `.502` outcome."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "source_context",
    "source_context_missing",
    "artifact_metadata",
    "artifact_paths_read",
    "failed_artifacts",
    "claim_boundaries",
    "llm_model_spec_audit",
    "arc_audit",
    "arc_live_levelup_claim_allowed",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
BOOL_FIELDS = (
    "structured_sota_claim_allowed",
    "sota_hard_soft_claim_allowed",
    "continuous_self_learning_evidence",
    "csl_claim_allowed",
    "sparse_repair_claim_allowed",
    "hardware_speedup_claim",
    "arc_live_levelup_claim_allowed",
)
INT_FIELDS = ("artifacts_expected", "artifacts_read", "arc_registry_delta", "reproduced_levels")
LIST_FIELDS = (
    "artifact_paths_read",
    "missing_artifacts",
    "skipped_by_gates",
    "flagged_artifacts",
    "honest_nulls",
    "clean_artifacts",
    "failed_artifacts",
    "docs_updated",
    "checks_run",
)
DEFAULT_DOCS_UPDATED = ("openspec/capabilities/research-reporting/spec.md",)
DEFAULT_CHECKS_RUN = (
    "PENDING: .venv/bin/pytest tests/python/test_experiment_5549_capstone_v502.py -q --no-cov -n 0",
    (
        "PENDING: .venv/bin/coverage run "
        "--include=python/carnot/experiment_5549_capstone_v502.py "
        "-m pytest tests/python/test_experiment_5549_capstone_v502.py -q --no-cov -n 0"
    ),
    (
        "PENDING: .venv/bin/coverage report "
        "--include=python/carnot/experiment_5549_capstone_v502.py --fail-under=100"
    ),
    "PENDING: .venv/bin/pytest tests/python -q",
    "PENDING: python scripts/check_spec_coverage.py",
    "PENDING: python scripts/root_clutter_sweep.py --check",
    "PENDING: git status --short -- research-roadmap.yaml scripts/research_conductor.py",
)


def _read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[str], list[str]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    paths_read: list[str] = []
    missing: list[str] = []
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if meta.get("exists") and meta.get("loadable"):
            paths_read.append(rel)
        else:
            missing.append(rel)
    return artifacts, metadata, paths_read, missing


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    return artifacts.get(rel_path.as_posix(), {})


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _status_label(payload: JsonMap) -> str:
    status = payload.get("status")
    if status is not None:
        return str(status).lower()
    verdict = _verdict(payload).lower()
    if verdict.startswith("blocked:"):
        return "blocked"
    if verdict.startswith("honest_null:") or "honest_null" in verdict:
        return "honest_null"
    if verdict.startswith("failed:"):
        return "failed"
    if verdict.startswith("complete:"):
        return "complete"
    return "unknown"


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _is_blocked(payload: JsonMap) -> bool:
    return _status_label(payload) == "blocked"


def _is_failed(payload: JsonMap) -> bool:
    return _status_label(payload) in {"failed", "error"}


def _is_honest_null(payload: JsonMap) -> bool:
    return _status_label(payload) == "honest_null"


def _clean_for_claim(payload: JsonMap) -> bool:
    return bool(payload) and not (
        _is_flagged(payload) or _is_blocked(payload) or _is_failed(payload) or _is_honest_null(payload)
    )


def _artifact_row(rel_path: str, payload: JsonMap, metadata: JsonMap) -> JsonDict:
    return {
        "artifact_path": rel_path,
        "status": payload.get("status", _status_label(payload)),
        "honest_verdict": payload.get("honest_verdict"),
        "flagged_adversarial": bool(payload.get("flagged_adversarial")),
        "inference_substrate": payload.get("inference_substrate"),
        "sha256": metadata.get(rel_path, {}).get("sha256") if isinstance(metadata, Mapping) else None,
    }


def classify_artifacts(
    artifacts: Mapping[str, JsonMap], metadata: JsonMap
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    flagged: list[JsonDict] = []
    skipped: list[JsonDict] = []
    failed: list[JsonDict] = []
    honest_nulls: list[JsonDict] = []
    clean: list[JsonDict] = []
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        rel = rel_path.as_posix()
        payload = artifacts.get(rel, {})
        if not payload:
            continue
        row = _artifact_row(rel, payload, metadata)
        if _is_flagged(payload):
            row["skip_reason"] = "flagged_adversarial"
            row["corrigendum_pending"] = payload.get("corrigendum_pending", [])
            flagged.append(row)
        elif _is_blocked(payload):
            row["skip_reason"] = "blocked_or_gated"
            skipped.append(row)
        elif _is_failed(payload):
            row["failure_reason"] = "failed_terminal_status"
            failed.append(row)
        elif _is_honest_null(payload):
            row["null_reason"] = payload.get("failure_mode") or "clean_honest_null"
            honest_nulls.append(row)
        else:
            clean.append(row)
    return flagged, skipped, failed, honest_nulls, clean


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    return int(value) if isinstance(value, int | float | str) and str(value).lstrip("-").isdigit() else 0


def _structured_sota_claim_allowed(artifacts: Mapping[str, JsonMap]) -> bool:
    exp5540 = _payload(artifacts, Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"))
    rows_requested = _int(exp5540, "rows_requested")
    return bool(
        _clean_for_claim(exp5540)
        and exp5540.get("gates_clean")
        and exp5540.get("adversarial_clean")
        and rows_requested > 0
        and _int(exp5540, "rows_emitted") == rows_requested
        and _int(exp5540, "schema_valid_rows") == rows_requested
        and float(exp5540.get("exact_validator_accuracy") or 0.0) >= 1.0
        and _int(exp5540, "missing_candidate_rows") == 0
    )


def _claim_booleans(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    exp5540 = _payload(artifacts, Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"))
    exp5542 = _payload(
        artifacts, Path("results/experiment_5542_csl_residue_metric_independence_corrigendum.json")
    )
    exp5543 = _payload(artifacts, Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"))
    exp5544 = _payload(artifacts, Path("results/experiment_5544_cross_model_sota_csl_transfer.json"))
    exp5545 = _payload(artifacts, Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json"))
    exp5546 = _payload(artifacts, Path("results/experiment_5546_hardware_receipt_substrate_corrigendum.json"))

    structured = _structured_sota_claim_allowed(artifacts)
    continuous_self_learning = bool(
        _clean_for_claim(exp5542)
        and exp5542.get("csl_residue_tautology_resolved")
        and exp5542.get("nonidentical_metric_evidence")
    )
    csl_allowed = bool(
        continuous_self_learning
        and _clean_for_claim(exp5543)
        and exp5543.get("csl_five_arm_ready")
        and _clean_for_claim(exp5544)
        and exp5544.get("csl_claim_allowed")
        and exp5544.get("no_weight_mutation")
    )
    return {
        "structured_sota_claim_allowed": structured,
        "sota_hard_soft_claim_allowed": bool(
            structured and _clean_for_claim(exp5540) and exp5540.get("sota_hard_soft_claim_allowed")
        ),
        "continuous_self_learning_evidence": continuous_self_learning,
        "csl_claim_allowed": csl_allowed,
        "sparse_repair_claim_allowed": bool(
            _clean_for_claim(exp5545)
            and exp5545.get("sparse_repair_fsm_ready")
            and exp5545.get("exact_validator_all_repairs_checked")
            and _int(exp5545, "unchecked_repair_count") == 0
        ),
        "hardware_speedup_claim": bool(
            _clean_for_claim(exp5546)
            and exp5546.get("hardware_speedup_claim")
            and exp5546.get("matched_timing_available")
        ),
    }


def _arc_claims(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    exp5548 = _payload(artifacts, Path("results/experiment_5548_arc_clean_live_levelup.json"))
    live_success = bool(
        _clean_for_claim(exp5548)
        and exp5548.get("solve_provenance") == "live_agent_self_discovery"
        and exp5548.get("offline_reproduced")
        and _int(exp5548, "registry_delta") > 0
        and _int(exp5548, "reproduced_levels") > 0
    )
    return {
        "arc_live_levelup_claim_allowed": live_success,
        "arc_registry_delta": _int(exp5548, "registry_delta") if live_success else 0,
        "reproduced_levels": _int(exp5548, "reproduced_levels") if live_success else 0,
        "arc_audit": {
            "source_artifact": "results/experiment_5548_arc_clean_live_levelup.json",
            "solve_provenance": exp5548.get("solve_provenance"),
            "offline_reproduced": bool(exp5548.get("offline_reproduced")),
            "registry_delta_raw": _int(exp5548, "registry_delta"),
            "reproduced_levels_raw": _int(exp5548, "reproduced_levels"),
            "capstone_counted_as_levelup_attempt": False,
        },
    }


def _model_spec_complete(spec: JsonMap) -> bool:
    hf_id = spec.get("hf_id")
    path = str(spec.get("model_path") or spec.get("model_filename") or "")
    quant = spec.get("preferred_quant") or spec.get("quantization")
    local_present = bool(
        spec.get("local_model_present")
        or spec.get("local_path_available")
        or (isinstance(spec.get("file_receipt"), Mapping) and spec["file_receipt"].get("exists"))
    )
    return bool(hf_id in MANDATED_HF_IDS and path.endswith(".gguf") and quant == "Q4_K_M" and local_present)


def _no_model_specs_explained(payload: JsonMap, rel_path: str) -> bool:
    substrate = str(payload.get("inference_substrate") or "")
    return bool(
        rel_path in {path.as_posix() for path in AUXILIARY_ARTIFACT_PATHS}
        or payload.get("no_model_specs_required")
        or "no_llm" in substrate
        or substrate == INFERENCE_SUBSTRATE
    )


def audit_model_specs(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    llm_rows: list[JsonDict] = []
    no_llm_rows: list[JsonDict] = []
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        rel = rel_path.as_posix()
        payload = artifacts.get(rel, {})
        if not payload:
            continue
        specs = payload.get("model_specs")
        if isinstance(specs, list) and specs:
            hf_ids = sorted({str(spec.get("hf_id")) for spec in specs if isinstance(spec, Mapping)})
            llm_rows.append(
                {
                    "artifact_path": rel,
                    "hf_ids": hf_ids,
                    "mandated_hf_ids_present": sorted(MANDATED_HF_IDS),
                    "all_specs_complete": all(
                        _model_spec_complete(spec) for spec in specs if isinstance(spec, Mapping)
                    ),
                }
            )
        else:
            no_llm_rows.append(
                {
                    "artifact_path": rel,
                    "absence_explained": _no_model_specs_explained(payload, rel),
                    "inference_substrate": payload.get("inference_substrate"),
                    "no_model_specs_required": bool(payload.get("no_model_specs_required")),
                }
            )
    return {
        "llm_bearing_artifacts": llm_rows,
        "no_llm_or_aggregation_artifacts": no_llm_rows,
        "all_mandated_specs_present": bool(llm_rows)
        and all(set(row["hf_ids"]) >= set(MANDATED_HF_IDS) and row["all_specs_complete"] for row in llm_rows),
        "all_absences_explained": all(row["absence_explained"] for row in no_llm_rows),
    }


def build_artifact(
    artifacts: Mapping[str, JsonMap],
    artifact_metadata: JsonMap,
    artifact_paths_read: Sequence[str],
    missing_artifacts: Sequence[str],
    source_context: Sequence[JsonMap],
    source_context_missing: Sequence[str],
    *,
    checks_run: Sequence[str],
    docs_updated: Sequence[str],
    roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    flagged, skipped, failed, honest_nulls, clean = classify_artifacts(artifacts, artifact_metadata)
    booleans = _claim_booleans(artifacts)
    arc = _arc_claims(artifacts)
    protected = {
        ROADMAP_RELATIVE_PATH.as_posix(): not roadmap_modified,
        CONDUCTOR_RELATIVE_PATH.as_posix(): not conductor_modified,
    }
    status_prefix = "blocked:" if missing_artifacts else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": [dict(row) for row in source_context],
        "source_context_missing": list(source_context_missing),
        "artifact_metadata": dict(artifact_metadata),
        "artifact_paths_read": list(artifact_paths_read),
        "failed_artifacts": failed,
        "claim_boundaries": [
            "SOTA structured and hard/soft claims stay false because Exp5540 is an honest null with missing rows.",
            "CSL evidence exists at the residue-corrigendum layer, but Exp5543 is flagged and Exp5544 is blocked.",
            "Sparse repair is allowed only as exact-checked FSM evidence; no speedup is claimed.",
            "Hardware speedup remains false because Exp5546 has receipt hygiene without matched timing.",
            "ARC live level-up remains false because Exp5548 did not offline-reproduce a new level.",
        ],
        "llm_model_spec_audit": audit_model_specs(artifacts),
        "arc_audit": arc["arc_audit"],
        "milestone": MILESTONE,
        "task_range": TASK_RANGE,
        "artifacts_expected": len(EXPECTED_ARTIFACT_PATHS),
        "artifacts_read": len(artifact_paths_read),
        "missing_artifacts": list(missing_artifacts),
        "skipped_by_gates": skipped,
        "flagged_artifacts": flagged,
        "honest_nulls": honest_nulls,
        "clean_artifacts": clean,
        "protected_files_unchanged": protected,
        "docs_updated": list(docs_updated),
        "checks_run": list(checks_run),
        "inference_substrate": INFERENCE_SUBSTRATE,
        **booleans,
        "arc_live_levelup_claim_allowed": arc["arc_live_levelup_claim_allowed"],
        "arc_registry_delta": arc["arc_registry_delta"],
        "reproduced_levels": arc["reproduced_levels"],
    }
    payload["honest_verdict"] = (
        f"{status_prefix} .502 capstone read {payload['artifacts_read']}/"
        f"{payload['artifacts_expected']} expected artifacts; "
        f"missing={len(payload['missing_artifacts'])}; flagged={len(flagged)}; "
        f"skipped_by_gates={len(skipped)}; honest_nulls={len(honest_nulls)}; "
        f"structured_sota_claim_allowed={payload['structured_sota_claim_allowed']}; "
        f"sota_hard_soft_claim_allowed={payload['sota_hard_soft_claim_allowed']}; "
        f"continuous_self_learning_evidence={payload['continuous_self_learning_evidence']}; "
        f"csl_claim_allowed={payload['csl_claim_allowed']}; "
        f"sparse_repair_claim_allowed={payload['sparse_repair_claim_allowed']}; "
        f"hardware_speedup_claim={payload['hardware_speedup_claim']}; "
        f"arc_registry_delta={payload['arc_registry_delta']}"
    )
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run_capstone(
    root: Path = REPO_ROOT,
    *,
    checks_run: Sequence[str] = DEFAULT_CHECKS_RUN,
    docs_updated: Sequence[str] = DEFAULT_DOCS_UPDATED,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, paths_read, missing_artifacts = _read_artifacts(root)
    source_context, source_missing = _read_source_context(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    return build_artifact(
        artifacts,
        metadata,
        paths_read,
        missing_artifacts,
        source_context,
        source_missing,
        checks_run=checks_run,
        docs_updated=docs_updated,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    for field in INT_FIELDS:
        if field in payload and not isinstance(payload[field], int):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or set(protected) != {
        ROADMAP_RELATIVE_PATH.as_posix(),
        CONDUCTOR_RELATIVE_PATH.as_posix(),
    }:
        errors.append("protected_files_unchanged")
    model_audit = payload.get("llm_model_spec_audit")
    if isinstance(model_audit, Mapping) and not model_audit.get("all_mandated_specs_present"):
        errors.append("llm_model_spec_audit")
    if payload.get("hardware_speedup_claim") is not False:
        errors.append("hardware_speedup_claim")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("task_range") != TASK_RANGE:
        errors.append("task_range")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    checks_run: Sequence[str] = DEFAULT_CHECKS_RUN,
    docs_updated: Sequence[str] = DEFAULT_DOCS_UPDATED,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root,
        checks_run=checks_run,
        docs_updated=docs_updated,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - unit tests exercise validate_artifact directly
        raise ValueError(f"invalid Exp5549 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5549 artifact")
    args = parser.parse_args(argv)
    artifact = write_capstone() if args.write else run_capstone()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
