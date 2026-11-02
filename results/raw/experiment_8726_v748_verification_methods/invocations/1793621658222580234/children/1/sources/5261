"""Build the Exp 3110 SOTA model-spec/cache manifest corrigendum.

Spec refs: REQ-REPORT-3110, SCENARIO-REPORT-3110.

This module turns the already checked-in `.289` SOTA/cache evidence into a
single machine-readable authority. It deliberately does not probe the live
Hugging Face cache or run inference, because the corrigendum is about
documenting the cache status that downstream artifacts already relied on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.290"
SCHEMA = "carnot.sota_model_spec_cache_manifest_corrigendum.v1"
ARTIFACT = "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1"
OUTPUT_REL_PATH = Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json")
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.py"
)

EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3100_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
MATRIX_V23_REL_PATH = Path("results/experiment_3107_cross_corpus_matrix_v23.json")
CAPSTONE_V289_REL_PATH = Path("results/experiment_3108_capstone_v289.json")
EXPERIMENT_TEMPLATE_REL_PATH = Path("scripts/experiment_template.py")

QWEN_MOE_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_DENSE_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA_MIDDLE_MOE_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATORY_HEADLINE_MODEL_IDS = (QWEN_MOE_ID, GEMMA_DENSE_ID, GEMMA_MIDDLE_MOE_ID)
LEGACY_SMOKE_TEST_MODEL_IDS = ("google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B")
SOURCE_PATHS = (
    ("exp3099_local_sota_confidence_abstention_panel", EXP3099_REL_PATH),
    ("exp3100_z3_oracle_feedback", EXP3100_REL_PATH),
    ("exp3107_cross_corpus_matrix_v23", MATRIX_V23_REL_PATH),
    ("exp3108_capstone_v289", CAPSTONE_V289_REL_PATH),
    ("experiment_template_cached_sota_pair_policy", EXPERIMENT_TEMPLATE_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a checksum for source provenance when the file exists."""

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
    """REQ-REPORT-3110: build the cache manifest from checked-in authorities only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3099 = read_json_object(root_path / EXP3099_REL_PATH)
    exp3100 = read_json_object(root_path / EXP3100_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V23_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V289_REL_PATH)
    source_artifacts = [
        _source_artifact(root_path, role, rel_path) for role, rel_path in SOURCE_PATHS
    ]
    missing_sources = [row for row in source_artifacts if row["present"] is not True]

    cache_rows = _cache_rows(exp3099, exp3100)
    present_model_ids = _present_model_ids(cache_rows)
    missing_model_ids = [
        model_id for model_id in MANDATORY_HEADLINE_MODEL_IDS if model_id not in present_model_ids
    ]
    cached_sota_pair_available = _cached_sota_pair_available(exp3099, exp3100)
    selected_headline_model_ids = _selected_headline_model_ids(exp3099, present_model_ids)
    headline_claim_allowed = bool(selected_headline_model_ids)
    ready = not missing_sources

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "sota_model_manifest_ready": ready,
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "present_model_ids": present_model_ids,
        "missing_model_ids": missing_model_ids,
        "cached_sota_pair_available": cached_sota_pair_available,
        "selected_headline_model_ids": selected_headline_model_ids,
        "smoke_test_model_ids": list(LEGACY_SMOKE_TEST_MODEL_IDS),
        "headline_claim_allowed": headline_claim_allowed,
        "downstream_usage": _downstream_usage(
            selected_headline_model_ids,
            cached_sota_pair_available,
            headline_claim_allowed,
        ),
        "cache_evidence": _cache_evidence(cache_rows),
        "matrix_reported_model_spec_gaps": _headline_model_spec_gaps(matrix, capstone),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "missing_source_artifacts": missing_sources,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3110 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifact(root: Path, role: str, rel_path: Path) -> JsonDict:
    path = root / rel_path
    return {
        "path": rel_path.as_posix(),
        "role": role,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def _cache_rows(exp3099: Mapping[str, Any], exp3100: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    rows.extend(_rows_from_model_specs("exp3099.model_specs", exp3099.get("model_specs")))
    rows.extend(_rows_from_model_specs("exp3100.model_specs", exp3100.get("model_specs")))
    substrate = exp3100.get("inference_substrate")
    if isinstance(substrate, Mapping):
        rows.extend(
            _rows_from_model_specs(
                "exp3100.inference_substrate.model_cache_status",
                substrate.get("model_cache_status"),
            )
        )
    return rows


def _rows_from_model_specs(source_field: str, value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    rows: list[JsonDict] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        hf_id = str(item.get("hf_id") or "")
        if not hf_id:
            continue
        rows.append(
            {
                "hf_id": hf_id,
                "cached": _is_cached(item),
                "selected": item.get("selected") is True,
                "model_path": item.get("model_path"),
                "cache_status": item.get("cache_status"),
                "source_field": source_field,
            }
        )
    return rows


def _is_cached(row: Mapping[str, Any]) -> bool:
    return (
        row.get("cached") is True
        or row.get("cache_present") is True
        or str(row.get("cache_status") or "").lower() == "cached"
    )


def _present_model_ids(cache_rows: list[Mapping[str, Any]]) -> list[str]:
    present = {
        str(row.get("hf_id"))
        for row in cache_rows
        if row.get("cached") is True and row.get("hf_id") in MANDATORY_HEADLINE_MODEL_IDS
    }
    return [model_id for model_id in MANDATORY_HEADLINE_MODEL_IDS if model_id in present]


def _selected_headline_model_ids(
    exp3099: Mapping[str, Any],
    present_model_ids: list[str],
) -> list[str]:
    present = set(present_model_ids)
    selected = {
        str(row.get("hf_id"))
        for row in exp3099.get("model_specs", [])
        if isinstance(row, Mapping) and row.get("selected") is True
    }
    selected.update(
        str(model_id) for model_id in exp3099.get("models_used", []) if isinstance(model_id, str)
    )
    return [
        model_id
        for model_id in MANDATORY_HEADLINE_MODEL_IDS
        if model_id in selected and model_id in present
    ]


def _cached_sota_pair_available(exp3099: Mapping[str, Any], exp3100: Mapping[str, Any]) -> bool:
    pair = exp3099.get("cached_sota_pair")
    status = exp3100.get("cached_sota_pair_status")
    exp3100_substrate = exp3100.get("inference_substrate")
    return (
        isinstance(pair, Mapping)
        and pair.get("ready") is True
        or isinstance(status, Mapping)
        and status.get("available") is True
        or isinstance(exp3100_substrate, Mapping)
        and exp3100_substrate.get("cached_sota_pair_available") is True
    )


def _cache_evidence(cache_rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    evidence: list[JsonDict] = []
    for model_id in MANDATORY_HEADLINE_MODEL_IDS:
        model_rows = [row for row in cache_rows if row.get("hf_id") == model_id]
        evidence.append(
            {
                "hf_id": model_id,
                "cached": any(row.get("cached") is True for row in model_rows),
                "selected": any(row.get("selected") is True for row in model_rows),
                "source_fields": sorted({str(row.get("source_field")) for row in model_rows}),
            }
        )
    return evidence


def _headline_model_spec_gaps(
    matrix: Mapping[str, Any], capstone: Mapping[str, Any]
) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    for payload in (matrix, capstone):
        for gap in payload.get("headline_model_spec_gaps", []):
            if isinstance(gap, Mapping):
                gap_row = dict(gap)
                if gap_row not in gaps:
                    gaps.append(gap_row)
    return gaps


def _downstream_usage(
    selected_headline_model_ids: list[str],
    cached_sota_pair_available: bool,
    headline_claim_allowed: bool,
) -> JsonDict:
    return {
        "solver_only_tasks": {
            "allowed_without_cached_sota_pair": True,
            "headline_claim_allowed_from_solver_only": False,
            "required_action": (
                "Proceed with exact solver/test-oracle evidence, cite this manifest, "
                "and do not report live-LLM headline evidence from solver-only runs."
            ),
        },
        "live_llm_headline_tasks": {
            "requires_mandated_cached_model": True,
            "minimum_selected_mandated_cached_models": 1,
            "allowed_model_ids": selected_headline_model_ids,
            "headline_claim_allowed": headline_claim_allowed,
            "when_no_selected_mandated_cached_model": "report headline_claim_allowed=false",
        },
        "pair_or_comparative_headline_tasks": {
            "requires_cached_sota_pair_available": True,
            "cached_sota_pair_available": cached_sota_pair_available,
            "headline_claim_allowed": cached_sota_pair_available
            and len(selected_headline_model_ids) >= 2,
        },
        "legacy_small_models": {
            "model_ids": list(LEGACY_SMOKE_TEST_MODEL_IDS),
            "allowed_only_for_cpu_smoke_tests": True,
            "headline_claim_allowed": False,
        },
    }


def _inference_substrate() -> JsonDict:
    return {
        "kind": "corrigendum_from_checked_in_artifacts",
        "source": "exp3099_exp3100_matrix_v23_capstone_v289",
        "executes_models": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "cache_probe_performed": False,
        "local_repo_only": True,
    }


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("sota_model_manifest_ready") is not True:
        return (
            "blocked_sota_model_manifest_preconditions: missing_source_artifacts="
            f"{len(artifact.get('missing_source_artifacts', []))}"
        )
    return (
        "complete: sota_model_manifest_ready=true; "
        f"cached_sota_pair_available={artifact['cached_sota_pair_available']}; "
        f"selected_headline_model_ids={len(artifact['selected_headline_model_ids'])}; "
        f"headline_claim_allowed={artifact['headline_claim_allowed']}; "
        f"missing_model_ids={len(artifact['missing_model_ids'])}"
    )
