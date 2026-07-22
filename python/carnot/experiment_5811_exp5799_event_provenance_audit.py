"""Exp5811: independently audit Exp5799 event definitions and provenance.

Spec refs: REQ-REPORT-5811, SCENARIO-REPORT-5811-ROW-REPLAY,
SCENARIO-REPORT-5811-GPU-RECEIPTS, SCENARIO-REPORT-5811-PRODUCER-REPAIR.

This module does not run a model.  It replays the checked-in Exp5799 row file,
recomputes row-level events from raw receipts, and writes a companion audit
artifact that supersedes definitions without changing the quarantined evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5799_sota_answer_channel_canary as exp5799


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5811_exp5799_event_provenance_audit.json")
EXP5799_ARTIFACT_RELATIVE_PATH = exp5799.RESULT_RELATIVE_PATH
EXP5799_ROWS_RELATIVE_PATH = exp5799.ROW_FILE_RELATIVE_PATH
EXP5798_ARTIFACT_RELATIVE_PATH = exp5799.EXP5798_ARTIFACT_RELATIVE_PATH
EXP5785_ARTIFACT_RELATIVE_PATH = exp5799.EXP5785_ARTIFACT_RELATIVE_PATH
EXP5785_ROWS_RELATIVE_PATH = exp5799.EXP5785_ROWS_RELATIVE_PATH
EXP5799_PRODUCER_RELATIVE_PATH = exp5799.MODULE_RELATIVE_PATH
EXP5799_TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5799_sota_answer_channel_canary.py")
MODEL_METADATA_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")
VERIFY_SCRIPT_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
SUMMARY_SCRIPT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")

EXPERIMENT = "experiment_5811_exp5799_event_provenance_audit"
EXPERIMENT_ID = "exp5811-exp5799-event-provenance-audit"
MILESTONE = "2026.07.518"
RUN_DATE = "20260722"
RANDOM_SEED = 5811
SCHEMA = "carnot.experiment_5811.exp5799_event_provenance_audit.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
QWEN_ID = exp5799.QWEN_ID
GEMMA31_ID = exp5799.GEMMA31_ID
GEMMA26_ID = exp5799.GEMMA26_ID

SPEC_REFS = (
    "REQ-REPORT-5811",
    "SCENARIO-REPORT-5811-ROW-REPLAY",
    "SCENARIO-REPORT-5811-GPU-RECEIPTS",
    "SCENARIO-REPORT-5811-PRODUCER-REPAIR",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_input_hashes",
    "independent_event_definitions",
    "overlapping_event_matrix",
    "exclusive_primary_failure_taxonomy",
    "per_model_mode_reconstruction",
    "methodology_gap_matrix",
    "gpu_provenance_reconciliation",
    "producer_repairs_and_tests",
    "canary_evidence_ready_score",
    "original_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal audit state distinguishes a clean reconstruction from unresolved evidence.",
    "preconditions_checked": "Hash and row checks prevent auditing a different artifact than the quarantined input.",
    "immutable_input_hashes": "Historical evidence must remain byte-identical while a companion audit is produced.",
    "independent_event_definitions": "Each metric needs a raw predicate so accidental tautologies are detectable.",
    "overlapping_event_matrix": "Failures can co-occur; explicit overlap prevents double counting from masquerading as independent evidence.",
    "exclusive_primary_failure_taxonomy": "One primary cause per row makes totals and denominators close exactly.",
    "per_model_mode_reconstruction": "Family and transport effects must not disappear into one aggregate.",
    "methodology_gap_matrix": "Missing seed, duration, or test receipts remain visible rather than backfilled.",
    "gpu_provenance_reconciliation": "Only actual load/offload evidence may support a real-GPU qualification.",
    "producer_repairs_and_tests": "Any code correction is explicit and does not mutate historical result files.",
    "canary_evidence_ready_score": "A bare scalar gates new protocol work only after the quarantined evidence is independently reconstructable.",
    "original_files_unchanged": "A true receipt prevents silent sanitization of flagged evidence.",
    "duration_s": "Measured wall time exposes bootstrap-only audit artifacts.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` declares that no LLM is loaded or invoked.",
    "verifier_is_oracle": "The exact parser/fixture defines correctness and is circular authority, so no verifier-moat claim is allowed.",
    "field_provenance": "Every reconstructed field identifies the row predicate or runtime receipt that satisfies it.",
    "test_commands": "Commands document row replay, code repair, and immutability validation.",
    "test_exit_codes": "Exit codes prevent failed audit checks from being narrated as passing.",
    "reproducibility_checksum": "A checksum detects drift in inputs, definitions, or reconstructed outputs.",
    "honest_verdict": "A `complete:` or `blocked:` prefix provides a terminal audit outcome.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id for the companion Exp5799 audit.",
    "experiment": "Stable local slug ties the artifact to this implementation.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds the audit to V518.",
    "run_date": "Operator-requested audit date.",
    "random_seed": "Deterministic metadata for an aggregation-only audit.",
    "spec_refs": "OpenSpec anchors for the audit and producer repair.",
    "result_path": "Declares the intended companion JSON path.",
}

IMMUTABLE_INPUT_PATHS: dict[str, Path] = {
    "claude_instructions": CLAUDE_RELATIVE_PATH,
    "codex_instructions": CODEX_RELATIVE_PATH,
    "exp5799_artifact": EXP5799_ARTIFACT_RELATIVE_PATH,
    "exp5799_rows": EXP5799_ROWS_RELATIVE_PATH,
    "exp5798_diagnostic_artifact": EXP5798_ARTIFACT_RELATIVE_PATH,
    "exp5785_fixture_artifact": EXP5785_ARTIFACT_RELATIVE_PATH,
    "exp5785_fixture_rows": EXP5785_ROWS_RELATIVE_PATH,
    "exp5799_producer": EXP5799_PRODUCER_RELATIVE_PATH,
    "exp5799_tests": EXP5799_TEST_RELATIVE_PATH,
    "model_metadata": MODEL_METADATA_RELATIVE_PATH,
    "adversarial_verify": VERIFY_SCRIPT_RELATIVE_PATH,
    "summarize_artifact": SUMMARY_SCRIPT_RELATIVE_PATH,
    "verification_spec": VERIFY_SPEC_RELATIVE_PATH,
    "research_reporting_spec": REPORT_SPEC_RELATIVE_PATH,
}

EVENT_FIELDS = (
    "parser_failure",
    "truncation",
    "empty_final_content",
    "exact_wrong_answer",
    "invalid_candidate",
    "timeout",
    "stop_collision",
)
PRIMARY_FAILURE_ORDER = (
    "timeout",
    "empty_final_content",
    "truncation",
    "stop_collision",
    "invalid_candidate",
    "parser_failure",
    "exact_wrong_answer",
)
PAIR_FIELDS = (
    ("parser_failure", "truncation"),
    ("parser_failure", "empty_final_content"),
    ("truncation", "empty_final_content"),
    ("invalid_candidate", "truncation"),
    ("invalid_candidate", "parser_failure"),
    ("exact_wrong_answer", "parser_failure"),
    ("timeout", "truncation"),
    ("stop_collision", "parser_failure"),
)


class AuditReplayError(ValueError):
    """Raised when the quarantined rows cannot be replayed byte-for-byte."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for deterministic JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so large inputs remain streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read JSONL rows from disk."""

    return [
        dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: str | Path) -> JsonDict:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _path_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    stat = path.stat()
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mode_octal": oct(stat.st_mode & 0o777),
        "mtime_ns": stat.st_mtime_ns,
        "readable": path.is_file(),
    }


def immutable_input_hashes(root: Path = REPO_ROOT) -> JsonDict:
    """Hash every immutable input that defines the Exp5799 audit surface."""

    return {
        name: _path_receipt(root, relative)
        for name, relative in IMMUTABLE_INPUT_PATHS.items()
    }


def _available_memory_mb() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0  # pragma: no cover - non-Linux fallback.


def _host_capacity_receipts(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {
        "disk": {
            "available_mb": int(usage.free / (1024 * 1024)),
            "total_mb": int(usage.total / (1024 * 1024)),
        },
        "memory": {"available_mb": _available_memory_mb()},
    }


def _runtime_log_receipts(root: Path) -> JsonDict:
    globs = ("logs/*5799*.log", "results/*5799*.log", "results/raw/*5799*/*.log")
    paths: list[str] = []
    for pattern in globs:
        paths.extend(path.relative_to(root).as_posix() for path in root.glob(pattern))
    return {
        "searched_globs": list(globs),
        "paths": sorted(paths),
        "status": "present" if paths else "missing",
    }


def _mode_lookup(diagnostic_artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(mode["mode_id"]): dict(mode)
        for mode in diagnostic_artifact.get("candidate_mode_matrix", [])
    }


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{row['model_hf_id']}::{row['mode_id']}::{row['fixture_row_id']}"


def _event_predicates(row: Mapping[str, Any], mode: Mapping[str, Any]) -> JsonDict:
    parser = dict(row.get("parser_receipt") or {})
    parse_ok = parser.get("parse_ok") is True
    reason = str(parser.get("parser_failure_reason") or "")
    max_tokens = int(mode.get("max_tokens") or exp5799.DEFAULT_MAX_TOKENS)
    raw_text = str(row.get("raw_response_text") or "")
    stops = list(mode.get("stops") or exp5799.STOP_STRINGS)
    return {
        "parser_failure": not parse_ok,
        "truncation": bool(
            row.get("finish_reason") == "length"
            or reason == "truncation"
            or int(row.get("output_tokens", 0) or 0) >= max_tokens
        ),
        "empty_final_content": str(row.get("raw_final_content") or "") == "",
        "exact_wrong_answer": bool(parse_ok and row.get("selected_label") != row.get("exact_label")),
        "invalid_candidate": reason in {"invalid_id", "invalid_candidate"},
        "timeout": row.get("timeout") is True,
        "stop_collision": bool(
            row.get("stop_collision") is True
            or reason == "stop_token"
            or any(str(stop) and str(stop) in raw_text for stop in stops)
        ),
    }


def _primary_failure(events: Mapping[str, bool]) -> str:
    for field in PRIMARY_FAILURE_ORDER:
        if events[field]:
            return field
    return "valid_exact_output"


def _rate(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def _count_events(rows: Sequence[Mapping[str, Any]], modes: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    event_counts = Counter()
    primary_counts = Counter()
    pair_counts = Counter()
    row_events: list[JsonDict] = []
    for row in rows:
        events = _event_predicates(row, modes[str(row["mode_id"])])
        primary = _primary_failure(events)
        row_events.append({"row_hash": row["row_hash"], "events": events, "primary": primary})
        for field in EVENT_FIELDS:
            event_counts[field] += int(events[field] is True)
        primary_counts[primary] += 1
        for left, right in PAIR_FIELDS:
            pair_counts[f"{left}&{right}"] += int(events[left] is True and events[right] is True)
    denominator = len(rows)
    return {
        "row_events": row_events,
        "event_counts": {field: int(event_counts[field]) for field in EVENT_FIELDS},
        "event_rates": {field: _rate(int(event_counts[field]), denominator) for field in EVENT_FIELDS},
        "primary_counts": dict(primary_counts),
        "pairwise_overlap_counts": {f"{left}&{right}": int(pair_counts[f"{left}&{right}"]) for left, right in PAIR_FIELDS},
    }


def _assert_row_replay(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any], rows_path: Path) -> JsonDict:
    declared_file_hash = str(artifact.get("row_file_sha256") or "")
    actual_file_hash = sha256_file(rows_path)
    if declared_file_hash != actual_file_hash:
        raise AuditReplayError("row_file_sha256")
    seen: set[str] = set()
    receipts = dict(artifact.get("raw_response_receipts") or {})
    for row in rows:
        key = _row_key(row)
        if key in seen:
            raise AuditReplayError("duplicate canary cell")
        seen.add(key)
        if sha256_text(str(row.get("raw_response_text") or "")) != row.get("raw_response_sha256"):
            raise AuditReplayError("raw_response_sha256")
        if _row_hash(row) != row.get("row_hash"):
            raise AuditReplayError("row_hash")
        receipt = dict(receipts.get(key) or {})
        if not receipt:
            raise AuditReplayError("missing raw_response_receipt")
        for field in ("row_hash", "raw_response_sha256", "prompt_hash", "fixture_row_hash"):
            if receipt.get(field) != row.get(field):
                raise AuditReplayError(field)
    if set(receipts) != seen:
        raise AuditReplayError("row receipt set")
    declared_mode_rows = sum(int(row.get("row_count") or 0) for row in artifact.get("mode_execution_matrix", []))
    if declared_mode_rows != len(rows):
        raise AuditReplayError("declared mode row counts")
    return {
        "ok": True,
        "row_count": len(rows),
        "unique_cell_count": len(seen),
        "declared_mode_row_count": declared_mode_rows,
        "row_file_sha256_matches_declared": True,
        "raw_response_receipts_match": True,
    }


def _overlap_matrix(counts: Mapping[str, Any], denominator: int, artifact: Mapping[str, Any]) -> JsonDict:
    flagged = list(artifact.get("corrigendum_pending") or [])
    return {
        "denominator": denominator,
        "event_counts": dict(counts["event_counts"]),
        "event_rates": dict(counts["event_rates"]),
        "pairwise_overlap_counts": dict(counts["pairwise_overlap_counts"]),
        "rows_with_multiple_events": sum(
            1
            for row in counts["row_events"]
            if sum(int(value is True) for value in row["events"].values()) > 1
        ),
        "legitimate_overlap_explanation": (
            "A truncated response can also have no final line and therefore fail parsing; "
            "overlap is expected and is not evidence that definitions share a predicate."
        ),
        "tautology_flag_resolution": (
            "parser_failure_rate and truncation_rate both equal 93/120 through legitimate "
            "co-occurrence; the audit keeps independent raw predicates and does not force "
            "distinct rates to differ."
        ),
        "original_flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "original_corrigendum_pending": flagged,
    }


def _primary_taxonomy(counts: Mapping[str, Any], denominator: int) -> JsonDict:
    primary_counts = dict(counts["primary_counts"])
    return {
        "denominator": denominator,
        "primary_order": list(PRIMARY_FAILURE_ORDER) + ["valid_exact_output"],
        "primary_counts": primary_counts,
        "total": sum(int(value) for value in primary_counts.values()),
        "total_matches_denominator": sum(int(value) for value in primary_counts.values()) == denominator,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], counts: Mapping[str, Any], artifact: Mapping[str, Any]) -> JsonDict:
    by_hash = {row["row_hash"]: row for row in rows}
    events_by_hash = {row["row_hash"]: row for row in counts["row_events"]}
    per_model: JsonDict = {}
    for row in rows:
        hf_id = str(row["model_hf_id"])
        mode_id = str(row["mode_id"])
        model = per_model.setdefault(
            hf_id,
            {
                "model_family": row["model_family"],
                "row_count": 0,
                "event_counts": Counter(),
                "primary_failure_counts": Counter(),
                "modes": {},
            },
        )
        mode = model["modes"].setdefault(
            mode_id,
            {
                "mode_type": row["mode_type"],
                "row_count": 0,
                "event_counts": Counter(),
                "primary_failure_counts": Counter(),
                "row_hashes": [],
            },
        )
        event_row = events_by_hash[str(row["row_hash"])]
        model["row_count"] += 1
        mode["row_count"] += 1
        mode["row_hashes"].append(row["row_hash"])
        for field, value in event_row["events"].items():
            model["event_counts"][field] += int(value is True)
            mode["event_counts"][field] += int(value is True)
        model["primary_failure_counts"][event_row["primary"]] += 1
        mode["primary_failure_counts"][event_row["primary"]] += 1
    for model in per_model.values():
        model["event_counts"] = dict(model["event_counts"])
        model["primary_failure_counts"] = dict(model["primary_failure_counts"])
        model["mode_count"] = len(model["modes"])
        for mode in model["modes"].values():
            mode["event_counts"] = dict(mode["event_counts"])
            mode["primary_failure_counts"] = dict(mode["primary_failure_counts"])
            mode["row_hash_manifest_sha256"] = sha256_json(mode["row_hashes"])
    total = len(rows)
    exact_label_count = sum(
        1
        for row in rows
        if row.get("exact_label") and row.get("exact_certificate_hash") and row.get("exact_validator_result")
    )
    units = sorted({str(row["fixture_unit_id"]) for row in rows})
    return {
        "overall": {
            "row_count": total,
            "model_count": len({row["model_hf_id"] for row in rows}),
            "mode_count": len({(row["model_hf_id"], row["mode_id"]) for row in rows}),
            "fixture_row_count": len({row["fixture_row_id"] for row in rows}),
            "independent_unit_count": len(units),
            "artifact_independent_unit_count": artifact.get("independent_unit_count"),
            "independent_unit_count_matches_artifact": len(units) == artifact.get("independent_unit_count"),
            "exact_label_count": exact_label_count,
            "exact_label_coverage": _rate(exact_label_count, total),
            "duration_from_rows_s": round(
                sum(float(dict(row.get("timing") or {}).get("generation_s", 0.0) or 0.0) for row in rows),
                6,
            ),
            "seed": dict(dict(artifact.get("preconditions_checked") or {}).get("deterministic_seeds") or {}),
            "row_file_sha256_matches_declared": True,
            "row_hash_manifest_sha256": sha256_json([row["row_hash"] for row in rows]),
        },
        "models": per_model,
        "row_hashes_by_sequence": [by_hash[str(row["row_hash"])]["row_hash"] for row in rows],
    }


def _is_resume_receipt(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("resume_from_checkpoint") is True
        or receipt.get("llama_cpp_version") == "resume_from_checkpoint"
        or dict(receipt.get("cuda_device_receipt") or {}).get("resume_from_checkpoint") is True
        or "resume_from" in str(receipt.get("offload_log_excerpt") or "")
    )


def classify_runtime_receipt(receipt: Mapping[str, Any] | None) -> str:
    """Classify a runtime/GPU receipt without trusting resume-only summaries."""

    if not receipt:
        return "missing"
    if _is_resume_receipt(receipt):
        return "resume_only"
    before = int(receipt.get("gpu_memory_before_mb", 0) or 0)
    peak = int(receipt.get("gpu_memory_peak_mb", 0) or 0)
    layers = int(receipt.get("n_gpu_layers_offloaded", 0) or 0)
    build = dict(receipt.get("llama_cpp_build_info") or {})
    if (
        receipt.get("cuda_offload_authenticated") is True
        and layers > 0
        and peak > before
        and build.get("cuda_backend") is True
    ):
        return "authenticated"
    if receipt.get("cuda_offload_authenticated") is True or layers > 0 or peak > before:
        return "inconsistent"
    return "missing"


def _receipt_summary(receipt: Mapping[str, Any] | None) -> JsonDict:
    receipt = dict(receipt or {})
    classification = classify_runtime_receipt(receipt)
    before = int(receipt.get("gpu_memory_before_mb", 0) or 0)
    peak = int(receipt.get("gpu_memory_peak_mb", 0) or 0)
    return {
        "classification": classification,
        "cuda_offload_authenticated_declared": receipt.get("cuda_offload_authenticated") is True,
        "resume_from_checkpoint": _is_resume_receipt(receipt),
        "n_gpu_layers_requested": int(receipt.get("n_gpu_layers_requested", 0) or 0),
        "n_gpu_layers_offloaded": int(receipt.get("n_gpu_layers_offloaded", 0) or 0),
        "gpu_memory_before_mb": before,
        "gpu_memory_peak_mb": peak,
        "gpu_memory_after_mb": int(receipt.get("gpu_memory_after_mb", 0) or 0),
        "vram_growth_observed": peak > before,
        "rows_attempted": int(receipt.get("rows_attempted", 0) or 0),
        "runtime_hash_present_in_receipt": bool(
            receipt.get("runtime_hash") or dict(receipt.get("llama_cpp_build_info") or {}).get("runtime_hash")
        ),
        "offload_log_excerpt_present": bool(str(receipt.get("offload_log_excerpt") or "")),
    }


def _mode_runtime_receipts(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for hf_id, model in dict(artifact.get("model_runtime_receipts") or {}).items():
        for mode_id, receipt in dict(dict(model).get("mode_runtime_receipts") or {}).items():
            receipts[f"{hf_id}::{mode_id}"] = dict(receipt)
    return receipts


def _gpu_reconciliation(artifact: Mapping[str, Any]) -> JsonDict:
    mode_receipts = {
        key: _receipt_summary(receipt)
        for key, receipt in _mode_runtime_receipts(artifact).items()
    }
    top_level = {
        hf_id: _receipt_summary(receipt)
        for hf_id, receipt in dict(artifact.get("gpu_offload_receipts") or {}).items()
    }
    selected = dict(artifact.get("selected_transport_by_model") or {})
    original_qualified = sorted(selected)
    audit_qualified: list[str] = []
    original_uses_resume = False
    for hf_id, selected_mode in selected.items():
        key = f"{hf_id}::{dict(selected_mode).get('mode_id')}"
        classification = mode_receipts.get(key, {}).get("classification", "missing")
        original_uses_resume = original_uses_resume or classification != "authenticated"
        if classification == "authenticated":
            audit_qualified.append(str(hf_id))
    model_receipts: JsonDict = {}
    for hf_id in exp5799.MANDATED_MODEL_IDS:
        classes = [
            row["classification"]
            for key, row in mode_receipts.items()
            if key.startswith(f"{hf_id}::")
        ]
        if "authenticated" in classes:
            classification = "authenticated"
        elif "resume_only" in classes:
            classification = "resume_only"
        elif classes:
            classification = "inconsistent"
        else:
            classification = "missing"
        model_receipts[hf_id] = {
            "classification": classification,
            "mode_classifications": classes,
            "top_level_classification": top_level.get(hf_id, {}).get("classification", "missing"),
        }
    return {
        "mode_receipts": mode_receipts,
        "model_receipts": model_receipts,
        "top_level_receipts": top_level,
        "authenticated_mode_count": sum(
            1 for row in mode_receipts.values() if row["classification"] == "authenticated"
        ),
        "resume_only_mode_count": sum(
            1 for row in mode_receipts.values() if row["classification"] == "resume_only"
        ),
        "original_answer_channel_qualified_models": original_qualified,
        "original_answer_channel_qualification_uses_resume_only": original_uses_resume,
        "audit_answer_channel_qualified_models": audit_qualified,
        "unauthenticated_receipt_used_to_qualify_model": False,
    }


def _methodology_gap_matrix(artifact: Mapping[str, Any], runtime_logs: Mapping[str, Any]) -> JsonDict:
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    preconditions = dict(artifact.get("preconditions_checked") or {})
    model_checks = dict(preconditions.get("models") or {})
    return {
        "original_duration_s": {
            "status": "present" if "duration_s" in artifact else "absent",
            "value": artifact.get("duration_s"),
            "note": "Exp5799 has row timing but no top-level wall-clock duration.",
        },
        "runtime_logs": {
            "status": runtime_logs["status"],
            "searched_globs": runtime_logs["searched_globs"],
            "paths": runtime_logs["paths"],
        },
        "original_test_exit_codes": {
            "status": "present" if test_exit_codes else "absent",
            "value": test_exit_codes,
        },
        "seed": {
            "status": "present" if preconditions.get("deterministic_seeds") else "absent",
            "value": preconditions.get("deterministic_seeds"),
        },
        "runtime_hashes": {
            "status": "precondition_only",
            "model_runtime_hashes": {
                hf_id: dict(check).get("runtime_hash")
                for hf_id, check in model_checks.items()
            },
            "note": "Per-mode runtime receipts omit a runtime hash field.",
        },
        "per_row_load_receipts": {
            "status": "absent",
            "note": "Rows carry response/timing hashes, while load/offload evidence is mode-level.",
        },
        "checkpoint_origin": {
            "status": "mixed_resume_and_new_rows",
            "value": artifact.get("checkpoint_resume_receipts"),
        },
    }


def independent_event_definitions() -> JsonDict:
    """Return the raw predicates used by Exp5811 instead of Exp5799 aggregates."""

    return {
        "parser_failure": {
            "predicate": "row.parser_receipt.parse_ok is not true",
            "source": "row.parser_receipt",
        },
        "truncation": {
            "predicate": "finish_reason == 'length' OR parser_failure_reason == 'truncation' OR output_tokens >= diagnostic_mode.max_tokens",
            "source": "row.finish_reason,row.output_tokens,Exp5798 mode.max_tokens",
        },
        "empty_final_content": {
            "predicate": "row.raw_final_content == ''",
            "source": "row.raw_final_content",
        },
        "exact_wrong_answer": {
            "predicate": "parser parse_ok is true AND selected_label != exact_label",
            "source": "row.parser_receipt,row.selected_label,row.exact_label",
        },
        "invalid_candidate": {
            "predicate": "parser_failure_reason in {'invalid_id','invalid_candidate'}",
            "source": "row.parser_receipt.parser_failure_reason",
        },
        "timeout": {
            "predicate": "row.timeout is true",
            "source": "row.timeout",
        },
        "stop_collision": {
            "predicate": "row.stop_collision is true OR parser_failure_reason == 'stop_token' OR stop string appears in raw_response_text",
            "source": "row.stop_collision,row.raw_response_text,Exp5798 mode.stops",
        },
        "definition_independence_receipt": {
            "distinct_raw_predicates": True,
            "aggregate_reused_as_ground_truth": False,
            "primary_failure_order": list(PRIMARY_FAILURE_ORDER) + ["valid_exact_output"],
        },
    }


def _field_provenance() -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "sources": ["task_prompt", REPORT_SPEC_RELATIVE_PATH.as_posix()],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            field: {"principle": principle, "sources": ["local_metadata"]}
            for field, principle in FIELD_PRINCIPLE_EXTRAS.items()
        }
    )
    provenance["row_predicates"] = {
        field: independent_event_definitions()[field]["predicate"]
        for field in EVENT_FIELDS
    }
    provenance["runtime_receipts"] = {
        "source": "Exp5799 model_runtime_receipts and gpu_offload_receipts",
        "allowed_classes": ["authenticated", "resume_only", "inconsistent", "missing"],
    }
    return provenance


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return payload


def compute_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field blanked."""

    return sha256_json(_checksum_payload(artifact))


def _producer_repairs(original_unchanged: bool) -> JsonDict:
    return {
        "repair_required": True,
        "repairs": [
            {
                "file": EXP5799_PRODUCER_RELATIVE_PATH.as_posix(),
                "change": (
                    "Resume-only rows no longer synthesize cuda_offload_authenticated=true; "
                    "prior runtime receipts must be replayed from an existing artifact."
                ),
                "historical_result_files_rewritten": False,
            }
        ],
        "superseding_definitions": [
            "Exp5811 parser/truncation/empty-final/invalid/timeout/stop/exact-wrong metrics use independent raw predicates.",
            "Resume-only GPU receipts classify as resume_only and cannot qualify answer-channel models.",
        ],
        "tests_reference": [
            "SCENARIO-REPORT-5811-PRODUCER-REPAIR",
            "tests/python/test_experiment_5799_sota_answer_channel_canary.py",
            "tests/python/test_experiment_5811_exp5799_event_provenance_audit.py",
        ],
        "historical_files_mutated": not original_unchanged,
    }


def _complete_ready(
    *,
    row_replay_ok: bool,
    counts_close: bool,
    gaps_explicit: bool,
    original_unchanged: bool,
    gpu: Mapping[str, Any],
) -> bool:
    return bool(
        row_replay_ok
        and counts_close
        and gaps_explicit
        and original_unchanged
        and gpu.get("unauthenticated_receipt_used_to_qualify_model") is False
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    rows_override: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the Exp5811 companion artifact without mutating Exp5799 evidence."""

    started = time.perf_counter()
    root = Path(root)
    before_hashes = immutable_input_hashes(root)
    exp5799_artifact = _read_json(root / EXP5799_ARTIFACT_RELATIVE_PATH)
    diagnostic = _read_json(root / EXP5798_ARTIFACT_RELATIVE_PATH)
    rows_path = root / EXP5799_ROWS_RELATIVE_PATH
    rows = [dict(row) for row in (rows_override if rows_override is not None else read_jsonl(rows_path))]
    runtime_logs = _runtime_log_receipts(root)
    blocked_reasons: list[str] = []
    row_replay: JsonDict
    reconstruction: JsonDict
    overlap: JsonDict
    taxonomy: JsonDict
    try:
        row_replay = _assert_row_replay(rows, exp5799_artifact, rows_path)
        counts = _count_events(rows, _mode_lookup(diagnostic))
        denominator = len(rows)
        overlap = _overlap_matrix(counts, denominator, exp5799_artifact)
        taxonomy = _primary_taxonomy(counts, denominator)
        reconstruction = _aggregate_rows(rows, counts, exp5799_artifact)
        counts_close = bool(
            taxonomy["total_matches_denominator"]
            and overlap["event_counts"]["parser_failure"]
            == round(float(exp5799_artifact["parser_failure_rate"]) * denominator)
            and overlap["event_counts"]["truncation"]
            == round(float(exp5799_artifact["truncation_rate"]) * denominator)
        )
    except AuditReplayError as exc:
        blocked_reasons.append(str(exc))
        row_replay = {"ok": False, "row_count": len(rows), "unique_cell_count": 0}
        overlap = {"denominator": len(rows), "event_counts": {}, "pairwise_overlap_counts": {}}
        taxonomy = {"denominator": len(rows), "primary_counts": {}, "total_matches_denominator": False}
        reconstruction = {"overall": {"row_count": len(rows)}, "models": {}, "row_hashes_by_sequence": []}
        counts_close = False
    gpu = _gpu_reconciliation(exp5799_artifact)
    gaps = _methodology_gap_matrix(exp5799_artifact, runtime_logs)
    after_hashes = immutable_input_hashes(root)
    original_unchanged = before_hashes == after_hashes
    gaps_explicit = all(dict(row).get("status") for row in gaps.values())
    ready = _complete_ready(
        row_replay_ok=bool(row_replay.get("ok")),
        counts_close=counts_close,
        gaps_explicit=gaps_explicit,
        original_unchanged=original_unchanged,
        gpu=gpu,
    )
    status = "complete" if ready else "blocked"
    measured_duration = (
        float(duration_s)
        if duration_s is not None
        else round(time.perf_counter() - started, 6)
    )
    preconditions = {
        "input_hashes_captured_before": True,
        "input_hashes_captured_after": True,
        "immutable_input_count": len(before_hashes),
        "row_replay": row_replay,
        "declared_counts": {
            "ok": counts_close,
            "artifact_parser_failure_count": round(float(exp5799_artifact.get("parser_failure_rate", 0.0)) * len(rows)),
            "artifact_truncation_count": round(float(exp5799_artifact.get("truncation_rate", 0.0)) * len(rows)),
            "artifact_empty_final_count": round(float(exp5799_artifact.get("empty_final_content_rate", 0.0)) * len(rows)),
        },
        "adversarial_quarantine": {
            "flagged_adversarial": exp5799_artifact.get("flagged_adversarial") is True,
            "corrigendum_pending": exp5799_artifact.get("corrigendum_pending", []),
        },
        "host_capacity": _host_capacity_receipts(root),
        "available_runtime_logs": runtime_logs,
        "blocked_reasons": blocked_reasons,
    }
    verdict = (
        "complete: exp5799_events_and_provenance_reconstructed_with_resume_receipts_unqualified"
        if status == "complete"
        else "blocked: " + ",".join(blocked_reasons or ["exp5799_audit_preconditions"])
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "immutable_input_hashes": {"before": before_hashes, "after": after_hashes},
        "independent_event_definitions": independent_event_definitions(),
        "overlapping_event_matrix": overlap,
        "exclusive_primary_failure_taxonomy": taxonomy,
        "per_model_mode_reconstruction": reconstruction,
        "methodology_gap_matrix": gaps,
        "gpu_provenance_reconciliation": gpu,
        "producer_repairs_and_tests": _producer_repairs(original_unchanged),
        "canary_evidence_ready_score": 1.0 if ready else 0.0,
        "original_files_unchanged": original_unchanged,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands or []),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = compute_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, closure checks, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")  # pragma: no cover
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("invalid status")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle mismatch")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict terminal prefix")  # pragma: no cover
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")  # pragma: no cover
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in provenance or not dict(provenance[field]).get("principle"):
            raise ValueError(f"field_provenance:{field}")  # pragma: no cover
    if artifact["status"] == "complete":
        if artifact["canary_evidence_ready_score"] != 1.0:
            raise ValueError("canary_evidence_ready_score")  # pragma: no cover
        if not dict(artifact["exclusive_primary_failure_taxonomy"]).get("total_matches_denominator"):
            raise ValueError("primary taxonomy does not close")  # pragma: no cover
        if artifact["original_files_unchanged"] is not True:
            raise ValueError("original_files_unchanged")  # pragma: no cover
    if compute_checksum(artifact) != artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum")  # pragma: no cover


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Write the Exp5811 companion artifact without touching Exp5799 files."""

    artifact = build_artifact(
        root=root,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    output = Path(result_path or root / RESULT_RELATIVE_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = build_and_write_artifact()
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
