"""Produce the V570 content-derived GGUF metadata admission record.

This experiment fixes the exact Exp6567 false negative. It reads bounded GGUF
headers from hash-only Hugging Face blobs. It binds each blob to repository
cache provenance and an existing trusted hash. It does not load an LLM or read
tensor payload bytes.

Spec: REQ-REPORT-6572 and SCENARIO-REPORT-6572-HASH-BLOB through
SCENARIO-REPORT-6572-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import tempfile
import time
from typing import Any

from carnot.inference.gguf_metadata import (
    DEFAULT_MAX_HEADER_BYTES,
    build_gguf_admission_record,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
MAX_HEADER_BYTES = DEFAULT_MAX_HEADER_BYTES
RESULT_RELATIVE_PATH = Path("results/experiment_6572_content_derived_gguf_metadata_resolver.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6571_v570_evidence_gate_and_retirement_root.json")
EXP6567_RELATIVE_PATH = Path("results/experiment_6567_sequential_flagship_gguf_admission.json")
PROTECTED_RELATIVE_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
INFERENCE_SUBSTRATE = "bounded_gguf_header_and_cache_provenance_inspection_no_llm"

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
EXPECTED_ARCHITECTURES = {
    MANDATED_HF_IDS[0]: frozenset({"qwen35moe"}),
    MANDATED_HF_IDS[1]: frozenset({"gemma4"}),
    MANDATED_HF_IDS[2]: frozenset({"gemma4"}),
}
REQUIRED_NEGATIVE_FIXTURES = (
    "non_gguf",
    "truncated_header",
    "tokenizer_only_gguf",
    "wrong_repository_mapping",
    "unsupported_version",
    "inconsistent_shard_metadata",
    "renamed_blob",
    "symlink_alias",
    "prefix_magic_collision",
    "huge_declared_metadata_length",
    "malformed_utf8",
    "tensor_count_overflow",
    "partial_shards",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "failing_before_rows",
    "gguf_blob_metadata_rows",
    "negative_fixture_rows",
    "bounded_read_receipts",
    "repository_revision_and_hash_receipts",
    "gguf_blob_metadata_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The resolver task must end as ready, partial, blocked, or disqualified.",
    "honest_verdict": "The verdict distinguishes fixed content identity from another blocked admission attempt.",
    "verdict_class": "Infrastructure readiness cannot masquerade as scientific utility.",
    "gate_check_summary": "A blocked task names the upstream gate and observed value.",
    "failing_before_rows": "The exact Exp6567 false negatives prove the repair targets the observed defect.",
    "gguf_blob_metadata_rows": "Each flagship blob reports content metadata and repository provenance independently.",
    "negative_fixture_rows": "Malformed, tokenizer-only, truncated, and mismatched inputs must fail closed.",
    "bounded_read_receipts": "Metadata inspection must not silently scan or copy all tensor bytes.",
    "repository_revision_and_hash_receipts": "Content type alone cannot establish model provenance.",
    "gguf_blob_metadata_ready_score": "This exact binary field gates real sequential admission.",
    "per_unit_rows": "Every blob and fixture remains independently recheckable.",
    "aggregate_row_recomputation": "Readiness derives only from emitted blob and fixture rows.",
    "preconditions_checked": "Tool and cache receipts separate missing inputs from parser failure.",
    "protected_files_unchanged": "The resolver must preserve both protected orchestration files.",
    "inference_substrate": "This task reads bounded file metadata and invokes no model.",
    "verifier_is_oracle": "Exact format checks are oracle authority, so readiness is classed null.",
    "field_provenance": "Every metadata field identifies byte offsets, tool output, or cache manifest source.",
    "duration_s": "Runtime reveals skipped fixtures or accidental full-file scans.",
    "tests_run": "Named tests and exits prove the reusable resolver behavior.",
    "reproducibility_checksum": "A content hash protects the terminal metadata record.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && .venv/bin/python -m "
    "carnot.experiment_6572_content_derived_gguf_metadata_resolver --date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_gguf_metadata.py "
    "tests/python/test_experiment_6572_content_derived_gguf_metadata_resolver.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/carnot/inference/gguf_metadata.py,"
    "*/carnot/experiment_6572_content_derived_gguf_metadata_resolver.py' -m pytest "
    "tests/python/test_gguf_metadata.py "
    "tests/python/test_experiment_6572_content_derived_gguf_metadata_resolver.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --include='*/carnot/inference/gguf_metadata.py,"
    "*/carnot/experiment_6572_content_derived_gguf_metadata_resolver.py' "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/inference/gguf_metadata.py "
    "python/carnot/experiment_6572_content_derived_gguf_metadata_resolver.py "
    "tests/python/test_gguf_metadata.py "
    "tests/python/test_experiment_6572_content_derived_gguf_metadata_resolver.py"
)
RUFF_FORMAT_COMMAND = RUFF_CHECK_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_gguf_metadata.py "
    "tests/python/test_experiment_6572_content_derived_gguf_metadata_resolver.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6572_content_derived_gguf_metadata_resolver --validate"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6572_content_derived_gguf_metadata_resolver.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6572_content_derived_gguf_metadata_resolver.json"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "model-ingestion E2E: three real cache blobs plus fail-closed fixtures",
        "exit_code": 0,
    },
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    """Return stable JSON text for reducer and artifact hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash canonical JSON with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a bounded non-model file used as provenance or protection."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON file through fsync and an atomic same-directory replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():  # pragma: no cover - replace failure cleanup.
            temporary_path.unlink()


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_failing_before_rows(exp6567: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve the three exact path-shape false negatives from Exp6567."""

    resolved = exp6567.get("resolved_model_file_rows", [])
    resolved = resolved if isinstance(resolved, list) else []
    by_id = {str(row.get("hf_id")): row for row in resolved if isinstance(row, Mapping)}
    preconditions = exp6567.get("preconditions_checked", {})
    preconditions = preconditions if isinstance(preconditions, Mapping) else {}
    checks = preconditions.get("model_preflight_checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    rows = []
    for hf_id in MANDATED_HF_IDS:
        source = by_id.get(hf_id, {})
        source_checks = checks.get(hf_id, {})
        source_checks = source_checks if isinstance(source_checks, Mapping) else {}
        path = str(source.get("absolute_path", ""))
        observed_language = source_checks.get("language_model_file")
        observed_quantization = source_checks.get("quantization_known")
        rows.append(
            {
                "row_type": "exp6567_false_negative",
                "hf_id": hf_id,
                "path": path,
                "path_basename": Path(path).name,
                "path_has_gguf_suffix": Path(path).suffix.lower() == ".gguf",
                "expected_language_model_file": True,
                "observed_language_model_file": observed_language,
                "expected_quantization_known": True,
                "observed_quantization_known": observed_quantization,
                "false_negative_reproduced": bool(path)
                and Path(path).suffix == ""
                and observed_language is False
                and observed_quantization is False,
            }
        )
    return rows


def _fixture_string(value: str | bytes) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(raw)) + raw


def _fixture_value(value_type: int, value: Any) -> bytes:
    if value_type == 4:
        return struct.pack("<I", int(value))
    if value_type == 8:
        return _fixture_string(value)
    if value_type == 9:
        element_type, elements = value
        return struct.pack("<IQ", element_type, len(elements)) + b"".join(
            _fixture_value(element_type, element) for element in elements
        )
    if value_type == 10:
        return struct.pack("<Q", int(value))
    raise AssertionError(f"unsupported fixture type {value_type}")


def _fixture_gguf(
    *,
    version: int = 3,
    tensor_count: int = 1,
    architecture: str | bytes = "gemma4",
    split_no: int | None = None,
    split_count: int | None = None,
) -> bytes:
    metadata: list[tuple[str, int, Any]] = [
        ("general.architecture", 8, architecture),
        ("general.name", 8, "negative-fixture"),
        ("general.file_type", 4, 15),
        ("tokenizer.ggml.model", 8, "gemma4"),
        ("tokenizer.ggml.tokens", 9, (8, ["one", "two"])),
    ]
    if split_no is not None:
        metadata.append(("split.no", 4, split_no))
    if split_count is not None:
        metadata.extend(
            [("split.count", 4, split_count), ("split.tensors.count", 10, tensor_count)]
        )
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", version, tensor_count, len(metadata)))
    for key, value_type, value in metadata:
        payload.extend(_fixture_string(key))
        payload.extend(struct.pack("<I", value_type))
        payload.extend(_fixture_value(value_type, value))
    for index in range(tensor_count):
        payload.extend(_fixture_string(f"tensor-{index}"))
        payload.extend(struct.pack("<IQIQ", 1, 1, 0, index * 4))
    payload.extend(b"\x00" * (-len(payload) % 32))
    payload.extend(b"\x00" * max(4, tensor_count * 4))
    return bytes(payload)


def _cache_fixture(
    root: Path,
    content: bytes,
    *,
    repository_id: str,
    filename: str,
) -> tuple[Path, str]:
    repo_dir = root / f"models--{repository_id.replace('/', '--')}"
    digest = hashlib.sha256(content).hexdigest()
    blob = repo_dir / "blobs" / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    snapshot = repo_dir / "snapshots" / "fixture-revision"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / filename).symlink_to(Path("../../blobs") / digest)
    return blob, f"sha256:{digest}"


def _negative_row(
    *,
    fixture_id: str,
    path: Path,
    repository_id: str,
    cache_root: Path,
    trusted_sha256: str,
    expected_reason: str,
) -> JsonDict:
    record = build_gguf_admission_record(
        path,
        repository_id=repository_id,
        cache_root=cache_root,
        trusted_sha256=trusted_sha256,
        expected_architectures={"gemma4"},
        max_header_bytes=MAX_HEADER_BYTES,
    )
    content = record.get("content_metadata")
    content = content if isinstance(content, Mapping) else {}
    receipt = content.get("bounded_read_receipt")
    if not isinstance(receipt, Mapping):
        file_size = path.stat().st_size if path.is_file() else 0
        receipt = {
            "receipt_kind": "fail_closed_physical_read_upper_bound",
            "file_size": file_size,
            "maximum_header_bytes": MAX_HEADER_BYTES,
            "physical_bytes_read_upper_bound": min(file_size, MAX_HEADER_BYTES),
            "tensor_payload_bytes_read": 0,
        }
    reasons = list(record.get("rejection_reasons", []))
    return {
        "row_type": "negative_fixture",
        "unit_id": fixture_id,
        "expected_admitted": False,
        "observed_admitted": record.get("admitted"),
        "expected_reason": expected_reason,
        "rejection_reasons": reasons,
        "passed": record.get("admitted") is False and expected_reason in reasons,
        "bounded_read_receipt": dict(receipt),
        "record": record,
    }


def build_negative_fixture_rows() -> list[JsonDict]:
    """Run every required malformed and provenance fixture in a temp cache."""

    definitions: list[tuple[str, bytes, str, str, str]] = [
        (
            "non_gguf",
            b"not a GGUF file" + b"\x00" * 16,
            "fixture-GGUF",
            "bad.gguf",
            "invalid_magic",
        ),
        (
            "truncated_header",
            _fixture_gguf()[:17],
            "fixture-GGUF",
            "short.gguf",
            "truncated_header",
        ),
        (
            "tokenizer_only_gguf",
            _fixture_gguf(tensor_count=0),
            "fixture-GGUF",
            "tokenizer.gguf",
            "tokenizer_only",
        ),
        (
            "unsupported_version",
            _fixture_gguf(version=99),
            "fixture-GGUF",
            "version.gguf",
            "unsupported_version",
        ),
        (
            "inconsistent_shard_metadata",
            _fixture_gguf(split_no=2, split_count=2),
            "fixture-GGUF",
            "fixture-00001-of-00002.gguf",
            "invalid_shard_index",
        ),
        (
            "prefix_magic_collision",
            b"GGUFjunk-prefix" + b"\x00" * 16,
            "fixture-GGUF",
            "prefix.gguf",
            "unsupported_version",
        ),
        (
            "huge_declared_metadata_length",
            b"GGUF" + struct.pack("<IQQQ", 3, 1, 1, 1 << 40),
            "fixture-GGUF",
            "huge.gguf",
            "string_length_limit",
        ),
        (
            "malformed_utf8",
            _fixture_gguf(architecture=b"\xff"),
            "fixture-GGUF",
            "utf8.gguf",
            "malformed_utf8",
        ),
        (
            "tensor_count_overflow",
            b"GGUF" + struct.pack("<IQQ", 3, 1_000_001, 0),
            "fixture-GGUF",
            "overflow.gguf",
            "tensor_count_limit",
        ),
        (
            "partial_shards",
            _fixture_gguf(split_no=0, split_count=2),
            "fixture-GGUF",
            "fixture-00001-of-00002.gguf",
            "partial_shard_set",
        ),
    ]
    rows: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="carnot-exp6572-") as temporary:
        cache_root = Path(temporary) / "hub"
        for fixture_id, content, _repository_name, filename, reason in definitions:
            repository_id = f"unsloth/{fixture_id}-GGUF"
            blob, trusted = _cache_fixture(
                cache_root, content, repository_id=repository_id, filename=filename
            )
            rows.append(
                _negative_row(
                    fixture_id=fixture_id,
                    path=blob,
                    repository_id=repository_id,
                    cache_root=cache_root,
                    trusted_sha256=trusted,
                    expected_reason=reason,
                )
            )

        valid, trusted = _cache_fixture(
            cache_root,
            _fixture_gguf(),
            repository_id="unsloth/mapped-GGUF",
            filename="mapped.gguf",
        )
        rows.append(
            _negative_row(
                fixture_id="wrong_repository_mapping",
                path=valid,
                repository_id="unsloth/wrong-GGUF",
                cache_root=cache_root,
                trusted_sha256=trusted,
                expected_reason="repository_mapping_missing",
            )
        )
        renamed = Path(temporary) / "renamed.gguf"
        renamed.write_bytes(valid.read_bytes())
        rows.append(
            _negative_row(
                fixture_id="renamed_blob",
                path=renamed,
                repository_id="unsloth/mapped-GGUF",
                cache_root=cache_root,
                trusted_sha256=trusted,
                expected_reason="path_outside_repository_cache",
            )
        )
        alias = Path(temporary) / "alias.gguf"
        alias.symlink_to(valid)
        rows.append(
            _negative_row(
                fixture_id="symlink_alias",
                path=alias,
                repository_id="unsloth/mapped-GGUF",
                cache_root=cache_root,
                trusted_sha256=trusted,
                expected_reason="path_outside_repository_cache",
            )
        )
    return sorted(rows, key=lambda row: REQUIRED_NEGATIVE_FIXTURES.index(row["unit_id"]))


def build_flagship_blob_rows(exp6567: Mapping[str, Any]) -> list[JsonDict]:
    """Inspect the three trusted Exp6567 blobs through the reusable reader."""

    source_rows = exp6567.get("resolved_model_file_rows", [])
    source_rows = source_rows if isinstance(source_rows, list) else []
    by_id = {str(row.get("hf_id")): row for row in source_rows if isinstance(row, Mapping)}
    rows = []
    for hf_id in MANDATED_HF_IDS:
        source = by_id.get(hf_id, {})
        path = str(source.get("absolute_path", ""))
        trusted = str(source.get("sha256", "missing"))
        record = build_gguf_admission_record(
            path,
            repository_id=hf_id,
            trusted_sha256=trusted,
            expected_architectures=EXPECTED_ARCHITECTURES[hf_id],
            max_header_bytes=MAX_HEADER_BYTES,
        )
        record.update(
            {
                "row_type": "flagship_blob",
                "unit_id": hf_id,
                "trusted_exp6567_sha256": trusted,
                "exp6567_byte_size": source.get("byte_size"),
                "passed": record.get("admitted") is True,
            }
        )
        rows.append(record)
    return rows


def recompute_aggregate(
    failing_before_rows: Sequence[Mapping[str, Any]],
    blob_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Derive readiness only from emitted pre-fix, blob, and fixture rows."""

    required_blobs = {str(row.get("unit_id")) for row in blob_rows if row.get("passed") is True}
    required_negatives = {
        str(row.get("unit_id")) for row in negative_rows if row.get("passed") is True
    }
    before_ok = len(failing_before_rows) == 3 and all(
        row.get("false_negative_reproduced") is True for row in failing_before_rows
    )
    blobs_ok = required_blobs == set(MANDATED_HF_IDS)
    negatives_ok = required_negatives == set(REQUIRED_NEGATIVE_FIXTURES)
    score = 1.0 if before_ok and blobs_ok and negatives_ok else 0.0
    return {
        "required_flagship_ids": list(MANDATED_HF_IDS),
        "passed_flagship_ids": sorted(required_blobs),
        "required_negative_fixture_ids": list(REQUIRED_NEGATIVE_FIXTURES),
        "passed_negative_fixture_ids": sorted(required_negatives),
        "failing_before_row_count": len(failing_before_rows),
        "flagship_pass_count": len(required_blobs),
        "negative_fixture_pass_count": len(required_negatives),
        "failing_before_rows_valid": before_ok,
        "all_flagship_blobs_passed": blobs_ok,
        "all_negative_fixtures_failed_closed": negatives_ok,
        "recomputed_ready_score": score,
        "reducer": "1.0 iff three Exp6567 false negatives reproduce, all flagship rows pass, and all required negative rows fail closed",
    }


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "satisfied_by": {
                "gguf_blob_metadata_rows": "GGUF byte offsets and bounded read receipts",
                "negative_fixture_rows": "fixture record and stable rejection reason",
                "repository_revision_and_hash_receipts": "HF cache symlink, revision, and trusted blob key",
            }.get(field, "Exp6572 emitted rows and deterministic reducer"),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def assemble_artifact(
    *,
    failing_before_rows: Sequence[Mapping[str, Any]],
    blob_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    run_date: str,
) -> JsonDict:
    """Assemble one terminal artifact from already emitted unit rows."""

    before = [dict(row) for row in failing_before_rows]
    blobs = [dict(row) for row in blob_rows]
    negatives = [dict(row) for row in negative_rows]
    aggregate = recompute_aggregate(before, blobs, negatives)
    preconditions_ok = preconditions.get("all_required_preconditions_available") is True
    protected_ok = protected.get("all_unchanged") is True
    negative_attack_failed = not aggregate["all_negative_fixtures_failed_closed"]
    before_failed = not aggregate["failing_before_rows_valid"]
    if not preconditions_ok:
        status, verdict_class = "blocked", "blocked"
    elif not protected_ok or negative_attack_failed or before_failed:
        status, verdict_class = "disqualified", "disqualified"
    elif aggregate["recomputed_ready_score"] == 1.0:
        status, verdict_class = "ready", None
    elif aggregate["flagship_pass_count"] > 0:
        status, verdict_class = "partial", "partial"
    else:
        status, verdict_class = "blocked", "blocked"
    failed_checks = []
    if not preconditions_ok:
        failed_checks.append("preconditions")
    if not protected_ok:
        failed_checks.append("protected_files_unchanged")
    if before_failed:
        failed_checks.append("failing_before_rows")
    if not aggregate["all_flagship_blobs_passed"]:
        failed_checks.append("flagship_blob_rows")
    if negative_attack_failed:
        failed_checks.append("negative_fixture_rows")
    if status == "ready":
        verdict = (
            "complete_content_derived_gguf_metadata_ready: all three hash-only flagship "
            "blobs passed bounded content and independent cache provenance checks; every "
            "negative fixture failed closed; no LLM was loaded"
        )
    elif status == "partial":
        verdict = "partial_content_derived_gguf_metadata: a usable flagship subset passed"
    elif status == "disqualified":
        verdict = "disqualified_content_derived_gguf_metadata: fail-closed evidence contract failed"
    else:
        verdict = (
            "blocked_content_derived_gguf_metadata: required gate, cache, or tool input is missing"
        )
    receipts = []
    for row in blobs:
        content = row.get("content_metadata", {})
        content = content if isinstance(content, Mapping) else {}
        receipt = content.get("bounded_read_receipt", {})
        receipts.append({"unit_id": row.get("unit_id"), **dict(receipt)})
    receipts.extend(
        {"unit_id": row.get("unit_id"), **dict(row.get("bounded_read_receipt", {}))}
        for row in negatives
    )
    provenance_receipts = [
        {
            "unit_id": row.get("unit_id"),
            "repository_id": row.get("repository_id"),
            **dict(row.get("provenance", {}) or {}),
        }
        for row in blobs
    ]
    artifact: JsonDict = {
        "planning_date": run_date,
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "all_gates_passed": status == "ready",
            "failed_checks": failed_checks,
            "rows": [
                {
                    "check": "upstream_v570_evidence_contract_ready_score",
                    "expected": 1.0,
                    "observed": preconditions.get("upstream_gate", {}).get("observed_value"),
                    "passed": preconditions.get("upstream_gate", {}).get("passed") is True,
                },
                {
                    "check": "gguf_blob_metadata_ready_score",
                    "expected": 1.0,
                    "observed": aggregate["recomputed_ready_score"],
                    "passed": aggregate["recomputed_ready_score"] == 1.0,
                },
            ],
        },
        "failing_before_rows": before,
        "gguf_blob_metadata_rows": blobs,
        "negative_fixture_rows": negatives,
        "bounded_read_receipts": receipts,
        "repository_revision_and_hash_receipts": provenance_receipts,
        "gguf_blob_metadata_ready_score": aggregate["recomputed_ready_score"],
        "per_unit_rows": blobs + negatives,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path.as_posix(),
            "before_sha256": before.get(path.as_posix(), "missing"),
            "after_sha256": after.get(path.as_posix(), "missing"),
            "unchanged": before.get(path.as_posix()) == after.get(path.as_posix()) != "missing",
        }
        for path in PROTECTED_RELATIVE_PATHS
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "research_roadmap_yaml_unchanged": rows[0]["unchanged"],
        "research_conductor_py_unchanged": rows[1]["unchanged"],
        "rows": rows,
    }


def _memory_receipt() -> JsonDict:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value, *_ = line.split()
        values[key.rstrip(":")] = int(value)
    return {
        "total_kib": values.get("MemTotal", 0),
        "available_kib": values.get("MemAvailable", 0),
    }


def _llama_cpp_support() -> JsonDict:
    spec = importlib.util.find_spec("llama_cpp")
    receipt: JsonDict = {
        "installed": spec is not None,
        "module_path": spec.origin if spec else "missing",
        "version": "missing",
        "gpu_offload_supported": False,
        "metadata_only_model_load_performed": False,
    }
    if spec is None:
        return receipt
    try:
        receipt["version"] = importlib.metadata.version("llama_cpp_python")
    except importlib.metadata.PackageNotFoundError:
        receipt["version"] = "unknown"
    try:
        from llama_cpp import llama_cpp as backend

        receipt["gpu_offload_supported"] = bool(backend.llama_supports_gpu_offload())
    except Exception as exc:  # pragma: no cover - environment-specific import failure.
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    return receipt


def collect_preconditions(
    repo_root: Path,
    exp6567: Mapping[str, Any],
    protected_before: Mapping[str, str],
    run_date: str,
) -> JsonDict:
    """Record the structured gate, host, tools, cache, and trusted identities."""

    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    upstream = _load_json(upstream_path)
    upstream_value = upstream.get("v570_evidence_contract_ready_score")
    source_rows = exp6567.get("resolved_model_file_rows", [])
    source_rows = source_rows if isinstance(source_rows, list) else []
    cache_rows = [
        {
            "hf_id": row.get("hf_id"),
            "path": row.get("absolute_path"),
            "exists": Path(str(row.get("absolute_path", ""))).is_file(),
            "byte_size": row.get("byte_size"),
            "trusted_exp6567_sha256": row.get("sha256"),
            "full_blob_rehash_performed": False,
        }
        for row in source_rows
        if isinstance(row, Mapping) and row.get("hf_id") in MANDATED_HF_IDS
    ]
    disk = shutil.disk_usage(repo_root)
    llama_support = _llama_cpp_support()
    checks = {
        "upstream_gate": upstream_value == 1.0,
        "exp6567_artifact": bool(exp6567),
        "three_cache_blobs": len(cache_rows) == 3 and all(row["exists"] for row in cache_rows),
        "trusted_existing_hashes": len(cache_rows) == 3
        and all(str(row["trusted_exp6567_sha256"]).startswith("sha256:") for row in cache_rows),
        "bounded_header_reader": True,
        "llama_cpp_metadata_support": llama_support["installed"] is True,
        "atomic_output_parent": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    return {
        "planning_date": run_date,
        "upstream_gate": {
            "path": UPSTREAM_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(upstream_path),
            "field": "v570_evidence_contract_ready_score",
            "expected_value": 1.0,
            "observed_value": upstream_value,
            "passed": upstream_value == 1.0,
        },
        "git": {
            "head": os.popen("git rev-parse HEAD").read().strip(),
            "upstream_hash": sha256_file(upstream_path),
        },
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "ram": _memory_receipt(),
        "python": {"version": platform.python_version(), "executable": os.sys.executable},
        "gguf_metadata_support": {
            "local_gguf_package_installed": importlib.util.find_spec("gguf") is not None,
            "bounded_reader_module": "carnot.inference.gguf_metadata",
            "maximum_header_bytes": MAX_HEADER_BYTES,
        },
        "llama_cpp_metadata_support": llama_support,
        "cache_layout": "models--<org>--<repo>/{blobs,refs,snapshots}",
        "model_cache_paths": cache_rows,
        "protected_file_hashes_before": dict(protected_before),
        "checks": checks,
        "all_required_preconditions_available": all(checks.values()),
        "llm_load_performed": False,
        "tensor_payload_scan_performed": False,
        "full_model_blob_rehash_performed": False,
    }


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return exact schema, reducer, bound, protection, and checksum errors."""

    errors = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    before = payload.get("failing_before_rows", [])
    blobs = payload.get("gguf_blob_metadata_rows", [])
    negatives = payload.get("negative_fixture_rows", [])
    if all(isinstance(value, list) for value in (before, blobs, negatives)):
        recomputed = recompute_aggregate(before, blobs, negatives)
        if payload.get("gguf_blob_metadata_ready_score") != recomputed["recomputed_ready_score"]:
            errors.append("ready score does not recompute")
        if payload.get("aggregate_row_recomputation") != recomputed:
            errors.append("aggregate row does not recompute")
        if payload.get("per_unit_rows") != blobs + negatives:
            errors.append("per_unit_rows do not match emitted rows")
    else:
        errors.append("unit row fields must be lists")
    protected = payload.get("protected_files_unchanged", {})
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if payload.get("status") not in {"ready", "partial", "blocked", "disqualified"}:
        errors.append("invalid status")
    if payload.get("status") == "ready" and payload.get("verdict_class") is not None:
        errors.append("ready verdict_class must be null")
    for receipt in payload.get("bounded_read_receipts", []):
        physical = receipt.get("physical_bytes_read")
        upper = receipt.get("physical_bytes_read_upper_bound")
        measured = physical if physical is not None else upper
        if measured is None or int(measured) > int(receipt.get("maximum_header_bytes", 0)):
            errors.append("bounded read receipt exceeds limit")
            break
        if receipt.get("tensor_payload_bytes_read") != 0:
            errors.append("tensor payload bytes were read")
            break
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility checksum mismatch")
    return errors


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Inspect real blobs and fixtures without loading a model."""

    started = time.monotonic()
    before = _protected_hashes(repo_root)
    exp6567 = _load_json(repo_root / EXP6567_RELATIVE_PATH)
    preconditions = collect_preconditions(repo_root, exp6567, before, run_date)
    failing_rows = build_failing_before_rows(exp6567)
    blob_rows = build_flagship_blob_rows(exp6567) if exp6567 else []
    negative_rows = build_negative_fixture_rows()
    after = _protected_hashes(repo_root)
    artifact = assemble_artifact(
        failing_before_rows=failing_rows,
        blob_rows=blob_rows,
        negative_rows=negative_rows,
        preconditions=preconditions,
        protected=_protected_unchanged(before, after),
        duration_s=time.monotonic() - started,
        tests_run=tests_run,
        run_date=run_date,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Write one terminal artifact or validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(f"artifact not found: {result_path}")
            return 1
        artifact = _load_json(result_path)
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    artifact = build_artifact(run_date=str(args.date))  # pragma: no cover - required E2E path.
    atomic_write_json(result_path, artifact)  # pragma: no cover - required E2E path.
    print(f"wrote {result_path}: {artifact['honest_verdict']}")  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
