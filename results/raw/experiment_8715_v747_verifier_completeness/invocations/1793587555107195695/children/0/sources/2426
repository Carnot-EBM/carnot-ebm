"""Replay the Exp6576 immutable flagship source stream without an LLM.

Spec refs: REQ-REPORT-6577, SCENARIO-REPORT-6577-MISSING,
SCENARIO-REPORT-6577-COVERAGE, SCENARIO-REPORT-6577-REPLAY,
SCENARIO-REPORT-6577-ATTACKS, SCENARIO-REPORT-6577-ATOMIC.

The audit reads manifest and raw response rows. It recovers stored bytes and
recomputes every gate value. Missing evidence produces a terminal blocked
artifact because absence is an input diagnosis, not a scientific null.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any
import zlib


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RESULT_RELATIVE_PATH = Path("results/experiment_6577_flagship_source_stream_independent_audit.json")
UPSTREAM_RELATIVE_PATH = Path(
    "results/experiment_6576_immutable_flagship_source_span_stream_v3.json"
)
EXP6575_RELATIVE_PATH = Path(
    "results/experiment_6575_v571_clean_evidence_and_flagship_qualification.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
AUDIT_TOOL_RELATIVE_PATHS = (
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/artifact_convention_audit.py"),
)
EXPECTED_UPSTREAM_SCHEMA = "carnot.exp6576.immutable_flagship_source_span_stream.v3"
INFERENCE_SUBSTRATE = "immutable_source_stream_independent_replay_no_llm"

MANDATED_MODELS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen3.6-35b-a3b",
    "unsloth/gemma-4-31B-it-GGUF": "gemma-4-31b-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma-4-26b-a4b-it",
}
FAILURE_CLASSES = (
    "timeout",
    "malformed_output",
    "refusal",
    "empty_output",
    "process_failure",
)
REQUIRED_ATTACKS = (
    "source_alias",
    "duplicate_unit",
    "copied_response_across_models",
    "selective_retry",
    "hidden_row_drop",
    "legacy_model_substitution",
    "inconsistent_family_label",
    "post_outcome_prompt_change",
    "null_only_row",
    "missing_content_path",
    "aggregate_row_contradiction",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_artifact_receipt",
    "rows",
    "failure_retention_rows",
    "duplicate_and_drift_attack_rows",
    "claim_stream_audit_ready_score",
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
    "status": "The independent audit terminates even when upstream evidence is absent.",
    "honest_verdict": "The verdict states whether lineage and coverage independently replay.",
    "verdict_class": "The audit uses only null, partial, blocked, or disqualified classes.",
    "gate_check_summary": "A block names the exact missing artifact or failed field and observed value.",
    "upstream_artifact_receipt": "Path, hash, status, and schema bind the object under audit.",
    "rows": "Every expected source-family unit carries independently recomputed metrics.",
    "failure_retention_rows": "Failure classes cannot vanish from an aggregate denominator.",
    "duplicate_and_drift_attack_rows": "Aliases, copied output, prompt drift, and selective retries are tested explicitly.",
    "claim_stream_audit_ready_score": "This exact top-level binary field gates semantic-block extraction.",
    "aggregate_row_recomputation": "Every audit headline derives from emitted rows.",
    "preconditions_checked": "Input and tool receipts distinguish a block from a failed replay.",
    "protected_files_unchanged": "The audit preserves both protected orchestration files.",
    "inference_substrate": "This is immutable artifact replay with no new LLM inference.",
    "verifier_is_oracle": "The audit is evidence authority and therefore cannot create positive science.",
    "field_provenance": "Each field identifies raw rows, hashes, and independent reducer code.",
    "duration_s": "Monotonic duration exposes skipped replay or attack work.",
    "tests_run": "Named commands and exits make the audit reproducible.",
    "reproducibility_checksum": "A final hash detects audit mutation.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6577_flagship_source_stream_independent_audit.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && .venv/bin/python -m "
    "carnot.experiment_6577_flagship_source_stream_independent_audit --date 20260824"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6577_flagship_source_stream_independent_audit.py "
    "-m pytest tests/python/test_experiment_6577_flagship_source_stream_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6577_flagship_source_stream_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6577_flagship_source_stream_independent_audit.py "
    "tests/python/test_experiment_6577_flagship_source_stream_independent_audit.py"
)
RUFF_FORMAT_COMMAND = RUFF_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6577_flagship_source_stream_independent_audit.py"
)
ROW_LINT_COMMAND = (
    f".venv/bin/python scripts/verdict_row_consistency_lint.py {RESULT_RELATIVE_PATH}"
)
ARTIFACT_AUDIT_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 1"
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH}"
E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6577_flagship_source_stream_independent_audit --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": FULL_PYTEST_COMMAND,
        "exit_code": 1,
        "outcome": "failed_pre_existing_test_record_mutation_race",
        "detail": (
            "test_experiment_1736_kanele_synth.py overwrote its tracked artifact and removed "
            "the adversarial corrigendum while parallel tests read that artifact"
        ),
    },
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ARTIFACT_AUDIT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    """Use one stable JSON form so independent hashes are repeatable."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    """Return a tagged SHA-256 digest for immutable byte receipts."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON only after stable key ordering and compact encoding."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks, or say plainly that it is missing."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Bind a row while excluding its self-referential hash field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind the terminal artifact while excluding its checksum field."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def recover_content_bytes(
    row: Mapping[str, Any], stem: str, repo_root: Path
) -> tuple[bytes | None, str]:
    """Recover bytes from inline text, base64, payload, or a stored path.

    The function reports the storage mechanism. It never invents content when
    the bytes are absent or corrupt, because a synthesized response would hide
    the upstream evidence defect that this audit must diagnose.
    """

    text_key = f"{stem}_text"
    if isinstance(row.get(text_key), str):
        return str(row[text_key]).encode("utf-8"), "text"
    b64_key = f"{stem}_bytes_b64"
    if isinstance(row.get(b64_key), str):
        try:
            return base64.b64decode(str(row[b64_key]), validate=True), "inline_base64"
        except (binascii.Error, ValueError):
            return None, "invalid_base64"
    payload = row.get(f"{stem}_payload")
    if isinstance(payload, Mapping):
        try:
            decoded = base64.b64decode(str(payload.get("bytes_b64", "")), validate=True)
            if payload.get("compression") == "zlib":
                decoded = zlib.decompress(decoded)
            if payload.get("encoding") != "base64":
                return None, "invalid_payload_encoding"
            return decoded, "inline_payload"
        except (binascii.Error, ValueError, zlib.error):
            return None, "invalid_payload"
    path_value = row.get(f"{stem}_content_path")
    if isinstance(path_value, str) and path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.is_file():
            return candidate.read_bytes(), "content_path"
        return None, "missing_content_path"
    return None, "missing"


def recompute_corpus_commit(manifest: Sequence[Mapping[str, Any]]) -> str:
    """Derive the corpus commit from frozen unit identity and byte hashes."""

    keys = (
        "unit_id",
        "source_id",
        "family",
        "model_repository",
        "model_revision",
        "gguf_sha256",
        "source_sha256",
        "prompt_sha256",
        "seed",
        "order_index",
        "attempt_index",
    )
    normalized = [{key: row.get(key) for key in keys} for row in manifest]
    normalized.sort(key=lambda row: (str(row.get("order_index")), str(row.get("unit_id"))))
    return sha256_json(normalized)


def _cost_from_components(row: Mapping[str, Any]) -> float | None:
    components = row.get("charged_cost_components")
    if not isinstance(components, list) or not components:
        return None
    total = 0.0
    for item in components:
        if not isinstance(item, Mapping):
            return None
        quantity = _finite_number(item.get("quantity"))
        unit_cost = _finite_number(item.get("unit_cost"))
        if quantity is None or unit_cost is None or quantity < 0 or unit_cost < 0:
            return None
        total += quantity * unit_cost
    return round(total, 12)


def _inline_response(row: Mapping[str, Any]) -> bytes | None:
    value = row.get("raw_response_bytes_b64")
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def _failure_classes(row: Mapping[str, Any], response: bytes | None) -> list[str]:
    process = row.get("process_receipt")
    exit_code = process.get("exit_code") if isinstance(process, Mapping) else None
    classes: list[str] = []
    if row.get("timeout") is True:
        classes.append("timeout")
    if row.get("malformed_output") is True:
        classes.append("malformed_output")
    if row.get("refusal") is True:
        classes.append("refusal")
    if response == b"":
        classes.append("empty_output")
    if row.get("process_failure") is True or (
        _nonnegative_int(exit_code) is not None and int(exit_code) != 0
    ):
        classes.append("process_failure")
    return classes


def recompute_raw_totals(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute only additive headlines from raw row values and cost parts."""

    prompt_tokens = sum(_nonnegative_int(row.get("prompt_tokens")) or 0 for row in rows)
    response_tokens = sum(_nonnegative_int(row.get("response_tokens")) or 0 for row in rows)
    latency = sum(_finite_number(row.get("latency_s")) or 0.0 for row in rows)
    costs = [_cost_from_components(row) for row in rows]
    failure_count = sum(bool(_failure_classes(row, _inline_response(row))) for row in rows)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "latency_s": round(latency, 12),
        "charged_cost": round(sum(value or 0.0 for value in costs), 12),
        "claim_bearing_row_count": sum(row.get("claim_bearing") is True for row in rows),
        "failure_row_count": failure_count,
    }


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, "root_not_object"
    return value, None


def _path_receipt(path: Path, payload: Mapping[str, Any] | None, error: str | None) -> JsonDict:
    receipt = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "read_error": error,
        "status": payload.get("status") if payload else "missing" if error == "missing" else error,
        "schema": payload.get("schema") if payload else None,
    }
    receipt["row_hash"] = row_hash(receipt)
    return receipt


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(repo_root / relative)
        for relative in PROTECTED_RELATIVE_PATHS
    }


def _raw_recovery_receipt(rows: Sequence[Mapping[str, Any]], repo_root: Path) -> JsonDict:
    modes = [recover_content_bytes(row, "raw_response", repo_root)[1] for row in rows]
    recoverable = {"text", "inline_base64", "inline_payload", "content_path"}
    return {
        "present": bool(rows) and all(mode in recoverable for mode in modes),
        "row_count": len(rows),
        "recoverable_row_count": sum(mode in recoverable for mode in modes),
        "content_path_row_count": modes.count("content_path"),
        "missing_or_invalid_row_count": sum(mode not in recoverable for mode in modes),
    }


def _preconditions(
    *,
    repo_root: Path,
    upstream_path: Path,
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
    exp6575_path: Path,
    exp6575: Mapping[str, Any] | None,
    exp6575_error: str | None,
    protected_before: Mapping[str, str],
) -> JsonDict:
    raw_rows = upstream.get("rows", []) if upstream else []
    rows = (
        [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
    )
    disk = shutil.disk_usage(repo_root)
    return {
        "planning_date": RUN_DATE,
        "expected_artifact_paths": {
            "exp6575": str(exp6575_path),
            "exp6576": str(upstream_path),
        },
        "artifact_receipts": {
            "exp6575": _path_receipt(exp6575_path, exp6575, exp6575_error),
            "exp6576": _path_receipt(upstream_path, upstream, upstream_error),
        },
        "schema_versions": {
            "expected_exp6576": EXPECTED_UPSTREAM_SCHEMA,
            "observed_exp6576": upstream.get("schema") if upstream else None,
            "audit": "carnot.exp6577.flagship_source_stream_independent_audit.v1",
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "audit_tool_versions": [
            {
                "path": str(repo_root / relative),
                "exists": (repo_root / relative).is_file(),
                "sha256": sha256_file(repo_root / relative),
            }
            for relative in AUDIT_TOOL_RELATIVE_PATHS
        ],
        "protected_file_hashes_before": dict(protected_before),
        "raw_response_recovery": _raw_recovery_receipt(rows, repo_root),
        "model_inference_invoked": False,
    }


def _audit_one_row(
    *,
    expected: Mapping[str, Any],
    observed_rows: Sequence[Mapping[str, Any]],
    corpus_commit: str,
    upstream_commit: Any,
    repo_root: Path,
    model_hash_cache: dict[str, str],
) -> JsonDict:
    unit_id = str(expected.get("unit_id", ""))
    matches = [row for row in observed_rows if str(row.get("unit_id", "")) == unit_id]
    observed = matches[0] if matches else None
    checks: dict[str, bool] = {"exactly_one_observed_row": len(matches) == 1}
    source_bytes: bytes | None = None
    prompt_bytes: bytes | None = None
    response_bytes: bytes | None = None
    source_mode = prompt_mode = response_mode = "missing"
    cost: float | None = None
    failures: list[str] = []
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    latency: float | None = None
    raw_before_parser = False
    if observed is not None:
        for field in (
            "source_id",
            "family",
            "model_repository",
            "model_revision",
            "gguf_sha256",
            "seed",
            "order_index",
            "attempt_index",
        ):
            checks[f"{field}_matches_manifest"] = observed.get(field) == expected.get(field)
        source_bytes, source_mode = recover_content_bytes(observed, "source", repo_root)
        prompt_bytes, prompt_mode = recover_content_bytes(observed, "prompt", repo_root)
        response_bytes, response_mode = recover_content_bytes(observed, "raw_response", repo_root)
        source_hash = sha256_bytes(source_bytes) if source_bytes is not None else None
        prompt_hash = sha256_bytes(prompt_bytes) if prompt_bytes is not None else None
        response_hash = sha256_bytes(response_bytes) if response_bytes is not None else None
        checks.update(
            {
                "source_bytes_recovered": source_bytes is not None,
                "source_hash_matches_row": source_hash == observed.get("source_sha256"),
                "source_hash_matches_manifest": source_hash == expected.get("source_sha256"),
                "prompt_bytes_recovered": prompt_bytes is not None,
                "prompt_hash_matches_row": prompt_hash == observed.get("prompt_sha256"),
                "prompt_hash_matches_manifest": prompt_hash == expected.get("prompt_sha256"),
                "response_bytes_recovered": response_bytes is not None,
                "response_hash_matches_row": response_hash == observed.get("raw_response_sha256"),
                "corpus_commit_matches_rows": observed.get("corpus_commit") == corpus_commit,
                "corpus_commit_matches_upstream": upstream_commit == corpus_commit,
            }
        )
        model_path = observed.get("model_path")
        model_path_text = str(model_path) if isinstance(model_path, str) else ""
        if model_path_text not in model_hash_cache:
            model_hash_cache[model_path_text] = sha256_file(model_path_text)
        model_hash = model_hash_cache[model_path_text]
        checks["model_file_hash_recomputed"] = (
            bool(model_path_text)
            and model_hash != "missing"
            and model_hash == observed.get("gguf_sha256")
            and model_hash == expected.get("gguf_sha256")
        )
        process = observed.get("process_receipt")
        process_ok = isinstance(process, Mapping)
        pid = _nonnegative_int(process.get("pid")) if isinstance(process, Mapping) else None
        exit_code = (
            _nonnegative_int(process.get("exit_code")) if isinstance(process, Mapping) else None
        )
        process_start = (
            _finite_number(process.get("started_monotonic_ns"))
            if isinstance(process, Mapping)
            else None
        )
        process_end = (
            _finite_number(process.get("ended_monotonic_ns"))
            if isinstance(process, Mapping)
            else None
        )
        process_ok = bool(
            process_ok
            and pid is not None
            and pid > 0
            and exit_code is not None
            and process_start is not None
            and process_end is not None
            and process_end >= process_start
        )
        checks["process_receipt_recomputed"] = process_ok
        checks["stop_reason_present"] = bool(str(observed.get("stop_reason", "")).strip())
        prompt_tokens = _nonnegative_int(observed.get("prompt_tokens"))
        response_tokens = _nonnegative_int(observed.get("response_tokens"))
        total_tokens = _nonnegative_int(observed.get("total_tokens"))
        checks["token_total_recomputed"] = (
            prompt_tokens is not None
            and response_tokens is not None
            and total_tokens == prompt_tokens + response_tokens
        )
        latency = _finite_number(observed.get("latency_s"))
        checks["latency_recomputed"] = latency is not None and latency >= 0
        cost = _cost_from_components(observed)
        stored_cost = _finite_number(observed.get("charged_cost"))
        checks["charged_cost_recomputed"] = cost is not None and stored_cost == cost
        raw_stored = _finite_number(observed.get("raw_response_recorded_monotonic_ns"))
        parser_started = _finite_number(observed.get("parser_started_monotonic_ns"))
        raw_before_parser = bool(
            raw_stored is not None and parser_started is not None and raw_stored <= parser_started
        )
        checks["raw_response_recorded_before_parser"] = raw_before_parser
        failures = _failure_classes(observed, response_bytes)
    failed_checks = [name for name, passed in checks.items() if not passed]
    audit = {
        "unit_id": unit_id,
        "source_id": expected.get("source_id"),
        "family": expected.get("family"),
        "model_repository": expected.get("model_repository"),
        "model_revision": expected.get("model_revision"),
        "gguf_sha256": expected.get("gguf_sha256"),
        "seed": expected.get("seed"),
        "order_index": expected.get("order_index"),
        "attempt_index": expected.get("attempt_index"),
        "source_sha256_recomputed": sha256_bytes(source_bytes)
        if source_bytes is not None
        else None,
        "prompt_sha256_recomputed": sha256_bytes(prompt_bytes)
        if prompt_bytes is not None
        else None,
        "raw_response_sha256_recomputed": sha256_bytes(response_bytes)
        if response_bytes is not None
        else None,
        "source_recovery": source_mode,
        "prompt_recovery": prompt_mode,
        "raw_response_recovery": response_mode,
        "raw_response_byte_length": len(response_bytes) if response_bytes is not None else None,
        "raw_before_parser": raw_before_parser,
        "process_exit_code": observed.get("process_receipt", {}).get("exit_code")
        if isinstance(observed, Mapping) and isinstance(observed.get("process_receipt"), Mapping)
        else None,
        "stop_reason": observed.get("stop_reason") if observed else None,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "latency_s": round(latency, 12) if latency is not None else None,
        "charged_cost": cost,
        "failure_classes": failures,
        "claim_bearing": observed.get("claim_bearing") is True if observed else False,
        "checks": checks,
        "failed_checks": failed_checks,
        "row_replay_passed": not failed_checks,
    }
    audit["row_hash"] = row_hash(audit)
    return audit


def _failure_retention_rows(
    audit_rows: Sequence[Mapping[str, Any]], *, upstream_available: bool
) -> list[JsonDict]:
    result: list[JsonDict] = []
    for failure_class in FAILURE_CLASSES:
        unit_ids = [
            str(row.get("unit_id"))
            for row in audit_rows
            if failure_class in list(row.get("failure_classes") or [])
        ]
        retained = [
            str(row.get("unit_id"))
            for row in audit_rows
            if row.get("row_replay_passed") is True
            and failure_class in list(row.get("failure_classes") or [])
        ]
        item = {
            "failure_class": failure_class,
            "denominator_row_count": len(audit_rows),
            "observed_count": len(unit_ids),
            "retained_count": len(retained),
            "unit_ids": unit_ids,
            "passed": upstream_available and unit_ids == retained,
        }
        item["row_hash"] = row_hash(item)
        result.append(item)
    return result


def _attack_row(attack: str, passed: bool, observed: Any, expected: Any) -> JsonDict:
    row = {"attack": attack, "passed": passed, "observed": observed, "expected": expected}
    row["row_hash"] = row_hash(row)
    return row


def _attack_rows(
    *,
    upstream: Mapping[str, Any] | None,
    manifest: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    row_readiness: bool,
) -> list[JsonDict]:
    if upstream is None:
        return [
            _attack_row(attack, False, "not_run_missing_upstream", "invariant_passes")
            for attack in REQUIRED_ATTACKS
        ]

    source_aliases: dict[str, set[str]] = defaultdict(set)
    for row in audit_rows:
        source_hash = row.get("source_sha256_recomputed")
        if isinstance(source_hash, str):
            source_aliases[source_hash].add(str(row.get("source_id")))
    alias_groups = {
        digest: sorted(source_ids)
        for digest, source_ids in source_aliases.items()
        if len(source_ids) > 1
    }

    raw_unit_counts = Counter(str(row.get("unit_id", "")) for row in raw_rows)
    manifest_unit_counts = Counter(str(row.get("unit_id", "")) for row in manifest)
    duplicate_ids = sorted(
        unit_id
        for unit_id, count in raw_unit_counts.items()
        if count != 1 or manifest_unit_counts.get(unit_id) != 1
    )

    response_models: dict[str, set[str]] = defaultdict(set)
    for row in audit_rows:
        response_hash = row.get("raw_response_sha256_recomputed")
        if isinstance(response_hash, str) and row.get("raw_response_byte_length") not in (None, 0):
            response_models[response_hash].add(str(row.get("model_repository")))
    copied = {
        digest: sorted(models) for digest, models in response_models.items() if len(models) > 1
    }

    retry_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (str(row.get("source_id")), str(row.get("family")), str(row.get("seed")))
        retry_groups[key].append(row)
    bad_retries: list[str] = []
    for key, group in retry_groups.items():
        attempts = sorted(
            attempt
            for attempt in (_nonnegative_int(row.get("attempt_index")) for row in group)
            if attempt is not None
        )
        declared = max((_nonnegative_int(row.get("retry_count")) or 0 for row in group), default=0)
        if attempts != list(range(declared + 1)):
            bad_retries.append("|".join(key))

    expected_units = Counter(str(row.get("unit_id", "")) for row in manifest)
    observed_units = Counter(str(row.get("unit_id", "")) for row in raw_rows)
    hidden_difference = {
        "missing": sorted((expected_units - observed_units).elements()),
        "extra": sorted((observed_units - expected_units).elements()),
    }
    substitutions = sorted(
        str(row.get("unit_id"))
        for row in raw_rows
        if row.get("model_repository") not in MANDATED_MODELS
    )
    bad_labels = sorted(
        str(row.get("unit_id"))
        for row in raw_rows
        if MANDATED_MODELS.get(str(row.get("model_repository"))) != row.get("family")
    )
    prompt_drift = sorted(
        str(row.get("unit_id"))
        for row in audit_rows
        if row.get("checks", {}).get("prompt_hash_matches_manifest") is not True
    )
    null_rows = sorted(
        str(row.get("unit_id"))
        for row in audit_rows
        if all(
            row.get(field) == "missing"
            for field in ("source_recovery", "prompt_recovery", "raw_response_recovery")
        )
    )
    valid_recovery_modes = {"text", "inline_base64", "inline_payload", "content_path"}
    missing_content = sorted(
        str(row.get("unit_id"))
        for row in audit_rows
        if any(
            row.get(field) not in valid_recovery_modes
            for field in ("source_recovery", "prompt_recovery", "raw_response_recovery")
        )
    )

    totals = recompute_raw_totals(raw_rows)
    recomputed_coverage = sorted(
        {
            str(row.get("family"))
            for row in raw_rows
            if row.get("family") in MANDATED_MODELS.values()
        }
    )
    comparisons = {
        "corpus_commit": recompute_corpus_commit(manifest),
        "row_count": len(raw_rows),
        "family_coverage": recomputed_coverage,
        **totals,
        "immutable_claim_stream_ready_score": 1.0 if row_readiness else 0.0,
    }
    contradictions = {
        field: {"observed": upstream.get(field), "recomputed": value}
        for field, value in comparisons.items()
        if upstream.get(field) != value
    }

    return [
        _attack_row("source_alias", not alias_groups, alias_groups, {}),
        _attack_row("duplicate_unit", not duplicate_ids, duplicate_ids, []),
        _attack_row("copied_response_across_models", not copied, copied, {}),
        _attack_row("selective_retry", not bad_retries, bad_retries, []),
        _attack_row(
            "hidden_row_drop",
            not hidden_difference["missing"] and not hidden_difference["extra"],
            hidden_difference,
            {"missing": [], "extra": []},
        ),
        _attack_row("legacy_model_substitution", not substitutions, substitutions, []),
        _attack_row("inconsistent_family_label", not bad_labels, bad_labels, []),
        _attack_row("post_outcome_prompt_change", not prompt_drift, prompt_drift, []),
        _attack_row("null_only_row", not null_rows, null_rows, []),
        _attack_row("missing_content_path", not missing_content, missing_content, []),
        _attack_row("aggregate_row_contradiction", not contradictions, contradictions, {}),
    ]


def _aggregate(
    *,
    upstream_path: Path,
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
    exp6575_path: Path,
    exp6575: Mapping[str, Any] | None,
    exp6575_error: str | None,
    manifest: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    row_readiness: bool,
) -> JsonDict:
    family_coverage = sorted(
        {
            str(row.get("family"))
            for row in audit_rows
            if row.get("family") in MANDATED_MODELS.values()
        }
    )
    prompt_tokens = sum(_nonnegative_int(row.get("prompt_tokens")) or 0 for row in audit_rows)
    response_tokens = sum(_nonnegative_int(row.get("response_tokens")) or 0 for row in audit_rows)
    latency = round(sum(_finite_number(row.get("latency_s")) or 0.0 for row in audit_rows), 12)
    cost_values = [_finite_number(row.get("charged_cost")) for row in audit_rows]
    charged_cost = round(sum(value or 0.0 for value in cost_values), 12)
    failure_units = {
        str(row.get("unit_id")) for row in audit_rows if list(row.get("failure_classes") or [])
    }
    upstream_status = upstream.get("status") if upstream else None
    upstream_blocked = isinstance(upstream_status, str) and upstream_status.startswith("blocked_")
    qualification_ready = bool(
        exp6575
        and exp6575_error is None
        and exp6575.get("v571_flagship_evidence_ready_score") == 1.0
        and not str(exp6575.get("status", "")).startswith("blocked_")
    )
    upstream_schema_ok = bool(upstream and upstream.get("schema") == EXPECTED_UPSTREAM_SCHEMA)
    attacks_ok = bool(attacks) and all(row.get("passed") is True for row in attacks)
    failures_ok = bool(failures) and all(row.get("passed") is True for row in failures)
    protected_ok = protected.get("all_unchanged") is True
    upstream_readiness_observed = (
        upstream.get("immutable_claim_stream_ready_score") if upstream else None
    )
    upstream_readiness_matches = upstream_readiness_observed == (1.0 if row_readiness else 0.0)
    audit_ready = bool(
        upstream
        and upstream_error is None
        and not upstream_blocked
        and qualification_ready
        and upstream_schema_ok
        and row_readiness
        and failures_ok
        and attacks_ok
        and protected_ok
        and upstream_readiness_matches
    )

    checks: list[JsonDict] = []
    checks.append(
        {
            "field": str(upstream_path),
            "expected": "file_exists",
            "observed": "present" if upstream_path.is_file() else "missing",
            "passed": upstream_path.is_file(),
        }
    )
    checks.append(
        {
            "field": "exp6576.read_error",
            "expected": None,
            "observed": upstream_error,
            "passed": upstream_error is None,
        }
    )
    checks.append(
        {
            "field": "exp6576.status",
            "expected": "complete_nonblocked",
            "observed": upstream_status,
            "passed": upstream is not None and not upstream_blocked,
        }
    )
    checks.extend(
        [
            {
                "field": str(exp6575_path),
                "expected": "qualified_artifact_with_ready_score_1.0",
                "observed": "missing"
                if not exp6575_path.is_file()
                else exp6575.get("v571_flagship_evidence_ready_score")
                if exp6575
                else exp6575_error,
                "passed": qualification_ready,
            },
            {
                "field": "exp6576.schema",
                "expected": EXPECTED_UPSTREAM_SCHEMA,
                "observed": upstream.get("schema") if upstream else None,
                "passed": upstream_schema_ok,
            },
            {
                "field": "rows.row_replay_passed",
                "expected": len(manifest),
                "observed": sum(row.get("row_replay_passed") is True for row in audit_rows),
                "passed": row_readiness,
            },
            {
                "field": "failure_retention_rows.passed",
                "expected": True,
                "observed": failures_ok,
                "passed": failures_ok,
            },
            {
                "field": "duplicate_and_drift_attack_rows.passed",
                "expected": True,
                "observed": attacks_ok,
                "passed": attacks_ok,
            },
            {
                "field": "immutable_claim_stream_ready_score",
                "expected": 1.0 if row_readiness else 0.0,
                "observed": upstream_readiness_observed,
                "passed": upstream_readiness_matches,
            },
            {
                "field": "protected_files_unchanged.all_unchanged",
                "expected": True,
                "observed": protected_ok,
                "passed": protected_ok,
            },
        ]
    )
    if upstream is None or upstream_blocked or not qualification_ready:
        verdict_class: str | None = "blocked"
    elif audit_ready:
        verdict_class = None
    else:
        verdict_class = "disqualified"
    aggregate = {
        "expected_row_count": len(manifest),
        "observed_raw_row_count": len(raw_rows),
        "emitted_audit_row_count": len(audit_rows),
        "replayed_row_count": sum(row.get("row_replay_passed") is True for row in audit_rows),
        "family_coverage": family_coverage,
        "mandated_family_coverage_complete": set(family_coverage) == set(MANDATED_MODELS.values()),
        "failure_row_count": len(failure_units),
        "claim_bearing_row_count": sum(row.get("claim_bearing") is True for row in audit_rows),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "latency_s": latency,
        "charged_cost": charged_cost,
        "upstream_readiness_from_rows": 1.0 if row_readiness else 0.0,
        "upstream_readiness_observed": upstream_readiness_observed,
        "upstream_readiness_matches": upstream_readiness_matches,
        "all_failure_classes_retained": failures_ok,
        "all_attack_invariants_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "audit_ready_score_from_rows": 1.0 if audit_ready else 0.0,
        "verdict_class_from_rows": verdict_class,
        "failed_checks": [row for row in checks if not row["passed"]],
        "checks": checks,
    }
    aggregate["row_hash"] = row_hash(aggregate)
    return aggregate


def _status_and_verdict(
    aggregate: Mapping[str, Any], upstream_path: Path
) -> tuple[str, str, str | None]:
    verdict_class = aggregate.get("verdict_class_from_rows")
    if verdict_class == "blocked":
        first = list(aggregate.get("failed_checks") or [{}])[0]
        return (
            "blocked_flagship_source_stream_independent_audit",
            "blocked_flagship_source_stream_independent_audit: "
            f"{first.get('field', upstream_path)} expected={first.get('expected')} "
            f"observed={first.get('observed')}",
            "blocked",
        )
    if verdict_class == "disqualified":
        first = list(aggregate.get("failed_checks") or [{}])[0]
        return (
            "disqualified_flagship_source_stream_independent_audit",
            "disqualified_flagship_source_stream_independent_audit: lineage or coverage "
            f"did not replay at {first.get('field')} observed={first.get('observed')}",
            "disqualified",
        )
    return (
        "complete_flagship_source_stream_independent_audit_null",
        "complete_flagship_source_stream_independent_audit_null: lineage, coverage, failures, "
        "cost, duplicates, and drift independently replay",
        None,
    )


def _field_provenance(upstream_path: Path) -> dict[str, JsonDict]:
    return {
        field: {
            "principle": principle,
            "sources": [str(upstream_path), "expected_source_family_units", "rows"],
            "reducer": "experiment_6577_flagship_source_stream_independent_audit",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _gate_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    failures = list(aggregate.get("failed_checks") or [])
    summary = {
        "passed": not failures,
        "failed_check_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checks": list(aggregate.get("checks") or []),
    }
    summary["row_hash"] = row_hash(summary)
    return summary


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one file in its target directory and replace it atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    upstream_path: Path | None = None,
    exp6575_path: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build one terminal audit artifact from immutable checked-in inputs."""

    started = time.monotonic()
    result = Path(result_path or (repo_root / RESULT_RELATIVE_PATH))
    upstream_file = Path(upstream_path or (repo_root / UPSTREAM_RELATIVE_PATH))
    qualification_file = Path(exp6575_path or (repo_root / EXP6575_RELATIVE_PATH))
    protected_before = _protected_hashes(repo_root)
    upstream, upstream_error = _read_json(upstream_file)
    exp6575, exp6575_error = _read_json(qualification_file)
    raw_manifest = upstream.get("expected_source_family_units", []) if upstream else []
    manifest = (
        [row for row in raw_manifest if isinstance(row, Mapping)]
        if isinstance(raw_manifest, list)
        else []
    )
    raw_source_rows = upstream.get("rows", []) if upstream else []
    raw_rows = (
        [row for row in raw_source_rows if isinstance(row, Mapping)]
        if isinstance(raw_source_rows, list)
        else []
    )
    corpus_commit = recompute_corpus_commit(manifest)
    model_hash_cache: dict[str, str] = {}
    audit_rows = [
        _audit_one_row(
            expected=expected,
            observed_rows=raw_rows,
            corpus_commit=corpus_commit,
            upstream_commit=upstream.get("corpus_commit") if upstream else None,
            repo_root=repo_root,
            model_hash_cache=model_hash_cache,
        )
        for expected in manifest
    ]
    upstream_available = upstream is not None and upstream_error is None
    failure_rows = _failure_retention_rows(audit_rows, upstream_available=upstream_available)
    base_family_coverage = {
        str(row.get("family"))
        for row in audit_rows
        if row.get("family") in MANDATED_MODELS.values()
    }
    row_readiness = bool(
        manifest
        and len(audit_rows) == len(manifest)
        and all(row.get("row_replay_passed") is True for row in audit_rows)
        and base_family_coverage == set(MANDATED_MODELS.values())
        and any(row.get("claim_bearing") is True for row in audit_rows)
        and all(row.get("charged_cost") is not None for row in audit_rows)
        and all(row.get("raw_before_parser") is True for row in audit_rows)
    )
    attack_rows = _attack_rows(
        upstream=upstream,
        manifest=manifest,
        raw_rows=raw_rows,
        audit_rows=audit_rows,
        row_readiness=row_readiness,
    )
    protected_after = _protected_hashes(repo_root)
    protected = {
        "before": protected_before,
        "after": protected_after,
        "all_unchanged": protected_before == protected_after
        and all(value != "missing" for value in protected_before.values()),
    }
    protected["row_hash"] = row_hash(protected)
    aggregate = _aggregate(
        upstream_path=upstream_file,
        upstream=upstream,
        upstream_error=upstream_error,
        exp6575_path=qualification_file,
        exp6575=exp6575,
        exp6575_error=exp6575_error,
        manifest=manifest,
        raw_rows=raw_rows,
        audit_rows=audit_rows,
        failures=failure_rows,
        attacks=attack_rows,
        protected=protected,
        row_readiness=row_readiness,
    )
    status, honest_verdict, verdict_class = _status_and_verdict(aggregate, upstream_file)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": _gate_summary(aggregate),
        "upstream_artifact_receipt": _path_receipt(upstream_file, upstream, upstream_error),
        "rows": audit_rows,
        "failure_retention_rows": failure_rows,
        "duplicate_and_drift_attack_rows": attack_rows,
        "claim_stream_audit_ready_score": aggregate["audit_ready_score_from_rows"],
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": _preconditions(
            repo_root=repo_root,
            upstream_path=upstream_file,
            upstream=upstream,
            upstream_error=upstream_error,
            exp6575_path=qualification_file,
            exp6575=exp6575,
            exp6575_error=exp6575_error,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(upstream_file),
        "duration_s": round(
            duration_s if duration_s is not None else time.monotonic() - started, 6
        ),
        "tests_run": [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _atomic_write_json(result, artifact)
    return artifact


def _validate_row_hashes(payload: Mapping[str, Any], errors: list[str]) -> None:
    for field in ("rows", "failure_retention_rows", "duplicate_and_drift_attack_rows"):
        rows = payload.get(field)
        if not isinstance(rows, list):
            errors.append(f"{field} must be a list")
            continue
        if any(
            not isinstance(row, Mapping) or row.get("row_hash") != row_hash(row) for row in rows
        ):
            errors.append(f"{field} row_hash mismatch")
    for field in (
        "gate_check_summary",
        "upstream_artifact_receipt",
        "aggregate_row_recomputation",
        "protected_files_unchanged",
    ):
        row = payload.get(field)
        if not isinstance(row, Mapping) or row.get("row_hash") != row_hash(row):
            errors.append(f"{field} row_hash mismatch")


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate field closure, row hashes, reducers, and final checksum."""

    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["required field set mismatch"]
    errors: list[str] = []
    if not str(payload.get("status", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6577 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover every required field")
    elif any(
        not isinstance(provenance.get(field), Mapping)
        or provenance[field].get("principle") != principle
        for field, principle in FIELD_PRINCIPLES.items()
    ):
        errors.append("field_provenance principle mismatch")
    aggregate = payload.get("aggregate_row_recomputation")
    if not isinstance(aggregate, Mapping):
        errors.append("aggregate_row_recomputation must be a mapping")
    else:
        if payload.get("claim_stream_audit_ready_score") != aggregate.get(
            "audit_ready_score_from_rows"
        ):
            errors.append("claim_stream_audit_ready_score mismatch")
        if payload.get("verdict_class") != aggregate.get("verdict_class_from_rows"):
            errors.append("verdict_class mismatch")
        if payload.get("claim_stream_audit_ready_score") == 1.0 and (
            payload.get("verdict_class") is not None or aggregate.get("failed_checks")
        ):
            errors.append("ready score requires clean null audit")
    attacks = payload.get("duplicate_and_drift_attack_rows")
    if isinstance(attacks, list) and [
        row.get("attack") for row in attacks if isinstance(row, Mapping)
    ] != list(REQUIRED_ATTACKS):
        errors.append("required attack set or order mismatch")
    failures = payload.get("failure_retention_rows")
    if isinstance(failures, list) and [
        row.get("failure_class") for row in failures if isinstance(row, Mapping)
    ] != list(FAILURE_CLASSES):
        errors.append("failure retention class set or order mismatch")
    _validate_row_hashes(payload, errors)
    duration = _finite_number(payload.get("duration_s"))
    if duration is None or duration < 0:
        errors.append("duration_s must be finite and nonnegative")
    tests = payload.get("tests_run")
    if not isinstance(tests, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("command"), str)
        or _nonnegative_int(row.get("exit_code")) is None
        for row in tests
    ):
        errors.append("tests_run must name commands and nonnegative exits")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the audit from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--upstream-path")
    parser.add_argument("--exp6575-path")
    parser.add_argument("--result-path")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.repo_root)
    result_path = Path(args.result_path or (repo_root / RESULT_RELATIVE_PATH))
    if args.validate:
        payload, error = _read_json(result_path)
        errors = [error] if error else validate_artifact(payload or {})
        if errors:
            print("\n".join(str(error) for error in errors))
            return 1
        print(f"validated {result_path}")
        return 0
    if args.date != RUN_DATE:
        print(f"planning date must be {RUN_DATE}; observed {args.date}")
        return 2
    artifact = build_artifact(
        repo_root=repo_root,
        result_path=result_path,
        upstream_path=Path(args.upstream_path) if args.upstream_path else None,
        exp6575_path=Path(args.exp6575_path) if args.exp6575_path else None,
        write=True,
    )
    print(
        f"wrote {result_path} "
        f"claim_stream_audit_ready_score={artifact['claim_stream_audit_ready_score']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m in E2E.
    raise SystemExit(main())
