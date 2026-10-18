"""Run the immutable Exp6582 dense Gemma source shard.

The dense-family task uses the tested Exp6581 lifecycle machinery. A scoped
configuration binds every reused reducer and runtime helper to Gemma 4 31B.
The scope restores shared state after each call, so Exp6581 cannot inherit a
Gemma identity. This task measures runtime evidence and source completeness.
It does not make a model-quality or cross-family claim.

Spec: REQ-REPORT-6582 and SCENARIO-REPORT-6582-GATE-BLOCK through
SCENARIO-REPORT-6582-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_6572_content_derived_gguf_metadata_resolver as gguf_fixtures
from carnot import experiment_6581_qwen36_flagship_source_shard as shared
from carnot.inference.gguf_metadata import build_gguf_admission_record


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "exp6582-gemma4-31b-flagship-source-shard"
RUN_DATE = "20260824"
GEMMA_REPOSITORY_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA_ARCHITECTURE = "gemma4"
RANDOM_SEED = 6582
READINESS_FIELD = "gemma4_31b_family_source_shard_ready_score"
INFERENCE_SUBSTRATE = "live_llama_cpp_cuda_one_family_source_shard"
RESULT_RELATIVE_PATH = Path("results/experiment_6582_gemma4_31b_flagship_source_shard.json")
RAW_CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6582_gemma4_31b_flagship_source_shard.raw")
PROTOCOL_RELATIVE_PATH = shared.PROTOCOL_RELATIVE_PATH
SPEC_RELATIVE_PATH = shared.SPEC_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6582_gemma4_31b_flagship_source_shard.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6582_gemma4_31b_flagship_source_shard.py")
PROTECTED_RELATIVE_PATHS = shared.PROTECTED_RELATIVE_PATHS
GATE_CONTRACTS = shared.GATE_CONTRACTS

RECOVERY_TOLERANCE_MB = shared.RECOVERY_TOLERANCE_MB
REQUIRED_NEGATIVE_FIXTURE_IDS = shared.REQUIRED_NEGATIVE_FIXTURE_IDS
REQUIRED_ATTACK_IDS = shared.REQUIRED_ATTACK_IDS
REQUIRED_ARTIFACT_FIELDS = tuple(
    READINESS_FIELD if field == "qwen36_family_source_shard_ready_score" else field
    for field in shared.REQUIRED_ARTIFACT_FIELDS
)

FIELD_PRINCIPLES = {
    "status": "The one-family task closes even on a timeout or gate block.",
    "honest_verdict": "The verdict states runtime and source-shard completeness without a quality claim.",
    "verdict_class": "Evidence readiness uses null, partial, blocked, or disqualified.",
    "gate_check_summary": "A block names the exact same-roadmap field and observed value.",
    "model_specs": "Only the mandated dense Gemma family can satisfy readiness.",
    "model_revision_and_hash_receipt": "Repository, revision, GGUF bytes, architecture, quantization, and tokenizer bind the run.",
    "rows": "Every source unit carries raw output, runtime, failure, token, timing, and cost metrics.",
    "raw_response_receipts": "Raw response bytes precede parsing and preserve all outcomes.",
    "process_and_gpu_receipts": "PID, command, offload, and repeated GPU samples prove live local execution.",
    "checkpoint_receipts": "Periodic content hashes prevent another long no-artifact loss.",
    "parser_diagnostic_rows": "Claim segmentation is diagnostic and cannot filter the source shard.",
    "unload_and_recovery_rows": "The family releases GPU state before the task closes.",
    "attack_rows": "Substitution, stale execution, drift, retries, and row loss fail closed.",
    READINESS_FIELD: "This exact binary field is owned by Exp6582 and consumed by Exp6584.",
    "aggregate_row_recomputation": "Coverage, failures, latency, tokens, and cost derive only from rows.",
    "seeds": "Explicit seeds bind all source-unit generations.",
    "preconditions_checked": "Gate, model, source, resource, CUDA, and protected-file receipts are explicit.",
    "protected_files_unchanged": "The task preserves both protected orchestration files.",
    "inference_substrate": "The live llama.cpp CUDA path selects the correct structural checks.",
    "verifier_is_oracle": "This task creates evidence and makes no verifier-backed utility claim.",
    "field_provenance": "Every field names its raw rows, hashes, and reducer.",
    "duration_s": "Monotonic duration exposes truncated family work.",
    "tests_run": "Named commands, exits, and durations make the shard reproducible.",
    "reproducibility_checksum": "A final hash detects terminal mutation.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6582_gemma4_31b_flagship_source_shard.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6582_gemma4_31b_flagship_source_shard.py "
    "-m pytest tests/python/test_experiment_6582_gemma4_31b_flagship_source_shard.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6582_gemma4_31b_flagship_source_shard.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"


# These helpers are byte- and family-neutral. Aliases keep one canonical hash,
# parsing, cost, gate, and unload implementation for the independent audit.
canonical_json = shared.canonical_json
sha256_bytes = shared.sha256_bytes
sha256_text = shared.sha256_text
sha256_json = shared.sha256_json
sha256_file = shared.sha256_file
load_json = shared.load_json
row_hash = shared.row_hash
artifact_checksum = shared.artifact_checksum
build_gate_receipts = shared.build_gate_receipts
compose_request_bytes = shared.compose_request_bytes
segment_claim_sentences = shared.segment_claim_sentences
classify_raw_response = shared.classify_raw_response
cost_from_components = shared.cost_from_components
build_parser_diagnostic = shared.build_parser_diagnostic
finalize_terminal_row = shared.finalize_terminal_row
unload_checks = shared.unload_checks
build_attack_rows = shared.build_attack_rows


@contextmanager
def _family_configuration() -> Iterator[None]:
    """Bind shared lifecycle code to dense Gemma for one scoped operation."""

    overrides: dict[str, Any] = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "QWEN_REPOSITORY_ID": GEMMA_REPOSITORY_ID,
        "QWEN_ARCHITECTURE": GEMMA_ARCHITECTURE,
        "RANDOM_SEED": RANDOM_SEED,
        "RESULT_RELATIVE_PATH": RESULT_RELATIVE_PATH,
        "RAW_CHECKPOINT_RELATIVE_PATH": RAW_CHECKPOINT_RELATIVE_PATH,
        "MODULE_RELATIVE_PATH": MODULE_RELATIVE_PATH,
        "TEST_RELATIVE_PATH": TEST_RELATIVE_PATH,
        "FOCUSED_TEST_COMMAND": FOCUSED_TEST_COMMAND,
        "COVERAGE_RUN_COMMAND": COVERAGE_RUN_COMMAND,
        "COVERAGE_REPORT_COMMAND": COVERAGE_REPORT_COMMAND,
        "RUFF_CHECK_COMMAND": RUFF_CHECK_COMMAND,
        "RUFF_FORMAT_COMMAND": RUFF_FORMAT_COMMAND,
        "SPEC_COVERAGE_COMMAND": SPEC_COVERAGE_COMMAND,
    }
    original = {name: getattr(shared, name) for name in overrides}
    original_pair = shared.cached_sota_pair

    def dense_cached_pair() -> list[dict] | None:
        """Resolve Qwen plus dense Gemma so the registry receipt includes this family."""

        return original_pair(model_indices=(0, 2))

    for name, value in overrides.items():
        setattr(shared, name, value)
    shared.cached_sota_pair = dense_cached_pair
    try:
        yield
    finally:
        shared.cached_sota_pair = original_pair
        for name, value in original.items():
            setattr(shared, name, value)


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Validate the frozen Exp6580 mapping for the exact dense Gemma task."""

    with _family_configuration():
        errors = shared.validate_frozen_protocol(protocol)
    return [
        "gemma4_31b_family_contract_mismatch" if error == "qwen_family_contract_mismatch" else error
        for error in errors
    ]


def metadata_receipt_passes(receipt: Mapping[str, Any]) -> bool:
    """Recompute dense Gemma content identity and cache provenance."""

    with _family_configuration():
        return shared.metadata_receipt_passes(receipt)


def _wrong_architecture_fixture_row() -> JsonDict:
    """Build a bounded valid Qwen GGUF that cannot satisfy dense Gemma identity."""

    with tempfile.TemporaryDirectory(prefix="exp6582-negative-") as temporary:
        cache_root = Path(temporary)
        blob, trusted = gguf_fixtures._cache_fixture(
            cache_root,
            gguf_fixtures._fixture_gguf(architecture="qwen35moe"),
            repository_id=GEMMA_REPOSITORY_ID,
            filename="wrong-architecture.gguf",
        )
        record = build_gguf_admission_record(
            blob,
            repository_id=GEMMA_REPOSITORY_ID,
            cache_root=cache_root,
            trusted_sha256=trusted,
            expected_architectures={GEMMA_ARCHITECTURE},
        )
        reasons = list(record.get("rejection_reasons", []))
        return {
            "row_type": "negative_fixture",
            "unit_id": "wrong_architecture",
            "expected_admitted": False,
            "observed_admitted": record.get("admitted"),
            "expected_reason": "architecture_mismatch",
            "rejection_reasons": reasons,
            "passed": record.get("admitted") is False and "architecture_mismatch" in reasons,
            "bounded_read_receipt": record.get("content_metadata", {}).get(
                "bounded_read_receipt", {}
            ),
            "record": record,
        }


def build_negative_metadata_fixture_rows() -> list[JsonDict]:
    """Replay four generic malformed fixtures plus wrong architecture."""

    source = {row.get("unit_id"): dict(row) for row in gguf_fixtures.build_negative_fixture_rows()}
    source["wrong_architecture"] = _wrong_architecture_fixture_row()
    return [
        source.get(fixture_id, {"unit_id": fixture_id, "passed": False})
        for fixture_id in REQUIRED_NEGATIVE_FIXTURE_IDS
    ]


def build_raw_terminal_row(**kwargs: Any) -> JsonDict:
    """Build one raw terminal row with the dense family seed and identity."""

    with _family_configuration():
        return shared.build_raw_terminal_row(**kwargs)


def write_raw_checkpoint(checkpoint_dir: Path, raw_row: Mapping[str, Any]) -> JsonDict:
    """Write one raw-before-derived checkpoint under the dense family scope."""

    with _family_configuration():
        return shared.write_raw_checkpoint(checkpoint_dir, raw_row)


def process_and_gpu_checks(receipt: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute fresh-process, one-family residency, and CUDA receipts."""

    with _family_configuration():
        return shared.process_and_gpu_checks(receipt)


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute dense-family readiness from retained rows and receipts."""

    with _family_configuration():
        return shared.recompute_aggregate(payload)


def _field_provenance() -> JsonDict:
    """Name the raw evidence and deterministic reducer for every field."""

    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": ["source_protocol", "rows", "hash receipts", "lifecycle receipts"],
            "reducer": "recompute_aggregate"
            if field in {READINESS_FIELD, "aggregate_row_recomputation"}
            else "direct receipt or deterministic assembly",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _normalize_report(payload: Mapping[str, Any]) -> JsonDict:
    """Replace Qwen-owned labels after shared deterministic assembly."""

    report = dict(payload)
    ready = float(report.pop("qwen36_family_source_shard_ready_score", 0.0) or 0.0)
    report["schema"] = "carnot.experiment_6582_gemma4_31b_flagship_source_shard.v1"
    report["task_id"] = TASK_ID
    report[READINESS_FIELD] = ready
    report["field_provenance"] = _field_provenance()
    if ready == 1.0:
        report["honest_verdict"] = (
            "complete_gemma4_31b_runtime_and_immutable_source_shard_without_quality_claim"
        )
    elif report.get("status") == "partial":
        report["honest_verdict"] = (
            "partial_gemma4_31b_runtime_or_source_shard_incomplete_without_quality_claim"
        )
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def build_report(**kwargs: Any) -> JsonDict:
    """Assemble one terminal dense Gemma artifact and recompute readiness."""

    with _family_configuration():
        payload = shared.build_report(**kwargs)
    return _normalize_report(payload)


def build_blocked_report(
    *,
    gates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    metadata_receipt: Mapping[str, Any],
    negative_fixture_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    reason: str,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Close a structured or environment block without starting the model."""

    report = build_report(
        gates=gates,
        protocol=protocol,
        metadata_receipt=metadata_receipt,
        negative_fixture_rows=negative_fixture_rows,
        rows=[],
        checkpoint_receipts=[],
        parser_diagnostic_rows=[],
        process_receipt={},
        unload_rows=[],
        attack_rows=build_attack_rows(),
        preconditions=preconditions,
        protected=protected,
        duration_s=duration_s,
        tests_run=tests_run,
        run_date=run_date,
    )
    report["status"] = "blocked"
    report["honest_verdict"] = f"blocked_{reason}_without_quality_claim"
    report["verdict_class"] = "blocked"
    report[READINESS_FIELD] = 0.0
    report["aggregate_row_recomputation"]["ready_score"] = 0.0
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Validate one artifact without trusting its stored readiness field."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(report))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if report.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    expected_specs = [
        {"repository_id": GEMMA_REPOSITORY_ID, "expected_architecture": GEMMA_ARCHITECTURE}
    ]
    if report.get("model_specs") != expected_specs:
        errors.append("model_specs_mismatch")
    if set(report.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    aggregate = recompute_aggregate(report)
    if report.get(READINESS_FIELD) != aggregate["ready_score"]:
        errors.append("ready_score_mismatch")
    stored = report.get("aggregate_row_recomputation", {})
    if stored.get("ready_score") != aggregate["ready_score"]:
        errors.append("aggregate_ready_score_mismatch")
    if report.get("verdict_class") is None and aggregate["ready_score"] != 1.0:
        errors.append("null_verdict_without_ready_shard")
    if report.get("verdict_class") == "blocked" and report.get("rows"):
        errors.append("blocked_report_started_rows")
    if report.get("reproducibility_checksum") != artifact_checksum(report):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(path: Path, report: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically replace one same-directory terminal artifact."""

    errors = validate_report(report)
    if errors:
        raise ValueError(";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".exp6582-final-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "byte_count": len(encoded),
        "atomic_replace": True,
    }


def normalize_preconditions(preconditions: Mapping[str, Any]) -> JsonDict:
    """Rename inherited diagnostic keys so every receipt names dense Gemma."""

    replacements = {
        "positive_qwen_metadata": "positive_gemma4_31b_metadata",
        "cached_sota_pair_contains_qwen": "cached_sota_pair_contains_gemma4_31b",
        "fresh_qwen_process": "fresh_gemma4_31b_process",
    }
    result = deepcopy(dict(preconditions))
    checks = result.get("checks", {})
    result["checks"] = {replacements.get(key, key): value for key, value in checks.items()}
    result["failed_preconditions"] = [
        replacements.get(str(name), str(name)) for name in result.get("failed_preconditions", [])
    ]
    return result


def family_task_deadline(protocol: Mapping[str, Any], *, now: float) -> float:
    """Start the frozen model budget after verification and resource checks."""

    timeout_s = float(protocol.get("prompt_seed_budget_contract", {}).get("timeout_s", 4200))
    return now + timeout_s


def _resolve_metadata_receipt() -> JsonDict:  # pragma: no cover - live cache receipt.
    """Resolve and content-bind the exact dense Gemma GGUF."""

    with _family_configuration():
        return shared._resolve_metadata_receipt()


def _checkpoint_tests(repo_root: Path) -> list[JsonDict]:  # pragma: no cover - live checks.
    """Run the focused, coverage, full, lint, format, and spec checks once."""

    commands = (
        (FOCUSED_TEST_COMMAND, 180.0),
        (COVERAGE_RUN_COMMAND, 180.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (RUFF_CHECK_COMMAND, 60.0),
        (RUFF_FORMAT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
    )
    return [shared._run_named_test(command, repo_root, timeout) for command, timeout in commands]


def _collect_preconditions(
    *args: Any, **kwargs: Any
) -> tuple[JsonDict, Path, JsonDict]:  # pragma: no cover - live host receipts.
    """Collect gates, resources, CUDA state, cache identity, and process state."""

    with _family_configuration():
        preconditions, server, initial = shared._collect_preconditions(*args, **kwargs)
    return normalize_preconditions(preconditions), server, initial


def _run_live_shard(
    *args: Any, **kwargs: Any
) -> tuple[  # pragma: no cover - live GPU work.
    list[JsonDict], list[JsonDict], list[JsonDict], JsonDict, list[JsonDict]
]:
    """Run one fresh embedded-tokenizer llama.cpp process for dense Gemma."""

    with _family_configuration():
        return shared._run_live_shard(*args, **kwargs)


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover - live workflow.
    """Run gates, checks, one dense-family shard, cleanup, and atomic output."""

    start = time.monotonic()
    protected_before = shared._hash_protected(repo_root)
    gates = build_gate_receipts(repo_root)
    protocol = load_json(repo_root / PROTOCOL_RELATIVE_PATH)
    metadata = _resolve_metadata_receipt()
    negative_rows = build_negative_metadata_fixture_rows()
    tests_run = _checkpoint_tests(repo_root)
    preconditions, server, _initial = _collect_preconditions(
        repo_root, gates, protocol, metadata, negative_rows, tests_run
    )
    preconditions["protected_file_hashes_before"] = protected_before
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    process_receipt: JsonDict = {}
    unload_rows: list[JsonDict] = []
    task_deadline = family_task_deadline(protocol, now=time.monotonic())
    if preconditions["all_required_preconditions_available"]:
        preconditions["model_process_started"] = True
        rows, checkpoints, diagnostics, process_receipt, unload_rows = _run_live_shard(
            repo_root=repo_root,
            protocol=protocol,
            metadata=metadata,
            server=server,
            selected_gpu=int(preconditions["selected_gpu"]),
            task_deadline=task_deadline,
        )
    protected = shared._compare_protected(protected_before, shared._hash_protected(repo_root))
    preconditions["protected_file_hashes_after"] = {
        row["path"]: row["after_sha256"] for row in protected["rows"]
    }
    if not preconditions["all_required_preconditions_available"]:
        structured_failed = any(row.get("passed") is not True for row in gates)
        reason = "structured_gate_failed" if structured_failed else "precondition_failed"
        artifact = build_blocked_report(
            gates=gates,
            protocol=protocol,
            metadata_receipt=metadata,
            negative_fixture_rows=negative_rows,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
            reason=reason,
            run_date=run_date,
        )
    else:
        artifact = build_report(
            gates=gates,
            protocol=protocol,
            metadata_receipt=metadata,
            negative_fixture_rows=negative_rows,
            rows=rows,
            checkpoint_receipts=checkpoints,
            parser_diagnostic_rows=diagnostics,
            process_receipt=process_receipt,
            unload_rows=unload_rows,
            attack_rows=build_attack_rows(),
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
            run_date=run_date,
        )
    atomic_write_report(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the dense Gemma one-family shard artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_RELATIVE_PATH)
    if args.validate:
        errors = validate_report(load_json(output))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 1 if errors else 0
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                READINESS_FIELD: artifact[READINESS_FIELD],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
