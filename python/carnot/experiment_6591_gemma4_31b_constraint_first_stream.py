"""Run one immutable Gemma constraint-first stream through llama.cpp.

The module rechecks every frozen Exp6587 method byte before model load. It then
runs direct, always-on CFR, and routed CFR arms in one fresh dense Gemma process.
Raw stages reach durable unit checkpoints before the next unit starts. Exact
fixture checks own release, so the artifact makes no CFR benefit claim.

Spec refs: REQ-REPORT-6591 and SCENARIO-REPORT-6591-FROZEN through
SCENARIO-REPORT-6591-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any

from carnot import experiment_6582_gemma4_31b_flagship_source_shard as gemma_shard
from carnot import experiment_6590_qwen36_constraint_first_stream as shared
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
TASK_ID = "exp6591-gemma4-31b-constraint-first-stream"
GEMMA_REPOSITORY_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA_ARCHITECTURE = "gemma4"
INFERENCE_SUBSTRATE = "fresh_local_gemma4_31b_gguf_cfr_inference"
EXACT_CHECKER_NAME = shared.EXACT_CHECKER_NAME

RESULT_RELATIVE_PATH = Path("results/experiment_6591_gemma4_31b_constraint_first_stream.json")
CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6591_gemma4_31b_constraint_first_stream.raw")
METHOD_RELATIVE_PATH = shared.METHOD_RELATIVE_PATH
LAUNCH_RELATIVE_PATH = shared.LAUNCH_RELATIVE_PATH
SPEC_RELATIVE_PATH = shared.SPEC_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6591_gemma4_31b_constraint_first_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6591_gemma4_31b_constraint_first_stream.py")
PROTECTED_RELATIVE_PATHS = shared.PROTECTED_RELATIVE_PATHS

ARM_ORDER = shared.ARM_ORDER
STAGE_ORDER = shared.STAGE_ORDER
FAILURE_CLASSES = shared.FAILURE_CLASSES
REQUIRED_ATTACK_IDS = shared.REQUIRED_ATTACK_IDS

MODEL_SPECS = [
    {
        "name": "Gemma4-31B-it",
        "repository_id": GEMMA_REPOSITORY_ID,
        "expected_architecture": GEMMA_ARCHITECTURE,
        "quantization": "Q4_K_M",
        "headline_eligible": True,
    }
]

LOAD_TIMEOUT_S = shared.LOAD_TIMEOUT_S
PER_GENERATION_TIMEOUT_S = shared.PER_GENERATION_TIMEOUT_S
TASK_TIMEOUT_S = shared.TASK_TIMEOUT_S
SHUTDOWN_TIMEOUT_S = shared.SHUTDOWN_TIMEOUT_S
RECOVERY_TIMEOUT_S = shared.RECOVERY_TIMEOUT_S
RECOVERY_TOLERANCE_MB = shared.RECOVERY_TOLERANCE_MB
GPU_LOAD_DELTA_MIN_MB = shared.GPU_LOAD_DELTA_MIN_MB
TELEMETRY_INTERVAL_S = shared.TELEMETRY_INTERVAL_S
CONTEXT_SIZE = shared.CONTEXT_SIZE
LAUNCH_MAX_OUTPUT_TOKENS = shared.LAUNCH_MAX_OUTPUT_TOKENS

READINESS_FIELD = "gemma31_cfr_rows_ready_score"
REQUIRED_ARTIFACT_FIELDS = tuple(
    READINESS_FIELD if field == "qwen_cfr_rows_ready_score" else field
    for field in shared.REQUIRED_ARTIFACT_FIELDS
)

FIELD_PRINCIPLES = {
    "status": "The stream ends as complete rows, bounded partial rows, or a named precondition block.",
    "honest_verdict": "The verdict reports Gemma row completeness without claiming CFR benefit.",
    "verdict_class": "A complete source stream is null evidence infrastructure.",
    "gate_check_summary": "A block names the exact gate, cache, resource, drift, or runtime value.",
    "per_unit_rows": "Every source unit and arm carries raw stages, exact results, failures, tokens, and latency.",
    "model_spec_and_identity": "The mandated Gemma GGUF spec and content-derived local file identity bind inference.",
    "prompt_source_router_hashes": "The Exp6587 byte-frozen method cannot drift by family or outcome.",
    "raw_stage_receipts": "Direct, Stage 1, and Stage 2 bytes remain separate and immutable.",
    "exact_checker_receipts": "Whitelisted executable checks, not the model, decide validity.",
    "checkpoint_receipts": "Each completed unit survives later timeout or process failure.",
    "gpu_process_receipts": "Ownership, offload, memory, utilization, and clean unload bind the local run.",
    "failure_rows": "Timeouts, parse failures, unsupported constraints, contradictions, and exact rejection remain evidence.",
    READINESS_FIELD: "This exact binary field gates the independent comparison.",
    "attack_rows": "Drift, leakage, substitution, dropped failures, and aggregate-only claims fail closed.",
    "preconditions_checked": "Gates, hashes, cache, resources, ownership, budgets, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task declares fresh local Gemma GGUF inference through llama.cpp.",
    "verifier_is_oracle": "Exact checks define row validity, so later exact wins are circular-positive.",
    "field_provenance": "Every field names source rows, raw bytes, process receipts, and reducer code.",
    "duration_s": "Monotonic duration exposes smoke-only or truncated execution.",
    "tests_run": "Named commands, exits, durations, and focused scope make validation reproducible.",
    "reproducibility_checksum": "A final content hash protects the immutable stream.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6591_gemma4_31b_constraint_first_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6591_gemma4_31b_constraint_first_stream.py "
    "-m pytest tests/python/test_experiment_6591_gemma4_31b_constraint_first_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6591_gemma4_31b_constraint_first_stream.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"

canonical_json = shared.canonical_json
sha256_bytes = shared.sha256_bytes
sha256_text = shared.sha256_text
sha256_json = shared.sha256_json
sha256_file = shared.sha256_file
load_json = shared.load_json
row_hash = shared.row_hash
artifact_checksum = shared.artifact_checksum
build_gate_receipt = shared.build_gate_receipt
build_frozen_hash_receipt = shared.build_frozen_hash_receipt
empty_failure_flags = shared.empty_failure_flags
make_stage_receipt = shared.make_stage_receipt
_decode_stage = shared._decode_stage
parse_stage1_proposals = shared.parse_stage1_proposals
cost_from_stages = shared.cost_from_stages
_stage1_leaks_answer = shared._stage1_leaks_answer
build_arm_row = shared.build_arm_row
finalize_unit_row = shared.finalize_unit_row
build_raw_stage_receipts = shared.build_raw_stage_receipts
build_exact_checker_receipts = shared.build_exact_checker_receipts
build_failure_rows = shared.build_failure_rows
_stage_authentic = shared._stage_authentic
_arm_authentic = shared._arm_authentic
_unit_authentic = shared._unit_authentic
_checkpoint_prefixes_ready = shared._checkpoint_prefixes_ready
_gate_summary = shared._gate_summary


@contextmanager
def _family_configuration() -> Iterator[None]:
    """Bind the shared row engine to dense Gemma for one scoped operation."""

    overrides: dict[str, Any] = {
        "RUN_DATE": RUN_DATE,
        "TASK_ID": TASK_ID,
        "QWEN_REPOSITORY_ID": GEMMA_REPOSITORY_ID,
        "QWEN_ARCHITECTURE": GEMMA_ARCHITECTURE,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "RESULT_RELATIVE_PATH": RESULT_RELATIVE_PATH,
        "CHECKPOINT_RELATIVE_PATH": CHECKPOINT_RELATIVE_PATH,
        "MODULE_RELATIVE_PATH": MODULE_RELATIVE_PATH,
        "TEST_RELATIVE_PATH": TEST_RELATIVE_PATH,
        "MODEL_SPECS": MODEL_SPECS,
        "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
        "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
        "FOCUSED_TEST_COMMAND": FOCUSED_TEST_COMMAND,
        "COVERAGE_RUN_COMMAND": COVERAGE_RUN_COMMAND,
        "COVERAGE_REPORT_COMMAND": COVERAGE_REPORT_COMMAND,
        "RUFF_CHECK_COMMAND": RUFF_CHECK_COMMAND,
        "RUFF_FORMAT_COMMAND": RUFF_FORMAT_COMMAND,
        "SPEC_COVERAGE_COMMAND": SPEC_COVERAGE_COMMAND,
        "write_unit_checkpoint": write_unit_checkpoint,
    }
    original = {name: getattr(shared, name) for name in overrides}
    for name, value in overrides.items():
        setattr(shared, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(shared, name, value)


def write_unit_checkpoint(
    checkpoint_dir: Path, completed_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Durably replace one Gemma completed-prefix checkpoint after a full unit."""

    rows = [dict(row) for row in completed_rows]
    payload = {
        "schema": "carnot.exp6591.completed_unit_prefix.v1",
        "completed_unit_count": len(rows),
        "completed_unit_ids": [row.get("unit_id") for row in rows],
        "completed_unit_rows": rows,
        "prefix_hash": sha256_json(rows),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(rows[-1].get("unit_id", "empty")).removeprefix("sha256:")[:12] if rows else "empty"
    target = checkpoint_dir / f"unit-{len(rows):02d}-{suffix}.json"
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_dir, prefix=".exp6591-unit-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "completed_unit_count": len(rows),
        "completed_unit_ids": payload["completed_unit_ids"],
        "completed_unit_row_hashes": [row.get("row_hash") for row in rows],
        "prefix_hash": payload["prefix_hash"],
        "absolute_path": str(target.resolve()),
        "checkpoint_sha256": sha256_file(target),
        "byte_count": len(encoded),
        "written_monotonic_ns": time.monotonic_ns(),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def model_identity_checks(receipt: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute dense Gemma cache, tokenizer, and CUDA-build identity."""

    with _family_configuration():
        return shared.model_identity_checks(receipt)


def process_lifecycle_checks(receipts: Mapping[str, Any]) -> dict[str, bool]:
    """Recheck one owned dense Gemma process, offload, and clean unload."""

    with _family_configuration():
        return shared.process_lifecycle_checks(receipts)


def stream_reducer(payload: Mapping[str, Any], *, require_attack_rows: bool = True) -> JsonDict:
    """Recompute binary Gemma readiness from independently checkable receipts."""

    with _family_configuration():
        reduction = shared.stream_reducer(payload, require_attack_rows=require_attack_rows)
    reduction["reducer"] = str(reduction["reducer"]).replace("Qwen", "Gemma")
    return reduction


def build_attack_rows(base_payload: Mapping[str, Any]) -> list[JsonDict]:
    """Apply each Gemma-specific mutation and retain the reducer's zero score."""

    mutations = {
        "prompt_drift": lambda value: value["prompt_source_router_hashes"]["checks"].update(
            prompt_hashes=False
        ),
        "post_outcome_unit_loss": lambda value: value["per_unit_rows"].pop(),
        "stage_overwrite": lambda value: value["per_unit_rows"][0]["arms"][1]["raw_stages"][
            "stage2"
        ].update(raw_byte_count=-1),
        "answer_leakage": lambda value: value["per_unit_rows"][0]["arms"][1]["failure"].update(
            stage1_answer_leakage=True, any=True
        ),
        "uncharged_stage1": lambda value: value["per_unit_rows"][0]["arms"][1]["tokens"].update(
            stage1_charged=False
        ),
        "family_label_substitution": lambda value: value["model_spec_and_identity"][
            "identity"
        ].update(repository_id="unsloth/Qwen3.6-35B-A3B-GGUF"),
        "legacy_model_substitution": lambda value: value["model_spec_and_identity"]["model_specs"][
            0
        ].update(repository_id="google/gemma-4-E4B-it"),
        "aggregate_only_output": lambda value: value.update(per_unit_rows=[]),
        "ready_score_with_missing_rows": lambda value: value["per_unit_rows"].pop(),
    }
    base_ready = stream_reducer(base_payload, require_attack_rows=False)["ready_score"]
    rows = []
    for attack_id in REQUIRED_ATTACK_IDS:
        candidate = deepcopy(base_payload)
        if base_ready == 1.0:
            mutations[attack_id](candidate)
        score = stream_reducer(candidate, require_attack_rows=False)["ready_score"]
        rows.append(
            {
                "attack_id": attack_id,
                "candidate_ready_score": score,
                "expected_ready_score": 0.0,
                "passed": score == 0.0,
                "reducer": "stream_reducer(require_attack_rows=False)",
            }
        )
    return rows


def _field_provenance() -> dict[str, JsonDict]:
    """Name raw receipts and deterministic reducers for every required field."""

    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": [
                "Exp6587 frozen method bytes",
                "per_unit_rows raw stages",
                "exact checker rows",
                "checkpoint and GPU lifecycle receipts",
            ],
            "reducer": "stream_reducer"
            if field in {READINESS_FIELD, "failure_rows", "attack_rows"}
            else "direct receipt or deterministic assembly",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    *,
    run_date: str,
    gate_receipt: Mapping[str, Any],
    frozen_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    gpu_receipts: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one terminal Gemma stream and derive its completeness score."""

    payload: JsonDict = {
        "status": "assembling",
        "honest_verdict": "partial_gemma31_cfr_rows_incomplete_without_benefit_claim",
        "verdict_class": "partial",
        "gate_check_summary": _gate_summary(gate_receipt, frozen_receipt, preconditions),
        "per_unit_rows": [dict(row) for row in per_unit_rows],
        "model_spec_and_identity": dict(model_identity),
        "prompt_source_router_hashes": dict(frozen_receipt),
        "raw_stage_receipts": build_raw_stage_receipts(per_unit_rows),
        "exact_checker_receipts": build_exact_checker_receipts(per_unit_rows),
        "checkpoint_receipts": [dict(row) for row in checkpoint_receipts],
        "gpu_process_receipts": dict(gpu_receipts),
        "failure_rows": build_failure_rows(per_unit_rows),
        READINESS_FIELD: 0.0,
        "attack_rows": [],
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    payload["attack_rows"] = build_attack_rows(payload)
    reduction = stream_reducer(payload)
    ready = reduction["ready_score"]
    payload[READINESS_FIELD] = ready
    payload["stream_recomputation"] = reduction
    payload["planning_date"] = run_date
    payload["task_id"] = TASK_ID
    if ready == 1.0:
        payload["status"] = "complete"
        payload["honest_verdict"] = (
            "complete: every frozen Gemma CFR unit, arm, raw stage, exact check, checkpoint, "
            "cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made"
        )
        payload["verdict_class"] = None
    elif protected.get("all_unchanged") is not True:
        payload["status"] = "disqualified"
        payload["honest_verdict"] = "disqualified_protected_file_changed_without_benefit_claim"
        payload["verdict_class"] = "disqualified"
    elif not per_unit_rows and payload["gate_check_summary"]["first_failure"] is not None:
        payload["status"] = "blocked"
        name = str(payload["gate_check_summary"]["first_failure"]["check"])
        payload["honest_verdict"] = f"blocked_{name}_without_benefit_claim"
        payload["verdict_class"] = "blocked"
    else:
        payload["status"] = "partial"
        payload["honest_verdict"] = "partial_gemma31_cfr_rows_incomplete_without_benefit_claim"
        payload["verdict_class"] = "partial"
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_blocked_report(
    *,
    run_date: str,
    gate_receipt: Mapping[str, Any],
    frozen_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Close a named precondition block without starting model inference."""

    return build_report(
        run_date=run_date,
        gate_receipt=gate_receipt,
        frozen_receipt=frozen_receipt,
        model_identity=model_identity,
        per_unit_rows=[],
        checkpoint_receipts=[],
        gpu_receipts={},
        preconditions=preconditions,
        protected=protected,
        duration_s=duration_s,
        tests_run=tests_run,
    )


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    """Validate terminal schema, readiness, verdict, and checksum without trust."""

    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    if payload.get("model_spec_and_identity", {}).get("model_specs") != MODEL_SPECS:
        errors.append("model_specs_mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    reduction = stream_reducer(payload)
    if payload.get(READINESS_FIELD) != reduction["ready_score"]:
        errors.append("ready_score_mismatch")
    if payload.get("verdict_class") is None and reduction["ready_score"] != 1.0:
        errors.append("null_verdict_without_ready_stream")
    if payload.get("verdict_class") == "blocked":
        if payload.get("per_unit_rows"):
            errors.append("blocked_report_started_rows")
        if payload.get("gate_check_summary", {}).get("first_failure") is None:
            errors.append("blocked_report_missing_gate_value")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, replace, and directory-sync one terminal artifact."""

    errors = validate_report(payload)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".exp6591-final-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def _checkpoint_tests(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    """Run focused tests, coverage, lint, format, and spec checks once."""

    commands = (
        (FOCUSED_TEST_COMMAND, 180.0),
        (COVERAGE_RUN_COMMAND, 180.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (RUFF_CHECK_COMMAND, 60.0),
        (RUFF_FORMAT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
    )
    return [shared._run_command(text.split(), repo_root, timeout) for text, timeout in commands]


_host_resources = shared._host_resources
_protected_hashes = shared._protected_hashes
_protected_receipt = shared._protected_receipt


def _resolve_model_identity() -> JsonDict:  # pragma: no cover
    """Resolve the dense Gemma GGUF and its content-derived local identity."""

    identity = gemma_shard._resolve_metadata_receipt()  # noqa: SLF001
    pair = cached_sota_pair(model_indices=(0, 2)) or []
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    build = shared.runtime_helpers._llama_cpp_build_receipt(server)  # noqa: SLF001
    return {
        "model_specs": deepcopy(MODEL_SPECS),
        "identity": identity,
        "cached_sota_pair": pair,
        "llama_cpp_build": build,
        "embedded_tokenizer_used": True,
        "auto_tokenizer_used": False,
        "download_performed": False,
    }


def _collect_preconditions(  # pragma: no cover
    *args: Any, **kwargs: Any
) -> tuple[JsonDict, Path, JsonDict]:
    """Collect gates, hashes, resources, cache, CUDA, and process state."""

    with _family_configuration():
        return shared._collect_preconditions(*args, **kwargs)


def _run_live_stream(
    *args: Any, **kwargs: Any
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Run one fresh embedded-tokenizer llama.cpp process for dense Gemma."""

    with _family_configuration():
        return shared._run_live_stream(*args, **kwargs)


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Run gates, one dense Gemma stream, cleanup, and atomic output."""

    start = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    gate = build_gate_receipt(repo_root)
    method = load_json(repo_root / METHOD_RELATIVE_PATH)
    frozen = build_frozen_hash_receipt(repo_root, method)
    model_identity = _resolve_model_identity()
    tests_run = _checkpoint_tests(repo_root)
    preconditions, server, _initial = _collect_preconditions(
        repo_root, gate, frozen, model_identity, tests_run
    )
    preconditions["protected_file_hashes_before"] = protected_before
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    gpu_receipts: JsonDict = {}
    if preconditions["all_required_preconditions_available"]:
        preconditions["model_process_started"] = True
        rows, checkpoints, gpu_receipts = _run_live_stream(
            repo_root=repo_root,
            method=method,
            model_identity=model_identity,
            server=server,
            selected_gpu=int(preconditions["selected_gpu"]),
            task_deadline=start + TASK_TIMEOUT_S,
        )
    protected_after = _protected_hashes(repo_root)
    protected = _protected_receipt(protected_before, protected_after)
    preconditions["protected_file_hashes_after"] = protected_after
    if not preconditions["all_required_preconditions_available"]:
        artifact = build_blocked_report(
            run_date=run_date,
            gate_receipt=gate,
            frozen_receipt=frozen,
            model_identity=model_identity,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
        )
    else:
        artifact = build_report(
            run_date=run_date,
            gate_receipt=gate,
            frozen_receipt=frozen,
            model_identity=model_identity,
            per_unit_rows=rows,
            checkpoint_receipts=checkpoints,
            gpu_receipts=gpu_receipts,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
        )
    atomic_write_report(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the immutable dense Gemma CFR stream artifact."""

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
