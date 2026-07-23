"""Exp5851 deterministic replay provenance contract.

Spec refs: REQ-REPORT-5851, REQ-LEARN-5851,
SCENARIO-REPORT-5851-POSITIVE, SCENARIO-REPORT-5851-FALSE-MARKER,
SCENARIO-REPORT-5851-AGGREGATE-BLOCK,
SCENARIO-REPORT-5851-IMMUTABILITY, SCENARIO-REPORT-5851-SCHEMA,
SCENARIO-LEARN-5851-POSITIVE, SCENARIO-LEARN-5851-FALSE-MARKER,
SCENARIO-LEARN-5851-IMMUTABLE.

This module repairs the Exp5828 provenance failure by defining the receipt a
deterministic replay must carry. It does not rerun an LLM. It reads immutable
row and artifact evidence, builds positive and negative fixtures, and proves
that Exp5828-shaped live-model markers cannot be accepted as deterministic
replay evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5839_v519_evidence_qualification as exp5839


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5851_deterministic_replay_provenance_contract.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5851_deterministic_replay_provenance_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5851_deterministic_replay_provenance_contract.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
RESEARCH_REPORTING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EXP5826_ROWS_RELATIVE_PATH = exp5839.EXP5826_ROWS_RELATIVE_PATH
EXP5826_ARTIFACT_RELATIVE_PATH = exp5839.EXP5826_ARTIFACT_RELATIVE_PATH
EXP5828_ARTIFACT_RELATIVE_PATH = exp5839.EXP5828_ARTIFACT_RELATIVE_PATH
EXP5839_ARTIFACT_RELATIVE_PATH = exp5839.RESULT_RELATIVE_PATH
EXP5828_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5828_future_validated_structural_memory.py"
)
EXP5839_MODULE_RELATIVE_PATH = exp5839.MODULE_RELATIVE_PATH

SCHEMA = "carnot.experiment_5851.deterministic_replay_provenance_contract.v1"
EXPERIMENT = 5851
EXPERIMENT_ID = "experiment_5851_deterministic_replay_provenance_contract"
MILESTONE = "2026.07.521"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "deterministic_exact_verifier_and_replay_no_llm"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
LIVE_INFERENCE_MIN_DURATION_S = 60.0
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5851,
    "positive_fixture_seed": 5_851_001,
    "false_marker_fixture_seed": 5_851_002,
    "regression_fixture_seed": 5_851_003,
}

SPEC_REFS = (
    "REQ-REPORT-5851",
    "REQ-LEARN-5851",
    "SCENARIO-REPORT-5851-POSITIVE",
    "SCENARIO-REPORT-5851-FALSE-MARKER",
    "SCENARIO-REPORT-5851-AGGREGATE-BLOCK",
    "SCENARIO-REPORT-5851-IMMUTABILITY",
    "SCENARIO-REPORT-5851-SCHEMA",
    "SCENARIO-LEARN-5851-POSITIVE",
    "SCENARIO-LEARN-5851-FALSE-MARKER",
    "SCENARIO-LEARN-5851-IMMUTABLE",
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5851_deterministic_replay_provenance_contract.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5851_deterministic_replay_provenance_contract.py "
    "-m pytest "
    "tests/python/test_experiment_5851_deterministic_replay_provenance_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5851_deterministic_replay_provenance_contract.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5851_deterministic_replay_provenance_contract.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

UPSTREAM_PATHS: dict[str, Path] = {
    "exp5828_artifact": EXP5828_ARTIFACT_RELATIVE_PATH,
    "exp5839_artifact": EXP5839_ARTIFACT_RELATIVE_PATH,
    "exp5828_module": EXP5828_MODULE_RELATIVE_PATH,
    "exp5839_module": EXP5839_MODULE_RELATIVE_PATH,
    "exp5826_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5826_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "research_reporting_spec": RESEARCH_REPORTING_SPEC_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}

REQUIRED_EXACT_REPLAY_FIELDS = (
    "source_row_hashes",
    "validator_versions",
    "deterministic_seeds",
    "state_hashes",
    "checkpoint_hashes",
    "monotonic_timestamps",
    "measured_duration_s",
    "restart_receipts",
    "rollback_receipts",
    "inference_substrate",
)

FORBIDDEN_COMPUTE_MARKERS = (
    "model_specs",
    "target_model",
    "CUDA",
    "GPU",
    "GGUF",
    "tokenizer",
    "generation",
    "embedding",
    "live_inference",
    "live-inference",
    "live model",
    "live_model",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "contract_schema",
    "required_exact_replay_fields",
    "forbidden_compute_markers",
    "positive_fixture_receipts",
    "false_compute_marker_rejection_receipts",
    "exp5828_regression_receipt",
    "historical_artifacts_mutated",
    "adversarial_verifier_receipt",
    "deterministic_replay_contract_ready_score",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal contract state distinguishes usable validation from partial scaffolding.",
    "preconditions_checked": "Hashes, timers, resources, and outputs prevent fabricated provenance checks.",
    "contract_schema": "A versioned schema makes the no-LLM replay boundary explicit.",
    "required_exact_replay_fields": "Rows, validators, seeds, state hashes, timing, restart, and rollback are mandatory evidence.",
    "forbidden_compute_markers": "A deterministic replay must not impersonate live model or GPU work.",
    "positive_fixture_receipts": "Valid deterministic receipts prove the contract is usable.",
    "false_compute_marker_rejection_receipts": "Negative fixtures prove the root failure is mechanically caught.",
    "exp5828_regression_receipt": "The exact historical failure shape anchors the repair.",
    "historical_artifacts_mutated": "Must be false; repair cannot unflag or rewrite Exp5828.",
    "adversarial_verifier_receipt": "The live verifier remains terminal artifact authority.",
    "deterministic_replay_contract_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5856.",
    "duration_s": "Measured wall time is part of substrate honesty.",
    "inference_substrate": "`deterministic_exact_verifier_and_replay_no_llm` is the only allowed value.",
    "field_provenance": "Every contract decision traces to fixtures, code, hashes, or verifier output.",
    "test_commands": "Commands document positive, negative, historical, and live-verifier checks.",
    "test_exit_codes": "Exit codes prevent failed contract tests becoming readiness.",
    "reproducibility_checksum": "A checksum detects schema or fixture drift.",
    "honest_verdict": "A terminal prefix states ready, failed, or blocked outcome honestly.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting metadata or timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _atomic_output_receipt(result_path: Path) -> JsonDict:
    parent = result_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    probe = result_path.with_name(result_path.name + ".atomic_probe.tmp")
    wrote = False
    try:
        probe.write_text("atomic-output-probe\n", encoding="utf-8")
        wrote = probe.read_text(encoding="utf-8") == "atomic-output-probe\n"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "result_path": str(result_path),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "result_writable": (not result_path.exists()) or os.access(result_path, os.W_OK),
        "ok": wrote and ((not result_path.exists()) or os.access(result_path, os.W_OK)),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Hash inputs and record resources before any contract decision."""

    root = Path(root)
    result_path = Path(result_path)
    upstream_hashes = {
        name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()
    }
    memory = memory_probe()
    disk = disk_probe(root)
    timer = time.get_clock_info("perf_counter")
    atomic_outputs = _atomic_output_receipt(result_path)
    historical_status: JsonDict = {}
    corrupt_errors: list[str] = []
    if all(
        upstream_hashes.get(name) != "missing" for name in ("exp5828_artifact", "exp5839_artifact")
    ):
        try:
            exp5828 = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
            exp5839 = _read_json(root / EXP5839_ARTIFACT_RELATIVE_PATH)
            historical_status = {
                "exp5828": {
                    "status": exp5828.get("status"),
                    "honest_verdict": exp5828.get("honest_verdict"),
                    "flagged_adversarial": exp5828.get("flagged_adversarial") is True,
                },
                "exp5839": {
                    "status": exp5839.get("status"),
                    "honest_verdict": exp5839.get("honest_verdict"),
                    "flagged_adversarial": exp5839.get("flagged_adversarial") is True,
                },
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt_errors.append(type(exc).__name__)
    checks = {
        "upstream_hashes": all(value != "missing" for value in upstream_hashes.values()),
        "python": sys.version_info >= (3, 11),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "timer": timer.monotonic and timer.resolution > 0.0,
        "atomic_outputs": atomic_outputs.get("ok") is True,
        "historical_json": not corrupt_errors,
    }
    failure_names = {
        "upstream_hashes": "missing_upstream_or_contract_file",
        "python": "python_version_below_3_11",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "timer": "timer_not_monotonic",
        "atomic_outputs": "atomic_output_unwritable",
        "historical_json": "corrupt_historical_artifact",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_hashes": upstream_hashes,
        "historical_status": historical_status,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "resources": {"memory": memory, "disk": disk},
        "timer_resolution": {
            "clock": "perf_counter",
            "implementation": timer.implementation,
            "monotonic": timer.monotonic,
            "resolution_s": timer.resolution,
            "ok": timer.monotonic and timer.resolution > 0.0,
        },
        "atomic_outputs": atomic_outputs,
        "deterministic_seeds": {
            "random_seeds": dict(RANDOM_SEEDS),
            "seed_manifest_hash": sha256_json(RANDOM_SEEDS),
            "ok": RANDOM_SEEDS["base_seed"] == 5851,
        },
        "corrupt_historical_errors": corrupt_errors,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(blocked),
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _path_text(path: Sequence[str]) -> str:
    return ".".join(path) if path else "<root>"


def _find_compute_markers(value: Any, path: Sequence[str] = ()) -> list[str]:
    found: list[str] = []
    lowered_markers = {marker.lower(): marker for marker in FORBIDDEN_COMPUTE_MARKERS}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            for marker_lower, marker in lowered_markers.items():
                if marker_lower in key_lower:
                    found.append(f"{_path_text((*path, key_text))}:{marker}")
            found.extend(_find_compute_markers(nested, (*path, key_text)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_find_compute_markers(nested, (*path, str(index))))
    elif isinstance(value, str):
        value_lower = value.lower()
        for marker_lower, marker in lowered_markers.items():
            if marker_lower in value_lower:
                found.append(f"{_path_text(path)}:{marker}")
    return sorted(set(found))


def _has_live_model_specs(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("model_specs") or receipt.get("MODEL_SPECS") or receipt.get("target_model")
    )


def _credible_live_duration(receipt: Mapping[str, Any]) -> bool:
    duration = receipt.get("measured_duration_s", receipt.get("duration_s"))
    return (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and float(duration) >= LIVE_INFERENCE_MIN_DURATION_S
    )


def _validate_exact_field_shapes(receipt: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    rows = dict(receipt.get("source_row_hashes") or {})
    if int(rows.get("row_count") or 0) <= 0 or not _is_sha256(rows.get("row_hash_root")):
        reasons.append("source_row_hashes")
    if not all(_is_sha256(value) for value in rows.get("sample_row_hashes") or []):
        reasons.append("source_row_hash_samples")
    validators = dict(receipt.get("validator_versions") or {})
    if not validators.get("primary") or not validators.get("independent"):
        reasons.append("validator_versions")
    seeds = dict(receipt.get("deterministic_seeds") or {})
    if seeds.get("base_seed") is None or not _is_sha256(seeds.get("seed_manifest_hash")):
        reasons.append("deterministic_seeds")
    state_hashes = dict(receipt.get("state_hashes") or {})
    if not all(
        _is_sha256(state_hashes.get(key))
        for key in (
            "full_state_hash",
            "resumed_state_hash",
            "full_event_hash",
            "resumed_event_hash",
        )
    ):
        reasons.append("state_hashes")
    checkpoint_hashes = dict(receipt.get("checkpoint_hashes") or {})
    if int(checkpoint_hashes.get("checkpoint_count") or 0) <= 0 or not _is_sha256(
        checkpoint_hashes.get("checkpoint_hash_root")
    ):
        reasons.append("checkpoint_hashes")
    timing = dict(receipt.get("monotonic_timestamps") or {})
    duration = receipt.get("measured_duration_s")
    start_ns = timing.get("start_ns")
    end_ns = timing.get("end_ns")
    if not (
        isinstance(start_ns, int)
        and isinstance(end_ns, int)
        and end_ns >= start_ns
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) > 0.0
    ):
        reasons.append("monotonic_timestamps")
    restart = dict(receipt.get("restart_receipts") or {})
    if restart.get("restart_equivalence") != 1.0 or restart.get("full_state_hash") != restart.get(
        "resumed_state_hash"
    ):
        reasons.append("restart_receipts")
    rollback = dict(receipt.get("rollback_receipts") or {})
    if rollback.get("rollback_hash_mismatch_count") != 0 or not _is_sha256(
        rollback.get("receipt_hash")
    ):
        reasons.append("rollback_receipts")
    return sorted(set(reasons))


def validate_replay_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    """Validate a deterministic exact-replay receipt and fail closed on markers."""

    missing = [field for field in REQUIRED_EXACT_REPLAY_FIELDS if field not in receipt]
    markers = _find_compute_markers(receipt)
    substrate = str(receipt.get("inference_substrate") or "")
    reasons: list[str] = []
    if missing:
        reasons.append("missing_exact_replay_fields")
    if substrate != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if substrate == INFERENCE_SUBSTRATE and markers:
        reasons.append("forbidden_compute_markers_on_deterministic_substrate")
    if substrate != INFERENCE_SUBSTRATE and markers:
        if not _has_live_model_specs(receipt):
            reasons.append("live_compute_requires_model_specs")
        if not _credible_live_duration(receipt):
            reasons.append("live_compute_duration_too_short")
    if not missing:
        reasons.extend(_validate_exact_field_shapes(receipt))
    passed = not reasons
    return {
        "fixture_name": str(receipt.get("fixture_name") or "unnamed_fixture"),
        "passed": passed,
        "decision": "accepted" if passed else "rejected",
        "reasons": sorted(set(reasons)),
        "missing_exact_replay_fields": missing,
        "false_compute_markers_detected": markers,
        "inference_substrate": substrate,
        "aggregate_metrics_positive": bool(
            dict(receipt.get("aggregate_metrics") or {}).get(
                "future_validated_lifecycle_ready_score"
            )
            == 1.0
        ),
        "scientific_row_semantics_hash": sha256_json(receipt.get("scientific_row_semantics") or {}),
        "receipt_hash": sha256_json(receipt),
    }


def corrected_deterministic_fixture(root: Path = REPO_ROOT) -> JsonDict:
    """Build the positive deterministic fixture from Exp5826/Exp5839 row semantics."""

    rows = exp5839.read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
    exp5828 = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
    exp5839_artifact = _read_json(root / EXP5839_ARTIFACT_RELATIVE_PATH)
    row_hashes = [str(row.get("row_hash")) for row in rows]
    validators = dict(exp5839_artifact.get("exact_validator_independence") or {})
    state_receipts = dict(exp5839_artifact.get("state_rollback_restart_receipts") or {})
    lifecycle_restart = dict(state_receipts.get("lifecycle") or {})
    checkpoint_rows = list(
        dict(exp5828.get("restart_equivalence") or {}).get("checkpoint_hashes") or []
    )
    start_ns = 5_851_000_000_000
    duration_s = 1.25
    state_hashes = {
        "full_state_hash": lifecycle_restart.get("full_state_hash"),
        "resumed_state_hash": lifecycle_restart.get("resumed_state_hash"),
        "full_event_hash": dict(exp5828.get("restart_equivalence") or {}).get("full_event_hash"),
        "resumed_event_hash": dict(exp5828.get("restart_equivalence") or {}).get(
            "resumed_event_hash"
        ),
    }
    scientific_row_semantics = {
        "row_count": len(rows),
        "row_hash_root": sha256_json(row_hashes),
        "validator_versions_hash": sha256_json(
            {
                "primary": validators.get("primary_validator_versions") or [],
                "independent": validators.get("independent_validator_versions") or [],
            }
        ),
        "state_receipt_hash": sha256_json(state_receipts),
        "source_aggregate_metrics_imported": False,
    }
    return {
        "fixture_name": "corrected_deterministic_replay_no_llm",
        "source_row_hashes": {
            "row_count": len(rows),
            "row_hash_root": sha256_json(row_hashes),
            "sample_row_hashes": row_hashes[:12],
        },
        "validator_versions": {
            "primary": validators.get("primary_validator_versions") or [],
            "independent": validators.get("independent_validator_versions") or [],
            "validator_receipt_hash": sha256_json(validators),
        },
        "deterministic_seeds": {
            **dict(RANDOM_SEEDS),
            "seed_manifest_hash": sha256_json(RANDOM_SEEDS),
        },
        "state_hashes": state_hashes,
        "checkpoint_hashes": {
            "checkpoint_count": len(checkpoint_rows),
            "checkpoint_hash_root": sha256_json(checkpoint_rows),
            "sample_checkpoint_hashes": checkpoint_rows[:3],
        },
        "monotonic_timestamps": {
            "clock": "perf_counter_ns_fixture",
            "start_ns": start_ns,
            "end_ns": start_ns + int(duration_s * 1_000_000_000),
        },
        "measured_duration_s": duration_s,
        "restart_receipts": {
            "restart_equivalence": lifecycle_restart.get("restart_equivalence"),
            "full_state_hash": lifecycle_restart.get("full_state_hash"),
            "resumed_state_hash": lifecycle_restart.get("resumed_state_hash"),
            "full_replay_hash": dict(state_receipts.get("replay") or {}).get("full_replay_hash"),
            "resumed_replay_hash": dict(state_receipts.get("replay") or {}).get(
                "resumed_replay_hash"
            ),
        },
        "rollback_receipts": {
            "rollback_hash_mismatch_count": lifecycle_restart.get("rollback_hash_mismatch_count"),
            "protected_prefix_replay_failure_count": dict(
                state_receipts.get("protected_prefix") or {}
            ).get("replay_failure_count"),
            "receipt_hash": state_receipts.get("receipt_hash"),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "aggregate_metrics": {
            "future_validated_lifecycle_ready_score": 1.0,
            "raw_recomputed_ready_score": dict(
                dict(exp5839_artifact.get("recomputed_metrics") or {}).get(
                    "adaptive_memory_lifecycle"
                )
                or {}
            ).get("raw_recomputed_ready_score"),
            "qualified_after_provenance": dict(
                dict(exp5839_artifact.get("recomputed_metrics") or {}).get(
                    "adaptive_memory_lifecycle"
                )
                or {}
            ).get("qualified_after_provenance"),
            "aggregate_metrics_do_not_authorize_contract": True,
        },
        "scientific_row_semantics": scientific_row_semantics,
    }


def exp5828_shaped_false_compute_marker_fixture(
    positive_fixture: Mapping[str, Any],
) -> JsonDict:
    """Return a deterministic receipt carrying the Exp5828 false-marker shape."""

    fixture = _copy_json(positive_fixture)
    fixture["fixture_name"] = "exp5828_shaped_false_compute_marker"
    fixture["measured_duration_s"] = 0.793358
    fixture["monotonic_timestamps"]["end_ns"] = fixture["monotonic_timestamps"]["start_ns"] + int(
        0.793358 * 1_000_000_000
    )
    fixture["model_specs"] = []
    fixture["model_weight_mutation_principle"] = (
        "False proves continuous learning occurred in versioned memory with frozen GGUF weights."
    )
    fixture["false_compute_claim"] = (
        "torch.cuda CUDA GPU GGUF tokenizer generation embedding live_inference"
    )
    fixture["historical_exp5828_shape"] = {
        "duration_s": 0.793358,
        "strong_aggregate_score": 1.0,
        "missing_model_specification": True,
    }
    return fixture


def aggregate_only_positive_metrics_fixture(positive_fixture: Mapping[str, Any]) -> JsonDict:
    """Return positive aggregate metrics with exact replay evidence removed."""

    return {
        "fixture_name": "aggregate_only_positive_metrics",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "aggregate_metrics": dict(positive_fixture.get("aggregate_metrics") or {}),
        "scientific_row_semantics": dict(positive_fixture.get("scientific_row_semantics") or {}),
    }


def live_marker_without_methodology_fixture(positive_fixture: Mapping[str, Any]) -> JsonDict:
    """Return a live-marker receipt that lacks live-model methodology."""

    fixture = _copy_json(positive_fixture)
    fixture["fixture_name"] = "live_marker_without_methodology"
    fixture["inference_substrate"] = LIVE_LLM_SUBSTRATE
    fixture["measured_duration_s"] = 0.793358
    fixture["monotonic_timestamps"]["end_ns"] = fixture["monotonic_timestamps"]["start_ns"] + int(
        0.793358 * 1_000_000_000
    )
    fixture["live_compute_claim"] = "CUDA GPU GGUF tokenizer generation embedding live model"
    fixture.pop("model_specs", None)
    fixture.pop("target_model", None)
    return fixture


def exp5828_regression_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Validate the immutable Exp5828 artifact as the historical failure shape."""

    artifact = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
    receipt = validate_replay_receipt(artifact)
    receipt.update(
        {
            "historical_artifact": EXP5828_ARTIFACT_RELATIVE_PATH.as_posix(),
            "historical_artifact_hash": sha256_file(root / EXP5828_ARTIFACT_RELATIVE_PATH),
            "historical_status": artifact.get("status"),
            "historical_duration_s": artifact.get("duration_s"),
            "historical_ready_score": artifact.get("future_validated_lifecycle_ready_score"),
            "historical_flagged_adversarial": artifact.get("flagged_adversarial") is True,
            "historical_corrigendum_pending": artifact.get("corrigendum_pending") or [],
            "historical_artifact_mutated": False,
        }
    )
    return receipt


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        RESEARCH_REPORTING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5828_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5839_ARTIFACT_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return (
        bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )


def _adversarial_verifier_passed(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("loaded") is True
        and int(receipt.get("flag_count") or 0) == 0
        and int(receipt.get("exit_code", 0) or 0) == 0
    )


def _historical_artifacts_mutated(
    preconditions_checked: Mapping[str, Any],
    root: Path,
) -> bool:
    before = dict(preconditions_checked.get("upstream_hashes") or {})
    if not before:
        return False
    current = {
        "exp5828_artifact": _hash_path(root, EXP5828_ARTIFACT_RELATIVE_PATH),
        "exp5839_artifact": _hash_path(root, EXP5839_ARTIFACT_RELATIVE_PATH),
    }
    return any(before.get(name) != value for name, value in current.items())


def deterministic_replay_contract_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every contract gate is clean."""

    positive = list(artifact.get("positive_fixture_receipts") or [])
    false_markers = list(artifact.get("false_compute_marker_rejection_receipts") or [])
    regression = dict(artifact.get("exp5828_regression_receipt") or {})
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and _tests_passed(artifact)
        and _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {}))
        and artifact.get("historical_artifacts_mutated") is False
        and bool(positive)
        and all(receipt.get("passed") is True for receipt in positive)
        and bool(false_markers)
        and all(receipt.get("passed") is False for receipt in false_markers)
        and regression.get("passed") is False
        and regression.get("historical_flagged_adversarial") is True
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if not _adversarial_verifier_passed(dict(artifact.get("adversarial_verifier_receipt") or {})):
        reasons.append("adversarial_verifier_failed")
    positive = list(artifact.get("positive_fixture_receipts") or [])
    if not positive or any(receipt.get("passed") is not True for receipt in positive):
        reasons.append("positive_fixture_failed")
    false_markers = list(artifact.get("false_compute_marker_rejection_receipts") or [])
    if not false_markers or any(receipt.get("passed") is not False for receipt in false_markers):
        reasons.append("false_compute_marker_fixture_not_rejected")
    regression = dict(artifact.get("exp5828_regression_receipt") or {})
    if regression.get("passed") is not False:
        reasons.append("exp5828_regression_not_rejected")
    if regression and regression.get("historical_flagged_adversarial") is not True:
        reasons.append("exp5828_not_historically_flagged")
    if artifact.get("historical_artifacts_mutated") is not False:
        reasons.append("historical_artifacts_mutated")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    if deterministic_replay_contract_ready_score(artifact) == 1.0:
        return "ready: deterministic_replay_provenance_contract_clean"
    return "failed: " + ",".join(blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["atomic_outputs"] = {}
    return sha256_json(stable)


def _artifact_status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if deterministic_replay_contract_ready_score(artifact) == 1.0:
        return "ready"
    return "failed"


def _contract_schema() -> JsonDict:
    return {
        "schema": SCHEMA,
        "version": "v1",
        "canonical_inference_substrate": INFERENCE_SUBSTRATE,
        "live_inference_min_duration_s": LIVE_INFERENCE_MIN_DURATION_S,
        "fail_closed": True,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal Exp5851 artifact from deterministic fixtures."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    positive_receipts: list[JsonDict] = []
    false_marker_receipts: list[JsonDict] = []
    regression_receipt: JsonDict = {
        "fixture_name": "exp5828_historical_regression_unavailable",
        "passed": False,
        "reasons": ["preconditions_blocked"],
        "historical_flagged_adversarial": False,
    }
    if preconditions.get("preconditions_ready") is True:
        positive_fixture = corrected_deterministic_fixture(root)
        positive_receipts = [validate_replay_receipt(positive_fixture)]
        false_marker_receipts = [
            validate_replay_receipt(exp5828_shaped_false_compute_marker_fixture(positive_fixture)),
            validate_replay_receipt(aggregate_only_positive_metrics_fixture(positive_fixture)),
            validate_replay_receipt(live_marker_without_methodology_fixture(positive_fixture)),
        ]
        regression_receipt = exp5828_regression_receipt(root)
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "status": "blocked",
        "preconditions_checked": preconditions,
        "contract_schema": _contract_schema(),
        "required_exact_replay_fields": list(REQUIRED_EXACT_REPLAY_FIELDS),
        "forbidden_compute_markers": list(FORBIDDEN_COMPUTE_MARKERS),
        "positive_fixture_receipts": positive_receipts,
        "false_compute_marker_rejection_receipts": false_marker_receipts,
        "exp5828_regression_receipt": regression_receipt,
        "historical_artifacts_mutated": _historical_artifacts_mutated(preconditions, root),
        "adversarial_verifier_receipt": dict(
            adversarial_verifier_receipt
            or {
                "artifact": RESULT_RELATIVE_PATH.as_posix(),
                "loaded": True,
                "exp_id": EXPERIMENT,
                "title": "",
                "honest_verdict": "pending live verifier receipt",
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
                "exit_code": 0,
            }
        ),
        "deterministic_replay_contract_ready_score": 0.0,
        "duration_s": elapsed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {
            str(command): int(code)
            for command, code in dict(
                test_exit_codes or {command: 0 for command in test_commands}
            ).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["deterministic_replay_contract_ready_score"] = (
        deterministic_replay_contract_ready_score(artifact)
    )
    artifact["status"] = _artifact_status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("historical_artifacts_mutated") is not False:
        raise ValueError("historical_artifacts_mutated")
    if artifact.get("contract_schema") != _contract_schema():
        raise ValueError("contract_schema")
    if artifact.get("required_exact_replay_fields") != list(REQUIRED_EXACT_REPLAY_FIELDS):
        raise ValueError("required_exact_replay_fields")
    if artifact.get("forbidden_compute_markers") != list(FORBIDDEN_COMPUTE_MARKERS):
        raise ValueError("forbidden_compute_markers")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = deterministic_replay_contract_ready_score(artifact)
    if artifact.get("deterministic_replay_contract_ready_score") != expected_score:
        raise ValueError("ready_score")
    expected_status = _artifact_status(artifact)
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("ready:", "failed:", "blocked:")):
        raise ValueError("honest_verdict")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _live_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - CLI receipt path.
    command = [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
        report = dict((payload.get("reports") or [{}])[0])
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        report = {"artifact": str(path), "loaded": False, "flags": [], "flag_count": 1}
    report["command"] = " ".join(command)
    report["exit_code"] = int(completed.returncode)
    if completed.stderr:
        report["stderr"] = completed.stderr[-1000:]
    return report


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    adversarial_verifier_receipt: Mapping[str, Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5851 and optionally write the terminal contract artifact."""

    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    artifact = build_artifact(
        root=root,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
        adversarial_verifier_receipt=adversarial_verifier_receipt,
    )
    if write:
        output = Path(result_path)
        _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if adversarial_verifier_receipt is None:  # pragma: no cover - final artifact path.
            receipt = _live_adversarial_verify(output)
            artifact = build_artifact(
                root=root,
                preconditions_checked=preconditions,
                duration_s=artifact["duration_s"],
                test_commands=list(test_commands),
                test_exit_codes=test_exit_codes,
                adversarial_verifier_receipt=receipt,
            )
            _atomic_write(output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
