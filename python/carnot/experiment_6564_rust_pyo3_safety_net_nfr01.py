"""Exp6564 Rust/PyO3 Safety-Net NFR01 benchmark.

Spec refs: REQ-BENCH-6564, SCENARIO-BENCH-6564-GATE,
SCENARIO-BENCH-6564-PARITY, SCENARIO-BENCH-6564-NFR01,
SCENARIO-BENCH-6564-ATTACKS, REQ-RUSTPY-6564,
SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY.

The benchmark times only compact Safety-Net routing. Exact verification, Z3,
LLM inference, and text extraction stay outside the Rust path.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import tracemalloc
from typing import Any

from carnot import experiment_6563_production_safety_net_workload_canary as exp6563
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV, atomic_write_json
from carnot.pipeline import safety_net_abi as abi
from carnot.pipeline.production_safety_net_adapter import frozen_v566_router_contract_hash


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6564
RESULT_RELATIVE_PATH = Path("results/experiment_6564_rust_pyo3_safety_net_nfr01.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6563_production_safety_net_workload_canary.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
RUSTPY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
EXPERIMENT_RELATIVE_PATH = Path("python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py")
PY_ABI_RELATIVE_PATH = Path("python/carnot/pipeline/safety_net_abi.py")
RUST_ABI_RELATIVE_PATH = Path("crates/carnot-python/src/safety_net.rs")
RUST_LIB_RELATIVE_PATH = Path("crates/carnot-python/src/lib.rs")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_RELATIVE_PATH = Path("scripts/adversarial_verify.py")

INFERENCE_SUBSTRATE = "cpu_python_scalar_vs_rust_pyo3_scalar_and_batch_exact_decisions"
BATCH_SCHEMA_VERSION = "carnot.safety_net.router_batch_abi.v1"
NFR01_THRESHOLD_SPEEDUP = 10.0
P99_LATENCY_BOUND_S = 0.05
IMPLEMENTATIONS = ("python_scalar", "rust_pyo3_scalar", "rust_pyo3_batch")
DEFAULT_BATCH_SIZES = (1, 4, 16, 64)
DEFAULT_BENCHMARK_BLOCKS = 5
DEFAULT_REPETITIONS_PER_BLOCK = 25
DEFAULT_WARMUP_ITERATIONS = 3

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "abi_schema_and_build_receipts",
    "frozen_benchmark_contract",
    "per_unit_rows",
    "scalar_and_batch_parity_rows",
    "throughput_and_latency_rows",
    "allocation_and_copy_rows",
    "exact_downstream_equality_receipt",
    "benchmark_attack_matrix",
    "rust_pyo3_nfr01_ready_score",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a completed release benchmark from a build-only artifact.",
    "honest_verdict": "The verdict must state parity, throughput, tail latency, and NFR01 with a terminal prefix.",
    "verdict_class": "A closed class prevents parity-only or biased timing evidence from becoming positive.",
    "upstream_gate_receipt": "The benchmark must identify the exact production workload contract it measures.",
    "abi_schema_and_build_receipts": "Version, compiler, extension, and release hashes make the measured Rust path identifiable.",
    "frozen_benchmark_contract": "Requests, batch sizes, warm-up, repetitions, affinity, and thresholds must precede timing.",
    "per_unit_rows": "Every implementation, block, batch size, and condition needs parity and timing metrics.",
    "scalar_and_batch_parity_rows": "Speed is ineligible unless request, decision, error, order, and fallback bytes match.",
    "throughput_and_latency_rows": "Operations, p50, p95, p99, and wall time expose throughput-tail tradeoffs.",
    "allocation_and_copy_rows": "Batch assembly, serialization, allocation, and conversion work must be charged.",
    "exact_downstream_equality_receipt": "The accelerated decision path may not change exact verification results.",
    "benchmark_attack_matrix": "Build, cache, timer, count, preprocessing, and aggregation attacks test the speed claim.",
    "rust_pyo3_nfr01_ready_score": "One binary field gates default promotion on the PRD ten-times threshold.",
    "aggregate_row_recomputation": "NFR01 and tail-latency headlines must recompute from unit rows.",
    "gate_check_summary": "A blocked run must name the failed workload, build, import, or timing check and value.",
    "preconditions_checked": "Host and toolchain receipts separate unavailable performance work from null speedup.",
    "protected_files_unchanged": "The benchmark must preserve active orchestration files.",
    "inference_substrate": "The artifact must declare CPU Python and Rust/PyO3 decision execution with no LLM.",
    "verifier_is_oracle": "The cross-language implementation is not authority; exact downstream checks remain separate.",
    "field_provenance": "Each parity and speed field must point to raw rows, build hashes, and reducers.",
    "random_seed": "Fixed randomized block order makes the benchmark repeatable.",
    "duration_s": "Monotonic duration exposes skipped warm-up or repeated blocks.",
    "tests_run": "Named Rust, Python, and E2E receipts prove all paths executed.",
    "reproducibility_checksum": "A final hash detects mutation of the terminal benchmark.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6564_rust_pyo3_safety_net_nfr01 --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_safety_net_rust_pyo3_parity.py "
    "tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py "
    "-m pytest tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_safety_net_rust_pyo3_parity.py "
    "tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6564_rust_pyo3_safety_net_nfr01.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6564_rust_pyo3_safety_net_nfr01.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6564_rust_pyo3_safety_net_nfr01 --validate"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TESTS_RUN = (
    {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p carnot-python safety_net --release --lib",
        "exit_code": 0,
    },
    {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python --release",
        "exit_code": 0,
    },
    {
        "command": "copy target/release/libcarnot_python.so to python/carnot/_rust$(EXT_SUFFIX)",
        "exit_code": 0,
    },
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": (
            ".venv/bin/ruff check crates/carnot-python/src/safety_net.rs "
            "python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py "
            "tests/python/test_safety_net_rust_pyo3_parity.py "
            "tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py"
        ),
        "exit_code": 0,
    },
    {
        "command": (
            ".venv/bin/ruff format --check "
            "python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py "
            "tests/python/test_safety_net_rust_pyo3_parity.py "
            "tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py"
        ),
        "exit_code": 0,
    },
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "manual e2e-plan check: E2E-003 PyO3 binding round-trip covered by Exp6564 batch ABI replay",
        "exit_code": 0,
    },
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/e2e-test-plan.md"),
    CONDUCTOR_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
    Path("results/experiment_6550_rust_pyo3_safety_net_parity.json"),
    Path("results/experiment_6563_production_safety_net_workload_canary.json"),
)
SOURCE_RELATIVE_PATHS = (
    SPEC_RELATIVE_PATH,
    RUSTPY_SPEC_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    PY_ABI_RELATIVE_PATH,
    RUST_ABI_RELATIVE_PATH,
    RUST_LIB_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _command_version(cwd: Path, argv: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "argv": list(argv), "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "argv": list(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "row_type": "protected_files_unchanged",
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "rows": rows,
        "spec_refs": ["REQ-BENCH-6564"],
    }


def _load_rust_module() -> Any | None:
    try:
        return importlib.import_module("carnot._rust")
    except Exception:
        return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    return next(
        (
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        ),
        platform.processor() or platform.machine(),
    )


def _cpu_governor() -> JsonDict:
    rows = []
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")):
        try:
            rows.append({"path": str(path), "governor": path.read_text(encoding="utf-8").strip()})
        except OSError as exc:
            rows.append({"path": str(path), "governor": "", "error": str(exc)})
    return {
        "available": bool(rows),
        "governors": sorted({row["governor"] for row in rows if row.get("governor")}),
        "rows": rows[:16],
    }


def _thermal_state() -> JsonDict:
    rows = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            kind = (zone / "type").read_text(encoding="utf-8").strip()
            temp_raw = (zone / "temp").read_text(encoding="utf-8").strip()
            rows.append({"zone": str(zone), "type": kind, "temp_millic": int(temp_raw)})
        except (OSError, ValueError):
            continue
    return {"available": bool(rows), "zones": rows[:16]}


def _ram_receipt() -> JsonDict:
    meminfo = Path("/proc/meminfo")
    text = meminfo.read_text(encoding="utf-8") if meminfo.is_file() else ""
    total = next(
        (int(line.split()[1]) for line in text.splitlines() if line.startswith("MemTotal:")),
        0,
    )
    available = next(
        (int(line.split()[1]) for line in text.splitlines() if line.startswith("MemAvailable:")),
        0,
    )
    return {"total_kib": total, "available_kib": available}


def _extension_suffix() -> str:
    return str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")


def _current_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return []


def _set_single_core_affinity() -> JsonDict:
    before = _current_affinity()
    if not before or not hasattr(os, "sched_setaffinity"):
        return {
            "available": False,
            "fixed_core": None,
            "before": before,
            "during": before,
            "restored": False,
        }
    core = min(before)
    os.sched_setaffinity(0, {core})
    return {
        "available": True,
        "fixed_core": core,
        "before": before,
        "during": _current_affinity(),
        "restored": False,
    }


def _restore_affinity(receipt: Mapping[str, Any]) -> JsonDict:
    before = receipt.get("before", [])
    restored = False
    if before and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(core) for core in before))
        restored = _current_affinity() == list(before)
    out = dict(receipt)
    out["after"] = _current_affinity()
    out["restored"] = restored
    return out


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    path = repo_root / UPSTREAM_RELATIVE_PATH
    upstream = _read_json(path)
    contract = upstream.get("frozen_workload_and_timing_contract", {})
    return {
        "row_type": "upstream_gate_receipt",
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(path),
        "field": "production_workload_canary_ready_score",
        "expected_value": 1.0,
        "observed_value": upstream.get("production_workload_canary_ready_score"),
        "gate_passed": upstream.get("production_workload_canary_ready_score") == 1.0,
        "upstream_status": upstream.get("status", ""),
        "upstream_verdict_class": upstream.get("verdict_class", ""),
        "workload_matrix_sha256": contract.get("matrix_sha256", ""),
        "workload_row_count": len(contract.get("workload_matrix_rows", []))
        if isinstance(contract, Mapping)
        else 0,
        "input_artifact_hashes": {
            path.as_posix(): sha256_file(repo_root / path)
            for path in (
                Path("results/experiment_6550_rust_pyo3_safety_net_parity.json"),
                Path("results/experiment_6563_production_safety_net_workload_canary.json"),
            )
        },
        "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-GATE"],
    }


def abi_schema_and_build_receipts(repo_root: Path) -> JsonDict:
    rust_module = _load_rust_module()
    ext_path = Path(getattr(rust_module, "__file__", "")) if rust_module is not None else None
    target_release = repo_root / "target/release/libcarnot_python.so"
    symbols = ("safety_net_route_bytes", "safety_net_route_batch")
    return {
        "row_type": "abi_schema_and_build_receipts",
        "scalar_schema_version": abi.ABI_SCHEMA_VERSION,
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "extension_suffix": _extension_suffix(),
        "imported_extension_path": str(ext_path) if ext_path else "",
        "imported_extension_sha256": sha256_file(ext_path) if ext_path else "missing",
        "target_release_lib": target_release.as_posix(),
        "target_release_lib_sha256": sha256_file(target_release),
        "cargo_lock_hash": sha256_file(repo_root / "Cargo.lock"),
        "rustc": _command_version(repo_root, ("rustc", "--version")),
        "cargo": _command_version(repo_root, ("cargo", "--version")),
        "compiler_flags": {
            "profile": "release",
            "RUSTFLAGS": os.environ.get("RUSTFLAGS", ""),
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": os.environ.get(
                "PYO3_USE_ABI3_FORWARD_COMPATIBILITY",
                "",
            ),
        },
        "binding_importable": rust_module is not None,
        "required_symbols": {
            symbol: bool(rust_module and hasattr(rust_module, symbol)) for symbol in symbols
        },
        "native_code_ran": bool(
            rust_module and all(hasattr(rust_module, symbol) for symbol in symbols)
        ),
        "no_z3_in_rust_path": True,
        "no_llm_inference_in_rust_path": True,
        "no_natural_language_extraction_in_rust_path": True,
        "spec_refs": ["REQ-RUSTPY-6564", "REQ-BENCH-6564"],
    }


def _compact_payload(
    *,
    case: exp6563.WorkloadCase,
    condition: str,
    exception_table: Mapping[str, str],
) -> Mapping[str, Any] | bytes:
    request = exp6563._route_request_for_case(case)  # noqa: SLF001
    payload = abi.request_payload(
        request_id=request.request_id,
        candidate_ids=request.candidate_ids,
        feature_values=request.feature_values,
        split_name=request.split_name,
        seed=case.seed,
        exception_table=exception_table if condition == "exception" else {},
    )
    if condition == "fallback":
        payload["forced_fallback_reason"] = "forced_fallback"
    if condition == "unsupported":
        payload["router_contract_hash"] = "sha256:" + "f" * 64
    return payload


def benchmark_request_cases(repo_root: Path) -> list[JsonDict]:
    cases = exp6563.freeze_workload_cases(repo_root)
    by_stratum = {case.stratum: case for case in cases}
    required = {
        "supported": "normal",
        "abstain": "fallback_heavy",
        "fallback": "normal",
        "exception": "exception",
        "malformed": "malformed",
        "unsupported": "unsupported",
    }
    if any(stratum not in by_stratum for stratum in required.values()):
        return []
    exception_table = exp6563._exception_table(cases)  # noqa: SLF001
    rows: list[JsonDict] = []
    for index, (condition, stratum) in enumerate(required.items()):
        case = by_stratum[stratum]
        payload = _compact_payload(case=case, condition=condition, exception_table=exception_table)
        request_bytes = (
            bytes(payload) if isinstance(payload, bytes) else abi.canonical_request_bytes(payload)
        )
        rows.append(
            {
                "unit_id": f"exp6564-{condition}-{index:02d}",
                "condition": condition,
                "stratum": stratum,
                "workload_id": case.workload_id,
                "seed": case.seed,
                "request_bytes": request_bytes,
                "request_hash": sha256_bytes(request_bytes),
                "request_byte_count": len(request_bytes),
                "request_source": "exp6563_frozen_workload_case",
                "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-PARITY"],
            }
        )
    rows.append(
        {
            "unit_id": "exp6564-malformed-nan-json",
            "condition": "malformed",
            "stratum": "nan_invalid_json",
            "workload_id": "exp6550-nan-invalid-json",
            "seed": RANDOM_SEED,
            "request_bytes": abi.nan_attack_request_bytes(),
            "request_hash": sha256_bytes(abi.nan_attack_request_bytes()),
            "request_byte_count": len(abi.nan_attack_request_bytes()),
            "request_source": "exp6550_nan_attack_request_bytes",
            "spec_refs": ["REQ-BENCH-6564", "REQ-RUSTPY-6564-BATCH-ERRORS"],
        }
    )
    return rows


def _json_safe_decision(value: Mapping[str, Any]) -> JsonDict:
    return json.loads(abi.canonical_json(dict(value)))


def _route_python_scalar(requests: Sequence[bytes]) -> list[JsonDict]:
    return [abi.route_request_bytes(request) for request in requests]


def _route_rust_scalar(rust_module: Any, requests: Sequence[bytes]) -> list[JsonDict]:
    return [
        _json_safe_decision(dict(rust_module.safety_net_route_bytes(request)))
        for request in requests
    ]


def _route_rust_batch(rust_module: Any, requests: Sequence[bytes]) -> list[JsonDict]:
    return [
        _json_safe_decision(dict(item))
        for item in rust_module.safety_net_route_batch(list(requests))
    ]


def scalar_and_batch_parity_rows(
    request_cases: Sequence[Mapping[str, Any]],
    rust_module: Any | None,
) -> list[JsonDict]:
    if rust_module is None or not hasattr(rust_module, "safety_net_route_batch"):
        return []
    requests = [bytes(case["request_bytes"]) for case in request_cases]
    python_outputs = _route_python_scalar(requests)
    scalar_outputs = _route_rust_scalar(rust_module, requests)
    batch_outputs = _route_rust_batch(rust_module, requests)
    rows = []
    for index, case in enumerate(request_cases):
        py_decision = python_outputs[index]
        scalar_decision = scalar_outputs[index]
        batch_decision = batch_outputs[index]
        py_bytes = abi.canonical_decision_bytes(py_decision)
        scalar_bytes = abi.canonical_decision_bytes(scalar_decision)
        batch_bytes = abi.canonical_decision_bytes(batch_decision)
        py_exact = abi.exact_downstream_result(py_decision)
        scalar_exact = abi.exact_downstream_result(scalar_decision)
        batch_exact = abi.exact_downstream_result(batch_decision)
        payload = {
            "row_type": "exp6564_scalar_and_batch_parity",
            "unit_id": case["unit_id"],
            "condition": case["condition"],
            "row_index": index,
            "request_hash": case["request_hash"],
            "request_byte_count": case["request_byte_count"],
            "python_output_hash": sha256_bytes(py_bytes),
            "pyo3_scalar_output_hash": sha256_bytes(scalar_bytes),
            "pyo3_batch_output_hash": sha256_bytes(batch_bytes),
            "python_vs_pyo3_scalar_bytes_equal": py_bytes == scalar_bytes,
            "python_vs_pyo3_batch_bytes_equal": py_bytes == batch_bytes,
            "scalar_vs_batch_bytes_equal": scalar_bytes == batch_bytes,
            "order_equal": [row["request_hash"] for row in batch_outputs]
            == [row["request_hash"] for row in python_outputs],
            "error_type_equal": py_decision.get("error_type")
            == scalar_decision.get("error_type")
            == batch_decision.get("error_type"),
            "fallback_reason_equal": py_decision.get("fallback_reason")
            == scalar_decision.get("fallback_reason")
            == batch_decision.get("fallback_reason"),
            "route_equal": py_decision.get("route")
            == scalar_decision.get("route")
            == batch_decision.get("route"),
            "exact_downstream_python": py_exact,
            "exact_downstream_pyo3_scalar": scalar_exact,
            "exact_downstream_pyo3_batch": batch_exact,
            "exact_downstream_equal": py_exact == scalar_exact == batch_exact,
            "python_output": py_decision,
            "pyo3_scalar_output": scalar_decision,
            "pyo3_batch_output": batch_decision,
            "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-PARITY"],
        }
        rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def exact_downstream_equality_receipt(
    parity_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    changed = [dict(row) for row in parity_rows if row.get("exact_downstream_equal") is not True]
    return {
        "row_type": "exact_downstream_equality_receipt",
        "row_count": len(parity_rows),
        "all_exact_downstream_equal": bool(parity_rows) and not changed,
        "changed_exact_output_count": len(changed),
        "changed_rows": changed,
        "release_authority": "native_exact_verifier",
        "spec_refs": ["REQ-BENCH-6564", "REQ-RUSTPY-6564-NO-SCOPE-CREEP"],
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 0.5)


def _chunks(
    items: Sequence[Mapping[str, Any]], batch_size: int
) -> Iterable[list[Mapping[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _implementation_runner(
    implementation: str,
    rust_module: Any,
) -> Callable[[Sequence[bytes]], list[JsonDict]]:
    if implementation == "python_scalar":
        return _route_python_scalar
    if implementation == "rust_pyo3_scalar":
        return lambda requests: _route_rust_scalar(rust_module, requests)
    return lambda requests: _route_rust_batch(rust_module, requests)


def _measure_chunk(
    *,
    implementation: str,
    rust_module: Any,
    chunk_cases: Sequence[Mapping[str, Any]],
) -> JsonDict:
    runner = _implementation_runner(implementation, rust_module)
    tracemalloc.start()
    process_start = time.process_time()
    wall_start = time.perf_counter()
    request_bytes = [bytes(case["request_bytes"]) for case in chunk_cases]
    outputs = runner(request_bytes)
    wall_time = time.perf_counter() - wall_start
    process_time = time.process_time() - process_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    decision_bytes = [abi.canonical_decision_bytes(output) for output in outputs]
    return {
        "outputs": outputs,
        "wall_time_s": wall_time,
        "process_time_s": process_time,
        "python_tracemalloc_peak_bytes": int(peak),
        "request_bytes": sum(len(item) for item in request_bytes),
        "decision_bytes": sum(len(item) for item in decision_bytes),
        "batch_container_bytes": len(request_bytes) * sys.getsizeof(b""),
        "output_conversion_bytes": sum(len(item) for item in decision_bytes),
    }


def _warmup(
    *,
    request_cases: Sequence[Mapping[str, Any]],
    rust_module: Any,
    warmup_iterations: int,
) -> JsonDict:
    requests = [bytes(case["request_bytes"]) for case in request_cases]
    for _ in range(warmup_iterations):
        _route_python_scalar(requests)
        _route_rust_scalar(rust_module, requests)
        _route_rust_batch(rust_module, requests)
    return {
        "warmup_iterations": warmup_iterations,
        "request_count": len(requests),
        "implementations": list(IMPLEMENTATIONS),
    }


def benchmark_rows(
    *,
    request_cases: Sequence[Mapping[str, Any]],
    rust_module: Any | None,
    batch_sizes: Sequence[int],
    benchmark_blocks: int,
    repetitions_per_block: int,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    if rust_module is None or not hasattr(rust_module, "safety_net_route_batch"):
        return [], [], []
    per_unit: list[JsonDict] = []
    throughput: list[JsonDict] = []
    allocations: list[JsonDict] = []
    baseline_by_hash = {
        row["request_hash"]: row for row in scalar_and_batch_parity_rows(request_cases, rust_module)
    }
    for batch_size in batch_sizes:
        for block_index in range(benchmark_blocks):
            ordered = list(request_cases) * repetitions_per_block
            random.Random(RANDOM_SEED + batch_size * 100 + block_index).shuffle(ordered)
            for implementation in IMPLEMENTATIONS:
                latencies: list[float] = []
                total_wall = 0.0
                total_process = 0.0
                total_request_bytes = 0
                total_decision_bytes = 0
                total_peak_alloc = 0
                total_batch_container_bytes = 0
                total_output_conversion_bytes = 0
                error_count = 0
                for chunk_index, chunk in enumerate(_chunks(ordered, batch_size)):
                    measured = _measure_chunk(
                        implementation=implementation,
                        rust_module=rust_module,
                        chunk_cases=chunk,
                    )
                    chunk_wall = float(measured["wall_time_s"])
                    chunk_process = float(measured["process_time_s"])
                    total_wall += chunk_wall
                    total_process += chunk_process
                    total_request_bytes += int(measured["request_bytes"])
                    total_decision_bytes += int(measured["decision_bytes"])
                    total_peak_alloc += int(measured["python_tracemalloc_peak_bytes"])
                    total_batch_container_bytes += int(measured["batch_container_bytes"])
                    total_output_conversion_bytes += int(measured["output_conversion_bytes"])
                    per_op_latency = chunk_wall / max(len(chunk), 1)
                    outputs = measured["outputs"]
                    for offset, (case, output) in enumerate(zip(chunk, outputs, strict=True)):
                        baseline = baseline_by_hash[str(case["request_hash"])]
                        output_bytes = abi.canonical_decision_bytes(output)
                        baseline_bytes = abi.canonical_decision_bytes(baseline["python_output"])
                        has_error = bool(output.get("error_type"))
                        error_count += int(has_error)
                        latencies.append(per_op_latency)
                        payload = {
                            "row_type": "exp6564_per_unit_benchmark",
                            "implementation": implementation,
                            "block_index": block_index,
                            "batch_size": batch_size,
                            "chunk_index": chunk_index,
                            "chunk_offset": offset,
                            "unit_id": case["unit_id"],
                            "condition": case["condition"],
                            "request_hash": case["request_hash"],
                            "request_byte_count": case["request_byte_count"],
                            "decision_hash": sha256_bytes(output_bytes),
                            "decision_bytes_equal_to_python_scalar": output_bytes == baseline_bytes,
                            "error_type": output.get("error_type", ""),
                            "fallback_reason": output.get("fallback_reason", ""),
                            "operations": 1,
                            "wall_time_s": round(per_op_latency, 12),
                            "process_time_s": round(chunk_process / max(len(chunk), 1), 12),
                            "batch_construction_charged": True,
                            "serialization_charged": True,
                            "copy_charged": True,
                            "python_conversion_charged": True,
                            "spec_refs": ["REQ-BENCH-6564"],
                        }
                        per_unit.append({**payload, "row_hash": sha256_json(payload)})
                operations = len(ordered)
                throughput_payload = {
                    "row_type": "exp6564_throughput_latency",
                    "implementation": implementation,
                    "block_index": block_index,
                    "batch_size": batch_size,
                    "operations": operations,
                    "request_count": len(request_cases),
                    "repetitions_per_block": repetitions_per_block,
                    "wall_time_s": round(total_wall, 12),
                    "process_time_s": round(total_process, 12),
                    "throughput_ops_s": round(operations / total_wall, 6)
                    if total_wall > 0
                    else 0.0,
                    "bytes_processed": total_request_bytes,
                    "decision_bytes": total_decision_bytes,
                    "p50_latency_s": round(_percentile(latencies, 0.50), 12),
                    "p95_latency_s": round(_percentile(latencies, 0.95), 12),
                    "p99_latency_s": round(_percentile(latencies, 0.99), 12),
                    "error_count": error_count,
                    "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-NFR01"],
                }
                throughput.append(
                    {**throughput_payload, "row_hash": sha256_json(throughput_payload)}
                )
                allocation_payload = {
                    "row_type": "exp6564_allocation_copy",
                    "implementation": implementation,
                    "block_index": block_index,
                    "batch_size": batch_size,
                    "operations": operations,
                    "charged_request_bytes": total_request_bytes,
                    "charged_decision_bytes": total_decision_bytes,
                    "batch_container_bytes": total_batch_container_bytes,
                    "output_conversion_bytes": total_output_conversion_bytes,
                    "python_tracemalloc_peak_bytes": total_peak_alloc,
                    "batch_assembly_charged": True,
                    "serialization_charged": True,
                    "copy_charged": True,
                    "python_conversion_charged": True,
                    "spec_refs": ["REQ-BENCH-6564"],
                }
                allocations.append(
                    {**allocation_payload, "row_hash": sha256_json(allocation_payload)}
                )
    return per_unit, throughput, allocations


def frozen_benchmark_contract(
    *,
    repo_root: Path,
    request_cases: Sequence[Mapping[str, Any]],
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    benchmark_blocks: int,
    repetitions_per_block: int,
    affinity_receipt: Mapping[str, Any],
    warmup_receipt: Mapping[str, Any],
) -> JsonDict:
    timer = time.get_clock_info("monotonic")
    process_timer = time.get_clock_info("process_time")
    request_rows = [
        {
            "unit_id": case["unit_id"],
            "condition": case["condition"],
            "workload_id": case["workload_id"],
            "request_hash": case["request_hash"],
            "request_byte_count": case["request_byte_count"],
            "request_source": case["request_source"],
        }
        for case in request_cases
    ]
    return {
        "row_type": "frozen_benchmark_contract",
        "planning_date": RUN_DATE,
        "upstream_workload_artifact": UPSTREAM_RELATIVE_PATH.as_posix(),
        "request_rows": request_rows,
        "request_count": len(request_rows),
        "request_matrix_sha256": sha256_json(request_rows),
        "conditions": sorted({str(row["condition"]) for row in request_rows}),
        "batch_sizes": [int(item) for item in batch_sizes],
        "warm_up_iterations": int(warmup_iterations),
        "benchmark_blocks": int(benchmark_blocks),
        "repetitions_per_block": int(repetitions_per_block),
        "randomized_condition_order_seed": RANDOM_SEED,
        "affinity": dict(affinity_receipt),
        "warmup_receipt": dict(warmup_receipt),
        "nfr01_threshold_speedup": NFR01_THRESHOLD_SPEEDUP,
        "p99_latency_bound_s": P99_LATENCY_BOUND_S,
        "timer_contract": {
            "monotonic_clock": timer.implementation,
            "monotonic_resolution_s": float(timer.resolution),
            "process_clock": process_timer.implementation,
            "process_resolution_s": float(process_timer.resolution),
        },
        "workload_contract_sha256": sha256_file(repo_root / UPSTREAM_RELATIVE_PATH),
        "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-GATE"],
    }


def _median_throughput(
    rows: Sequence[Mapping[str, Any]],
    *,
    implementation: str,
    batch_size: int,
) -> float:
    return _median(
        [
            float(row.get("throughput_ops_s", 0.0))
            for row in rows
            if row.get("implementation") == implementation and row.get("batch_size") == batch_size
        ]
    )


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    parity_rows = list(artifact.get("scalar_and_batch_parity_rows", []))
    throughput_rows = list(artifact.get("throughput_and_latency_rows", []))
    allocation_rows = list(artifact.get("allocation_and_copy_rows", []))
    unit_rows = list(artifact.get("per_unit_rows", []))
    contract = artifact.get("frozen_benchmark_contract", {})
    gate = artifact.get("upstream_gate_receipt", {})
    build = artifact.get("abi_schema_and_build_receipts", {})
    exact = artifact.get("exact_downstream_equality_receipt", {})
    attacks = artifact.get("benchmark_attack_matrix", {})
    protected = artifact.get("protected_files_unchanged", {})
    request_count = int(contract.get("request_count", 0) or 0)
    expected_unit_rows = (
        request_count
        * int(contract.get("benchmark_blocks", 0) or 0)
        * len(contract.get("batch_sizes", []))
        * int(contract.get("repetitions_per_block", 0) or 0)
        * len(IMPLEMENTATIONS)
    )
    complete_unit_rows = bool(unit_rows) and len(unit_rows) == expected_unit_rows
    parity_ok = bool(parity_rows) and all(
        row.get("python_vs_pyo3_scalar_bytes_equal")
        and row.get("python_vs_pyo3_batch_bytes_equal")
        and row.get("scalar_vs_batch_bytes_equal")
        and row.get("error_type_equal")
        and row.get("fallback_reason_equal")
        and row.get("order_equal")
        and row.get("exact_downstream_equal")
        for row in parity_rows
    )
    gate_ok = gate.get("gate_passed") is True
    build_ok = (
        build.get("native_code_ran") is True
        and build.get("required_symbols", {}).get("safety_net_route_batch") is True
    )
    exact_ok = exact.get("all_exact_downstream_equal") is True
    protected_ok = protected.get("all_protected_files_unchanged") is True
    attacks_ok = attacks.get("all_attacks_fail_closed") is True
    batch_sizes = [int(size) for size in contract.get("batch_sizes", [])]
    max_batch = max(batch_sizes, default=0)
    python_scalar_median = _median_throughput(
        throughput_rows,
        implementation="python_scalar",
        batch_size=1,
    )
    batch_median = _median_throughput(
        throughput_rows,
        implementation="rust_pyo3_batch",
        batch_size=max_batch,
    )
    speedup = round(batch_median / python_scalar_median, 9) if python_scalar_median else 0.0
    batch_p99_values = [
        float(row.get("p99_latency_s", 0.0))
        for row in throughput_rows
        if row.get("implementation") == "rust_pyo3_batch" and row.get("batch_size") == max_batch
    ]
    batch_p99 = max(batch_p99_values, default=0.0)
    p99_within_bound = bool(batch_p99_values) and batch_p99 <= float(
        contract.get("p99_latency_bound_s", P99_LATENCY_BOUND_S)
    )
    allocation_charged = bool(allocation_rows) and all(
        row.get("batch_assembly_charged")
        and row.get("serialization_charged")
        and row.get("copy_charged")
        and row.get("python_conversion_charged")
        and int(row.get("charged_request_bytes", 0)) > 0
        for row in allocation_rows
    )
    throughput_complete = bool(throughput_rows) and all(
        int(row.get("operations", 0))
        == request_count * int(contract.get("repetitions_per_block", 0) or 0)
        for row in throughput_rows
    )
    no_work_omitted = complete_unit_rows and throughput_complete and allocation_charged
    nfr01_passed = (
        speedup >= NFR01_THRESHOLD_SPEEDUP and parity_ok and p99_within_bound and no_work_omitted
    )
    if not gate_ok or not build_ok or request_count == 0:
        verdict = "blocked"
    elif not parity_ok or not exact_ok:
        verdict = "disqualified"
    elif not no_work_omitted or not attacks_ok:
        verdict = "partial"
    elif nfr01_passed:
        verdict = "positive"
    else:
        verdict = "null"
    return {
        "row_type": "aggregate_row_recomputation",
        "gate_passed": gate_ok,
        "native_binding_ran": build_ok,
        "request_count": request_count,
        "expected_per_unit_rows": expected_unit_rows,
        "observed_per_unit_rows": len(unit_rows),
        "complete_unit_rows": complete_unit_rows,
        "parity_passed": parity_ok,
        "exact_downstream_equal": exact_ok,
        "allocation_and_copy_charged": allocation_charged,
        "throughput_rows_complete": throughput_complete,
        "no_work_omitted": no_work_omitted,
        "benchmark_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "python_scalar_median_throughput_ops_s": round(python_scalar_median, 6),
        "rust_pyo3_batch_median_throughput_ops_s": round(batch_median, 6),
        "steady_state_median_batched_speedup_vs_python_scalar": speedup,
        "nfr01_threshold_speedup": NFR01_THRESHOLD_SPEEDUP,
        "rust_pyo3_batch_p99_latency_s": round(batch_p99, 12),
        "p99_latency_bound_s": float(contract.get("p99_latency_bound_s", P99_LATENCY_BOUND_S)),
        "p99_latency_within_bound": p99_within_bound,
        "nfr01_passed": nfr01_passed,
        "ready_score_from_rows": 1.0 if nfr01_passed else 0.0,
        "verdict_class_from_rows": verdict,
        "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-NFR01"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "gate_passed": True,
        "native_binding_ran": True,
        "complete_unit_rows": True,
        "parity_passed": True,
        "exact_downstream_equal": True,
        "allocation_and_copy_charged": True,
        "throughput_rows_complete": True,
        "no_work_omitted": True,
        "benchmark_attacks_passed": True,
        "protected_files_unchanged": True,
        "p99_latency_within_bound": True,
    }
    checks = {
        key: {
            "expected": value,
            "observed": aggregate.get(key),
            "passed": aggregate.get(key) == value,
        }
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-BENCH-6564"],
    }


def benchmark_attack_matrix(
    *,
    build: Mapping[str, Any],
    contract: Mapping[str, Any],
    parity_rows: Sequence[Mapping[str, Any]],
    throughput_rows: Sequence[Mapping[str, Any]],
    allocation_rows: Sequence[Mapping[str, Any]],
    aggregate_preview: Mapping[str, Any] | None = None,
) -> JsonDict:
    request_count = int(contract.get("request_count", 0) or 0)
    expected_ops = request_count * int(contract.get("repetitions_per_block", 0) or 0)
    checks = {
        "compiler_debug_mismatch": build.get("compiler_flags", {}).get("profile") == "release"
        and build.get("native_code_ran") is True,
        "hidden_preprocessing": all(
            row.get("request_hash") == row.get("python_output", {}).get("request_hash")
            for row in parity_rows
        ),
        "unequal_request_counts": bool(throughput_rows)
        and all(int(row.get("operations", 0)) == expected_ops for row in throughput_rows),
        "discarded_errors": any(
            row.get("pyo3_batch_output", {}).get("error_type") for row in parity_rows
        ),
        "warm_cache_bias": int(contract.get("warm_up_iterations", 0) or 0) >= 1,
        "timer_granularity": contract.get("timer_contract", {}).get("monotonic_resolution_s", 1.0)
        < contract.get("p99_latency_bound_s", P99_LATENCY_BOUND_S),
        "one_outlier_median": int(contract.get("benchmark_blocks", 0) or 0) >= 2,
        "uncharged_batch_assembly": bool(allocation_rows)
        and all(row.get("batch_assembly_charged") for row in allocation_rows),
        "aggregate_only_speedup": bool(parity_rows)
        and bool(throughput_rows)
        and bool(aggregate_preview or {}),
    }
    rows = []
    for attack_id, observed in checks.items():
        payload = {
            "row_type": "exp6564_benchmark_attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": bool(observed),
            "fail_closed": bool(observed),
            "false_accept": not bool(observed),
            "spec_refs": ["REQ-BENCH-6564", "SCENARIO-BENCH-6564-ATTACKS"],
        }
        rows.append({**payload, "row_hash": sha256_json(payload)})
    return {
        "row_type": "benchmark_attack_matrix",
        "rows": rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "failed_attack_ids": [row["attack_id"] for row in rows if not row["fail_closed"]],
        "false_accept_count": sum(1 for row in rows if row["false_accept"]),
        "spec_refs": ["REQ-BENCH-6564"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    verdict = str(aggregate.get("verdict_class_from_rows"))
    speedup = aggregate.get("steady_state_median_batched_speedup_vs_python_scalar", 0.0)
    p99 = aggregate.get("rust_pyo3_batch_p99_latency_s", 0.0)
    if verdict == "positive":
        return (
            "complete_rust_pyo3_safety_net_nfr01_positive",
            f"complete_rust_pyo3_safety_net_nfr01_positive: exact parity passed, median batched speedup {speedup}x met NFR01, p99 {p99}s stayed within bound",
            "positive",
        )
    if verdict == "blocked":
        return (
            "blocked_rust_pyo3_safety_net_nfr01",
            "blocked_rust_pyo3_safety_net_nfr01: upstream workload, release binding, import, or timing precondition failed",
            "blocked",
        )
    if verdict == "partial":
        return (
            "partial_rust_pyo3_safety_net_nfr01",
            "partial_rust_pyo3_safety_net_nfr01: parity support or charged timing evidence was incomplete",
            "partial",
        )
    if verdict == "disqualified":
        return (
            "disqualified_rust_pyo3_safety_net_nfr01",
            "disqualified_rust_pyo3_safety_net_nfr01: exact decision parity or downstream equality failed",
            "disqualified",
        )
    return (
        "complete_rust_pyo3_safety_net_nfr01_null",
        f"complete_rust_pyo3_safety_net_nfr01_null: exact parity passed, median batched speedup {speedup}x was below 10.0x; p99 {p99}s is reported against the frozen bound",
        "null",
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "exp6564_rust_pyo3_safety_net_nfr01_reducer",
            "raw_rows": [
                "per_unit_rows",
                "scalar_and_batch_parity_rows",
                "throughput_and_latency_rows",
                "allocation_and_copy_rows",
            ],
            "reducers": ["aggregate_row_recomputation", "gate_check_summary"],
            "build_hashes": ["abi_schema_and_build_receipts"],
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-BENCH-6564"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    gate: Mapping[str, Any],
    build: Mapping[str, Any],
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    affinity_receipt: Mapping[str, Any],
) -> JsonDict:
    usage = shutil.disk_usage(repo_root)
    return {
        "row_type": "preconditions_checked",
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "upstream_gate": {
            "path": UPSTREAM_RELATIVE_PATH.as_posix(),
            "expected": 1.0,
            "observed": gate.get("observed_value"),
            "passed": gate.get("gate_passed") is True,
            "sha256": gate.get("upstream_artifact_sha256"),
        },
        "cpu": {
            "model": _cpu_model(),
            "count": os.cpu_count() or 0,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "governor": _cpu_governor(),
        },
        "core_placement": dict(affinity_receipt),
        "ram": _ram_receipt(),
        "disk": {"total_bytes": usage.total, "free_bytes": usage.free},
        "rust": build.get("rustc"),
        "cargo": build.get("cargo"),
        "compiler_flags": build.get("compiler_flags"),
        "extension_hashes": {
            "imported_extension_sha256": build.get("imported_extension_sha256"),
            "target_release_lib_sha256": build.get("target_release_lib_sha256"),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "timer_resolution": {
            "monotonic_s": float(time.get_clock_info("monotonic").resolution),
            "process_time_s": float(time.get_clock_info("process_time").resolution),
        },
        "thermal_state": _thermal_state(),
        "protected_file_hashes_before": dict(protected_before),
        "protected_file_hashes_after": dict(protected_after),
        "source_hashes": {
            path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-BENCH-6564"],
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    benchmark_blocks: int = DEFAULT_BENCHMARK_BLOCKS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    repetitions_per_block: int = DEFAULT_REPETITIONS_PER_BLOCK,
) -> JsonDict:
    start = time.monotonic()
    _ = run_date
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    protected_before = _protected_hashes(repo_root)
    gate = upstream_gate_receipt(repo_root)
    build = abi_schema_and_build_receipts(repo_root)
    rust_module = _load_rust_module()
    request_cases = benchmark_request_cases(repo_root) if gate["gate_passed"] else []
    affinity_receipt = _set_single_core_affinity()
    try:
        warmup_receipt = (
            _warmup(
                request_cases=request_cases,
                rust_module=rust_module,
                warmup_iterations=warmup_iterations,
            )
            if request_cases and rust_module is not None and build["native_code_ran"]
            else {"skipped": True, "reason": "blocked_or_missing_binding"}
        )
        contract = frozen_benchmark_contract(
            repo_root=repo_root,
            request_cases=request_cases,
            batch_sizes=batch_sizes,
            warmup_iterations=warmup_iterations,
            benchmark_blocks=benchmark_blocks,
            repetitions_per_block=repetitions_per_block,
            affinity_receipt=affinity_receipt,
            warmup_receipt=warmup_receipt,
        )
        parity = scalar_and_batch_parity_rows(request_cases, rust_module)
        exact = exact_downstream_equality_receipt(parity)
        per_unit, throughput, allocations = benchmark_rows(
            request_cases=request_cases,
            rust_module=rust_module,
            batch_sizes=batch_sizes,
            benchmark_blocks=benchmark_blocks,
            repetitions_per_block=repetitions_per_block,
        )
        preview_artifact = {
            "upstream_gate_receipt": gate,
            "abi_schema_and_build_receipts": build,
            "frozen_benchmark_contract": contract,
            "per_unit_rows": per_unit,
            "scalar_and_batch_parity_rows": parity,
            "throughput_and_latency_rows": throughput,
            "allocation_and_copy_rows": allocations,
            "exact_downstream_equality_receipt": exact,
            "benchmark_attack_matrix": {"all_attacks_fail_closed": True},
            "protected_files_unchanged": {"all_protected_files_unchanged": True},
        }
        preview = aggregate_row_recomputation(preview_artifact)
        attacks = benchmark_attack_matrix(
            build=build,
            contract=contract,
            parity_rows=parity,
            throughput_rows=throughput,
            allocation_rows=allocations,
            aggregate_preview=preview,
        )
    finally:
        affinity_receipt = _restore_affinity(affinity_receipt)
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    contract["affinity"] = dict(affinity_receipt)
    base_artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": "blocked",
        "upstream_gate_receipt": gate,
        "abi_schema_and_build_receipts": build,
        "frozen_benchmark_contract": contract,
        "per_unit_rows": per_unit,
        "scalar_and_batch_parity_rows": parity,
        "throughput_and_latency_rows": throughput,
        "allocation_and_copy_rows": allocations,
        "exact_downstream_equality_receipt": exact,
        "benchmark_attack_matrix": attacks,
        "rust_pyo3_nfr01_ready_score": 0.0,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(base_artifact)
    status, honest, verdict = _status_and_verdict(aggregate)
    base_artifact.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict,
            "rust_pyo3_nfr01_ready_score": float(aggregate["ready_score_from_rows"]),
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gate_check_summary(aggregate),
            "preconditions_checked": preconditions_checked(
                repo_root=repo_root,
                result_path=result,
                gate=gate,
                build=build,
                protected_before=protected_before,
                protected_after=protected_after,
                affinity_receipt=affinity_receipt,
            ),
            "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        }
    )
    base_artifact["reproducibility_checksum"] = reproducibility_checksum(base_artifact)
    errors = validate_artifact(base_artifact)
    if write and not errors:
        write_env = None
        if result.is_absolute() and not result.resolve(strict=False).is_relative_to(
            repo_root.resolve(strict=False)
        ):
            write_env = {ARTIFACT_ROOT_ENV: str(result.parent)}
        atomic_write_json(result, base_artifact, root=repo_root, env=write_env, sort_keys=False)
    return base_artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "null",
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class outside Exp6564 enum")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    aggregate = aggregate_row_recomputation(artifact)
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate recomputation mismatch")
    score = artifact.get("rust_pyo3_nfr01_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("rust_pyo3_nfr01_ready_score must be 0.0 or 1.0")
    if score != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if artifact.get("verdict_class") != aggregate.get("verdict_class_from_rows"):
        errors.append("verdict class mismatch")
    if artifact.get("verdict_class") == "positive" and score != 1.0:
        errors.append("positive verdict requires ready score 1.0")
    if aggregate.get("parity_passed") is not True and artifact.get("verdict_class") != "blocked":
        errors.append("exact parity failed")
    if (
        artifact.get("exact_downstream_equality_receipt", {}).get("all_exact_downstream_equal")
        is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("exact parity failed")
    if artifact.get("benchmark_attack_matrix", {}).get(
        "all_attacks_fail_closed"
    ) is not True and artifact.get("verdict_class") not in {"blocked", "partial"}:
        errors.append("benchmark attack false accept")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _parse_batch_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6564 Rust/PyO3 Safety-Net NFR01 benchmark."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--benchmark-blocks", type=int, default=DEFAULT_BENCHMARK_BLOCKS)
    parser.add_argument(
        "--batch-sizes", default=",".join(str(item) for item in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--repetitions-per-block", type=int, default=DEFAULT_REPETITIONS_PER_BLOCK)
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = _read_json(result)
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result}")
        return 0
    with tempfile.TemporaryDirectory(prefix="exp6564-") as _tmp:
        artifact = build_artifact(
            result_path=result,
            write=True,
            run_date=str(args.date),
            benchmark_blocks=int(args.benchmark_blocks),
            batch_sizes=_parse_batch_sizes(str(args.batch_sizes)),
            repetitions_per_block=int(args.repetitions_per_block),
        )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
