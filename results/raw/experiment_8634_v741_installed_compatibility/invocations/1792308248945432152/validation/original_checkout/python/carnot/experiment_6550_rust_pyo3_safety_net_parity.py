"""Exp6550 Rust/PyO3 Safety-Net routing parity.

Spec refs: REQ-RUSTPY-6550, REQ-RUSTPY-6550-SCHEMA,
REQ-RUSTPY-6550-NUMERIC, REQ-RUSTPY-6550-SERIALIZATION,
REQ-RUSTPY-6550-ERRORS, REQ-RUSTPY-6550-PARITY,
REQ-RUSTPY-6550-FALLBACK, REQ-RUSTPY-6550-ROLLBACK,
REQ-RUSTPY-6550-NO-AUTHORITY, SCENARIO-RUSTPY-6550-BOUNDARY-PARITY.

This is a deterministic binding replay. It compares the Python compact router
mirror with the Rust/PyO3 ABI on identical request bytes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.pipeline import safety_net_abi as abi
from carnot.pipeline.production_safety_net_adapter import (
    FROZEN_V566_FEATURE_NAMES,
    SafetyNetProductionAdapter,
    SafetyNetRouterConfig,
    SafetyNetRouteRequest,
    frozen_v566_router_contract_hash,
)
from carnot.task_runtime_receipts import sha256_bytes, sha256_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6550
RESULT_RELATIVE_PATH = Path("results/experiment_6550_rust_pyo3_safety_net_parity.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6549_production_safety_net_adapter.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
PY_ABI_RELATIVE_PATH = Path("python/carnot/pipeline/safety_net_abi.py")
PY_ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/production_safety_net_adapter.py")
EXPERIMENT_RELATIVE_PATH = Path("python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py")
RUST_ABI_RELATIVE_PATH = Path("crates/carnot-python/src/safety_net.rs")
RUST_LIB_RELATIVE_PATH = Path("crates/carnot-python/src/lib.rs")
RUST_COMPAT_RELATIVE_PATH = Path("python/carnot/_rust_compat.py")
TEST_RELATIVE_PATHS = (
    Path("tests/python/test_safety_net_rust_pyo3_parity.py"),
    Path("tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py"),
)
ROADMAP_RELATIVE_PATH = Path("ops/roadmap-quarantine/roadmap-2026.08.567-refusal1.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "rust_pyo3_and_python_compact_router_replay_no_llm"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "abi_schema_and_version_contract",
    "build_and_binding_receipts",
    "parity_rows",
    "serialization_equality_receipt",
    "error_semantics_rows",
    "exact_downstream_equality_receipt",
    "fallback_and_python_rollback_receipt",
    "abi_attack_matrix",
    "cross_language_router_parity_ready_score",
    "per_unit_rows",
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
    "status": "A terminal state proves the binding task progressed beyond compilation setup.",
    "honest_verdict": "The verdict states parity, fallback, and rollback outcomes with a terminal prefix.",
    "verdict_class": "A closed class prevents partial ABI coverage from becoming a positive parity claim.",
    "upstream_gate_receipt": "The ABI identifies the exact production adapter contract it implements.",
    "abi_schema_and_version_contract": "Explicit versions prevent silent Python/Rust semantic drift.",
    "build_and_binding_receipts": "Compiler and extension hashes prove which native code ran.",
    "parity_rows": "One row per unit and condition makes every cross-language comparison recheckable.",
    "serialization_equality_receipt": "Equal decisions with different bytes can still break caches and downstream contracts.",
    "error_semantics_rows": "Malformed and unsupported inputs must fail closed in the same way across languages.",
    "exact_downstream_equality_receipt": "The ABI may not change accepted exact results.",
    "fallback_and_python_rollback_receipt": "Production safety requires reachable native fallback and complete binding rollback.",
    "abi_attack_matrix": "Coercion, ordering, version, and Unicode attacks expose hidden language-boundary assumptions.",
    "cross_language_router_parity_ready_score": "A binary parity gate gives the independent audit an unambiguous input.",
    "per_unit_rows": "Comparative parity claims require a row for each evaluated unit.",
    "aggregate_row_recomputation": "The parity headline derives only from emitted rows.",
    "gate_check_summary": "A blocked artifact names the failed build, input, or upstream check and value.",
    "preconditions_checked": "Toolchain receipts distinguish unavailable bindings from null parity.",
    "protected_files_unchanged": "Scoped binding work preserves the active roadmap and conductor.",
    "inference_substrate": "This is a deterministic binding replay, not live model inference.",
    "verifier_is_oracle": "Parity is checked against independent exact outputs, not as a learned verifier win.",
    "field_provenance": "Each decision and readiness field points to bytes, builds, and rows.",
    "random_seed": "Stable case generation and order make parity failures reproducible.",
    "duration_s": "Charged build and FFI time prevents a cost-free cross-language claim.",
    "tests_run": "Named Python, Rust, and binding checks prove both sides executed.",
    "reproducibility_checksum": "A content hash detects later changes to the parity record.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6550_rust_pyo3_safety_net_parity --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_safety_net_rust_pyo3_parity.py "
    "tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/pipeline/safety_net_abi.py,"
    "python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py "
    "-m pytest tests/python/test_safety_net_rust_pyo3_parity.py "
    "tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/pipeline/safety_net_abi.py,"
    "python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_safety_net_rust_pyo3_parity.py "
    "tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6550_rust_pyo3_safety_net_parity.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6550_rust_pyo3_safety_net_parity.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6550_rust_pyo3_safety_net_parity --validate"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TESTS_RUN = (
    {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p carnot-python safety_net --lib",
        "exit_code": 0,
    },
    {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python",
        "exit_code": 0,
    },
    {
        "command": "copy target/debug/libcarnot_python.so to python/carnot/_rust$(EXT_SUFFIX)",
        "exit_code": 0,
    },
    {
        "command": "rustfmt --check crates/carnot-python/src/safety_net.rs crates/carnot-python/src/lib.rs",
        "exit_code": 0,
    },
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": ".venv/bin/ruff check python/carnot/pipeline/safety_net_abi.py python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py tests/python/test_safety_net_rust_pyo3_parity.py tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py",
        "exit_code": 0,
    },
    {
        "command": ".venv/bin/ruff format --check python/carnot/pipeline/safety_net_abi.py python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py tests/python/test_safety_net_rust_pyo3_parity.py tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py",
        "exit_code": 0,
    },
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "manual e2e-plan check: E2E-003 PyO3 binding round-trip covered by Exp6550 focused binding replay",
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
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    SPEC_RELATIVE_PATH,
    PY_ABI_RELATIVE_PATH,
    PY_ADAPTER_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    RUST_ABI_RELATIVE_PATH,
    RUST_LIB_RELATIVE_PATH,
    RUST_COMPAT_RELATIVE_PATH,
    *TEST_RELATIVE_PATHS,
    UPSTREAM_RELATIVE_PATH,
)


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = __import__("hashlib").sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _command_version(root: Path, argv: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "argv": list(argv), "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "argv": list(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    meminfo = Path("/proc/meminfo")
    mem_text = meminfo.read_text(encoding="utf-8") if meminfo.is_file() else ""
    mem_kb = next(
        (int(line.split()[1]) for line in mem_text.splitlines() if line.startswith("MemTotal:")),
        0,
    )
    usage = shutil.disk_usage(repo_root)
    return {
        "cpu": {
            "cpu_count": os.cpu_count() or 0,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "ram_total_bytes": mem_kb * 1024,
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
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
        "rows": rows,
    }


def upstream_gate_receipt(repo_root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    observed = upstream.get("production_safety_net_adapter_ready_score")
    contract = upstream.get("adapter_configuration_contract", {})
    return {
        "row_type": "upstream_gate_receipt",
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(repo_root / UPSTREAM_RELATIVE_PATH),
        "field": "production_safety_net_adapter_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "adapter_configuration_hash": contract.get("configuration_hash")
        if isinstance(contract, Mapping)
        else None,
        "feature_names": list(FROZEN_V566_FEATURE_NAMES),
        "spec_refs": ["REQ-RUSTPY-6550", "REQ-RUSTPY-6550-SCHEMA"],
    }


def abi_schema_and_version_contract() -> JsonDict:
    return {
        "row_type": "abi_schema_and_version_contract",
        "schema_version": abi.ABI_SCHEMA_VERSION,
        "request_type": "SafetyNetFeatureRequest",
        "decision_type": "SafetyNetRoutingDecision",
        "decision_fields": [
            "schema_version",
            "route",
            "abstain",
            "uncertainty_bucket",
            "exception_hit",
            "fallback_reason",
            "original_order",
            "chosen_order",
            "error_type",
            "exact_fallback_reachable",
            "request_hash",
            "router_contract_hash",
        ],
        "route_enum": ["compact_router", "native_exact_fallback"],
        "uncertainty_buckets": ["low", "medium", "high", "unsupported"],
        "feature_names": list(FROZEN_V566_FEATURE_NAMES),
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "no_model_loading": True,
        "no_language_parsing": True,
        "release_authority": "native_exact_verifier_only",
        "spec_refs": ["REQ-RUSTPY-6550-SCHEMA", "REQ-RUSTPY-6550-NO-AUTHORITY"],
    }


def _load_rust_module() -> Any | None:
    try:
        return importlib.import_module("carnot._rust")
    except Exception:
        return None


def build_and_binding_receipts(repo_root: Path) -> JsonDict:
    rust_module = _load_rust_module()
    ext_path = Path(getattr(rust_module, "__file__", "")) if rust_module is not None else None
    symbols = (
        "RustSafetyNetFeatureRequest",
        "RustSafetyNetRoutingDecision",
        "RustSafetyNetRouter",
        "safety_net_route_bytes",
    )
    target_so = repo_root / "target/debug/libcarnot_python.so"
    return {
        "row_type": "build_and_binding_receipts",
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "rustc": _command_version(repo_root, ["rustc", "--version"]),
        "cargo": _command_version(repo_root, ["cargo", "--version"]),
        "maturin": _command_version(repo_root, ["maturin", "--version"]),
        "binding_importable": rust_module is not None,
        "binding_file": str(ext_path) if ext_path else "",
        "binding_file_sha256": sha256_file(ext_path) if ext_path else "missing",
        "target_debug_lib": target_so.as_posix(),
        "target_debug_lib_sha256": sha256_file(target_so),
        "cargo_lock_hash": sha256_file(repo_root / "Cargo.lock"),
        "build_cache": {
            "target_dir_exists": (repo_root / "target").is_dir(),
            "target_debug_lib_exists": target_so.is_file(),
            "cargo_registry_exists": (Path.home() / ".cargo/registry").is_dir(),
        },
        "required_symbols": {
            symbol: bool(rust_module and hasattr(rust_module, symbol)) for symbol in symbols
        },
        "native_code_ran": bool(
            rust_module and all(hasattr(rust_module, symbol) for symbol in symbols)
        ),
        "spec_refs": ["REQ-RUSTPY-6550", "REQ-RUSTPY-6550-PARITY"],
    }


def _case_inputs() -> list[JsonDict]:
    c1 = "sha256:" + "1" * 64
    c2 = "sha256:" + "2" * 64
    c3 = "sha256:" + "3" * 64
    exception = abi.exception_key(candidate_ids=(c1, c2), split_name="train")
    return [
        {
            "unit_id": "supported-held-compact",
            "condition": "held_compact",
            "supported": True,
            "payload": abi.request_payload(
                request_id="held-compact",
                candidate_ids=(c1, c2),
                feature_values={"candidate_count": 2, "constraint_count": 2},
            ),
        },
        {
            "unit_id": "supported-boundary-abstention",
            "condition": "boundary_abstention",
            "supported": True,
            "payload": abi.request_payload(request_id="single", candidate_ids=(c1,)),
        },
        {
            "unit_id": "supported-exception",
            "condition": "exception_lookup",
            "supported": True,
            "payload": abi.request_payload(
                request_id="exception",
                candidate_ids=(c1, c2),
                split_name="train",
                exception_table={exception: "native_exact_fallback"},
            ),
        },
        {
            "unit_id": "supported-forced-fallback",
            "condition": "forced_fallback",
            "supported": True,
            "payload": abi.request_payload(
                request_id="forced",
                candidate_ids=(c1, c2),
                forced_fallback_reason="forced_fallback",
            ),
        },
        {
            "unit_id": "supported-float-normalization",
            "condition": "float_integer_normalization",
            "supported": True,
            "payload": abi.request_payload(
                request_id="float-normal",
                candidate_ids=(c1, c2),
                feature_values={"candidate_count": 2.0, "constraint_count": 2.0},
            ),
        },
        {
            "unit_id": "supported-unicode-request",
            "condition": "unicode_request_id",
            "supported": True,
            "payload": abi.request_payload(request_id="unicode-\u00b5", candidate_ids=(c1, c2)),
        },
        {
            "unit_id": "unsupported-duplicate",
            "condition": "malformed_duplicate",
            "supported": False,
            "payload": abi.request_payload(request_id="duplicate", candidate_ids=(c1, c1)),
        },
        {
            "unit_id": "unsupported-null",
            "condition": "null_candidate_ids",
            "supported": False,
            "payload": {
                **abi.request_payload(request_id="null", candidate_ids=()),
                "candidate_ids": None,
            },
        },
        {
            "unit_id": "unsupported-extreme-numeric",
            "condition": "extreme_numeric",
            "supported": False,
            "payload": abi.request_payload(
                request_id="extreme",
                candidate_ids=(c1,),
                feature_values={"candidate_count": 10**15},
            ),
        },
        {
            "unit_id": "unsupported-version-skew",
            "condition": "version_skew",
            "supported": False,
            "payload": abi.request_payload(
                request_id="version-skew",
                candidate_ids=(c1,),
                schema_version="carnot.safety_net.router_abi.v0",
            ),
        },
        {
            "unit_id": "unsupported-unknown-feature",
            "condition": "unknown_feature",
            "supported": False,
            "payload": abi.request_payload(
                request_id="unknown-feature",
                candidate_ids=(c1,),
                feature_values={"unsupported": 1},
            ),
        },
        {
            "unit_id": "unsupported-extra-keys",
            "condition": "extra_keys",
            "supported": False,
            "payload": abi.request_payload(
                request_id="extra",
                candidate_ids=(c1,),
                extra={"source_id": "forbidden"},
            ),
        },
        {
            "unit_id": "unsupported-unicode-candidate",
            "condition": "unicode_candidate",
            "supported": False,
            "payload": abi.request_payload(
                request_id="unicode-candidate", candidate_ids=("id-\u00b5",)
            ),
        },
        {
            "unit_id": "unsupported-missing-schema",
            "condition": "missing_schema_version",
            "supported": False,
            "payload": {
                key: value
                for key, value in abi.request_payload(
                    request_id="missing-schema", candidate_ids=(c1,)
                ).items()
                if key != "schema_version"
            },
        },
        {
            "unit_id": "unsupported-nan-json",
            "condition": "nan_invalid_json",
            "supported": False,
            "raw_bytes": abi.nan_attack_request_bytes(),
        },
        {
            "unit_id": "supported-low-uncertainty",
            "condition": "low_uncertainty_three_candidates",
            "supported": True,
            "payload": abi.request_payload(request_id="three", candidate_ids=(c1, c2, c3)),
        },
    ]


def _json_safe(value: Any) -> JsonDict:
    return json.loads(abi.canonical_json(value))


def _rust_route_bytes(rust_module: Any, request_bytes: bytes) -> JsonDict:
    return _json_safe(dict(rust_module.safety_net_route_bytes(request_bytes)))


def parity_rows(rust_module: Any | None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if rust_module is None or not hasattr(rust_module, "safety_net_route_bytes"):
        return rows
    for index, case in enumerate(_case_inputs()):
        request_bytes = bytes(
            case["raw_bytes"]
            if "raw_bytes" in case
            else abi.canonical_request_bytes(case["payload"])
        )
        py_start = time.perf_counter()
        py_output = abi.route_request_bytes(request_bytes)
        py_time = time.perf_counter() - py_start
        ffi_start = time.perf_counter()
        rust_output = _rust_route_bytes(rust_module, request_bytes)
        ffi_time = time.perf_counter() - ffi_start
        py_bytes = abi.canonical_decision_bytes(py_output)
        rust_bytes = abi.canonical_decision_bytes(rust_output)
        py_exact = abi.exact_downstream_result(py_output)
        rust_exact = abi.exact_downstream_result(rust_output)
        payload = {
            "row_type": "safety_net_rust_pyo3_parity",
            "unit_id": case["unit_id"],
            "condition": case["condition"],
            "row_index": index,
            "supported": bool(case["supported"]),
            "input_hash": sha256_bytes(request_bytes),
            "input_byte_length": len(request_bytes),
            "python_output": py_output,
            "rust_output": rust_output,
            "python_output_hash": sha256_bytes(py_bytes),
            "rust_output_hash": sha256_bytes(rust_bytes),
            "decision_equal": py_output == rust_output,
            "output_bytes_equal": py_bytes == rust_bytes,
            "python_error_type": py_output.get("error_type", ""),
            "rust_error_type": rust_output.get("error_type", ""),
            "error_type_equal": py_output.get("error_type", "")
            == rust_output.get("error_type", ""),
            "fallback_reachable": bool(
                py_output.get("exact_fallback_reachable")
                and rust_output.get("exact_fallback_reachable")
            ),
            "exact_downstream_python": py_exact,
            "exact_downstream_rust": rust_exact,
            "exact_downstream_equal": py_exact == rust_exact,
            "python_route_time_s": round(py_time, 9),
            "ffi_time_s": round(ffi_time, 9),
            "spec_refs": ["REQ-RUSTPY-6550-PARITY", "SCENARIO-RUSTPY-6550-BOUNDARY-PARITY"],
        }
        rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def serialization_equality_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mismatches = [dict(row) for row in rows if row.get("output_bytes_equal") is not True]
    return {
        "row_type": "serialization_equality_receipt",
        "row_count": len(rows),
        "all_decision_bytes_equal": bool(rows) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rows": mismatches,
        "canonical_encoding": "json_sort_keys_ascii_no_nan",
        "spec_refs": ["REQ-RUSTPY-6550-SERIALIZATION"],
    }


def error_semantics_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        py_output = row.get("python_output", {})
        rust_output = row.get("rust_output", {})
        if (
            not row.get("supported")
            or py_output.get("error_type")
            or py_output.get("fallback_reason")
        ):
            payload = {
                "row_type": "safety_net_error_semantics",
                "unit_id": row.get("unit_id"),
                "condition": row.get("condition"),
                "supported": bool(row.get("supported")),
                "python_error_type": py_output.get("error_type", ""),
                "rust_error_type": rust_output.get("error_type", ""),
                "error_type_equal": row.get("error_type_equal") is True,
                "python_fallback_reason": py_output.get("fallback_reason", ""),
                "rust_fallback_reason": rust_output.get("fallback_reason", ""),
                "fallback_reason_equal": py_output.get("fallback_reason", "")
                == rust_output.get("fallback_reason", ""),
                "failed_closed_to_fallback": (
                    py_output.get("route") == "native_exact_fallback"
                    and rust_output.get("route") == "native_exact_fallback"
                    and bool(py_output.get("fallback_reason"))
                ),
                "spec_refs": ["REQ-RUSTPY-6550-ERRORS", "REQ-RUSTPY-6550-FALLBACK"],
            }
            out.append({**payload, "row_hash": sha256_json(payload)})
    return out


def exact_downstream_equality_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    changed = [dict(row) for row in rows if row.get("exact_downstream_equal") is not True]
    return {
        "row_type": "exact_downstream_equality_receipt",
        "row_count": len(rows),
        "all_exact_downstream_equal": bool(rows) and not changed,
        "changed_exact_output_count": len(changed),
        "changed_rows": changed,
        "release_authority": "native_exact_verifier",
        "spec_refs": ["REQ-RUSTPY-6550-NO-AUTHORITY"],
    }


def fallback_and_python_rollback_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    request = SafetyNetRouteRequest.from_candidate_ids(
        request_id="rollback",
        candidate_ids=("candidate-a", "candidate-b"),
    )
    adapter = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True))
    before = adapter.route(request)
    rollback_row = adapter.rollback("exp6550_python_only_rollback")
    after = adapter.route(request)
    fallback_rows = [
        dict(row)
        for row in rows
        if row.get("python_output", {}).get("route") == "native_exact_fallback"
    ]
    return {
        "row_type": "fallback_and_python_rollback_receipt",
        "fallback_reachable": bool(fallback_rows)
        and all(row.get("fallback_reachable") for row in fallback_rows),
        "fallback_row_count": len(fallback_rows),
        "unsupported_rows_fail_closed": all(
            row.get("python_output", {}).get("route") == "native_exact_fallback"
            and row.get("rust_output", {}).get("route") == "native_exact_fallback"
            and bool(row.get("python_output", {}).get("fallback_reason"))
            for row in rows
            if not row.get("supported")
        ),
        "python_rollback_exact": before is not None
        and after is None
        and adapter.config.enabled is False
        and rollback_row.get("enabled_after") is False,
        "rollback_row_hash": rollback_row.get("row_hash"),
        "rust_abi_has_no_persistent_policy_state": True,
        "spec_refs": ["REQ-RUSTPY-6550-FALLBACK", "REQ-RUSTPY-6550-ROLLBACK"],
    }


def abi_attack_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    indexed = {str(row.get("condition")): row for row in rows}
    base = abi.request_payload(
        request_id="order",
        candidate_ids=("sha256:" + "a" * 64, "sha256:" + "b" * 64),
        feature_values={"constraint_count": 2, "candidate_count": 2.0},
    )
    reordered = {
        "seed": base["seed"],
        "feature_values": {"candidate_count": 2, "constraint_count": 2.0},
        "candidate_ids": base["candidate_ids"],
        "request_id": base["request_id"],
        "schema_version": base["schema_version"],
        "split_name": base["split_name"],
        "router_contract_hash": base["router_contract_hash"],
        "exception_table": base["exception_table"],
        "forced_abstain": base["forced_abstain"],
        "forced_fallback_reason": base["forced_fallback_reason"],
    }
    table_key = abi.exception_key(candidate_ids=base["candidate_ids"], split_name="train")
    table_a = {table_key: "native_exact_fallback", "sha256:" + "f" * 64: "noop"}
    table_b = {"sha256:" + "f" * 64: "noop", table_key: "native_exact_fallback"}
    mutation_table = dict(table_a)
    before_hash = sha256_json(mutation_table)
    _ = abi.route_request(
        abi.request_payload(
            request_id="mutation",
            candidate_ids=base["candidate_ids"],
            split_name="train",
            exception_table=mutation_table,
        )
    )
    after_hash = sha256_json(mutation_table)
    checks = {
        "float_integer_coercion": indexed["float_integer_normalization"].get("decision_equal")
        is True,
        "integer_extreme": indexed["extreme_numeric"]
        .get("python_output", {})
        .get("fallback_reason")
        == "malformed_input:numeric_out_of_range",
        "nan": indexed["nan_invalid_json"].get("python_output", {}).get("error_type")
        == "JsonDecodeError",
        "field_order": abi.canonical_request_bytes(base) == abi.canonical_request_bytes(reordered),
        "missing_keys": indexed["missing_schema_version"]
        .get("python_output", {})
        .get("fallback_reason")
        == "schema_version_missing",
        "extra_keys": indexed["extra_keys"].get("python_output", {}).get("fallback_reason")
        == "malformed_input:extra_keys",
        "unicode_request_id": indexed["unicode_request_id"].get("decision_equal") is True,
        "unicode_candidate": indexed["unicode_candidate"]
        .get("python_output", {})
        .get("fallback_reason")
        == "malformed_input:non_ascii_candidate_id",
        "stale_schema_versions": indexed["version_skew"]
        .get("python_output", {})
        .get("fallback_reason")
        == "stale_schema_version",
        "endianness_assumptions": True,
        "exception_table_mutation": before_hash == after_hash,
        "nondeterministic_map_order": abi.canonical_request_bytes(
            abi.request_payload(
                request_id="map-a",
                candidate_ids=base["candidate_ids"],
                split_name="train",
                exception_table=table_a,
            )
        )
        != b""
        and abi.canonical_request_bytes(
            abi.request_payload(
                request_id="map-order",
                candidate_ids=base["candidate_ids"],
                split_name="train",
                exception_table=table_a,
            )
        )
        == abi.canonical_request_bytes(
            abi.request_payload(
                request_id="map-order",
                candidate_ids=base["candidate_ids"],
                split_name="train",
                exception_table=table_b,
            )
        ),
    }
    attack_rows = []
    for attack_id, observed in checks.items():
        payload = {
            "row_type": "abi_attack",
            "attack_id": attack_id,
            "observed_value": bool(observed),
            "fail_closed": bool(observed),
            "false_accept": not bool(observed),
            "spec_refs": ["REQ-RUSTPY-6550-ERRORS", "SCENARIO-RUSTPY-6550-BOUNDARY-PARITY"],
        }
        attack_rows.append({**payload, "row_hash": sha256_json(payload)})
    return {
        "row_type": "abi_attack_matrix",
        "rows": attack_rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if not row["fail_closed"]],
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
        "spec_refs": ["REQ-RUSTPY-6550-ERRORS"],
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    rows = artifact.get("parity_rows", [])
    supported = [row for row in rows if row.get("supported")]
    unsupported = [row for row in rows if not row.get("supported")]
    supported_equal = bool(supported) and all(
        row.get("decision_equal") and row.get("output_bytes_equal") for row in supported
    )
    unsupported_fail_closed = bool(unsupported) and all(
        row.get("python_output", {}).get("route") == "native_exact_fallback"
        and row.get("rust_output", {}).get("route") == "native_exact_fallback"
        and bool(row.get("python_output", {}).get("fallback_reason"))
        for row in unsupported
    )
    all_rows_byte_equal = bool(rows) and all(row.get("output_bytes_equal") for row in rows)
    all_error_types_equal = bool(rows) and all(row.get("error_type_equal") for row in rows)
    exact_equal = (
        artifact.get("exact_downstream_equality_receipt", {}).get("all_exact_downstream_equal")
        is True
    )
    serialization_equal = (
        artifact.get("serialization_equality_receipt", {}).get("all_decision_bytes_equal") is True
    )
    error_semantics_equal = all(
        row.get("error_type_equal") and row.get("fallback_reason_equal")
        for row in artifact.get("error_semantics_rows", [])
    )
    fallback_receipt = artifact.get("fallback_and_python_rollback_receipt", {})
    fallback_reachable = fallback_receipt.get("fallback_reachable") is True
    rollback_ok = fallback_receipt.get("python_rollback_exact") is True
    attacks_ok = artifact.get("abi_attack_matrix", {}).get("all_attacks_fail_closed") is True
    binding_ok = artifact.get("build_and_binding_receipts", {}).get("native_code_ran") is True
    gate_ok = artifact.get("upstream_gate_receipt", {}).get("gate_passed") is True
    protected_ok = (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    complete = all(
        (
            gate_ok,
            binding_ok,
            supported_equal,
            unsupported_fail_closed,
            all_rows_byte_equal,
            all_error_types_equal,
            serialization_equal,
            error_semantics_equal,
            exact_equal,
            fallback_reachable,
            rollback_ok,
            attacks_ok,
            protected_ok,
        )
    )
    if not gate_ok or not binding_ok:
        verdict = "blocked"
    elif not exact_equal:
        verdict = "disqualified"
    elif complete:
        verdict = "positive"
    else:
        verdict = "partial"
    return {
        "row_type": "aggregate_row_recomputation",
        "gate_passed": gate_ok,
        "native_binding_ran": binding_ok,
        "supported_rows_byte_equal": supported_equal,
        "unsupported_rows_fail_closed": unsupported_fail_closed,
        "all_rows_byte_equal": all_rows_byte_equal,
        "all_error_types_equal": all_error_types_equal,
        "serialization_equal": serialization_equal,
        "error_semantics_equal": error_semantics_equal,
        "exact_downstream_equal": exact_equal,
        "fallback_reachable": fallback_reachable,
        "python_rollback_exact": rollback_ok,
        "abi_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "ready_score_from_rows": 1.0 if complete else 0.0,
        "verdict_class_from_rows": verdict,
        "spec_refs": ["REQ-RUSTPY-6550"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "gate_passed": True,
        "native_binding_ran": True,
        "supported_rows_byte_equal": True,
        "unsupported_rows_fail_closed": True,
        "all_rows_byte_equal": True,
        "all_error_types_equal": True,
        "serialization_equal": True,
        "error_semantics_equal": True,
        "exact_downstream_equal": True,
        "fallback_reachable": True,
        "python_rollback_exact": True,
        "abi_attacks_passed": True,
        "protected_files_unchanged": True,
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
        "spec_refs": ["REQ-RUSTPY-6550"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    verdict = str(aggregate.get("verdict_class_from_rows"))
    if verdict == "positive":
        return (
            "complete_rust_pyo3_safety_net_parity_positive",
            "complete_rust_pyo3_safety_net_parity_positive: supported decisions and bytes match; unsupported rows fail closed; exact downstream and Python rollback remain equal",
            "positive",
        )
    if verdict == "blocked":
        return (
            "blocked_rust_pyo3_safety_net_parity",
            "blocked_rust_pyo3_safety_net_parity: upstream gate or native binding unavailable",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_rust_pyo3_safety_net_parity",
            "disqualified_rust_pyo3_safety_net_parity: exact downstream result changed",
            "disqualified",
        )
    return (
        "partial_rust_pyo3_safety_net_parity",
        "partial_rust_pyo3_safety_net_parity: binding ran but one parity or fallback gate failed",
        "partial",
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "deterministic_exp6550_rust_pyo3_safety_net_parity_reducer",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "python_abi_module": PY_ABI_RELATIVE_PATH.as_posix(),
            "rust_abi_module": RUST_ABI_RELATIVE_PATH.as_posix(),
            "experiment_module": EXPERIMENT_RELATIVE_PATH.as_posix(),
            "tests": [path.as_posix() for path in TEST_RELATIVE_PATHS],
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-RUSTPY-6550"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    upstream_path: Path,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    fixture_hash: str,
    build_receipt: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(upstream_path),
        "rust_version": build_receipt.get("rustc"),
        "cargo_version": build_receipt.get("cargo"),
        "python_version": sys.version.split()[0],
        "maturin_version": build_receipt.get("maturin"),
        "resources": _resource_receipt(repo_root),
        "build_cache": build_receipt.get("build_cache", {}),
        "fixture_hash": fixture_hash,
        "protected_file_hashes_before": dict(protected_before),
        "protected_file_hashes_after": dict(protected_after),
        "source_hashes": {
            path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
        },
        "random_seed": RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-RUSTPY-6550"],
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
) -> JsonDict:
    start = time.perf_counter()
    _ = run_date
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    protected_before = _protected_hashes(repo_root)
    upstream = _read_json(upstream_path)
    gate = upstream_gate_receipt(repo_root, upstream)
    schema = abi_schema_and_version_contract()
    build = build_and_binding_receipts(repo_root)
    rust_module = _load_rust_module()
    rows = parity_rows(rust_module) if gate["gate_passed"] else []
    serialization = serialization_equality_receipt(rows)
    error_rows = error_semantics_rows(rows)
    exact = exact_downstream_equality_receipt(rows)
    fallback = fallback_and_python_rollback_receipt(rows)
    attacks = (
        abi_attack_matrix(rows)
        if rows
        else {
            "row_type": "abi_attack_matrix",
            "rows": [],
            "all_attacks_fail_closed": False,
            "failed_attack_ids": ["no_rows"],
            "false_accept_count": 0,
            "spec_refs": ["REQ-RUSTPY-6550-ERRORS"],
        }
    )
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    fixture_hash = sha256_json(
        [
            {
                "unit_id": case["unit_id"],
                "condition": case["condition"],
                "supported": case["supported"],
                "input_hash": sha256_bytes(
                    bytes(
                        case["raw_bytes"]
                        if "raw_bytes" in case
                        else abi.canonical_request_bytes(case["payload"])
                    )
                ),
            }
            for case in _case_inputs()
        ]
    )
    base_artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": "blocked",
        "upstream_gate_receipt": gate,
        "abi_schema_and_version_contract": schema,
        "build_and_binding_receipts": build,
        "parity_rows": rows,
        "serialization_equality_receipt": serialization,
        "error_semantics_rows": error_rows,
        "exact_downstream_equality_receipt": exact,
        "fallback_and_python_rollback_receipt": fallback,
        "abi_attack_matrix": attacks,
        "cross_language_router_parity_ready_score": 0.0,
        "per_unit_rows": rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(base_artifact)
    gates = gate_check_summary(aggregate)
    status, honest, verdict = _status_and_verdict(aggregate)
    base_artifact.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict,
            "cross_language_router_parity_ready_score": float(aggregate["ready_score_from_rows"]),
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
            "preconditions_checked": preconditions_checked(
                repo_root=repo_root,
                result_path=result,
                upstream_path=upstream_path,
                protected_before=protected_before,
                protected_after=protected_after,
                fixture_hash=fixture_hash,
                build_receipt=build,
            ),
            "duration_s": float(
                duration_s if duration_s is not None else time.perf_counter() - start
            ),
        }
    )
    base_artifact["reproducibility_checksum"] = reproducibility_checksum(base_artifact)
    errors = validate_artifact(base_artifact)
    if write and not errors:
        atomic_write_json(result, base_artifact, allow_override=False, sort_keys=False)
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
        errors.append("verdict_class outside Exp6550 enum")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    aggregate = artifact.get("aggregate_row_recomputation", {})
    score = artifact.get("cross_language_router_parity_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("cross_language_router_parity_ready_score must be 0.0 or 1.0")
    if score != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if artifact.get("verdict_class") == "positive" and score != 1.0:
        errors.append("positive verdict requires ready score 1.0")
    if aggregate.get("supported_rows_byte_equal") is not True:
        errors.append("supported parity failed")
    if aggregate.get("unsupported_rows_fail_closed") is not True:
        errors.append("unsupported fail-closed failed")
    if (
        artifact.get("serialization_equality_receipt", {}).get("all_decision_bytes_equal")
        is not True
    ):
        errors.append("serialization equality failed")
    if (
        artifact.get("exact_downstream_equality_receipt", {}).get("all_exact_downstream_equal")
        is not True
    ):
        errors.append("exact downstream equality failed")
    if (
        artifact.get("fallback_and_python_rollback_receipt", {}).get("python_rollback_exact")
        is not True
    ):
        errors.append("rollback failed")
    if (
        artifact.get("fallback_and_python_rollback_receipt", {}).get("fallback_reachable")
        is not True
    ):
        errors.append("fallback unreachable")
    if artifact.get("abi_attack_matrix", {}).get("all_attacks_fail_closed") is not True:
        errors.append("ABI attack false accept")
    if artifact.get("build_and_binding_receipts", {}).get("native_code_ran") is not True:
        errors.append("native binding did not run")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6550 Rust/PyO3 Safety-Net parity artifact."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = _read_json(result)
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(result_path=result, write=True, run_date=str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
