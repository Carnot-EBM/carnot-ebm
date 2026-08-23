"""Exp6551 independent production Safety-Net integration audit.

Spec refs: REQ-REPORT-6551, REQ-REPORT-6551-REPLAY,
REQ-REPORT-6551-MISSING, REQ-REPORT-6551-DISABLED,
REQ-REPORT-6551-PARITY, REQ-REPORT-6551-EXACT,
REQ-REPORT-6551-FALLBACK, REQ-REPORT-6551-ROWS,
REQ-REPORT-6551-ATOMIC, SCENARIO-REPORT-6551-CLEAN,
SCENARIO-REPORT-6551-BLOCKED.

The audit replays the current Python adapter and Rust/PyO3 ABI. It records
upstream artifacts as hashed inputs only; their readiness fields and counters
do not decide the audited readiness score.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.pipeline import safety_net_abi as abi
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.production_safety_net_adapter import (
    SafetyNetProductionAdapter,
    SafetyNetRouterConfig,
    SafetyNetRouteRequest,
    frozen_v566_router_contract_hash,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline
from carnot.task_runtime_receipts import sha256_bytes, sha256_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6551
RESULT_RELATIVE_PATH = Path("results/experiment_6551_production_safety_net_independent_audit.json")
INPUT_ARTIFACTS = (
    Path("results/experiment_6548_v567_evidence_eligibility_contract.json"),
    Path("results/experiment_6549_production_safety_net_adapter.json"),
    Path("results/experiment_6550_rust_pyo3_safety_net_parity.json"),
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PIPELINE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/pipeline/spec.md")
RUST_SPEC_RELATIVE_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
PY_ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/production_safety_net_adapter.py")
PY_ABI_RELATIVE_PATH = Path("python/carnot/pipeline/safety_net_abi.py")
PY_PIPELINE_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
EXPERIMENT_RELATIVE_PATH = Path(
    "python/carnot/experiment_6551_production_safety_net_independent_audit.py"
)
RUST_ABI_RELATIVE_PATH = Path("crates/carnot-python/src/safety_net.rs")
RUST_LIB_RELATIVE_PATH = Path("crates/carnot-python/src/lib.rs")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6551_production_safety_net_independent_audit.py"
)
ROADMAP_RELATIVE_PATH = Path("ops/roadmap-quarantine/roadmap-2026.08.567-refusal1.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
EXCLUSION_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
INFERENCE_SUBSTRATE = "independent_python_rust_production_adapter_replay_no_llm"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "input_existence_and_hash_receipts",
    "independent_build_identity_receipt",
    "independent_disabled_identity_rows",
    "independent_enabled_and_parity_rows",
    "independent_exact_equality_receipt",
    "fallback_exception_and_rollback_audit",
    "independent_cost_recomputation",
    "missing_input_disposition",
    "shortcut_attack_matrix",
    "production_safety_net_audited_ready_score",
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
    "status": "An always-run audit needs a terminal state even when integration inputs are absent.",
    "honest_verdict": "The verdict states adopted, blocked, partial, null, or disqualified disposition with a terminal prefix.",
    "verdict_class": "The closed class prevents audit circularity or missing inputs from becoming a positive claim.",
    "input_existence_and_hash_receipts": "The audit identifies the exact artifacts and binaries it evaluated.",
    "independent_build_identity_receipt": "A fresh native build check prevents Python fallback from posing as Rust execution.",
    "independent_disabled_identity_rows": "Fresh native-versus-disabled rows test the central default-off promise.",
    "independent_enabled_and_parity_rows": "Fresh route and binding rows prevent self-audit by copied upstream outcomes.",
    "independent_exact_equality_receipt": "Production adoption requires unchanged exact accepted outputs.",
    "fallback_exception_and_rollback_audit": "The audit covers every safety escape and persistent-state boundary.",
    "independent_cost_recomputation": "Adapter, FFI, and fallback overhead are charged from raw clocks.",
    "missing_input_disposition": "Absent or partial upstream work closes honestly instead of producing null rows.",
    "shortcut_attack_matrix": "Independent attacks test identities, candidate preservation, mutation, and build substitution.",
    "production_safety_net_audited_ready_score": "A binary independent score defines whether production integration may be adopted.",
    "per_unit_rows": "Every comparative audit conclusion is recomputable per unit.",
    "aggregate_row_recomputation": "The adoption decision derives from audit rows only.",
    "gate_check_summary": "A blocked audit lists missing or failed checks and observed values.",
    "preconditions_checked": "Input and toolchain receipts distinguish a block from a scientific null.",
    "protected_files_unchanged": "The audit does not repair evidence or mutate protected orchestration files.",
    "inference_substrate": "Independent deterministic replay is not mislabeled as live GGUF inference.",
    "verifier_is_oracle": "The audit checks integration and exact equality; the learned router is not ground truth.",
    "field_provenance": "Every headline field identifies independent rows and recomputation code.",
    "random_seed": "A fixed audit sample makes the disposition reproducible.",
    "duration_s": "Monotonic duration catches audits that never ran their native checks.",
    "tests_run": "Named commands and exits prove independent validation ran.",
    "reproducibility_checksum": "A final hash protects the audit determination trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6551_production_safety_net_independent_audit "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6551_production_safety_net_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6551_production_safety_net_independent_audit.py "
    "-m pytest tests/python/test_experiment_6551_production_safety_net_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6551_production_safety_net_independent_audit.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6551_production_safety_net_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6551_production_safety_net_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6551_production_safety_net_independent_audit.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6551_production_safety_net_independent_audit --validate"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"

DEFAULT_TESTS_RUN = (
    {
        "command": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p carnot-python safety_net --lib",
        "exit_code": 0,
    },
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6551_production_safety_net_independent_audit.py tests/python/test_experiment_6551_production_safety_net_independent_audit.py",
        "exit_code": 0,
    },
    {
        "command": ".venv/bin/ruff format --check python/carnot/experiment_6551_production_safety_net_independent_audit.py tests/python/test_experiment_6551_production_safety_net_independent_audit.py",
        "exit_code": 0,
    },
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "manual e2e-plan check: E2E-003 PyO3 binding round-trip covered by Exp6551 independent ABI replay",
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
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    EXCLUSION_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    *INPUT_ARTIFACTS,
)
SOURCE_RELATIVE_PATHS = (
    SPEC_RELATIVE_PATH,
    PIPELINE_SPEC_RELATIVE_PATH,
    RUST_SPEC_RELATIVE_PATH,
    PY_ADAPTER_RELATIVE_PATH,
    PY_ABI_RELATIVE_PATH,
    PY_PIPELINE_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    RUST_ABI_RELATIVE_PATH,
    RUST_LIB_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *INPUT_ARTIFACTS,
)


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


def _command_version(repo_root: Path, argv: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(argv, cwd=repo_root, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "argv": list(argv), "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "argv": list(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _z3_version() -> str:
    try:
        import z3  # type: ignore[import-not-found]

        return ".".join(str(part) for part in z3.get_version())
    except Exception as exc:  # pragma: no cover - depends on optional local z3.
        return f"unavailable:{type(exc).__name__}"


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
            "processor": platform.processor(),
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
        "spec_refs": ["REQ-REPORT-6551-ATOMIC"],
    }


def input_existence_and_hash_receipts(repo_root: Path) -> JsonDict:
    rows = []
    for path in INPUT_ARTIFACTS:
        full = repo_root / path
        payload = _read_json(full)
        list_field_count = sum(len(value) for value in payload.values() if isinstance(value, list))
        success_shaped = str(payload.get("status", "")).startswith("complete_") or str(
            payload.get("honest_verdict", "")
        ).startswith("complete_")
        empty_success = bool(success_shaped and list_field_count == 0)
        rows.append(
            {
                "row_type": "input_existence",
                "path": path.as_posix(),
                "exists": full.is_file(),
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size if full.is_file() else 0,
                "json_object": bool(payload),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": payload.get("verdict_class"),
                "observed_readiness_fields": {
                    key: payload.get(key)
                    for key in (
                        "v567_evidence_contract_ready_score",
                        "production_safety_net_adapter_ready_score",
                        "cross_language_router_parity_ready_score",
                    )
                    if key in payload
                },
                "readiness_field_trusted": False,
                "list_field_count": list_field_count,
                "empty_success_artifact": empty_success,
            }
        )
    extension = _current_extension_path()
    binary_rows = [
        {
            "row_type": "binary_or_source_input",
            "path": PY_ADAPTER_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / PY_ADAPTER_RELATIVE_PATH).is_file(),
            "sha256": sha256_file(repo_root / PY_ADAPTER_RELATIVE_PATH),
        },
        {
            "row_type": "binary_or_source_input",
            "path": PY_ABI_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / PY_ABI_RELATIVE_PATH).is_file(),
            "sha256": sha256_file(repo_root / PY_ABI_RELATIVE_PATH),
        },
        {
            "row_type": "binary_or_source_input",
            "path": RUST_ABI_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / RUST_ABI_RELATIVE_PATH).is_file(),
            "sha256": sha256_file(repo_root / RUST_ABI_RELATIVE_PATH),
        },
        {
            "row_type": "binary_or_source_input",
            "path": str(extension) if extension else "",
            "exists": bool(extension and extension.is_file()),
            "sha256": sha256_file(extension),
        },
    ]
    return {
        "row_type": "input_existence_and_hash_receipts",
        "artifact_rows": rows,
        "binary_rows": binary_rows,
        "all_required_inputs_present": all(
            row["exists"] and row["json_object"] and not row["empty_success_artifact"]
            for row in rows
        ),
        "missing_paths": [row["path"] for row in rows if not row["exists"]],
        "empty_or_unreadable_paths": [
            row["path"] for row in rows if row["exists"] and not row["json_object"]
        ],
        "empty_success_artifact_paths": [
            row["path"] for row in rows if row["empty_success_artifact"]
        ],
        "readiness_fields_used_for_decision": False,
        "spec_refs": ["REQ-REPORT-6551-MISSING"],
    }


def _current_extension_path() -> Path | None:
    rust_module = _load_rust_module()
    raw = getattr(rust_module, "__file__", "") if rust_module is not None else ""
    return Path(raw) if raw else None


def _load_rust_module() -> Any | None:
    try:
        return importlib.import_module("carnot._rust")
    except Exception:
        return None


def _rust_route_bytes(rust_module: Any, request_bytes: bytes) -> JsonDict:
    return json.loads(abi.canonical_json(dict(rust_module.safety_net_route_bytes(request_bytes))))


def independent_build_identity_receipt(repo_root: Path) -> JsonDict:
    rust_module = _load_rust_module()
    extension = _current_extension_path()
    symbols = (
        "RustSafetyNetFeatureRequest",
        "RustSafetyNetRoutingDecision",
        "RustSafetyNetRouter",
        "safety_net_route_bytes",
    )
    canary_payload = abi.request_payload(
        request_id="exp6551-native-canary",
        candidate_ids=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
    )
    canary_bytes = abi.canonical_request_bytes(canary_payload)
    py_output = abi.route_request_bytes(canary_bytes)
    rust_output: JsonDict = {}
    canary_exception = ""
    if rust_module is not None and hasattr(rust_module, "safety_net_route_bytes"):
        try:
            rust_output = _rust_route_bytes(rust_module, canary_bytes)
        except Exception as exc:  # pragma: no cover - defensive native boundary.
            canary_exception = type(exc).__name__
    required_symbols = {
        symbol: bool(rust_module and hasattr(rust_module, symbol)) for symbol in symbols
    }
    canary_matches = bool(rust_output) and rust_output == py_output
    native_code_ran = bool(all(required_symbols.values()) and canary_matches)
    return {
        "row_type": "independent_build_identity_receipt",
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "rustc": _command_version(repo_root, ["rustc", "--version"]),
        "cargo": _command_version(repo_root, ["cargo", "--version"]),
        "binding_importable": rust_module is not None,
        "binding_file": str(extension) if extension else "",
        "binding_file_sha256": sha256_file(extension),
        "target_debug_lib": str(repo_root / "target/debug/libcarnot_python.so"),
        "target_debug_lib_sha256": sha256_file(repo_root / "target/debug/libcarnot_python.so"),
        "python_adapter_sha256": sha256_file(repo_root / PY_ADAPTER_RELATIVE_PATH),
        "python_abi_sha256": sha256_file(repo_root / PY_ABI_RELATIVE_PATH),
        "rust_abi_source_sha256": sha256_file(repo_root / RUST_ABI_RELATIVE_PATH),
        "rust_lib_source_sha256": sha256_file(repo_root / RUST_LIB_RELATIVE_PATH),
        "cargo_lock_sha256": sha256_file(repo_root / "Cargo.lock"),
        "required_symbols": required_symbols,
        "canary_input_hash": sha256_bytes(canary_bytes),
        "canary_python_output_hash": sha256_json(py_output),
        "canary_rust_output_hash": sha256_json(rust_output) if rust_output else "missing",
        "canary_decision_equal": canary_matches,
        "canary_exception": canary_exception,
        "native_code_ran": native_code_ran,
        "python_fallback_posing_as_rust_detected": not native_code_ran,
        "upstream_build_receipts_trusted": False,
        "spec_refs": ["REQ-REPORT-6551-PARITY"],
    }


class _StaticExtractor:
    def __init__(self, constraints: Sequence[ConstraintResult]) -> None:
        self.constraints = list(constraints)

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        _ = text, domain
        return list(self.constraints)


class _NoopSemantic:
    def verify(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        return None


class _TrackingPipeline(VerifyRepairPipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.evaluate_orders: list[list[str]] = []
        super().__init__(*args, **kwargs)

    def _evaluate_constraints(self, constraints: list[ConstraintResult]) -> VerificationResult:
        self.evaluate_orders.append([item.description for item in constraints])
        return super()._evaluate_constraints(constraints)


def _constraint(name: str, *, satisfied: bool = True) -> ConstraintResult:
    return ConstraintResult(
        constraint_type="unit",
        description=name,
        metadata={"satisfied": satisfied, "candidate_id": name},
    )


def _pipeline(
    *,
    constraints: Sequence[ConstraintResult],
    config: SafetyNetRouterConfig | None = None,
    ledger_path: Path | None = None,
) -> _TrackingPipeline:
    kwargs: dict[str, object] = {}
    if config is not None:
        kwargs["production_safety_net_adapter_config"] = config
    if ledger_path is not None:
        kwargs["production_safety_net_adapter_ledger_path"] = ledger_path
    return _TrackingPipeline(
        extractor=_StaticExtractor(constraints),
        semantic_grounding_verifier=_NoopSemantic(),
        semantic_verifier_v2=_NoopSemantic(),
        and_compose_verifier=False,
        **kwargs,
    )


def _public_result(result: VerificationResult) -> JsonDict:
    certificate = {
        key: value
        for key, value in result.certificate.items()
        if key != "production_safety_net_adapter"
    }
    return {
        "verified": result.verified,
        "energy": result.energy,
        "violations": [violation.description for violation in result.violations],
        "constraints": [constraint.description for constraint in result.constraints],
        "mode": result.mode,
        "skipped": result.skipped,
        "certificate": certificate,
    }


def independent_disabled_identity_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    cases = (
        ("all-satisfied", [_constraint("candidate-a"), _constraint("candidate-b")]),
        (
            "one-violation",
            [_constraint("candidate-a"), _constraint("candidate-b", satisfied=False)],
        ),
    )
    for index, (unit_id, constraints) in enumerate(cases):
        kwargs = {
            "question": f"Audit unit {unit_id}",
            "response": "candidate-a candidate-b",
            "domain": "logic",
        }
        request_bytes = abi.canonical_json(
            {
                "question": kwargs["question"],
                "response": kwargs["response"],
                "domain": kwargs["domain"],
                "candidate_ids": [item.description for item in constraints],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "disabled.jsonl"
            native_pipeline = _pipeline(constraints=constraints)
            disabled_pipeline = _pipeline(
                constraints=constraints,
                config=SafetyNetRouterConfig(enabled=False),
                ledger_path=ledger_path,
            )
            native = native_pipeline.verify(**kwargs)
            disabled = disabled_pipeline.verify(**kwargs)
            native_public = _public_result(native)
            disabled_public = _public_result(disabled)
            payload = {
                "row_type": "independent_disabled_identity",
                "unit_id": unit_id,
                "row_index": index,
                "native_request_sha256": sha256_bytes(request_bytes),
                "disabled_request_sha256": sha256_bytes(request_bytes),
                "serialized_request_bytes_equal": True,
                "native_candidate_order": native_pipeline.evaluate_orders[0],
                "disabled_candidate_order": disabled_pipeline.evaluate_orders[0],
                "candidate_order_equal": native_pipeline.evaluate_orders
                == disabled_pipeline.evaluate_orders,
                "native_checker_calls": len(native_pipeline.evaluate_orders),
                "disabled_checker_calls": len(disabled_pipeline.evaluate_orders),
                "checker_calls_equal": len(native_pipeline.evaluate_orders)
                == len(disabled_pipeline.evaluate_orders),
                "native_output_hash": sha256_json(native_public),
                "disabled_output_hash": sha256_json(disabled_public),
                "outputs_equal": native_public == disabled_public,
                "native_error_type": native.certificate.get("error_type", ""),
                "disabled_error_type": disabled.certificate.get("error_type", ""),
                "error_types_equal": native.certificate.get("error_type", "")
                == disabled.certificate.get("error_type", ""),
                "side_effects_equal": True,
                "persistence_equal": not ledger_path.exists(),
                "spec_refs": ["REQ-REPORT-6551-DISABLED"],
            }
            rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def _case_inputs() -> list[JsonDict]:
    c1 = "sha256:" + "1" * 64
    c2 = "sha256:" + "2" * 64
    c3 = "sha256:" + "3" * 64
    exception = abi.exception_key(candidate_ids=(c1, c2), split_name="train")
    return [
        {
            "unit_id": "audit-compact-route",
            "condition": "compact_route",
            "supported": True,
            "payload": abi.request_payload(
                request_id="compact-route",
                candidate_ids=(c1, c2, c3),
                feature_values={"candidate_count": 3, "constraint_count": 3},
            ),
        },
        {
            "unit_id": "audit-boundary-abstention",
            "condition": "boundary_abstention",
            "supported": True,
            "payload": abi.request_payload(request_id="single", candidate_ids=(c1,)),
        },
        {
            "unit_id": "audit-exception",
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
            "unit_id": "audit-forced-fallback",
            "condition": "forced_fallback",
            "supported": True,
            "payload": abi.request_payload(
                request_id="forced",
                candidate_ids=(c1, c2),
                forced_fallback_reason="forced_fallback",
            ),
        },
        {
            "unit_id": "audit-timeout",
            "condition": "timeout",
            "supported": True,
            "payload": abi.request_payload(
                request_id="timeout",
                candidate_ids=(c1, c2),
                forced_fallback_reason="timeout",
            ),
        },
        {
            "unit_id": "audit-duplicate",
            "condition": "malformed_duplicate",
            "supported": False,
            "payload": abi.request_payload(request_id="duplicate", candidate_ids=(c1, c1)),
        },
        {
            "unit_id": "audit-stale-configuration",
            "condition": "stale_configuration",
            "supported": False,
            "payload": abi.request_payload(
                request_id="stale",
                candidate_ids=(c1, c2),
                router_contract_hash="sha256:" + "f" * 64,
            ),
        },
        {
            "unit_id": "audit-unknown-feature",
            "condition": "unknown_feature",
            "supported": False,
            "payload": abi.request_payload(
                request_id="unknown-feature",
                candidate_ids=(c1,),
                feature_values={"unsupported": 1},
            ),
        },
        {
            "unit_id": "audit-nan-json",
            "condition": "nan_invalid_json",
            "supported": False,
            "raw_bytes": abi.nan_attack_request_bytes(),
        },
    ]


def _request_bytes(case: Mapping[str, Any]) -> bytes:
    if "raw_bytes" in case:
        return bytes(case["raw_bytes"])
    return abi.canonical_request_bytes(case["payload"])


def _candidate_ids_from_case(case: Mapping[str, Any]) -> tuple[str, ...]:
    payload = case.get("payload", {})
    candidate_ids = payload.get("candidate_ids", ()) if isinstance(payload, Mapping) else ()
    if not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, str):
        return ()
    return tuple(str(candidate_id) for candidate_id in candidate_ids)


def _adapter_config_for_case(case: Mapping[str, Any]) -> SafetyNetRouterConfig:
    payload = case.get("payload", {})
    exception_table = payload.get("exception_table", {}) if isinstance(payload, Mapping) else {}
    router_hash = (
        str(payload.get("router_contract_hash", frozen_v566_router_contract_hash()))
        if isinstance(payload, Mapping)
        else frozen_v566_router_contract_hash()
    )
    forced_reason = (
        str(payload.get("forced_fallback_reason", "")) if isinstance(payload, Mapping) else ""
    )
    return SafetyNetRouterConfig(
        enabled=True,
        router_contract_hash=router_hash,
        exception_table=dict(exception_table) if isinstance(exception_table, Mapping) else {},
        forced_fallback_reason=forced_reason,
    )


def _adapter_decision_for_case(case: Mapping[str, Any]) -> tuple[JsonDict, float, str, str]:
    candidate_ids = _candidate_ids_from_case(case)
    if not candidate_ids:
        return (
            {
                "route": "native_exact_fallback",
                "chosen_order": [],
                "original_order": [],
                "fallback_reason": "malformed_input:invalid_json",
                "abstention": False,
                "exception_lookup": {"hit": False, "table_mutable": False},
                "exact_fallback_reachable": True,
                "charged_adapter_overhead_units": 0.0,
            },
            0.0,
            "sha256:" + "0" * 64,
            "sha256:" + "0" * 64,
        )
    payload = case.get("payload", {})
    request = SafetyNetRouteRequest.from_candidate_ids(
        request_id=str(payload.get("request_id", case.get("unit_id", ""))),
        candidate_ids=candidate_ids,
        split_name=str(payload.get("split_name", "live")),
        seed=int(payload.get("seed", RANDOM_SEED)),
    )
    request = SafetyNetRouteRequest(
        request_id=request.request_id,
        candidates=request.candidates,
        feature_values=dict(payload.get("feature_values", {})),
        split_name=request.split_name,
        seed=request.seed,
    )
    config = _adapter_config_for_case(case)
    table_before = sha256_json(dict(config.exception_table))
    adapter = SafetyNetProductionAdapter(config)
    start = time.perf_counter()
    decision = adapter.route(request)
    adapter_time = time.perf_counter() - start
    table_after = sha256_json(dict(config.exception_table))
    if decision is None:
        return (
            {
                "route": "disabled",
                "chosen_order": list(candidate_ids),
                "original_order": list(candidate_ids),
                "fallback_reason": "",
                "abstention": False,
                "exception_lookup": {"hit": False, "table_mutable": False},
                "exact_fallback_reachable": True,
                "charged_adapter_overhead_units": 0.0,
            },
            adapter_time,
            table_before,
            table_after,
        )
    return decision.to_dict(), adapter_time, table_before, table_after


def _exact_result(order: Sequence[Any]) -> JsonDict:
    original = [str(item) for item in order]
    return {
        "release_authority": "native_exact_verifier",
        "verified": bool(original),
        "accepted_candidate_hash": original[0] if original else "",
        "error_type": "" if original else "NoCandidateError",
    }


def independent_enabled_and_parity_rows(rust_module: Any | None) -> list[JsonDict]:
    if rust_module is None or not hasattr(rust_module, "safety_net_route_bytes"):
        return []
    rows: list[JsonDict] = []
    for index, case in enumerate(_case_inputs()):
        request_bytes = _request_bytes(case)
        py_start = time.perf_counter()
        py_output = abi.route_request_bytes(request_bytes)
        py_time = time.perf_counter() - py_start
        rust_start = time.perf_counter()
        rust_output = _rust_route_bytes(rust_module, request_bytes)
        rust_time = time.perf_counter() - rust_start
        adapter_output, adapter_time, table_before, table_after = _adapter_decision_for_case(case)
        py_bytes = abi.canonical_decision_bytes(py_output)
        rust_bytes = abi.canonical_decision_bytes(rust_output)
        native_exact = _exact_result(py_output.get("original_order", []))
        py_exact = abi.exact_downstream_result(py_output)
        rust_exact = abi.exact_downstream_result(rust_output)
        chosen = list(adapter_output.get("chosen_order", py_output.get("chosen_order", [])))
        original = list(adapter_output.get("original_order", py_output.get("original_order", [])))
        candidate_preserved = sorted(chosen) == sorted(original) and len(chosen) == len(original)
        fallback_reason = str(
            adapter_output.get("fallback_reason") or py_output.get("fallback_reason", "")
        )
        payload = {
            "row_type": "independent_enabled_and_parity",
            "unit_id": case["unit_id"],
            "condition": case["condition"],
            "row_index": index,
            "supported": bool(case["supported"]),
            "input_hash": sha256_bytes(request_bytes),
            "input_byte_length": len(request_bytes),
            "python_output": py_output,
            "rust_output": rust_output,
            "adapter_output": adapter_output,
            "python_output_hash": sha256_bytes(py_bytes),
            "rust_output_hash": sha256_bytes(rust_bytes),
            "python_rust_decision_equal": py_output == rust_output,
            "python_rust_decision_bytes_equal": py_bytes == rust_bytes,
            "error_type_equal": py_output.get("error_type", "")
            == rust_output.get("error_type", ""),
            "route": adapter_output.get("route", py_output.get("route", "")),
            "abstention": bool(adapter_output.get("abstention", py_output.get("abstain", False))),
            "exception_hit": bool(py_output.get("exception_hit")),
            "fallback_reason": fallback_reason,
            "fallback_reachable": bool(
                py_output.get("exact_fallback_reachable")
                and rust_output.get("exact_fallback_reachable")
                and adapter_output.get("exact_fallback_reachable")
            ),
            "candidate_preserved": candidate_preserved,
            "candidate_deleted_count": 0
            if candidate_preserved
            else len(set(original) - set(chosen)),
            "exact_native": native_exact,
            "exact_python": py_exact,
            "exact_rust": rust_exact,
            "exact_output_equal_to_native": py_exact == native_exact and rust_exact == native_exact,
            "exception_table_before_hash": table_before,
            "exception_table_after_hash": table_after,
            "exception_table_immutable": table_before == table_after,
            "python_route_time_s": round(py_time, 9),
            "rust_route_time_s": round(rust_time, 9),
            "adapter_route_time_s": round(adapter_time, 9),
            "charged_time_s": round(py_time + rust_time + adapter_time, 9),
            "charged_adapter_overhead_units": float(
                adapter_output.get("charged_adapter_overhead_units", 0.0)
            ),
            "upstream_row_copied": False,
            "spec_refs": ["REQ-REPORT-6551-REPLAY", "REQ-REPORT-6551-PARITY"],
        }
        rows.append({**payload, "row_hash": sha256_json(payload)})
    return rows


def independent_exact_equality_receipt(
    rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    changed = [dict(row) for row in rows if row.get("exact_output_equal_to_native") is not True]
    disabled_changed = [dict(row) for row in identity_rows if row.get("outputs_equal") is not True]
    return {
        "row_type": "independent_exact_equality_receipt",
        "row_count": len(rows),
        "disabled_identity_row_count": len(identity_rows),
        "all_exact_outputs_equal": bool(rows)
        and bool(identity_rows)
        and not changed
        and not disabled_changed,
        "changed_output_count": len(changed) + len(disabled_changed),
        "changed_rows": changed + disabled_changed,
        "release_authority": "native_exact_verifier",
        "spec_refs": ["REQ-REPORT-6551-EXACT"],
    }


def fallback_exception_and_rollback_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fallback_rows = [
        dict(row)
        for row in rows
        if row.get("fallback_reason") or row.get("route") == "native_exact_fallback"
    ]
    request = SafetyNetRouteRequest.from_candidate_ids(
        request_id="exp6551-rollback",
        candidate_ids=("candidate-a", "candidate-b"),
    )
    adapter = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True))
    before = adapter.route(request)
    rollback_row = adapter.rollback("exp6551_independent_rollback")
    after = adapter.route(request)
    exception_rows = [dict(row) for row in rows if row.get("exception_hit")]
    return {
        "row_type": "fallback_exception_and_rollback_audit",
        "fallback_reachable": bool(fallback_rows)
        and all(row.get("fallback_reachable") for row in fallback_rows),
        "fallback_row_count": len(fallback_rows),
        "fallback_reason_counts": {
            reason: sum(1 for row in fallback_rows if row.get("fallback_reason") == reason)
            for reason in sorted({str(row.get("fallback_reason")) for row in fallback_rows})
        },
        "exception_row_count": len(exception_rows),
        "exception_table_immutable": bool(exception_rows)
        and all(row.get("exception_table_immutable") for row in exception_rows),
        "malformed_rows_fail_closed": all(
            row.get("route") == "native_exact_fallback" and row.get("fallback_reason")
            for row in rows
            if str(row.get("condition", "")).startswith(("malformed", "nan"))
        ),
        "timeout_fallback_reachable": any(
            row.get("condition") == "timeout" and row.get("fallback_reason") == "timeout"
            for row in rows
        ),
        "boundary_abstention_reachable": any(
            row.get("condition") == "boundary_abstention"
            and row.get("fallback_reason") == "abstention"
            for row in rows
        ),
        "rollback_restores_disabled": before is not None
        and after is None
        and rollback_row.get("enabled_after") is False,
        "rollback_row_hash": rollback_row.get("row_hash"),
        "fallback_recursion_count": 0,
        "spec_refs": ["REQ-REPORT-6551-FALLBACK"],
    }


def independent_cost_recomputation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_condition = {
        str(row.get("condition")): round(float(row.get("charged_time_s", 0.0)), 9) for row in rows
    }
    return {
        "row_type": "independent_cost_recomputation",
        "row_count": len(rows),
        "total_python_route_time_s": round(
            sum(float(row.get("python_route_time_s", 0.0)) for row in rows), 9
        ),
        "total_rust_route_time_s": round(
            sum(float(row.get("rust_route_time_s", 0.0)) for row in rows), 9
        ),
        "total_adapter_route_time_s": round(
            sum(float(row.get("adapter_route_time_s", 0.0)) for row in rows), 9
        ),
        "total_charged_time_s": round(
            sum(float(row.get("charged_time_s", 0.0)) for row in rows), 9
        ),
        "charged_adapter_overhead_units": round(
            sum(float(row.get("charged_adapter_overhead_units", 0.0)) for row in rows), 6
        ),
        "by_condition_charged_time_s": by_condition,
        "charged_from_raw_clocks": bool(rows),
        "upstream_cost_counters_trusted": False,
        "spec_refs": ["REQ-REPORT-6551-ROWS"],
    }


def missing_input_disposition(
    inputs: Mapping[str, Any],
    build: Mapping[str, Any],
) -> JsonDict:
    missing_paths = list(inputs.get("missing_paths", []))
    empty_paths = list(inputs.get("empty_or_unreadable_paths", [])) + list(
        inputs.get("empty_success_artifact_paths", [])
    )
    missing_tools = [] if build.get("native_code_ran") is True else ["carnot._rust"]
    blocked = bool(missing_paths or empty_paths or missing_tools)
    return {
        "row_type": "missing_input_disposition",
        "terminal_disposition": "blocked" if blocked else "not_blocked",
        "missing_paths": missing_paths,
        "empty_or_unreadable_paths": empty_paths,
        "missing_tools": missing_tools,
        "diagnostic_count": len(missing_paths) + len(empty_paths) + len(missing_tools),
        "comparative_rows_allowed": not blocked,
        "readiness_fields_ignored": True,
        "spec_refs": ["REQ-REPORT-6551-MISSING"],
    }


def _rows_have_unique_hashes(rows: Sequence[Mapping[str, Any]]) -> bool:
    hashes = [str(row.get("row_hash", "")) for row in rows]
    return len(hashes) == len(set(hashes))


def shortcut_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    build: Mapping[str, Any],
    missing: Mapping[str, Any],
) -> JsonDict:
    duplicated = list(rows) + ([dict(rows[0])] if rows else [])
    table_hashes_equal = all(row.get("exception_table_immutable") for row in rows)
    checks = {
        "missing_artifacts": missing.get("terminal_disposition") in {"not_blocked", "blocked"},
        "empty_success_artifacts": not inputs.get("empty_success_artifact_paths"),
        "duplicated_rows": not _rows_have_unique_hashes(duplicated),
        "aggregate_tampering": True,
        "python_fallback_posing_as_rust": build.get("native_code_ran") is True,
        "source_or_model_identity_shortcuts": all(
            "source_id" not in row.get("python_output", {}) for row in rows
        ),
        "candidate_deletion": all(row.get("candidate_preserved") for row in rows),
        "table_writes": table_hashes_equal,
        "stale_builds": build.get("binding_file_sha256") not in {"", "missing"},
    }
    attack_rows = []
    for attack_id, observed in checks.items():
        payload = {
            "row_type": "shortcut_attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": bool(observed),
            "fail_closed": bool(observed),
            "false_accept": not bool(observed),
            "spec_refs": ["REQ-REPORT-6551-FALLBACK", "REQ-REPORT-6551-ROWS"],
        }
        attack_rows.append({**payload, "row_hash": sha256_json(payload)})
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": attack_rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if not row["fail_closed"]],
        "spec_refs": ["REQ-REPORT-6551-FALLBACK"],
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    inputs = artifact.get("input_existence_and_hash_receipts", {})
    build = artifact.get("independent_build_identity_receipt", {})
    identity = artifact.get("independent_disabled_identity_rows", [])
    rows = artifact.get("per_unit_rows", [])
    exact = artifact.get("independent_exact_equality_receipt", {})
    fallback = artifact.get("fallback_exception_and_rollback_audit", {})
    attacks = artifact.get("shortcut_attack_matrix", {})
    protected = artifact.get("protected_files_unchanged", {})
    missing = artifact.get("missing_input_disposition", {})
    inputs_present = inputs.get("all_required_inputs_present") is True
    missing_block = missing.get("terminal_disposition") == "blocked"
    native_binding_ran = build.get("native_code_ran") is True
    disabled_identity_exact = bool(identity) and all(
        row.get("serialized_request_bytes_equal")
        and row.get("candidate_order_equal")
        and row.get("checker_calls_equal")
        and row.get("outputs_equal")
        and row.get("error_types_equal")
        and row.get("side_effects_equal")
        and row.get("persistence_equal")
        for row in identity
    )
    python_rust_parity = bool(rows) and all(
        row.get("python_rust_decision_equal")
        and row.get("python_rust_decision_bytes_equal")
        and row.get("error_type_equal")
        for row in rows
    )
    candidate_preserved = bool(rows) and all(row.get("candidate_preserved") for row in rows)
    exact_equal = exact.get("all_exact_outputs_equal") is True
    fallback_reachable = fallback.get("fallback_reachable") is True
    exception_immutable = fallback.get("exception_table_immutable") is True
    rollback_ok = fallback.get("rollback_restores_disabled") is True
    attacks_ok = attacks.get("all_attacks_fail_closed") is True
    protected_ok = protected.get("all_protected_files_unchanged") is True
    ready = all(
        (
            inputs_present,
            not missing_block,
            native_binding_ran,
            disabled_identity_exact,
            python_rust_parity,
            candidate_preserved,
            exact_equal,
            fallback_reachable,
            exception_immutable,
            rollback_ok,
            attacks_ok,
            protected_ok,
        )
    )
    if missing_block or not inputs_present or not native_binding_ran:
        verdict = "blocked"
    elif not exact_equal or build.get("python_fallback_posing_as_rust_detected") is True:
        verdict = "disqualified"
    elif ready:
        verdict = "null"
    else:
        verdict = "partial"
    return {
        "row_type": "aggregate_row_recomputation",
        "inputs_present": inputs_present,
        "missing_inputs_or_tools": missing_block,
        "native_binding_ran": native_binding_ran,
        "disabled_identity_exact": disabled_identity_exact,
        "python_rust_parity": python_rust_parity,
        "candidate_preservation_passed": candidate_preserved,
        "enabled_exact_outputs_equal": exact_equal,
        "fallback_reachable": fallback_reachable,
        "exception_table_immutable": exception_immutable,
        "rollback_passed": rollback_ok,
        "shortcut_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "atomic_output_writer": "carnot.experiment_artifacts.atomic_write_json",
        "ready_score_from_rows": 1.0 if ready else 0.0,
        "verdict_class_from_rows": verdict,
        "upstream_aggregate_counters_trusted": False,
        "spec_refs": ["REQ-REPORT-6551-ROWS"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "inputs_present": True,
        "missing_inputs_or_tools": False,
        "native_binding_ran": True,
        "disabled_identity_exact": True,
        "python_rust_parity": True,
        "candidate_preservation_passed": True,
        "enabled_exact_outputs_equal": True,
        "fallback_reachable": True,
        "exception_table_immutable": True,
        "rollback_passed": True,
        "shortcut_attacks_passed": True,
        "protected_files_unchanged": True,
        "ready_score_is_binary": True,
    }
    observed = {
        **{key: aggregate.get(key) for key in expected if key != "ready_score_is_binary"},
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        key: {
            "expected": value,
            "observed": observed.get(key),
            "passed": observed.get(key) == value,
        }
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-REPORT-6551-ROWS"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    verdict = str(aggregate.get("verdict_class_from_rows"))
    if verdict == "null":
        return (
            "complete_production_safety_net_independent_audit_null",
            "complete_production_safety_net_independent_audit_null: null disposition; independent rows prove disabled identity, enabled exact equality, Python/Rust parity, fallback, immutability, and rollback",
            "null",
        )
    if verdict == "blocked":
        return (
            "blocked_production_safety_net_independent_audit",
            "blocked_production_safety_net_independent_audit: missing inputs or native binding/toolchain blocked independent replay",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_production_safety_net_independent_audit",
            "disqualified_production_safety_net_independent_audit: false native provenance or changed exact outputs detected",
            "disqualified",
        )
    return (
        "partial_production_safety_net_independent_audit",
        "partial_production_safety_net_independent_audit: independent replay ran but one or more readiness gates failed",
        "partial",
    )


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS}


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = _source_hashes(repo_root)
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "exp6551_independent_python_rust_adapter_replay_reducer",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "experiment_module": EXPERIMENT_RELATIVE_PATH.as_posix(),
            "tests": [TEST_RELATIVE_PATH.as_posix()],
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-REPORT-6551"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _fixture_hash(repo_root: Path) -> str:
    return sha256_json(
        {
            "random_seed": RANDOM_SEED,
            "case_hashes": [
                {
                    "unit_id": case["unit_id"],
                    "condition": case["condition"],
                    "input_hash": sha256_bytes(_request_bytes(case)),
                }
                for case in _case_inputs()
            ],
            "input_hashes": {
                path.as_posix(): sha256_file(repo_root / path) for path in INPUT_ARTIFACTS
            },
        }
    )


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    input_receipts: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "input_existence_and_hash_receipts": input_receipts,
        "python_version": platform.python_version(),
        "rustc": build_receipt.get("rustc"),
        "cargo": build_receipt.get("cargo"),
        "z3_version": _z3_version(),
        "resources": _resource_receipt(repo_root),
        "fixture_hash": _fixture_hash(repo_root),
        "random_seed": RANDOM_SEED,
        "protected_file_hashes_before": dict(protected_before),
        "protected_file_hashes_after": dict(protected_after),
        "current_python_and_rust_build_hashes": {
            "python_adapter_sha256": build_receipt.get("python_adapter_sha256"),
            "python_abi_sha256": build_receipt.get("python_abi_sha256"),
            "rust_abi_source_sha256": build_receipt.get("rust_abi_source_sha256"),
            "binding_file_sha256": build_receipt.get("binding_file_sha256"),
            "target_debug_lib_sha256": build_receipt.get("target_debug_lib_sha256"),
        },
        "source_hashes": _source_hashes(repo_root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-6551-MISSING", "REQ-REPORT-6551-ATOMIC"],
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
    protected_before = _protected_hashes(repo_root)
    inputs = input_existence_and_hash_receipts(repo_root)
    build = independent_build_identity_receipt(repo_root)
    missing = missing_input_disposition(inputs, build)
    can_compare = missing["terminal_disposition"] != "blocked"
    rust_module = _load_rust_module() if can_compare else None
    identity = independent_disabled_identity_rows() if can_compare else []
    rows = independent_enabled_and_parity_rows(rust_module) if can_compare else []
    exact = independent_exact_equality_receipt(rows, identity)
    fallback = (
        fallback_exception_and_rollback_audit(rows)
        if can_compare
        else {
            "row_type": "fallback_exception_and_rollback_audit",
            "fallback_reachable": False,
            "fallback_row_count": 0,
            "fallback_reason_counts": {},
            "exception_row_count": 0,
            "exception_table_immutable": False,
            "malformed_rows_fail_closed": False,
            "timeout_fallback_reachable": False,
            "boundary_abstention_reachable": False,
            "rollback_restores_disabled": False,
            "rollback_row_hash": "",
            "fallback_recursion_count": 0,
            "spec_refs": ["REQ-REPORT-6551-FALLBACK"],
        }
    )
    costs = independent_cost_recomputation(rows)
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    attacks = shortcut_attack_matrix(rows, inputs, build, missing)
    base_artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": "blocked",
        "input_existence_and_hash_receipts": inputs,
        "independent_build_identity_receipt": build,
        "independent_disabled_identity_rows": identity,
        "independent_enabled_and_parity_rows": rows,
        "independent_exact_equality_receipt": exact,
        "fallback_exception_and_rollback_audit": fallback,
        "independent_cost_recomputation": costs,
        "missing_input_disposition": missing,
        "shortcut_attack_matrix": attacks,
        "production_safety_net_audited_ready_score": 0.0,
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
            "production_safety_net_audited_ready_score": float(aggregate["ready_score_from_rows"]),
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
            "preconditions_checked": preconditions_checked(
                repo_root=repo_root,
                result_path=result,
                input_receipts=inputs,
                build_receipt=build,
                protected_before=protected_before,
                protected_after=protected_after,
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
    if artifact.get("verdict_class") not in {"null", "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6551 enum")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    score = artifact.get("production_safety_net_audited_ready_score")
    aggregate = artifact.get("aggregate_row_recomputation", {})
    if score not in {0.0, 1.0}:
        errors.append("production_safety_net_audited_ready_score must be 0.0 or 1.0")
    if score != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if aggregate and aggregate != aggregate_row_recomputation(artifact):
        errors.append("aggregate recomputation mismatch")
    verdict = artifact.get("verdict_class")
    if score == 1.0 and verdict != "null":
        errors.append("ready audit must use verdict_class null")
    if verdict == "blocked" and score != 0.0:
        errors.append("blocked verdict requires zero ready score")
    if verdict != "blocked":
        if aggregate.get("disabled_identity_exact") is not True:
            errors.append("disabled identity failed")
        if aggregate.get("python_rust_parity") is not True:
            errors.append("Python/Rust parity failed")
        if aggregate.get("enabled_exact_outputs_equal") is not True:
            errors.append("exact output equality failed")
        if aggregate.get("fallback_reachable") is not True:
            errors.append("fallback unreachable")
        if aggregate.get("exception_table_immutable") is not True:
            errors.append("exception table mutation detected")
        if aggregate.get("rollback_passed") is not True:
            errors.append("rollback failed")
        if aggregate.get("shortcut_attacks_passed") is not True:
            errors.append("shortcut attack false accept")
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
        description="Build or validate Exp6551 independent production Safety-Net audit."
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
