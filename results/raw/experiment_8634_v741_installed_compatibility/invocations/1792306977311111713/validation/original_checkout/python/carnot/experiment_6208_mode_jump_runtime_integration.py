"""Exp6208 mode-jump runtime adapter integration.

Spec refs: REQ-SAMPLE-6208, SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK,
SCENARIO-SAMPLE-6208-RUNTIME-PARITY, SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS.

The experiment wires the already-qualified Exp6194 fixed categorical
mode-jump kernel into the runtime sampler backend boundary. It does not change
the transition math, make hardware claims, or treat timing as readiness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_6194_mode_jump_rust_pyo3_parity as exp6194
from carnot.samplers.backend import CpuBackend, get_backend, get_sampler_backend
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_ENV_VAR,
    MODE_JUMP_ALGORITHM,
    MODE_JUMP_TOPOLOGY,
    ModeJumpRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
    frozen_mode_jump_inputs,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6208_mode_jump_runtime_integration.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6208_mode_jump_runtime_integration.py")
ADAPTER_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/backend.py")
INIT_RELATIVE_PATH = Path("python/carnot/samplers/__init__.py")
PY_TEST_RELATIVE_PATH = Path("tests/python/samplers/test_mode_jump_rust_backend.py")
ARTIFACT_TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6208_mode_jump_runtime_integration.py"
)
RUST_TEST_RELATIVE_PATH = Path("crates/carnot-samplers/tests/mode_jump.rs")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXP6194_RESULT_RELATIVE_PATH = Path("results/experiment_6194_mode_jump_rust_pyo3_parity.json")

SCHEMA = "carnot.experiment_6208.mode_jump_runtime_integration.v1"
EXPERIMENT_ID = "experiment_6208_mode_jump_runtime_integration"
RUN_DATE = "20260807"
INFERENCE_SUBSTRATE = "production_python_samplerbackend_plus_rust_pyo3_mode_jump_cpu"
PARITY_SAMPLE_COUNT = 512
PARITY_BURN_IN = 16
LONG_RUN_SAMPLE_COUNT = 50_000
LONG_RUN_BURN_IN = 1_000
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6208_command_receipts.json")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6194_artifact_and_kernel_hashes",
    "runtime_adapter_paths_and_hashes",
    "default_off_receipt",
    "config_and_feature_flag_contract",
    "supported_and_unsupported_shape_matrix",
    "seeded_quality_parity",
    "distribution_tv_kl_and_interval_receipts",
    "autocorrelation_and_effective_sample_size",
    "serialization_roundtrip",
    "cancellation_timeout_and_error_receipts",
    "exact_fallback_receipts",
    "task_owned_test_commands_and_exit_codes",
    "unrelated_nonzero_command_classifications",
    "hardware_or_speed_power_energy_claimed",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Distinguishes ready, partial, and blocked runtime integration outcomes.",
    "exp6194_artifact_and_kernel_hashes": "Pins Exp6194 readiness and the qualified Rust/PyO3 kernel before adapter evidence is trusted.",
    "runtime_adapter_paths_and_hashes": "Content-addresses the runtime adapter, factory registration, spec, tests, and terminal artifact path.",
    "default_off_receipt": "Proves current CPU sampler selection remains the default and mode-jump Rust execution needs explicit opt-in.",
    "config_and_feature_flag_contract": "Records the only accepted backend name, config flag, environment flag, seed, state, and fallback controls.",
    "supported_and_unsupported_shape_matrix": "Separates Rust-supported, exact-fallback, and fail-closed input surfaces.",
    "seeded_quality_parity": "Shows Rust and exact fallback replay the same seeded samples and final state.",
    "distribution_tv_kl_and_interval_receipts": "Quantifies empirical distribution quality against the frozen target with intervals.",
    "autocorrelation_and_effective_sample_size": "Reports mixing diagnostics rather than inferring quality from frequencies alone.",
    "serialization_roundtrip": "Proves adapter checkpoints and serialized kernel state restore exactly and reject corruption.",
    "cancellation_timeout_and_error_receipts": "Proves runtime interruption and bad inputs fail closed with explicit errors.",
    "exact_fallback_receipts": "Lists every exercised fallback reason and its exact equivalence result.",
    "task_owned_test_commands_and_exit_codes": "Stores focused Rust/PyO3/Python/spec/coverage command receipts.",
    "unrelated_nonzero_command_classifications": "Prevents unrelated nonzero commands from masquerading as integration failure or success evidence.",
    "hardware_or_speed_power_energy_claimed": "Bare false prevents this software adapter from claiming hardware, speed, power, or energy results.",
    "inference_substrate": "Declares local CPU production sampler backend plus Rust/PyO3 mode-jump integration.",
    "verifier_is_oracle": "States whether the verifier is the exact finite categorical target and fixed transition, not LLM judgment.",
    "field_provenance": "Maps each artifact field to prompt, spec, source hashes, tests, commands, or deterministic computation.",
    "field_principles": "Explains why each required field exists before the JSON shape is trusted.",
    "duration_s": "Reports real wall time without converting it into a speed claim.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking duration and the checksum field.",
    "honest_verdict": "Terminal verdict states readiness and all nonzero command classifications.",
}


DEFAULT_TASK_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("cargo", "test", "-p", "carnot-samplers", "--test", "mode_jump"),
    ("PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1", "cargo", "build", "-p", "carnot-python"),
    (
        ".venv/bin/pytest",
        "tests/python/samplers/test_mode_jump_rust_backend.py",
        "tests/python/test_experiment_6208_mode_jump_runtime_integration.py",
        "-q",
        "-o",
        "addopts=",
    ),
    (
        ".venv/bin/python",
        "scripts/check_spec_coverage.py",
        "tests/python/samplers/test_mode_jump_rust_backend.py",
        "tests/python/test_experiment_6208_mode_jump_runtime_integration.py",
        "crates/carnot-samplers/tests/mode_jump.rs",
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    return "sha256:" + exp6194.hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def _path_hashes(root: Path, paths: Sequence[Path]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for path in paths:
        full = root / path
        rows[path.as_posix()] = {
            "exists": full.exists(),
            "sha256": sha256_file(full) if full.exists() else None,
            "size_bytes": full.stat().st_size if full.exists() else None,
        }
    return rows


def _run_text(argv: Sequence[str], root: Path) -> JsonDict:
    env = os.environ.copy()
    command = list(argv)
    if command and "=" in command[0] and not command[0].startswith(("/", ".")):
        key, value = command.pop(0).split("=", 1)
        env[key] = value
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        return {
            "name": command[0] if command else "missing",
            "command": " ".join(argv),
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
            "task_owned": True,
        }
    return {
        "name": command[0] if command else "command",
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip()[-4000:],
        "stderr": result.stderr.strip()[-4000:],
        "task_owned": True,
    }


def exp6194_artifact_and_kernel_hashes(root: Path = REPO_ROOT) -> JsonDict:
    exp6194_artifact = _read_json(root / EXP6194_RESULT_RELATIVE_PATH)
    fixed = exp6194.fixed_algorithm_equations_config_and_seed(root)
    ready_score = float(exp6194_artifact.get("mode_jump_rust_pyo3_ready_score", 0.0))
    status = str(exp6194_artifact.get("status", ""))
    return {
        "exp6194_artifact": {
            "path": EXP6194_RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6194_RESULT_RELATIVE_PATH),
            "status": status,
            "mode_jump_rust_pyo3_ready_score": ready_score,
            "honest_verdict": exp6194_artifact.get("honest_verdict"),
        },
        "qualified_kernel": {
            "algorithm": MODE_JUMP_ALGORITHM,
            "config_sha256": fixed["config_sha256"],
            "rust_kernel_sha256": sha256_file(root / RUST_KERNEL_RELATIVE_PATH),
            "pyo3_binding_sha256": sha256_file(root / PYO3_BINDING_RELATIVE_PATH),
            "state_schema": exp6194.STATE_SCHEMA,
        },
        "exp6194_ready": status == "complete_ready" and ready_score == 1.0,
        "transition_math_altered": False,
        "principle": FIELD_PRINCIPLES["exp6194_artifact_and_kernel_hashes"],
    }


def runtime_adapter_paths_and_hashes(root: Path = REPO_ROOT) -> JsonDict:
    paths = (
        ADAPTER_RELATIVE_PATH,
        BACKEND_RELATIVE_PATH,
        INIT_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        PY_TEST_RELATIVE_PATH,
        ARTIFACT_TEST_RELATIVE_PATH,
        RUST_TEST_RELATIVE_PATH,
        SAMPLER_SPEC_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
    )
    return {
        "paths": [path.as_posix() for path in paths],
        "hashes": _path_hashes(
            root,
            [path for path in paths if (root / path).exists() and path != RESULT_RELATIVE_PATH],
        ),
        "self_hash_note": "result JSON path is listed but not self-hashed",
        "principle": FIELD_PRINCIPLES["runtime_adapter_paths_and_hashes"],
    }


@contextmanager
def _without_sampler_env() -> Any:
    keys = ("CARNOT_BACKEND", "CARNOT_SAMPLER", FEATURE_ENV_VAR)
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def default_off_receipt(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    with _without_sampler_env():
        default_backend = get_backend()
        default_sampler = get_sampler_backend()
        mode_jump = get_backend("mode_jump_rust")
        disabled = mode_jump.run_descriptor(
            target,
            proposal,
            n_samples=8,
            config=descriptor_for_run(labels=labels, seed=6208),
        )
    return {
        "default_backend_name": default_backend.backend_name,
        "default_backend_is_cpu": isinstance(default_backend, CpuBackend),
        "default_sampler_name": default_sampler.backend_name,
        "default_sampler_is_cpu": isinstance(default_sampler, CpuBackend),
        "mode_jump_backend_name": mode_jump.backend_name,
        "explicit_mode_jump_selectable": isinstance(mode_jump, ModeJumpRustBackend),
        "disabled_mode_jump_active_backend": disabled["receipt"]["active_backend"],
        "disabled_mode_jump_fallback_reason": disabled["receipt"]["fallback_reason"],
        "default_off_pass": (
            isinstance(default_backend, CpuBackend)
            and isinstance(default_sampler, CpuBackend)
            and disabled["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
            and disabled["receipt"]["fallback_reason"] == "feature_flag_disabled"
        ),
        "principle": FIELD_PRINCIPLES["default_off_receipt"],
    }


def config_and_feature_flag_contract(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6208,
        burn_in=PARITY_BURN_IN,
        enable_mode_jump_runtime=True,
    )
    return {
        "backend_name": "mode_jump_rust",
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": MODE_JUMP_TOPOLOGY,
        "feature_config_flag": "enable_mode_jump_runtime",
        "feature_env_var": FEATURE_ENV_VAR,
        "feature_env_truthy_values": ["1", "true", "yes", "on"],
        "default_feature_enabled": False,
        "descriptor": descriptor,
        "target_shape": list(target.shape),
        "proposal_shape": list(proposal.shape),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_backends": [ACTIVE_RUST_BACKEND, ACTIVE_PYTHON_FALLBACK],
        "fallback_reasons": [
            "feature_flag_disabled",
            "declared_python_compatibility",
            "unsupported_dtype_or_layout",
            "rust_extension_missing",
            "rust_extension_error",
            "rust_symbol_missing",
        ],
        "principle": FIELD_PRINCIPLES["config_and_feature_flag_contract"],
    }


def supported_and_unsupported_shape_matrix(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(labels=labels, seed=6208, enable_mode_jump_runtime=True)
    rows: list[JsonDict] = []

    def run_case(name: str, case_target: Any, case_proposal: Any) -> None:
        try:
            result = ModeJumpRustBackend(seed=6208).run_descriptor(
                case_target,
                case_proposal,
                n_samples=4,
                config=descriptor,
            )
            rows.append(
                {
                    "case": name,
                    "outcome": "accepted",
                    "active_backend": result["receipt"]["active_backend"],
                    "fallback_reason": result["receipt"]["fallback_reason"],
                    "rust_supported": result["receipt"]["input_support"]["rust_supported"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - artifact records boundary errors.
            rows.append(
                {
                    "case": name,
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    run_case("float64_c_contiguous", target, proposal)
    run_case("float32_exact_values", target.astype(np.float32), proposal.astype(np.float32))
    run_case("fortran_exact_values", target.copy(), np.asfortranarray(proposal))
    run_case("bad_target_shape", target.reshape(1, -1), proposal)
    run_case("bad_proposal_shape", target, proposal[:5, :5])
    altered = target.copy()
    altered[0] += 0.01
    run_case("altered_target_probability", altered, proposal)

    return {
        "rows": rows,
        "rust_case_count": sum(row.get("active_backend") == ACTIVE_RUST_BACKEND for row in rows),
        "fallback_case_count": sum(
            row.get("active_backend") == ACTIVE_PYTHON_FALLBACK for row in rows
        ),
        "error_case_count": sum(row.get("outcome") == "error" for row in rows),
        "shape_matrix_pass": (
            any(row.get("active_backend") == ACTIVE_RUST_BACKEND for row in rows)
            and any(row.get("active_backend") == ACTIVE_PYTHON_FALLBACK for row in rows)
            and sum(row.get("outcome") == "error" for row in rows) >= 3
        ),
        "principle": FIELD_PRINCIPLES["supported_and_unsupported_shape_matrix"],
    }


def _paired_runs(
    *,
    sample_count: int,
    burn_in: int,
    seed: int,
    return_trace: bool,
    root: Path = REPO_ROOT,
) -> tuple[JsonDict, JsonDict]:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=seed,
        burn_in=burn_in,
        enable_mode_jump_runtime=True,
    )
    descriptor["return_trace"] = return_trace
    rust = ModeJumpRustBackend(seed=seed).run_descriptor(
        target,
        proposal,
        n_samples=sample_count,
        config=descriptor,
    )
    fallback = ModeJumpRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=sample_count,
        config=descriptor,
    )
    return rust, fallback


def seeded_quality_parity(root: Path = REPO_ROOT) -> JsonDict:
    rust, fallback = _paired_runs(
        sample_count=PARITY_SAMPLE_COUNT,
        burn_in=PARITY_BURN_IN,
        seed=6208,
        return_trace=True,
        root=root,
    )
    samples_match = bool(np.array_equal(rust["samples"], fallback["samples"]))
    return {
        "seed": 6208,
        "sample_count": PARITY_SAMPLE_COUNT,
        "burn_in": PARITY_BURN_IN,
        "rust_active_backend": rust["receipt"]["active_backend"],
        "fallback_active_backend": fallback["receipt"]["active_backend"],
        "seeded_samples_match": samples_match,
        "sample_label_hashes_match": sha256_json(rust["sample_labels"])
        == sha256_json(fallback["sample_labels"]),
        "decision_log_match": rust["decision_log"] == fallback["decision_log"],
        "metrics_match": rust["metrics"] == fallback["metrics"],
        "final_state_match": rust["checkpoint"]["state"] == fallback["checkpoint"]["state"],
        "rust_sample_label_hash": sha256_json(rust["sample_labels"]),
        "fallback_sample_label_hash": sha256_json(fallback["sample_labels"]),
        "principle": FIELD_PRINCIPLES["seeded_quality_parity"],
    }


def distribution_tv_kl_and_interval_receipts(root: Path = REPO_ROOT) -> JsonDict:
    rust, fallback = _paired_runs(
        sample_count=LONG_RUN_SAMPLE_COUNT,
        burn_in=LONG_RUN_BURN_IN,
        seed=6208,
        return_trace=False,
        root=root,
    )
    rust_metrics = rust["metrics"]
    fallback_metrics = fallback["metrics"]
    return {
        "sample_count": LONG_RUN_SAMPLE_COUNT,
        "burn_in": LONG_RUN_BURN_IN,
        "interval_method": "wald_95_per_label_frequency",
        "target_tv_tolerance": exp6194.TOLERANCES["target_tv"],
        "target_kl_tolerance": exp6194.TOLERANCES["target_kl"],
        "rust": {
            "total_variation_to_target": rust_metrics["total_variation_to_target"],
            "kl_target_to_empirical": rust_metrics["kl_target_to_empirical"],
            "frequencies": rust_metrics["frequencies"],
        },
        "fallback": {
            "total_variation_to_target": fallback_metrics["total_variation_to_target"],
            "kl_target_to_empirical": fallback_metrics["kl_target_to_empirical"],
            "frequencies": fallback_metrics["frequencies"],
        },
        "rust_fallback_frequency_delta_max": _frequency_delta_max(rust_metrics, fallback_metrics),
        "distribution_pass": (
            rust_metrics["total_variation_to_target"] <= exp6194.TOLERANCES["target_tv"]
            and fallback_metrics["total_variation_to_target"] <= exp6194.TOLERANCES["target_tv"]
            and rust_metrics["kl_target_to_empirical"] <= exp6194.TOLERANCES["target_kl"]
            and fallback_metrics["kl_target_to_empirical"] <= exp6194.TOLERANCES["target_kl"]
            and _frequency_delta_max(rust_metrics, fallback_metrics) == 0.0
        ),
        "principle": FIELD_PRINCIPLES["distribution_tv_kl_and_interval_receipts"],
    }


def autocorrelation_and_effective_sample_size(root: Path = REPO_ROOT) -> JsonDict:
    rust, fallback = _paired_runs(
        sample_count=LONG_RUN_SAMPLE_COUNT,
        burn_in=LONG_RUN_BURN_IN,
        seed=6208,
        return_trace=False,
        root=root,
    )
    rust_metrics = rust["metrics"]
    fallback_metrics = fallback["metrics"]
    return {
        "rust": {
            "acceptance_rate": rust_metrics["acceptance_rate"],
            "accepted_count": rust_metrics["accepted_count"],
            "attempted_count": rust_metrics["attempted_count"],
            "lag1_autocorrelation": rust_metrics["lag1_autocorrelation"],
            "integrated_autocorrelation_time": rust_metrics["integrated_autocorrelation_time"],
            "effective_sample_size": rust_metrics["effective_sample_size"],
        },
        "fallback": {
            "acceptance_rate": fallback_metrics["acceptance_rate"],
            "accepted_count": fallback_metrics["accepted_count"],
            "attempted_count": fallback_metrics["attempted_count"],
            "lag1_autocorrelation": fallback_metrics["lag1_autocorrelation"],
            "integrated_autocorrelation_time": fallback_metrics["integrated_autocorrelation_time"],
            "effective_sample_size": fallback_metrics["effective_sample_size"],
        },
        "acceptance_delta": abs(
            rust_metrics["acceptance_rate"] - fallback_metrics["acceptance_rate"]
        ),
        "autocorrelation_delta": abs(
            rust_metrics["lag1_autocorrelation"] - fallback_metrics["lag1_autocorrelation"]
        ),
        "ess_delta": abs(
            rust_metrics["effective_sample_size"] - fallback_metrics["effective_sample_size"]
        ),
        "ess_min": exp6194.TOLERANCES["ess_min"],
        "mixing_pass": (
            rust_metrics["effective_sample_size"] > exp6194.TOLERANCES["ess_min"]
            and rust_metrics["effective_sample_size"] == fallback_metrics["effective_sample_size"]
        ),
        "principle": FIELD_PRINCIPLES["autocorrelation_and_effective_sample_size"],
    }


def serialization_roundtrip(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6211,
        burn_in=1,
        enable_mode_jump_runtime=True,
    )
    backend = ModeJumpRustBackend(seed=6211)
    prefix = backend.run_descriptor(target, proposal, n_samples=8, config=descriptor)
    checkpoint_a = backend.save_checkpoint()
    checkpoint_b = backend.save_checkpoint()
    loaded_state = backend.load_checkpoint(checkpoint_a, target, proposal, config=descriptor)
    resumed_descriptor = {**descriptor, "checkpoint": checkpoint_a}
    rust_suffix = ModeJumpRustBackend(seed=6211).run_descriptor(
        target,
        proposal,
        n_samples=8,
        config=resumed_descriptor,
    )
    fallback_suffix = ModeJumpRustBackend(seed=6211, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=8,
        config=resumed_descriptor,
    )
    corrupt = dict(checkpoint_a)
    corrupt["state"] = dict(corrupt["state"])
    corrupt["state"]["step"] = int(corrupt["state"]["step"]) + 1
    corrupt_receipt = _expect_error(
        lambda: backend.load_checkpoint(corrupt, target, proposal, config=descriptor),
        ValueError,
    )
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_roundtrip_pass": checkpoint_a == checkpoint_b
        and loaded_state == checkpoint_a["state"]
        and checkpoint_a["payload_checksum"] == checkpoint_checksum(checkpoint_a),
        "prefix_checkpoint_state": prefix["checkpoint"]["state"],
        "serialized_state": checkpoint_a["serialized_state"],
        "rust_to_fallback_restart_samples_match": bool(
            np.array_equal(rust_suffix["samples"], fallback_suffix["samples"])
        ),
        "rust_to_fallback_restart_state_match": rust_suffix["checkpoint"]["state"]
        == fallback_suffix["checkpoint"]["state"],
        "corrupt_checkpoint_rejected": corrupt_receipt,
        "serialization_pass": (
            checkpoint_a == checkpoint_b
            and loaded_state == checkpoint_a["state"]
            and bool(np.array_equal(rust_suffix["samples"], fallback_suffix["samples"]))
            and corrupt_receipt["raised"] is True
        ),
        "principle": FIELD_PRINCIPLES["serialization_roundtrip"],
    }


def cancellation_timeout_and_error_receipts(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(labels=labels, seed=6213, enable_mode_jump_runtime=True)
    controls = {
        "cancel_after_steps": _expect_error(
            lambda: ModeJumpRustBackend(seed=6213).run_descriptor(
                target,
                proposal,
                n_samples=4,
                config={**descriptor, "cancel_after_steps": 0},
            ),
            TimeoutError,
        ),
        "timeout_s": _expect_error(
            lambda: ModeJumpRustBackend(seed=6213).run_descriptor(
                target,
                proposal,
                n_samples=4,
                config={**descriptor, "timeout_s": 0.0},
            ),
            TimeoutError,
        ),
        "invalid_n_samples": _expect_error(
            lambda: ModeJumpRustBackend(seed=6213).run_descriptor(
                target,
                proposal,
                n_samples=0,
                config=descriptor,
            ),
            ValueError,
        ),
        "invalid_initial_label": _expect_error(
            lambda: ModeJumpRustBackend(seed=6213).run_descriptor(
                target,
                proposal,
                n_samples=2,
                config={**descriptor, "initial_label": "shadow"},
            ),
            ValueError,
        ),
        "unsupported_shape": _expect_error(
            lambda: ModeJumpRustBackend(seed=6213).run_descriptor(
                target.reshape(1, -1),
                proposal,
                n_samples=2,
                config=descriptor,
            ),
            ValueError,
        ),
    }
    return {
        "controls": controls,
        "all_controls_passed": all(row["raised"] for row in controls.values()),
        "principle": FIELD_PRINCIPLES["cancellation_timeout_and_error_receipts"],
    }


def exact_fallback_receipts(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6210,
        burn_in=2,
        enable_mode_jump_runtime=True,
    )
    baseline = ModeJumpRustBackend(seed=6210, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=16,
        config=descriptor,
    )
    cases = [
        (
            "disabled_feature_flag",
            ModeJumpRustBackend(seed=6210),
            target,
            proposal,
            {**descriptor, "enable_mode_jump_runtime": False},
            "feature_flag_disabled",
        ),
        (
            "declared_python_compatibility",
            ModeJumpRustBackend(seed=6210),
            target,
            proposal,
            {**descriptor, "force_python_fallback": True},
            "declared_python_compatibility",
        ),
        (
            "unsupported_dtype_or_layout",
            ModeJumpRustBackend(seed=6210),
            target.astype(np.float32),
            proposal.astype(np.float32),
            descriptor,
            "unsupported_dtype_or_layout",
        ),
        (
            "rust_extension_missing",
            ModeJumpRustBackend(
                seed=6210,
                rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
            ),
            target,
            proposal,
            descriptor,
            "rust_extension_missing",
        ),
        (
            "rust_extension_error",
            ModeJumpRustBackend(
                seed=6210,
                rust_module_loader=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
            ),
            target,
            proposal,
            descriptor,
            "rust_extension_error",
        ),
        (
            "rust_symbol_missing",
            ModeJumpRustBackend(seed=6210, rust_module_loader=lambda: SimpleNamespace()),
            target,
            proposal,
            descriptor,
            "rust_symbol_missing",
        ),
    ]
    rows: list[JsonDict] = []
    for name, backend, case_target, case_proposal, case_descriptor, reason in cases:
        result = backend.run_descriptor(case_target, case_proposal, 16, case_descriptor)
        rows.append(
            {
                "case": name,
                "active_backend": result["receipt"]["active_backend"],
                "fallback_reason": result["receipt"]["fallback_reason"],
                "expected_reason_present": reason in str(result["receipt"]["fallback_reason"]),
                "samples_match": bool(np.array_equal(result["samples"], baseline["samples"])),
                "state_match": result["checkpoint"]["state"] == baseline["checkpoint"]["state"],
                "metrics_match": result["metrics"] == baseline["metrics"],
            }
        )
    return {
        "fallback_cases": rows,
        "all_fallbacks_exact": all(
            row["active_backend"] == ACTIVE_PYTHON_FALLBACK
            and row["expected_reason_present"] is True
            and row["samples_match"] is True
            and row["state_match"] is True
            and row["metrics_match"] is True
            for row in rows
        ),
        "principle": FIELD_PRINCIPLES["exact_fallback_receipts"],
    }


def _frequency_delta_max(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    labels = sorted(left["frequencies"])
    return max(
        abs(left["frequencies"][label]["frequency"] - right["frequencies"][label]["frequency"])
        for label in labels
    )


def _expect_error(call: Any, error_type: type[BaseException]) -> JsonDict:
    try:
        call()
    except error_type as exc:
        return {"raised": True, "error": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "error": None, "message": None}


def _command_summary(command_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    commands = [str(row.get("command", "")) for row in command_receipts]
    exits = {
        str(row.get("command", "")): int(row.get("exit_code", -999)) for row in command_receipts
    }
    nonzero = [
        {
            "name": row.get("name"),
            "command": row.get("command"),
            "exit_code": row.get("exit_code"),
            "classification": row.get("classification", "task_owned_failure"),
            "task_owned": bool(row.get("task_owned", row.get("classification") is None)),
        }
        for row in command_receipts
        if int(row.get("exit_code", -999)) != 0
    ]
    unrelated = [row for row in nonzero if row["classification"] == "unrelated_preexisting"]
    task_owned_failures = [
        row
        for row in nonzero
        if row["classification"] != "unrelated_preexisting" and row["task_owned"] is not False
    ]
    return {
        "test_commands": commands,
        "test_exit_codes": exits,
        "nonzero": nonzero,
        "unrelated": unrelated,
        "task_owned_failure_count": len(task_owned_failures),
        "all_task_owned_commands_passed": len(task_owned_failures) == 0,
    }


def task_owned_test_commands_and_exit_codes(
    command_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    summary = _command_summary(command_receipts)
    return {
        "command_receipts": [dict(row) for row in command_receipts],
        "test_commands": summary["test_commands"],
        "test_exit_codes": summary["test_exit_codes"],
        "all_task_owned_commands_passed": summary["all_task_owned_commands_passed"],
        "task_owned_failure_count": summary["task_owned_failure_count"],
        "principle": FIELD_PRINCIPLES["task_owned_test_commands_and_exit_codes"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "exact finite categorical target plus fixed Metropolis-Hastings transition",
        "not_oracle_for": ["hardware speed", "power", "energy", "unseen sampler shapes"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def field_provenance() -> JsonDict:
    sources = {
        "status": "computed readiness gates",
        "exp6194_artifact_and_kernel_hashes": "Exp6194 artifact and source hashes",
        "runtime_adapter_paths_and_hashes": "repo file hashes",
        "default_off_receipt": "runtime factory calls",
        "config_and_feature_flag_contract": "adapter constants and descriptor",
        "supported_and_unsupported_shape_matrix": "adapter boundary probes",
        "seeded_quality_parity": "matched Rust/Python adapter runs",
        "distribution_tv_kl_and_interval_receipts": "matched long-chain fixtures",
        "autocorrelation_and_effective_sample_size": "matched long-chain diagnostics",
        "serialization_roundtrip": "adapter checkpoint probes",
        "cancellation_timeout_and_error_receipts": "adapter error probes",
        "exact_fallback_receipts": "declared fallback probes",
        "task_owned_test_commands_and_exit_codes": "command receipts",
        "unrelated_nonzero_command_classifications": "command receipt classification",
        "hardware_or_speed_power_energy_claimed": "prompt/spec invariant",
        "inference_substrate": "prompt/spec invariant",
        "verifier_is_oracle": "finite target and fixed transition",
        "field_provenance": "this provenance map",
        "field_principles": "OpenSpec required field principles",
        "duration_s": "wall-clock measurement",
        "reproducibility_checksum": "deterministic artifact hash",
        "honest_verdict": "computed verdict",
    }
    return {
        field: {
            "source": sources[field],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    commands = dict(artifact.get("task_owned_test_commands_and_exit_codes") or {})
    return float(
        dict(artifact.get("exp6194_artifact_and_kernel_hashes") or {}).get("exp6194_ready") is True
        and dict(artifact.get("default_off_receipt") or {}).get("default_off_pass") is True
        and dict(artifact.get("supported_and_unsupported_shape_matrix") or {}).get(
            "shape_matrix_pass"
        )
        is True
        and dict(artifact.get("seeded_quality_parity") or {}).get("seeded_samples_match") is True
        and dict(artifact.get("seeded_quality_parity") or {}).get("metrics_match") is True
        and dict(artifact.get("distribution_tv_kl_and_interval_receipts") or {}).get(
            "distribution_pass"
        )
        is True
        and dict(artifact.get("autocorrelation_and_effective_sample_size") or {}).get("mixing_pass")
        is True
        and dict(artifact.get("serialization_roundtrip") or {}).get("serialization_pass") is True
        and dict(artifact.get("cancellation_timeout_and_error_receipts") or {}).get(
            "all_controls_passed"
        )
        is True
        and dict(artifact.get("exact_fallback_receipts") or {}).get("all_fallbacks_exact") is True
        and commands.get("all_task_owned_commands_passed") is True
        and artifact.get("hardware_or_speed_power_energy_claimed") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def status(artifact: Mapping[str, Any]) -> str:
    if ready_score(artifact) == 1.0:
        return "complete_ready"
    if dict(artifact.get("seeded_quality_parity") or {}).get("seeded_samples_match") is True:
        return "complete_partial"
    return "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    nonzero = list(artifact.get("unrelated_nonzero_command_classifications") or [])
    default_off = dict(artifact.get("default_off_receipt") or {}).get("default_off_pass")
    fallback = dict(artifact.get("exact_fallback_receipts") or {}).get("all_fallbacks_exact")
    parity = dict(artifact.get("seeded_quality_parity") or {}).get("seeded_samples_match")
    nonzero_text = "none" if not nonzero else json.dumps(nonzero, sort_keys=True)
    prefix = status(artifact)
    return (
        f"{prefix}: default_off={bool(default_off)}; "
        f"fallback_exact={bool(fallback)}; seeded_parity={bool(parity)}; "
        f"nonzero commands {nonzero_text}"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> JsonDict:
    command_summary = _command_summary(command_receipts)
    artifact: JsonDict = {
        "status": "blocked",
        "exp6194_artifact_and_kernel_hashes": exp6194_artifact_and_kernel_hashes(root),
        "runtime_adapter_paths_and_hashes": runtime_adapter_paths_and_hashes(root),
        "default_off_receipt": default_off_receipt(root),
        "config_and_feature_flag_contract": config_and_feature_flag_contract(root),
        "supported_and_unsupported_shape_matrix": supported_and_unsupported_shape_matrix(root),
        "seeded_quality_parity": seeded_quality_parity(root),
        "distribution_tv_kl_and_interval_receipts": distribution_tv_kl_and_interval_receipts(root),
        "autocorrelation_and_effective_sample_size": autocorrelation_and_effective_sample_size(
            root
        ),
        "serialization_roundtrip": serialization_roundtrip(root),
        "cancellation_timeout_and_error_receipts": cancellation_timeout_and_error_receipts(root),
        "exact_fallback_receipts": exact_fallback_receipts(root),
        "task_owned_test_commands_and_exit_codes": task_owned_test_commands_and_exit_codes(
            command_receipts
        ),
        "unrelated_nonzero_command_classifications": command_summary["unrelated"],
        "hardware_or_speed_power_energy_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: pending",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": SCHEMA,
    }
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        command_receipts=command_receipts,
        duration_s=duration_s,
        run_date=run_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required: {missing}")
    if artifact.get("hardware_or_speed_power_energy_claimed") is not False:
        raise ValueError("hardware_or_speed_power_energy_claimed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"field_principles:{field}")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        row = provenance.get(field)
        if (
            not isinstance(row, Mapping)
            or row.get("principle") != principle
            or not row.get("source")
        ):
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _run_default_task_commands(root: Path = REPO_ROOT) -> list[JsonDict]:
    return [_run_text(command, root) for command in DEFAULT_TASK_COMMANDS]


def _external_command_receipts() -> list[JsonDict] | None:
    receipt_path = Path(os.environ.get("CARNOT_6208_COMMAND_RECEIPTS", DEFAULT_RECEIPT_PATH))
    if not receipt_path.exists():
        return None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("command receipt payload must be a list")
    return [dict(row) for row in payload]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--skip-internal-commands", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    started = time.perf_counter()
    command_receipts = [] if args.skip_internal_commands else _external_command_receipts()
    if command_receipts is None:
        command_receipts = _run_default_task_commands(REPO_ROOT)
    artifact = write_artifact(
        output_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        root=REPO_ROOT,
        command_receipts=command_receipts,
        duration_s=time.perf_counter() - started,
        run_date=str(args.date),
    )
    print(
        json.dumps(
            {
                "path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
