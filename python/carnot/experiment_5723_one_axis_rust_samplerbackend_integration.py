"""Exp5723 one-axis Rust SamplerBackend production integration.

Spec refs: REQ-SAMPLE-5723, SCENARIO-SAMPLE-5723.

This experiment promotes the already-validated one-axis Rust/PyO3 kernel across
Carnot's production ``SamplerBackend`` boundary. It keeps the algorithm fixed:
no ladder changes, no two-axis exchange, and no timing or hardware claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5715_one_axis_tempering_rust_quality_restart as exp5715
from carnot.samplers.backend import CpuBackend, SamplerBackend, get_backend
from carnot.samplers.one_axis_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    ENERGY_CONVENTION,
    ONE_AXIS_ALGORITHM,
    ONE_AXIS_BACKEND_SPEC_REFS,
    ONE_AXIS_TOPOLOGY,
    OneAxisRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5723_one_axis_rust_samplerbackend_integration.json")

EXPERIMENT = 5723
EXPERIMENT_ID = "exp5723-one-axis-rust-samplerbackend-integration"
MILESTONE = "2026.07.523"
RUN_DATE = "2026-07-19"
SCHEMA = "carnot.experiment_5723.one_axis_rust_samplerbackend_integration.v1"
SPEC_REFS = tuple(ONE_AXIS_BACKEND_SPEC_REFS)
INFERENCE_SUBSTRATE = "production_python_samplerbackend_plus_rust_pyo3_one_axis"
TERMINAL_PREFIXES = ("complete:", "blocked:")
DEFAULT_RANDOM_SEEDS = (5723, 5724, 5725)
FROZEN_TOLERANCES = {"energy": 1e-12, "proposal": 1e-12, "swap": 1e-12}

RUST_SOURCE_PATH = Path("crates/carnot-samplers/src/one_axis_tempering.rs")
PYO3_BINDING_PATH = Path("crates/carnot-python/src/one_axis_tempering.rs")
PYTHON_ADAPTER_PATH = Path("python/carnot/samplers/one_axis_rust_backend.py")
FACTORY_PATH = Path("python/carnot/samplers/backend.py")

BROKEN_CONTROL_IDS = (
    "broken_extension",
    "wrong_symbol",
    "malformed_descriptor",
    "unsupported_topology",
    "corrupt_checkpoint",
    "seed_mismatch",
    "energy_sign",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required production integration field exists before the sampler backend can be marked ready.",
    "upstream_gate_receipts": "Pins Exp5714 exact parity, Exp5715 hard-instance quality/restart, source hashes, algorithm identity, checkpoint schema, and two-axis-closed receipts.",
    "source_algorithm_hash": "Binds the adapter to the frozen one-axis algorithm recipe rather than a renamed or tuned variant.",
    "rust_source_hash": "Content-addresses the Rust kernel used by the adapter.",
    "pyo3_binding_hash": "Content-addresses the PyO3 binding symbols used by the adapter.",
    "python_adapter_hash": "Content-addresses the production Python adapter that implements fallback and receipts.",
    "sampler_backend_protocol": "Proves the adapter satisfies the production SamplerBackend boundary.",
    "factory_registration_receipt": "Proves explicit factory registration while preserving the cpu default.",
    "supported_descriptor_schema": "Declares the only descriptor accepted for the promoted one-axis path.",
    "unsupported_case_policy": "Separates exact fallback cases from fail-closed malformed or out-of-scope cases.",
    "seed_semantics": "Records the portable LCG state and seed mismatch rule used for replay and restart.",
    "temperature_ladder_receipt": "Proves the Exp5714/5715 one-axis beta ladder was not changed.",
    "swap_schedule_receipt": "Proves exchange remains label-only adjacent one-axis swapping.",
    "transition_budget_receipt": "Accounts for corrected transitions and swap attempts without adding work silently.",
    "energy_parity_max_error": "Quantifies production-adapter energy parity against the exact Python fallback.",
    "proposal_parity_max_error": "Quantifies production-adapter corrected proposal parity.",
    "swap_parity_max_error": "Quantifies production-adapter swap-ratio parity.",
    "decision_log_parity": "Proves the recorded production decision log matches across Rust and exact fallback.",
    "checkpoint_schema_version": "Pins the production adapter checkpoint schema.",
    "checkpoint_roundtrip_pass": "Proves duplicate save/load preserves state and receipts.",
    "python_to_rust_restart_pass": "Proves a Python fallback checkpoint can resume through the Rust adapter.",
    "rust_to_python_restart_pass": "Proves a Rust adapter checkpoint can resume through exact Python fallback.",
    "fallback_cases": "Lists every exercised exact fallback case and the recorded reason.",
    "fallback_equivalence_pass": "Gates fallback only when samples, checkpoints, and decision logs match exactly.",
    "exact_fallback_equivalence_score": "Provides a scalar gate equal to 1.0 only when all fallback cases are exact.",
    "broken_control_results": "Documents broken-extension, wrong-symbol, malformed descriptor, unsupported topology, corrupt checkpoint, seed mismatch, and energy-sign controls.",
    "one_axis_samplerbackend_ready_score": "Equals 1.0 only when upstream gates, protocol/factory exposure, parity, restart, fallback, and controls all pass.",
    "two_axis_code_added": "Bare false keeps retired penalty-axis exchange closed.",
    "timing_claimed": "Bare false prevents integration readiness from becoming a timing claim.",
    "hardware_speedup_claimed": "Bare false prevents PyO3 routing from becoming a hardware claim.",
    "inference_substrate": "Declares production Python SamplerBackend plus Rust/PyO3 one-axis execution with no LLM or board participation.",
    "random_seeds": "Records replay seeds for production adapter parity, fallback, and restart checks.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether production SamplerBackend integration is ready.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for reproducible artifact hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using Carnot's SHA-256 convention."""

    return exp5714.sha256_json(value)


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for provenance receipts."""

    return exp5714.file_sha256(path)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp5723 production integration artifact."""

    root_path = Path(root)
    seeds = tuple(int(seed) for seed in random_seeds)
    if not seeds:
        raise ValueError("random_seeds must not be empty")
    parity_seed = seeds[0]
    checkpoint_seed = seeds[1] if len(seeds) > 1 else seeds[0]
    fallback_seed = seeds[-1]
    parity = adapter_parity_receipt(parity_seed)
    fallback = fallback_receipts(fallback_seed)
    checkpoint = checkpoint_and_restart_receipts(checkpoint_seed)
    broken = broken_control_results(fallback_seed)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": upstream_gate_receipts(root_path),
        "source_algorithm_hash": exp5714.source_algorithm_hash(),
        "rust_source_hash": file_sha256(root_path / RUST_SOURCE_PATH),
        "pyo3_binding_hash": file_sha256(root_path / PYO3_BINDING_PATH),
        "python_adapter_hash": file_sha256(root_path / PYTHON_ADAPTER_PATH),
        "sampler_backend_protocol": sampler_backend_protocol_receipt(),
        "factory_registration_receipt": factory_registration_receipt(root_path),
        "supported_descriptor_schema": supported_descriptor_schema(),
        "unsupported_case_policy": unsupported_case_policy(),
        "seed_semantics": seed_semantics(),
        "temperature_ladder_receipt": temperature_ladder_receipt(),
        "swap_schedule_receipt": swap_schedule_receipt(),
        "transition_budget_receipt": parity["transition_budget_receipt"],
        "energy_parity_max_error": parity["energy_parity_max_error"],
        "proposal_parity_max_error": parity["proposal_parity_max_error"],
        "swap_parity_max_error": parity["swap_parity_max_error"],
        "decision_log_parity": parity["decision_log_parity"],
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_roundtrip_pass": checkpoint["checkpoint_roundtrip_pass"],
        "python_to_rust_restart_pass": checkpoint["python_to_rust_restart_pass"],
        "rust_to_python_restart_pass": checkpoint["rust_to_python_restart_pass"],
        "fallback_cases": fallback,
        "fallback_equivalence_pass": all(row["equivalent"] is True for row in fallback),
        "exact_fallback_equivalence_score": 1.0
        if all(row["equivalent"] is True for row in fallback)
        else 0.0,
        "broken_control_results": broken,
        "one_axis_samplerbackend_ready_score": 0.0,
        "two_axis_code_added": False,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in seeds],
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: production integration gates not evaluated",
    }
    artifact["one_axis_samplerbackend_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def upstream_gate_receipts(root: Path) -> JsonDict:
    """Verify Exp5714/5715 artifacts and return readiness receipts."""

    exp5714_path = root / exp5714.RESULT_RELATIVE_PATH
    exp5715_path = root / exp5715.RESULT_RELATIVE_PATH
    exp5714_payload = _read_json(exp5714_path)
    exp5715_payload = _read_json(exp5715_path)
    exp5714.validate_artifact(exp5714_payload)
    exp5715.validate_artifact(exp5715_payload)
    algorithm_hash = exp5714.source_algorithm_hash()
    return {
        "exp5714": {
            "path": exp5714.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": file_sha256(exp5714_path),
            "ready_score": exp5714_payload["one_axis_rust_parity_ready_score"],
            "ready": exp5714_payload["one_axis_rust_parity_ready_score"] == 1.0,
            "algorithm_hash_matches": exp5714_payload["source_algorithm_hash"] == algorithm_hash,
            "two_axis_closed": exp5714_payload["two_axis_code_added"] is False,
            "timing_claimed": exp5714_payload["timing_claimed"],
            "hardware_speedup_claimed": exp5714_payload["hardware_speedup_claimed"],
        },
        "exp5715": {
            "path": exp5715.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": file_sha256(exp5715_path),
            "ready_score": exp5715_payload["one_axis_rust_quality_ready_score"],
            "ready": exp5715_payload["one_axis_rust_quality_ready_score"] == 1.0,
            "checkpoint_schema": exp5715_payload["checkpoint_schema_version"],
            "checkpoint_schema_matches": exp5715_payload["checkpoint_schema_version"]
            == exp5715.CHECKPOINT_SCHEMA_VERSION,
            "two_axis_closed": exp5715_payload["two_axis_arm_count"] == 0,
            "timing_claimed": exp5715_payload["timing_claimed"],
            "hardware_speedup_claimed": exp5715_payload["hardware_speedup_claimed"],
        },
        "two_axis_closed_receipts": {
            "exp5714_two_axis_code_added": exp5714_payload["two_axis_code_added"],
            "exp5715_two_axis_arm_count": exp5715_payload["two_axis_arm_count"],
            "closed": exp5714_payload["two_axis_code_added"] is False
            and exp5715_payload["two_axis_arm_count"] == 0,
        },
    }


def sampler_backend_protocol_receipt() -> JsonDict:
    """Return production protocol conformance evidence."""

    backend = OneAxisRustBackend()
    return {
        "backend_class": "OneAxisRustBackend",
        "backend_name": backend.backend_name,
        "is_sampler_backend": isinstance(backend, SamplerBackend),
        "methods": {
            name: callable(getattr(backend, name, None))
            for name in ("sample", "minimize_energy", "set_constraints", "dual_update_step")
        },
        "set_constraints_noop": backend.set_constraints(lambda state: state) is None,
        "dual_update_step_noop": backend.dual_update_step(0.1) is None,
    }


def factory_registration_receipt(root: Path) -> JsonDict:
    """Return explicit factory registration and default-preservation evidence."""

    default_backend = get_backend()
    one_axis = get_backend("one_axis_rust")
    return {
        "factory_path": FACTORY_PATH.as_posix(),
        "factory_hash": file_sha256(root / FACTORY_PATH),
        "explicit_backend_name": one_axis.backend_name,
        "explicit_backend_class": type(one_axis).__name__,
        "default_backend_name": default_backend.backend_name,
        "default_backend_class": type(default_backend).__name__,
        "default_backend_preserved": isinstance(default_backend, CpuBackend)
        and default_backend.backend_name == "cpu",
        "registered_without_default_change": isinstance(one_axis, OneAxisRustBackend)
        and default_backend.backend_name == "cpu",
    }


def supported_descriptor_schema() -> JsonDict:
    """Return the frozen descriptor schema for the production adapter."""

    return {
        "algorithm": ONE_AXIS_ALGORITHM,
        "topology": ONE_AXIS_TOPOLOGY,
        "required_fields": [
            "algorithm",
            "topology",
            "source_algorithm_hash",
            "beta_ladder",
            "proposal_std",
            "drift_scale",
            "seed",
        ],
        "optional_fields": [
            "initial_states",
            "initial_labels",
            "burn_in_sweeps",
            "checkpoint",
            "force_python_fallback",
        ],
        "spin_values": [-1, 1],
        "sample_return_convention": "bool array where True means +1 and False means -1",
    }


def unsupported_case_policy() -> JsonDict:
    """Declare exact fallback versus fail-closed cases."""

    return {
        "missing_extension": "exact_python_fallback",
        "broken_extension": "exact_python_fallback",
        "wrong_symbol": "exact_python_fallback",
        "unsupported_dtype_or_layout": "exact_python_fallback",
        "declared_python_compatibility": "exact_python_fallback",
        "malformed_descriptor": "fail_closed",
        "unsupported_topology": "fail_closed",
        "invalid_shape": "fail_closed",
        "corrupt_checkpoint": "fail_closed",
        "seed_mismatch": "fail_closed",
        "two_axis_request": "fail_closed",
    }


def seed_semantics() -> JsonDict:
    """Return the portable seed semantics used by adapter receipts."""

    return {
        "rng": "portable_lcg_u64",
        "lcg_a": exp5714.LCG_A,
        "lcg_c": exp5714.LCG_C,
        "seed_range": "0 <= seed < 2**64",
        "checkpoint_carries_rng_state": True,
        "seed_mismatch_policy": "fail_closed",
    }


def temperature_ladder_receipt() -> JsonDict:
    """Return beta-ladder identity evidence."""

    return {
        "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
        "beta_ladder_hash": exp5715.beta_ladder_hash(),
        "matches_exp5714": tuple(exp5714.BETA_LADDER) == tuple(exp5715.BETA_LADDER),
        "source_algorithm_hash": exp5714.source_algorithm_hash(),
    }


def swap_schedule_receipt() -> JsonDict:
    """Return label-only one-axis schedule evidence."""

    core = exp5714.PythonOneAxisTemperingCore(exp5714.default_config())
    trace = core.scheduler_trace()
    return {
        "scheduler_trace": trace,
        "label_only_adjacent_swaps": trace[-2:] == ["swap:0-1", "swap:1-2"],
        "state_copy_swaps_allowed": False,
        "two_axis_exchange": False,
    }


def adapter_parity_receipt(seed: int) -> JsonDict:
    """Compare production Rust routing against exact Python fallback."""

    biases, couplings, descriptor = _fixture(seed, burn_in_sweeps=1)
    rust_result = OneAxisRustBackend(seed=seed).run_descriptor(
        biases,
        couplings,
        n_samples=3,
        config=descriptor,
    )
    python_result = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        biases,
        couplings,
        n_samples=3,
        config=descriptor,
    )
    errors = _decision_log_errors(rust_result["decision_log"], python_result["decision_log"])
    return {
        **errors,
        "decision_log_parity": rust_result["decision_log"] == python_result["decision_log"],
        "transition_budget_receipt": {
            "rust_active_backend": rust_result["receipt"]["active_backend"],
            "python_active_backend": python_result["receipt"]["active_backend"],
            "matched": rust_result["receipt"]["transition_budget"]
            == python_result["receipt"]["transition_budget"],
            "budget": rust_result["receipt"]["transition_budget"],
            "algorithm": ONE_AXIS_ALGORITHM,
            "ladder_unchanged": rust_result["checkpoint"]["beta_ladder"]
            == [float(beta) for beta in exp5714.BETA_LADDER],
        },
    }


def checkpoint_and_restart_receipts(seed: int) -> JsonDict:
    """Verify duplicate save/load and cross-language restart through the adapter."""

    biases, couplings, descriptor = _fixture(seed, burn_in_sweeps=1)
    rust_backend = OneAxisRustBackend(seed=seed)
    rust_prefix = rust_backend.run_descriptor(biases, couplings, 2, descriptor)
    checkpoint_a = rust_backend.save_checkpoint()
    checkpoint_b = rust_backend.save_checkpoint()
    roundtrip_state = rust_backend.load_checkpoint(
        checkpoint_a,
        biases,
        couplings,
        config=descriptor,
    )
    checkpoint_roundtrip_pass = (
        checkpoint_a == checkpoint_b
        and checkpoint_a["payload_checksum"] == checkpoint_checksum(checkpoint_a)
        and roundtrip_state == checkpoint_a["state"]
        and rust_prefix["checkpoint"] == checkpoint_a
    )

    python_prefix_backend = OneAxisRustBackend(seed=seed, prefer_rust=False)
    python_prefix = python_prefix_backend.run_descriptor(biases, couplings, 2, descriptor)
    python_checkpoint = python_prefix_backend.save_checkpoint()
    python_to_rust_config = {**descriptor, "checkpoint": python_checkpoint, "burn_in_sweeps": 0}
    python_to_rust = OneAxisRustBackend(seed=seed).run_descriptor(
        biases,
        couplings,
        2,
        python_to_rust_config,
    )
    python_to_python = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        biases,
        couplings,
        2,
        python_to_rust_config,
    )

    rust_to_python_config = {**descriptor, "checkpoint": checkpoint_a, "burn_in_sweeps": 0}
    rust_to_python = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        biases,
        couplings,
        2,
        rust_to_python_config,
    )
    rust_to_rust = OneAxisRustBackend(seed=seed).run_descriptor(
        biases,
        couplings,
        2,
        rust_to_python_config,
    )

    return {
        "checkpoint_roundtrip_pass": checkpoint_roundtrip_pass,
        "python_to_rust_restart_pass": _same_run_suffix(python_to_rust, python_to_python)
        and python_prefix["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK,
        "rust_to_python_restart_pass": _same_run_suffix(rust_to_python, rust_to_rust)
        and rust_prefix["receipt"]["active_backend"] == ACTIVE_RUST_BACKEND,
    }


def fallback_receipts(seed: int) -> list[JsonDict]:
    """Exercise every declared exact fallback case."""

    biases, couplings, descriptor = _fixture(seed, burn_in_sweeps=1)
    float32_biases = np.asarray(biases, dtype=np.float32)
    float32_couplings = np.asarray(couplings, dtype=np.float32)
    cases = [
        (
            "missing_extension",
            biases,
            couplings,
            descriptor,
            OneAxisRustBackend(
                seed=seed,
                rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
            ),
            "rust_extension_missing",
        ),
        (
            "broken_extension",
            biases,
            couplings,
            descriptor,
            OneAxisRustBackend(
                seed=seed,
                rust_module_loader=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
            ),
            "rust_extension_error",
        ),
        (
            "wrong_symbol",
            biases,
            couplings,
            descriptor,
            OneAxisRustBackend(seed=seed, rust_module_loader=lambda: SimpleNamespace()),
            "rust_symbol_missing",
        ),
        (
            "unsupported_dtype",
            float32_biases,
            float32_couplings,
            descriptor,
            OneAxisRustBackend(seed=seed),
            "unsupported_dtype_or_layout",
        ),
        (
            "unsupported_layout",
            biases,
            np.asfortranarray(couplings),
            descriptor,
            OneAxisRustBackend(seed=seed),
            "unsupported_dtype_or_layout",
        ),
        (
            "declared_python_compatibility",
            biases,
            couplings,
            {**descriptor, "force_python_fallback": True},
            OneAxisRustBackend(seed=seed),
            "declared_python_compatibility",
        ),
    ]
    rows: list[JsonDict] = []
    for case_id, case_biases, case_couplings, case_descriptor, backend, reason in cases:
        baseline = OneAxisRustBackend(seed=seed, prefer_rust=False).run_descriptor(
            case_biases,
            case_couplings,
            2,
            case_descriptor,
        )
        result = backend.run_descriptor(case_biases, case_couplings, 2, case_descriptor)
        equivalent = _same_run_suffix(result, baseline)
        rows.append(
            {
                "case_id": case_id,
                "active_backend": result["receipt"]["active_backend"],
                "fallback_reason": result["receipt"]["fallback_reason"],
                "expected_reason_fragment": reason,
                "equivalent": equivalent
                and result["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
                and reason in str(result["receipt"]["fallback_reason"]),
            }
        )
    return rows


def broken_control_results(seed: int) -> list[JsonDict]:
    """Run preregistered broken controls for the production adapter."""

    biases, couplings, descriptor = _fixture(seed, burn_in_sweeps=1)
    controls: list[JsonDict] = []
    broken = OneAxisRustBackend(
        seed=seed,
        rust_module_loader=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
    ).run_descriptor(biases, couplings, 2, descriptor)
    controls.append(
        {
            "control_id": "broken_extension",
            "passed": broken["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
            and "rust_extension_error" in str(broken["receipt"]["fallback_reason"]),
            "policy": "exact_python_fallback",
        }
    )
    wrong = OneAxisRustBackend(seed=seed, rust_module_loader=lambda: SimpleNamespace())
    wrong_result = wrong.run_descriptor(biases, couplings, 2, descriptor)
    controls.append(
        {
            "control_id": "wrong_symbol",
            "passed": wrong_result["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
            and "rust_symbol_missing" in str(wrong_result["receipt"]["fallback_reason"]),
            "policy": "exact_python_fallback",
        }
    )
    malformed = deepcopy(descriptor)
    malformed.pop("algorithm")
    controls.append(
        {
            "control_id": "malformed_descriptor",
            "passed": _raises_value_error(
                lambda: OneAxisRustBackend(seed=seed).run_descriptor(
                    biases,
                    couplings,
                    2,
                    malformed,
                )
            ),
            "policy": "fail_closed",
        }
    )
    controls.append(
        {
            "control_id": "unsupported_topology",
            "passed": _raises_value_error(
                lambda: OneAxisRustBackend(seed=seed).run_descriptor(
                    biases,
                    couplings,
                    2,
                    {**descriptor, "topology": "two_axis_temperature_penalty_exchange"},
                )
            ),
            "policy": "fail_closed",
        }
    )
    prefix_backend = OneAxisRustBackend(seed=seed)
    prefix_backend.run_descriptor(biases, couplings, 2, descriptor)
    checkpoint = prefix_backend.save_checkpoint()
    corrupt = deepcopy(checkpoint)
    corrupt["state"]["sweep"] = int(corrupt["state"]["sweep"]) + 1
    controls.append(
        {
            "control_id": "corrupt_checkpoint",
            "passed": _raises_value_error(
                lambda: prefix_backend.load_checkpoint(
                    corrupt,
                    biases,
                    couplings,
                    config=descriptor,
                )
            ),
            "policy": "fail_closed",
        }
    )
    controls.append(
        {
            "control_id": "seed_mismatch",
            "passed": _raises_value_error(
                lambda: prefix_backend.load_checkpoint(
                    checkpoint,
                    biases,
                    couplings,
                    config={**descriptor, "seed": int(descriptor["seed"]) + 1},
                )
            ),
            "policy": "fail_closed",
        }
    )
    controls.append(
        OneAxisRustBackend(seed=seed).energy_sign_control(biases, couplings, descriptor)
    )
    controls[-1]["passed"] = bool(controls[-1].pop("rejected"))
    controls[-1]["policy"] = "reject_wrong_energy_sign"
    return controls


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5723 fields and fail closed on manual promotion edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("two_axis_code_added") is not False:
        raise ValueError("two_axis_code_added must be false")
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("one_axis_samplerbackend_ready_score") != ready_score(payload):
        raise ValueError("one_axis_samplerbackend_ready_score mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start complete: or blocked:")
    if verdict != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return the downstream scalar gate for production SamplerBackend readiness."""

    receipts = payload.get("upstream_gate_receipts", {})
    gates = [
        isinstance(receipts, Mapping)
        and receipts.get("exp5714", {}).get("ready") is True
        and receipts.get("exp5714", {}).get("algorithm_hash_matches") is True
        and receipts.get("exp5715", {}).get("ready") is True
        and receipts.get("exp5715", {}).get("checkpoint_schema_matches") is True
        and receipts.get("two_axis_closed_receipts", {}).get("closed") is True,
        payload.get("sampler_backend_protocol", {}).get("is_sampler_backend") is True,
        payload.get("factory_registration_receipt", {}).get("registered_without_default_change")
        is True,
        payload.get("transition_budget_receipt", {}).get("matched") is True,
        float(payload.get("energy_parity_max_error", 1.0)) <= FROZEN_TOLERANCES["energy"],
        float(payload.get("proposal_parity_max_error", 1.0)) <= FROZEN_TOLERANCES["proposal"],
        float(payload.get("swap_parity_max_error", 1.0)) <= FROZEN_TOLERANCES["swap"],
        payload.get("decision_log_parity") is True,
        payload.get("checkpoint_schema_version") == CHECKPOINT_SCHEMA_VERSION,
        payload.get("checkpoint_roundtrip_pass") is True,
        payload.get("python_to_rust_restart_pass") is True,
        payload.get("rust_to_python_restart_pass") is True,
        payload.get("fallback_equivalence_pass") is True,
        payload.get("exact_fallback_equivalence_score") == 1.0,
        isinstance(payload.get("broken_control_results"), list)
        and {row.get("control_id") for row in payload["broken_control_results"]}
        == set(BROKEN_CONTROL_IDS)
        and all(row.get("passed") is True for row in payload["broken_control_results"]),
        payload.get("two_axis_code_added") is False,
        payload.get("timing_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the terminal honest verdict for Exp5723."""

    if ready_score(payload) == 1.0:
        return "complete: one-axis Rust/PyO3 kernel is exposed through SamplerBackend with exact fallback, restart receipts, and no timing or hardware claim"
    return "blocked: one-axis Rust SamplerBackend integration gate failed"


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal JSON artifact and return its path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(seed: int, *, burn_in_sweeps: int) -> tuple[np.ndarray, np.ndarray, JsonDict]:
    config = exp5714.default_config()
    state = exp5714.default_state(seed=seed)
    descriptor = descriptor_for_run(
        seed=seed,
        initial_states=state.states.astype(int).tolist(),
        initial_labels=list(state.labels),
        burn_in_sweeps=burn_in_sweeps,
    )
    return (
        np.asarray(config.fields, dtype=np.float64),
        np.asarray(config.couplings, dtype=np.float64),
        descriptor,
    )


def _decision_log_errors(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> JsonDict:
    energy_error = 0.0
    proposal_error = 0.0
    swap_error = 0.0
    if len(left) != len(right):
        return {
            "energy_parity_max_error": 1.0,
            "proposal_parity_max_error": 1.0,
            "swap_parity_max_error": 1.0,
        }
    for lhs, rhs in zip(left, right, strict=True):
        for key in ("current_energy", "proposed_energy"):
            if key in lhs:
                energy_error = max(energy_error, abs(float(lhs[key]) - float(rhs[key])))
        for key in ("proposal_log_forward", "proposal_log_reverse"):
            if key in lhs:
                proposal_error = max(proposal_error, abs(float(lhs[key]) - float(rhs[key])))
        if "log_ratio" in lhs:
            swap_error = max(swap_error, abs(float(lhs["log_ratio"]) - float(rhs["log_ratio"])))
    return {
        "energy_parity_max_error": energy_error,
        "proposal_parity_max_error": proposal_error,
        "swap_parity_max_error": swap_error,
    }


def _same_run_suffix(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        np.array_equal(left["samples"], right["samples"])
        and left["samples_spin"] == right["samples_spin"]
        and left["decision_log"] == right["decision_log"]
        and left["checkpoint"]["state"] == right["checkpoint"]["state"]
    )


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _raises_value_error(call: Any) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT, random_seeds=DEFAULT_RANDOM_SEEDS)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
