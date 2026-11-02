"""Exp6220 mode-jump runtime A/B measurement.

Spec refs: REQ-SAMPLE-6220,
SCENARIO-SAMPLE-6220-MATCHED-RUNTIME-QUALITY,
SCENARIO-SAMPLE-6220-UNSUPPORTED-FIXTURE-BOUNDARY,
SCENARIO-SAMPLE-6220-STATE-FALLBACK-NOCLAIM.

This experiment measures the Exp6208 runtime choice. It does not change the
mode-jump backend contract. Unsupported fixtures are recorded as boundary
evidence instead of being forced through a wider sampler surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import numpy as np

from carnot import experiment_6194_mode_jump_rust_pyo3_parity as exp6194
from carnot import experiment_6208_mode_jump_runtime_integration as exp6208
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
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
RESULT_RELATIVE_PATH = Path("results/experiment_6220_mode_jump_runtime_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6220_mode_jump_runtime_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6220_mode_jump_runtime_ab.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
FACTORY_RELATIVE_PATH = Path("python/carnot/samplers/backend.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")
EXP6194_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6194_mode_jump_rust_pyo3_parity.py")
EXP6208_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6208_mode_jump_runtime_integration.py"
)
EXP6194_RESULT_RELATIVE_PATH = Path("results/experiment_6194_mode_jump_rust_pyo3_parity.json")
EXP6208_RESULT_RELATIVE_PATH = Path("results/experiment_6208_mode_jump_runtime_integration.json")
EXP6166_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6166_mode_jumping_factor_thermalization.json"
)
EXP6180_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6180_exp6166_reproducibility_adjudication.json"
)

SCHEMA = "carnot.experiment_6220.mode_jump_runtime_ab.v1"
EXPERIMENT_ID = "experiment_6220_mode_jump_runtime_ab"
RUN_DATE = "20260809"
INFERENCE_SUBSTRATE = "local_cpu_software_mode_jump_runtime_ab_no_hardware"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6220_command_receipts.json")

SEED = 6220
BURN_IN = 128
SAMPLE_COUNT = 4096
RESTART_PREFIX_COUNT = 32
RESTART_SUFFIX_COUNT = 32
TIMING_REPEATS = 5

TOLERANCES: JsonDict = {
    "target_tv": 0.02,
    "target_kl": 0.002,
    "energy_mean_abs_error": 0.03,
    "energy_variance_abs_error": 0.05,
    "mode_mass_abs_error": 0.03,
    "ess_min": 1000.0,
    "autocorrelation_abs_max": 0.8,
    "frequency_delta_max": 0.0,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_exp6194_and_exp6208_paths_hashes",
    "backend_paths_build_and_abi_receipts",
    "preregistered_fixture_seed_schedule_matrix",
    "matched_arm_configuration",
    "support_validity_by_fixture_arm",
    "observable_and_energy_error_by_fixture_arm",
    "ess_and_autocorrelation_by_fixture_arm",
    "transition_and_mode_occupancy_counts",
    "serialization_and_restart_receipts",
    "fallback_trigger_and_exactness",
    "cpu_thread_and_wall_time_receipts",
    "quality_gate_passed",
    "timing_claim_allowed",
    "default_off_preserved",
    "fpga_tsu_power_hardware_claim_count",
    "task_owned_and_preexisting_test_classification",
    "sampler_runtime_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates ready, partial, and blocked Exp6220 outcomes.",
    "upstream_exp6194_and_exp6208_paths_hashes": "Pins the qualified parity and runtime-integration evidence before this A/B run is trusted.",
    "backend_paths_build_and_abi_receipts": "Records adapter, factory, Rust kernel, PyO3 binding, import, and ABI availability.",
    "preregistered_fixture_seed_schedule_matrix": "Freezes fixtures, seeds, schedules, burn-in, counts, observables, tolerances, and timing rule before outcomes.",
    "matched_arm_configuration": "Proves the fallback and runtime arms use matched seeds, target, proposal, burn-in, and sample counts.",
    "support_validity_by_fixture_arm": "Separates accepted support, exact fallback, Rust execution, and fail-closed unsupported fixtures.",
    "observable_and_energy_error_by_fixture_arm": "Reports observable, mode-mass, and energy-moment errors before timing.",
    "ess_and_autocorrelation_by_fixture_arm": "Reports mixing quality before any runtime interpretation.",
    "transition_and_mode_occupancy_counts": "Shows accepted/proposed transitions and mode occupancy for supported runs.",
    "serialization_and_restart_receipts": "Proves checkpoint round trip, restart equivalence, and malformed state rejection.",
    "fallback_trigger_and_exactness": "Proves disabled, forced, unavailable, and dtype/layout fallback paths remain exact.",
    "cpu_thread_and_wall_time_receipts": "Stores CPU/thread context and diagnostic wall-clock measurements only.",
    "quality_gate_passed": "Bare boolean gates timing interpretation after support and quality checks.",
    "timing_claim_allowed": "Bare boolean stays false unless quality passes and uncertainty excludes parity.",
    "default_off_preserved": "Bare true only when the production default remains CPU and Rust execution needs explicit opt-in.",
    "fpga_tsu_power_hardware_claim_count": "Bare integer must be `0` for this software-only task.",
    "task_owned_and_preexisting_test_classification": "Keeps task-owned failures separate from pre-existing repository-wide nonzero checks.",
    "sampler_runtime_ready_score": "Summarizes readiness from default-off, support, quality, state, fallback, tests, and no-claim gates.",
    "protected_files_unchanged": "Confirms conductor and reconciliation files were not modified.",
    "inference_substrate": "Declares local CPU software A/B measurement, not hardware, TSU, GPU, or LLM inference.",
    "verifier_is_oracle": "States that the verifier is exact finite enumeration plus fixed transition receipts.",
    "field_provenance": "Maps every required field to prompt, spec, source hashes, tests, commands, or computed fixtures.",
    "field_principles": "Explains why each required field exists.",
    "test_commands": "Records focused tests, coverage, artifact command, required suite, and applicable E2E receipts.",
    "test_exit_codes": "Stores exit codes for every recorded command.",
    "duration_s": "Reports real wall time without padding.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile duration and checksum blanked.",
    "honest_verdict": "Uses a terminal prefix and states support, quality, timing, fallback, default-off, and nonzero-command classifications.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _fixture_payload_hash(payload: Mapping[str, Any]) -> str:
    stable = {
        "name": payload["name"],
        "labels": payload["labels"],
        "target_probabilities": [float(value) for value in payload["target_probabilities"]],
        "proposal_probabilities": np.asarray(payload["proposal_probabilities"]).tolist(),
        "fixture_class": payload["fixture_class"],
    }
    return sha256_json(stable)


def _fixtures(root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    multimodal: JsonDict = {
        "name": "multimodal_exp6194",
        "fixture_class": "multimodal_supported",
        "labels": labels,
        "target_probabilities": target.astype(float).tolist(),
        "proposal_probabilities": proposal.astype(float).tolist(),
        "mode_labels": {
            "left_mode": ["left_peak", "left_shoulder"],
            "right_mode": ["right_peak", "right_shoulder"],
            "valley": ["valley_left", "valley_right"],
        },
        "expected_adapter_support": True,
        "source": EXP6194_RESULT_RELATIVE_PATH.as_posix(),
    }
    unimodal_target = np.asarray([0.86, 0.06, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    unimodal: JsonDict = {
        "name": "unimodal_contract_probe",
        "fixture_class": "unimodal_unsupported_by_exp6208_contract",
        "labels": labels,
        "target_probabilities": unimodal_target.astype(float).tolist(),
        "proposal_probabilities": proposal.astype(float).tolist(),
        "mode_labels": {"single_mode": ["left_peak"], "tail": labels[1:]},
        "expected_adapter_support": False,
        "source": "preregistered_exp6220_contract_probe",
    }
    for fixture in (multimodal, unimodal):
        fixture["fixture_sha256"] = _fixture_payload_hash(fixture)
    return {fixture["name"]: fixture for fixture in (multimodal, unimodal)}


def preregistered_fixture_seed_schedule_matrix(root: Path = REPO_ROOT) -> JsonDict:
    fixtures = _fixtures(root)
    return {
        "matrix_sha256": sha256_json(
            {
                "fixtures": {name: fixture["fixture_sha256"] for name, fixture in fixtures.items()},
                "seed": SEED,
                "burn_in": BURN_IN,
                "sample_count": SAMPLE_COUNT,
                "timing_repeats": TIMING_REPEATS,
                "tolerances": TOLERANCES,
            }
        ),
        "fixtures": {
            name: {
                "fixture_sha256": fixture["fixture_sha256"],
                "fixture_class": fixture["fixture_class"],
                "expected_adapter_support": fixture["expected_adapter_support"],
                "source": fixture["source"],
            }
            for name, fixture in fixtures.items()
        },
        "seed": SEED,
        "schedule": {
            "algorithm": MODE_JUMP_ALGORITHM,
            "topology": MODE_JUMP_TOPOLOGY,
            "burn_in": BURN_IN,
            "retained_sample_count": SAMPLE_COUNT,
            "temperature_or_beta_changed_from_exp6208": False,
            "stopping_rule_changed_from_outcomes": False,
        },
        "observables": [
            "per_label_frequency",
            "mode_mass",
            "energy_mean",
            "energy_variance",
            "total_variation",
            "kl_target_to_empirical",
            "ess",
            "lag1_autocorrelation",
            "transition_counts",
        ],
        "tolerances": dict(TOLERANCES),
        "timing_rule": (
            "diagnostic unless all fixture arms are supported, quality gates pass, "
            "and 95% mean wall-time intervals exclude parity"
        ),
        "principle": FIELD_PRINCIPLES["preregistered_fixture_seed_schedule_matrix"],
    }


def matched_arm_configuration(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    target_hash = sha256_json(target.astype(float).tolist())
    proposal_hash = sha256_json(proposal.astype(float).tolist())
    return {
        "arms": {
            "fallback_exact": {
                "backend_class": "ModeJumpRustBackend",
                "prefer_rust": False,
                "expected_active_backend": ACTIVE_PYTHON_FALLBACK,
            },
            "mode_jump_runtime": {
                "backend_class": "ModeJumpRustBackend",
                "prefer_rust": True,
                "expected_active_backend": ACTIVE_RUST_BACKEND,
                "feature_config_flag": "enable_mode_jump_runtime",
            },
        },
        "matched_seed": SEED,
        "matched_burn_in": BURN_IN,
        "matched_sample_count": SAMPLE_COUNT,
        "matched_label_order": labels,
        "matched_multimodal_target_hash": target_hash,
        "matched_multimodal_proposal_hash": proposal_hash,
        "couplings_or_proposal_changed_between_arms": False,
        "temperature_or_stopping_changed_between_arms": False,
        "principle": FIELD_PRINCIPLES["matched_arm_configuration"],
    }


def _descriptor(labels: Sequence[str], *, return_trace: bool = True) -> JsonDict:
    descriptor = descriptor_for_run(
        labels=labels,
        seed=SEED,
        burn_in=BURN_IN,
        enable_mode_jump_runtime=True,
    )
    descriptor["return_trace"] = bool(return_trace)
    return descriptor


def _run_arm(fixture: Mapping[str, Any], arm_name: str, *, return_trace: bool) -> JsonDict:
    labels = [str(label) for label in fixture["labels"]]
    target = np.asarray(fixture["target_probabilities"], dtype=np.float64)
    proposal = np.asarray(fixture["proposal_probabilities"], dtype=np.float64)
    backend = ModeJumpRustBackend(seed=SEED, prefer_rust=arm_name == "mode_jump_runtime")
    started = time.perf_counter()
    try:
        result = backend.run_descriptor(
            target,
            proposal,
            n_samples=SAMPLE_COUNT,
            config=_descriptor(labels, return_trace=return_trace),
        )
    except Exception as exc:  # noqa: BLE001 - boundary result is part of the receipt.
        return {
            "success": False,
            "fixture": fixture["name"],
            "arm": arm_name,
            "elapsed_s": _stable_float(time.perf_counter() - started),
            "support_valid": False,
            "fail_closed": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "success": True,
        "fixture": fixture["name"],
        "arm": arm_name,
        "elapsed_s": _stable_float(time.perf_counter() - started),
        "support_valid": True,
        "fail_closed": False,
        "active_backend": result["receipt"]["active_backend"],
        "fallback_reason": result["receipt"]["fallback_reason"],
        "result": result,
    }


def _measure_fixture_arms(root: Path = REPO_ROOT) -> dict[str, dict[str, JsonDict]]:
    rows: dict[str, dict[str, JsonDict]] = {}
    for fixture_name, fixture in _fixtures(root).items():
        rows[fixture_name] = {
            "fallback_exact": _run_arm(fixture, "fallback_exact", return_trace=True),
            "mode_jump_runtime": _run_arm(fixture, "mode_jump_runtime", return_trace=True),
        }
    return rows


def support_validity_by_fixture_arm(measurements: Mapping[str, Mapping[str, JsonDict]]) -> JsonDict:
    fixtures: dict[str, JsonDict] = {}
    for fixture_name, arms in measurements.items():
        fixtures[fixture_name] = {}
        for arm_name, row in arms.items():
            fixtures[fixture_name][arm_name] = {
                key: value
                for key, value in row.items()
                if key
                in {
                    "support_valid",
                    "fail_closed",
                    "active_backend",
                    "fallback_reason",
                    "error_type",
                    "message",
                    "elapsed_s",
                }
            }
    all_required_supported = all(
        row["support_valid"] for arms in fixtures.values() for row in arms.values()
    )
    unsupported_fail_closed = all(
        row["fail_closed"]
        for arms in fixtures.values()
        for row in arms.values()
        if row["support_valid"] is False
    )
    return {
        "fixtures": fixtures,
        "all_required_fixture_arms_supported": all_required_supported,
        "unsupported_fixture_arms_fail_closed": unsupported_fail_closed,
        "supported_fixture_arm_count": sum(
            row["support_valid"] for arms in fixtures.values() for row in arms.values()
        ),
        "principle": FIELD_PRINCIPLES["support_validity_by_fixture_arm"],
    }


def _mode_masses(
    labels: Sequence[str], mode_labels: Mapping[str, Sequence[str]], rows: Any
) -> dict[str, float]:
    if isinstance(rows, np.ndarray):
        target_lookup = {label: float(rows[index]) for index, label in enumerate(labels)}
        return {
            mode: sum(target_lookup[str(label)] for label in members)
            for mode, members in mode_labels.items()
        }
    counts = Counter(rows)
    total = sum(counts.values())
    masses: dict[str, float] = {}
    for mode, members in mode_labels.items():
        masses[mode] = sum(counts[str(label)] for label in members) / total
    return masses


def _energy_moments(
    labels: Sequence[str], target: np.ndarray, sample_labels: Sequence[str]
) -> JsonDict:
    energies = {label: -math.log(float(target[index])) for index, label in enumerate(labels)}
    exact_mean = sum(float(target[index]) * energies[label] for index, label in enumerate(labels))
    exact_second = sum(
        float(target[index]) * energies[label] * energies[label]
        for index, label in enumerate(labels)
    )
    sample_values = [energies[str(label)] for label in sample_labels]
    sample_mean = float(np.mean(sample_values))
    sample_variance = float(np.var(sample_values))
    exact_variance = exact_second - exact_mean * exact_mean
    return {
        "exact_energy_mean": _stable_float(exact_mean),
        "sample_energy_mean": _stable_float(sample_mean),
        "energy_mean_abs_error": _stable_float(abs(sample_mean - exact_mean)),
        "exact_energy_variance": _stable_float(exact_variance),
        "sample_energy_variance": _stable_float(sample_variance),
        "energy_variance_abs_error": _stable_float(abs(sample_variance - exact_variance)),
    }


def observable_and_energy_error_by_fixture_arm(
    root: Path,
    measurements: Mapping[str, Mapping[str, JsonDict]],
) -> JsonDict:
    fixtures = _fixtures(root)
    rows: dict[str, JsonDict] = {}
    all_pass = True
    for fixture_name, arms in measurements.items():
        rows[fixture_name] = {}
        fixture = fixtures[fixture_name]
        labels = [str(label) for label in fixture["labels"]]
        target = np.asarray(fixture["target_probabilities"], dtype=np.float64)
        exact_modes = _mode_masses(labels, fixture["mode_labels"], target)
        fallback_labels = (
            arms["fallback_exact"]["result"]["sample_labels"]
            if arms["fallback_exact"]["success"]
            else None
        )
        for arm_name, run in arms.items():
            if not run["success"]:
                rows[fixture_name][arm_name] = {
                    "support_valid": False,
                    "quality_evaluated": False,
                    "skipped_reason": run["message"],
                }
                continue
            result = run["result"]
            metrics = result["metrics"]
            sample_labels = [str(label) for label in result["sample_labels"]]
            sample_modes = _mode_masses(labels, fixture["mode_labels"], sample_labels)
            mode_errors = {
                mode: _stable_float(abs(sample_modes[mode] - exact_modes[mode]))
                for mode in exact_modes
            }
            energy = _energy_moments(labels, target, sample_labels)
            samples_match = (
                sample_labels == fallback_labels
                if fallback_labels is not None
                else arm_name == "fallback_exact"
            )
            row_pass = (
                metrics["total_variation_to_target"] <= TOLERANCES["target_tv"]
                and metrics["kl_target_to_empirical"] <= TOLERANCES["target_kl"]
                and energy["energy_mean_abs_error"] <= TOLERANCES["energy_mean_abs_error"]
                and energy["energy_variance_abs_error"] <= TOLERANCES["energy_variance_abs_error"]
                and max(mode_errors.values()) <= TOLERANCES["mode_mass_abs_error"]
                and (arm_name == "fallback_exact" or samples_match)
            )
            all_pass = all_pass and row_pass
            rows[fixture_name][arm_name] = {
                "support_valid": True,
                "quality_evaluated": True,
                "samples_match_fallback": bool(samples_match),
                "total_variation_to_target": metrics["total_variation_to_target"],
                "kl_target_to_empirical": metrics["kl_target_to_empirical"],
                "mode_masses_exact": {
                    mode: _stable_float(value) for mode, value in exact_modes.items()
                },
                "mode_masses_empirical": {
                    mode: _stable_float(value) for mode, value in sample_modes.items()
                },
                "mode_mass_abs_errors": mode_errors,
                "max_mode_mass_abs_error": max(mode_errors.values()),
                **energy,
                "error_pass": row_pass,
            }
    return {
        "fixtures": rows,
        "tolerances": dict(TOLERANCES),
        "all_supported_errors_within_tolerance": all_pass,
        "principle": FIELD_PRINCIPLES["observable_and_energy_error_by_fixture_arm"],
    }


def ess_and_autocorrelation_by_fixture_arm(
    measurements: Mapping[str, Mapping[str, JsonDict]],
) -> JsonDict:
    fixtures: dict[str, JsonDict] = {}
    all_pass = True
    for fixture_name, arms in measurements.items():
        fixtures[fixture_name] = {}
        for arm_name, run in arms.items():
            if not run["success"]:
                fixtures[fixture_name][arm_name] = {
                    "support_valid": False,
                    "mixing_evaluated": False,
                }
                continue
            metrics = run["result"]["metrics"]
            row_pass = (
                metrics["effective_sample_size"] >= TOLERANCES["ess_min"]
                and abs(metrics["lag1_autocorrelation"]) <= TOLERANCES["autocorrelation_abs_max"]
            )
            all_pass = all_pass and row_pass
            fixtures[fixture_name][arm_name] = {
                "support_valid": True,
                "mixing_evaluated": True,
                "acceptance_rate": metrics["acceptance_rate"],
                "accepted_count": metrics["accepted_count"],
                "attempted_count": metrics["attempted_count"],
                "lag1_autocorrelation": metrics["lag1_autocorrelation"],
                "integrated_autocorrelation_time": metrics["integrated_autocorrelation_time"],
                "effective_sample_size": metrics["effective_sample_size"],
                "mixing_pass": row_pass,
            }
    return {
        "fixtures": fixtures,
        "ess_min": TOLERANCES["ess_min"],
        "all_supported_mixing_pass": all_pass,
        "principle": FIELD_PRINCIPLES["ess_and_autocorrelation_by_fixture_arm"],
    }


def _transition_counts(decision_log: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        after = str(event["state_after"]["current_label"])
        counts[f"{before}->{after}"] += 1
    return dict(sorted(counts.items()))


def transition_and_mode_occupancy_counts(
    root: Path,
    measurements: Mapping[str, Mapping[str, JsonDict]],
) -> JsonDict:
    fixtures = _fixtures(root)
    rows: dict[str, JsonDict] = {}
    all_recorded = True
    for fixture_name, arms in measurements.items():
        rows[fixture_name] = {}
        fixture = fixtures[fixture_name]
        for arm_name, run in arms.items():
            if not run["success"]:
                rows[fixture_name][arm_name] = {"support_valid": False, "counts_recorded": False}
                continue
            result = run["result"]
            labels = [str(label) for label in result["sample_labels"]]
            mode_counts = {
                mode: sum(label in members for label in labels)
                for mode, members in fixture["mode_labels"].items()
            }
            rows[fixture_name][arm_name] = {
                "support_valid": True,
                "counts_recorded": True,
                "sample_count": len(labels),
                "per_label_counts": {
                    label: result["metrics"]["frequencies"][label]["count"]
                    for label in result["metrics"]["frequencies"]
                },
                "mode_occupancy_counts": mode_counts,
                "transition_counts": _transition_counts(result["decision_log"]),
                "accepted_count": result["metrics"]["accepted_count"],
                "attempted_count": result["metrics"]["attempted_count"],
            }
    return {
        "fixtures": rows,
        "all_supported_counts_recorded": all_recorded,
        "principle": FIELD_PRINCIPLES["transition_and_mode_occupancy_counts"],
    }


def serialization_and_restart_receipts(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=SEED,
        burn_in=1,
        enable_mode_jump_runtime=True,
    )
    descriptor["return_trace"] = True
    backend = ModeJumpRustBackend(seed=SEED)
    prefix = backend.run_descriptor(target, proposal, RESTART_PREFIX_COUNT, descriptor)
    checkpoint = backend.save_checkpoint()
    loaded_state = backend.load_checkpoint(checkpoint, target, proposal, config=descriptor)
    resumed_descriptor = {**descriptor, "checkpoint": checkpoint}
    runtime_suffix = ModeJumpRustBackend(seed=SEED).run_descriptor(
        target,
        proposal,
        RESTART_SUFFIX_COUNT,
        resumed_descriptor,
    )
    fallback_suffix = ModeJumpRustBackend(seed=SEED, prefer_rust=False).run_descriptor(
        target,
        proposal,
        RESTART_SUFFIX_COUNT,
        resumed_descriptor,
    )
    corrupt = json.loads(json.dumps(checkpoint))
    corrupt["state"]["step"] = int(corrupt["state"]["step"]) + 1
    corrupt_error = _expect_error(
        lambda: backend.load_checkpoint(corrupt, target, proposal, config=descriptor),
        ValueError,
    )
    pass_value = (
        loaded_state == checkpoint["state"]
        and checkpoint["payload_checksum"] == checkpoint_checksum(checkpoint)
        and runtime_suffix["sample_labels"] == fallback_suffix["sample_labels"]
        and runtime_suffix["checkpoint"]["state"] == fallback_suffix["checkpoint"]["state"]
        and corrupt_error["raised"] is True
    )
    return {
        "prefix_active_backend": prefix["receipt"]["active_backend"],
        "checkpoint_payload_checksum": checkpoint["payload_checksum"],
        "checkpoint_checksum_valid": checkpoint["payload_checksum"]
        == checkpoint_checksum(checkpoint),
        "loaded_state_matches": loaded_state == checkpoint["state"],
        "serialized_state": checkpoint["serialized_state"],
        "runtime_to_fallback_restart_samples_match": runtime_suffix["sample_labels"]
        == fallback_suffix["sample_labels"],
        "runtime_to_fallback_restart_state_match": runtime_suffix["checkpoint"]["state"]
        == fallback_suffix["checkpoint"]["state"],
        "malformed_state_rejection": corrupt_error,
        "python_rust_pyo3_consistency": {
            "runtime_active_backend": runtime_suffix["receipt"]["active_backend"],
            "fallback_active_backend": fallback_suffix["receipt"]["active_backend"],
            "decision_logs_match": runtime_suffix["decision_log"]
            == fallback_suffix["decision_log"],
        },
        "serialization_pass": pass_value,
        "principle": FIELD_PRINCIPLES["serialization_and_restart_receipts"],
    }


def fallback_trigger_and_exactness(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    base_descriptor = descriptor_for_run(
        labels=labels,
        seed=SEED,
        burn_in=4,
        enable_mode_jump_runtime=True,
    )
    baseline = ModeJumpRustBackend(seed=SEED, prefer_rust=False).run_descriptor(
        target,
        proposal,
        64,
        base_descriptor,
    )
    cases = (
        (
            "feature_flag_disabled",
            ModeJumpRustBackend(seed=SEED),
            target,
            proposal,
            {**base_descriptor, "enable_mode_jump_runtime": False},
            "feature_flag_disabled",
        ),
        (
            "declared_python_compatibility",
            ModeJumpRustBackend(seed=SEED),
            target,
            proposal,
            {**base_descriptor, "force_python_fallback": True},
            "declared_python_compatibility",
        ),
        (
            "unsupported_dtype_or_layout",
            ModeJumpRustBackend(seed=SEED),
            target.astype(np.float32),
            proposal.astype(np.float32),
            base_descriptor,
            "unsupported_dtype_or_layout",
        ),
        (
            "rust_extension_missing",
            ModeJumpRustBackend(
                seed=SEED,
                rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
            ),
            target,
            proposal,
            base_descriptor,
            "rust_extension_missing",
        ),
    )
    rows = []
    for name, backend, case_target, case_proposal, descriptor, expected_reason in cases:
        result = backend.run_descriptor(case_target, case_proposal, 64, descriptor)
        rows.append(
            {
                "case": name,
                "active_backend": result["receipt"]["active_backend"],
                "fallback_reason": result["receipt"]["fallback_reason"],
                "expected_reason_present": expected_reason
                in str(result["receipt"]["fallback_reason"]),
                "sample_labels_match": result["sample_labels"] == baseline["sample_labels"],
                "state_match": result["checkpoint"]["state"] == baseline["checkpoint"]["state"],
                "metrics_match": result["metrics"] == baseline["metrics"],
            }
        )
    return {
        "fallback_cases": rows,
        "all_fallbacks_exact": all(
            row["active_backend"] == ACTIVE_PYTHON_FALLBACK
            and row["expected_reason_present"]
            and row["sample_labels_match"]
            and row["state_match"]
            and row["metrics_match"]
            for row in rows
        ),
        "principle": FIELD_PRINCIPLES["fallback_trigger_and_exactness"],
    }


def _time_one_run(root: Path, arm_name: str) -> float:
    fixture = _fixtures(root)["multimodal_exp6194"]
    result = _run_arm(fixture, arm_name, return_trace=False)
    if not result["success"]:
        raise RuntimeError(result["message"])
    return float(result["elapsed_s"])


def _interval(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return [_stable_float(values[0]), _stable_float(values[0])]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    half_width = 1.96 * stdev / math.sqrt(len(values))
    return [_stable_float(mean - half_width), _stable_float(mean + half_width)]


def cpu_thread_and_wall_time_receipts(root: Path = REPO_ROOT) -> JsonDict:
    timings = {
        "fallback_exact": [
            _stable_float(_time_one_run(root, "fallback_exact")) for _ in range(TIMING_REPEATS)
        ],
        "mode_jump_runtime": [
            _stable_float(_time_one_run(root, "mode_jump_runtime")) for _ in range(TIMING_REPEATS)
        ],
    }
    intervals = {arm: _interval(values) for arm, values in timings.items()}
    fallback_interval = intervals["fallback_exact"]
    runtime_interval = intervals["mode_jump_runtime"]
    excludes = (
        fallback_interval[1] < runtime_interval[0] or runtime_interval[1] < fallback_interval[0]
    )
    return {
        "cpu": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "os_cpu_count": os.cpu_count(),
            "thread_env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
            },
        },
        "fixture": "multimodal_exp6194",
        "repeat_count": TIMING_REPEATS,
        "wall_time_seconds": {
            arm: {
                "values": values,
                "mean": _stable_float(statistics.mean(values)),
                "median": _stable_float(statistics.median(values)),
                "min": _stable_float(min(values)),
                "max": _stable_float(max(values)),
                "mean_95_interval": intervals[arm],
            }
            for arm, values in timings.items()
        },
        "uncertainty_excludes_parity": bool(excludes),
        "timing_is_diagnostic": True,
        "timing_claim_condition": "requires quality_gate_passed and all fixture arms supported",
        "principle": FIELD_PRINCIPLES["cpu_thread_and_wall_time_receipts"],
    }


def default_off_preserved(root: Path = REPO_ROOT) -> bool:
    return bool(exp6208.default_off_receipt(root)["default_off_pass"])


def upstream_exp6194_and_exp6208_paths_hashes(root: Path = REPO_ROOT) -> JsonDict:
    exp6194_artifact = _read_json(root / EXP6194_RESULT_RELATIVE_PATH)
    exp6208_artifact = _read_json(root / EXP6208_RESULT_RELATIVE_PATH)
    return {
        "paths": _path_hashes(
            root,
            (
                EXP6166_RESULT_RELATIVE_PATH,
                EXP6180_RESULT_RELATIVE_PATH,
                EXP6194_MODULE_RELATIVE_PATH,
                EXP6194_RESULT_RELATIVE_PATH,
                EXP6208_MODULE_RELATIVE_PATH,
                EXP6208_RESULT_RELATIVE_PATH,
            ),
        ),
        "exp6194_status": exp6194_artifact.get("status"),
        "exp6194_ready_score": exp6194_artifact.get("mode_jump_rust_pyo3_ready_score"),
        "exp6208_status": exp6208_artifact.get("status"),
        "exp6208_hardware_claimed": exp6208_artifact.get("hardware_or_speed_power_energy_claimed"),
        "principle": FIELD_PRINCIPLES["upstream_exp6194_and_exp6208_paths_hashes"],
    }


def backend_paths_build_and_abi_receipts(root: Path = REPO_ROOT) -> JsonDict:
    try:
        from carnot import _rust

        rust_import_available = True
        rust_import_error = None
        required_symbols_present = all(
            hasattr(_rust, name)
            for name in ("RustModeJumpConfig", "RustModeJumpCore", "RustModeJumpState")
        )
    except Exception as exc:  # pragma: no cover - this is a live ABI precondition receipt.
        rust_import_available = False
        rust_import_error = f"{type(exc).__name__}: {exc}"
        required_symbols_present = False
    return {
        "paths": _path_hashes(
            root,
            (
                BACKEND_RELATIVE_PATH,
                FACTORY_RELATIVE_PATH,
                RUST_KERNEL_RELATIVE_PATH,
                PYO3_BINDING_RELATIVE_PATH,
                MODULE_RELATIVE_PATH,
                TEST_RELATIVE_PATH,
                SAMPLER_SPEC_RELATIVE_PATH,
            ),
        ),
        "rust_import_available": rust_import_available,
        "rust_import_error": rust_import_error,
        "required_symbols_present": required_symbols_present,
        "abi_receipt": {
            "module": "carnot._rust",
            "symbols": ["RustModeJumpConfig", "RustModeJumpCore", "RustModeJumpState"],
            "feature_env_var": FEATURE_ENV_VAR,
        },
        "principle": FIELD_PRINCIPLES["backend_paths_build_and_abi_receipts"],
    }


def protected_files_unchanged(root: Path = REPO_ROOT) -> JsonDict:
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", *[path.as_posix() for path in PROTECTED_FILES]],
        cwd=root,
        check=False,
    )
    return {
        "paths": [path.as_posix() for path in PROTECTED_FILES],
        "unchanged": diff.returncode == 0,
        "git_diff_exit_code": diff.returncode,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "exact finite categorical target plus fixed Exp6194 transition receipts",
        "unsupported_fixture_handling": "fail_closed support oracle, not sampled quality evidence",
        "not_oracle_for": ["hardware speed", "power", "TSU behavior", "unseen targets"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def _expect_error(call: Any, error_type: type[BaseException]) -> JsonDict:
    try:
        call()
    except error_type as exc:
        return {"raised": True, "error": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "error": None, "message": None}


def _preexisting_nonzero_from_exp6208(root: Path) -> list[JsonDict]:
    exp6208_artifact = _read_json(root / EXP6208_RESULT_RELATIVE_PATH)
    return [
        {**dict(row), "source": EXP6208_RESULT_RELATIVE_PATH.as_posix()}
        for row in exp6208_artifact.get("unrelated_nonzero_command_classifications", [])
    ]


def task_owned_and_preexisting_test_classification(
    root: Path,
    command_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    receipts = [dict(row) for row in command_receipts]
    task_owned = [row for row in receipts if bool(row.get("task_owned", True)) is True]
    task_owned_failures = [row for row in task_owned if int(row.get("exit_code", -999)) != 0]
    current_preexisting = [
        row
        for row in receipts
        if bool(row.get("task_owned", True)) is False and int(row.get("exit_code", -999)) != 0
    ]
    preexisting = [*_preexisting_nonzero_from_exp6208(root), *current_preexisting]
    return {
        "command_receipts": receipts,
        "task_owned_commands": task_owned,
        "task_owned_failures": task_owned_failures,
        "task_owned_failure_count": len(task_owned_failures),
        "all_task_owned_commands_passed": len(task_owned_failures) == 0,
        "preexisting_repository_wide_nonzero_commands": preexisting,
        "preexisting_nonzero_count": len(preexisting),
        "principle": FIELD_PRINCIPLES["task_owned_and_preexisting_test_classification"],
    }


def field_provenance() -> JsonDict:
    sources = {
        "status": "computed readiness gates",
        "upstream_exp6194_and_exp6208_paths_hashes": "upstream artifacts and file hashes",
        "backend_paths_build_and_abi_receipts": "source hashes and carnot._rust import check",
        "preregistered_fixture_seed_schedule_matrix": "frozen fixture matrix in this module",
        "matched_arm_configuration": "Exp6208 descriptor and fixed target/proposal",
        "support_validity_by_fixture_arm": "matched adapter run receipts",
        "observable_and_energy_error_by_fixture_arm": "exact finite target and sample labels",
        "ess_and_autocorrelation_by_fixture_arm": "Exp6208 adapter metrics",
        "transition_and_mode_occupancy_counts": "decision logs and sample labels",
        "serialization_and_restart_receipts": "checkpoint and restart probes",
        "fallback_trigger_and_exactness": "fallback control probes",
        "cpu_thread_and_wall_time_receipts": "local CPU context and perf_counter timings",
        "quality_gate_passed": "support and quality gate computation",
        "timing_claim_allowed": "preregistered timing gate computation",
        "default_off_preserved": "Exp6208 default-off receipt",
        "fpga_tsu_power_hardware_claim_count": "prompt and spec invariant",
        "task_owned_and_preexisting_test_classification": "command receipts and Exp6208 classifications",
        "sampler_runtime_ready_score": "computed readiness score",
        "protected_files_unchanged": "git diff over protected paths",
        "inference_substrate": "prompt and spec invariant",
        "verifier_is_oracle": "exact finite target and fixed transition",
        "field_provenance": "this provenance map",
        "field_principles": "OpenSpec required field principles",
        "test_commands": "command receipts",
        "test_exit_codes": "command receipts",
        "duration_s": "wall-clock measurement",
        "reproducibility_checksum": "deterministic artifact hash",
        "honest_verdict": "computed verdict",
    }
    return {
        field: {"source": sources[field], "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_commands(command_receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("command", "")) for row in command_receipts]


def _test_exit_codes(command_receipts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(row.get("command", "")): int(row.get("exit_code", -999)) for row in command_receipts
    }


def quality_gate_passed(artifact: Mapping[str, Any]) -> bool:
    support = dict(artifact.get("support_validity_by_fixture_arm") or {})
    errors = dict(artifact.get("observable_and_energy_error_by_fixture_arm") or {})
    mixing = dict(artifact.get("ess_and_autocorrelation_by_fixture_arm") or {})
    counts = dict(artifact.get("transition_and_mode_occupancy_counts") or {})
    return bool(
        support.get("all_required_fixture_arms_supported") is True
        and errors.get("all_supported_errors_within_tolerance") is True
        and mixing.get("all_supported_mixing_pass") is True
        and counts.get("all_supported_counts_recorded") is True
    )


def timing_claim_allowed(artifact: Mapping[str, Any]) -> bool:
    timing = dict(artifact.get("cpu_thread_and_wall_time_receipts") or {})
    return bool(quality_gate_passed(artifact) and timing.get("uncertainty_excludes_parity") is True)


def sampler_runtime_ready_score(artifact: Mapping[str, Any]) -> float:
    classification = dict(artifact.get("task_owned_and_preexisting_test_classification") or {})
    core = [
        artifact.get("default_off_preserved") is True,
        artifact.get("fpga_tsu_power_hardware_claim_count") == 0,
        dict(artifact.get("observable_and_energy_error_by_fixture_arm") or {}).get(
            "all_supported_errors_within_tolerance"
        )
        is True,
        dict(artifact.get("ess_and_autocorrelation_by_fixture_arm") or {}).get(
            "all_supported_mixing_pass"
        )
        is True,
        dict(artifact.get("serialization_and_restart_receipts") or {}).get("serialization_pass")
        is True,
        dict(artifact.get("fallback_trigger_and_exactness") or {}).get("all_fallbacks_exact")
        is True,
        classification.get("all_task_owned_commands_passed") is True,
        dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True,
    ]
    support_ready = (
        dict(artifact.get("support_validity_by_fixture_arm") or {}).get(
            "all_required_fixture_arms_supported"
        )
        is True
    )
    if all(core) and support_ready and artifact.get("timing_claim_allowed") is True:
        return 1.0
    if all(core):
        return 0.75
    return 0.0


def status(artifact: Mapping[str, Any]) -> str:
    classification = dict(artifact.get("task_owned_and_preexisting_test_classification") or {})
    if classification.get("task_owned_failure_count", 0) > 0:
        return "blocked"
    if sampler_runtime_ready_score(artifact) == 1.0:
        return "complete_ready"
    if sampler_runtime_ready_score(artifact) > 0.0:
        return "complete_partial"
    return "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    support = dict(artifact.get("support_validity_by_fixture_arm") or {})
    classification = dict(artifact.get("task_owned_and_preexisting_test_classification") or {})
    task_failures = classification.get("task_owned_failures", [])
    preexisting = classification.get("preexisting_repository_wide_nonzero_commands", [])
    task_text = "none" if not task_failures else json.dumps(task_failures, sort_keys=True)
    preexisting_text = "none" if not preexisting else json.dumps(preexisting, sort_keys=True)
    return (
        f"{status(artifact)}: default_off={artifact.get('default_off_preserved') is True}; "
        f"all_fixture_arms_supported={support.get('all_required_fixture_arms_supported') is True}; "
        f"quality_gate={artifact.get('quality_gate_passed') is True}; "
        f"timing_claim_allowed={artifact.get('timing_claim_allowed') is True}; "
        f"hardware_claim_count={artifact.get('fpga_tsu_power_hardware_claim_count')}; "
        f"task_owned_failures={task_text}; preexisting_nonzero={preexisting_text}"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> JsonDict:
    measurements = _measure_fixture_arms(root)
    artifact: JsonDict = {
        "status": "blocked",
        "upstream_exp6194_and_exp6208_paths_hashes": upstream_exp6194_and_exp6208_paths_hashes(
            root
        ),
        "backend_paths_build_and_abi_receipts": backend_paths_build_and_abi_receipts(root),
        "preregistered_fixture_seed_schedule_matrix": preregistered_fixture_seed_schedule_matrix(
            root
        ),
        "matched_arm_configuration": matched_arm_configuration(root),
        "support_validity_by_fixture_arm": support_validity_by_fixture_arm(measurements),
        "observable_and_energy_error_by_fixture_arm": observable_and_energy_error_by_fixture_arm(
            root, measurements
        ),
        "ess_and_autocorrelation_by_fixture_arm": ess_and_autocorrelation_by_fixture_arm(
            measurements
        ),
        "transition_and_mode_occupancy_counts": transition_and_mode_occupancy_counts(
            root, measurements
        ),
        "serialization_and_restart_receipts": serialization_and_restart_receipts(root),
        "fallback_trigger_and_exactness": fallback_trigger_and_exactness(root),
        "cpu_thread_and_wall_time_receipts": cpu_thread_and_wall_time_receipts(root),
        "quality_gate_passed": False,
        "timing_claim_allowed": False,
        "default_off_preserved": default_off_preserved(root),
        "fpga_tsu_power_hardware_claim_count": 0,
        "task_owned_and_preexisting_test_classification": task_owned_and_preexisting_test_classification(
            root, command_receipts
        ),
        "sampler_runtime_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": _test_commands(command_receipts),
        "test_exit_codes": _test_exit_codes(command_receipts),
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: pending",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": SCHEMA,
    }
    artifact["quality_gate_passed"] = quality_gate_passed(artifact)
    artifact["timing_claim_allowed"] = timing_claim_allowed(artifact)
    artifact["sampler_runtime_ready_score"] = sampler_runtime_ready_score(artifact)
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
    if artifact.get("fpga_tsu_power_hardware_claim_count") != 0:
        raise ValueError("fpga_tsu_power_hardware_claim_count")
    if artifact.get("default_off_preserved") is not True:
        raise ValueError("default_off_preserved")
    if artifact.get("timing_claim_allowed") != timing_claim_allowed(artifact):
        raise ValueError("timing_claim_allowed")
    if artifact.get("quality_gate_passed") != quality_gate_passed(artifact):
        raise ValueError("quality_gate_passed")
    if artifact.get("sampler_runtime_ready_score") != sampler_runtime_ready_score(artifact):
        raise ValueError("sampler_runtime_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field in REQUIRED_ARTIFACT_FIELDS:
        row = provenance.get(field)
        if not isinstance(row, Mapping) or not row.get("source") or not row.get("principle"):
            raise ValueError("field_provenance")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _run_command(command: str, root: Path) -> JsonDict:  # pragma: no cover - live receipt path.
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=root,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": command.split()[0],
        "command": command,
        "exit_code": result.returncode,
        "duration_s": _stable_float(time.perf_counter() - started),
        "stdout_tail": result.stdout.strip()[-2000:],
        "stderr_tail": result.stderr.strip()[-2000:],
        "task_owned": True,
        "classification": "task_owned",
    }


def _external_command_receipts() -> list[JsonDict] | None:
    receipt_path = Path(os.environ.get("CARNOT_6220_COMMAND_RECEIPTS", DEFAULT_RECEIPT_PATH))
    if not receipt_path.exists():
        return None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("command receipt payload must be a list")
    return [dict(row) for row in payload]


def _run_default_task_commands(
    root: Path,
) -> list[JsonDict]:  # pragma: no cover - live receipt path.
    commands = (
        ".venv/bin/pytest tests/python/test_experiment_6220_mode_jump_runtime_ab.py -q -o addopts=",
        ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6220_mode_jump_runtime_ab.py -m pytest tests/python/test_experiment_6220_mode_jump_runtime_ab.py -q --no-cov -o addopts=",
        ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6220_mode_jump_runtime_ab.py --fail-under=100",
        ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6220_mode_jump_runtime_ab.py",
        ".venv/bin/pytest tests/python/test_pyo3_integration.py tests/python/test_e2e_serialization.py -q -o addopts=",
        ".venv/bin/pytest tests/python -q",
    )
    return [_run_command(command, root) for command in commands]


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - covered by CLI smoke patch.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    command_receipts = _external_command_receipts()
    if command_receipts is None:
        command_receipts = _run_default_task_commands(REPO_ROOT)
    artifact = write_artifact(
        output_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        root=REPO_ROOT,
        command_receipts=command_receipts,
        duration_s=time.perf_counter() - started,
        run_date=args.date,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
