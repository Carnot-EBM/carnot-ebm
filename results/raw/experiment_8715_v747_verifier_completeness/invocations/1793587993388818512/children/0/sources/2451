"""Scale a reusable spectral spin-block sampler and check Rust parity.

The treatment partitions spin variables from the coupling graph. Each selected
block uses an exact conditional heat-bath draw. Exact enumeration evaluates
`n=16`; separate long chains evaluate `n=32`. Neither evaluator generates a
treatment transition. The result supports CPU software claims only.

Spec refs: REQ-SAMPLER-6612,
SCENARIO-SAMPLER-6612-INDEPENDENT-SCALE-EVIDENCE,
SCENARIO-SAMPLER-6612-RUST-PARITY-AND-FAIL-CLOSED-VERDICT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from carnot.samplers.spectral_k_block import (
    BlockChainResult,
    SpinPartition,
    build_random_blocks,
    build_spectral_blocks,
    ising_energy,
    run_python_chain,
    run_rust_chain,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6612_spectral_k_block_scale_rust_parity.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6612_spectral_k_block_scale_rust_parity.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py")
SAMPLER_RELATIVE_PATH = Path("python/carnot/samplers/spectral_k_block.py")
SAMPLER_TEST_RELATIVE_PATH = Path("tests/python/samplers/test_spectral_k_block.py")
RUST_RELATIVE_PATH = Path("crates/carnot-samplers/src/spectral_k_block.rs")
RUST_TEST_RELATIVE_PATH = Path("crates/carnot-samplers/tests/spectral_k_block.rs")
PYO3_RELATIVE_PATH = Path("crates/carnot-python/src/spectral_k_block.rs")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
RUST_SPEC_RELATIVE_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")

RUN_DATE = "20260825"
EXPERIMENT_ID = "experiment_6612_spectral_k_block_scale_rust_parity"
SCHEMA_VERSION = "carnot.experiment_6612.spectral_k_block_scale_rust_parity.v1"
INFERENCE_SUBSTRATE = "cpu_python_rust_frustrated_spectral_k_block_sampling_no_llm"
ARMS = (
    "sequential_gibbs",
    "random_k_block",
    "spectral_k_block_python",
    "spectral_k_block_rust",
)
ALLOWED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
PARITY_SAMPLE_MISMATCH_TOLERANCE = 0.0
PARITY_MOMENT_TOLERANCE = 1.0e-12
STATIONARY_NONINFERIORITY_MARGIN = 0.05
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen chain, fixture, and reference budgets for one artifact."""

    treatment_seeds: tuple[int, ...] = (6612, 6613, 6614, 6615, 6616)
    burn_in: int = 1_000
    retained_samples: int = 10_000
    reference_seeds: tuple[int, ...] = (16612, 26612, 36612)
    reference_burn_in: int = 3_000
    reference_retained_samples: int = 20_000
    fixtures_per_size: int = 6
    block_size: int = 4


@dataclass(frozen=True)
class FrustratedFixture:
    """One immutable mixed-sign Ising system with a guaranteed odd cycle."""

    fixture_id: str
    n_spins: int
    generation_seed: int
    couplings: np.ndarray = field(compare=False, repr=False)
    fields: np.ndarray = field(compare=False, repr=False)
    temperature: float
    fixture_sha256: str
    has_non_bipartite_cycle: bool = True
    competing_modes: bool = True


@dataclass(frozen=True)
class _ReferenceStats:
    receipt: JsonDict = field(compare=False)
    energy_mean: float
    moment_mean: np.ndarray = field(compare=False, repr=False)
    energy_interval: tuple[float, float]
    moment_half_width_max: float
    exact_probabilities: np.ndarray | None = field(compare=False, repr=False)


DEFAULT_TEST_COMMANDS = (
    "cargo test -p carnot-samplers --test spectral_k_block --quiet",
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p carnot-python spectral_k_block --quiet",
    ".venv/bin/pytest tests/python/samplers/test_spectral_k_block.py tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=*/python/carnot/samplers/spectral_k_block.py,*/python/carnot/experiment_6612_spectral_k_block_scale_rust_parity.py -m pytest tests/python/samplers/test_spectral_k_block.py tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=*/python/carnot/samplers/spectral_k_block.py,*/python/carnot/experiment_6612_spectral_k_block_scale_rust_parity.py --fail-under=100",
    "rustfmt --edition 2021 --check crates/carnot-samplers/src/spectral_k_block.rs crates/carnot-samplers/tests/spectral_k_block.rs crates/carnot-python/src/spectral_k_block.rs",
    "cargo clippy -p carnot-samplers --test spectral_k_block -- -D warnings -A clippy::type-complexity -A clippy::too-many-arguments -A clippy::needless-range-loop",
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check -p carnot-python",
    ".venv/bin/ruff check python/carnot/samplers/spectral_k_block.py python/carnot/experiment_6612_spectral_k_block_scale_rust_parity.py tests/python/samplers/test_spectral_k_block.py tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py",
    ".venv/bin/ruff format --check python/carnot/samplers/spectral_k_block.py python/carnot/experiment_6612_spectral_k_block_scale_rust_parity.py tests/python/samplers/test_spectral_k_block.py tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/samplers/test_spectral_k_block.py tests/python/test_experiment_6612_spectral_k_block_scale_rust_parity.py",
    ".venv/bin/pytest tests/python/test_e2e_training_sampling.py tests/python/test_e2e_serialization.py -q -o addopts=",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6612_spectral_k_block_scale_rust_parity.json",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6612_spectral_k_block_scale_rust_parity.json",
)


def _test_receipt_blockers(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Name every missing, failed, or malformed mandatory verification receipt."""

    by_command = {str(row.get("command")): row for row in receipts}
    blockers: list[JsonDict] = []
    for command in DEFAULT_TEST_COMMANDS:
        row = by_command.get(command)
        if row is None or row.get("exit_code") != 0 or float(row.get("duration_s", -1.0)) < 0.0:
            blockers.append(
                {
                    "gate": "toolchain_or_test_failure",
                    "command": command,
                    "exit_code": None if row is None else row.get("exit_code"),
                }
            )
    return blockers


REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "fixture_and_reference_receipts",
    "sampler_implementation_receipts",
    "partition_rows",
    "stationary_quality_summary",
    "efficiency_summary",
    "rust_python_parity_rows",
    "spectral_scale_ready_score",
    "hardware_path_receipt",
    "claim_boundaries",
    "attack_rows",
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
    "status": "The scale task ends with complete software evidence or a named reference or parity block.",
    "honest_verdict": "The verdict separates stationary quality, transition efficiency, wall efficiency, and parity.",
    "verdict_class": "Use the closed enum; positive is allowed only for independent held software evidence.",
    "gate_check_summary": "Any block names the failed fixture, reference, chain, parity, cost, toolchain, or protection value.",
    "per_unit_rows": "Every size, fixture, seed, and arm carries stationary, ESS, transition, wall, parity, and failure metrics.",
    "fixture_and_reference_receipts": "Couplings, seeds, hashes, target statistics, uncertainty, and reference independence are explicit.",
    "sampler_implementation_receipts": "Reusable Python and Rust paths bind code hashes and sampler-interface identity.",
    "partition_rows": "Spectral and random partitions, setup cost, block balance, and hashes remain inspectable.",
    "stationary_quality_summary": "Energy and moment errors and noninferiority recompute from per-unit rows.",
    "efficiency_summary": "ESS per transition, ESS per wall second, setup, sampling, and total time remain separate.",
    "rust_python_parity_rows": "Matched seeds and fixtures expose distribution, moment, and cost deltas within fixed tolerance.",
    "spectral_scale_ready_score": "This exact binary field reports complete reference, comparison, parity, and cost replay.",
    "hardware_path_receipt": "Arithmetic, memory, RNG, and parallelism needs are stated without claiming attached hardware execution.",
    "claim_boundaries": "The task is distinct from retired homotopy argmin, PIMI, FPGA, TSU, and general hardware performance.",
    "attack_rows": "Scope, reference, burn-in, charge, setup, identity, RNG, parity, hardware, and mutation attacks fail closed.",
    "preconditions_checked": "Prior result, toolchains, resources, fixtures, references, chains, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain original hashes.",
    "inference_substrate": "The task declares CPU Python and Rust Ising sampling with no LLM and no hardware board.",
    "verifier_is_oracle": "Independent target statistics evaluate sampling but do not generate treatment transitions.",
    "field_provenance": "Every field names fixtures, seeds, chain rows, code hashes, timers, and reducers.",
    "duration_s": "Monotonic duration covers all retained chains and parity runs.",
    "tests_run": "Named Rust, Python, lint, spec, artifact, adversarial, and E2E commands include exits and durations.",
    "reproducibility_checksum": "A final hash protects the software result.",
}


def canonical_json(value: Any) -> str:
    """Serialize one evidence value with stable keys and separators."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fixture_hash(
    fixture_id: str,
    generation_seed: int,
    couplings: np.ndarray,
    fields: np.ndarray,
    temperature: float,
) -> str:
    return sha256_json(
        {
            "fixture_id": fixture_id,
            "generation_seed": generation_seed,
            "couplings": couplings.tolist(),
            "fields": fields.tolist(),
            "temperature": temperature,
        }
    )


def frozen_frustrated_fixtures() -> tuple[FrustratedFixture, ...]:
    """Generate the preregistered matrices from fixed, inspectable seeds."""

    fixtures: list[FrustratedFixture] = []
    for n_spins in (16, 32):
        for fixture_index in range(6):
            generation_seed = 661_200 + 100 * n_spins + fixture_index
            rng = np.random.default_rng(generation_seed)
            couplings = np.zeros((n_spins, n_spins), dtype=np.float64)
            for left in range(n_spins):
                right = (left + 1) % n_spins
                sign = -1.0 if (left + fixture_index) % 3 == 0 else 1.0
                couplings[left, right] = couplings[right, left] = sign * float(
                    rng.uniform(0.35, 0.8)
                )
            couplings[0, 1] = couplings[1, 0] = float(rng.uniform(0.65, 0.9))
            couplings[1, 2] = couplings[2, 1] = float(rng.uniform(0.65, 0.9))
            couplings[0, 2] = couplings[2, 0] = -float(rng.uniform(0.65, 0.9))
            for chord in range(n_spins):
                left = int(rng.integers(0, n_spins))
                right = (left + 3 + 2 * chord) % n_spins
                if left == right:  # pragma: no cover - defensive for future fixture sizes.
                    right = (right + 1) % n_spins
                sign = -1.0 if chord % 2 else 1.0
                weight = sign * float(rng.uniform(0.15, 0.55))
                couplings[left, right] = couplings[right, left] = weight
            np.fill_diagonal(couplings, 0.0)
            fields = np.zeros(n_spins, dtype=np.float64)
            temperature = 0.8 + 0.08 * (fixture_index % 3)
            fixture_id = f"frustrated_mixed_modes_n{n_spins}_f{fixture_index}"
            fixtures.append(
                FrustratedFixture(
                    fixture_id=fixture_id,
                    n_spins=n_spins,
                    generation_seed=generation_seed,
                    couplings=couplings,
                    fields=fields,
                    temperature=temperature,
                    fixture_sha256=_fixture_hash(
                        fixture_id,
                        generation_seed,
                        couplings,
                        fields,
                        temperature,
                    ),
                )
            )
    return tuple(fixtures)


def _selected_fixtures(config: ExperimentConfig) -> tuple[FrustratedFixture, ...]:
    if not 1 <= config.fixtures_per_size <= 6:
        raise ValueError("fixtures_per_size must be in [1, 6]")
    return tuple(
        fixture
        for fixture in frozen_frustrated_fixtures()
        if int(fixture.fixture_id.rsplit("f", 1)[1]) < config.fixtures_per_size
    )


def _enumerate_states(n_spins: int) -> np.ndarray:
    indices = np.arange(1 << int(n_spins), dtype=np.uint64)[:, None]
    bits = (indices >> np.arange(int(n_spins), dtype=np.uint64)) & 1
    return (2 * bits.astype(np.int8) - 1).astype(np.int8)


def _energies(samples: np.ndarray, fixture: FrustratedFixture) -> np.ndarray:
    spins = np.asarray(samples, dtype=np.float64)
    return (
        -0.5 * np.einsum("bi,ij,bj->b", spins, fixture.couplings, spins, optimize=True)
        - spins @ fixture.fields
    )


def _interval(values: Sequence[float]) -> JsonDict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "sample_size": 1}
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(array), 1.96)
    half = critical * float(np.std(array, ddof=1)) / sqrt(len(array))
    return {
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
        "sample_size": int(len(array)),
    }


def _initial_state(fixture: FrustratedFixture, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{fixture.fixture_sha256}:{seed}:initial".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.where(rng.integers(0, 2, size=fixture.n_spins) == 1, 1, -1).astype(np.int8)


def _domain_seed(fixture: FrustratedFixture, seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{fixture.fixture_sha256}:{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _exact_reference(fixture: FrustratedFixture) -> _ReferenceStats:
    states = _enumerate_states(fixture.n_spins)
    energies = _energies(states, fixture)
    log_weights = -energies / fixture.temperature
    weights = np.exp(log_weights - float(np.max(log_weights)))
    probabilities = weights / float(np.sum(weights))
    energy_mean = float(probabilities @ energies)
    moment_mean = probabilities @ states.astype(np.float64)
    receipt = {
        "fixture_id": fixture.fixture_id,
        "n_spins": fixture.n_spins,
        "reference_method": "exact_enumeration",
        "reference_seeds": [],
        "state_count": int(len(states)),
        "target_energy_mean": energy_mean,
        "target_moment_mean": moment_mean.tolist(),
        "energy_uncertainty_interval": [energy_mean, energy_mean],
        "moment_uncertainty_half_width_max": 0.0,
        "reference_sample_hashes": [],
        "target_statistics_sha256": sha256_json(
            {"energy": energy_mean, "moment": moment_mean.tolist()}
        ),
        "independent_of_treatment": True,
        "treatment_samples_used": False,
        "role": "post_sampling_evaluator_only",
    }
    return _ReferenceStats(
        receipt=receipt,
        energy_mean=energy_mean,
        moment_mean=moment_mean,
        energy_interval=(energy_mean, energy_mean),
        moment_half_width_max=0.0,
        exact_probabilities=probabilities,
    )


def _long_chain_reference(fixture: FrustratedFixture, config: ExperimentConfig) -> _ReferenceStats:
    blocks = tuple((index,) for index in range(fixture.n_spins))
    energy_means: list[float] = []
    moment_means: list[np.ndarray] = []
    sample_hashes: list[str] = []
    rows: list[JsonDict] = []
    for seed in config.reference_seeds:
        run = run_python_chain(
            fixture.couplings,
            fixture.fields,
            fixture.temperature,
            blocks,
            _initial_state(fixture, seed),
            seed=_domain_seed(fixture, seed, "independent_reference"),
            burn_in=config.reference_burn_in,
            retained_samples=config.reference_retained_samples,
        )
        chain_energies = _energies(run.samples, fixture)
        energy_mean = float(np.mean(chain_energies))
        moment_mean = np.mean(run.samples.astype(np.float64), axis=0)
        energy_means.append(energy_mean)
        moment_means.append(moment_mean)
        sample_hashes.append(run.sample_sha256)
        rows.append(
            {
                "seed": seed,
                "domain_seed": run.rng_initial_state,
                "burn_in": config.reference_burn_in,
                "retained_samples": config.reference_retained_samples,
                "transitions": run.transitions,
                "sample_sha256": run.sample_sha256,
                "energy_mean": energy_mean,
                "moment_mean": moment_mean.tolist(),
            }
        )
    energy_interval = _interval(energy_means)
    moment_matrix = np.stack(moment_means)
    moment_intervals = [_interval(moment_matrix[:, index]) for index in range(fixture.n_spins)]
    moment_mean = np.mean(moment_matrix, axis=0)
    half_width = max(
        max(
            abs(float(interval["upper"]) - float(interval["mean"])),
            abs(float(interval["mean"]) - float(interval["lower"])),
        )
        for interval in moment_intervals
    )
    receipt = {
        "fixture_id": fixture.fixture_id,
        "n_spins": fixture.n_spins,
        "reference_method": "independent_long_chains",
        "reference_seeds": list(config.reference_seeds),
        "reference_burn_in": config.reference_burn_in,
        "reference_retained_samples_per_chain": config.reference_retained_samples,
        "reference_chain_rows": rows,
        "target_energy_mean": float(energy_interval["mean"]),
        "target_moment_mean": moment_mean.tolist(),
        "energy_uncertainty_interval": [
            float(energy_interval["lower"]),
            float(energy_interval["upper"]),
        ],
        "moment_uncertainty_intervals": moment_intervals,
        "moment_uncertainty_half_width_max": half_width,
        "reference_sample_hashes": sample_hashes,
        "target_statistics_sha256": sha256_json(
            {"energy": energy_interval, "moments": moment_intervals}
        ),
        "independent_of_treatment": True,
        "treatment_samples_used": False,
        "role": "post_sampling_evaluator_only",
    }
    return _ReferenceStats(
        receipt=receipt,
        energy_mean=float(energy_interval["mean"]),
        moment_mean=moment_mean,
        energy_interval=(
            float(energy_interval["lower"]),
            float(energy_interval["upper"]),
        ),
        moment_half_width_max=half_width,
        exact_probabilities=None,
    )


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate autocorrelation time with positive paired FFT sums."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 1.0
    centered = array - float(np.mean(array))
    variance_sum = float(centered @ centered)
    if variance_sum <= 1.0e-18:
        return 1.0
    fft_size = 1 << (2 * array.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[: array.size]
    covariance /= np.arange(array.size, 0, -1, dtype=np.float64)
    correlations = covariance / covariance[0]
    positive_sum = 0.0
    for lag in range(1, len(correlations) - 1, 2):
        pair = float(correlations[lag] + correlations[lag + 1])
        if pair <= 0.0:
            break
        positive_sum += pair
    return max(1.0, min(float(array.size), 1.0 + 2.0 * positive_sum))


def _sample_indices(samples: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(samples.shape[1], dtype=np.uint64))[None, :]
    return np.sum((samples > 0).astype(np.uint64) * powers, axis=1).astype(np.int64)


def _evaluate_chain(
    fixture: FrustratedFixture,
    reference: _ReferenceStats,
    run: BlockChainResult,
) -> JsonDict:
    energies = _energies(run.samples, fixture)
    moment = np.mean(run.samples.astype(np.float64), axis=0)
    observables = [
        energies,
        np.sum(run.samples.astype(np.float64), axis=1),
        *(run.samples[:, index] for index in range(fixture.n_spins)),
    ]
    tau = max(integrated_autocorrelation_time(values) for values in observables)
    ess = float(len(run.samples) / tau)
    energy_error = abs(float(np.mean(energies)) - reference.energy_mean)
    moment_error = np.abs(moment - reference.moment_mean)
    distribution_error: float | None = None
    if reference.exact_probabilities is not None:
        counts = np.bincount(
            _sample_indices(run.samples), minlength=len(reference.exact_probabilities)
        )
        empirical = counts.astype(np.float64) / len(run.samples)
        distribution_error = float(0.5 * np.sum(np.abs(empirical - reference.exact_probabilities)))
    return {
        "energy_mean": float(np.mean(energies)),
        "reference_energy_mean": reference.energy_mean,
        "energy_error_abs": energy_error,
        "moment_mean": moment.tolist(),
        "reference_moment_mean": reference.moment_mean.tolist(),
        "moment_error_linf": float(np.max(moment_error)),
        "moment_error_l2": float(np.linalg.norm(moment_error)),
        "total_variation_error": distribution_error,
        "stationary_error_score": float(energy_error / fixture.n_spins + np.max(moment_error)),
        "integrated_autocorrelation_time": tau,
        "effective_sample_size": ess,
    }


def _partition_row(fixture: FrustratedFixture, partition: SpinPartition) -> JsonDict:
    sizes = list(partition.block_sizes)
    return {
        "fixture_id": fixture.fixture_id,
        "n_spins": fixture.n_spins,
        "partition_kind": partition.kind,
        "blocks": [list(block) for block in partition.blocks],
        "block_count": len(partition.blocks),
        "block_sizes": sizes,
        "block_balance_max_minus_min": max(sizes) - min(sizes),
        "setup_time_s": partition.setup_time_s,
        "partition_seed": partition.seed,
        "partition_sha256": partition.sha256,
        "source": partition.source,
        "failure": None,
    }


def _run_arm(
    fixture: FrustratedFixture,
    seed: int,
    arm: str,
    partition: SpinPartition | None,
    config: ExperimentConfig,
) -> BlockChainResult:
    if arm == "sequential_gibbs":
        blocks = tuple((index,) for index in range(fixture.n_spins))
        domain = "sequential_control"
        engine = "python"
    elif arm == "random_k_block":
        if partition is None:  # pragma: no cover - guarded by the experiment matrix.
            raise ValueError("random partition missing")
        blocks = partition.blocks
        domain = "random_control"
        engine = "python"
    else:
        if partition is None:  # pragma: no cover - guarded by the experiment matrix.
            raise ValueError("spectral partition missing")
        blocks = partition.blocks
        domain = "spectral_matched_parity"
        engine = "rust" if arm.endswith("rust") else "python"
    kwargs = {
        "seed": _domain_seed(fixture, seed, domain),
        "burn_in": config.burn_in,
        "retained_samples": config.retained_samples,
    }
    runner = run_rust_chain if engine == "rust" else run_python_chain
    return runner(
        fixture.couplings,
        fixture.fields,
        fixture.temperature,
        blocks,
        _initial_state(fixture, seed),
        **kwargs,
    )


def _failure_row(
    fixture: FrustratedFixture,
    seed: int,
    arm: str,
    config: ExperimentConfig,
    error: Exception,
) -> JsonDict:
    return {
        "row_id": f"{fixture.fixture_id}:seed{seed}:{arm}",
        "fixture_id": fixture.fixture_id,
        "n_spins": fixture.n_spins,
        "seed": seed,
        "arm": arm,
        "burn_in": config.burn_in,
        "retained_sample_count": 0,
        "transitions": 0,
        "spins_updated": 0,
        "energy_mean": None,
        "reference_energy_mean": None,
        "energy_error_abs": None,
        "moment_mean": None,
        "reference_moment_mean": None,
        "moment_error_linf": None,
        "moment_error_l2": None,
        "total_variation_error": None,
        "stationary_error_score": None,
        "integrated_autocorrelation_time": None,
        "effective_sample_size": None,
        "ess_per_transition": None,
        "ess_per_wall_second": None,
        "partition_kind": None,
        "partition_sha256": None,
        "setup_time_s": 0.0,
        "sampling_time_s": 0.0,
        "total_time_s": 0.0,
        "rng_algorithm": None,
        "rng_domain": None,
        "rng_initial_state": None,
        "rng_final_state": None,
        "initial_state_sha256": sha256_bytes(_initial_state(fixture, seed).tobytes()),
        "sample_sha256": None,
        "parity_sample_mismatch_fraction": None,
        "parity_energy_mean_delta": None,
        "parity_moment_linf_delta": None,
        "failure": f"{type(error).__name__}:{error}",
    }


def _paired_summaries(per_unit_rows: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    successful = [row for row in per_unit_rows if row["failure"] is None]
    baseline = {
        (row["fixture_id"], row["seed"]): row
        for row in successful
        if row["arm"] == "sequential_gibbs"
    }
    stationary_rows: list[JsonDict] = []
    efficiency_rows: list[JsonDict] = []
    for arm in ARMS[1:]:
        candidate = {
            (row["fixture_id"], row["seed"]): row for row in successful if row["arm"] == arm
        }
        keys = sorted(set(baseline) & set(candidate))
        stationary_effects = [
            float(candidate[key]["stationary_error_score"])
            - float(baseline[key]["stationary_error_score"])
            for key in keys
        ]
        transition_effects = [
            float(candidate[key]["ess_per_transition"])
            / max(float(baseline[key]["ess_per_transition"]), 1.0e-18)
            - 1.0
            for key in keys
        ]
        wall_effects = [
            float(candidate[key]["ess_per_wall_second"])
            / max(float(baseline[key]["ess_per_wall_second"]), 1.0e-18)
            - 1.0
            for key in keys
        ]
        charged_time_effects = [
            float(baseline[key]["total_time_s"])
            / max(float(candidate[key]["total_time_s"]), 1.0e-18)
            - 1.0
            for key in keys
        ]
        stationary_interval = _interval(stationary_effects or [0.0])
        transition_interval = _interval(transition_effects or [0.0])
        wall_interval = _interval(wall_effects or [0.0])
        charged_interval = _interval(charged_time_effects or [0.0])
        stationary_noninferior = bool(
            keys and float(stationary_interval["upper"]) <= STATIONARY_NONINFERIORITY_MARGIN
        )
        transition_gain = bool(
            keys
            and float(transition_interval["mean"]) > 0.0
            and float(transition_interval["lower"]) >= 0.0
        )
        wall_gain = bool(
            keys
            and (
                (float(wall_interval["mean"]) > 0.0 and float(wall_interval["lower"]) >= 0.0)
                or (
                    float(charged_interval["mean"]) > 0.0
                    and float(charged_interval["lower"]) >= 0.0
                )
            )
        )
        stationary_rows.append(
            {
                "arm": arm,
                "matched_row_count": len(keys),
                "stationary_error_candidate_minus_gibbs": stationary_interval,
                "noninferiority_margin": STATIONARY_NONINFERIORITY_MARGIN,
                "stationary_noninferior": stationary_noninferior,
            }
        )
        efficiency_rows.append(
            {
                "arm": arm,
                "matched_row_count": len(keys),
                "ess_per_transition_fractional_gain": transition_interval,
                "ess_per_wall_second_fractional_gain": wall_interval,
                "charged_total_time_fractional_gain": charged_interval,
                "transition_gain_with_nonnegative_lower_bound": transition_gain,
                "wall_or_charged_time_gain_with_nonnegative_lower_bound": wall_gain,
                "software_win": bool(stationary_noninferior and wall_gain),
                "algorithmic_only": bool(
                    stationary_noninferior and transition_gain and not wall_gain
                ),
            }
        )
    return (
        {
            "definition": "energy_error/n_spins plus maximum absolute spin-moment error",
            "noninferiority_margin": STATIONARY_NONINFERIORITY_MARGIN,
            "rows": stationary_rows,
            "reducer": "paired preregistered fixture-seed rows",
        },
        {
            "definition": "minimum observable ESS with transitions and charged wall separated",
            "rows": efficiency_rows,
            "arm_means": [
                {
                    "arm": arm,
                    "mean_ess_per_transition": float(
                        np.mean(
                            [row["ess_per_transition"] for row in successful if row["arm"] == arm]
                        )
                    ),
                    "mean_ess_per_wall_second": float(
                        np.mean(
                            [row["ess_per_wall_second"] for row in successful if row["arm"] == arm]
                        )
                    ),
                    "mean_setup_time_s": float(
                        np.mean([row["setup_time_s"] for row in successful if row["arm"] == arm])
                    ),
                    "mean_sampling_time_s": float(
                        np.mean([row["sampling_time_s"] for row in successful if row["arm"] == arm])
                    ),
                    "mean_total_time_s": float(
                        np.mean([row["total_time_s"] for row in successful if row["arm"] == arm])
                    ),
                }
                for arm in ARMS
                if any(row["arm"] == arm for row in successful)
            ],
            "reducer": "paired intervals and arithmetic means from per_unit_rows",
        },
    )


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): sha256_file(root / path) if (root / path).is_file() else None
        for path in PROTECTED_RELATIVE_PATHS
    }


def _command_version(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover - live block path.
        return f"unavailable:{type(error).__name__}:{error}"
    return result.stdout.strip() or result.stderr.strip()


def _cpu_receipt() -> JsonDict:
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
                break
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "available_affinity_cpu_count": affinity,
        "available_ram_bytes": None if mem_available_kib is None else mem_available_kib * 1024,
        "parallelism_used": "one chain at a time; Rust kernel single-threaded",
    }


def _sampler_receipts(root: Path) -> JsonDict:
    paths = (
        SAMPLER_RELATIVE_PATH,
        RUST_RELATIVE_PATH,
        PYO3_RELATIVE_PATH,
        Path("python/carnot/samplers/backend.py"),
    )
    sampler_source = (root / SAMPLER_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "interface_identity": "SamplerBackend:SpectralKBlockBackend",
        "python_entrypoints": [
            "build_spectral_blocks",
            "build_random_blocks",
            "run_python_chain",
            "run_rust_chain",
            "SpectralKBlockBackend.sample",
        ],
        "rust_identity": "carnot_samplers::spectral_k_block::SpectralKBlockCore",
        "pyo3_identity": "carnot._rust.spectral_k_block.RustSpectralKBlockCore",
        "rng_identity": "lcg64_pcg_constants_top53_uniform_v1",
        "energy_convention": "-0.5*s^T*J*s - h^T*s",
        "source_hashes": {path.as_posix(): sha256_file(root / path) for path in paths},
        "sampler_imports_experiment_module": "experiment_" in sampler_source,
        "silent_rust_fallback": False,
    }


def _field_provenance() -> dict[str, JsonDict]:
    sources = {
        "status": ["gate_check_summary", "attack_rows", "per_unit_rows"],
        "honest_verdict": [
            "stationary_quality_summary",
            "efficiency_summary",
            "rust_python_parity_rows",
        ],
        "verdict_class": ["stationary_quality_summary", "efficiency_summary", "gate_check_summary"],
        "gate_check_summary": ["per_unit_rows", "fixture_and_reference_receipts", "attack_rows"],
        "per_unit_rows": [
            "fixture_and_reference_receipts",
            "partition_rows",
            "chain timers",
            "ESS reducer",
        ],
        "fixture_and_reference_receipts": [
            "frozen_frustrated_fixtures",
            "exact enumeration",
            "independent reference chains",
        ],
        "sampler_implementation_receipts": [
            "source file hashes",
            "SamplerBackend identity",
            "PyO3 identity",
        ],
        "partition_rows": ["build_spectral_blocks", "build_random_blocks", "partition timers"],
        "stationary_quality_summary": ["per_unit_rows", "paired interval reducer"],
        "efficiency_summary": ["per_unit_rows", "paired interval reducer", "monotonic timers"],
        "rust_python_parity_rows": ["matched spectral Python/Rust chain rows"],
        "spectral_scale_ready_score": ["reference, arm, parity, and cost row completeness checks"],
        "hardware_path_receipt": ["sampler operation census", "source inspection"],
        "claim_boundaries": ["REQ-SAMPLER-6612-BOUNDARY", "ops/exclusion_manifest.yaml"],
        "attack_rows": [
            "row matrix",
            "reference receipts",
            "partition identity",
            "claim boundaries",
        ],
        "preconditions_checked": [
            "protected hashes",
            "Exp6597 hash",
            "tool versions",
            "CPU and RAM receipt",
        ],
        "protected_files_unchanged": ["protected hashes before and after build_artifact"],
        "inference_substrate": ["REQ-SAMPLER-6612-BOUNDARY"],
        "verifier_is_oracle": ["REQ-SAMPLER-6612-REFERENCE"],
        "field_provenance": ["REQUIRED_ARTIFACT_FIELDS", "FIELD_PRINCIPLES"],
        "duration_s": ["time.perf_counter around references and all treatment chains"],
        "tests_run": ["CARNOT_6612_TEST_RECEIPTS external receipt file"],
        "reproducibility_checksum": ["canonical JSON with checksum blanked"],
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    config: ExperimentConfig | None = None,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run references and every preregistered arm, then reduce one artifact."""

    started = time.perf_counter()
    config = config or ExperimentConfig()
    if not config.treatment_seeds or not config.reference_seeds:
        raise ValueError("treatment and reference seeds must be non-empty")
    if config.burn_in <= 0 or config.retained_samples <= 0:
        raise ValueError("treatment burn-in and retained samples must be positive")
    if config.reference_burn_in <= 0 or config.reference_retained_samples <= 0:
        raise ValueError("reference burn-in and retained samples must be positive")
    fixtures = _selected_fixtures(config)
    protected_before = _protected_hashes(root)
    sampler_receipts = _sampler_receipts(root)
    partition_rows: list[JsonDict] = []
    reference_receipts: list[JsonDict] = []
    per_unit_rows: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    partitions: dict[str, tuple[SpinPartition, SpinPartition]] = {}
    references: dict[str, _ReferenceStats] = {}

    for fixture in fixtures:
        spectral = build_spectral_blocks(fixture.couplings, config.block_size)
        random = build_random_blocks(
            fixture.n_spins,
            config.block_size,
            fixture.generation_seed + 77,
            forbidden_hash=spectral.sha256,
        )
        partitions[fixture.fixture_id] = (spectral, random)
        partition_rows.extend((_partition_row(fixture, spectral), _partition_row(fixture, random)))
        reference = (
            _exact_reference(fixture)
            if fixture.n_spins == 16
            else _long_chain_reference(fixture, config)
        )
        references[fixture.fixture_id] = reference
        reference_receipts.append(
            {
                "fixture_id": fixture.fixture_id,
                "n_spins": fixture.n_spins,
                "generation_seed": fixture.generation_seed,
                "fixture_sha256": fixture.fixture_sha256,
                "couplings": fixture.couplings.tolist(),
                "fields": fixture.fields.tolist(),
                "temperature": fixture.temperature,
                "has_non_bipartite_cycle": fixture.has_non_bipartite_cycle,
                "mixed_sign_couplings": bool(
                    np.any(fixture.couplings > 0.0) and np.any(fixture.couplings < 0.0)
                ),
                "competing_modes": fixture.competing_modes,
                "matrix_frozen_before_sampling": True,
                **reference.receipt,
            }
        )

    for fixture in fixtures:
        spectral, random = partitions[fixture.fixture_id]
        reference = references[fixture.fixture_id]
        for seed in config.treatment_seeds:
            seed_rows: dict[str, JsonDict] = {}
            for arm in ARMS:
                partition = (
                    random if arm == "random_k_block" else spectral if "spectral" in arm else None
                )
                try:
                    run = _run_arm(fixture, seed, arm, partition, config)
                    metrics = _evaluate_chain(fixture, reference, run)
                    setup_time = 0.0 if partition is None else partition.setup_time_s
                    total_time = setup_time + run.sample_time_s
                    rng_domain = (
                        "sequential_control"
                        if arm == "sequential_gibbs"
                        else "random_control"
                        if arm == "random_k_block"
                        else "spectral_matched_parity"
                    )
                    row = {
                        "row_id": f"{fixture.fixture_id}:seed{seed}:{arm}",
                        "fixture_id": fixture.fixture_id,
                        "n_spins": fixture.n_spins,
                        "seed": seed,
                        "arm": arm,
                        "burn_in": config.burn_in,
                        "retained_sample_count": config.retained_samples,
                        "transitions": run.transitions,
                        "spins_updated": run.spins_updated,
                        **metrics,
                        "ess_per_transition": metrics["effective_sample_size"] / run.transitions,
                        "ess_per_wall_second": metrics["effective_sample_size"] / total_time,
                        "partition_kind": "single_spin" if partition is None else partition.kind,
                        "partition_sha256": None if partition is None else partition.sha256,
                        "setup_time_s": setup_time,
                        "sampling_time_s": run.sample_time_s,
                        "total_time_s": total_time,
                        "rng_algorithm": run.rng_algorithm,
                        "rng_domain": rng_domain,
                        "rng_initial_state": run.rng_initial_state,
                        "rng_final_state": run.rng_final_state,
                        "initial_state_sha256": sha256_bytes(
                            _initial_state(fixture, seed).tobytes()
                        ),
                        "sample_sha256": run.sample_sha256,
                        "parity_sample_mismatch_fraction": None,
                        "parity_energy_mean_delta": None,
                        "parity_moment_linf_delta": None,
                        "failure": None,
                    }
                    row["_samples"] = run.samples
                except Exception as error:  # pragma: no cover - live fail-closed receipt.
                    row = _failure_row(fixture, seed, arm, config, error)
                seed_rows[arm] = row
                per_unit_rows.append(row)
            python_row = seed_rows["spectral_k_block_python"]
            rust_row = seed_rows["spectral_k_block_rust"]
            if python_row["failure"] is None and rust_row["failure"] is None:
                python_samples = python_row.pop("_samples")
                rust_samples = rust_row.pop("_samples")
                mismatch = float(np.mean(python_samples != rust_samples))
                energy_delta = abs(
                    float(python_row["energy_mean"]) - float(rust_row["energy_mean"])
                )
                moment_delta = float(
                    np.max(
                        np.abs(
                            np.asarray(python_row["moment_mean"])
                            - np.asarray(rust_row["moment_mean"])
                        )
                    )
                )
                parity = {
                    "fixture_id": fixture.fixture_id,
                    "n_spins": fixture.n_spins,
                    "seed": seed,
                    "python_sample_sha256": python_row["sample_sha256"],
                    "rust_sample_sha256": rust_row["sample_sha256"],
                    "sample_mismatch_fraction": mismatch,
                    "sample_mismatch_tolerance": PARITY_SAMPLE_MISMATCH_TOLERANCE,
                    "distribution_total_variation_delta": mismatch,
                    "energy_mean_delta": energy_delta,
                    "moment_linf_delta": moment_delta,
                    "moment_tolerance": PARITY_MOMENT_TOLERANCE,
                    "sampling_cost_delta_s_rust_minus_python": float(rust_row["sampling_time_s"])
                    - float(python_row["sampling_time_s"]),
                    "transitions_delta": int(rust_row["transitions"])
                    - int(python_row["transitions"]),
                    "spins_updated_delta": int(rust_row["spins_updated"])
                    - int(python_row["spins_updated"]),
                    "rng_final_state_equal": rust_row["rng_final_state"]
                    == python_row["rng_final_state"],
                    "passed": bool(
                        mismatch <= PARITY_SAMPLE_MISMATCH_TOLERANCE
                        and energy_delta <= PARITY_MOMENT_TOLERANCE
                        and moment_delta <= PARITY_MOMENT_TOLERANCE
                        and rust_row["transitions"] == python_row["transitions"]
                        and rust_row["spins_updated"] == python_row["spins_updated"]
                        and rust_row["rng_final_state"] == python_row["rng_final_state"]
                    ),
                    "failure": None,
                }
                for row in (python_row, rust_row):
                    row["parity_sample_mismatch_fraction"] = mismatch
                    row["parity_energy_mean_delta"] = energy_delta
                    row["parity_moment_linf_delta"] = moment_delta
            else:  # pragma: no cover - live Rust/toolchain block path.
                python_row.pop("_samples", None)
                rust_row.pop("_samples", None)
                parity = {
                    "fixture_id": fixture.fixture_id,
                    "n_spins": fixture.n_spins,
                    "seed": seed,
                    "passed": False,
                    "failure": python_row["failure"] or rust_row["failure"],
                }
            seed_rows["sequential_gibbs"].pop("_samples", None)
            seed_rows["random_k_block"].pop("_samples", None)
            parity_rows.append(parity)

    stationary_summary, efficiency_summary = _paired_summaries(per_unit_rows)
    expected_rows = len(fixtures) * len(config.treatment_seeds) * len(ARMS)
    failures = [row for row in per_unit_rows if row["failure"] is not None]
    references_complete = len(reference_receipts) == len(fixtures) and all(
        row["independent_of_treatment"] is True and row["treatment_samples_used"] is False
        for row in reference_receipts
    )
    parity_complete = len(parity_rows) == len(fixtures) * len(config.treatment_seeds) and all(
        row.get("passed") is True for row in parity_rows
    )
    cost_complete = all(
        row["failure"] is None
        and row["sampling_time_s"] > 0.0
        and row["total_time_s"] >= row["sampling_time_s"]
        for row in per_unit_rows
    )
    row_matrix_complete = len(per_unit_rows) == expected_rows
    protected_after = _protected_hashes(root)
    protected = {
        "before": protected_before,
        "after": protected_after,
        "all_unchanged": protected_before == protected_after
        and all(value is not None for value in protected_before.values()),
    }
    claim_boundaries = {
        "software_claim_only": True,
        "attached_hardware_execution": False,
        "general_hardware_performance_claim": False,
        "fpga_execution_claim": False,
        "tsu_execution_claim": False,
        "pimi_method_used": False,
        "retired_phase3_homotopy_argmin_used": False,
        "hubo_reduction_used": False,
        "scope": "CPU Python and Rust frustrated Ising block heat-bath sampling",
    }
    random_spectral_distinct = all(
        group[0]["partition_sha256"] != group[1]["partition_sha256"]
        and group[0]["blocks"] != group[1]["blocks"]
        for group in (
            [row for row in partition_rows if row["fixture_id"] == fixture.fixture_id]
            for fixture in fixtures
        )
    )
    rng_domains_valid = all(
        len(
            {
                row["rng_initial_state"]
                for row in per_unit_rows
                if row["fixture_id"] == fixture.fixture_id
                and row["seed"] == seed
                and row["arm"] in {"sequential_gibbs", "random_k_block", "spectral_k_block_python"}
            }
        )
        == 3
        for fixture in fixtures
        for seed in config.treatment_seeds
    )
    attacks = [
        {
            "attack_id": "retired_homotopy_hubo_substitution",
            "passed": not claim_boundaries["retired_phase3_homotopy_argmin_used"]
            and not claim_boundaries["hubo_reduction_used"],
        },
        {"attack_id": "pimi_adjacency_reuse", "passed": not claim_boundaries["pimi_method_used"]},
        {"attack_id": "treatment_defined_reference", "passed": references_complete},
        {
            "attack_id": "burn_in_deletion",
            "passed": all(row["burn_in"] == config.burn_in for row in per_unit_rows),
        },
        {
            "attack_id": "transition_undercharging",
            "passed": all(
                row["failure"] is not None
                or row["transitions"] == config.burn_in + config.retained_samples
                for row in per_unit_rows
            ),
        },
        {
            "attack_id": "spin_update_undercharging",
            "passed": all(
                row["failure"] is not None or row["spins_updated"] >= row["transitions"]
                for row in per_unit_rows
            ),
        },
        {
            "attack_id": "setup_omission",
            "passed": all(
                row["failure"] is not None
                or row["arm"] == "sequential_gibbs"
                or row["setup_time_s"] > 0.0
                for row in per_unit_rows
            ),
        },
        {
            "attack_id": "identical_random_and_spectral_partitions",
            "passed": random_spectral_distinct,
        },
        {"attack_id": "shared_mutable_rng_state", "passed": rng_domains_valid},
        {
            "attack_id": "parity_tolerance_inflation",
            "passed": PARITY_SAMPLE_MISMATCH_TOLERANCE == 0.0
            and PARITY_MOMENT_TOLERANCE == 1.0e-12
            and parity_complete,
        },
        {
            "attack_id": "fpga_or_tsu_wording_upgrade",
            "passed": not claim_boundaries["attached_hardware_execution"]
            and not claim_boundaries["general_hardware_performance_claim"],
        },
        {"attack_id": "protected_file_mutation", "passed": protected["all_unchanged"]},
        {
            "attack_id": "failure_row_deletion",
            "passed": row_matrix_complete and len(per_unit_rows) == expected_rows,
        },
    ]
    scale_ready = float(
        row_matrix_complete
        and not failures
        and references_complete
        and parity_complete
        and cost_complete
        and all(row["passed"] for row in attacks)
    )
    rust_stationary = next(
        row for row in stationary_summary["rows"] if row["arm"] == "spectral_k_block_rust"
    )
    rust_efficiency = next(
        row for row in efficiency_summary["rows"] if row["arm"] == "spectral_k_block_rust"
    )
    receipts = [dict(row) for row in (test_receipts or [])]
    test_blockers = _test_receipt_blockers(receipts)
    blockers: list[JsonDict] = []
    blockers.extend(
        {
            "fixture_id": row["fixture_id"],
            "seed": row["seed"],
            "arm": row["arm"],
            "gate": "chain_failure",
            "observed": row["failure"],
        }
        for row in failures
    )
    blockers.extend(
        {
            "fixture_id": row["fixture_id"],
            "seed": row["seed"],
            "gate": "rust_python_parity",
            "observed": row.get("failure") or row,
        }
        for row in parity_rows
        if row.get("passed") is not True
    )
    blockers.extend(
        {"gate": row["attack_id"], "observed": False}
        for row in attacks
        if row["passed"] is not True
    )
    blockers.extend(test_blockers)
    blocked = bool(blockers or scale_ready != 1.0)
    if blocked:
        verdict_class = "blocked"
        status = "blocked_reference_parity_cost_or_protection"
        honest_verdict = (
            "blocked_reference_parity_cost_or_protection: stationary, transition, wall, "
            "and parity conclusions are withheld; CPU software only"
        )
    else:
        software_win = bool(rust_efficiency["software_win"])
        algorithmic_only = bool(rust_efficiency["algorithmic_only"])
        verdict_class = "positive" if software_win else "partial" if algorithmic_only else "null"
        status = "complete"
        honest_verdict = (
            f"{verdict_class}: stationary_noninferior={rust_stationary['stationary_noninferior']}; "
            f"transition_gain={rust_efficiency['transition_gain_with_nonnegative_lower_bound']}; "
            f"wall_or_charged_time_gain={rust_efficiency['wall_or_charged_time_gain_with_nonnegative_lower_bound']}; "
            f"rust_python_parity={parity_complete}; CPU software only"
        )
    tests_complete = not test_blockers
    source_paths = (
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        SAMPLER_RELATIVE_PATH,
        SAMPLER_TEST_RELATIVE_PATH,
        RUST_RELATIVE_PATH,
        RUST_TEST_RELATIVE_PATH,
        PYO3_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        RUST_SPEC_RELATIVE_PATH,
        Path("ops/exclusion_manifest.yaml"),
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_date": str(run_date),
        "random_seed": config.treatment_seeds[0],
        "n_samples": config.retained_samples,
        "sample_size_rationale": "At least 10,000 retained samples per default arm satisfy the n>=64 MCMC floor and expose ESS convergence.",
        "experiment_config": json.loads(canonical_json(asdict(config))),
        "spec_refs": [
            "REQ-SAMPLER-6612",
            "SCENARIO-SAMPLER-6612-INDEPENDENT-SCALE-EVIDENCE",
            "SCENARIO-SAMPLER-6612-RUST-PARITY-AND-FAIL-CLOSED-VERDICT",
            "REQ-RUSTPY-6612",
        ],
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": blockers,
        "per_unit_rows": per_unit_rows,
        "fixture_and_reference_receipts": reference_receipts,
        "sampler_implementation_receipts": sampler_receipts,
        "partition_rows": partition_rows,
        "stationary_quality_summary": stationary_summary,
        "efficiency_summary": efficiency_summary,
        "rust_python_parity_rows": parity_rows,
        "spectral_scale_ready_score": scale_ready,
        "hardware_path_receipt": {
            "attached_hardware_executed": False,
            "arithmetic": "bounded 2^block_size conditional weights plus dense local fields",
            "memory": "O(n^2 + retained_samples*n) host memory",
            "rng": "two deterministic 64-bit LCG draws per transition",
            "parallelism": "single-chain scalar core; independent fixture-seed rows can run in parallel later",
            "software_observation_only": True,
            "latency_power_energy_claimed": False,
        },
        "claim_boundaries": claim_boundaries,
        "attack_rows": attacks,
        "preconditions_checked": {
            "planning_date": str(run_date),
            "protected_hashes_before": protected_before,
            "prior_exp6597_path": "results/experiment_6597_spectral_k_block_ising_canary.json",
            "prior_exp6597_sha256": sha256_file(
                root / "results/experiment_6597_spectral_k_block_ising_canary.json"
            ),
            "python_version": platform.python_version(),
            "rustc_version": _command_version(("rustc", "--version")),
            "cargo_version": _command_version(("cargo", "--version")),
            "cpu_topology_and_ram": _cpu_receipt(),
            "fixture_generation_seeds": [fixture.generation_seed for fixture in fixtures],
            "fixture_hashes_frozen_before_sampling": {
                fixture.fixture_id: fixture.fixture_sha256 for fixture in fixtures
            },
            "treatment_seeds": list(config.treatment_seeds),
            "reference_seeds": list(config.reference_seeds),
            "chain_lengths": {
                "treatment_burn_in": config.burn_in,
                "treatment_retained": config.retained_samples,
                "reference_burn_in": config.reference_burn_in,
                "reference_retained_per_chain": config.reference_retained_samples,
            },
            "reference_methods": {"n16": "exact_enumeration", "n32": "independent_long_chains"},
            "gpu_used": False,
            "hardware_board_used": False,
            "llm_used": False,
            "source_hashes": {path.as_posix(): sha256_file(root / path) for path in source_paths},
            "tests_complete": tests_complete,
        },
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(time.perf_counter() - started, 1.0e-12),
        "tests_run": receipts,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash all artifact content after blanking this self-reference."""

    stable = json.loads(canonical_json(payload))
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(payload: Mapping[str, Any]) -> bool:
    """Reject missing, circular, undercharged, rebranded, or edited evidence."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")
    if payload["verdict_class"] not in ALLOWED_VERDICT_CLASSES:
        raise ValueError("verdict_class unsupported")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if payload["protected_files_unchanged"].get("all_unchanged") is not True:
        raise ValueError("protected files changed")
    boundaries = payload["claim_boundaries"]
    if (
        boundaries.get("attached_hardware_execution") is not False
        or boundaries.get("general_hardware_performance_claim") is not False
    ):
        raise ValueError("hardware claim is forbidden")
    config = payload["experiment_config"]
    fixture_ids = {row["fixture_id"] for row in payload["fixture_and_reference_receipts"]}
    expected = {
        (fixture_id, seed, arm)
        for fixture_id in fixture_ids
        for seed in config["treatment_seeds"]
        for arm in ARMS
    }
    rows = payload["per_unit_rows"]
    observed = {(row["fixture_id"], row["seed"], row["arm"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("row matrix is incomplete or duplicated")
    if any(row["burn_in"] != config["burn_in"] or row["burn_in"] <= 0 for row in rows):
        raise ValueError("burn-in mismatch or deletion")
    ready = payload["spectral_scale_ready_score"]
    if ready not in {0.0, 1.0}:
        raise ValueError("spectral_scale_ready_score must be binary")
    if ready == 1.0:
        total = config["burn_in"] + config["retained_samples"]
        if any(row["transitions"] != total for row in rows):
            raise ValueError("transition charge mismatch")
        if any(
            row["arm"] != "sequential_gibbs" and float(row["setup_time_s"]) <= 0.0 for row in rows
        ):
            raise ValueError("setup charge omitted")
        if any(row["failure"] is not None for row in rows):
            raise ValueError("ready artifact contains chain failure")
    references = payload["fixture_and_reference_receipts"]
    if any(
        row.get("independent_of_treatment") is not True
        or row.get("treatment_samples_used") is not False
        for row in references
    ):
        raise ValueError("reference independence failed")
    parity_rows = payload["rust_python_parity_rows"]
    if ready == 1.0 and (
        len(parity_rows) != len(fixture_ids) * len(config["treatment_seeds"])
        or any(
            row.get("passed") is not True
            or float(row.get("sample_mismatch_fraction", 1.0)) > PARITY_SAMPLE_MISMATCH_TOLERANCE
            for row in parity_rows
        )
    ):
        raise ValueError("parity row missing or outside tolerance")
    partitions = payload["partition_rows"]
    for fixture_id in fixture_ids:
        fixture_partitions = [row for row in partitions if row["fixture_id"] == fixture_id]
        if (
            len(fixture_partitions) != 2
            or len({row["partition_sha256"] for row in fixture_partitions}) != 2
        ):
            raise ValueError("partition identity or row matrix failed")
    if ready == 1.0 and not all(row.get("passed") is True for row in payload["attack_rows"]):
        raise ValueError("attack row failed")
    provenance = payload["field_provenance"]
    if any(
        field not in provenance or "principle" not in provenance[field]
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        raise ValueError("field_provenance incomplete")
    if ready == 0.0 and (
        not str(payload["status"]).startswith("blocked_")
        or not str(payload["honest_verdict"]).startswith("blocked_")
        or not payload["gate_check_summary"]
    ):
        raise ValueError("blocked artifact lacks named gate summary")
    return True


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted replacement only.
            temporary.unlink()


def _load_test_receipts() -> list[JsonDict]:
    path_text = os.environ.get("CARNOT_6612_TEST_RECEIPTS")
    if not path_text:
        return []
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("test receipt file must contain a list")
    return [dict(row) for row in payload]


def run_experiment(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | None = None,
) -> JsonDict:
    """Run the scale matrix, validate it, and write the terminal artifact."""

    artifact = build_artifact(root=root, run_date=run_date, test_receipts=_load_test_receipts())
    validate_artifact(artifact)
    write_json_atomic(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date, output_path=args.output)
    print(
        canonical_json(
            {
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "spectral_scale_ready_score": artifact["spectral_scale_ready_score"],
                "rows": len(artifact["per_unit_rows"]),
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
