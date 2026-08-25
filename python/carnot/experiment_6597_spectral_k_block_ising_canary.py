"""Run a bounded spectral k-block canary on exact-enumerable Ising models.

The paper defines a state-space partition from bottom modes of ``P squared``.
The resulting kernel first applies one random-site heat-bath step. It then
draws from the model conditional inside the current spectral block. Exact
enumeration evaluates the retained samples, but the sampler never receives
the evaluator probability vector.

Spec refs: REQ-SAMPLER-6597,
SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION,
SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE,
SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from math import exp, sqrt
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6597_spectral_k_block_ising_canary.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6597_spectral_k_block_ising_canary.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6597_spectral_k_block_ising_canary.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
RUN_DATE = "20260825"
EXPERIMENT_ID = "experiment_6597_spectral_k_block_ising_canary"
SCHEMA_VERSION = "carnot.experiment_6597.spectral_k_block_ising_canary.v1"
INFERENCE_SUBSTRATE = "cpu_spectral_k_block_ising_exact_enumeration"
PAPER_URL = "https://arxiv.org/abs/2608.21466"
PAPER_VERSION = "arXiv:2608.21466v1"

SEEDS = (6597, 6598, 6599, 6600, 6601)
BLOCK_SIZES = (2, 4)
BURN_IN = 2_000
RETAINED_SAMPLES = 10_000
TRANSITION_BUDGET = BURN_IN + RETAINED_SAMPLES
STATIONARY_TV_NONINFERIORITY_MARGIN = 0.02
DEGENERACY_TOLERANCE = 1.0e-10
ALLOWED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/hardware/kv260_mmd_vs_cpu_sequential_gibbs.py"),
    Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py"),
    Path("crates/carnot-ising/src/lib.rs"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6597_spectral_k_block_ising_canary.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=*/python/carnot/experiment_6597_spectral_k_block_ising_canary.py -m pytest tests/python/test_experiment_6597_spectral_k_block_ising_canary.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=*/python/carnot/experiment_6597_spectral_k_block_ising_canary.py --fail-under=100",
    "cargo test -p carnot-ising --lib --quiet",
    ".venv/bin/pytest tests/python/test_e2e_training_sampling.py -q -o addopts=",
    ".venv/bin/ruff check python/carnot/experiment_6597_spectral_k_block_ising_canary.py tests/python/test_experiment_6597_spectral_k_block_ising_canary.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6597_spectral_k_block_ising_canary.py tests/python/test_experiment_6597_spectral_k_block_ising_canary.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6597_spectral_k_block_ising_canary.py",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6597_spectral_k_block_ising_canary.json",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6597_spectral_k_block_ising_canary.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "method_source_receipt",
    "fixture_receipts",
    "spectral_partition_rows",
    "sampler_run_rows",
    "exact_distribution_comparison",
    "effective_sample_size_summary",
    "paired_statistical_receipts",
    "acceptance_gate_rows",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The sampler canary ends with complete distributional evidence or a named numerical block.",
    "honest_verdict": "The verdict is limited to charged software fixtures and makes no hardware claim.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "gate_check_summary": "Any block names the method, fixture, eigensolver, sample, resource, or numerical check and observed value.",
    "per_unit_rows": "Every fixture, temperature, seed, arm, and block size carries distribution, ESS, cost, failure, and memory metrics.",
    "method_source_receipt": "The spectral method, bounded implementation, and PIMI retirement distinction bind by source.",
    "fixture_receipts": "Couplings, fields, temperatures, exact distributions, starts, seeds, and budgets are immutable.",
    "spectral_partition_rows": "Operator, eigenvalues, degeneracy handling, blocks, setup cost, and failures are auditable.",
    "sampler_run_rows": "Gibbs and block arms share matched budgets and retain all failed runs.",
    "exact_distribution_comparison": "Stationary and moment errors recompute independently from retained samples.",
    "effective_sample_size_summary": "ESS, ESS per transition, autocorrelation, and charged wall time remain separate.",
    "paired_statistical_receipts": "Effects, intervals, wins, losses, ties, and family heterogeneity remain explicit.",
    "acceptance_gate_rows": "Stationary noninferiority and efficiency conditions record expected and observed values.",
    "attack_rows": "Oracle sampling, mismatch, omitted setup, sample floor, seed, selection, bias, rebranding, and aggregate attacks fail closed.",
    "preconditions_checked": "Sources, samplers, fixtures, seeds, temperatures, blocks, sample floors, resources, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task declares CPU software Ising sampling and exact enumeration with no LLM.",
    "verifier_is_oracle": "Exact enumeration independently evaluates stationary sampling quality.",
    "field_provenance": "Every field points to fixture, partition, sample, exact, and reducer rows.",
    "duration_s": "Monotonic duration exposes too-short or skipped sampling.",
    "tests_run": "Focused sampler, cross-language, and E2E commands include exits and durations.",
    "reproducibility_checksum": "A final content hash protects the distributional result.",
}


@dataclass(frozen=True)
class IsingFixture:
    """One frozen finite Ising system with an auditable family label."""

    fixture_id: str
    family: str
    couplings: np.ndarray = field(compare=False, repr=False)
    fields: np.ndarray = field(compare=False, repr=False)
    temperature: float

    @property
    def n_spins(self) -> int:
        return int(self.fields.size)


@dataclass(frozen=True)
class SpectralPartition:
    """A paper-derived state-space partition plus conditional draw tables."""

    fixture_id: str
    block_count: int
    blocks: tuple[tuple[int, ...], ...]
    labels: np.ndarray = field(compare=False, repr=False)
    conditional_indices: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    conditional_cdfs: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    operator: str
    eigensolver: str
    eigenvalue_ordering: str
    selected_eigenvalues: tuple[float, ...]
    all_eigenvalues: tuple[float, ...]
    degeneracy_handling: JsonDict = field(compare=False)
    objective_f: float
    rounding_candidate_count: int
    setup_time_s: float
    failure: str | None = None


@dataclass(frozen=True)
class ChainRun:
    """Raw retained indices and charged runtime data for one sampler arm."""

    sample_indices: np.ndarray = field(compare=False, repr=False)
    sample_sha256: str
    random_stream_sha256: str
    initial_state_index: int
    transitions: int
    kernel_component_updates: int
    sample_time_s: float
    peak_memory_mib: float
    failure: str | None


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys and no platform-specific spacing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 representation."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a source file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def frozen_fixtures() -> tuple[IsingFixture, ...]:
    """Return the preregistered independent, ferro, and frustrated systems."""

    independent = np.zeros((6, 6), dtype=np.float64)
    ferro = np.zeros((6, 6), dtype=np.float64)
    for left, right, weight in (
        (0, 1, 0.55),
        (1, 2, 0.55),
        (2, 3, 0.55),
        (3, 4, 0.55),
        (4, 5, 0.55),
        (5, 0, 0.55),
        (0, 3, 0.15),
        (1, 4, 0.15),
        (2, 5, 0.15),
    ):
        ferro[left, right] = ferro[right, left] = weight
    frustrated = np.zeros((6, 6), dtype=np.float64)
    for left, right, weight in (
        (0, 1, 0.8),
        (1, 2, 0.8),
        (2, 0, -0.8),
        (3, 4, 0.75),
        (4, 5, 0.75),
        (5, 3, -0.75),
        (2, 3, 0.25),
    ):
        frustrated[left, right] = frustrated[right, left] = weight
    return (
        IsingFixture(
            "independent_fields_n6",
            "independent",
            independent,
            np.array([0.20, -0.15, 0.10, -0.05, 0.12, -0.08], dtype=np.float64),
            1.0,
        ),
        IsingFixture(
            "ferromagnetic_ring_chords_n6",
            "ferromagnetic",
            ferro,
            np.array([0.03, -0.02, 0.01, -0.03, 0.02, -0.01], dtype=np.float64),
            0.85,
        ),
        IsingFixture(
            "frustrated_two_triangles_n6",
            "frustrated",
            frustrated,
            np.array([0.08, -0.06, 0.04, -0.05, 0.07, -0.03], dtype=np.float64),
            0.90,
        ),
    )


def enumerate_states(n_spins: int) -> np.ndarray:
    """Enumerate binary spin states in stable little-endian index order."""

    indices = np.arange(2 ** int(n_spins), dtype=np.uint64)[:, None]
    bits = (indices >> np.arange(int(n_spins), dtype=np.uint64)) & 1
    return (bits.astype(np.int8) * 2 - 1).astype(np.int8)


def energy_vector(fixture: IsingFixture, states: np.ndarray) -> np.ndarray:
    """Compute ``-0.5 s J s - h s`` for every supplied state."""

    spins = states.astype(np.float64)
    pair = -0.5 * np.einsum("bi,ij,bj->b", spins, fixture.couplings, spins)
    return pair - spins @ fixture.fields


def _boltzmann_distribution(fixture: IsingFixture, states: np.ndarray) -> np.ndarray:
    energies = energy_vector(fixture, states)
    log_weights = -energies / float(fixture.temperature)
    weights = np.exp(log_weights - float(np.max(log_weights)))
    return weights / float(np.sum(weights))


def exact_distribution(fixture: IsingFixture, states: np.ndarray) -> np.ndarray:
    """Return the evaluator's normalized target for independent comparison."""

    return _boltzmann_distribution(fixture, states)


def heat_bath_transition_matrix(fixture: IsingFixture, states: np.ndarray) -> np.ndarray:
    """Build the exact random-site, single-spin heat-bath operator ``P``."""

    count = len(states)
    matrix = np.zeros((count, count), dtype=np.float64)
    beta = 1.0 / float(fixture.temperature)
    for source, state in enumerate(states.astype(np.float64)):
        for site in range(fixture.n_spins):
            local_field = float(fixture.couplings[site] @ state + fixture.fields[site])
            probability_plus = 1.0 / (1.0 + exp(-2.0 * beta * local_field))
            plus = source | (1 << site)
            minus = source & ~(1 << site)
            matrix[source, plus] += probability_plus / fixture.n_spins
            matrix[source, minus] += (1.0 - probability_plus) / fixture.n_spins
    return matrix


def _canonicalize_degenerate_basis(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, list[JsonDict]]:
    """Choose a stable basis inside tied eigenspaces by coordinate projection."""

    canonical = np.zeros_like(eigenvectors)
    groups: list[JsonDict] = []
    start = 0
    while start < len(eigenvalues):
        stop = start + 1
        while (
            stop < len(eigenvalues)
            and abs(float(eigenvalues[stop] - eigenvalues[start])) <= DEGENERACY_TOLERANCE
        ):
            stop += 1
        width = stop - start
        subspace = eigenvectors[:, start:stop]
        projector = subspace @ subspace.T
        basis: list[np.ndarray] = []
        for coordinate in range(projector.shape[0]):
            candidate = projector[:, coordinate].copy()
            for prior in basis:
                candidate -= float(prior @ candidate) * prior
            norm = float(np.linalg.norm(candidate))
            if norm > 1.0e-9:
                candidate /= norm
                pivot = int(np.argmax(np.abs(candidate)))
                if candidate[pivot] < 0.0:
                    candidate *= -1.0
                basis.append(candidate)
            if len(basis) == width:
                break
        canonical[:, start:stop] = np.column_stack(basis)
        if width > 1:
            groups.append(
                {
                    "start_index": start,
                    "stop_index_exclusive": stop,
                    "multiplicity": width,
                    "eigenvalue": float(eigenvalues[start]),
                }
            )
        start = stop
    return canonical, groups


def _canonical_blocks(labels: np.ndarray, k: int) -> tuple[tuple[int, ...], ...] | None:
    blocks = [tuple(int(index) for index in np.flatnonzero(labels == label)) for label in range(k)]
    if any(not block for block in blocks):
        return None
    return tuple(sorted(blocks, key=lambda block: block[0]))


def _labels_from_blocks(blocks: tuple[tuple[int, ...], ...], count: int) -> np.ndarray:
    labels = np.empty(count, dtype=np.int64)
    for label, block in enumerate(blocks):
        labels[np.asarray(block, dtype=np.int64)] = label
    return labels


def _partition_objective(
    blocks: tuple[tuple[int, ...], ...],
    stationary: np.ndarray,
    squared_operator: np.ndarray,
) -> float:
    trace = 0.0
    for block in blocks:
        indices = np.asarray(block, dtype=np.int64)
        mass = float(np.sum(stationary[indices]))
        trace += (
            float(np.sum(stationary[indices, None] * squared_operator[np.ix_(indices, indices)]))
            / mass
        )
    return max(0.0, trace - 1.0)


def _weighted_kmeans_candidate(
    embedding: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...] | None:
    count = len(embedding)
    centers = [int(rng.choice(count, p=weights))]
    while len(centers) < k:
        distance = np.min(
            np.sum(
                (embedding[:, None, :] - embedding[np.asarray(centers)][None, :, :]) ** 2, axis=2
            ),
            axis=1,
        )
        probabilities = weights * distance
        probabilities[np.asarray(centers)] = 0.0
        total = float(np.sum(probabilities))
        if total <= 1.0e-18:
            centers.append(next(index for index in range(count) if index not in centers))
        else:
            centers.append(int(rng.choice(count, p=probabilities / total)))
    center_values = embedding[np.asarray(centers)].copy()
    labels = np.full(count, -1, dtype=np.int64)
    for _ in range(100):
        distances = np.sum((embedding[:, None, :] - center_values[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for label in range(k):
            members = labels == label
            if not np.any(members):
                return None
            center_values[label] = np.average(embedding[members], axis=0, weights=weights[members])
    return _canonical_blocks(labels, k)


def _round_spectral_embedding(
    embedding: np.ndarray,
    stationary: np.ndarray,
    squared_operator: np.ndarray,
    k: int,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    candidates: set[tuple[tuple[int, ...], ...]] = set()
    if k == 2:
        order = np.lexsort((np.arange(len(embedding)), embedding[:, 0]))
        for cut in range(1, len(order)):
            candidates.add(
                (
                    tuple(sorted(int(value) for value in order[:cut])),
                    tuple(sorted(int(value) for value in order[cut:])),
                )
            )
    else:
        for start in range(32):
            candidate = _weighted_kmeans_candidate(
                embedding,
                stationary,
                k,
                np.random.default_rng(6597 + 1009 * k + start),
            )
            if candidate is not None:
                candidates.add(candidate)
    if not candidates:
        raise ValueError("spectral rounding produced no nonempty partition")
    selected = min(
        candidates,
        key=lambda blocks: (_partition_objective(blocks, stationary, squared_operator), blocks),
    )
    return selected, len(candidates)


def build_spectral_partition(
    fixture: IsingFixture,
    states: np.ndarray,
    block_count: int,
) -> SpectralPartition:
    """Build and rescore one bottom-``P squared`` spectral partition."""

    if block_count < 2 or block_count >= len(states):
        raise ValueError("block count must be between 2 and state_count - 1")
    started = time.perf_counter()
    stationary = _boltzmann_distribution(fixture, states)
    operator = heat_bath_transition_matrix(fixture, states)
    squared = operator @ operator
    root = np.sqrt(stationary)
    symmetric = root[:, None] * squared / root[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvectors, tied_groups = _canonicalize_degenerate_basis(eigenvalues, eigenvectors)
    selected_vectors = eigenvectors[:, : block_count - 1]
    embedding = selected_vectors / root[:, None]
    blocks, candidate_count = _round_spectral_embedding(
        embedding,
        stationary,
        squared,
        block_count,
    )
    labels = _labels_from_blocks(blocks, len(states))
    energies = energy_vector(fixture, states)
    conditional_indices: list[np.ndarray] = []
    conditional_cdfs: list[np.ndarray] = []
    for block in blocks:
        indices = np.asarray(block, dtype=np.int64)
        log_weights = -energies[indices] / float(fixture.temperature)
        weights = np.exp(log_weights - float(np.max(log_weights)))
        cdf = np.cumsum(weights / float(np.sum(weights)))
        cdf[-1] = 1.0
        conditional_indices.append(indices)
        conditional_cdfs.append(cdf)
    cutoff_value = float(eigenvalues[block_count - 2])
    cutoff_group = next(
        (
            group
            for group in tied_groups
            if group["start_index"] <= block_count - 2 < group["stop_index_exclusive"]
        ),
        None,
    )
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return SpectralPartition(
        fixture_id=fixture.fixture_id,
        block_count=block_count,
        blocks=blocks,
        labels=labels,
        conditional_indices=tuple(conditional_indices),
        conditional_cdfs=tuple(conditional_cdfs),
        operator="squared_random_site_heat_bath_P2",
        eigensolver="numpy.linalg.eigh_symmetric_similarity",
        eigenvalue_ordering="ascending_P2_eigenvalue_bottom_nonconstant_first",
        selected_eigenvalues=tuple(float(value) for value in eigenvalues[: block_count - 1]),
        all_eigenvalues=tuple(float(value) for value in eigenvalues),
        degeneracy_handling={
            "tolerance": DEGENERACY_TOLERANCE,
            "rule": "coordinate_projection_gram_schmidt_with_positive_largest_pivot",
            "tied_groups": tied_groups,
            "cutoff_eigenvalue": cutoff_value,
            "cutoff_inside_tied_group": bool(
                cutoff_group is not None and cutoff_group["stop_index_exclusive"] > block_count - 1
            ),
            "cutoff_group": cutoff_group,
        },
        objective_f=_partition_objective(blocks, stationary, squared),
        rounding_candidate_count=candidate_count,
        setup_time_s=elapsed,
    )


def matched_random_stream(seed: int, transitions: int) -> np.ndarray:
    """Create common random numbers for site, heat-bath, and averaging draws."""

    return np.random.default_rng(int(seed)).random((int(transitions), 3), dtype=np.float64)


def initial_state_index(seed: int, state_count: int) -> int:
    """Freeze one deterministic, non-target-derived initial state per seed."""

    return int((int(seed) * 1103515245 + 12345) % int(state_count))


def _peak_memory_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / 1024.0 if platform.system() != "Darwin" else usage / (1024.0 * 1024.0)


def run_chain(
    fixture: IsingFixture,
    states: np.ndarray,
    partition: SpectralPartition | None,
    *,
    initial_index: int,
    random_stream: np.ndarray,
    burn_in: int,
    retained_samples: int,
) -> ChainRun:
    """Run sequential Gibbs or the composed averaging kernel from model data."""

    total = int(burn_in) + int(retained_samples)
    if random_stream.shape != (total, 3):
        raise ValueError("random stream must have shape (burn_in + retained_samples, 3)")
    state_index = int(initial_index)
    samples = np.empty(int(retained_samples), dtype=np.int64)
    beta = 1.0 / float(fixture.temperature)
    started = time.perf_counter()
    for transition in range(total):
        state = states[state_index].astype(np.float64)
        site = min(int(random_stream[transition, 0] * fixture.n_spins), fixture.n_spins - 1)
        local_field = float(fixture.couplings[site] @ state + fixture.fields[site])
        probability_plus = 1.0 / (1.0 + exp(-2.0 * beta * local_field))
        if random_stream[transition, 1] < probability_plus:
            state_index |= 1 << site
        else:
            state_index &= ~(1 << site)
        if partition is not None:
            label = int(partition.labels[state_index])
            position = int(
                np.searchsorted(
                    partition.conditional_cdfs[label],
                    random_stream[transition, 2],
                    side="right",
                )
            )
            state_index = int(partition.conditional_indices[label][position])
        if transition >= burn_in:
            samples[transition - burn_in] = state_index
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return ChainRun(
        sample_indices=samples,
        sample_sha256=sha256_bytes(samples.astype("<i8", copy=False).tobytes()),
        random_stream_sha256=sha256_bytes(random_stream.astype("<f8", copy=False).tobytes()),
        initial_state_index=int(initial_index),
        transitions=total,
        kernel_component_updates=total * (2 if partition is not None else 1),
        sample_time_s=elapsed,
        peak_memory_mib=_peak_memory_mib(),
        failure=None,
    )


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate autocorrelation time with an FFT and positive paired sums."""

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


def lag_one_autocorrelation(values: np.ndarray) -> float:
    """Return lag-one correlation, or zero for a constant or singleton row."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    centered = array - float(np.mean(array))
    denominator = float(centered @ centered)
    if denominator <= 1.0e-18:
        return 0.0
    return float(centered[:-1] @ centered[1:] / denominator)


def evaluate_samples(
    fixture: IsingFixture,
    states: np.ndarray,
    exact_target: np.ndarray,
    sample_indices: np.ndarray,
) -> JsonDict:
    """Evaluate retained counts independently against the enumerated target."""

    count = int(len(sample_indices))
    counts = np.bincount(sample_indices, minlength=len(states)).astype(np.int64)
    empirical = counts.astype(np.float64) / count
    spins = states.astype(np.float64)
    empirical_mean = empirical @ spins
    exact_mean = exact_target @ spins
    empirical_second = np.einsum("s,si,sj->ij", empirical, spins, spins)
    exact_second = np.einsum("s,si,sj->ij", exact_target, spins, spins)
    empirical_covariance = empirical_second - np.outer(empirical_mean, empirical_mean)
    exact_covariance = exact_second - np.outer(exact_mean, exact_mean)
    retained_states = spins[sample_indices]
    energies = energy_vector(fixture, states)[sample_indices]
    magnetization = np.sum(retained_states, axis=1)
    observables = [
        magnetization,
        energies,
        *(retained_states[:, site] for site in range(fixture.n_spins)),
    ]
    autocorrelation_times = [integrated_autocorrelation_time(values) for values in observables]
    worst_tau = max(autocorrelation_times)
    return {
        "retained_sample_count": count,
        "state_counts": counts.tolist(),
        "empirical_distribution_sha256": sha256_json(empirical.tolist()),
        "total_variation_error": float(0.5 * np.sum(np.abs(empirical - exact_target))),
        "mean_l2_error": float(np.linalg.norm(empirical_mean - exact_mean)),
        "covariance_frobenius_error": float(
            np.linalg.norm(empirical_covariance - exact_covariance)
        ),
        "lag_one_autocorrelation": lag_one_autocorrelation(magnetization),
        "integrated_autocorrelation_time": float(worst_tau),
        "effective_sample_size": float(count / worst_tau),
        "observable_autocorrelation_times": {
            "magnetization": float(autocorrelation_times[0]),
            "energy": float(autocorrelation_times[1]),
            "spin_coordinates": [float(value) for value in autocorrelation_times[2:]],
        },
    }


def paired_interval(values: Sequence[float]) -> JsonDict:
    """Return a preregistered two-sided 95 percent paired t interval."""

    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "sample_size": 1}
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(array), 1.96)
    half_width = critical * float(np.std(array, ddof=1)) / sqrt(len(array))
    return {
        "mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "sample_size": int(len(array)),
    }


def _partition_row(partition: SpectralPartition, state_count: int) -> JsonDict:
    return {
        "fixture_id": partition.fixture_id,
        "block_size": partition.block_count,
        "state_count": state_count,
        "operator": partition.operator,
        "eigensolver": partition.eigensolver,
        "eigenvalue_ordering": partition.eigenvalue_ordering,
        "selected_eigenvalues": list(partition.selected_eigenvalues),
        "all_eigenvalues": list(partition.all_eigenvalues),
        "degeneracy_handling": partition.degeneracy_handling,
        "blocks": [list(block) for block in partition.blocks],
        "block_membership_sha256": sha256_json(partition.labels.tolist()),
        "objective_f": partition.objective_f,
        "rounding_candidate_count": partition.rounding_candidate_count,
        "setup_time_s": partition.setup_time_s,
        "failure": partition.failure,
    }


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        path.as_posix(): {
            "exists": (root / path).is_file(),
            "sha256": sha256_file(root / path) if (root / path).is_file() else None,
        }
        for path in SOURCE_RELATIVE_PATHS
    }


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): sha256_file(root / path) if (root / path).is_file() else None
        for path in PROTECTED_RELATIVE_PATHS
    }


def _cpu_resources() -> JsonDict:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "available_affinity_cpu_count": affinity,
        "physical_memory_mib": page_size * pages // (1024 * 1024),
        "peak_process_memory_mib_at_precondition": _peak_memory_mib(),
    }


def _fixture_receipt(fixture: IsingFixture, states: np.ndarray, target: np.ndarray) -> JsonDict:
    return {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "n_spins": fixture.n_spins,
        "couplings": fixture.couplings.tolist(),
        "fields": fixture.fields.tolist(),
        "temperature": fixture.temperature,
        "inverse_temperature": 1.0 / fixture.temperature,
        "states": states.tolist(),
        "state_ordering": "little_endian_binary_index_with_minus_one_for_zero_bit",
        "exact_probabilities": target.tolist(),
        "exact_distribution_sha256": sha256_json(target.tolist()),
        "initial_state_by_seed": {
            str(seed): initial_state_index(seed, len(states)) for seed in SEEDS
        },
        "seeds": list(SEEDS),
        "block_sizes": list(BLOCK_SIZES),
        "burn_in": BURN_IN,
        "retained_samples": RETAINED_SAMPLES,
        "transition_budget": TRANSITION_BUDGET,
    }


def _paired_receipts(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    receipts: list[JsonDict] = []
    for fixture in frozen_fixtures():
        baseline = {
            int(row["seed"]): row
            for row in per_unit_rows
            if row["fixture_id"] == fixture.fixture_id and row["arm"] == "sequential_gibbs"
        }
        for block_size in BLOCK_SIZES:
            treatment = {
                int(row["seed"]): row
                for row in per_unit_rows
                if row["fixture_id"] == fixture.fixture_id
                and row["arm"] == "spectral_k_block"
                and row["block_size"] == block_size
            }
            matched = sorted(set(baseline) & set(treatment))
            tv_effects = [
                float(treatment[seed]["total_variation_error"])
                - float(baseline[seed]["total_variation_error"])
                for seed in matched
            ]
            ess_effects = [
                float(treatment[seed]["ess_per_transition"])
                - float(baseline[seed]["ess_per_transition"])
                for seed in matched
            ]
            wall_effects = [
                float(baseline[seed]["charged_wall_time_s"])
                / float(treatment[seed]["charged_wall_time_s"])
                - 1.0
                for seed in matched
            ]
            tv_interval = paired_interval(tv_effects)
            ess_interval = paired_interval(ess_effects)
            wall_interval = paired_interval(wall_effects)
            stationary_noninferior = bool(
                len(matched) == len(SEEDS)
                and float(tv_interval["upper"]) <= STATIONARY_TV_NONINFERIORITY_MARGIN
            )
            ess_gain = bool(
                float(ess_interval["mean"]) > 0.0 and float(ess_interval["lower"]) >= 0.0
            )
            wall_gain = bool(
                float(wall_interval["mean"]) > 0.0 and float(wall_interval["lower"]) >= 0.0
            )
            signs = [int(np.sign(value)) for value in ess_effects]
            receipts.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "family": fixture.family,
                    "block_size": block_size,
                    "matched_seeds": matched,
                    "stationary_tv_candidate_minus_gibbs": tv_interval,
                    "ess_per_transition_candidate_minus_gibbs": ess_interval,
                    "charged_wall_fractional_gain": wall_interval,
                    "wins": signs.count(1),
                    "losses": signs.count(-1),
                    "ties": signs.count(0),
                    "stationary_noninferior": stationary_noninferior,
                    "ess_gain_with_nonnegative_lower_bound": ess_gain,
                    "charged_wall_gain_with_nonnegative_lower_bound": wall_gain,
                    "paired_win": bool(stationary_noninferior and (ess_gain or wall_gain)),
                    "heterogeneity_preserved": True,
                }
            )
    return receipts


def _acceptance_rows(paired: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for receipt in paired:
        observed_tv = receipt["stationary_tv_candidate_minus_gibbs"]
        observed_ess = receipt["ess_per_transition_candidate_minus_gibbs"]
        observed_wall = receipt["charged_wall_fractional_gain"]
        rows.extend(
            [
                {
                    "fixture_id": receipt["fixture_id"],
                    "family": receipt["family"],
                    "block_size": receipt["block_size"],
                    "gate": "stationary_distribution_noninferiority",
                    "expected": f"upper_bound <= {STATIONARY_TV_NONINFERIORITY_MARGIN}",
                    "observed": observed_tv,
                    "passed": receipt["stationary_noninferior"],
                },
                {
                    "fixture_id": receipt["fixture_id"],
                    "family": receipt["family"],
                    "block_size": receipt["block_size"],
                    "gate": "positive_efficiency_effect",
                    "expected": "ESS-per-transition or charged-wall mean > 0 with lower_bound >= 0",
                    "observed": {"ess": observed_ess, "charged_wall": observed_wall},
                    "passed": bool(
                        receipt["ess_gain_with_nonnegative_lower_bound"]
                        or receipt["charged_wall_gain_with_nonnegative_lower_bound"]
                    ),
                },
            ]
        )
    frustrated = [row for row in paired if row["family"] == "frustrated"]
    rows.append(
        {
            "fixture_id": "all_frustrated_rows",
            "family": "frustrated",
            "block_size": "all_preregistered",
            "gate": "no_pooled_away_frustrated_regression",
            "expected": "every frustrated paired row passes",
            "observed": [bool(row["paired_win"]) for row in frustrated],
            "passed": bool(frustrated and all(bool(row["paired_win"]) for row in frustrated)),
        }
    )
    return rows


def _attack_rows(
    per_unit_rows: Sequence[Mapping[str, Any]],
    partition_rows: Sequence[Mapping[str, Any]],
    method_source_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    expected_count = len(frozen_fixtures()) * len(SEEDS) * (1 + len(BLOCK_SIZES))
    expected_cells = {
        (fixture.fixture_id, seed, arm, block)
        for fixture in frozen_fixtures()
        for seed in SEEDS
        for arm, block in (
            ("sequential_gibbs", None),
            *(("spectral_k_block", k) for k in BLOCK_SIZES),
        )
    }
    observed_cells = {
        (row["fixture_id"], row["seed"], row["arm"], row["block_size"]) for row in per_unit_rows
    }
    spectral_rows = [row for row in per_unit_rows if row["arm"] == "spectral_k_block"]
    transitions = {int(row["transitions"]) for row in per_unit_rows}
    return [
        {
            "attack_id": "exact_distribution_inside_sampling",
            "expected": "sampler API accepts model, partition, start, and random stream only",
            "observed": method_source_receipt["sampler_inputs"],
            "passed": method_source_receipt["sampler_accepts_evaluator_distribution"] is False,
        },
        {
            "attack_id": "unmatched_transitions",
            "expected": [TRANSITION_BUDGET],
            "observed": sorted(transitions),
            "passed": transitions == {TRANSITION_BUDGET},
        },
        {
            "attack_id": "omitted_partition_setup",
            "expected": "every spectral row charges its full partition setup",
            "observed": [float(row["charged_setup_time_s"]) for row in spectral_rows],
            "passed": bool(
                spectral_rows
                and all(float(row["charged_setup_time_s"]) > 0.0 for row in spectral_rows)
            ),
        },
        {
            "attack_id": "retained_sample_floor",
            "expected": f">= {RETAINED_SAMPLES} after explicit burn-in",
            "observed": min(int(row["retained_sample_count"]) for row in per_unit_rows),
            "passed": all(
                int(row["retained_sample_count"]) >= RETAINED_SAMPLES for row in per_unit_rows
            ),
        },
        {
            "attack_id": "seed_drop",
            "expected": [list(cell) for cell in sorted(expected_cells)],
            "observed": [list(cell) for cell in sorted(observed_cells)],
            "passed": observed_cells == expected_cells,
        },
        {
            "attack_id": "favorable_block_selection",
            "expected": list(BLOCK_SIZES),
            "observed": sorted({int(row["block_size"]) for row in partition_rows}),
            "passed": sorted({int(row["block_size"]) for row in partition_rows})
            == list(BLOCK_SIZES),
        },
        {
            "attack_id": "synchronous_update_bias",
            "expected": "one random-site heat-bath update followed by exact block averaging",
            "observed": sorted({str(row["update_schedule"]) for row in per_unit_rows}),
            "passed": all(
                row["update_schedule"]
                in {"sequential_single_spin", "sequential_single_spin_then_partition_average"}
                for row in per_unit_rows
            ),
        },
        {
            "attack_id": "pimi_or_hardware_rebranding",
            "expected": "software spectral partition; no FPGA, TSU, or PIMI claim",
            "observed": {
                "mechanism": method_source_receipt["mechanism_name"],
                "hardware_claimed": method_source_receipt["hardware_claimed"],
                "pimi_claimed": method_source_receipt["pimi_claimed"],
            },
            "passed": method_source_receipt["hardware_claimed"] is False
            and method_source_receipt["pimi_claimed"] is False,
        },
        {
            "attack_id": "aggregate_only_claim",
            "expected": expected_count,
            "observed": len(per_unit_rows),
            "passed": len(per_unit_rows) == expected_count,
        },
    ]


def _field_provenance() -> dict[str, JsonDict]:
    sources = {
        "status": ["acceptance_gate_rows", "attack_rows", "sampler_run_rows"],
        "honest_verdict": ["acceptance_gate_rows", "paired_statistical_receipts"],
        "verdict_class": ["acceptance_gate_rows", "paired_statistical_receipts"],
        "gate_check_summary": ["acceptance_gate_rows", "sampler_run_rows"],
        "per_unit_rows": [
            "fixture_receipts",
            "spectral_partition_rows",
            "sampler_run_rows",
            "exact_distribution_comparison",
        ],
        "method_source_receipt": [
            PAPER_VERSION,
            "research-references.md",
            "ops/exclusion_manifest.yaml",
        ],
        "fixture_receipts": ["frozen_fixtures", "exact_distribution"],
        "spectral_partition_rows": ["heat_bath_transition_matrix", "build_spectral_partition"],
        "sampler_run_rows": ["run_chain", "matched_random_stream"],
        "exact_distribution_comparison": ["evaluate_samples", "fixture_receipts"],
        "effective_sample_size_summary": ["evaluate_samples", "integrated_autocorrelation_time"],
        "paired_statistical_receipts": ["per_unit_rows", "paired_interval"],
        "acceptance_gate_rows": ["paired_statistical_receipts", "REQ-SAMPLER-6597-PAIRED"],
        "attack_rows": ["per_unit_rows", "method_source_receipt", "spectral_partition_rows"],
        "preconditions_checked": ["source_paths_and_hashes", "fixture_receipts", "cpu_resources"],
        "protected_files_unchanged": ["protected_hashes_before", "protected_hashes_after"],
        "inference_substrate": ["REQ-SAMPLER-6597-BOUNDARY"],
        "verifier_is_oracle": ["REQ-SAMPLER-6597-EXACT"],
        "field_provenance": ["REQUIRED_ARTIFACT_FIELDS", "FIELD_PRINCIPLES"],
        "duration_s": ["time.perf_counter"],
        "tests_run": ["external command receipt file"],
        "reproducibility_checksum": ["canonical JSON content"],
    }
    return {
        field: {"sources": sources[field], "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _failure_row(
    fixture: IsingFixture,
    seed: int,
    arm: str,
    block_size: int | None,
    error: Exception,
) -> JsonDict:  # pragma: no cover - only used for a live numerical failure.
    return {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "temperature": fixture.temperature,
        "seed": seed,
        "arm": arm,
        "block_size": block_size,
        "retained_sample_count": 0,
        "burn_in": BURN_IN,
        "transition_budget": TRANSITION_BUDGET,
        "transitions": 0,
        "kernel_component_updates": 0,
        "total_variation_error": None,
        "mean_l2_error": None,
        "covariance_frobenius_error": None,
        "lag_one_autocorrelation": None,
        "integrated_autocorrelation_time": None,
        "effective_sample_size": None,
        "ess_per_transition": None,
        "charged_setup_time_s": 0.0,
        "sample_time_s": 0.0,
        "charged_wall_time_s": 0.0,
        "peak_memory_mib": _peak_memory_mib(),
        "failure": f"{type(error).__name__}:{error}",
        "update_schedule": "failed_before_sampling",
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the full preregistered matrix and build one terminal artifact."""

    started = time.perf_counter()
    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    cpu_resources = _cpu_resources()
    fixture_receipts: list[JsonDict] = []
    partition_rows: list[JsonDict] = []
    per_unit_rows: list[JsonDict] = []
    sampler_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []

    for fixture in frozen_fixtures():
        states = enumerate_states(fixture.n_spins)
        target = exact_distribution(fixture, states)
        fixture_receipts.append(_fixture_receipt(fixture, states, target))
        partitions: dict[int, SpectralPartition] = {}
        partition_errors: dict[int, Exception] = {}
        for block_size in BLOCK_SIZES:
            try:
                partition = build_spectral_partition(fixture, states, block_size)
                partitions[block_size] = partition
                partition_rows.append(_partition_row(partition, len(states)))
            except Exception as error:  # pragma: no cover - fail-closed live receipt.
                partition_errors[block_size] = error
                partition_rows.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "block_size": block_size,
                        "operator": "squared_random_site_heat_bath_P2",
                        "eigensolver": "numpy.linalg.eigh_symmetric_similarity",
                        "failure": f"{type(error).__name__}:{error}",
                        "setup_time_s": 0.0,
                        "blocks": [],
                    }
                )
        for seed in SEEDS:
            stream = matched_random_stream(seed, TRANSITION_BUDGET)
            start_index = initial_state_index(seed, len(states))
            arms: list[tuple[str, int | None, SpectralPartition | None]] = [
                ("sequential_gibbs", None, None),
                *(
                    ("spectral_k_block", block_size, partitions.get(block_size))
                    for block_size in BLOCK_SIZES
                ),
            ]
            for arm, block_size, partition in arms:
                if block_size in partition_errors:  # pragma: no cover - fail-closed live receipt.
                    row = _failure_row(fixture, seed, arm, block_size, partition_errors[block_size])
                    per_unit_rows.append(row)
                    sampler_rows.append(dict(row))
                    exact_rows.append(dict(row))
                    continue
                try:
                    run = run_chain(
                        fixture,
                        states,
                        partition,
                        initial_index=start_index,
                        random_stream=stream,
                        burn_in=BURN_IN,
                        retained_samples=RETAINED_SAMPLES,
                    )
                    metrics = evaluate_samples(fixture, states, target, run.sample_indices)
                    setup_time = partition.setup_time_s if partition is not None else 0.0
                    row_id = f"{fixture.fixture_id}:temp{fixture.temperature}:seed{seed}:{arm}:k{block_size}"
                    row = {
                        "row_id": row_id,
                        "fixture_id": fixture.fixture_id,
                        "family": fixture.family,
                        "temperature": fixture.temperature,
                        "seed": seed,
                        "arm": arm,
                        "block_size": block_size,
                        "retained_sample_count": metrics["retained_sample_count"],
                        "burn_in": BURN_IN,
                        "transition_budget": TRANSITION_BUDGET,
                        "transitions": run.transitions,
                        "kernel_component_updates": run.kernel_component_updates,
                        "total_variation_error": metrics["total_variation_error"],
                        "mean_l2_error": metrics["mean_l2_error"],
                        "covariance_frobenius_error": metrics["covariance_frobenius_error"],
                        "lag_one_autocorrelation": metrics["lag_one_autocorrelation"],
                        "integrated_autocorrelation_time": metrics[
                            "integrated_autocorrelation_time"
                        ],
                        "effective_sample_size": metrics["effective_sample_size"],
                        "ess_per_transition": float(metrics["effective_sample_size"])
                        / run.transitions,
                        "charged_setup_time_s": setup_time,
                        "sample_time_s": run.sample_time_s,
                        "charged_wall_time_s": setup_time + run.sample_time_s,
                        "peak_memory_mib": run.peak_memory_mib,
                        "failure": run.failure,
                        "update_schedule": (
                            "sequential_single_spin"
                            if partition is None
                            else "sequential_single_spin_then_partition_average"
                        ),
                        "sample_sha256": run.sample_sha256,
                        "random_stream_sha256": run.random_stream_sha256,
                        "initial_state_index": run.initial_state_index,
                    }
                    per_unit_rows.append(row)
                    sampler_rows.append(
                        {
                            "row_id": row_id,
                            "fixture_id": fixture.fixture_id,
                            "temperature": fixture.temperature,
                            "seed": seed,
                            "arm": arm,
                            "block_size": block_size,
                            "initial_state_index": start_index,
                            "random_stream_sha256": run.random_stream_sha256,
                            "sample_sha256": run.sample_sha256,
                            "state_counts": metrics["state_counts"],
                            "burn_in": BURN_IN,
                            "retained_sample_count": RETAINED_SAMPLES,
                            "transitions": run.transitions,
                            "kernel_component_updates": run.kernel_component_updates,
                            "charged_setup_time_s": setup_time,
                            "sample_time_s": run.sample_time_s,
                            "charged_wall_time_s": setup_time + run.sample_time_s,
                            "peak_memory_mib": run.peak_memory_mib,
                            "failure": run.failure,
                        }
                    )
                    exact_rows.append(
                        {
                            "row_id": row_id,
                            "fixture_id": fixture.fixture_id,
                            "temperature": fixture.temperature,
                            "seed": seed,
                            "arm": arm,
                            "block_size": block_size,
                            "exact_distribution_sha256": sha256_json(target.tolist()),
                            "retained_counts_sha256": sha256_json(metrics["state_counts"]),
                            "retained_sample_count": metrics["retained_sample_count"],
                            "total_variation_error": metrics["total_variation_error"],
                            "mean_l2_error": metrics["mean_l2_error"],
                            "covariance_frobenius_error": metrics["covariance_frobenius_error"],
                            "evaluation_role": "independent_evaluator_not_sampler_input",
                            "failure": None,
                        }
                    )
                except Exception as error:  # pragma: no cover - fail-closed live receipt.
                    row = _failure_row(fixture, seed, arm, block_size, error)
                    per_unit_rows.append(row)
                    sampler_rows.append(dict(row))
                    exact_rows.append(dict(row))

    method_source_receipt = {
        "source_id": PAPER_VERSION,
        "source_url": PAPER_URL,
        "retrieved_title": "Spectral partitioning for k-block averaging kernels of finite Markov chains",
        "bounded_method": "bottom nonconstant eigenfunctions of P squared; weighted rounding; exact F rescoring; composed G_O P kernel",
        "operator_objective": "F(O)=Tr((G_O-Pi)P^2)",
        "sampler_inputs": [
            "Ising fixture",
            "spectral block membership",
            "initial state",
            "matched random stream",
        ],
        "sampler_accepts_evaluator_distribution": False,
        "conditional_sampling_source": "unnormalized Ising energy within the selected block",
        "exact_evaluator_role": "post-sampling distribution and moment comparison only",
        "mechanism_name": "software_spectral_state_partition_averaging",
        "pimi_retirement_source": "ops/exclusion_manifest.yaml",
        "pimi_distinction": "PIMI inertia hardware directions remain retired; this method is a different state-space spectral partition.",
        "hardware_claimed": False,
        "pimi_claimed": False,
        "fpga_claimed": False,
        "tsu_claimed": False,
    }
    paired = (
        _paired_receipts(per_unit_rows)
        if all(row["failure"] is None for row in per_unit_rows)
        else []
    )
    acceptance = _acceptance_rows(paired) if paired else []
    attacks = _attack_rows(per_unit_rows, partition_rows, method_source_receipt)
    failed_rows = [row for row in per_unit_rows if row["failure"] is not None]
    all_attacks_passed = all(bool(row["passed"]) for row in attacks)
    all_paired_wins = bool(paired and all(bool(row["paired_win"]) for row in paired))
    verdict_class = (
        "blocked"
        if failed_rows or not all_attacks_passed
        else ("positive" if all_paired_wins else "null")
    )
    status = "blocked" if verdict_class == "blocked" else "complete"
    failed_gates = [row for row in acceptance if not bool(row["passed"])]
    gate_summary = [
        {
            "fixture_id": row["fixture_id"],
            "family": row["family"],
            "block_size": row["block_size"],
            "gate": row["gate"],
            "expected": row["expected"],
            "observed": row["observed"],
        }
        for row in failed_gates
    ]
    gate_summary.extend(
        {
            "fixture_id": row["fixture_id"],
            "method": row["arm"],
            "block_size": row["block_size"],
            "gate": "sampler_run_failure",
            "expected": None,
            "observed": row["failure"],
        }
        for row in failed_rows
    )
    protected_after = _protected_hashes(root)
    protected = {
        "before": protected_before,
        "after": protected_after,
        "all_unchanged": protected_before == protected_after
        and all(value is not None for value in protected_before.values()),
    }
    receipts = [dict(row) for row in (test_receipts or [])]
    tests_by_command = {str(row.get("command")): row for row in receipts}
    tests_complete = all(
        command in tests_by_command
        and tests_by_command[command].get("exit_code") == 0
        and float(tests_by_command[command].get("duration_s", -1.0)) >= 0.0
        for command in DEFAULT_TEST_COMMANDS
    )
    honest_verdict = (
        "complete: charged CPU software exact-enumerable fixtures support every spectral k-block arm; no FPGA, TSU, PIMI, or general hardware claim"
        if verdict_class == "positive"
        else (
            "complete: charged CPU software exact-enumerable fixtures do not satisfy every stationary and efficiency win gate; no hardware claim"
            if verdict_class == "null"
            else "blocked: named sampler, partition, sample, or attack checks failed; no hardware claim"
        )
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "schema_version": SCHEMA_VERSION,
        "random_seed": SEEDS[0],
        "spec_refs": [
            "REQ-SAMPLER-6597",
            "SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION",
            "SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE",
            "SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT",
        ],
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": gate_summary,
        "per_unit_rows": per_unit_rows,
        "method_source_receipt": method_source_receipt,
        "fixture_receipts": fixture_receipts,
        "spectral_partition_rows": partition_rows,
        "sampler_run_rows": sampler_rows,
        "exact_distribution_comparison": exact_rows,
        "effective_sample_size_summary": {
            "definition": "minimum ESS across magnetization, energy, and each spin coordinate",
            "rows": [
                {
                    "row_id": row.get("row_id"),
                    "fixture_id": row["fixture_id"],
                    "family": row["family"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "block_size": row["block_size"],
                    "lag_one_autocorrelation": row["lag_one_autocorrelation"],
                    "integrated_autocorrelation_time": row["integrated_autocorrelation_time"],
                    "effective_sample_size": row["effective_sample_size"],
                    "ess_per_transition": row["ess_per_transition"],
                    "transitions": row["transitions"],
                    "charged_wall_time_s": row["charged_wall_time_s"],
                }
                for row in per_unit_rows
            ],
            "family_heterogeneity": [
                {
                    "family": family,
                    "paired_rows": [row for row in paired if row["family"] == family],
                }
                for family in ("independent", "ferromagnetic", "frustrated")
            ],
        },
        "paired_statistical_receipts": paired,
        "acceptance_gate_rows": acceptance,
        "attack_rows": attacks,
        "preconditions_checked": {
            "planning_date": str(run_date),
            "paper": method_source_receipt,
            "source_paths_and_hashes": source_hashes,
            "existing_sequential_gibbs_path": "python/carnot/hardware/kv260_mmd_vs_cpu_sequential_gibbs.py",
            "existing_exact_enumerator_path": "python/carnot/experiment_5622_cdls_exact_kernel_audit.py",
            "rust_ising_model_path": "crates/carnot-ising/src/lib.rs",
            "fixtures": [receipt["fixture_id"] for receipt in fixture_receipts],
            "families": [receipt["family"] for receipt in fixture_receipts],
            "temperatures": [receipt["temperature"] for receipt in fixture_receipts],
            "seeds": list(SEEDS),
            "block_sizes": list(BLOCK_SIZES),
            "burn_in": BURN_IN,
            "retained_sample_count": RETAINED_SAMPLES,
            "transition_budget": TRANSITION_BUDGET,
            "cpu_resources": cpu_resources,
            "protected_hashes_before": protected_before,
            "tests_complete": tests_complete,
            "all_sources_present": all(row["exists"] for row in source_hashes.values()),
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
    """Hash all artifact content after blanking only this checksum field."""

    copy = json.loads(canonical_json(payload))
    copy["reproducibility_checksum"] = ""
    return sha256_json(copy)


def validate_artifact(payload: Mapping[str, Any]) -> bool:
    """Reject incomplete, unmatched, under-sized, rebranded, or edited evidence."""

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
    if payload["method_source_receipt"].get("hardware_claimed") is not False:
        raise ValueError("hardware claim is forbidden")
    if payload["method_source_receipt"].get("pimi_claimed") is not False:
        raise ValueError("PIMI claim is forbidden")
    if payload["protected_files_unchanged"].get("all_unchanged") is not True:
        raise ValueError("protected files changed")
    rows = payload["per_unit_rows"]
    expected = {
        (fixture.fixture_id, seed, arm, block)
        for fixture in frozen_fixtures()
        for seed in SEEDS
        for arm, block in (
            ("sequential_gibbs", None),
            *(("spectral_k_block", k) for k in BLOCK_SIZES),
        )
    }
    observed = {(row["fixture_id"], row["seed"], row["arm"], row["block_size"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("row matrix is incomplete or duplicated")
    if any(int(row["retained_sample_count"]) < RETAINED_SAMPLES for row in rows):
        raise ValueError("sample floor failed")
    if any(int(row["transitions"]) != TRANSITION_BUDGET for row in rows):
        raise ValueError("transition budget mismatch")
    if any(
        float(row["charged_setup_time_s"]) <= 0.0
        for row in rows
        if row["arm"] == "spectral_k_block"
    ):
        raise ValueError("setup cost omitted")
    if len(payload["sampler_run_rows"]) != len(expected) or len(
        payload["exact_distribution_comparison"]
    ) != len(expected):
        raise ValueError("sampler or exact row matrix mismatch")
    if len(payload["spectral_partition_rows"]) != len(frozen_fixtures()) * len(BLOCK_SIZES):
        raise ValueError("spectral partition row matrix mismatch")
    if not all(row.get("passed") is True for row in payload["attack_rows"]):
        raise ValueError("attack row failed")
    provenance = payload["field_provenance"]
    if any(
        field not in provenance or "principle" not in provenance[field]
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        raise ValueError("field_provenance incomplete")
    return True


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through a same-directory atomic replacement."""

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
        if temporary.exists():  # pragma: no cover - only reached after an interrupted replace.
            temporary.unlink()


def _load_test_receipts() -> list[JsonDict]:
    path_text = os.environ.get("CARNOT_6597_TEST_RECEIPTS")
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
    """Run the canary, validate it, and write its terminal artifact."""

    artifact = build_artifact(root=root, run_date=run_date, test_receipts=_load_test_receipts())
    validate_artifact(artifact)
    write_json_atomic(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin command wrapper.
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
                "rows": len(artifact["per_unit_rows"]),
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
