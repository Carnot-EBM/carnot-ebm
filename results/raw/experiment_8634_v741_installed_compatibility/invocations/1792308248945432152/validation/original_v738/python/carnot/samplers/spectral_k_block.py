"""Reusable exact block heat-bath sampling for finite Ising models.

The partition groups spin variables, not complete Ising states. A transition
enumerates only the selected bounded block. This preserves the Ising target
without requiring full-state enumeration at `n=32`. The Rust and Python paths
share one explicit 64-bit random stream so parity differences stay observable.

Spec refs: REQ-SAMPLER-6612,
SCENARIO-SAMPLER-6612-REUSABLE-PARTITION-AND-TRANSITION,
REQ-RUSTPY-6612, SCENARIO-RUSTPY-6612-MATCHED-CHAIN-PARITY.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from math import exp
import time
from typing import Any

import numpy as np


LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
UINT64_MASK = (1 << 64) - 1
MAX_BLOCK_SIZE = 16


@dataclass(frozen=True)
class SpinPartition:
    """One complete spin partition with measured construction cost."""

    kind: str
    blocks: tuple[tuple[int, ...], ...]
    setup_time_s: float
    seed: int | None
    sha256: str
    source: str

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self.blocks)


@dataclass(frozen=True)
class BlockChainResult:
    """Retained spins and counters needed to charge one complete chain."""

    samples: np.ndarray = field(compare=False, repr=False)
    final_state: np.ndarray = field(compare=False, repr=False)
    sample_sha256: str
    rng_algorithm: str
    rng_initial_state: int
    rng_final_state: int
    transitions: int
    spins_updated: int
    sample_time_s: float
    engine: str


class _Lcg64:
    """Small cross-language RNG with a fully specified 53-bit float mapping."""

    def __init__(self, state: int) -> None:
        if not 0 <= int(state) <= UINT64_MASK:
            raise ValueError("seed must fit in unsigned 64 bits")
        self.state = int(state)

    def uniform(self) -> float:
        self.state = (self.state * LCG_A + LCG_C) & UINT64_MASK
        return float(self.state >> 11) * (1.0 / float(1 << 53))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sample_hash(samples: np.ndarray) -> str:
    return (
        "sha256:"
        + hashlib.sha256(np.asarray(samples, dtype=np.int8, order="C").tobytes()).hexdigest()
    )


def _validate_couplings(couplings: object) -> np.ndarray:
    matrix = np.asarray(couplings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("couplings must be a square non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("couplings must be finite")
    if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("couplings must be symmetric")
    return matrix


def _validate_model(
    couplings: object,
    fields: object,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _validate_couplings(couplings)
    field_values = np.asarray(fields, dtype=np.float64)
    if field_values.shape != (matrix.shape[0],):
        raise ValueError("fields length must match couplings dimension")
    if not np.all(np.isfinite(field_values)):
        raise ValueError("fields must be finite")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return matrix, field_values


def _canonical_blocks(blocks: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(sorted(int(index) for index in block)) for block in blocks)
    return tuple(sorted(normalized, key=lambda block: block[0] if block else -1))


def _validate_partition(
    blocks: Sequence[Sequence[int]], n_spins: int
) -> tuple[tuple[int, ...], ...]:
    normalized = _canonical_blocks(blocks)
    if not normalized or any(not block or len(block) > MAX_BLOCK_SIZE for block in normalized):
        raise ValueError(
            f"partition blocks must be non-empty and contain at most {MAX_BLOCK_SIZE} spins"
        )
    members = [index for block in normalized for index in block]
    if sorted(members) != list(range(n_spins)):
        raise ValueError("partition must contain every spin exactly once")
    return normalized


def build_spectral_blocks(couplings: object, block_size: int) -> SpinPartition:
    """Group nearby coupling-graph eigenvector coordinates into balanced blocks."""

    started = time.perf_counter()
    matrix = _validate_couplings(couplings)
    n_spins = matrix.shape[0]
    if not 1 <= int(block_size) <= min(n_spins, MAX_BLOCK_SIZE):
        raise ValueError(f"block_size must be in [1, {min(n_spins, MAX_BLOCK_SIZE)}]")
    adjacency = np.abs(matrix).copy()
    np.fill_diagonal(adjacency, 0.0)
    degree = np.sum(adjacency, axis=1)
    safe_root = np.sqrt(np.maximum(degree, 1.0e-15))
    laplacian = np.eye(n_spins) - adjacency / safe_root[:, None] / safe_root[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    coordinate_count = min(3, max(1, n_spins - 1))
    coordinates = eigenvectors[:, 1 : 1 + coordinate_count]
    for column in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[pivot, column] < 0.0:
            coordinates[:, column] *= -1.0
    keys: tuple[np.ndarray, ...] = tuple(
        coordinates[:, column] for column in reversed(range(coordinates.shape[1]))
    ) + (np.arange(n_spins),)
    order = np.lexsort(keys)
    blocks = _canonical_blocks(
        [order[start : start + int(block_size)].tolist() for start in range(0, n_spins, block_size)]
    )
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return SpinPartition(
        kind="spectral",
        blocks=blocks,
        setup_time_s=elapsed,
        seed=None,
        sha256=_sha256_json(blocks),
        source="normalized_absolute_coupling_laplacian_bottom_coordinates",
    )


def build_random_blocks(
    n_spins: int,
    block_size: int,
    seed: int,
    *,
    forbidden_hash: str | None = None,
) -> SpinPartition:
    """Create a seeded balanced control and reject spectral identity."""

    started = time.perf_counter()
    if int(n_spins) <= 0:
        raise ValueError("n_spins must be positive")
    if not 1 <= int(block_size) <= min(int(n_spins), MAX_BLOCK_SIZE):
        raise ValueError(f"block_size must be in [1, {min(int(n_spins), MAX_BLOCK_SIZE)}]")
    blocks: tuple[tuple[int, ...], ...] | None = None
    partition_hash = ""
    selected_seed = int(seed)
    for offset in range(128):
        candidate_seed = int(seed) + offset
        order = np.random.default_rng(candidate_seed).permutation(int(n_spins))
        candidate = _canonical_blocks(
            [
                order[start : start + int(block_size)].tolist()
                for start in range(0, int(n_spins), int(block_size))
            ]
        )
        candidate_hash = _sha256_json(candidate)
        if candidate_hash != forbidden_hash:
            blocks = candidate
            partition_hash = candidate_hash
            selected_seed = candidate_seed
            break
    if blocks is None:
        raise ValueError("random partition remained identical to forbidden partition")
    return SpinPartition(
        kind="random",
        blocks=blocks,
        setup_time_s=max(time.perf_counter() - started, 1.0e-12),
        seed=selected_seed,
        sha256=partition_hash,
        source="numpy_pcg64_seeded_permutation_control",
    )


def ising_energy(spins: object, couplings: object, fields: object) -> float:
    """Compute the shared `-0.5*s^T*J*s - h^T*s` convention."""

    matrix, field_values = _validate_model(couplings, fields, 1.0)
    state = np.asarray(spins, dtype=np.int8)
    if state.shape != (matrix.shape[0],) or np.any((state != -1) & (state != 1)):
        raise ValueError("spins must match the model and contain only -1 or +1")
    floating = state.astype(np.float64)
    return float(-0.5 * floating @ matrix @ floating - field_values @ floating)


def _conditional_log_weight(
    state: np.ndarray,
    couplings: np.ndarray,
    fields: np.ndarray,
    temperature: float,
    block: tuple[int, ...],
    assignment: int,
) -> float:
    proposed = [1.0 if (assignment >> local) & 1 else -1.0 for local in range(len(block))]
    in_block = [False] * len(state)
    for index in block:
        in_block[index] = True
    exponent = 0.0
    for local, spin_index in enumerate(block):
        outside_field = float(fields[spin_index])
        for other in range(len(state)):
            if not in_block[other]:
                outside_field += float(couplings[spin_index, other]) * float(state[other])
        exponent += proposed[local] * outside_field
        for later in range(local + 1, len(block)):
            exponent += (
                float(couplings[spin_index, block[later]]) * proposed[local] * proposed[later]
            )
    return exponent / float(temperature)


def run_python_chain(
    couplings: object,
    fields: object,
    temperature: float,
    blocks: Sequence[Sequence[int]],
    initial_state: object,
    *,
    seed: int,
    burn_in: int,
    retained_samples: int,
) -> BlockChainResult:
    """Run the exact block kernel with charged burn-in and retained steps."""

    matrix, field_values = _validate_model(couplings, fields, temperature)
    partition = _validate_partition(blocks, matrix.shape[0])
    state = np.asarray(initial_state, dtype=np.int8).copy()
    if state.shape != (matrix.shape[0],) or np.any((state != -1) & (state != 1)):
        raise ValueError("spins must match the model and contain only -1 or +1")
    if int(burn_in) < 0:
        raise ValueError("burn_in must be nonnegative")
    if int(retained_samples) <= 0:
        raise ValueError("retained_samples must be positive")
    total = int(burn_in) + int(retained_samples)
    rng = _Lcg64(int(seed))
    samples = np.empty((int(retained_samples), matrix.shape[0]), dtype=np.int8)
    spins_updated = 0
    started = time.perf_counter()
    for transition in range(total):
        block_index = min(int(rng.uniform() * len(partition)), len(partition) - 1)
        draw = rng.uniform()
        block = partition[block_index]
        log_weights = [
            _conditional_log_weight(
                state, matrix, field_values, float(temperature), block, assignment
            )
            for assignment in range(1 << len(block))
        ]
        maximum = max(log_weights)
        weights = [exp(value - maximum) for value in log_weights]
        threshold = draw * sum(weights)
        cumulative = 0.0
        selected = len(weights) - 1
        for assignment, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                selected = assignment
                break
        for local, spin_index in enumerate(block):
            state[spin_index] = 1 if (selected >> local) & 1 else -1
        spins_updated += len(block)
        if transition >= int(burn_in):
            samples[transition - int(burn_in)] = state
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return BlockChainResult(
        samples=samples,
        final_state=state,
        sample_sha256=_sample_hash(samples),
        rng_algorithm="lcg64_pcg_constants_top53_uniform_v1",
        rng_initial_state=int(seed),
        rng_final_state=rng.state,
        transitions=total,
        spins_updated=spins_updated,
        sample_time_s=elapsed,
        engine="python",
    )


def run_rust_chain(
    couplings: object,
    fields: object,
    temperature: float,
    blocks: Sequence[Sequence[int]],
    initial_state: object,
    *,
    seed: int,
    burn_in: int,
    retained_samples: int,
    rust_module_loader: Callable[[], Any] | None = None,
) -> BlockChainResult:
    """Run the PyO3 kernel and surface missing bindings without fallback."""

    matrix, field_values = _validate_model(couplings, fields, temperature)
    partition = _validate_partition(blocks, matrix.shape[0])
    state = np.asarray(initial_state, dtype=np.int8)
    if state.shape != (matrix.shape[0],) or np.any((state != -1) & (state != 1)):
        raise ValueError("spins must match the model and contain only -1 or +1")
    try:
        rust = (
            rust_module_loader()
            if rust_module_loader is not None
            else importlib.import_module("carnot._rust")
        )
        binding = rust.spectral_k_block
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Rust spectral k-block binding unavailable") from error
    started = time.perf_counter()
    config = binding.RustSpectralKBlockConfig(
        matrix.tolist(),
        field_values.tolist(),
        float(temperature),
        [list(block) for block in partition],
    )
    core = binding.RustSpectralKBlockCore(config)
    rust_state = binding.RustSpectralKBlockState(state.tolist(), int(seed), 0, 0)
    result = core.run_chain(rust_state, int(burn_in), int(retained_samples))
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    samples = np.asarray(result["samples"], dtype=np.int8)
    final = result["final_state"]
    return BlockChainResult(
        samples=samples,
        final_state=np.asarray(final["spins"], dtype=np.int8),
        sample_sha256=_sample_hash(samples),
        rng_algorithm="lcg64_pcg_constants_top53_uniform_v1",
        rng_initial_state=int(seed),
        rng_final_state=int(final["rng_state"]),
        transitions=int(result["transitions"]),
        spins_updated=int(result["spins_updated"]),
        sample_time_s=elapsed,
        engine="rust_pyo3",
    )


@dataclass
class SpectralKBlockBackend:
    """SamplerBackend adapter for the reusable Python or Rust block kernel."""

    seed: int = 6612
    engine: str = "rust"
    block_size: int = 4
    rust_module_loader: Callable[[], Any] | None = None
    last_result: BlockChainResult | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return f"spectral_k_block_{self.engine}"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        config = {
            "temperature": 1.0 / float(beta),
            "burn_in": int(n_steps),
            "block_size": self.block_size,
        }
        return self.sample(biases, couplings, n_samples, config)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        temperature = float(config.get("temperature", 1.0))
        matrix, fields = _validate_model(couplings, biases, temperature)
        blocks = config.get("blocks")
        if blocks is None:
            blocks = build_spectral_blocks(
                matrix, int(config.get("block_size", self.block_size))
            ).blocks
        initial = np.asarray(
            config.get("initial_state", np.ones(matrix.shape[0], dtype=np.int8)),
            dtype=np.int8,
        )
        kwargs = {
            "seed": int(config.get("seed", self.seed)),
            "burn_in": int(config.get("burn_in", 0)),
            "retained_samples": int(n_samples),
        }
        if self.engine == "python":
            result = run_python_chain(matrix, fields, temperature, blocks, initial, **kwargs)
        elif self.engine == "rust":
            result = run_rust_chain(
                matrix,
                fields,
                temperature,
                blocks,
                initial,
                rust_module_loader=self.rust_module_loader,
                **kwargs,
            )
        else:
            raise ValueError("engine must be 'python' or 'rust'")
        self.last_result = result
        return np.asarray(result.samples > 0, dtype=np.bool_)

    def set_constraints(self, constraints: Any) -> None:
        """Keep primal-dual hooks explicit; this Ising kernel has none."""

        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """Keep primal-dual hooks explicit; this Ising kernel has none."""

        return None
