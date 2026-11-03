"""FPGA Ising sampler backend with a software-mode control-plane model.

**Researcher summary:**
    Provides a KV260-oriented Ising sampler design surface in Python first:
    sparse runtime coupling upload, AXI-Lite control/status windows, sample
    trigger/readback, and safe CPU fallback when no FPGA overlay is present.

**Detailed explanation for engineers:**
    The real FPGA path will eventually drive a PYNQ overlay on the Kria KV260.
    Until the bitstream exists, this module models the same MMIO contract in
    software so tests can validate the control plane and buffer formats. The
    software overlay stores quantized biases and sparse couplings in AXI-Lite
    windows, runs the existing CPU sampler when ``START`` is written, and
    exposes packed sample words for readback exactly as hardware would.

Spec: REQ-SAMPLE-005, REQ-SAMPLE-006
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from carnot.samplers.backend import CpuBackend

logger = logging.getLogger(__name__)


def _signed_from_u16(word: int) -> int:
    value = word & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _quantize_word(value: float, frac_bits: int) -> int:
    scaled = int(round(float(value) * (1 << frac_bits)))
    clipped = min(max(scaled, -(1 << 15)), (1 << 15) - 1)
    return clipped & 0xFFFF


def _dequantize_word(word: int, frac_bits: int) -> float:
    return _signed_from_u16(word) / float(1 << frac_bits)


def _pack_edge_word(neighbor: int, weight: float, frac_bits: int) -> int:
    return ((_quantize_word(weight, frac_bits) & 0xFFFF) << 16) | (neighbor & 0x0FFF)


def _unpack_edge_word(word: int, frac_bits: int) -> tuple[int, float]:
    neighbor = word & 0x0FFF
    weight = _dequantize_word((word >> 16) & 0xFFFF, frac_bits)
    return neighbor, weight


def _pack_sample_words(sample: np.ndarray) -> list[int]:
    words: list[int] = []
    flat = np.asarray(sample, dtype=bool)
    for start in range(0, flat.shape[0], 32):
        word = 0
        for bit_index, value in enumerate(flat[start : start + 32]):
            if bool(value):
                word |= 1 << bit_index
        words.append(word)
    return words


def unpack_sample_words(words: list[int], n_spins: int) -> np.ndarray:
    """Expand packed 32-bit sample words into a boolean spin vector.

    Spec: REQ-SAMPLE-006
    """
    bits = np.zeros(n_spins, dtype=bool)
    for word_index, word in enumerate(words):
        base = word_index * 32
        for bit_index in range(32):
            spin_index = base + bit_index
            if spin_index >= n_spins:
                break
            bits[spin_index] = bool((word >> bit_index) & 0x1)
    return bits


class RegisterIO(Protocol):
    """Minimal MMIO contract shared by real and simulated FPGA transports."""

    def write(self, offset: int, value: int) -> None: ...

    def read(self, offset: int) -> int: ...


@dataclass(frozen=True)
class FPGAArchitecture:
    """Static design target for the KV260-class p-bit array.

    Spec: REQ-SAMPLE-005
    """

    max_spins: int = 4096
    tile_size: int = 128
    max_degree: int = 32
    frac_bits: int = 8

    @property
    def tile_count(self) -> int:
        return self.max_spins // self.tile_size


@dataclass(frozen=True)
class AXILiteRegisterMap:
    """AXI-Lite register and buffer layout for runtime Ising uploads.

    Spec: REQ-SAMPLE-005
    """

    CONTROL: int = 0x0000
    STATUS: int = 0x0004
    SPIN_COUNT: int = 0x0008
    SAMPLE_COUNT: int = 0x000C
    WARMUP_STEPS: int = 0x0010
    STEPS_PER_SAMPLE: int = 0x0014
    BETA_INIT: int = 0x0018
    BETA_FINAL: int = 0x001C
    RUN_FLAGS: int = 0x0020

    CONTROL_START: int = 0x0001
    CONTROL_RESET: int = 0x0002
    CONTROL_CLEAR_RESULTS: int = 0x0004

    STATUS_READY: int = 0x0001
    STATUS_BUSY: int = 0x0002
    STATUS_DONE: int = 0x0004
    STATUS_ERROR: int = 0x0008

    RUN_MINIMIZE: int = 0x0001

    BIAS_BASE: int = 0x1000
    ROW_PTR_BASE: int = 0x2000
    EDGE_BASE: int = 0x4000
    SAMPLE_BASE: int = 0x8010

    def bias_offset(self, index: int) -> int:
        return self.BIAS_BASE + 4 * index

    def row_ptr_offset(self, index: int) -> int:
        return self.ROW_PTR_BASE + 4 * index

    def edge_offset(self, index: int) -> int:
        return self.EDGE_BASE + 4 * index

    def sample_offset(self, sample_index: int, word_index: int, words_per_sample: int) -> int:
        return self.SAMPLE_BASE + 4 * ((sample_index * words_per_sample) + word_index)


@dataclass(frozen=True)
class CompiledIsingProblem:
    """Sparse, quantized upload buffers for the FPGA control plane.

    Spec: REQ-SAMPLE-005
    """

    n_spins: int
    bias_words: np.ndarray
    row_ptr: np.ndarray
    edge_words: np.ndarray
    architecture: FPGAArchitecture

    def dequantized_biases(self) -> np.ndarray:
        values = [
            _dequantize_word(int(word), self.architecture.frac_bits) for word in self.bias_words
        ]
        return np.asarray(values, dtype=np.float32)

    def to_dense_couplings(self) -> np.ndarray:
        dense = np.zeros((self.n_spins, self.n_spins), dtype=np.float32)
        for row_index in range(self.n_spins):
            start = int(self.row_ptr[row_index])
            stop = int(self.row_ptr[row_index + 1])
            for edge_word in self.edge_words[start:stop]:
                neighbor, weight = _unpack_edge_word(int(edge_word), self.architecture.frac_bits)
                dense[row_index, neighbor] = weight
        return dense


def compile_sparse_problem(
    biases: np.ndarray,
    couplings: np.ndarray,
    architecture: FPGAArchitecture | None = None,
) -> CompiledIsingProblem:
    """Compile a dense Ising problem into the sparse FPGA upload format.

    Spec: REQ-SAMPLE-005
    """
    arch = architecture or FPGAArchitecture()
    b = np.asarray(biases, dtype=np.float32)
    couplings_array = np.asarray(couplings, dtype=np.float32)

    if b.ndim != 1 or couplings_array.shape != (b.shape[0], b.shape[0]):
        raise ValueError("biases must be 1-D and couplings must be square")

    n_spins = int(b.shape[0])
    if n_spins > arch.max_spins:
        raise ValueError(f"n_spins={n_spins} exceeds max_spins={arch.max_spins}")

    couplings_array = couplings_array.copy()
    np.fill_diagonal(couplings_array, 0.0)

    bias_words = np.asarray([_quantize_word(value, arch.frac_bits) for value in b], dtype=np.uint32)
    row_ptr: list[int] = [0]
    edge_words: list[int] = []
    for row in range(n_spins):
        neighbors = np.nonzero(couplings_array[row])[0]
        if int(neighbors.shape[0]) > arch.max_degree:
            raise ValueError(f"row {row} exceeds max_degree={arch.max_degree}")
        for neighbor in neighbors:
            weight = float(couplings_array[row, neighbor])
            edge_words.append(_pack_edge_word(int(neighbor), weight, arch.frac_bits))
        row_ptr.append(len(edge_words))

    return CompiledIsingProblem(
        n_spins=n_spins,
        bias_words=bias_words,
        row_ptr=np.asarray(row_ptr, dtype=np.uint32),
        edge_words=np.asarray(edge_words, dtype=np.uint32),
        architecture=arch,
    )


def default_overlay_factory(bitfile_path: str | None) -> RegisterIO | None:
    """Load the default PYNQ MMIO transport when a bitfile is available.

    Spec: REQ-SAMPLE-006
    """
    if bitfile_path is None:
        return None

    bitfile = Path(bitfile_path)
    if not bitfile.exists():
        return None

    try:
        pynq = importlib.import_module("pynq")
    except ImportError:
        return None

    overlay = pynq.Overlay(str(bitfile), download=True)
    return getattr(getattr(overlay, "carnot_ising_0", None), "mmio", None)


@dataclass
class SoftwareFPGAOverlay:
    """Software model of the AXI-Lite Ising overlay.

    Spec: REQ-SAMPLE-006
    """

    architecture: FPGAArchitecture = field(default_factory=FPGAArchitecture)
    register_map: AXILiteRegisterMap = field(default_factory=AXILiteRegisterMap)
    seed: int = 42
    _memory: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _compiled_problem: CompiledIsingProblem | None = field(default=None, init=False, repr=False)
    _cpu_backend: CpuBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cpu_backend = CpuBackend(seed=self.seed)
        self._memory[self.register_map.STATUS] = self.register_map.STATUS_READY

    @property
    def compiled_problem(self) -> CompiledIsingProblem | None:
        return self._compiled_problem

    def write(self, offset: int, value: int) -> None:
        self._memory[offset] = int(value) & 0xFFFFFFFF
        if offset != self.register_map.CONTROL:
            return

        if value & self.register_map.CONTROL_RESET:
            self._memory.clear()
            self._memory[self.register_map.STATUS] = self.register_map.STATUS_READY
            self._compiled_problem = None
            return

        if value & self.register_map.CONTROL_CLEAR_RESULTS:
            self._clear_results()

        if value & self.register_map.CONTROL_START:
            self._run()

    def read(self, offset: int) -> int:
        return self._memory.get(offset, 0)

    def _clear_results(self) -> None:
        for address in list(self._memory):
            if address >= self.register_map.SAMPLE_BASE:
                del self._memory[address]

    def _run(self) -> None:
        regmap = self.register_map
        self._memory[regmap.STATUS] = regmap.STATUS_BUSY

        n_spins = int(self._memory.get(regmap.SPIN_COUNT, 0))
        n_samples = int(self._memory.get(regmap.SAMPLE_COUNT, 0))
        warmup_steps = int(self._memory.get(regmap.WARMUP_STEPS, 0))
        steps_per_sample = int(self._memory.get(regmap.STEPS_PER_SAMPLE, 20))
        beta_final = _dequantize_word(
            int(self._memory.get(regmap.BETA_FINAL, 0)),
            self.architecture.frac_bits,
        )
        run_flags = int(self._memory.get(regmap.RUN_FLAGS, 0))

        bias_words = [self._memory.get(regmap.bias_offset(index), 0) for index in range(n_spins)]
        row_ptr = [
            self._memory.get(regmap.row_ptr_offset(index), 0) for index in range(n_spins + 1)
        ]
        edge_count = int(row_ptr[-1]) if row_ptr else 0
        edge_words = [self._memory.get(regmap.edge_offset(index), 0) for index in range(edge_count)]
        self._compiled_problem = CompiledIsingProblem(
            n_spins=n_spins,
            bias_words=np.asarray(bias_words, dtype=np.uint32),
            row_ptr=np.asarray(row_ptr, dtype=np.uint32),
            edge_words=np.asarray(edge_words, dtype=np.uint32),
            architecture=self.architecture,
        )

        biases = self._compiled_problem.dequantized_biases()
        couplings = self._compiled_problem.to_dense_couplings()
        if run_flags & regmap.RUN_MINIMIZE:
            samples = self._cpu_backend.minimize_energy(
                biases=biases,
                couplings=couplings,
                n_samples=n_samples,
                n_steps=warmup_steps,
                beta=beta_final,
            )
        else:
            samples = self._cpu_backend.sample(
                biases=biases,
                couplings=couplings,
                n_samples=n_samples,
                config={
                    "beta": beta_final,
                    "n_warmup": warmup_steps,
                    "steps_per_sample": steps_per_sample,
                },
            )

        words_per_sample = max(1, (n_spins + 31) // 32)
        for sample_index, sample in enumerate(samples):
            for word_index, word in enumerate(_pack_sample_words(sample)):
                address = regmap.sample_offset(sample_index, word_index, words_per_sample)
                self._memory[address] = word
        self._memory[regmap.STATUS] = regmap.STATUS_READY | regmap.STATUS_DONE


OverlayFactory = Callable[[str | None], RegisterIO | None]


@dataclass
class FPGAIsingSampler:
    """Sampler backend that targets a real or simulated FPGA control plane.

    Spec: REQ-SAMPLE-006
    """

    seed: int = 42
    bitfile_path: str | None = None
    mode: Literal["auto", "hardware", "software", "cpu"] = "auto"
    allow_cpu_fallback: bool = True
    architecture: FPGAArchitecture = field(default_factory=FPGAArchitecture)
    register_map: AXILiteRegisterMap = field(default_factory=AXILiteRegisterMap)
    overlay_factory: OverlayFactory | None = None
    transport: RegisterIO | None = field(init=False, default=None)
    last_upload: CompiledIsingProblem | None = field(init=False, default=None)
    using_cpu_fallback: bool = field(init=False, default=False)
    _backend_name: str = field(init=False, default="cpu_fallback")
    _cpu_backend: CpuBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cpu_backend = CpuBackend(seed=self.seed)
        self._resolve_transport()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _resolve_transport(self) -> None:
        if self.mode == "software":
            self.transport = SoftwareFPGAOverlay(
                architecture=self.architecture,
                register_map=self.register_map,
                seed=self.seed,
            )
            self.using_cpu_fallback = False
            self._backend_name = "fpga_sim"
            return

        if self.mode == "cpu":
            self.transport = None
            self.using_cpu_fallback = True
            self._backend_name = "cpu_fallback"
            return

        if self.mode not in {"auto", "hardware"}:
            raise ValueError(f"Unsupported FPGA sampler mode {self.mode!r}")

        factory = self.overlay_factory or default_overlay_factory
        self.transport = factory(self.bitfile_path)
        if self.transport is not None:
            self.using_cpu_fallback = False
            self._backend_name = "fpga"
            return

        if self.mode == "hardware" and not self.allow_cpu_fallback:
            raise RuntimeError("FPGA overlay unavailable for hardware mode")

        self.using_cpu_fallback = True
        self._backend_name = "cpu_fallback"

    def upload_problem(self, biases: np.ndarray, couplings: np.ndarray) -> CompiledIsingProblem:
        """Compile and write biases/couplings into the active transport windows.

        Spec: REQ-SAMPLE-006
        """
        compiled = compile_sparse_problem(biases, couplings, architecture=self.architecture)
        self.last_upload = compiled
        if self.transport is None:
            return compiled

        for index, word in enumerate(compiled.bias_words):
            self.transport.write(self.register_map.bias_offset(index), int(word))
        for index, word in enumerate(compiled.row_ptr):
            self.transport.write(self.register_map.row_ptr_offset(index), int(word))
        for index, word in enumerate(compiled.edge_words):
            self.transport.write(self.register_map.edge_offset(index), int(word))
        return compiled

    def _run_transport(
        self,
        compiled: CompiledIsingProblem,
        n_samples: int,
        warmup_steps: int,
        steps_per_sample: int,
        beta: float,
        minimize: bool,
    ) -> np.ndarray:
        if self.transport is None:
            raise RuntimeError("FPGA transport is not active")

        regmap = self.register_map
        self.transport.write(regmap.SPIN_COUNT, compiled.n_spins)
        self.transport.write(regmap.SAMPLE_COUNT, n_samples)
        self.transport.write(regmap.WARMUP_STEPS, warmup_steps)
        self.transport.write(regmap.STEPS_PER_SAMPLE, steps_per_sample)
        self.transport.write(regmap.BETA_INIT, _quantize_word(beta, self.architecture.frac_bits))
        self.transport.write(regmap.BETA_FINAL, _quantize_word(beta, self.architecture.frac_bits))
        self.transport.write(regmap.RUN_FLAGS, regmap.RUN_MINIMIZE if minimize else 0)
        self.transport.write(
            regmap.CONTROL,
            regmap.CONTROL_CLEAR_RESULTS | regmap.CONTROL_START,
        )

        status = self.transport.read(regmap.STATUS)
        if not status & regmap.STATUS_DONE:
            raise RuntimeError("FPGA sample run did not complete")

        words_per_sample = max(1, (compiled.n_spins + 31) // 32)
        sample_rows: list[np.ndarray] = []
        for sample_index in range(n_samples):
            words = [
                self.transport.read(
                    regmap.sample_offset(sample_index, word_index, words_per_sample)
                )
                for word_index in range(words_per_sample)
            ]
            sample_rows.append(unpack_sample_words(words, n_spins=compiled.n_spins))
        return np.asarray(sample_rows, dtype=bool)

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run annealing on FPGA, software model, or CPU fallback.

        Spec: REQ-SAMPLE-006
        """
        if self.using_cpu_fallback:
            return self._cpu_backend.minimize_energy(biases, couplings, n_samples, n_steps, beta)

        compiled = self.upload_problem(biases, couplings)
        return self._run_transport(
            compiled=compiled,
            n_samples=n_samples,
            warmup_steps=n_steps,
            steps_per_sample=20,
            beta=beta,
            minimize=True,
        )

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw fixed-temperature samples on FPGA, software model, or CPU fallback.

        Spec: REQ-SAMPLE-006
        """
        if self.using_cpu_fallback:
            return self._cpu_backend.sample(biases, couplings, n_samples, config)

        beta = float(config.get("beta", 10.0))
        warmup_steps = int(config.get("n_warmup", 500))
        steps_per_sample = int(config.get("steps_per_sample", 20))
        compiled = self.upload_problem(biases, couplings)
        return self._run_transport(
            compiled=compiled,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            steps_per_sample=steps_per_sample,
            beta=beta,
            minimize=False,
        )


def benchmark_fpga_sampler(
    biases: np.ndarray,
    couplings: np.ndarray,
    n_samples: int,
    n_steps: int,
    beta: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark the software FPGA path against the CPU backend.

    Spec: REQ-SAMPLE-006
    """
    fpga = FPGAIsingSampler(mode="software", seed=seed)
    cpu = CpuBackend(seed=seed)

    fpga_start = time.perf_counter()
    fpga_samples = fpga.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    fpga_seconds = time.perf_counter() - fpga_start

    cpu_start = time.perf_counter()
    cpu_samples = cpu.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    cpu_seconds = time.perf_counter() - cpu_start

    if fpga_samples.shape != cpu_samples.shape:
        logger.warning(
            "FPGA and CPU sample shapes differ: fpga=%s cpu=%s",
            fpga_samples.shape,
            cpu_samples.shape,
        )

    return {
        "fpga_backend": fpga.backend_name,
        "cpu_backend": cpu.backend_name,
        "n_spins": int(np.asarray(biases).shape[0]),
        "sample_shape": [int(dim) for dim in fpga_samples.shape],
        "fpga_seconds": fpga_seconds,
        "cpu_seconds": cpu_seconds,
    }
