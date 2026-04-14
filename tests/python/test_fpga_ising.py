"""Tests for the FPGA Ising sampler backend.

Spec coverage: REQ-SAMPLE-003, REQ-SAMPLE-005, REQ-SAMPLE-006,
               SCENARIO-SAMPLE-009, SCENARIO-SAMPLE-010,
               SCENARIO-SAMPLE-011
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from carnot.samplers.backend import get_backend
from carnot.samplers.fpga_ising import (
    AXILiteRegisterMap,
    FPGAArchitecture,
    FPGAIsingSampler,
    SoftwareFPGAOverlay,
    benchmark_fpga_sampler,
    compile_sparse_problem,
    default_overlay_factory,
    unpack_sample_words,
)


def _ferromagnetic_problem(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Create a small low-temperature Ising problem with an all-ones ground state."""
    biases = np.ones(n, dtype=np.float32) * 1.5
    couplings = np.ones((n, n), dtype=np.float32) * 0.4
    np.fill_diagonal(couplings, 0.0)
    return biases, couplings


class TestAXILiteRegisterMap:
    """REQ-SAMPLE-005: AXI-Lite windows cover the control plane and buffers."""

    def test_offsets_are_stable(self) -> None:
        """SCENARIO-SAMPLE-009: Control, bias, edge, and sample windows are addressable."""
        regmap = AXILiteRegisterMap()
        assert regmap.CONTROL == 0x0000
        assert regmap.STATUS == 0x0004
        assert FPGAArchitecture().tile_count == 32
        assert regmap.bias_offset(0) == 0x1000
        assert regmap.row_ptr_offset(3) == 0x200C
        assert regmap.edge_offset(2) == 0x4008
        assert regmap.sample_offset(sample_index=1, word_index=0, words_per_sample=2) == 0x8018


class TestSparseCompilation:
    """REQ-SAMPLE-005: Runtime couplings are compiled into sparse upload buffers."""

    def test_compile_round_trip_recovers_biases_and_couplings(self) -> None:
        """SCENARIO-SAMPLE-009: Sparse upload keeps the problem contents intact."""
        biases = np.array([0.5, -0.25, 0.0], dtype=np.float32)
        couplings = np.array(
            [
                [0.0, 0.75, 0.0],
                [0.75, 0.0, -0.5],
                [0.0, -0.5, 0.0],
            ],
            dtype=np.float32,
        )

        compiled = compile_sparse_problem(biases, couplings)

        np.testing.assert_allclose(compiled.dequantized_biases(), biases, atol=1 / 256)
        np.testing.assert_allclose(compiled.to_dense_couplings(), couplings, atol=1 / 256)
        assert compiled.row_ptr.tolist() == [0, 1, 3, 4]

    def test_compile_rejects_degree_overflow(self) -> None:
        """SCENARIO-SAMPLE-009: Per-spin sparsity limits are enforced."""
        biases = np.zeros(3, dtype=np.float32)
        couplings = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        with pytest.raises(ValueError, match="max_degree"):
            compile_sparse_problem(
                biases,
                couplings,
                architecture=FPGAArchitecture(max_degree=1),
            )

    def test_compile_rejects_bad_shapes_and_oversized_arrays(self) -> None:
        """SCENARIO-SAMPLE-009: Upload compilation validates array shape limits."""
        with pytest.raises(ValueError, match="square"):
            compile_sparse_problem(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
            )

        with pytest.raises(ValueError, match="max_spins"):
            compile_sparse_problem(
                np.zeros(5, dtype=np.float32),
                np.zeros((5, 5), dtype=np.float32),
                architecture=FPGAArchitecture(max_spins=4),
            )


class TestDefaultOverlayFactory:
    """REQ-SAMPLE-006: Overlay loading is optional and safe in non-FPGA environments."""

    def test_returns_none_without_bitfile(self) -> None:
        """SCENARIO-SAMPLE-010: Missing bitfile means no hardware transport."""
        assert default_overlay_factory(None) is None

    def test_returns_none_when_bitfile_path_is_missing(self) -> None:
        """SCENARIO-SAMPLE-010: Non-existent bitfiles are ignored safely."""
        assert default_overlay_factory("/tmp/definitely-missing-carnot.bit") is None

    def test_returns_none_when_pynq_is_missing(self, monkeypatch, tmp_path) -> None:
        """SCENARIO-SAMPLE-010: Auto-detect tolerates missing PYNQ installs."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")

        def _raise(_name: str) -> types.SimpleNamespace:
            raise ImportError("no pynq")

        monkeypatch.setattr("carnot.samplers.fpga_ising.importlib.import_module", _raise)
        assert default_overlay_factory(str(bitfile)) is None

    def test_returns_mmio_when_overlay_is_available(self, monkeypatch, tmp_path) -> None:
        """SCENARIO-SAMPLE-010: Hardware mode can bind to a PYNQ MMIO object."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        mmio = object()

        class FakeOverlay:
            def __init__(self, _path: str, download: bool = True) -> None:
                self.carnot_ising_0 = types.SimpleNamespace(mmio=mmio)

        monkeypatch.setattr(
            "carnot.samplers.fpga_ising.importlib.import_module",
            lambda _name: types.SimpleNamespace(Overlay=FakeOverlay),
        )

        assert default_overlay_factory(str(bitfile)) is mmio


class TestFPGAIsingSampler:
    """REQ-SAMPLE-006: FPGA backend supports software mode and CPU fallback."""

    def test_software_mode_uploads_and_reads_back_samples(self) -> None:
        """SCENARIO-SAMPLE-009: Software model executes the full register path."""
        biases, couplings = _ferromagnetic_problem(8)
        sampler = FPGAIsingSampler(mode="software", seed=7)

        samples = sampler.minimize_energy(
            biases,
            couplings,
            n_samples=3,
            n_steps=40,
            beta=8.0,
        )

        assert sampler.backend_name == "fpga_sim"
        assert isinstance(sampler.transport, SoftwareFPGAOverlay)
        assert samples.shape == (3, 8)
        assert samples.dtype == bool
        assert sampler.last_upload is not None
        assert sampler.transport.compiled_problem is not None
        np.testing.assert_allclose(
            sampler.last_upload.to_dense_couplings(),
            couplings,
            atol=1 / 256,
        )

        words = [
            sampler.transport.read(
                sampler.register_map.sample_offset(
                    sample_index=0,
                    word_index=word_index,
                    words_per_sample=1,
                )
            )
            for word_index in range(1)
        ]
        unpacked = unpack_sample_words(words, n_spins=8)
        np.testing.assert_array_equal(unpacked, samples[0])

    def test_software_overlay_reset_and_clear_results(self) -> None:
        """SCENARIO-SAMPLE-009: Control bits clear result RAM and reset uploaded state."""
        regmap = AXILiteRegisterMap()
        overlay = SoftwareFPGAOverlay(register_map=regmap)
        overlay.write(regmap.SAMPLE_BASE, 123)
        overlay.write(regmap.CONTROL, regmap.CONTROL_CLEAR_RESULTS)
        assert overlay.read(regmap.SAMPLE_BASE) == 0
        overlay.write(regmap.CONTROL, regmap.CONTROL_RESET)
        assert overlay.compiled_problem is None
        assert overlay.read(regmap.STATUS) == regmap.STATUS_READY

    def test_software_mode_sample_uses_fixed_temperature_defaults(self) -> None:
        """SCENARIO-SAMPLE-009: Software sampling uses the fixed-temperature path."""
        biases, couplings = _ferromagnetic_problem(7)
        sampler = FPGAIsingSampler(mode="software", seed=11)

        samples = sampler.sample(
            biases,
            couplings,
            n_samples=2,
            config={"beta": 4.0},
        )

        assert samples.shape == (2, 7)
        assert samples.dtype == bool

    def test_auto_mode_falls_back_to_cpu(self) -> None:
        """SCENARIO-SAMPLE-010: Missing overlay leaves the backend callable."""
        biases, couplings = _ferromagnetic_problem(6)
        sampler = FPGAIsingSampler(mode="auto", overlay_factory=lambda *_args, **_kwargs: None)

        samples = sampler.sample(
            biases,
            couplings,
            n_samples=4,
            config={"beta": 6.0, "n_warmup": 20},
        )

        assert sampler.backend_name == "cpu_fallback"
        assert sampler.using_cpu_fallback is True
        assert samples.shape == (4, 6)
        assert samples.dtype == bool

    def test_cpu_mode_uses_explicit_fallback_backend(self) -> None:
        """SCENARIO-SAMPLE-010: Explicit CPU mode remains sampler-compatible."""
        biases, couplings = _ferromagnetic_problem(5)
        sampler = FPGAIsingSampler(mode="cpu", seed=5)

        compiled = sampler.upload_problem(biases, couplings)
        samples = sampler.minimize_energy(
            biases,
            couplings,
            n_samples=2,
            n_steps=10,
            beta=3.0,
        )

        assert sampler.backend_name == "cpu_fallback"
        assert sampler.using_cpu_fallback is True
        assert compiled.n_spins == 5
        assert samples.shape == (2, 5)

    def test_hardware_mode_without_fallback_raises(self) -> None:
        """SCENARIO-SAMPLE-010: Strict hardware mode errors when no transport exists."""
        with pytest.raises(RuntimeError, match="FPGA overlay"):
            FPGAIsingSampler(
                mode="hardware",
                allow_cpu_fallback=False,
                overlay_factory=lambda *_args, **_kwargs: None,
            )

    def test_invalid_mode_raises(self) -> None:
        """SCENARIO-SAMPLE-010: Unsupported backend modes are rejected explicitly."""
        with pytest.raises(ValueError, match="mode"):
            FPGAIsingSampler(mode="quantum")

    def test_sampler_factory_exposes_fpga_backend(self) -> None:
        """REQ-SAMPLE-003, REQ-SAMPLE-006, REQ-SAMPLE-009: get_backend('fpga') resolves.

        As of Exp 289, get_backend('fpga') returns FpgaBackend (quantum-inspired
        sparse Ising with log-linear schedule), which wraps FPGAIsingSampler
        internally when CARNOT_KV260_BITFILE is set.
        """
        from carnot.samplers.fpga_backend import FpgaBackend

        backend = get_backend("fpga")
        assert isinstance(backend, FpgaBackend)

    def test_hardware_mode_reports_fpga_when_transport_exists(self) -> None:
        """SCENARIO-SAMPLE-010: A live transport keeps the backend on the FPGA path."""
        sampler = FPGAIsingSampler(
            mode="hardware",
            overlay_factory=lambda *_args, **_kwargs: SoftwareFPGAOverlay(),
        )
        assert sampler.backend_name == "fpga"
        assert sampler.using_cpu_fallback is False

    def test_transport_errors_when_results_are_not_ready(self) -> None:
        """SCENARIO-SAMPLE-010: Non-completing hardware runs raise a clear error."""

        class IdleTransport:
            def write(self, offset: int, value: int) -> None:
                self.offset = offset
                self.value = value

            def read(self, offset: int) -> int:
                return 0

        biases, couplings = _ferromagnetic_problem(4)
        sampler = FPGAIsingSampler(
            mode="hardware",
            overlay_factory=lambda *_args, **_kwargs: IdleTransport(),
        )

        with pytest.raises(RuntimeError, match="did not complete"):
            sampler.sample(biases, couplings, n_samples=1, config={"beta": 2.0})

    def test_run_transport_requires_active_transport(self) -> None:
        """SCENARIO-SAMPLE-010: Internal transport runner rejects missing MMIO."""
        biases, couplings = _ferromagnetic_problem(4)
        sampler = FPGAIsingSampler(mode="cpu")
        compiled = compile_sparse_problem(biases, couplings)

        with pytest.raises(RuntimeError, match="not active"):
            sampler._run_transport(  # noqa: SLF001 - deliberate branch coverage
                compiled=compiled,
                n_samples=1,
                warmup_steps=1,
                steps_per_sample=1,
                beta=1.0,
                minimize=False,
            )


class TestBenchmarkHelper:
    """REQ-SAMPLE-006: Benchmark helper returns comparable software and CPU metrics."""

    def test_benchmark_contract(self) -> None:
        """SCENARIO-SAMPLE-011: Benchmark result names both backends and the sample shape."""
        biases, couplings = _ferromagnetic_problem(10)

        result = benchmark_fpga_sampler(
            biases,
            couplings,
            n_samples=4,
            n_steps=25,
            beta=5.0,
            seed=3,
        )

        assert result["fpga_backend"] == "fpga_sim"
        assert result["cpu_backend"] == "cpu"
        assert result["n_spins"] == 10
        assert result["sample_shape"] == [4, 10]
        assert result["fpga_seconds"] >= 0.0
        assert result["cpu_seconds"] >= 0.0

    def test_benchmark_warns_when_shapes_diverge(self, caplog, monkeypatch) -> None:
        """SCENARIO-SAMPLE-011: Benchmark emits a warning on shape mismatches."""
        biases, couplings = _ferromagnetic_problem(3)

        monkeypatch.setattr(
            "carnot.samplers.fpga_ising.FPGAIsingSampler.minimize_energy",
            lambda self, *_args, **_kwargs: np.zeros((1, 3), dtype=bool),
        )
        monkeypatch.setattr(
            "carnot.samplers.fpga_ising.CpuBackend.minimize_energy",
            lambda self, *_args, **_kwargs: np.zeros((2, 3), dtype=bool),
        )

        benchmark_fpga_sampler(
            biases,
            couplings,
            n_samples=2,
            n_steps=3,
            beta=1.0,
        )

        assert "sample shapes differ" in caplog.text
