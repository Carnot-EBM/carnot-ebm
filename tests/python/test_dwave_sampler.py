"""Tests for the D-Wave sampler backend.

Spec coverage: REQ-SAMPLE-003, REQ-SAMPLE-007

All D-Wave Ocean SDK imports (dimod, neal, tabu, dwave.system) are mocked at
sys.modules level so the tests run in CI without the dwave-ocean-sdk package
installed. The mocks replicate the dimod SampleSet and BQM contracts that the
sampler code depends on.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal fake dimod module injected into sys.modules.
# The production code does `import dimod` inside _ising_to_bqm() so every
# test that exercises that path needs dimod available.
# ---------------------------------------------------------------------------


class _FakeBQM:
    """Minimal dimod.BinaryQuadraticModel stand-in for assertions."""

    def __init__(
        self,
        linear: dict[int, float],
        quadratic: dict[tuple[int, int], float],
        vartype: Any,
    ) -> None:
        self.linear = linear
        self.quadratic = quadratic
        self.vartype = vartype


class _FakeBinary:
    """Stands in for dimod.BINARY sentinel."""


_FAKE_BINARY = _FakeBinary()


def _make_fake_dimod() -> MagicMock:
    """Build a fake dimod module with just enough surface area for dwave_sampler."""
    m = MagicMock()
    m.BinaryQuadraticModel = _FakeBQM
    m.BINARY = _FAKE_BINARY
    return m


_FAKE_DIMOD = _make_fake_dimod()


@pytest.fixture(autouse=True)
def _patch_dimod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake dimod into sys.modules for every test in this file."""
    monkeypatch.setitem(sys.modules, "dimod", _FAKE_DIMOD)


# ---------------------------------------------------------------------------
# Imports under test (after fixture defined so import-time errors are deferred)
# ---------------------------------------------------------------------------

from carnot.samplers.backend import SamplerBackend, get_backend  # noqa: E402
from carnot.samplers.dwave_sampler import (  # noqa: E402
    DWaveSampler,
    _ising_to_bqm,
    _sample_set_to_array,
    benchmark_dwave_vs_cpu,
)


# ---------------------------------------------------------------------------
# Helpers: fake sampler / sample-set objects
# ---------------------------------------------------------------------------


def _make_sample_set(
    samples: list[dict[int, int]],
    energies: list[float] | None = None,
    num_occurrences: list[int] | None = None,
    chain_break_fraction: list[float] | None = None,
) -> MagicMock:
    """Build a minimal fake dimod.SampleSet.

    _sample_set_to_array calls ss.data(['sample', 'energy', 'num_occurrences'])
    and iterates the tuples. The QPU path reads ss.record.chain_break_fraction.
    """
    n = len(samples)
    energies = energies or [0.0] * n
    num_occurrences = num_occurrences or [1] * n

    rows = list(zip(samples, energies, num_occurrences))

    def _data(fields: list[str]):  # noqa: ANN001
        yield from rows

    ss = MagicMock()
    ss.data.side_effect = _data

    if chain_break_fraction is not None:
        ss.record.chain_break_fraction = np.array(chain_break_fraction)
    else:
        # hasattr(ss.record, 'chain_break_fraction') must return False.
        del ss.record.chain_break_fraction

    return ss


def _mock_neal(ss: Any) -> MagicMock:
    s = MagicMock()
    s.sample.return_value = ss
    return s


def _mock_tabu(ss: Any) -> MagicMock:
    s = MagicMock()
    s.sample.return_value = ss
    return s


def _mock_qpu(ss: Any, n_qubits: int = 5000, n_couplers: int = 15000) -> MagicMock:
    """Fake EmbeddingComposite(DWaveSampler(...))."""
    s = MagicMock()
    s.sample.return_value = ss
    s.child.properties = {
        "qubits": list(range(n_qubits)),
        "couplers": list(range(n_couplers)),
        "chip_id": "Advantage_system4.1",
    }
    return s


def _ferro(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """All-positive ferromagnetic problem: ground state is all-ones."""
    b = np.ones(n, dtype=np.float32) * 2.0
    J = np.ones((n, n), dtype=np.float32) * 0.5
    np.fill_diagonal(J, 0.0)
    return b, J


# ---------------------------------------------------------------------------
# BQM conversion
# ---------------------------------------------------------------------------


class TestIsingToBqm:
    """REQ-SAMPLE-003: Carnot Ising → dimod BQM conversion is correct."""

    def test_linear_terms_negated(self):
        """SCENARIO-SAMPLE-007: Linear biases are negated (minimization convention)."""
        b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
        J = np.zeros((3, 3), dtype=np.float64)
        bqm = _ising_to_bqm(b, J)
        assert isinstance(bqm, _FakeBQM)
        np.testing.assert_allclose(bqm.linear[0], -1.0)
        np.testing.assert_allclose(bqm.linear[1], 2.0)
        np.testing.assert_allclose(bqm.linear[2], -0.5)

    def test_quadratic_terms_negated_and_doubled(self):
        """SCENARIO-SAMPLE-007: Q_ij = -2 * J_ij for upper triangle."""
        b = np.zeros(3, dtype=np.float64)
        J = np.zeros((3, 3), dtype=np.float64)
        J[0, 1] = J[1, 0] = 0.5
        J[0, 2] = J[2, 0] = -0.3
        bqm = _ising_to_bqm(b, J)
        np.testing.assert_allclose(bqm.quadratic[(0, 1)], -1.0)  # -2 * 0.5
        np.testing.assert_allclose(bqm.quadratic[(0, 2)], 0.6)   # -2 * -0.3

    def test_zero_couplings_not_in_quadratic(self):
        """SCENARIO-SAMPLE-007: Zero couplings are omitted from quadratic dict."""
        b = np.ones(4, dtype=np.float64)
        J = np.zeros((4, 4), dtype=np.float64)
        bqm = _ising_to_bqm(b, J)
        assert len(bqm.quadratic) == 0

    def test_vartype_is_binary(self):
        """SCENARIO-SAMPLE-007: BQM uses BINARY vartype (Carnot {0,1} spins)."""
        b = np.zeros(2, dtype=np.float64)
        J = np.zeros((2, 2), dtype=np.float64)
        bqm = _ising_to_bqm(b, J)
        assert bqm.vartype is _FAKE_BINARY


# ---------------------------------------------------------------------------
# SampleSet → array conversion
# ---------------------------------------------------------------------------


class TestSampleSetToArray:
    """REQ-SAMPLE-003: dimod SampleSet → boolean NumPy array conversion."""

    def test_basic_shape(self):
        """SCENARIO-SAMPLE-007: Correct shape when samples match n_samples."""
        ss = _make_sample_set([{0: 1, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 0}])
        arr = _sample_set_to_array(ss, n_spins=3, n_samples=2)
        assert arr.shape == (2, 3)
        assert arr.dtype == bool

    def test_values_correct(self):
        """SCENARIO-SAMPLE-007: Sample dict values mapped to correct bool positions."""
        ss = _make_sample_set([{0: 1, 1: 0, 2: 1}])
        arr = _sample_set_to_array(ss, n_spins=3, n_samples=1)
        np.testing.assert_array_equal(arr[0], [True, False, True])

    def test_occurrences_expanded(self):
        """SCENARIO-SAMPLE-007: num_occurrences > 1 repeats the row."""
        ss = _make_sample_set([{0: 1, 1: 0}], num_occurrences=[3])
        arr = _sample_set_to_array(ss, n_spins=2, n_samples=3)
        assert arr.shape == (3, 2)
        assert np.all(arr[:, 0])

    def test_padding_when_fewer_samples(self):
        """SCENARIO-SAMPLE-007: Last row repeated when fewer than n_samples returned."""
        ss = _make_sample_set([{0: 1, 1: 1}])
        arr = _sample_set_to_array(ss, n_spins=2, n_samples=5)
        assert arr.shape == (5, 2)
        assert np.all(arr)

    def test_trimmed_when_more_samples(self):
        """SCENARIO-SAMPLE-007: Extra rows trimmed to exactly n_samples."""
        ss = _make_sample_set([{0: 1, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 1}])
        arr = _sample_set_to_array(ss, n_spins=2, n_samples=2)
        assert arr.shape == (2, 2)

    def test_empty_sample_set_returns_zeros(self):
        """SCENARIO-SAMPLE-007: Empty SampleSet returns all-zero array."""
        ss = _make_sample_set([])
        arr = _sample_set_to_array(ss, n_spins=4, n_samples=3)
        assert arr.shape == (3, 4)
        assert not np.any(arr)

    def test_occurrences_stops_early(self):
        """SCENARIO-SAMPLE-007: Expansion stops at n_samples even if occurrences exceed it."""
        ss = _make_sample_set([{0: 1, 1: 0}], num_occurrences=[10])
        arr = _sample_set_to_array(ss, n_spins=2, n_samples=3)
        assert arr.shape == (3, 2)


# ---------------------------------------------------------------------------
# DWaveSampler construction
# ---------------------------------------------------------------------------


class TestDWaveSamplerConstruction:
    """REQ-SAMPLE-003: DWaveSampler builds the correct underlying sampler per mode."""

    def test_unknown_mode_raises(self):
        """SCENARIO-SAMPLE-007: Unknown mode raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown DWaveSampler mode"):
            DWaveSampler(mode="invalid")

    def test_neal_backend_name(self):
        """SCENARIO-SAMPLE-007: Neal mode reports correct backend name."""
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=MagicMock()):
            sampler = DWaveSampler(mode="neal")
        assert sampler.backend_name == "dwave_neal"

    def test_tabu_backend_name(self):
        """SCENARIO-SAMPLE-007: Tabu mode reports correct backend name."""
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=MagicMock()):
            sampler = DWaveSampler(mode="tabu")
        assert sampler.backend_name == "dwave_tabu"

    def test_qpu_backend_name(self):
        """SCENARIO-SAMPLE-007: QPU mode passes leap_token to builder."""
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler") as mock_build:
            mock_build.return_value = MagicMock()
            DWaveSampler(mode="qpu", leap_token="fake-token")
        mock_build.assert_called_once_with("fake-token")

    def test_is_sampler_backend_protocol(self):
        """SCENARIO-SAMPLE-007: DWaveSampler conforms to SamplerBackend protocol."""
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=MagicMock()):
            sampler = DWaveSampler(mode="neal")
        assert isinstance(sampler, SamplerBackend)


# ---------------------------------------------------------------------------
# minimize_energy
# ---------------------------------------------------------------------------


class TestMinimizeEnergy:
    """REQ-SAMPLE-003: minimize_energy produces correct shapes and routes to backend."""

    def test_shape_neal(self):
        """SCENARIO-SAMPLE-007: Neal minimize_energy returns (n_samples, n_spins) bool array."""
        b, J = _ferro(8)
        ss = _make_sample_set([{i: 1 for i in range(8)}] * 5)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=_mock_neal(ss)):
            sampler = DWaveSampler(mode="neal")
        result = sampler.minimize_energy(b, J, n_samples=5, n_steps=200, beta=10.0)
        assert result.shape == (5, 8)
        assert result.dtype == bool

    def test_neal_passes_beta_range(self):
        """SCENARIO-SAMPLE-007: Neal backend receives beta_range=[0.1, beta]."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}] * 3)
        underlying = _mock_neal(ss)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="neal")
        sampler.minimize_energy(b, J, n_samples=3, n_steps=100, beta=5.0)
        kw = underlying.sample.call_args[1]
        assert kw["beta_range"] == [0.1, 5.0]
        assert kw["num_sweeps"] == 100
        assert kw["num_reads"] == 3

    def test_tabu_shape(self):
        """SCENARIO-SAMPLE-007: Tabu minimize_energy returns (n_samples, n_spins) bool array."""
        b, J = _ferro(6)
        ss = _make_sample_set([{i: 1 for i in range(6)}] * 4)
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=_mock_tabu(ss)):
            sampler = DWaveSampler(mode="tabu")
        result = sampler.minimize_energy(b, J, n_samples=4, n_steps=500, beta=8.0)
        assert result.shape == (4, 6)
        assert result.dtype == bool

    def test_tabu_passes_tenure(self):
        """SCENARIO-SAMPLE-007: Tabu backend receives tenure = n_steps // 10."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}] * 2)
        underlying = _mock_tabu(ss)
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="tabu")
        sampler.minimize_energy(b, J, n_samples=2, n_steps=200, beta=5.0)
        kw = underlying.sample.call_args[1]
        assert kw["tenure"] == 20  # 200 // 10
        assert kw["num_reads"] == 2

    def test_qpu_shape(self):
        """SCENARIO-SAMPLE-007: QPU minimize_energy returns (n_samples, n_spins) bool array."""
        b, J = _ferro(4)
        ss = _make_sample_set(
            [{i: 1 for i in range(4)}] * 3,
            chain_break_fraction=[0.05, 0.02, 0.01],
        )
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=_mock_qpu(ss)):
            sampler = DWaveSampler(mode="qpu")
        result = sampler.minimize_energy(b, J, n_samples=3, n_steps=100, beta=10.0)
        assert result.shape == (3, 4)
        assert result.dtype == bool

    def test_qpu_records_chain_break_fraction(self):
        """SCENARIO-SAMPLE-007: QPU path stores mean chain_break_fraction."""
        b, J = _ferro(4)
        ss = _make_sample_set(
            [{i: 1 for i in range(4)}] * 2,
            chain_break_fraction=[0.10, 0.20],
        )
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=_mock_qpu(ss)):
            sampler = DWaveSampler(mode="qpu")
        sampler.minimize_energy(b, J, n_samples=2, n_steps=100, beta=10.0)
        np.testing.assert_allclose(sampler.last_chain_break_fraction, 0.15, atol=1e-6)

    def test_qpu_no_chain_break_attribute(self):
        """SCENARIO-SAMPLE-007: QPU path handles missing chain_break_fraction gracefully."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 1 for i in range(4)}] * 2, chain_break_fraction=None)
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=_mock_qpu(ss)):
            sampler = DWaveSampler(mode="qpu")
        result = sampler.minimize_energy(b, J, n_samples=2, n_steps=100, beta=10.0)
        assert result.shape == (2, 4)
        assert sampler.last_chain_break_fraction == 0.0

    def test_n_steps_zero_clamped_to_one(self):
        """SCENARIO-SAMPLE-007: n_steps=0 is clamped to 1 to avoid dimod errors."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}])
        underlying = _mock_neal(ss)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="neal")
        sampler.minimize_energy(b, J, n_samples=1, n_steps=0, beta=5.0)
        kw = underlying.sample.call_args[1]
        assert kw["num_sweeps"] == 1


# ---------------------------------------------------------------------------
# sample (fixed temperature)
# ---------------------------------------------------------------------------


class TestSample:
    """REQ-SAMPLE-003: sample draws at fixed temperature."""

    def test_shape_neal(self):
        """SCENARIO-SAMPLE-007: Neal sample returns (n_samples, n_spins) bool array."""
        b, J = _ferro(6)
        ss = _make_sample_set([{i: 1 for i in range(6)}] * 4)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=_mock_neal(ss)):
            sampler = DWaveSampler(mode="neal")
        result = sampler.sample(b, J, n_samples=4, config={"beta": 10.0})
        assert result.shape == (4, 6)
        assert result.dtype == bool

    def test_neal_fixed_temp_beta_range(self):
        """SCENARIO-SAMPLE-007: Fixed-temperature sample sets beta_range=[beta, beta]."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}] * 2)
        underlying = _mock_neal(ss)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="neal")
        sampler.sample(b, J, n_samples=2, config={"beta": 7.0, "n_warmup": 300})
        kw = underlying.sample.call_args[1]
        assert kw["beta_range"] == [7.0, 7.0]
        assert kw["num_sweeps"] == 300

    def test_defaults_in_config(self):
        """SCENARIO-SAMPLE-007: Config defaults apply when keys are absent."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}] * 2)
        underlying = _mock_neal(ss)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="neal")
        sampler.sample(b, J, n_samples=2, config={})
        kw = underlying.sample.call_args[1]
        assert kw["beta_range"] == [10.0, 10.0]
        assert kw["num_sweeps"] == 1000

    def test_tabu_sample_shape(self):
        """SCENARIO-SAMPLE-007: Tabu sample returns correct shape."""
        b, J = _ferro(5)
        ss = _make_sample_set([{i: 1 for i in range(5)}] * 3)
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=_mock_tabu(ss)):
            sampler = DWaveSampler(mode="tabu")
        result = sampler.sample(b, J, n_samples=3, config={"beta": 5.0})
        assert result.shape == (3, 5)

    def test_qpu_sample_shape(self):
        """SCENARIO-SAMPLE-007: QPU sample returns correct shape."""
        b, J = _ferro(4)
        ss = _make_sample_set([{i: 0 for i in range(4)}] * 2)
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=_mock_qpu(ss)):
            sampler = DWaveSampler(mode="qpu")
        result = sampler.sample(b, J, n_samples=2, config={"beta": 5.0})
        assert result.shape == (2, 4)


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """REQ-SAMPLE-007: health_check reports backend info correctly."""

    def test_neal_health_check(self):
        """SCENARIO-SAMPLE-007: Neal health_check returns online=True, no size limits."""
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=MagicMock()):
            sampler = DWaveSampler(mode="neal")
        info = sampler.health_check()
        assert info["online"] is True
        assert info["mode"] == "neal"
        assert info["backend"] == "dwave_neal"
        assert info["max_variables"] is None
        assert info["qpu_name"] is None
        assert info["chain_break_fraction_last"] == 0.0

    def test_tabu_health_check(self):
        """SCENARIO-SAMPLE-007: Tabu health_check returns online=True, no size limits."""
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=MagicMock()):
            sampler = DWaveSampler(mode="tabu")
        info = sampler.health_check()
        assert info["online"] is True
        assert info["mode"] == "tabu"
        assert info["backend"] == "dwave_tabu"
        assert info["max_variables"] is None

    def test_qpu_health_check_online(self):
        """SCENARIO-SAMPLE-007: QPU health_check returns qubit/coupler counts and chip ID."""
        underlying = _mock_qpu(MagicMock(), n_qubits=5627, n_couplers=40279)
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="qpu")
        info = sampler.health_check()
        assert info["online"] is True
        assert info["mode"] == "qpu"
        assert info["max_variables"] == 5627
        assert info["max_couplers"] == 40279
        assert info["qpu_name"] == "Advantage_system4.1"
        assert "chain_break_fraction_last" in info

    def test_qpu_health_check_offline(self):
        """SCENARIO-SAMPLE-007: QPU health_check returns online=False when hardware unreachable."""
        underlying = MagicMock()
        type(underlying.child).properties = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("connection refused"))
        )
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=underlying):
            sampler = DWaveSampler(mode="qpu")
        info = sampler.health_check()
        assert info["online"] is False
        assert info["max_variables"] is None


# ---------------------------------------------------------------------------
# get_backend integration
# ---------------------------------------------------------------------------


class TestGetBackendDwave:
    """REQ-SAMPLE-003: get_backend factory resolves dwave_* names correctly."""

    def test_get_dwave_neal(self):
        """SCENARIO-SAMPLE-007: get_backend('dwave_neal') returns DWaveSampler(mode='neal')."""
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=MagicMock()):
            backend = get_backend("dwave_neal")
        assert isinstance(backend, DWaveSampler)
        assert backend.backend_name == "dwave_neal"

    def test_get_dwave_tabu(self):
        """SCENARIO-SAMPLE-007: get_backend('dwave_tabu') returns DWaveSampler(mode='tabu')."""
        with patch("carnot.samplers.dwave_sampler._build_tabu_sampler", return_value=MagicMock()):
            backend = get_backend("dwave_tabu")
        assert isinstance(backend, DWaveSampler)
        assert backend.backend_name == "dwave_tabu"

    def test_get_dwave_qpu(self):
        """SCENARIO-SAMPLE-007: get_backend('dwave_qpu') returns DWaveSampler(mode='qpu')."""
        with patch("carnot.samplers.dwave_sampler._build_qpu_sampler", return_value=MagicMock()):
            backend = get_backend("dwave_qpu")
        assert isinstance(backend, DWaveSampler)
        assert backend.backend_name == "dwave_qpu"

    def test_unknown_backend_still_raises(self):
        """SCENARIO-SAMPLE-007: Unknown name still raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sampler backend"):
            get_backend("dwave_unknown")


# ---------------------------------------------------------------------------
# Builder helpers (exercises import paths for coverage)
# ---------------------------------------------------------------------------


class TestBuilderHelpers:
    """REQ-SAMPLE-003: Builder helper functions import and instantiate correctly."""

    def test_build_neal_sampler(self):
        """SCENARIO-SAMPLE-007: _build_neal_sampler calls SimulatedAnnealingSampler()."""
        from carnot.samplers.dwave_sampler import _build_neal_sampler

        mock_cls = MagicMock()
        with patch.dict(sys.modules, {"neal": MagicMock(SimulatedAnnealingSampler=mock_cls)}):
            _build_neal_sampler()
        mock_cls.assert_called_once()

    def test_build_tabu_sampler(self):
        """SCENARIO-SAMPLE-007: _build_tabu_sampler calls TabuSampler()."""
        from carnot.samplers.dwave_sampler import _build_tabu_sampler

        mock_cls = MagicMock()
        with patch.dict(sys.modules, {"tabu": MagicMock(TabuSampler=mock_cls)}):
            _build_tabu_sampler()
        mock_cls.assert_called_once()

    def test_build_qpu_sampler(self):
        """SCENARIO-SAMPLE-007: _build_qpu_sampler builds EmbeddingComposite(DWaveSampler())."""
        from carnot.samplers.dwave_sampler import _build_qpu_sampler

        mock_raw_cls = MagicMock()
        mock_embed_cls = MagicMock()
        dwave_system_mock = MagicMock(
            DWaveSampler=mock_raw_cls,
            EmbeddingComposite=mock_embed_cls,
        )
        with patch.dict(sys.modules, {"dwave.system": dwave_system_mock}):
            _build_qpu_sampler("my-token")
        mock_raw_cls.assert_called_once_with(token="my-token")
        mock_embed_cls.assert_called_once()


# ---------------------------------------------------------------------------
# benchmark_dwave_vs_cpu
# ---------------------------------------------------------------------------


class TestBenchmarkDwaveVsCpu:
    """REQ-SAMPLE-007: benchmark_dwave_vs_cpu returns timing dict."""

    def test_returns_expected_keys(self):
        """SCENARIO-SAMPLE-007: Benchmark result contains all expected keys."""
        b, J = _ferro(8)
        ss = _make_sample_set([{i: 1 for i in range(8)}] * 10)
        with patch("carnot.samplers.dwave_sampler._build_neal_sampler", return_value=_mock_neal(ss)):
            result = benchmark_dwave_vs_cpu(b, J, n_samples=10, n_steps=50, beta=5.0)
        assert "dwave_neal_seconds" in result
        assert "cpu_seconds" in result
        assert "n_spins" in result
        assert "sample_shape" in result
        assert "cpu_sample_shape" in result
        assert result["n_spins"] == 8
        assert result["sample_shape"] == [10, 8]
        assert result["cpu_sample_shape"] == [10, 8]
