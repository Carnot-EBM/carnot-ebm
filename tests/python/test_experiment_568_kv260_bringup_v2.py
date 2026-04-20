"""Tests for Experiment 568: KV260 FPGA Bring-Up v2.

Spec coverage: REQ-SAMPLE-031,
               SCENARIO-SAMPLE-049, SCENARIO-SAMPLE-050, SCENARIO-SAMPLE-051

Design philosophy:
    - Tests that require physical KV260 hardware auto-skip when
      CARNOT_KV260_BITFILE is not set.
    - Tests covering the synthesis_required / CPU baseline path always run.
    - We never fabricate latency numbers: hardware_latency_us must be None
      when FPGA was not exercised.
    - We cover all new functions added by this experiment:
      _measure_cpu_baseline, _ensure_tcl_stub, and run_experiment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import scripts.experiment_568_kv260_bringup_v2 as exp568
from scripts.experiment_568_kv260_bringup_v2 import (
    CPU_BASELINE_REF_US,
    EXP_ID,
    LATENCY_TARGET_US,
    N_SPINS,
    N_TRIALS,
    SYNTHESIS_COMMAND,
    TCL_STUB_PATH,
    _ensure_tcl_stub,
    _measure_cpu_baseline,
    run_experiment,
)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031: Module-level constants are correct
# ---------------------------------------------------------------------------


class TestConstants:
    """REQ-SAMPLE-031: Core constants match the spec contract."""

    def test_exp_id_is_568(self) -> None:
        """REQ-SAMPLE-031: EXP_ID must be 568."""
        assert EXP_ID == 568

    def test_n_spins_is_100(self) -> None:
        """REQ-SAMPLE-031: N_SPINS=100 matches Exp 313 methodology."""
        assert N_SPINS == 100

    def test_n_trials_is_100(self) -> None:
        """REQ-SAMPLE-031: N_TRIALS=100 for statistical validity."""
        assert N_TRIALS == 100

    def test_latency_target_is_100us(self) -> None:
        """REQ-SAMPLE-031: LATENCY_TARGET_US=100 per spec."""
        assert LATENCY_TARGET_US == 100.0

    def test_synthesis_command_correct(self) -> None:
        """SCENARIO-SAMPLE-050: synthesis_command matches spec."""
        assert SYNTHESIS_COMMAND == "vivado -mode batch -source hardware/kv260/synth_ising.tcl"

    def test_cpu_baseline_ref_positive(self) -> None:
        """REQ-SAMPLE-031: CPU reference from Exp 313 is positive and realistic."""
        assert CPU_BASELINE_REF_US > 100_000  # at least 100ms


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031: _measure_cpu_baseline
# ---------------------------------------------------------------------------


class TestMeasureCpuBaseline:
    """REQ-SAMPLE-031-4: CPU baseline always measurable."""

    def test_returns_positive_float(self) -> None:
        """SCENARIO-SAMPLE-051: cpu_baseline_latency_us is positive."""
        latency = _measure_cpu_baseline(n_spins=10, n_trials=2)
        assert isinstance(latency, float)
        assert latency > 0.0

    def test_result_in_microseconds_range(self) -> None:
        """REQ-SAMPLE-031: Result is in microseconds (not ms or s)."""
        latency = _measure_cpu_baseline(n_spins=10, n_trials=2)
        # Very loose bounds: should be between 10μs and 10s for n_spins=10
        assert 1.0 < latency < 10_000_000.0

    def test_multiple_calls_consistent(self) -> None:
        """REQ-SAMPLE-031: Repeated calls return consistent order-of-magnitude results."""
        lat1 = _measure_cpu_baseline(n_spins=10, n_trials=2)
        lat2 = _measure_cpu_baseline(n_spins=10, n_trials=2)
        # Within 100x of each other — catches unit confusion (s vs μs)
        assert lat1 / lat2 < 100.0
        assert lat2 / lat1 < 100.0


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031: _ensure_tcl_stub
# ---------------------------------------------------------------------------


class TestEnsureTclStub:
    """SCENARIO-SAMPLE-050: synth_ising.tcl is created when absent."""

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-050: Returns True when TCL stub already exists."""
        fake_root = tmp_path
        tcl_dir = fake_root / "hardware" / "kv260"
        tcl_dir.mkdir(parents=True)
        (tcl_dir / "synth_ising.tcl").write_text("# existing stub\n")

        result = _ensure_tcl_stub(fake_root)
        assert result is True

    def test_returns_false_and_creates_file_when_absent(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-050: Creates stub and returns False when file is absent."""
        fake_root = tmp_path
        result = _ensure_tcl_stub(fake_root)
        assert result is False
        tcl_path = fake_root / TCL_STUB_PATH
        assert tcl_path.exists()

    def test_created_stub_contains_vivado_command(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-050: Created stub references Vivado synthesis."""
        _ensure_tcl_stub(tmp_path)
        content = (tmp_path / TCL_STUB_PATH).read_text()
        assert "synth_design" in content or "vivado" in content.lower()

    def test_created_stub_references_verilog(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-050: Created stub references the RTL source file."""
        _ensure_tcl_stub(tmp_path)
        content = (tmp_path / TCL_STUB_PATH).read_text()
        assert "ising_sampler" in content


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031 / SCENARIO-SAMPLE-050: synthesis_required path
# ---------------------------------------------------------------------------


class TestSynthesisRequiredPath:
    """SCENARIO-SAMPLE-050: Behaviour when CARNOT_KV260_BITFILE is unset."""

    def test_honest_verdict_synthesis_required(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-050: honest_verdict='synthesis_required' when bitfile unset."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["honest_verdict"] == "synthesis_required"

    def test_synthesis_command_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-050: synthesis_command is the Vivado batch command."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["synthesis_command"] == SYNTHESIS_COMMAND

    def test_bitfile_set_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-051: bitfile_set=False when env var absent."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["bitfile_set"] is False

    def test_hardware_latency_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: hardware_latency_us=None when not exercised."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["hardware_latency_us"] is None

    def test_fpga_alive_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: fpga_alive=False when hardware not exercised."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["fpga_alive"] is False

    def test_fpga_speedup_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-051: fpga_speedup=None when hardware not exercised."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["fpga_speedup"] is None


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031 / SCENARIO-SAMPLE-051: CPU baseline always present
# ---------------------------------------------------------------------------


class TestCpuBaselineAlwaysPresent:
    """SCENARIO-SAMPLE-051: cpu_baseline_latency_us present in every artifact."""

    def test_cpu_baseline_present_when_no_bitfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-051: CPU baseline measured when bitfile unset."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert "cpu_baseline_latency_us" in artifact
        assert artifact["cpu_baseline_latency_us"] > 0.0

    def test_cpu_baseline_is_positive_float(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-051: cpu_baseline_latency_us is a positive number."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert isinstance(artifact["cpu_baseline_latency_us"], float)
        assert artifact["cpu_baseline_latency_us"] > 0.0


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031: Artifact schema completeness
# ---------------------------------------------------------------------------


REQUIRED_ARTIFACT_FIELDS = [
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "schema",
    "bitfile_set",
    "hardware_latency_us",
    "cpu_baseline_latency_us",
    "fpga_speedup",
    "fpga_alive",
    "honest_verdict",
]


class TestArtifactSchema:
    """REQ-SAMPLE-031: Artifact has all required top-level fields."""

    def test_required_fields_present_synthesis_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: All required fields present on synthesis_required path."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        for field in REQUIRED_ARTIFACT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_experiment_id_is_568(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: experiment=568 in every artifact."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["experiment"] == 568

    def test_schema_identifies_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: schema field present and non-empty."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["schema"]  # non-empty

    def test_artifact_written_to_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: Artifact is written to disk when write_output=True."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        output = tmp_path / "results" / "exp568.json"
        run_experiment(output, write_output=True, _cpu_trials=2)
        assert output.exists()
        with output.open() as f:
            data = json.load(f)
        assert data["experiment"] == 568

    def test_status_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: status='success' on normal completion."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["status"] == "success"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-031: honest_verdict approved vocabulary
# ---------------------------------------------------------------------------

APPROVED_VERDICTS = frozenset(["hardware_working", "synthesis_required", "hardware_too_slow"])


class TestHonestVerdict:
    """REQ-SAMPLE-031-5: honest_verdict from approved vocabulary."""

    def test_verdict_in_approved_set_synthesis_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: verdict is from approved set on synthesis path."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        artifact = run_experiment(
            tmp_path / "exp568.json",
            write_output=False,
            _cpu_trials=2,
        )
        assert artifact["honest_verdict"] in APPROVED_VERDICTS

    def test_hardware_too_slow_when_fpga_exceeds_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: hardware_too_slow when FPGA latency >= 100μs."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/fake/path/carnot.bit")

        # Patch FpgaBackend measurement to return a slow value (1ms = 1000μs)
        with patch.object(exp568, "_measure_fpga_latency", return_value=1000.0):
            artifact = run_experiment(
                tmp_path / "exp568.json",
                write_output=False,
                _cpu_trials=2,
            )

        assert artifact["honest_verdict"] == "hardware_too_slow"
        assert artifact["fpga_alive"] is False
        assert artifact["hardware_latency_us"] == 1000.0

    def test_hardware_working_when_fpga_fast(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-049: hardware_working when FPGA latency < 100μs."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/fake/path/carnot.bit")

        # Patch FpgaBackend measurement to return a fast value (50μs)
        with patch.object(exp568, "_measure_fpga_latency", return_value=50.0):
            artifact = run_experiment(
                tmp_path / "exp568.json",
                write_output=False,
                _cpu_trials=2,
            )

        assert artifact["honest_verdict"] == "hardware_working"
        assert artifact["fpga_alive"] is True
        assert artifact["hardware_latency_us"] == 50.0

    def test_fpga_speedup_computed_when_hardware_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-051: fpga_speedup = cpu_baseline / hardware_latency."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/fake/path/carnot.bit")

        with patch.object(exp568, "_measure_fpga_latency", return_value=50.0), \
             patch.object(exp568, "_measure_cpu_baseline", return_value=50000.0):
            artifact = run_experiment(
                tmp_path / "exp568.json",
                write_output=False,
                _cpu_trials=2,
            )

        assert artifact["fpga_speedup"] is not None
        assert abs(artifact["fpga_speedup"] - 1000.0) < 1.0  # 50000 / 50 = 1000

    def test_fpga_benchmark_exception_gives_hardware_too_slow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-031: FpgaBackend exception → hardware_too_slow with null latency."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/fake/path/carnot.bit")

        with patch.object(exp568, "_measure_fpga_latency", side_effect=RuntimeError("PYNQ unavailable")):
            artifact = run_experiment(
                tmp_path / "exp568.json",
                write_output=False,
                _cpu_trials=2,
            )

        assert artifact["honest_verdict"] == "hardware_too_slow"
        assert artifact["hardware_latency_us"] is None
        assert artifact["fpga_alive"] is False


# ---------------------------------------------------------------------------
# Hardware path tests — auto-skip when KV260 not present
# ---------------------------------------------------------------------------

_hw_available = os.environ.get("CARNOT_KV260_BITFILE") is not None
HW_REASON = "KV260 hardware not available: CARNOT_KV260_BITFILE not set"


@pytest.mark.skipif(not _hw_available, reason=HW_REASON)
class TestHardwarePath:
    """SCENARIO-SAMPLE-049: Tests that require real KV260 hardware.

    These tests only run when CARNOT_KV260_BITFILE is set to a valid bitfile.
    """

    def test_hardware_working_verdict(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-049: honest_verdict='hardware_working' on real KV260."""
        artifact = run_experiment(tmp_path / "exp568_hw.json", write_output=False)
        assert artifact["honest_verdict"] == "hardware_working"

    def test_hardware_latency_within_target(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-049: mean_latency_us < 100 on real KV260."""
        artifact = run_experiment(tmp_path / "exp568_hw.json", write_output=False)
        assert artifact["hardware_latency_us"] is not None
        assert artifact["hardware_latency_us"] < LATENCY_TARGET_US

    def test_fpga_speedup_significant(self, tmp_path: Path) -> None:
        """SCENARIO-SAMPLE-049: FPGA is significantly faster than CPU baseline."""
        artifact = run_experiment(tmp_path / "exp568_hw.json", write_output=False)
        assert artifact["fpga_speedup"] is not None
        # Expect at least 100x speedup (358ms CPU vs <100μs FPGA)
        assert artifact["fpga_speedup"] > 100.0
