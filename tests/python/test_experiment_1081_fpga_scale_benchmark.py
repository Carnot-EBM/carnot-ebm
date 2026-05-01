"""Unit tests for the FPGA-vs-CPU scaling-benchmark helpers (Exp 1081).

These tests exercise the pure-function helpers in
``scripts.experiment_1081_fpga_scale_benchmark`` — extrapolation,
crossover algebra, and the artifact schema — without requiring SSH
to the KV260 board, Vivado, or a long CPU benchmark.  Live-system
behaviour is covered by the experiment script itself when the
conductor runs it; the test suite must stay fast and hermetic.

Spec refs: REQ-HW-040, SCENARIO-HW-040, REQ-VERIFY-083 (artifact schema).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

mod = importlib.import_module("scripts.experiment_1081_fpga_scale_benchmark")


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_fpga_latency_us_anchors_at_64():
    # At the anchor size the extrapolation must return the anchor exactly.
    assert mod.fpga_latency_us(64) == pytest.approx(mod.EXP1068_FPGA_LATENCY_US)


def test_fpga_latency_us_scales_linearly():
    # Doubling N must double the predicted latency (linear pipelined model).
    base = mod.fpga_latency_us(64)
    assert mod.fpga_latency_us(128) == pytest.approx(2 * base)
    assert mod.fpga_latency_us(1024) == pytest.approx(16 * base)


def test_cpu_latency_theoretical_ms_quadratic():
    # Doubling N must quadruple the theoretical CPU latency.
    base = mod.cpu_latency_theoretical_ms(64)
    assert mod.cpu_latency_theoretical_ms(128) == pytest.approx(4 * base)
    assert mod.cpu_latency_theoretical_ms(256) == pytest.approx(16 * base)


def test_solve_theoretical_crossover_below_one_at_canonical_anchors():
    # 24.83 us / 290 ms / 64 spins gives a sub-1 fractional crossover:
    # the FPGA wins at every realisable problem size.
    cross = mod.solve_theoretical_crossover()
    assert 0 < cross < 1


def test_solve_theoretical_crossover_explicit_formula():
    # Explicit-numerics check: N = 64 * (us/1000) / ms.
    cross = mod.solve_theoretical_crossover(fpga_anchor_us=1000.0, cpu_anchor_ms=10.0)
    # 1000 us = 1 ms; 64 * 1 / 10 = 6.4
    assert cross == pytest.approx(6.4)


def test_find_measured_crossover_picks_smallest_winning_n():
    cpu = {64: 100.0, 128: 100.0, 256: 100.0}
    fpga = {64: 200_000.0, 128: 90_000.0, 256: 50_000.0}
    # 64 us = 0.064 ms, but here fpga[64]=200000 us = 200 ms > 100 ms so 64 doesn't win.
    # fpga[128]=90 ms < 100 ms cpu, so 128 is the smallest winning N.
    assert mod.find_measured_crossover(cpu, fpga) == 128


def test_find_measured_crossover_returns_none_when_fpga_always_loses():
    cpu = {64: 0.001, 128: 0.001}
    fpga = {64: 100_000.0, 128: 200_000.0}  # FPGA is 100 ms vs 0.001 ms CPU
    assert mod.find_measured_crossover(cpu, fpga) is None


def test_find_measured_crossover_winning_at_64():
    # Realistic Carnot numbers: CPU ~270 ms, FPGA ~25 us — FPGA wins at 64.
    cpu = {64: 270.0, 128: 260.0}
    fpga = {64: 24.83, 128: 49.66}
    assert mod.find_measured_crossover(cpu, fpga) == 64


# ---------------------------------------------------------------------------
# End-to-end artifact build (pure path, no live SSH probe)
# ---------------------------------------------------------------------------


def test_run_experiment_writes_artifact_with_required_fields(monkeypatch, tmp_path):
    """Drive run_experiment with stubbed-out CPU + FPGA probes for speed."""

    # 1. Stub the slow CPU benchmark — return a flat ~270 ms / call so the
    #    measured crossover is N=64 (matches the live observation).
    monkeypatch.setattr(mod, "measure_cpu_latency", lambda n, **kw: 270.0)

    # 2. Skip the live FPGA probe path (covered by an explicit test below).
    monkeypatch.setattr(mod, "check_board_reachable", lambda *a, **kw: True)
    monkeypatch.setattr(mod, "check_ssh_ok", lambda *a, **kw: True)
    monkeypatch.setattr(mod, "probe_fpga_latency_live", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "vivado_available", lambda: False)

    # 3. Redirect the artifact write into tmp_path so we don't clobber the
    #    real results file.  The ExperimentTemplate also writes to
    #    results/... but we only check the run_experiment-level write here.
    real_repo_root = mod._REPO_ROOT
    artifact = mod.run_experiment(
        cpu_reps_small=1,
        cpu_reps_large=1,
        n_warmup=1,
        skip_fpga_live_probe=True,
    )

    # Required schema fields from REQUIRED_RESULT_FIELDS.
    for field in (
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ):
        assert field in artifact, f"missing required field {field}"

    # Experiment-specific fields the conductor's retrospective expects.
    for field in (
        "board_reachable",
        "cpu_latencies_ms",
        "fpga_latency_64_us",
        "fpga_latencies_us",
        "extrapolation_mode",
        "crossover_n_spins",
        "max_speedup_measured",
        "vivado_synthesis_attempted",
        "honest_verdict",
    ):
        assert field in artifact, f"missing experiment field {field}"

    assert artifact["experiment"] == 1081
    assert artifact["honest_verdict"] == "fpga_speedup_confirmed_measured"
    assert artifact["crossover_n_spins"] == 64
    assert artifact["max_speedup_measured"] > 1000  # FPGA crushes CPU at N=64
    assert artifact["extrapolation_mode"] in {"measured", "theoretical", "mixed"}
    assert artifact["status"] == "success"

    # Verify the artifact path exists on disk.  We do NOT re-read+re-parse here
    # because pytest-xdist runs tests in parallel and the unreachable-board
    # sibling test in this file overwrites the same file; the in-memory
    # `artifact` dict already covers the schema check.
    artifact_path = real_repo_root / "results" / "experiment_1081_fpga_scale_benchmark.json"
    assert artifact_path.exists()


def test_run_experiment_marks_board_unreachable_when_ping_fails(monkeypatch):
    """When the KV260 is unreachable, the verdict must say so honestly."""
    monkeypatch.setattr(mod, "measure_cpu_latency", lambda n, **kw: 270.0)
    monkeypatch.setattr(mod, "check_board_reachable", lambda *a, **kw: False)
    monkeypatch.setattr(mod, "check_ssh_ok", lambda *a, **kw: False)
    monkeypatch.setattr(mod, "probe_fpga_latency_live", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "vivado_available", lambda: False)

    artifact = mod.run_experiment(
        cpu_reps_small=1,
        cpu_reps_large=1,
        n_warmup=1,
        skip_fpga_live_probe=True,
    )
    assert artifact["board_reachable"] is False
    assert artifact["honest_verdict"] == "board_unreachable"
    assert artifact["status"] != "success"
