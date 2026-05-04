"""Tests for ``docs/figures/fig3_fpga_latency.py``.

Spec traces: REQ-PUBLISH-011, SCENARIO-PUBLISH-011.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG3_PATH = REPO_ROOT / "docs" / "figures" / "fig3_fpga_latency.py"
EXP1068_PATH = REPO_ROOT / "results" / "experiment_1068_kv260_smoke_test_v9.json"


def _load_fig3():
    spec = importlib.util.spec_from_file_location("fig3_fpga_latency_for_test", FIG3_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latency_is_loaded_from_exp1068_req_publish_011() -> None:
    """REQ-PUBLISH-011: fig3 uses the measured Exp 1068 FPGA latency."""

    fig3 = _load_fig3()
    exp1068 = json.loads(EXP1068_PATH.read_text(encoding="utf-8"))

    assert fig3.load_measured_fpga_latency_us(EXP1068_PATH) == pytest.approx(
        exp1068["hardware_latency_us"]
    )


def test_latency_bars_contain_only_measured_fpga_scenario_publish_011() -> None:
    """SCENARIO-PUBLISH-011: plotted data is the measured FPGA bar only."""

    fig3 = _load_fig3()
    bars = fig3.build_latency_bars(24.82834388501942)

    assert bars == [
        {
            "label": "KV260 FPGA Ising",
            "latency_us": 24.82834388501942,
            "source": "exp1068",
            "measurement": "measured",
        }
    ]
    assert all("CPU" not in bar["label"] for bar in bars)


def test_source_removes_unmeasured_cpu_baseline_req_publish_011() -> None:
    """REQ-PUBLISH-011: fig3 source excludes the disputed CPU baseline."""

    source = FIG3_PATH.read_text(encoding="utf-8")

    for banned in ["CPU_LATENCY_MS", "290.0", "order-of-magnitude", "speedup ~", "11680"]:
        assert banned not in source


def test_render_writes_outputs_from_measured_only_latency(tmp_path: Path) -> None:
    """REQ-PUBLISH-011: rendering writes both figure formats from Exp 1068."""

    fig3 = _load_fig3()

    fig3.render(outdir=tmp_path, results_path=EXP1068_PATH)

    png = tmp_path / "fig3_fpga_latency.png"
    pdf = tmp_path / "fig3_fpga_latency.pdf"
    assert png.exists()
    assert png.stat().st_size > 0
    assert pdf.exists()
    assert pdf.stat().st_size > 0
