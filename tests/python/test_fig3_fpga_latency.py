"""Regression tests for Figure 3 publication integrity.

Spec anchors: REQ-PUBLISH-011 and SCENARIO-PUBLISH-011 require Figure 3 to
display only measured KV260 FPGA latency data from Exp 1068 and to remove the
unmeasured CPU baseline / derived speedup headline.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG3_PATH = REPO_ROOT / "docs" / "figures" / "fig3_fpga_latency.py"
EXP1068_PATH = REPO_ROOT / "results" / "experiment_1068_kv260_smoke_test_v9.json"


def _load_fig3_module():
    spec = importlib.util.spec_from_file_location("fig3_fpga_latency", FIG3_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fig3_loads_exp1068_measured_latency_only():
    """REQ-PUBLISH-011: the plotted Figure 3 datum must come from Exp 1068."""
    fig3 = _load_fig3_module()
    exp1068 = json.loads(EXP1068_PATH.read_text(encoding="utf-8"))

    latency_us = fig3.load_measured_fpga_latency_us(EXP1068_PATH)
    bars = fig3.measured_latency_bars(EXP1068_PATH)

    assert latency_us == exp1068["hardware_latency_us"]
    assert bars == [
        {
            "label": "KV260 FPGA Ising\n(exp1068)",
            "latency_us": exp1068["hardware_latency_us"],
            "color": "#7b1c1c",
        }
    ]


def test_fig3_source_removes_unmeasured_cpu_speedup_headline():
    """SCENARIO-PUBLISH-011: no CPU estimate or derived speedup badge remains."""
    source = FIG3_PATH.read_text(encoding="utf-8")

    assert "CPU_LATENCY_MS" not in source
    assert "290.0" not in source
    assert "speedup ~" not in source
    assert "cpu_us / fpga_us" not in source


def test_fig3_render_writes_measured_only_outputs(tmp_path: Path):
    """REQ-PUBLISH-011: rendering writes figure files from the measured bar."""
    fig3 = _load_fig3_module()

    fig3.render(outdir=tmp_path, result_path=EXP1068_PATH)

    png = tmp_path / "fig3_fpga_latency.png"
    pdf = tmp_path / "fig3_fpga_latency.pdf"
    assert png.exists()
    assert png.stat().st_size > 0
    assert pdf.exists()
    assert pdf.stat().st_size > 0
