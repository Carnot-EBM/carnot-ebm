"""
Tests for experiment 1730 CIKAN FPGA benchmark.
"""

import json
import os
import shutil
import subprocess
import tempfile
from unittest import mock

# Need to make sure scripts dir is in path or import it directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.experiment_1730_cikan_fpga import run_experiment

# REQ-KAN-1730: CIKAN FPGA benchmark metrics
# SCENARIO-KAN-1730-1: Simulation fallback

def test_run_experiment_simulation():
    """Test running experiment when Vivado is not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = os.path.join(tmpdir, "out.json")
        tcl_script = os.path.join(tmpdir, "synth.tcl")
        lut_output = os.path.join(tmpdir, "lut.v")
        
        with mock.patch("shutil.which", return_value=None):
            results = run_experiment(output_json, tcl_script, lut_output)
            
        assert os.path.exists(output_json)
        assert os.path.exists(lut_output)
        
        with open(output_json) as f:
            data = json.load(f)
            assert data["vivado_found"] is False
            assert "metrics" in data
            assert data["metrics"]["lut_count"] == 1

def test_run_experiment_vivado_success():
    """Test running experiment when Vivado succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = os.path.join(tmpdir, "out.json")
        tcl_script = os.path.join(tmpdir, "synth.tcl")
        lut_output = os.path.join(tmpdir, "lut.v")
        
        with mock.patch("shutil.which", return_value="/fake/path/vivado"), \
             mock.patch("subprocess.run") as mock_run:
            results = run_experiment(output_json, tcl_script, lut_output)
            
        assert os.path.exists(output_json)
        assert results["synthesis_run"] is True
        assert results["metrics"]["fmax_mhz"] == 400.0

def test_run_experiment_vivado_failure():
    """Test running experiment when Vivado fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = os.path.join(tmpdir, "out.json")
        tcl_script = os.path.join(tmpdir, "synth.tcl")
        lut_output = os.path.join(tmpdir, "lut.v")
        
        with mock.patch("shutil.which", return_value="/fake/path/vivado"), \
             mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "vivado", stderr="error")):
            results = run_experiment(output_json, tcl_script, lut_output)
            
        assert os.path.exists(output_json)
        assert results["synthesis_run"] is False
        assert "synthesis_error" in results
        assert results["metrics"]["fmax_mhz"] == 450.0
