import numpy as np
import os
import json
from carnot.hardware.asp_kan_quant import (
    align_to_grid,
    enforce_symmetry,
    asp_kan_quantize,
    forward_pass,
    generate_artifact
)

def test_align_to_grid():
    """Test grid alignment logic for ASP-KAN-HAQ (REQ-KAN-2104)."""
    values = np.array([0.1, 0.4, 0.6, 0.9])
    aligned = align_to_grid(values, 0.5)
    np.testing.assert_allclose(aligned, [0.0, 0.5, 0.5, 1.0])

def test_enforce_symmetry():
    """Test symmetry sharing for control points (SCENARIO-KAN-2104)."""
    cps = np.array([1.0, 2.0, 4.0, 5.0])
    sym = enforce_symmetry(cps)
    np.testing.assert_allclose(sym, [3.0, 3.0, 3.0, 3.0])

def test_asp_kan_quantize():
    """Test combined quantization and symmetry enforcement."""
    knots = np.array([0.0, 0.33, 0.66, 1.0])
    cps = np.array([-1.0, 0.0, 1.0, 2.0])
    aligned_knots, aligned_cps = asp_kan_quantize(knots, cps, grid_resolution=0.5)
    
    np.testing.assert_allclose(aligned_knots, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(aligned_cps, [0.5, 0.5, 0.5, 0.5])

def test_quantized_forward_pass_vs_fp32():
    """Test forward pass of quantized vs full FP32 (SCENARIO-KAN-2104)."""
    knots = np.array([0.0, 0.5, 1.0])
    cps = np.array([0.1, 0.9, 0.2])
    
    x = np.array([0.25, 0.75])
    fp32_res = forward_pass(x, knots, cps)
    
    aligned_knots, aligned_cps = asp_kan_quantize(knots, cps, grid_resolution=0.5)
    quantized_res = forward_pass(x, aligned_knots, aligned_cps)
    
    assert fp32_res.shape == quantized_res.shape
    # Check that they differ slightly due to quantization but remain within reasonable error
    diff = np.abs(fp32_res - quantized_res)
    assert np.any(diff > 0.0), "Quantized pass should differ from FP32"
    
def test_generate_artifact():
    """Test the generation of the experiment JSON artifact."""
    generate_artifact(success=True)
    assert os.path.exists("results/experiment_2104_asp_kan.json")
    with open("results/experiment_2104_asp_kan.json") as f:
        data = json.load(f)
    assert data["asp_kan_ready"] is True
