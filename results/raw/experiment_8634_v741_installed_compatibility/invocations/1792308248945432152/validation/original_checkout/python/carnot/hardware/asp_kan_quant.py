import numpy as np
import json
import os

def align_to_grid(values, grid_resolution):
    """
    Align values to a quantization grid.
    
    Args:
        values (np.ndarray): The values to quantize.
        grid_resolution (float): The resolution of the quantization grid.
        
    Returns:
        np.ndarray: The aligned values.
    """
    return np.round(values / grid_resolution) * grid_resolution

def enforce_symmetry(control_points):
    """
    Enforce symmetry sharing on control points for ASP-KAN-HAQ.
    Forces control points to share values symmetrically.
    
    Args:
        control_points (np.ndarray): The control points of the B-spline.
        
    Returns:
        np.ndarray: The symmetrically shared control points.
    """
    n = len(control_points)
    symmetric_points = np.zeros_like(control_points, dtype=float)
    for i in range(n):
        symmetric_points[i] = (control_points[i] + control_points[n - 1 - i]) / 2.0
    return symmetric_points

def asp_kan_quantize(knots, control_points, grid_resolution):
    """
    Apply ASP-KAN-HAQ alignment and symmetry to knots and control points.
    
    Args:
        knots (np.ndarray): Spline knots.
        control_points (np.ndarray): Spline control points.
        grid_resolution (float): Grid spacing for quantization.
        
    Returns:
        tuple: (aligned_knots, aligned_control_points)
    """
    aligned_knots = align_to_grid(knots, grid_resolution)
    sym_cps = enforce_symmetry(control_points)
    aligned_cps = align_to_grid(sym_cps, grid_resolution)
    return aligned_knots, aligned_cps

def forward_pass(x, knots, control_points):
    """
    Mock FP32 or Quantized forward pass using simple interpolation.
    
    Args:
        x (np.ndarray): Input values.
        knots (np.ndarray): The spline knots.
        control_points (np.ndarray): The spline control points.
        
    Returns:
        np.ndarray: Evaluated spline values.
    """
    # Assuming knots and control_points are sorted by knots
    # Ensure knots are strictly increasing for interp, handle aligned duplicate knots gracefully
    unique_knots, indices = np.unique(knots, return_index=True)
    unique_cps = control_points[indices]
    return np.interp(x, unique_knots, unique_cps)

def generate_artifact(success=True):
    """
    Generate the experiment JSON artifact.
    """
    artifact = {"asp_kan_ready": success}
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2104_asp_kan.json", "w") as f:
        json.dump(artifact, f, indent=2)
