"""Tests for RKAN Lean 4 Formal Specification Exporter.

Spec references: REQ-KAN-1647, SCENARIO-KAN-1647.
"""

from carnot.models.rkan import RationalKANEnergyFunction
from carnot.models.rkan_lean_export import export_rkan_to_lean


def test_export_rkan_to_lean():
    """Test that a simple RKAN model can be exported to Lean 4."""
    model = RationalKANEnergyFunction(
        input_dim=2,
        edge_control_points={
            (0, 1): ["1/2", "1"],
        },
        bias_control_points=[
            ["0", "1"],
            ["-1", "0"],
        ],
    )
    
    lean_code = export_rkan_to_lean(model, module_name="TestSpec")
    
    assert "module TestSpec" in lean_code
    assert "def input_dim : Nat := 2" in lean_code
    assert "def edge_splines" in lean_code
    assert "1/2" in lean_code
    assert "def bias_splines" in lean_code
