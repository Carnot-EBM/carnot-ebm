"""Export RationalKANEnergyFunction to Lean 4 formal specification.

Spec references: REQ-KAN-1647, SCENARIO-KAN-1647.
"""

from carnot.models.rkan import RationalKANEnergyFunction, serialize_fraction


def export_rkan_to_lean(model: RationalKANEnergyFunction, module_name: str = "RKANSpec") -> str:
    """Export a RationalKANEnergyFunction to Lean 4."""
    
    lines = [
        f"module {module_name}",
        "import Mathlib.Data.Rat.Basic",
        "",
        f"def input_dim : Nat := {model.input_dim}",
        ""
    ]
    
    # Export edges
    lines.append("def edge_splines : List (Nat × Nat × List Rat) := [")
    edge_entries = []
    for (i, j), spline in model.edge_splines.items():
        points = [f"({serialize_fraction(p)} : Rat)" for p in spline.control_points]
        edge_entries.append(f"  ({i}, {j}, [{', '.join(points)}])")
    lines.append(",\n".join(edge_entries))
    lines.append("]")
    lines.append("")
    
    # Export biases
    lines.append("def bias_splines : List (List Rat) := [")
    bias_entries = []
    for spline in model.bias_splines:
        points = [f"({serialize_fraction(p)} : Rat)" for p in spline.control_points]
        bias_entries.append(f"  [{', '.join(points)}]")
    lines.append(",\n".join(bias_entries))
    lines.append("]")
    
    return "\n".join(lines) + "\n"
