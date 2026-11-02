"""KAN Symbolizer for extracting exact symbolic formulas.

This module implements the KANSymbolizer class that extracts piecewise
algebraic expressions from trained KAN energy functions and generates
SMT-verifiable AST strings, satisfying REQ-KAN-2060.
"""

import numpy as np
from typing import Dict, List, Any


class KANSymbolizer:
    """Extracts piecewise algebraic expressions from trained KAN energy functions.

    This supports generating SMT-verifiable expressions required by KAN4CBC.
    """

    def __init__(self, layer: Any) -> None:
        """Initialize the symbolizer with a trained KAN layer.

        Parameters
        ----------
        layer : Any
            A KAN layer, specifically expected to be UnivariateKAEMLayer.
        """
        self.layer = layer
        if hasattr(self.layer, "n_vars") and hasattr(self.layer, "n_knots"):
            self.n_vars = self.layer.n_vars
            self.n_knots = self.layer.n_knots
        else:
            raise ValueError("Layer must have n_vars and n_knots properties.")

    def extract_piecewise_polynomials(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract knot points and polynomial coefficients for each variable.

        Returns
        -------
        Dict[int, List[Dict[str, Any]]]
            A dictionary mapping variable index to a list of piecewise linear segments.
            Each segment contains 'interval' (x0, x1) and 'coeffs' (intercept, slope).
        """
        # Knots are assumed to be accessible via layer._knots or layer.knots
        if hasattr(self.layer, "_knots"):
            knots = np.array(self.layer._knots)
        elif hasattr(self.layer, "knots"):
            knots = np.array(self.layer.knots)
        else:
            raise ValueError("Layer must have _knots or knots property.")

        # Control points are assumed to be accessible via layer.control_points
        if hasattr(self.layer, "control_points"):
            control_points = np.array(self.layer.control_points)
        else:
            raise ValueError("Layer must have control_points property.")

        expressions = {}
        for i in range(self.n_vars):
            segments = []
            ctrl = control_points[i]
            for j in range(self.n_knots - 1):
                x0 = float(knots[j])
                x1 = float(knots[j + 1])
                c0 = float(ctrl[j])
                c1 = float(ctrl[j + 1])

                dx = x1 - x0
                if dx == 0:
                    slope = 0.0
                else:
                    slope = (c1 - c0) / dx
                intercept = c0 - slope * x0

                segments.append(
                    {
                        "interval": [x0, x1],
                        "coeffs": [intercept, slope],
                    }
                )
            expressions[i] = segments

        return expressions

    def to_ast_string(self) -> str:
        """Generate a symbolic AST string representing the constraints.

        Returns
        -------
        str
            A string representing the SMT-verifiable expression of the energy.
        """
        polynomials = self.extract_piecewise_polynomials()

        # Build an AST string. For simplicity, we create a sum of per-variable expressions.
        ast_parts = []
        for var_idx, segments in polynomials.items():
            var_name = f"x_{var_idx}"

            # Start from the first segment and build nested ite (if-then-else)
            var_ast = ""
            for j, seg in enumerate(segments):
                x0, _ = seg["interval"]
                intercept, slope = seg["coeffs"]

                expr = f"(+ {intercept} (* {slope} {var_name}))"

                if j == 0:
                    var_ast = expr
                else:
                    var_ast = f"(ite (< {var_name} {x0}) {var_ast} {expr})"

            ast_parts.append(var_ast)

        if len(ast_parts) == 1:
            return ast_parts[0]
        else:
            joined = " ".join(ast_parts)
            return f"(+ {joined})"
