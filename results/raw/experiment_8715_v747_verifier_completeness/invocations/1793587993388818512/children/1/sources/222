"""KAN hardware complexity analysis using arXiv 2604.03345 per-knot LUT estimates.

**Why this module exists:**
    Before committing FPGA synthesis time to a KAN energy model, we need a
    cost estimate.  arXiv 2604.03345 (Hardware-Oriented KAN Inference Complexity)
    provides per-knot LUT (Look-Up Table) estimates for piecewise-linear KAN
    activations: each piecewise-linear segment needs 2 comparators + 1 multiplier
    + 1 adder ≈ 8–12 LUTs.  We use the midpoint (10) as a conservative estimate.

**Architecture note:**
    KAEMEnergy (KANEnergyFunction) is graph-based, not a stacked MLP.  Its
    splines live on edges between spin nodes, not between MLP layers.  For an
    FPGA budget estimate we model it as an equivalent 2-layer KAN MLP:
      Layer 1: n_inputs → n_hidden  (each connection = one B-spline)
      Layer 2: n_hidden → 1         (each connection = one B-spline)
    This over-counts by roughly the difference between a fully-connected MLP
    and the actual sparse edge graph, so the estimate is conservative.

**LUT budget target:**
    iCE40 HX8K: 7,680 LUTs.

Spec: REQ-KAN-020
"""

from __future__ import annotations


class KANHardwareAnalyzer:
    """Estimate FPGA LUT cost for a 2-layer KAN energy model.

    Uses the per-segment formula from arXiv 2604.03345: each piecewise-linear
    segment in a KAN activation needs approximately 8–12 LUTs.  Default is 10
    (midpoint).

    The model is approximated as a 2-layer KAN MLP:
      - Layer 1: n_inputs → n_hidden
      - Layer 2: n_hidden → 1 (scalar energy output)

    Each connection is one B-spline with n_knots segments.

    Attributes:
        n_inputs: Number of spin inputs (N in Carnot terminology).
        n_hidden: Width of the hidden KAN layer.
        n_knots: Number of piecewise-linear knots per B-spline activation.
        luts_per_segment: LUT cost per piecewise-linear segment (default 10).
    """

    ICE40_HX8K_BUDGET: int = 7680

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_knots: int,
        luts_per_segment: int = 10,
    ) -> None:
        if n_inputs <= 0:
            raise ValueError("n_inputs must be > 0")
        if n_hidden <= 0:
            raise ValueError("n_hidden must be > 0")
        if n_knots < 2:
            raise ValueError("n_knots must be >= 2 (need at least one segment)")
        if luts_per_segment <= 0:
            raise ValueError("luts_per_segment must be > 0")

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_knots = n_knots
        self.luts_per_segment = luts_per_segment

    def lut_estimate_layer(self, fan_in: int, fan_out: int) -> int:
        """Estimate LUTs for one KAN layer with fan_in → fan_out connections.

        Per arXiv 2604.03345: each (fan_in, fan_out) pair has one B-spline
        with n_knots segments, each segment costing luts_per_segment LUTs.

        Args:
            fan_in: Number of input features to the layer.
            fan_out: Number of output features from the layer.

        Returns:
            Total LUT estimate for the layer.
        """
        return fan_in * fan_out * self.n_knots * self.luts_per_segment

    def total_lut_estimate(self) -> dict:
        """Estimate total LUTs for the full 2-layer KAN energy model.

        Returns a dict with per-layer and total counts, plus the iCE40 HX8K
        budget and a boolean indicating whether the model fits.

        Returns:
            Dict with keys: layer1_luts, layer2_luts, total_luts,
            ice40_hx8k_budget, within_budget.
        """
        layer1 = self.lut_estimate_layer(self.n_inputs, self.n_hidden)
        layer2 = self.lut_estimate_layer(self.n_hidden, 1)
        total = layer1 + layer2
        return {
            "layer1_luts": layer1,
            "layer2_luts": layer2,
            "total_luts": total,
            "ice40_hx8k_budget": self.ICE40_HX8K_BUDGET,
            "within_budget": total < self.ICE40_HX8K_BUDGET,
        }

    def synthesis_priority(self, ising_lut_count: int) -> str:
        """Determine which model should be synthesised first on iCE40 HX8K.

        Logic:
          - If KAN fits in budget AND is cheaper than Ising → KAN_PRIORITY
          - If Ising fits in budget AND is cheaper than KAN → ISING_PRIORITY
          - Both fit → BOTH_FEASIBLE (tie or KAN not cheaper)

        Args:
            ising_lut_count: Actual LUT count from a previous Ising synthesis
                (e.g. from Exp 859 pnr_lut_count field).

        Returns:
            "KAN_PRIORITY", "ISING_PRIORITY", or "BOTH_FEASIBLE".
        """
        kan_luts = self.total_lut_estimate()["total_luts"]
        if kan_luts < ising_lut_count and kan_luts < self.ICE40_HX8K_BUDGET:
            return "KAN_PRIORITY"
        elif ising_lut_count < kan_luts and ising_lut_count < self.ICE40_HX8K_BUDGET:
            return "ISING_PRIORITY"
        else:
            return "BOTH_FEASIBLE"

    def sensitivity_analysis(self, n_values: list[int]) -> dict[int, int]:
        """Compute total LUT estimate for several values of n_inputs.

        Useful for understanding how quickly LUT cost scales with problem size.

        Args:
            n_values: List of n_inputs values to evaluate.

        Returns:
            Dict mapping n_inputs → total_luts.
        """
        results: dict[int, int] = {}
        for n in n_values:
            analyzer = KANHardwareAnalyzer(
                n_inputs=n,
                n_hidden=self.n_hidden,
                n_knots=self.n_knots,
                luts_per_segment=self.luts_per_segment,
            )
            results[n] = analyzer.total_lut_estimate()["total_luts"]
        return results
