#!/usr/bin/env python3
"""Exp 866: KAN hardware complexity analysis — arXiv 2604.03345 LUT estimates.

**Why this experiment exists:**
    Exp 859 proved that the Ising N=8 energy oracle fits on iCE40 HX8K in 134 LUTs.
    Before investing synthesis time in the KAN (Kolmogorov-Arnold Network) energy
    model, we need to know whether it also fits in 7680 LUTs.

    arXiv 2604.03345 (Hardware-Oriented KAN Inference Complexity) gives per-knot
    LUT estimates for piecewise-linear KAN activations.  This experiment applies
    that formula to KAEMEnergy and determines which model should be synthesised
    first.

**Architecture note:**
    KAEMEnergy (KANEnergyFunction) is graph-based: B-spline edges between spin
    nodes, not a stacked MLP.  For a conservative FPGA budget estimate we model
    it as an equivalent 2-layer KAN MLP (n_inputs → n_hidden → 1).  The actual
    graph-based implementation will have fewer active splines (sparse edges),
    so the estimate is an upper bound.

    Actual KANConfig defaults (from python/carnot/models/kan.py):
      - num_knots  = 10  (piecewise-linear knot count per B-spline)
      - degree     = 3   (cubic; does not affect LUT count per 2604.03345)
      - sparse     = True, edge_density = 0.1  (only ~10% of edges exist)

    n_hidden = 16 is used as the worst-case MLP approximation.  With sparse
    edges the real cost would be ~10% of this estimate.

Spec: REQ-KAN-020
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root: python scripts/experiment_866_...py
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_template import ExperimentTemplate
from python.carnot.analysis.kan_hw_analysis import KANHardwareAnalyzer

# ---------------------------------------------------------------------------
# Configuration constants matching actual KAEMEnergy defaults
# ---------------------------------------------------------------------------
N_INPUTS_N8: int = 8          # Problem size for primary analysis
N_HIDDEN: int = 16            # Worst-case MLP hidden width approximation
N_KNOTS: int = 10             # KANConfig.num_knots default
LUTS_PER_SEGMENT: int = 10    # midpoint of arXiv 2604.03345 8–12 LUT range

# Ising N=8 baseline from Exp 859 (pnr_lut_count after place-and-route)
ISING_LUT_COUNT_EXP859: int = 134
ISING_LUT_FALLBACK: int = 144  # theoretical estimate if Exp 859 unavailable


def load_ising_lut_count() -> int:
    """Read pnr_lut_count from Exp 859 result, falling back to theoretical 144."""
    result_path = Path("results/experiment_859_ice40_n8_combinational.json")
    if result_path.exists():
        try:
            with result_path.open() as fh:
                data = json.load(fh)
            return int(data["pnr_lut_count"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return ISING_LUT_FALLBACK


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=866,
        title="KAN hardware complexity analysis — arXiv 2604.03345 LUT estimates",
        deliverable="results/experiment_866_kan_hardware_analysis.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Primary analysis: N=8
    analyzer = KANHardwareAnalyzer(
        n_inputs=N_INPUTS_N8,
        n_hidden=N_HIDDEN,
        n_knots=N_KNOTS,
        luts_per_segment=LUTS_PER_SEGMENT,
    )

    lut_estimate = analyzer.total_lut_estimate()
    ising_lut_count = load_ising_lut_count()
    priority = analyzer.synthesis_priority(ising_lut_count)

    kan_lut_n8 = lut_estimate["total_luts"]
    within_budget = lut_estimate["within_budget"]

    # Sensitivity analysis across N values
    sensitivity_raw = analyzer.sensitivity_analysis([4, 8, 16])
    sensitivity = {f"n{k}": v for k, v in sensitivity_raw.items()}

    # Determine honest verdict
    if within_budget:
        honest_verdict = "kan_fpga_roadmap_clear"
    else:
        honest_verdict = "kan_over_budget"

    payload = {
        "kan_lut_estimate_n8": kan_lut_n8,
        "ising_lut_count": ising_lut_count,
        "synthesis_priority": priority,
        "within_ice40_budget": within_budget,
        "kan_fpga_roadmap_clear": within_budget,
        "priority_determined": True,
        "sensitivity": sensitivity,
        "lut_detail": lut_estimate,
        "config": {
            "n_inputs": N_INPUTS_N8,
            "n_hidden": N_HIDDEN,
            "n_knots": N_KNOTS,
            "luts_per_segment": LUTS_PER_SEGMENT,
            "architecture_note": (
                "2-layer MLP upper bound; actual KAEMEnergy is sparse graph-based "
                "with edge_density=0.1, so real LUT cost is ~10% of this estimate"
            ),
        },
        "honest_verdict": honest_verdict,
        "analysis_complete": True,
    }

    artifact = tmpl.build_result(payload, status="success")

    output_path = Path("results/experiment_866_kan_hardware_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"KAN N=8 LUT estimate : {kan_lut_n8:,}")
    print(f"Ising N=8 LUT count  : {ising_lut_count}")
    print(f"Within iCE40 budget  : {within_budget}")
    print(f"Synthesis priority   : {priority}")
    print(f"Honest verdict       : {honest_verdict}")
    print(f"Sensitivity          : {sensitivity}")
    print(f"Deliverable          : {output_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
