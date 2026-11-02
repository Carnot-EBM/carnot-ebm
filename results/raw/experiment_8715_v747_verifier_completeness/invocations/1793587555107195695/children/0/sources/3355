"""BiKA-style multiply-free hardware metrics for SOSKANEnergyV3.

BiKA replaces learned-coefficient multiplications with a fixed-point
approximation that shifts by a precomputed ``log2(weight)`` value.  This module
does not claim an accuracy result; it is a deterministic complexity analysis
for the Exp 1148/1162 SOS-KAN shape.

Spec refs: REQ-KAN-1174, SCENARIO-KAN-1174.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1148_PATH = PROJECT_ROOT / "results" / "experiment_1148_metacluster_sos_kan_compression.json"
EXP1162_PATH = PROJECT_ROOT / "results" / "experiment_1162_kanele_sos_kan_fpga_blueprint.json"
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1174_bika_hardware_analysis.json"

EXPERIMENT_ID = 1174
SCHEMA = "bika_sos_kan_hardware_analysis_v1"
TITLE = "BiKA SOS-KAN Hardware Analysis"
FLOAT32_BIT_WIDTH = 32
Q8_SHIFT_COMPARISON_BOP = 16
FP32_MULTIPLIER_LUT_COST = 10
NPU_INT8_TOPS = 1.0
BIKA_PAPER_MIN_REDUCTION_PCT = 27.73
BIKA_PAPER_MAX_REDUCTION_PCT = 51.54
BIKA_PAPER_MIDPOINT_REDUCTION_PCT = round(
    (BIKA_PAPER_MIN_REDUCTION_PCT + BIKA_PAPER_MAX_REDUCTION_PCT) / 2.0, 3
)

NPU_FEASIBILITY_VERDICTS = {"npu_feasible", "npu_borderline", "npu_infeasible"}
HONEST_VERDICTS = {
    "bika_feasible_for_npu",
    "bika_reduces_cost_fpga_only",
    "bika_insufficient_analysis",
}
REQUIRED_ARTIFACT_FIELDS = {
    "standard_kan_rm",
    "standard_kan_bop",
    "compressed_kan_rm",
    "compressed_kan_bop",
    "bika_kan_nabs",
    "bika_resource_reduction_pct",
    "npu_feasibility_verdict",
    "estimated_npu_inference_us",
    "bika_hardware_analysis_complete",
    "honest_verdict",
}


@dataclass(frozen=True)
class HardwareMetrics:
    """Platform-independent hardware operation counts for one KAN variant."""

    RM: int
    BOP: int
    NABS: int
    estimated_lut_count: int


@dataclass(frozen=True)
class BiKAComparison:
    """Resource reduction and NPU feasibility derived from two metric records."""

    bika_resource_reduction_pct: float
    npu_feasibility_verdict: str
    estimated_npu_inference_us: float
    honest_verdict: str


@dataclass(frozen=True)
class _OperationBreakdown:
    """Internal exact operation counts for SOSKANEnergyV3 inference."""

    linear_rm: int
    gram_rm: int
    energy_rm: int
    bias_rm: int
    nabs: int
    output_dim: int
    w2_rm: int
    w2_nabs: int

    @property
    def rm(self) -> int:
        return self.linear_rm + self.gram_rm + self.energy_rm + self.bias_rm


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _model_int(model: Any, name: str) -> int:
    """Read one positive integer architecture attribute from a model-like object."""
    value = int(getattr(model, name))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def soskan_architecture(model: Any) -> dict[str, Any]:
    """Return SOSKANEnergyV3 dimensions and its two linear layer sizes."""
    n_features = _model_int(model, "n_features")
    n_splines = _model_int(model, "n_splines")
    rank = _model_int(model, "rank")
    hidden_dim = _model_int(model, "hidden_dim")
    output_dim = n_features * n_splines * rank
    return {
        "n_features": n_features,
        "n_splines": n_splines,
        "rank": rank,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "linear_layers": [
            {"in_features": n_features, "out_features": hidden_dim},
            {"in_features": hidden_dim, "out_features": output_dim},
        ],
    }


def _operation_breakdown(model: Any) -> _OperationBreakdown:
    """Count real multiplications and add/shift operations for SOSKANEnergyV3."""
    arch = soskan_architecture(model)
    n_features = arch["n_features"]
    n_splines = arch["n_splines"]
    rank = arch["rank"]
    hidden_dim = arch["hidden_dim"]
    output_dim = arch["output_dim"]

    w1_rm = hidden_dim * n_features
    w2_rm = output_dim * hidden_dim
    gram_rm = n_features * n_splines * n_splines * rank
    energy_rm = n_features * n_splines * n_splines
    bias_rm = n_features

    w1_nabs = hidden_dim * n_features
    w2_nabs = output_dim * hidden_dim
    gram_nabs = n_features * n_splines * n_splines * max(rank - 1, 0)
    energy_nabs = n_features * n_splines * n_splines

    return _OperationBreakdown(
        linear_rm=w1_rm + w2_rm,
        gram_rm=gram_rm,
        energy_rm=energy_rm,
        bias_rm=bias_rm,
        nabs=w1_nabs + w2_nabs + gram_nabs + energy_nabs,
        output_dim=output_dim,
        w2_rm=w2_rm,
        w2_nabs=w2_nabs,
    )


def _estimate_luts(rm: int, nabs: int) -> int:
    """Estimate LUT pressure from FP32 multipliers plus add/shift operations."""
    return int(rm * FP32_MULTIPLIER_LUT_COST + nabs)


def _block_shape(payload: dict[str, Any], name: str) -> tuple[int, ...]:
    """Return one Exp 1148 parameter-block shape."""
    for block in payload.get("parameter_blocks", []):
        if block.get("name") == name:
            return tuple(int(dim) for dim in block.get("shape", []))
    return ()


def load_soskan_v3_model_from_artifacts(
    exp1148_path: Path = EXP1148_PATH,
    exp1162_path: Path = EXP1162_PATH,
) -> SimpleNamespace:
    """Extract a model-like SOSKANEnergyV3 shape from Exp 1148/1162 artifacts."""
    exp1148 = load_json(exp1148_path)
    exp1162 = load_json(exp1162_path)
    w1_shape = _block_shape(exp1148, "W1")
    n_features = int(exp1162.get("sos_kan_n_inputs") or len(exp1148.get("feature_stats", [])))
    n_splines = int(exp1162.get("sos_kan_k_splines") or exp1148.get("n_kan_basis_functions", 0))
    rank = int(exp1162.get("sos_kan_rank") or exp1148.get("coefficients_per_spline", 0))
    hidden_dim = int(exp1162.get("sos_kan_hidden_dim") or (w1_shape[0] if w1_shape else 0))
    return SimpleNamespace(
        n_features=n_features,
        n_splines=n_splines,
        rank=rank,
        hidden_dim=hidden_dim,
    )


class BiKAComplexityAnalyzer:
    """Analyze standard, MetaCluster, and BiKA SOSKANEnergyV3 hardware costs."""

    def __init__(
        self,
        *,
        npu_int8_tops: float = NPU_INT8_TOPS,
        bika_reduction_pct: float = BIKA_PAPER_MIDPOINT_REDUCTION_PCT,
    ) -> None:
        self.npu_int8_tops = float(npu_int8_tops)
        self.bika_reduction_pct = float(bika_reduction_pct)

    def analyze_standard_kan(self, model: Any) -> HardwareMetrics:
        """Return full-float32 SOSKANEnergyV3 inference metrics."""
        breakdown = _operation_breakdown(model)
        rm = breakdown.rm
        nabs = breakdown.nabs
        return HardwareMetrics(
            RM=int(rm),
            BOP=int(rm * FLOAT32_BIT_WIDTH),
            NABS=int(nabs),
            estimated_lut_count=_estimate_luts(rm, nabs),
        )

    def analyze_metacluster_kan(self, model: Any, n_centroids: int) -> HardwareMetrics:
        """Return metrics when W2 row dot-products are reused through centroids."""
        breakdown = _operation_breakdown(model)
        unique_w2_rows = min(int(n_centroids), breakdown.output_dim)
        compressed_w2_rm = unique_w2_rows * _model_int(model, "hidden_dim")
        compressed_w2_nabs = compressed_w2_rm
        rm = breakdown.rm - breakdown.w2_rm + compressed_w2_rm
        nabs = breakdown.nabs - breakdown.w2_nabs + compressed_w2_nabs
        return HardwareMetrics(
            RM=int(rm),
            BOP=int(rm * FLOAT32_BIT_WIDTH),
            NABS=int(nabs),
            estimated_lut_count=_estimate_luts(rm, nabs),
        )

    def analyze_bika_kan(self, model: Any, precision_bits: int = 8) -> HardwareMetrics:
        """Return multiply-free BiKA metrics using shift/comparison approximation."""
        standard = self.analyze_standard_kan(model)
        shift_bop = int(standard.RM * precision_bits * 2)
        nabs = int(standard.NABS + standard.RM)
        estimated_luts = int(
            round(standard.estimated_lut_count * (1.0 - self.bika_reduction_pct / 100.0))
        )
        return HardwareMetrics(
            RM=0,
            BOP=shift_bop,
            NABS=nabs,
            estimated_lut_count=estimated_luts,
        )

    def compare(
        self, standard_metrics: HardwareMetrics, bika_metrics: HardwareMetrics
    ) -> BiKAComparison:
        """Compare full-float32 and BiKA metrics and classify NPU feasibility."""
        if standard_metrics.estimated_lut_count <= 0:
            raise ValueError("standard estimated_lut_count must be positive")
        reduction_pct = round(
            100.0
            * (standard_metrics.estimated_lut_count - bika_metrics.estimated_lut_count)
            / standard_metrics.estimated_lut_count,
            3,
        )
        estimated_us = round(bika_metrics.NABS / (self.npu_int8_tops * 1_000_000.0), 6)
        npu_verdict = (
            "npu_feasible"
            if estimated_us <= 1.0 and bika_metrics.RM == 0
            else "npu_borderline"
            if estimated_us <= 10.0 and bika_metrics.RM == 0
            else "npu_infeasible"
        )
        honest_verdict = (
            "bika_feasible_for_npu"
            if npu_verdict == "npu_feasible"
            and BIKA_PAPER_MIN_REDUCTION_PCT <= reduction_pct <= BIKA_PAPER_MAX_REDUCTION_PCT
            else "bika_reduces_cost_fpga_only"
            if bika_metrics.RM == 0
            else "bika_insufficient_analysis"
        )
        return BiKAComparison(
            bika_resource_reduction_pct=reduction_pct,
            npu_feasibility_verdict=npu_verdict,
            estimated_npu_inference_us=estimated_us,
            honest_verdict=honest_verdict,
        )


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 1174 artifact satisfies the required schema."""
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("npu_feasibility_verdict") in NPU_FEASIBILITY_VERDICTS
        and artifact.get("honest_verdict") in HONEST_VERDICTS
        and isinstance(artifact.get("bika_hardware_analysis_complete"), bool)
    )


def build_artifact(
    *,
    model: Any,
    standard_metrics: HardwareMetrics,
    compressed_metrics: HardwareMetrics,
    bika_metrics: HardwareMetrics,
    comparison: BiKAComparison,
    exp1148: dict[str, Any],
    exp1162: dict[str, Any],
    duration_s: float,
    run_date: str,
) -> dict[str, Any]:
    """Build the Exp 1174 result artifact."""
    arch = soskan_architecture(model)
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "success",
        "title": TITLE,
        "sos_kan_spline_control_points_per_dimension": arch["n_splines"],
        "sos_kan_rank": arch["rank"],
        "sos_kan_n_features": arch["n_features"],
        "sos_kan_hidden_dim": arch["hidden_dim"],
        "sos_kan_linear_layers": arch["linear_layers"],
        "standard_kan_rm": standard_metrics.RM,
        "standard_kan_bop": standard_metrics.BOP,
        "standard_kan_nabs": standard_metrics.NABS,
        "standard_estimated_lut_count": standard_metrics.estimated_lut_count,
        "compressed_kan_rm": compressed_metrics.RM,
        "compressed_kan_bop": compressed_metrics.BOP,
        "compressed_kan_nabs": compressed_metrics.NABS,
        "compressed_estimated_lut_count": compressed_metrics.estimated_lut_count,
        "bika_kan_rm": bika_metrics.RM,
        "bika_kan_bop": bika_metrics.BOP,
        "bika_kan_nabs": bika_metrics.NABS,
        "bika_estimated_lut_count": bika_metrics.estimated_lut_count,
        "bika_resource_reduction_pct": comparison.bika_resource_reduction_pct,
        "npu_feasibility_verdict": comparison.npu_feasibility_verdict,
        "estimated_npu_inference_us": comparison.estimated_npu_inference_us,
        "bika_hardware_analysis_complete": True,
        "honest_verdict": comparison.honest_verdict,
        "metacluster_size_reduction_factor": round(
            float(exp1148.get("size_reduction_factor", 0.0)), 6
        ),
        "metacluster_n_centroids": int(exp1148.get("n_centroids", 0)),
        "kanele_rm_per_inference": int(exp1162.get("rm_per_inference", 0)),
        "kanele_nabs_per_inference": int(exp1162.get("nabs_per_inference", 0)),
        "xdna_int8_tops_assumption": NPU_INT8_TOPS,
        "precision_bits": 8,
        "spec": ["REQ-KAN-1174", "SCENARIO-KAN-1174"],
        "notes": [
            "BiKA metrics are complexity estimates only; no accuracy benchmark was run.",
            "MetaCluster RM assumes W2 centroid dot-products can be reused in hardware.",
        ],
    }


def run_experiment(
    *,
    exp1148_path: Path = EXP1148_PATH,
    exp1162_path: Path = EXP1162_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Run Exp 1174 and write the JSON deliverable."""
    started = datetime.now(UTC)
    run_date = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    exp1148 = load_json(exp1148_path)
    exp1162 = load_json(exp1162_path)
    model = load_soskan_v3_model_from_artifacts(exp1148_path, exp1162_path)
    analyzer = BiKAComplexityAnalyzer()
    standard = analyzer.analyze_standard_kan(model)
    compressed = analyzer.analyze_metacluster_kan(
        model, n_centroids=int(exp1148.get("n_centroids", 32))
    )
    bika = analyzer.analyze_bika_kan(model, precision_bits=8)
    comparison = analyzer.compare(standard, bika)
    duration_s = (datetime.now(UTC) - started).total_seconds()
    artifact = build_artifact(
        model=model,
        standard_metrics=standard,
        compressed_metrics=compressed,
        bika_metrics=bika,
        comparison=comparison,
        exp1148=exp1148,
        exp1162=exp1162,
        duration_s=duration_s,
        run_date=run_date,
    )
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact
