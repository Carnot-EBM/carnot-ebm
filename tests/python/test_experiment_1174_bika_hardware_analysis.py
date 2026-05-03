"""Tests for Exp 1174 BiKA SOS-KAN hardware analysis.

Spec refs: REQ-KAN-1174, SCENARIO-KAN-1174.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.hardware.bika_analysis import (  # noqa: E402
    BIKA_PAPER_MAX_REDUCTION_PCT,
    BIKA_PAPER_MIN_REDUCTION_PCT,
    REQUIRED_ARTIFACT_FIELDS,
    BiKAComplexityAnalyzer,
    HardwareMetrics,
    artifact_has_required_fields,
    build_artifact,
    load_soskan_v3_model_from_artifacts,
    load_json,
    run_experiment,
    _block_shape,
)
from scripts import experiment_1174_bika_hardware_analysis as exp1174  # noqa: E402


def _soskan_shape() -> SimpleNamespace:
    return SimpleNamespace(n_features=3, n_splines=8, rank=4, hidden_dim=16)


def test_standard_soskan_v3_rm_bop_and_nabs_are_architecture_derived() -> None:
    """REQ-KAN-1174: standard metrics derive from SOSKANEnergyV3 layer sizes."""
    analyzer = BiKAComplexityAnalyzer()

    metrics = analyzer.analyze_standard_kan(_soskan_shape())

    assert metrics.RM == 2547
    assert metrics.BOP == 2547 * 32
    assert metrics.NABS == 2352
    assert metrics.estimated_lut_count > metrics.RM


def test_metacluster_metric_uses_centroid_reuse_for_w2_layer() -> None:
    """REQ-KAN-1174: MetaCluster compression lowers the W2 real-multiply count."""
    analyzer = BiKAComplexityAnalyzer()

    metrics = analyzer.analyze_metacluster_kan(_soskan_shape(), n_centroids=32)

    assert metrics.RM == 1523
    assert metrics.BOP == 1523 * 32
    assert (
        metrics.estimated_lut_count
        < analyzer.analyze_standard_kan(_soskan_shape()).estimated_lut_count
    )


def test_bika_metrics_remove_real_multiplications_and_count_shift_ops() -> None:
    """REQ-KAN-1174: BiKA replaces RM with 8-bit shift/comparison BOP and NABS."""
    analyzer = BiKAComplexityAnalyzer()
    standard = analyzer.analyze_standard_kan(_soskan_shape())

    bika = analyzer.analyze_bika_kan(_soskan_shape(), precision_bits=8)

    assert bika.RM == 0
    assert bika.BOP == standard.RM * 16
    assert bika.NABS == standard.NABS + standard.RM
    assert bika.estimated_lut_count < standard.estimated_lut_count


def test_compare_reports_paper_band_reduction_and_npu_feasibility() -> None:
    """REQ-KAN-1174: compare emits BiKA resource reduction and NPU verdict."""
    analyzer = BiKAComplexityAnalyzer()
    standard = analyzer.analyze_standard_kan(_soskan_shape())
    bika = analyzer.analyze_bika_kan(_soskan_shape(), precision_bits=8)

    comparison = analyzer.compare(standard, bika)

    assert BIKA_PAPER_MIN_REDUCTION_PCT <= comparison.bika_resource_reduction_pct
    assert comparison.bika_resource_reduction_pct <= BIKA_PAPER_MAX_REDUCTION_PCT
    assert comparison.npu_feasibility_verdict == "npu_feasible"
    assert comparison.estimated_npu_inference_us < 1.0
    assert comparison.honest_verdict == "bika_feasible_for_npu"


def test_build_artifact_has_required_schema_fields() -> None:
    """REQ-KAN-1174: artifact carries the required Exp 1174 schema."""
    analyzer = BiKAComplexityAnalyzer()
    standard = analyzer.analyze_standard_kan(_soskan_shape())
    compressed = analyzer.analyze_metacluster_kan(_soskan_shape(), n_centroids=32)
    bika = analyzer.analyze_bika_kan(_soskan_shape(), precision_bits=8)
    comparison = analyzer.compare(standard, bika)

    artifact = build_artifact(
        model=_soskan_shape(),
        standard_metrics=standard,
        compressed_metrics=compressed,
        bika_metrics=bika,
        comparison=comparison,
        exp1148={"size_reduction_factor": 5.026627, "n_centroids": 32},
        exp1162={"rm_per_inference": 24, "nabs_per_inference": 75},
        duration_s=0.1,
        run_date="2026-05-02T00:00:00Z",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert artifact["sos_kan_spline_control_points_per_dimension"] == 8
    assert artifact["sos_kan_linear_layers"] == [
        {"in_features": 3, "out_features": 16},
        {"in_features": 16, "out_features": 96},
    ]
    assert artifact["standard_kan_rm"] == 2547
    assert artifact["compressed_kan_rm"] == 1523
    assert artifact["bika_kan_nabs"] == bika.NABS
    assert artifact["bika_hardware_analysis_complete"] is True


def test_required_schema_rejects_incomplete_or_unknown_verdicts() -> None:
    """REQ-KAN-1174: artifact validation rejects missing fields and bad verdicts."""
    complete = {field: 1 for field in REQUIRED_ARTIFACT_FIELDS}
    complete["npu_feasibility_verdict"] = "npu_feasible"
    complete["honest_verdict"] = "bika_feasible_for_npu"
    complete["bika_hardware_analysis_complete"] = True

    assert artifact_has_required_fields(complete)
    assert not artifact_has_required_fields({"standard_kan_rm": 1})
    bad = dict(complete)
    bad["honest_verdict"] = "optimistic"
    assert not artifact_has_required_fields(bad)


def test_load_model_from_existing_artifacts_extracts_soskan_shape() -> None:
    """SCENARIO-KAN-1174: Exp 1148/1162 artifacts provide the SOS-KAN shape."""
    model = load_soskan_v3_model_from_artifacts(
        exp1148_path=exp1174.EXP1148_PATH,
        exp1162_path=exp1174.EXP1162_PATH,
    )

    assert model.n_features == 3
    assert model.n_splines == 8
    assert model.rank == 4
    assert model.hidden_dim == 16


def test_run_experiment_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-1174: runner writes the BiKA hardware analysis artifact."""
    deliverable = tmp_path / "experiment_1174_bika_hardware_analysis.json"

    artifact = run_experiment(deliverable_path=deliverable)

    payload = json.loads(deliverable.read_text())
    assert payload == artifact
    assert artifact_has_required_fields(payload)
    assert payload["bika_hardware_analysis_complete"] is True
    assert payload["honest_verdict"] in {
        "bika_feasible_for_npu",
        "bika_reduces_cost_fpga_only",
        "bika_insufficient_analysis",
    }


def test_script_main_uses_configured_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-KAN-1174: script entrypoint writes the configured artifact path."""
    deliverable = tmp_path / "experiment_1174.json"
    monkeypatch.setattr(exp1174, "DELIVERABLE", deliverable)

    assert exp1174.main() == 0
    payload = json.loads(deliverable.read_text())

    assert artifact_has_required_fields(payload)
    assert payload["standard_kan_rm"] == 2547
    assert payload["npu_feasibility_verdict"] == "npu_feasible"


def test_compare_rejects_zero_standard_luts() -> None:
    """REQ-KAN-1174: invalid comparison inputs fail clearly."""
    analyzer = BiKAComplexityAnalyzer()

    with pytest.raises(ValueError, match="standard"):
        analyzer.compare(
            HardwareMetrics(RM=0, BOP=0, NABS=0, estimated_lut_count=0),
            HardwareMetrics(RM=0, BOP=0, NABS=0, estimated_lut_count=0),
        )


def test_invalid_inputs_fail_clearly(tmp_path: Path) -> None:
    """REQ-KAN-1174: malformed JSON and invalid model dimensions are rejected."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_json)
    with pytest.raises(ValueError, match="n_features"):
        BiKAComplexityAnalyzer().analyze_standard_kan(
            SimpleNamespace(n_features=0, n_splines=8, rank=4, hidden_dim=16)
        )
    assert _block_shape({}, "W1") == ()
