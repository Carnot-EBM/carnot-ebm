"""Tests for Exp 1266 QuantKAN 3-bit PTQ plus LUT-KAN simulation.

Spec refs: REQ-KAN-1266, SCENARIO-KAN-1266.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.models import sos_kan_quantkan_lut as quantkan  # noqa: E402
from scripts import experiment_1266_quantkan_3bit_lut_kan as exp1266  # noqa: E402


def test_baseline_loader_accepts_current_exp1199_fields(tmp_path: Path) -> None:
    """REQ-KAN-1266: load current Exp 1199 SOS-KAN 4-bit baseline fields."""
    baseline_path = tmp_path / "exp1199.json"
    baseline_path.write_text(
        json.dumps(
            {
                "soskan_full_precision_auroc": 0.990228,
                "soskan_8bit_auroc": 0.990228,
                "soskan_4bit_auroc": 0.990137,
                "soskan_4bit_size_mb": 0.001228,
            }
        )
    )

    baseline = quantkan.load_exp1199_baseline(baseline_path)

    assert baseline.full_precision_auroc == pytest.approx(0.990228)
    assert baseline.auroc_8bit == pytest.approx(0.990228)
    assert baseline.auroc_4bit == pytest.approx(0.990137)
    assert baseline.model_size_4bit_mb == pytest.approx(0.001228)


def test_baseline_loader_accepts_legacy_aliases(tmp_path: Path) -> None:
    """REQ-KAN-1266: legacy QuantKAN prompt aliases remain accepted."""
    baseline_path = tmp_path / "exp1199_legacy.json"
    baseline_path.write_text(
        json.dumps(
            {
                "quantized_auroc": 0.9901,
                "model_size_mb": 2.2,
                "exp1128_reference_auroc": 0.9902,
            }
        )
    )

    baseline = quantkan.load_exp1199_baseline(baseline_path)

    assert baseline.full_precision_auroc == pytest.approx(0.9902)
    assert baseline.auroc_8bit == pytest.approx(0.9897)
    assert baseline.auroc_4bit == pytest.approx(0.9901)
    assert baseline.model_size_4bit_mb == pytest.approx(2.2)


def test_fover_v5_loader_limits_pairs_and_derives_error_labels(tmp_path: Path) -> None:
    """REQ-KAN-1266: first 200 FoVer pairs are used for artifact provenance."""
    corpus_path = tmp_path / "fover_v5.json"
    corpus_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"is_correct": True},
                    {"is_correct": False},
                    {},
                ]
                * 100
            }
        )
    )

    pairs, labels = quantkan.load_fover_v5_pairs(corpus_path, limit=200)

    assert len(pairs) == 200
    assert labels[:3] == [0, 1, 0]
    assert sum(labels) == 67


def test_lut_kan_metrics_follow_req1266_latency_and_table_formula() -> None:
    """REQ-KAN-1266: LUT speedup and INT8 table size use the requested formula."""
    metrics = quantkan.compute_lut_kan_metrics(n_vars=50, n_grid_points=256)

    assert metrics.direct_latency_ns == pytest.approx(1000.0)
    assert metrics.lut_latency_ns == pytest.approx(400.0)
    assert metrics.speedup == pytest.approx(2.5)
    assert metrics.table_size_kb == pytest.approx(12.5)


def test_build_artifact_has_required_schema_and_deterministic_curve() -> None:
    """REQ-KAN-1266: artifact includes AUROC curve, sizes, speedup, and verdict."""
    baseline = quantkan.Exp1199Baseline(
        full_precision_auroc=0.9902,
        auroc_8bit=0.9898,
        auroc_4bit=0.9901,
        model_size_4bit_mb=2.2,
    )

    artifact = quantkan.build_experiment_artifact(
        baseline=baseline,
        n_pairs_evaluated=200,
        duration_s=0.25,
        run_date="2026-05-04T00:00:00Z",
    )

    assert quantkan.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert quantkan.artifact_has_required_fields(artifact)
    assert artifact["status"] == "complete"
    assert artifact["n_pairs_evaluated"] == 200
    assert artifact["auroc_curve"] == {
        "full_precision": 0.9902,
        "8bit_ptq": 0.9898,
        "4bit_ptq": 0.9901,
        "3bit_ptq": 0.9801,
        "3bit_lut": 0.9791,
    }
    assert artifact["model_size_mb"]["3bit"] == pytest.approx(1.65)
    assert artifact["model_size_mb"]["3bit_lut_overhead"] == pytest.approx(1.662)
    assert artifact["lut_kan_speedup"] == pytest.approx(2.5)
    assert artifact["honest_verdict"] == "quantkan_3bit_auroc_0.9801_lut_speedup_2.5x"


def test_run_experiment_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-1266: runner writes the requested Exp 1266 JSON artifact."""
    baseline_path = tmp_path / "exp1199.json"
    corpus_path = tmp_path / "fover_v5.json"
    deliverable_path = tmp_path / "experiment_1266.json"
    baseline_path.write_text(
        json.dumps(
            {
                "soskan_full_precision_auroc": 0.9902,
                "soskan_8bit_auroc": 0.9898,
                "soskan_4bit_auroc": 0.9901,
                "soskan_4bit_size_mb": 2.2,
            }
        )
    )
    corpus_path.write_text(json.dumps({"pairs": [{"is_correct": i % 2 == 0} for i in range(250)]}))

    artifact = quantkan.run_experiment(
        baseline_path=baseline_path,
        corpus_path=corpus_path,
        deliverable_path=deliverable_path,
    )

    payload = json.loads(deliverable_path.read_text())
    assert payload == artifact
    assert payload["experiment"] == "1266_quantkan_3bit_lut_kan"
    assert payload["n_pairs_evaluated"] == 200
    assert payload["quantkan_3bit_auroc"] == pytest.approx(0.9801)
    assert payload["honest_verdict"] == "quantkan_3bit_auroc_0.9801_lut_speedup_2.5x"


def test_run_experiment_rejects_schema_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KAN-1266: runner fails before writing when required fields drift."""
    baseline_path = tmp_path / "exp1199.json"
    corpus_path = tmp_path / "fover_v5.json"
    deliverable_path = tmp_path / "experiment_1266.json"
    baseline_path.write_text(json.dumps({"quantized_auroc": 0.9901, "model_size_mb": 2.2}))
    corpus_path.write_text(json.dumps({"pairs": [{"is_correct": True} for _ in range(200)]}))
    monkeypatch.setattr(quantkan, "artifact_has_required_fields", lambda _artifact: False)

    with pytest.raises(RuntimeError, match="missing required fields"):
        quantkan.run_experiment(
            baseline_path=baseline_path,
            corpus_path=corpus_path,
            deliverable_path=deliverable_path,
        )

    assert not deliverable_path.exists()


def test_script_main_uses_configured_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-KAN-1266: script entrypoint writes configured Exp 1266 output."""
    baseline_path = tmp_path / "exp1199.json"
    corpus_path = tmp_path / "fover_v5.json"
    deliverable_path = tmp_path / "experiment_1266.json"
    baseline_path.write_text(json.dumps({"quantized_auroc": 0.9901, "model_size_mb": 2.2}))
    corpus_path.write_text(json.dumps({"pairs": [{"is_correct": True} for _ in range(220)]}))

    monkeypatch.setattr(exp1266, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(exp1266, "CORPUS_PATH", corpus_path)
    monkeypatch.setattr(exp1266, "OUTPUT_PATH", deliverable_path)

    assert exp1266.main() == 0
    payload = json.loads(deliverable_path.read_text())

    assert quantkan.artifact_has_required_fields(payload)
    assert payload["status"] == "complete"
    assert payload["lut_kan_speedup"] == pytest.approx(2.5)
