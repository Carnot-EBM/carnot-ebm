"""Tests for Exp 1162 KANELE SOS-KAN FPGA LUT blueprint.

Spec refs: REQ-KAN-1162, SCENARIO-KAN-1162.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.analysis import kanele_sos_kan_fpga as kanele  # noqa: E402
from scripts import experiment_1162_kanele_sos_kan_fpga_blueprint as exp1162  # noqa: E402


def _structure() -> kanele.SOSKANLUTStructure:
    return kanele.SOSKANLUTStructure(
        n_inputs=3,
        k_splines=8,
        n_knots=8,
        rank=4,
        hidden_dim=16,
        knot_positions=tuple(float(v) for v in np.linspace(-1.0, 1.0, 8)),
        model_source="test",
    )


def test_exp1148_structure_is_loaded_from_result_and_model_shape() -> None:
    """REQ-KAN-1162: derive SOSKANEnergyV3 dimensions from Exp 1148 structure."""
    structure = kanele.load_sos_kan_structure(kanele.EXP1148_PATH)

    assert structure.n_inputs == 3
    assert structure.k_splines == 8
    assert structure.n_knots == 8
    assert structure.rank == 4
    assert structure.hidden_dim == 16
    assert structure.knot_positions[0] == pytest.approx(-1.0)
    assert structure.knot_positions[-1] == pytest.approx(1.0)
    assert "SOSKANEnergyV3" in structure.model_source


def test_q8_lut_spec_samples_every_hat_basis_function() -> None:
    """REQ-KAN-1162: Q8 tables cover n_inputs * k_splines * 256 bytes."""
    lut_spec = kanele.build_q8_lut_spec(_structure(), n_lut_points=256)

    assert lut_spec.n_lut_points == 256
    assert lut_spec.storage_bytes == 3 * 8 * 256
    assert lut_spec.q8_tables.shape == (3, 8, 256)
    assert lut_spec.q8_tables.dtype == np.uint8
    assert lut_spec.q8_tables[0, 0, 0] == 255
    assert lut_spec.q8_tables[0, -1, -1] == 255
    assert np.array_equal(lut_spec.q8_tables[0], lut_spec.q8_tables[1])
    assert lut_spec.table_sha256


def test_lut_index_and_interpolation_follow_kanele_datapath() -> None:
    """SCENARIO-KAN-1162: index is floor((x + 1) / 2 * 255) with interpolation."""
    table = np.arange(256, dtype=np.uint8)

    assert kanele.lut_index_and_fraction(-2.0, n_lut_points=256) == (0, 0.0)
    assert kanele.lut_index_and_fraction(-1.0, n_lut_points=256) == (0, 0.0)
    assert kanele.lut_index_and_fraction(1.0, n_lut_points=256) == (254, 1.0)
    assert kanele.lut_index_and_fraction(2.0, n_lut_points=256) == (254, 1.0)
    assert kanele.interpolate_q8(table, -1.0) == pytest.approx(0.0)
    assert kanele.interpolate_q8(table, 1.0) == pytest.approx(1.0)
    assert kanele.interpolate_q8(table, 0.0) == pytest.approx(0.5, abs=1 / 255)


def test_complexity_metrics_and_latency_formula() -> None:
    """REQ-KAN-1162: RM/BOP/NABS and KV260 latency are deterministic."""
    metrics = kanele.compute_complexity_metrics(_structure(), q8_operand_width=8)

    assert metrics.rm_per_inference == 24
    assert metrics.bop_per_inference == 192
    assert metrics.nabs_per_inference == 75
    assert metrics.total_cycles == 36
    assert metrics.estimated_fpga_latency_us == pytest.approx(0.12)
    assert metrics.estimated_speedup_factor == pytest.approx((289.0 * 1000.0) / 0.12)


def test_build_artifact_has_required_schema_and_verdict() -> None:
    """REQ-KAN-1162: artifact carries all required fields and approved verdict."""
    structure = _structure()
    lut_spec = kanele.build_q8_lut_spec(structure)
    metrics = kanele.compute_complexity_metrics(structure)

    artifact = kanele.build_artifact(
        structure=structure,
        lut_spec=lut_spec,
        metrics=metrics,
        exp1148={
            "size_compressed_bytes": 2704,
            "auroc_compressed": 0.971753,
            "n_centroids": 32,
            "energy_correlation": 0.996599,
        },
        blueprint_written=True,
        duration_s=0.25,
        run_date="2026-05-02T00:00:00Z",
        blueprint_path=Path("hardware/kv260/sos_kan_lut_blueprint.md"),
    )

    assert kanele.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["sos_kan_n_inputs"] == 3
    assert artifact["sos_kan_k_splines"] == 8
    assert artifact["n_lut_points"] == 256
    assert artifact["lut_storage_bytes"] == 6144
    assert artifact["auroc_compressed"] == 0.9718
    assert artifact["kanele_fpga_blueprint_generated"] is True
    assert artifact["honest_verdict"] == "blueprint_generated_speedup_above_100x"


def test_structure_not_found_artifact_is_honest() -> None:
    """REQ-KAN-1162: missing SOS-KAN structure maps to the approved blocked verdict."""
    artifact = kanele.build_structure_not_found_artifact(
        reason="missing class",
        duration_s=0.1,
        run_date="2026-05-02T00:00:00Z",
        blueprint_path=Path("hardware/kv260/sos_kan_lut_blueprint.md"),
    )

    assert kanele.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["sos_kan_n_inputs"] == 0
    assert artifact["kanele_fpga_blueprint_generated"] is True
    assert artifact["blueprint_written"] is False
    assert artifact["honest_verdict"] == "sos_kan_structure_not_found"


def test_validation_and_blocked_branches_are_schema_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KAN-1162: invalid structure inputs fail clearly and still write a schema."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]")

    assert kanele.utc_now_iso().endswith("Z")
    with pytest.raises(ValueError, match="expected JSON object"):
        kanele.load_json(bad_json)
    assert kanele._parameter_block_shape({}, "W1") == ()
    with pytest.raises(kanele.SOSKANStructureError, match="Exp 1148 result"):
        kanele._extract_exp1148_structure_fields({})
    with pytest.raises(ValueError, match="at least two"):
        kanele.hat_basis_matrix(np.array([0.0]), (0.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        kanele.hat_basis_matrix(np.array([0.0]), (1.0, 0.0))
    with pytest.raises(ValueError, match="n_lut_points"):
        kanele.build_q8_lut_spec(_structure(), n_lut_points=1)
    with pytest.raises(ValueError, match="n_lut_points"):
        kanele.lut_index_and_fraction(0.0, n_lut_points=1)

    monkeypatch.setattr(kanele, "PROJECT_ROOT", tmp_path)
    with pytest.raises(kanele.SOSKANStructureError, match="source file not found"):
        kanele._resolve_sos_kan_energy_v3()
    fake_model_dir = tmp_path / "python" / "carnot" / "models"
    fake_model_dir.mkdir(parents=True)
    (fake_model_dir / "sos_kan.py").write_text("class NotSOSKAN: pass\n")
    with pytest.raises(kanele.SOSKANStructureError, match="was not found"):
        kanele._resolve_sos_kan_energy_v3()
    assert kanele.classify_honest_verdict(50.0) == "blueprint_generated_speedup_below_100x"

    blocked = kanele.run_experiment(
        exp1148_path=bad_json,
        deliverable_path=tmp_path / "blocked.json",
        blueprint_path=tmp_path / "blocked.md",
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"] == "sos_kan_structure_not_found"
    assert kanele.REQUIRED_ARTIFACT_FIELDS <= set(blocked)


def test_blueprint_writer_documents_no_vivado_lut_datapath(tmp_path: Path) -> None:
    """SCENARIO-KAN-1162: blueprint documents Q8 LUT lookup and interpolation."""
    structure = _structure()
    lut_spec = kanele.build_q8_lut_spec(structure)
    metrics = kanele.compute_complexity_metrics(structure)
    path = tmp_path / "sos_kan_lut_blueprint.md"

    assert kanele.write_blueprint(
        path, structure, lut_spec, metrics, {"auroc_compressed": 0.971753}
    )
    text = path.read_text()

    assert "REQ-KAN-1162" in text
    assert "No Vivado synthesis was run" in text
    assert "floor((x_j + 1) / 2 * 255)" in text
    assert "Q8" in text
    assert "linear interpolation" in text
    assert str(metrics.rm_per_inference) in text


def test_run_experiment_writes_deliverable_and_blueprint(tmp_path: Path) -> None:
    """SCENARIO-KAN-1162: runner writes the JSON artifact and markdown blueprint."""
    deliverable = tmp_path / "experiment_1162.json"
    blueprint = tmp_path / "sos_kan_lut_blueprint.md"

    artifact = kanele.run_experiment(
        exp1148_path=kanele.EXP1148_PATH,
        deliverable_path=deliverable,
        blueprint_path=blueprint,
    )

    payload = json.loads(deliverable.read_text())
    assert payload == artifact
    assert kanele.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["blueprint_written"] is True
    assert payload["blueprint_path"] == "hardware/kv260/sos_kan_lut_blueprint.md"
    assert payload["estimated_speedup_factor"] > 100.0
    assert blueprint.exists()


def test_script_main_uses_configured_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-KAN-1162: script entrypoint writes configured Exp 1162 artifacts."""
    deliverable = tmp_path / "experiment_1162.json"
    blueprint = tmp_path / "sos_kan_lut_blueprint.md"

    monkeypatch.setattr(exp1162, "DELIVERABLE", deliverable)
    monkeypatch.setattr(exp1162, "BLUEPRINT_PATH", blueprint)

    assert exp1162.main() == 0
    payload = json.loads(deliverable.read_text())

    assert payload["kanele_fpga_blueprint_generated"] is True
    assert payload["blueprint_written"] is True
    assert payload["honest_verdict"] in kanele.HONEST_VERDICTS
    assert blueprint.exists()
