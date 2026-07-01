"""Tests for Exp 5128 KAN certificate explanation audit.

Spec refs: REQ-KAN-5128, SCENARIO-KAN-5128.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5128_kan_certificate_explanation_v470 as mod
from scripts import experiment_5128_kan_certificate_explanation_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_kan_5128_spec_declares_certificate_explanation_contract() -> None:
    """REQ-KAN-5128: OpenSpec anchors breadth and cycle-consistent explanations."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("## REQ-KAN-5128")
    end = spec.index("## Implementation Status", start)
    section = spec[start:end]

    assert "SCENARIO-KAN-5128" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.MILESTONE in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_kan_5128_emits_independent_machine_readable_certificates() -> None:
    """REQ-KAN-5128: certificates span true, false-control, and abstention families."""

    certificates = mod.build_explainable_certificates(n_units=20, seed=mod.RANDOM_SEED + 20)
    families = {certificate["property"]["family"] for certificate in certificates}
    verdicts = {certificate["verdict"] for certificate in certificates}
    proof_statuses = {certificate["proof_status"] for certificate in certificates}

    assert len(certificates) >= 3
    assert families == {
        "global_energy_upper_bound",
        "refinement_error_budget",
        "false_low_threshold_control",
        "near_margin_residual_gap",
    }
    assert {"verified", "counterexample", "abstained"}.issubset(verdicts)
    assert "refuted_by_exact_witness" in proof_statuses
    assert "abstained_residual_gap" in proof_statuses
    for certificate in certificates:
        assert set(mod.REQUIRED_CERTIFICATE_FIELDS).issubset(certificate)
        assert certificate["property"]["id"].startswith("kan_n20_")
        assert certificate["margin"] >= 0.0
        assert certificate["abstraction_error"] >= 0.0

    summary = mod.validate_certificates(certificates)
    assert summary["certificate_soundness"] is True
    assert summary["false_property_detected"] is True
    assert summary["near_margin_abstained"] is True


def test_scenario_kan_5128_explanation_cycle_reconstructs_and_symbolically_checks() -> None:
    """SCENARIO-KAN-5128: explanation metadata round-trips through the symbolic validator."""

    certificates = mod.build_explainable_certificates(n_units=20, seed=mod.RANDOM_SEED + 20)
    records = mod.generate_explanation_records(certificates)

    assert len(records) == len(certificates)
    assert all(record["cycle_sound"] for record in records)
    assert records == mod.generate_explanation_records(certificates)
    for certificate, record in zip(certificates, records, strict=True):
        reconstructed = mod.reconstruct_metadata_from_explanation(record["explanation"])
        assert reconstructed == record["reconstructed_metadata"]
        assert (
            mod.symbolic_validate_explanation(certificate, record["explanation"])["valid"] is True
        )

    tampered = records[0]["explanation"].replace("verdict=verified", "verdict=counterexample")
    invalid = mod.symbolic_validate_explanation(certificates[0], tampered)
    assert invalid["valid"] is False
    assert "verdict" in invalid["mismatches"]


def test_req_kan_5128_artifact_reports_required_fields_and_controls(tmp_path: Path) -> None:
    """REQ-KAN-5128: artifact preserves baselines and reports breadth/cycle outcomes."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(artifact_path=artifact_path, run_date="20260701")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["exp5108_wall_loaded"] is True
    assert artifact["exp5114_baseline_loaded"] is True
    assert len(artifact["property_families"]) >= 3
    assert len(artifact["certificates_emitted"]) == len(artifact["property_families"])
    assert artifact["certificate_soundness"] is True
    assert artifact["false_property_detected"] is True
    assert artifact["near_margin_abstained"] is True
    assert artifact["explanation_cycle_soundness"] is True
    assert artifact["kan_certificate_breadth_ready"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["conductor_modified"] is False


def test_req_kan_5128_blocked_preconditions_validate_and_cli_writes(tmp_path: Path, capsys) -> None:
    """REQ-KAN-5128: missing baselines fail closed, while the CLI writes the artifact."""

    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    blocked = mod.build_artifact(root=missing_root, run_date="20260701")
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["exp5108_wall_loaded"] is False
    assert blocked["exp5114_baseline_loaded"] is False
    assert blocked["certificates_emitted"] == []
    assert blocked["explanation_records"] == []
    assert blocked["kan_certificate_breadth_ready"] is False
    assert blocked["flagged_adversarial"] is True

    output = tmp_path / "cli-result.json"
    assert script_mod.main(["--date", "20260701", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "success_kan_certificate_explanation_cycle_sound" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID


def test_req_kan_5128_validation_edges_cover_closed_world_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KAN-5128: malformed certificates and not-ready audits fail closed."""

    certificates = mod.build_explainable_certificates(n_units=20, seed=mod.RANDOM_SEED + 20)
    missing_field = dict(certificates[0])
    del missing_field["property"]
    non_mapping_property = dict(certificates[0], property="bad")
    negative_margin = dict(certificates[0], margin=-1.0)
    unknown_proof = dict(certificates[0], proof_status="unknown")

    assert mod._sha256_file(tmp_path / "missing.json") is None
    assert mod._certificate_sound(missing_field) is False
    assert mod._certificate_sound(non_mapping_property) is False
    assert mod._certificate_sound(negative_margin) is False
    assert mod._certificate_sound(unknown_proof) is False
    assert mod.reconstruct_metadata_from_explanation("not-a-certificate") == {}

    monkeypatch.setattr(
        mod,
        "validate_certificates",
        lambda _certificates: {
            "property_family_count": 4,
            "certificate_soundness": False,
            "false_property_detected": True,
            "near_margin_abstained": True,
        },
    )
    artifact = mod.build_artifact(run_date="20260701")
    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete_kan_certificate_explanation_breadth_not_ready"
    assert artifact["flagged_adversarial"] is True


def test_deliverable_file_validates_for_scenario_kan_5128() -> None:
    """SCENARIO-KAN-5128: committed JSON deliverable satisfies the V470 contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["kan_certificate_breadth_ready"] is True
    assert artifact["explanation_cycle_soundness"] is True
    assert artifact["false_property_detected"] is True
    assert artifact["near_margin_abstained"] is True
