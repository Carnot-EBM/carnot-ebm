"""Tests for Exp 5114 KAN abstraction-refinement post-wall diagnostic.

Spec refs: REQ-KAN-5114, SCENARIO-KAN-5114.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5114_kan_abstraction_refinement_post_wall_v469 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_kan_5114_spec_declares_post_wall_contract() -> None:
    """REQ-KAN-5114: OpenSpec anchors the post-wall diagnostic before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-5114" in spec
    assert "SCENARIO-KAN-5114" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert "post_wall_progress" in spec


def test_req_kan_5114_refinement_reduces_global_error_without_binary_search() -> None:
    """REQ-KAN-5114: local/global error bounds shrink under bounded refinement."""

    certificate = mod.build_refined_certificate(n_units=20, seed=mod.RANDOM_SEED + 20)

    assert certificate.n_units == 20
    assert certificate.binary_vars == 0
    assert certificate.constraints["milp_constraints"] == 0
    assert certificate.piece_budget["coarse_total_pieces"] == 20
    assert certificate.piece_budget["allocated_total_pieces"] <= 30
    assert certificate.piece_budget["native_total_pieces"] == 60
    assert certificate.refined_unit_count > 0
    assert certificate.initial_global_error_bound > certificate.global_error_bound >= 0.0
    assert certificate.certified_upper_bound >= certificate.exact_upper_bound
    assert certificate.witness_lower_bound == pytest.approx(certificate.exact_upper_bound)


def test_scenario_kan_5114_property_classes_are_sound_at_post_wall_n() -> None:
    """SCENARIO-KAN-5114: safe, false-control, and near-margin rows stay separable."""

    certificate = mod.build_refined_certificate(n_units=20, seed=mod.RANDOM_SEED + 20)
    outcomes = {row.property_class: row for row in mod.evaluate_property_classes(certificate)}

    assert outcomes["true_safe"].property_status == "verified"
    assert outcomes["true_safe"].property_holds is True
    assert outcomes["true_safe"].threshold > certificate.certified_upper_bound

    assert outcomes["false_counterexample"].property_status == "counterexample"
    assert outcomes["false_counterexample"].property_holds is False
    assert outcomes["false_counterexample"].counterexample is not None
    assert outcomes["false_counterexample"].threshold < certificate.exact_upper_bound

    assert outcomes["near_margin_abstain"].property_status == "abstained_margin"
    assert outcomes["near_margin_abstain"].property_holds is None
    assert outcomes["near_margin_abstain"].counterexample is None
    assert outcomes["near_margin_abstain"].threshold > certificate.exact_upper_bound
    assert outcomes["near_margin_abstain"].threshold < certificate.certified_upper_bound


def test_req_kan_5114_artifact_reports_progress_and_required_fields(tmp_path: Path) -> None:
    """REQ-KAN-5114: artifact compares against Exp 5108 and exposes required schema fields."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(artifact_path=artifact_path, run_date="20260701")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload["field_principles"])
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["technique_changed_from_exp5108"] is True
    assert artifact["exp5108_baseline_loaded"] is True
    assert artifact["attempted_n"] == 100
    assert artifact["solved_n"] == 100
    assert artifact["post_wall_progress"] is True
    assert artifact["certificate_soundness"] is True
    assert artifact["false_property_detected"] is True
    assert artifact["near_margin_abstained"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["binary_vars"]["100"] == 0
    assert artifact["constraints"]["100"]["milp_constraints"] == 0
    assert (
        artifact["piece_budget"]["100"]["allocated_total_pieces"]
        < artifact["piece_budget"]["100"]["native_total_pieces"]
    )
    assert len(artifact["seeds_or_checksums"]["reproducibility_checksum"]) == 64
    mod.validate_artifact(artifact)


def test_deliverable_file_validates_for_req_kan_5114() -> None:
    """SCENARIO-KAN-5114: committed deliverable JSON satisfies the post-wall contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["post_wall_progress"] is True
    assert artifact["attempted_n"] >= 100
    assert artifact["solved_n"] > artifact["exp5108_baseline"]["largest_n_reached"]


def test_req_kan_5114_defensive_paths_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-KAN-5114: defensive branches fail closed and the CLI writes the artifact."""

    missing_root = tmp_path / "missing-baseline"
    missing_root.mkdir()
    assert mod._sha256_file(missing_root / "missing.json") is None
    assert mod.load_exp5108_baseline(missing_root)["loaded"] is False

    blocked = mod.build_artifact(root=missing_root, unit_counts=(20,))
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["post_wall_progress"] is False
    mod.validate_artifact(blocked)

    complete = mod.build_artifact(unit_counts=())
    assert complete["honest_verdict"].startswith("complete_")
    assert complete["abstain_rate"] == 0.0
    mod.validate_artifact(complete)

    certificate = mod.build_refined_certificate(n_units=20, seed=mod.RANDOM_SEED + 20)
    good_outcome = mod.evaluate_property_classes(certificate)[0]

    bad_verified = replace(
        good_outcome,
        property_status="verified",
        threshold=certificate.certified_upper_bound - 1.0,
    )
    assert mod._row_sound(mod.DiagnosticRow(20, certificate, (bad_verified,))) is False

    bad_false_control = replace(
        good_outcome,
        property_class="false_counterexample",
        property_status="verified",
        threshold=certificate.certified_upper_bound + 1.0,
    )
    assert mod._row_sound(mod.DiagnosticRow(20, certificate, (bad_false_control,))) is False

    bad_counterexample = replace(
        good_outcome,
        property_class="synthetic",
        property_status="counterexample",
        counterexample=None,
        threshold=certificate.certified_upper_bound + 1.0,
    )
    assert mod._row_sound(mod.DiagnosticRow(20, certificate, (bad_counterexample,))) is False

    output = tmp_path / "cli-result.json"
    assert mod.main(["--date", "20260701", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "success_kan_abstraction_refinement_post_wall_progress" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
