"""Tests for Exp5471 deterministic guard-composition scale-up.

Spec refs: REQ-SAFE-5471, SCENARIO-SAFE-5471.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5470_rewrite_state_semantic_fixture_v497 as exp5470
from carnot import experiment_5471_guard_composition_scale_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5471_guard_composition_scale_v497.py -q"
)


def _artifact() -> dict:
    return mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])


def _rows_by_id() -> dict[str, dict]:
    return {row["candidate_id"]: row for row in mod.evaluate_candidates(mod.build_candidates())}


def test_req_safe_5471_spec_declares_guard_composition_contract() -> None:
    """REQ-SAFE-5471: OpenSpec anchors the deterministic composition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5471") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5471",
        "SCENARIO-SAFE-5471",
        str(mod.RESULT_RELATIVE_PATH),
        "license-transition guards",
        "semantic-graph guards",
        "deterministic distortion guards",
        "minimal-core feedback",
        "exactly one guard catches",
        "composed guards",
        "repair proposal scores",
        "minimal core IDs separately from semantic graph node IDs",
        "guard success",
        mod.INFERENCE_SUBSTRATE,
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_safe_5471_extends_exp5470_fixture_and_reports_guard_catches() -> None:
    """REQ-SAFE-5471: scaled rows identify single-guard and composed failures."""

    candidates = mod.build_candidates()
    rows = mod.evaluate_candidates(candidates)

    assert len(candidates) == 13
    assert len(candidates) > len(exp5470.build_candidates())
    assert {row["candidate_id"] for row in rows} == {
        candidate.candidate_id for candidate in candidates
    }

    single_guard_rows = [
        row for row in rows if row["expected_accept"] is False and len(row["caught_by_guards"]) == 1
    ]
    composed_rows = [
        row for row in rows if row["expected_accept"] is False and len(row["caught_by_guards"]) >= 2
    ]
    assert {tuple(row["caught_by_guards"]) for row in single_guard_rows} >= {
        ("distortion_guard",),
        ("license_transition_guard",),
        ("semantic_graph_guard",),
    }
    assert any(
        {"license_transition_guard", "semantic_graph_guard"}.issubset(
            set(row["caught_by_guards"])
        )
        for row in composed_rows
    )
    assert any(
        {"distortion_guard", "semantic_graph_guard"}.issubset(set(row["caught_by_guards"]))
        for row in composed_rows
    )

    for row in rows:
        verdict = row["exact_final_verdict"]
        assert verdict["final_authority"] == mod.EXACT_FINAL_AUTHORITY
        assert verdict["computed_from_repair_score"] is False
        assert verdict["matches_expected"] is True
        assert isinstance(row["repair_proposal"]["proposal_score"], float)
        assert all(
            result["evidence_source"] != "repair_proposal_score"
            for result in row["guard_results"].values()
        )
        core_ids = set(row["minimal_core_feedback"]["minimal_core_ids"])
        node_ids = set(row["semantic_graph_receipt"]["node_ids"])
        assert core_ids.isdisjoint(node_ids)


def test_scenario_safe_5471_minimal_core_and_semantic_receipts_are_separate() -> None:
    """SCENARIO-SAFE-5471: core IDs are feedback, graph node IDs are receipts."""

    rows = _rows_by_id()
    hidden = rows["5470-hidden-premise"]
    json_api = rows["5471-json-api-composed"]
    distortion_api = rows["5471-distortion-api-composed"]

    assert hidden["caught_by_guards"] == ["license_transition_guard"]
    assert hidden["minimal_core_feedback"]["minimal_core_ids"] == [
        "core:5470-hidden-premise:license_transition_guard:hidden_premise"
    ]
    assert hidden["minimal_core_feedback"]["unsatisfied_constraint_ids"] == [
        "constraint:license_transition_guard:hidden_premise"
    ]
    assert all(
        node_id.startswith("sem:")
        for node_id in hidden["semantic_graph_receipt"]["node_ids"]
    )

    assert json_api["caught_by_guards"] == [
        "license_transition_guard",
        "semantic_graph_guard",
    ]
    assert set(json_api["minimal_core_feedback"]["guard_to_core_ids"]) == {
        "license_transition_guard",
        "semantic_graph_guard",
    }
    assert set(distortion_api["caught_by_guards"]) == {
        "distortion_guard",
        "semantic_graph_guard",
    }
    assert "distortion_guard" in distortion_api["minimal_core_feedback"][
        "guard_to_core_ids"
    ]


def test_scenario_safe_5471_artifact_metrics_and_write_path(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5471: terminal JSON exposes required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND], write=True)
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixture_count"] == 13
    core_ids = {
        core_id
        for row in artifact["row_results"]
        for core_id in row["minimal_core_feedback"]["minimal_core_ids"]
    }
    assert artifact["minimal_core_count"] == len(core_ids)
    assert artifact["minimal_core_count"] >= 10
    assert artifact["semantic_graph_node_count"] >= artifact["fixture_count"]
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_reject_rate"] == pytest.approx(0.0)
    assert artifact["exact_final_agreement"] == pytest.approx(1.0)
    assert artifact["guard_composition_ready"] is True
    assert artifact["guided_decoding_quarantine_lifted"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["row_results"]
    )
    assert artifact["research_conductor_modified"] is False

    matrix = artifact["guard_overlap_matrix"]
    for guard_id in mod.GUARD_IDS:
        assert matrix[guard_id][guard_id]["count"] == artifact["guard_catch_counts"][
            guard_id
        ]
        assert matrix[guard_id][guard_id]["rate"] == pytest.approx(
            artifact["guard_catch_rates"][guard_id]
        )
    assert matrix["license_transition_guard"]["semantic_graph_guard"]["count"] >= 1
    assert matrix["distortion_guard"]["semantic_graph_guard"]["count"] >= 1
    assert mod.run(result_path=result_path, write=False)["guard_composition_ready"] is True


def test_req_safe_5471_guard_success_ignores_repair_proposal_scalar() -> None:
    """REQ-SAFE-5471: guard success is not the repair proposal score in disguise."""

    rows = mod.evaluate_candidates(mod.build_candidates())
    baseline = mod.derive_metrics(rows)
    perturbed = deepcopy(rows)
    for index, row in enumerate(perturbed):
        row["repair_proposal"]["proposal_score"] = float(index % 2)
        row["repair_proposal"]["selected_repair_action"] = "score_perturbed"

    perturbed_metrics = mod.derive_metrics(perturbed)
    assert perturbed_metrics["guard_catch_rates"] == baseline["guard_catch_rates"]
    assert perturbed_metrics["guard_overlap_matrix"] == baseline["guard_overlap_matrix"]
    assert perturbed_metrics["false_accept_rate"] == baseline["false_accept_rate"]
    assert perturbed_metrics["false_reject_rate"] == baseline["false_reject_rate"]
    assert perturbed_metrics["exact_final_agreement"] == baseline["exact_final_agreement"]

    artifact = _artifact()
    bad = deepcopy(artifact)
    row = next(row for row in bad["row_results"] if row["caught_by_guards"])
    guard_id = row["caught_by_guards"][0]
    row["guard_results"][guard_id]["evidence_source"] = "repair_proposal_score"
    row["row_checksum"] = mod.row_checksum(row)
    bad["row_provenance_checksum"] = mod.row_provenance_checksum(bad["row_results"])
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)

    with pytest.raises(ValueError, match="repair proposal score"):
        mod.validate_artifact(bad)


def test_req_safe_5471_validation_fails_closed_on_schema_or_authority_drift() -> None:
    """REQ-SAFE-5471: schema, exact authority, and readiness drift are rejected."""

    artifact = _artifact()

    missing = deepcopy(artifact)
    missing.pop("fixture_count")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_core_node = deepcopy(artifact)
    row = next(row for row in bad_core_node["row_results"] if row["minimal_core_feedback"]["minimal_core_ids"])
    node_id = row["semantic_graph_receipt"]["node_ids"][0]
    row["minimal_core_feedback"]["minimal_core_ids"][0] = node_id
    first_guard = row["caught_by_guards"][0]
    row["minimal_core_feedback"]["guard_to_core_ids"][first_guard][0] = node_id
    row["row_checksum"] = mod.row_checksum(row)
    bad_core_node["row_provenance_checksum"] = mod.row_provenance_checksum(
        bad_core_node["row_results"]
    )
    bad_core_node["minimal_core_count"] = mod.derive_metrics(
        bad_core_node["row_results"]
    )["minimal_core_count"]
    bad_core_node["reproducibility_checksum"] = mod.reproducibility_checksum(bad_core_node)
    with pytest.raises(ValueError, match="minimal core IDs must be separate"):
        mod.validate_artifact(bad_core_node)

    bad_false_accept = deepcopy(artifact)
    invalid = next(row for row in bad_false_accept["row_results"] if row["expected_accept"] is False)
    invalid["exact_final_verdict"]["accepted"] = True
    invalid["exact_final_verdict"]["violation_kinds"] = []
    invalid["row_checksum"] = mod.row_checksum(invalid)
    metrics = mod.derive_metrics(bad_false_accept["row_results"])
    bad_false_accept["false_accept_rate"] = metrics["false_accept_rate"]
    bad_false_accept["exact_final_agreement"] = metrics["exact_final_agreement"]
    bad_false_accept["row_provenance_checksum"] = mod.row_provenance_checksum(
        bad_false_accept["row_results"]
    )
    bad_false_accept["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_false_accept
    )
    with pytest.raises(ValueError, match="false_accept_rate=0.0"):
        mod.validate_artifact(bad_false_accept)

    bad_quarantine = deepcopy(artifact)
    bad_quarantine["guided_decoding_quarantine_lifted"] = True
    with pytest.raises(ValueError, match="quarantine"):
        mod.validate_artifact(bad_quarantine)

    bad_ready_type = deepcopy(artifact)
    bad_ready_type["guard_composition_ready"] = "yes"
    with pytest.raises(ValueError, match="guard_composition_ready must be boolean"):
        mod.validate_artifact(bad_ready_type)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)


def test_req_safe_5471_defensive_validation_branches() -> None:
    """REQ-SAFE-5471: defensive schema branches expose deterministic errors."""

    artifact = _artifact()

    bad_rows_type = deepcopy(artifact)
    bad_rows_type["row_results"] = "bad"
    assert "row_results must be a list" in "; ".join(
        mod.artifact_schema_errors(bad_rows_type)
    )

    bad_counts = deepcopy(artifact)
    bad_counts["guard_catch_counts"] = {}
    assert "guard_catch_counts must match row recomputation" in "; ".join(
        mod.artifact_schema_errors(bad_counts)
    )

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    assert "field_principles mismatch" in "; ".join(
        mod.artifact_schema_errors(bad_principles)
    )

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm"
    assert "inference_substrate mismatch" in "; ".join(
        mod.artifact_schema_errors(bad_substrate)
    )

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    assert "random_seed mismatch" in "; ".join(mod.artifact_schema_errors(bad_seed))

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    assert "honest_verdict must start" in "; ".join(
        mod.artifact_schema_errors(bad_verdict)
    )

    bad_provenance = deepcopy(artifact)
    bad_provenance["row_provenance_checksum"] = "0" * 64
    assert "row_provenance_checksum mismatch" in "; ".join(
        mod.artifact_schema_errors(bad_provenance)
    )

    bad_scaled = deepcopy(artifact)
    bad_scaled["row_results"] = bad_scaled["row_results"][: len(exp5470.build_candidates())]
    assert "guard_composition_ready requires scaled fixture_count" in "; ".join(
        mod.artifact_schema_errors(bad_scaled)
    )

    bad_guard_catches = deepcopy(artifact)
    for row in bad_guard_catches["row_results"]:
        row["guard_results"]["license_transition_guard"]["caught"] = False
        row["guard_results"]["license_transition_guard"]["violation_kinds"] = []
        if "license_transition_guard" in row["caught_by_guards"]:
            row["caught_by_guards"].remove("license_transition_guard")
    assert "guard_composition_ready requires license_transition_guard catches" in "; ".join(
        mod.artifact_schema_errors(bad_guard_catches)
    )

    bad_single_composed = deepcopy(artifact)
    for row in bad_single_composed["row_results"]:
        row["caught_by_guards"] = []
    single_composed_errors = "; ".join(mod.artifact_schema_errors(bad_single_composed))
    assert "guard_composition_ready requires single-guard failures" in single_composed_errors
    assert "guard_composition_ready requires composed-guard failures" in single_composed_errors

    bad_false_reject = deepcopy(artifact)
    valid = next(row for row in bad_false_reject["row_results"] if row["expected_accept"])
    valid["exact_final_verdict"]["accepted"] = False
    assert "guard_composition_ready requires false_reject_rate=0.0" in "; ".join(
        mod.artifact_schema_errors(bad_false_reject)
    )

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["description"] = "drifted"
    checksum_errors = "; ".join(mod.artifact_schema_errors(bad_checksum))
    assert "guard_composition_ready requires valid row checksums" in checksum_errors
    assert "row checksum mismatch" in checksum_errors

    bad_authority = deepcopy(artifact)
    bad_authority["row_results"][0]["exact_final_verdict"]["final_authority"] = "model"
    authority_errors = "; ".join(mod.artifact_schema_errors(bad_authority))
    assert "guard_composition_ready requires exact final authority" in authority_errors
    assert "final authority must be exact guard validators" in authority_errors

    bad_score_final = deepcopy(artifact)
    bad_score_final["row_results"][0]["exact_final_verdict"][
        "computed_from_repair_score"
    ] = True
    assert "final verdict must not use repair proposal score" in "; ".join(
        mod.artifact_schema_errors(bad_score_final)
    )

    bad_guard_results = deepcopy(artifact)
    bad_guard_results["row_results"][0]["guard_results"] = "bad"
    assert "guard_results must be a mapping" in "; ".join(
        mod.artifact_schema_errors(bad_guard_results)
    )

    bad_caught = deepcopy(artifact)
    bad_caught["row_results"][0]["caught_by_guards"] = ["semantic_graph_guard"]
    assert "caught_by_guards must match exact guard results" in "; ".join(
        mod.artifact_schema_errors(bad_caught)
    )

    bad_missing_caught = deepcopy(artifact)
    bad_missing_caught["row_results"][0]["guard_results"]["distortion_guard"].pop(
        "caught"
    )
    assert "distortion_guard missing caught field" in "; ".join(
        mod.artifact_schema_errors(bad_missing_caught)
    )


def test_deliverable_file_validates_for_scenario_safe_5471() -> None:
    """SCENARIO-SAFE-5471: checked-in deliverable satisfies the V497 contract."""

    path = REPO / mod.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["guard_composition_ready"] is True
    assert artifact["guided_decoding_quarantine_lifted"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
