"""Tests for Exp6497 factor-pool support stress.

Spec refs: REQ-CL-6497, SCENARIO-CL-6497-GATE,
SCENARIO-CL-6497-CAPACITY, SCENARIO-CL-6497-LIFECYCLE,
SCENARIO-CL-6497-SUPPORT, SCENARIO-CL-6497-ATTACKS,
SCENARIO-CL-6497-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6497_factor_pool_support_stress as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _artifact(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        **kwargs,
    )


def test_req_cl_6497_spec_declares_support_stress_contract() -> None:
    """REQ-CL-6497: OpenSpec owns the support-stress contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CL-6497") : text.index("REQ-CSL-6318")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CL-6497-GATE",
        "SCENARIO-CL-6497-CAPACITY",
        "SCENARIO-CL-6497-LIFECYCLE",
        "SCENARIO-CL-6497-SUPPORT",
        "SCENARIO-CL-6497-ATTACKS",
        "SCENARIO-CL-6497-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_cl_6497_gate_uses_exp6496_execution_not_science(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6497-GATE/ARTIFACT: execution completeness gates the run."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    gate = artifact["upstream_gate_receipt"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive")
    assert artifact["support_stress_complete_score"] == 1.0
    assert artifact["support_preserved_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    assert gate["path"].endswith("experiment_6496_continuous_factor_learning.json")
    assert gate["field"] == "csl_execution_complete_score"
    assert gate["expected"] == 1.0
    assert gate["observed"] == 1.0
    assert gate["observed_type"] == "float"
    assert gate["passed"] is True
    assert gate["science_field"] == "continuous_self_learning_ready_score"
    assert gate["science_observed"] == 0.0
    assert gate["science_required"] is False


def test_scenario_cl_6497_capacity_cells_are_complete_and_matched(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6497-CAPACITY: each capacity receives each event."""

    artifact = _artifact(tmp_path)
    aggregate = artifact["aggregate_row_recomputation"]
    manifest = artifact["frozen_stress_manifest"]

    assert set(manifest["capacities"]) == set(mod.CAPACITY_IDS)
    assert set(manifest["stress_conditions"]) == set(mod.STRESS_CONDITIONS)
    assert manifest["evaluation_outcomes_inspected_before_freeze"] is False
    assert aggregate["all_capacity_stress_cells_represented"] is True
    assert aggregate["identical_event_opportunities"] is True
    assert aggregate["capacity_respected"] is True

    expected = len(mod.CAPACITY_IDS) * len(artifact["stress_stream_rows"])
    assert len(artifact["capacity_arm_rows"]) == expected
    assert aggregate["capacity_arm_row_count"] == expected

    rows_by_capacity: dict[str, list[dict[str, Any]]] = {
        capacity_id: [
            row
            for row in artifact["capacity_arm_rows"]
            if row["capacity_id"] == capacity_id
        ]
        for capacity_id in mod.CAPACITY_IDS
    }
    expected_events = [row["stress_event_id"] for row in artifact["stress_stream_rows"]]
    for capacity_id, rows in rows_by_capacity.items():
        assert [row["stress_event_id"] for row in rows] == expected_events
        capacity = manifest["capacities"][capacity_id]["active_capacity"]
        assert all(row["occupancy_after"] <= capacity for row in rows)
        assert all(row["admission_opportunity_charged"] is True for row in rows)
    assert all(row["durable"] is False for row in rows_by_capacity["zero_frozen"])


def test_scenario_cl_6497_lifecycle_support_and_recommendation(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6497-LIFECYCLE/SUPPORT: recommendation is row-derived."""

    artifact = _artifact(tmp_path)
    aggregate = artifact["aggregate_row_recomputation"]
    recommendation = artifact["recommended_capacity"]

    assert recommendation["capacity_id"] == "medium_bounded"
    assert recommendation["source"] == "row_derived"
    assert recommendation["support_preserved"] is True
    assert aggregate["recommended_capacity"] == recommendation["capacity_id"]
    assert aggregate["support_preserved_score_from_rows"] == 1.0

    lifecycle_types = {row["lifecycle_type"] for row in artifact["eviction_rollback_restart_rows"]}
    assert {"eviction", "rollback", "restart", "tombstone", "recovery"}.issubset(
        lifecycle_types
    )
    assert all(row["recovery_time_events"] >= 0 for row in artifact["eviction_rollback_restart_rows"])

    regressions = [row for row in artifact["negative_transfer_rows"] if row["regression"]]
    assert regressions
    assert any(row["capacity_id"] == "overlarge_unbounded_probe" for row in regressions)
    assert all(row["baseline_capacity_id"] == "zero_frozen" for row in regressions)

    support_rows = artifact["future_support_rows"]
    assert support_rows
    for row in support_rows:
        assert row["planned_future_unit_count"] == len(mod.FUTURE_UNITS)
        assert row["support_computed_from"] == "all_planned_future_units"
        assert set(row["best_of_k_support"]) == {str(k) for k in mod.SUPPORT_BUDGETS}
    medium_rows = [row for row in support_rows if row["capacity_id"] == "medium_bounded"]
    assert all(row["material_support_loss"] is False for row in medium_rows)


def test_scenario_cl_6497_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6497-ATTACKS: all stress attacks fail closed."""

    artifact = _artifact(tmp_path)
    matrix = artifact["stress_attack_matrix"]
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    for attack_id in mod.ATTACK_IDS:
        assert by_id[attack_id]["fail_closed"] is True
        assert by_id[attack_id]["row_accounted"] is True


def test_scenario_cl_6497_validation_and_status_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6497-ARTIFACT: malformed artifacts cannot claim support."""

    clean = _artifact(tmp_path)
    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert len(mod._exp6496_sources({})) == 4

    missing = deepcopy(clean)
    del missing["status"]
    assert mod.validate_artifact(missing) == ["missing required field: status"]

    for field, message, value in (
        ("field_principles", "field_principles must cover exactly required fields", {}),
        ("field_provenance", "field_provenance must cover exactly required fields", {}),
        ("inference_substrate", "inference_substrate mismatch", "bad"),
        ("verifier_is_oracle", "verifier_is_oracle must be true", False),
    ):
        mutated = deepcopy(clean)
        mutated[field] = value
        mutated["reproducibility_checksum"] = mod.reproducibility_checksum(mutated)
        assert message in mod.validate_artifact(mutated)

    bad_rows = deepcopy(clean)
    bad_rows["per_unit_rows"] = bad_rows["per_unit_rows"][:-1]
    bad_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_rows)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_rows)

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    bad_recommendation = deepcopy(clean)
    bad_recommendation["recommended_capacity"] = {"capacity_id": "small_bounded"}
    bad_recommendation["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_recommendation
    )
    assert "recommended_capacity mismatch" in mod.validate_artifact(bad_recommendation)

    bad_complete = deepcopy(clean)
    bad_complete["support_stress_complete_score"] = 0.0
    bad_complete["reproducibility_checksum"] = mod.reproducibility_checksum(bad_complete)
    assert "support_stress_complete_score mismatch" in mod.validate_artifact(
        bad_complete
    )

    bad_score = deepcopy(clean)
    bad_score["support_preserved_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "support_preserved_score mismatch" in mod.validate_artifact(bad_score)

    bad_top_rows = deepcopy(clean)
    bad_top_rows["capacity_arm_rows"] = bad_top_rows["capacity_arm_rows"][:-1]
    bad_top_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_top_rows)
    assert "capacity_arm_rows mismatch" in mod.validate_artifact(bad_top_rows)

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"][
        "active_roadmap_and_conductor_unchanged"
    ] = False
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad_protected)

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "unknown"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad_verdict)

    no_candidate_rows = deepcopy(clean["per_unit_rows"])
    for row in no_candidate_rows:
        if row.get("row_type") == "future_support" and row.get("capacity_id") in {
            "small_bounded",
            "medium_bounded",
        }:
            row["material_support_loss"] = True
    assert mod._recommend_capacity_from_rows(no_candidate_rows)["capacity_id"] is None

    assert mod._status_and_verdict(1.0, 1.0, {"all_gates_passed": True})[0] == (
        "complete_positive"
    )
    assert mod._status_and_verdict(1.0, 0.0, {"all_gates_passed": True})[0] == (
        "complete_null"
    )
    assert mod._status_and_verdict(0.0, 0.0, {"all_gates_passed": True})[0] == (
        "disqualified"
    )
    assert mod._status_and_verdict(
        0.0,
        0.0,
        {"all_gates_passed": False, "blocked_reason": "blocked_gate"},
    )[0] == "blocked_factor_pool_support_stress"


def test_scenario_cl_6497_run_records_requested_date(tmp_path: Path) -> None:
    """SCENARIO-CL-6497-ARTIFACT: CLI runner writes the requested date."""

    artifact = mod.run(date="20260821", result_path=tmp_path / "artifact.json")
    written = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["preconditions_checked"]["requested_date"] == "20260821"
    assert mod.validate_artifact(artifact) == []
