"""Tests for Exp6496 chronological continuous factor learning.

Spec refs: REQ-CL-6496, SCENARIO-CL-6496-CHRONOLOGY,
SCENARIO-CL-6496-ADMISSION, SCENARIO-CL-6496-DOSE,
SCENARIO-CL-6496-FUTURE-SUPPORT, SCENARIO-CL-6496-LIFECYCLE,
SCENARIO-CL-6496-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6496_continuous_factor_learning as mod


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


def test_req_cl_6496_spec_declares_chronological_learning_contract() -> None:
    """REQ-CL-6496: OpenSpec owns the chronological learning contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CL-6496") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CL-6496-CHRONOLOGY",
        "SCENARIO-CL-6496-ADMISSION",
        "SCENARIO-CL-6496-DOSE",
        "SCENARIO-CL-6496-FUTURE-SUPPORT",
        "SCENARIO-CL-6496-LIFECYCLE",
        "SCENARIO-CL-6496-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_cl_6496_real_stream_completes_null_with_exact_no_writes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6496-CHRONOLOGY/ADMISSION/DOSE: real rows complete."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    aggregate = artifact["aggregate_row_recomputation"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["csl_execution_complete_score"] == 1.0
    assert artifact["continuous_self_learning_ready_score"] == 0.0

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"].endswith("experiment_6495_restarted_factor_pool_controller.json")
    assert gate["field"] == "factor_pool_controller_ready_score"
    assert gate["expected"] == 1.0
    assert gate["observed"] == 1.0
    assert gate["observed_type"] == "float"
    assert gate["passed"] is True

    proposal = artifact["proposal_stream_receipt"]
    assert proposal["proposal_count"] == 4
    assert proposal["exact_compile_count"] == 4
    assert proposal["new_llm_invocation_count"] == 0
    assert artifact["optional_causal_replay_receipt"]["present"] is True
    assert artifact["optional_causal_replay_receipt"]["allowed_use"] == "optional_context_only"

    opportunity_keys = {
        (row["arm_id"], row["chronology_index"], row["proposal_row_hash"])
        for row in artifact["event_rows"]
    }
    assert len(opportunity_keys) == len(mod.ARM_IDS) * proposal["proposal_count"]
    assert aggregate["every_event_opportunity_has_every_arm"] is True
    assert aggregate["durable_write_count"] == 0
    assert aggregate["unsafe_commit_count"] == 0
    assert aggregate["csl_execution_complete_score_from_rows"] == 1.0
    assert aggregate["continuous_self_learning_ready_score_from_rows"] == 0.0
    assert aggregate == mod.recompute_aggregates_from_rows(artifact["per_unit_rows"])

    for row in artifact["exact_admission_rows"]:
        assert row["exact_admission_passed"] is False
        assert row["durable_write_allowed"] is False
    for row in artifact["decision_action_rows"]:
        assert row["durable"] is False
        assert row["action_type"] == "no_write"
    for row in artifact["pool_state_rows"]:
        assert row["active_factor_count"] == 0
        assert row["active_factor_ids"] == []

    for row in artifact["dose_matching_rows"]:
        assert row["opportunity_count"] == proposal["proposal_count"]
        assert row["admitted_event_count"] == 0
        assert row["exposure_dose"] == 0
        assert row["matched_to_restarted"] is True
        assert row["reweighting_applied"] is False


def test_scenario_cl_6496_future_support_and_lifecycle_gates_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6496-FUTURE-SUPPORT/LIFECYCLE: readiness is conjunctive."""

    artifact = _artifact(tmp_path)
    attacks = artifact["lifecycle_attack_matrix"]
    by_id = {row["attack_id"]: row for row in attacks["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["unsafe_survivor_count"] == 0
    assert artifact["gate_check_summary"]["checks"]["lifecycle_attacks_closed"] is True
    assert artifact["gate_check_summary"]["checks"]["held_future_benefit"] is False

    future_by_arm = {row["arm_id"]: row for row in artifact["future_evaluation_rows"]}
    support_by_arm = {row["arm_id"]: row for row in artifact["future_support_rows"]}
    frozen = future_by_arm["frozen_no_update"]
    restarted = future_by_arm["restarted_reuse_spawn_defer"]
    assert restarted["held_future_utility"] == frozen["held_future_utility"]
    assert restarted["safety_regression_count"] == 0
    assert support_by_arm["restarted_reuse_spawn_defer"]["support_loss"] == 0


def test_scenario_cl_6496_optional_causal_replay_absence_is_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6496-ARTIFACT: missing Exp6492 is not hidden."""

    artifact = _artifact(tmp_path, exp6492_path=tmp_path / "missing_6492.json")
    receipt = artifact["optional_causal_replay_receipt"]

    assert receipt["present"] is False
    assert receipt["hash"] is None
    assert receipt["allowed_use"] == "absent_not_required"
    assert "optional_causal_replay_present" in artifact["preconditions_checked"]
    assert artifact["preconditions_checked"]["optional_causal_replay_present"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6496_artifact_validation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6496-ARTIFACT: malformed artifacts cannot claim readiness."""

    clean = _artifact(tmp_path)
    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

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

    bad_score = deepcopy(clean)
    bad_score["continuous_self_learning_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "continuous_self_learning_ready_score mismatch" in mod.validate_artifact(bad_score)

    bad_execution = deepcopy(clean)
    bad_execution["csl_execution_complete_score"] = 0.0
    bad_execution["reproducibility_checksum"] = mod.reproducibility_checksum(bad_execution)
    assert "csl_execution_complete_score mismatch" in mod.validate_artifact(bad_execution)

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"][
        "active_roadmap_and_conductor_unchanged"
    ] = False
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad_protected)

    bad_top_rows = deepcopy(clean)
    bad_top_rows["event_rows"] = bad_top_rows["event_rows"][:-1]
    bad_top_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_top_rows)
    assert "event_rows mismatch" in mod.validate_artifact(bad_top_rows)

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "unknown"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad_verdict)

    assert mod._status_and_verdict(1.0, 1.0, {"all_gates_passed": True})[0] == (
        "complete_positive"
    )
    assert mod._status_and_verdict(0.0, 0.0, {"all_gates_passed": True})[0] == (
        "disqualified"
    )
    assert mod._status_and_verdict(
        0.0,
        0.0,
        {"all_gates_passed": False, "blocked_reason": "blocked_fixture"},
    )[0] == "blocked_chronological_factor_learning"


def test_scenario_cl_6496_accepted_synthetic_rows_cover_durable_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6496-ADMISSION/ARTIFACT: accepted rows stay exact-gated."""

    accepted = [
        {
            "chronology_index": 0,
            "event_id": "synthetic-0",
            "source_unit_id": "unit-0",
            "source_family_id": "boolean_guard",
            "split": "development",
            "model_family": "qwen",
            "model_hf_id": "model-qwen",
            "request_id": "request-0",
            "proposal_row_hash": "sha256:" + "a" * 64,
            "response_sha256": "sha256:" + "b" * 64,
            "compile_outcome": "accept",
            "compile_reason": "synthetic_accept",
            "compile_row_hash": "sha256:" + "c" * 64,
            "factor_id": "factor_alpha",
            "semantic_hash": "sha256:" + "d" * 64,
            "exact_compile_oracle": True,
            "proposal_present": True,
            "future_held_utility_delta": 1.0,
            "immediate_exact_utility_delta": 0.5,
            "support_delta": 0,
            "safety_regression": False,
        },
        {
            "chronology_index": 1,
            "event_id": "synthetic-1",
            "source_unit_id": "unit-1",
            "source_family_id": "boolean_guard",
            "split": "development",
            "model_family": "qwen",
            "model_hf_id": "model-qwen",
            "request_id": "request-1",
            "proposal_row_hash": "sha256:" + "e" * 64,
            "response_sha256": "sha256:" + "f" * 64,
            "compile_outcome": "accept",
            "compile_reason": "synthetic_accept",
            "compile_row_hash": "sha256:" + "1" * 64,
            "factor_id": "factor_alpha",
            "semantic_hash": "sha256:" + "2" * 64,
            "exact_compile_oracle": True,
            "proposal_present": True,
            "future_held_utility_delta": 1.0,
            "immediate_exact_utility_delta": 0.5,
            "support_delta": -1,
            "safety_regression": False,
        },
    ]
    rows = mod.build_learning_rows(accepted)
    dose = mod._dose_matching_rows(accepted, rows["decision_action_rows"])
    eval_rows = mod._evaluation_rows(accepted, rows["decision_action_rows"])
    cells = mod._family_model_horizon_cells(accepted, rows["decision_action_rows"])
    lifecycle = mod._lifecycle_attack_matrix()
    per_unit = mod._per_unit_rows(
        rows={
            **rows,
            "dose_matching_rows": dose,
            **eval_rows,
            "family_model_horizon_cells": cells,
        },
        lifecycle_attack_matrix=lifecycle,
    )
    aggregate = mod.recompute_aggregates_from_rows(per_unit)

    assert aggregate["durable_write_count"] > 0
    assert any(row["action_type"] == "spawn_write" for row in rows["decision_action_rows"])
    assert any(row["action_type"] == "reuse_write" for row in rows["decision_action_rows"])
    assert any(row["admitted_event_count"] > 0 for row in cells)
    assert mod._admission_reason(
        {"compile_outcome": "accept", "exact_compile_oracle": False}
    ) == "accepted_without_exact_oracle_receipt"
    assert mod._decision_for_arm("fixed_threshold", accepted[0], 0.5, [])[3] == (
        "threshold_not_met"
    )

    written = tmp_path / "manual.json"
    assert mod.write_artifact({"ok": True}, written) == written
    assert json.loads(written.read_text()) == {"ok": True}

    run_path = tmp_path / "run.json"
    run_artifact = mod.run(date="20260821", result_path=run_path)
    assert run_path.is_file()
    assert run_artifact["preconditions_checked"]["requested_date"] == "20260821"
    assert mod.validate_artifact(run_artifact) == []
