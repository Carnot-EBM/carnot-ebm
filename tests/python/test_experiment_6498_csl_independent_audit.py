"""Tests for Exp6498 independent continuous-learning replay audit.

Spec refs: REQ-CL-6498, SCENARIO-CL-6498-INDEPENDENCE,
SCENARIO-CL-6498-REPLAY, SCENARIO-CL-6498-CLAIM,
SCENARIO-CL-6498-ATTACKS, SCENARIO-CL-6498-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6498_csl_independent_audit as mod


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


def test_req_cl_6498_spec_declares_independent_audit_contract() -> None:
    """REQ-CL-6498: OpenSpec owns the independent audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CL-6498") : text.index("REQ-CSL-6318")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CL-6498-INDEPENDENCE",
        "SCENARIO-CL-6498-REPLAY",
        "SCENARIO-CL-6498-CLAIM",
        "SCENARIO-CL-6498-ATTACKS",
        "SCENARIO-CL-6498-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_cl_6498_real_artifacts_replay_valid_null(tmp_path: Path) -> None:
    """SCENARIO-CL-6498-REPLAY/CLAIM: real rows audit cleanly but claim stays closed."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    aggregate = artifact["aggregate_row_recomputation"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_independent_audit"
    assert artifact["honest_verdict"].startswith("complete_independent_audit")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["csl_audit_ready_score"] == 1.0
    assert artifact["continuous_learning_claim_eligible"] is False

    gates = artifact["upstream_gate_receipts"]
    assert gates["all_structured_gates_passed"] is True
    by_id = {row["artifact_id"]: row for row in gates["rows"]}
    assert by_id["exp6496_execution"]["field"] == "csl_execution_complete_score"
    assert by_id["exp6496_execution"]["observed"] == 1.0
    assert by_id["exp6496_science"]["field"] == "continuous_self_learning_ready_score"
    assert by_id["exp6496_science"]["observed"] == 0.0
    assert by_id["exp6497_support"]["field"] == "support_preserved_score"
    assert by_id["exp6497_support"]["observed"] == 1.0

    reducer = artifact["independent_reducer_receipt"]
    assert reducer["forbidden_imports_clean"] is True
    assert reducer["forbidden_imports"] == list(mod.FORBIDDEN_IMPORTS)
    assert reducer["reducer_identity"] == mod.REDUCER_IDENTITY

    assert aggregate == mod.recompute_aggregates_from_rows(artifact["per_unit_rows"])
    assert aggregate["audit"]["csl_audit_ready_score_from_rows"] == 1.0
    assert aggregate["audit"]["continuous_learning_claim_eligible_from_rows"] is False
    assert aggregate["exp6496"]["continuous_self_learning_ready_score_from_rows"] == 0.0
    assert aggregate["exp6496"]["held_future_benefit"] is False
    assert artifact["gate_check_summary"]["checks"]["audit_rows_valid"] is True
    assert artifact["gate_check_summary"]["claim_failed_gates"] == ["held_future_benefit"]


def test_scenario_cl_6498_replay_rows_cover_chronology_actions_dose_and_support(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6498-REPLAY: all row families are replayed from raw rows."""

    artifact = _artifact(tmp_path)
    aggregate = artifact["aggregate_row_recomputation"]

    assert len(artifact["chronology_replay_rows"]) == 28
    assert {row["source_artifact"] for row in artifact["chronology_replay_rows"]} == {
        "exp6496",
        "exp6497",
    }
    assert all(row["chronology_valid"] for row in artifact["chronology_replay_rows"])
    assert all(row["event_identity_unique"] for row in artifact["chronology_replay_rows"])

    process_ids = {row["process_id"] for row in artifact["evidence_replay_rows"]}
    assert {
        "exp6496_exact_admission_evidence",
        "exp6497_stress_admission_and_exposure",
        "exp6497_restart_spending",
    }.issubset(process_ids)
    assert all(row["sequential_evidence_valid"] for row in artifact["evidence_replay_rows"])
    assert all(row["peek_charged_or_absent"] for row in artifact["evidence_replay_rows"])

    assert aggregate["exp6496"]["durable_write_count"] == 0
    assert aggregate["exp6496"]["dose_rows_matched"] is True
    assert aggregate["exp6497"]["support_preserved_score_from_rows"] == 1.0
    assert all(row["matched"] for row in artifact["action_store_match_rows"])
    assert any(row["row_family"] == "arm_dose" for row in artifact["dose_recomputation_rows"])
    assert any(row["row_family"] == "capacity_dose" for row in artifact["dose_recomputation_rows"])
    assert all(row["metric_matches_source"] for row in artifact["immediate_metric_rows"])
    assert all(row["metric_matches_source"] for row in artifact["future_metric_rows"])
    assert all(row["support_matches_source"] for row in artifact["support_recomputation_rows"])


def test_scenario_cl_6498_attack_matrix_and_discrepancies_are_rowed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6498-ATTACKS: shortcut probes fail closed."""

    artifact = _artifact(tmp_path)
    matrix = artifact["audit_attack_matrix"]
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    for attack_id in mod.ATTACK_IDS:
        assert by_id[attack_id]["fail_closed"] is True
        assert by_id[attack_id]["discrepancy_emitted_if_open"] is True
    for row in artifact["discrepancy_rows"]:
        assert {"json_pointer", "expected", "observed", "severity", "impact"}.issubset(row)
        assert row["severity"] != "critical"


def test_scenario_cl_6498_validation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6498-ARTIFACT: malformed audit artifacts cannot pass."""

    clean = _artifact(tmp_path)
    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._sequential_valid(clean) is True
    assert (
        mod._exp6496_durable_by_arm(
            {"decision_action_rows": [{"arm_id": "synthetic", "durable": True}]}
        )["synthetic"][0]["arm_id"]
        == "synthetic"
    )

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

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    bad_rows = deepcopy(clean)
    bad_rows["per_unit_rows"] = bad_rows["per_unit_rows"][:-1]
    bad_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_rows)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_rows)

    bad_aggregate = deepcopy(clean)
    bad_aggregate["aggregate_row_recomputation"]["audit"][
        "csl_audit_ready_score_from_rows"
    ] = 0.0
    bad_aggregate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_aggregate)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_aggregate)

    bad_claim = deepcopy(clean)
    bad_claim["continuous_learning_claim_eligible"] = True
    bad_claim["reproducibility_checksum"] = mod.reproducibility_checksum(bad_claim)
    assert "continuous_learning_claim_eligible mismatch" in mod.validate_artifact(
        bad_claim
    )

    bad_reducer = deepcopy(clean)
    bad_reducer["independent_reducer_receipt"]["forbidden_imports_clean"] = False
    bad_reducer["reproducibility_checksum"] = mod.reproducibility_checksum(bad_reducer)
    assert "independent_reducer_receipt forbidden imports" in mod.validate_artifact(
        bad_reducer
    )

    bad_top_rows = deepcopy(clean)
    bad_top_rows["chronology_replay_rows"] = bad_top_rows["chronology_replay_rows"][:-1]
    bad_top_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_top_rows)
    assert "chronology_replay_rows mismatch" in mod.validate_artifact(bad_top_rows)

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"][
        "active_roadmap_and_conductor_unchanged"
    ] = False
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad_protected)

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "unknown"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(
        bad_verdict
    )

    bad_status = deepcopy(clean)
    bad_status["status"] = "blocked_independent_audit"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    assert "status mismatch" in mod.validate_artifact(bad_status)

    bad_attack = deepcopy(clean)
    bad_attack["audit_attack_matrix"]["rows"] = bad_attack["audit_attack_matrix"]["rows"][:-1]
    bad_attack["reproducibility_checksum"] = mod.reproducibility_checksum(bad_attack)
    assert "audit_attack_matrix mismatch" in mod.validate_artifact(bad_attack)

    invalid_sequence = deepcopy(clean)
    invalid_sequence["evidence_replay_rows"][0]["sequential_evidence_valid"] = False
    assert mod._sequential_valid(invalid_sequence) is False

    rows = {
        "chronology_replay_rows": deepcopy(clean["chronology_replay_rows"]),
        "evidence_replay_rows": deepcopy(clean["evidence_replay_rows"]),
        "action_store_match_rows": deepcopy(clean["action_store_match_rows"]),
        "dose_recomputation_rows": deepcopy(clean["dose_recomputation_rows"]),
        "immediate_metric_rows": deepcopy(clean["immediate_metric_rows"]),
        "future_metric_rows": deepcopy(clean["future_metric_rows"]),
        "support_recomputation_rows": deepcopy(clean["support_recomputation_rows"]),
    }
    rows["chronology_replay_rows"][0]["chronology_valid"] = False
    upstream = deepcopy(clean["upstream_gate_receipts"])
    upstream["rows"][0]["observed"] = 0.0
    discrepancies = mod._discrepancy_rows(
        rows, clean["aggregate_row_recomputation"], upstream
    )
    assert {row["severity"] for row in discrepancies} == {"critical"}
    assert len(discrepancies) == 2

    assert mod._status_and_verdict(1.0, True, [])[1].endswith("claim is eligible")
    assert mod._status_and_verdict(0.0, False, ["bad_gate"])[0] == (
        "blocked_independent_audit"
    )
    manual_path = tmp_path / "manual.json"
    assert mod.write_artifact({"ok": True}, manual_path) == manual_path
    assert json.loads(manual_path.read_text()) == {"ok": True}


def test_scenario_cl_6498_run_records_requested_date(tmp_path: Path) -> None:
    """SCENARIO-CL-6498-ARTIFACT: CLI runner writes the requested date."""

    artifact = mod.run(date="20260821", result_path=tmp_path / "artifact.json")
    written = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["preconditions_checked"]["requested_date"] == "20260821"
    assert mod.validate_artifact(artifact) == []
