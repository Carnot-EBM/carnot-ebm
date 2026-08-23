"""Tests for Exp6563 measured production Safety-Net workload canary.

Spec refs: REQ-PIPELINE-6563, SCENARIO-PIPELINE-6563-IDENTITY,
SCENARIO-PIPELINE-6563-MEASURED-WORK,
SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK,
SCENARIO-PIPELINE-6563-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6563_production_safety_net_workload_canary as mod


REPO = Path(__file__).resolve().parents[2]
TESTS_RUN = [{"command": "focused-exp6563", "exit_code": 0}]


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """REQ-PIPELINE-6563: build the canary from checked-in fixtures."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=Path("/tmp/experiment_6563_test_result.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_pipeline_6563_spec_declares_measured_canary_contract() -> None:
    """REQ-PIPELINE-6563: OpenSpec owns the measured canary contract."""

    text = (REPO / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-PIPELINE-6563") :]

    for marker in (
        "SCENARIO-PIPELINE-6563-IDENTITY",
        "SCENARIO-PIPELINE-6563-MEASURED-WORK",
        "SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK",
        "SCENARIO-PIPELINE-6563-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_pipeline_6563_artifact_schema_and_scores(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-PIPELINE-6563-ATOMIC: terminal fields and scores recompute."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_production_safety_net_workload_canary_null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["production_workload_canary_ready_score"] == 1.0
    assert artifact["production_workload_promotion_candidate_score"] == 0.0
    assert artifact["upstream_gate_receipt"]["gate_passed"] is True
    assert artifact["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(artifact)
    assert artifact["aggregate_row_recomputation"]["headline_excludes_synthetic_cost_units"]
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_pipeline_6563_workload_matrix_is_frozen_and_family_blind(
    artifact: dict[str, Any],
) -> None:
    """REQ-PIPELINE-6563: workloads, timing, seeds, and fixture hashes are fixed."""

    contract = artifact["frozen_workload_and_timing_contract"]
    workload_rows = contract["workload_matrix_rows"]

    assert contract["family_blind"] is True
    assert contract["uses_checked_in_fixtures_only"] is True
    assert contract["warm_up_iterations"] == 1
    assert contract["conditions"] == list(mod.CONDITIONS)
    assert contract["random_seed"] == mod.RANDOM_SEED
    assert set(contract["required_strata"]) == set(mod.REQUIRED_STRATA)
    assert {row["stratum"] for row in workload_rows} >= set(mod.REQUIRED_STRATA)
    assert all(row["fixture_sha256"].startswith("sha256:") for row in workload_rows)
    assert all(row["request_forbidden_policy_features_present"] == [] for row in workload_rows)
    assert all(row["candidate_order_frozen"] for row in workload_rows)
    assert contract["timer_contract"]["monotonic_resolution_s"] > 0.0


def test_scenario_pipeline_6563_per_unit_rows_cover_conditions_and_direct_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-PIPELINE-6563-MEASURED-WORK: rows expose direct work metrics."""

    rows = artifact["per_unit_rows"]
    workload_ids = {
        row["workload_id"]
        for row in artifact["frozen_workload_and_timing_contract"]["workload_matrix_rows"]
    }
    expected_pairs = {
        (workload_id, condition) for workload_id in workload_ids for condition in mod.CONDITIONS
    }

    assert {(row["workload_id"], row["condition"]) for row in rows} == expected_pairs
    assert {row["condition"] for row in rows} == set(mod.CONDITIONS)
    assert any(row["route"] == "compact_router" for row in rows)
    assert any(row["fallback_reason"] == "abstention" for row in rows)
    assert any(row["fallback_reason"] == "forced_fallback" for row in rows)
    assert any(row["fallback_reason"] == "exception_table_hit" for row in rows)
    assert any(str(row["fallback_reason"]).startswith("malformed_input") for row in rows)
    assert any(row["fallback_reason"] == "stale_configuration" for row in rows)

    for row in rows:
        assert set(mod.PER_UNIT_REQUIRED_FIELDS) <= set(row)
        assert row["request_byte_count"] > 0
        assert row["serialization_bytes"] >= row["request_byte_count"]
        assert row["persistence_bytes"] >= 0
        assert row["process_time_s"] >= 0.0
        assert row["monotonic_wall_time_s"] >= 0.0
        assert row["exact_result_sha256"].startswith("sha256:")
        assert row["candidate_preserved"] is True
        assert row["exact_output_equal_to_native"] is True
        assert row["hidden_retry_count"] == 0
        assert row["headline_work_source"] == "direct_measured_receipts"
        assert row["synthetic_adapter_cost_units_diagnostic"] >= 0.0

    measured_rows = artifact["measured_work_and_latency_rows"]
    assert len(measured_rows) == len(rows)
    assert all(row["synthetic_cost_excluded_from_headline"] for row in measured_rows)


def test_scenario_pipeline_6563_disabled_identity_and_enabled_fallback_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-PIPELINE-6563-IDENTITY: disabled rows and enabled routes are visible."""

    identity_rows = artifact["disabled_identity_rows"]
    enabled_rows = artifact["enabled_route_and_fallback_rows"]
    exact = artifact["exact_output_and_candidate_receipt"]

    assert len(identity_rows) == len(
        artifact["frozen_workload_and_timing_contract"]["workload_matrix_rows"]
    )
    assert all(row["serialized_request_bytes_equal"] for row in identity_rows)
    assert all(row["candidate_order_equal"] for row in identity_rows)
    assert all(row["checker_calls_equal"] for row in identity_rows)
    assert all(row["outputs_equal"] for row in identity_rows)
    assert all(row["error_types_equal"] for row in identity_rows)
    assert all(row["side_effects_equal"] for row in identity_rows)
    assert all(row["persistence_equal"] for row in identity_rows)

    assert enabled_rows
    assert all(row["candidate_preserved"] for row in enabled_rows)
    assert all(row["exact_output_equal_to_native"] for row in enabled_rows)
    assert exact["all_exact_outputs_equal"] is True
    assert exact["all_candidates_preserved"] is True
    assert exact["candidate_deletion_count"] == 0


def test_scenario_pipeline_6563_restart_and_rollback_recover_exactly(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK: recovery paths are exact."""

    receipts = artifact["restart_and_rollback_receipts"]

    assert receipts["fallback_reachable"] is True
    assert receipts["restart_replayed"] is True
    assert receipts["restart_exact_output_equal"] is True
    assert receipts["rollback_exercised"] is True
    assert receipts["rollback_restores_disabled"] is True
    assert receipts["rollback_exact_output_equal"] is True
    assert receipts["ledger_persistence_visible"] is True
    assert any(row["condition"] == "rollback" for row in receipts["rollback_rows"])


def test_scenario_pipeline_6563_blocked_inputs_close_with_diagnostics(
    tmp_path: Path,
) -> None:
    """SCENARIO-PIPELINE-6563-ATOMIC: missing gate or fixtures block honestly."""

    blocked = mod.build_artifact(
        repo_root=tmp_path,
        result_path=Path("blocked.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert blocked["status"] == "blocked_production_safety_net_workload_canary"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["production_workload_canary_ready_score"] == 0.0
    assert blocked["production_workload_promotion_candidate_score"] == 0.0
    assert blocked["per_unit_rows"] == []
    assert blocked["upstream_gate_receipt"]["gate_passed"] is False
    assert blocked["gate_check_summary"]["all_gates_passed"] is False
    assert blocked["preconditions_checked"]["fixture_hashes"]
    assert mod.validate_artifact(blocked) == []


def test_req_pipeline_6563_helper_edges_and_verdict_classes(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-PIPELINE-6563: helper fallbacks and closed verdict classes are covered."""

    assert mod.sha256_file(None) == "missing"
    assert mod._field_value({"wrapped": {"value": 1.0}}, "wrapped") == 1.0  # noqa: SLF001
    assert mod._fixture_source_rows(tmp_path) == []  # noqa: SLF001
    assert mod.freeze_workload_cases(tmp_path) == []
    assert mod.per_unit_rows([]) == []
    assert all(
        item.startswith("sha256:")
        for item in mod._candidate_ids_from_source({}, 2)  # noqa: SLF001
    )

    fixture = tmp_path / mod.FIXTURE_RELATIVE_PATHS[0]
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "{bad json}\n"
        + json.dumps({"answer_space": [{"candidate_hash": "sha256:" + "1" * 64}]})
        + "\n",
        encoding="utf-8",
    )
    assert len(mod._fixture_source_rows(tmp_path, limit=1)) == 1  # noqa: SLF001

    positive = deepcopy(artifact)
    for row in positive["per_unit_rows"]:
        if row["condition"] == "enabled_adapter":
            row["checker_calls"] = 0
            row["monotonic_wall_time_s"] = 0.0
    positive["measured_work_and_latency_rows"] = mod.measured_work_and_latency_rows(
        positive["per_unit_rows"]
    )
    aggregate = mod.aggregate_row_recomputation(positive)
    assert aggregate["verdict_class_from_rows"] == "positive"
    assert mod._status_and_verdict(aggregate)[2] == "positive"  # noqa: SLF001
    assert mod._status_and_verdict({"verdict_class_from_rows": "partial"})[2] == "partial"  # noqa: SLF001
    assert mod._status_and_verdict({"verdict_class_from_rows": "disqualified"})[2] == "disqualified"  # noqa: SLF001

    positive.update(
        {
            "status": "complete_production_safety_net_workload_canary_positive",
            "honest_verdict": (
                "complete_production_safety_net_workload_canary_positive: "
                "disabled identity, exact equality, fallback, restart, rollback, "
                "and measured enabled-path benefit passed"
            ),
            "verdict_class": "positive",
            "production_workload_canary_ready_score": 1.0,
            "production_workload_promotion_candidate_score": 0.0,
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": mod.gate_check_summary(aggregate),
        }
    )
    positive["reproducibility_checksum"] = mod.reproducibility_checksum(positive)
    assert "positive verdict requires promotion score 1.0" in mod.validate_artifact(positive)


def test_req_pipeline_6563_validation_detects_tampering(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-PIPELINE-6563-ATOMIC: validation rejects unsafe positive claims."""

    mutations = [
        (
            lambda data: data.pop("status"),
            "required field set mismatch",
        ),
        (
            lambda data: data.__setitem__("inference_substrate", "wrong"),
            "inference_substrate mismatch",
        ),
        (
            lambda data: data.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must be false",
        ),
        (
            lambda data: data.__setitem__("honest_verdict", "not-terminal"),
            "honest_verdict terminal prefix mismatch",
        ),
        (
            lambda data: data.__setitem__("verdict_class", "surprise"),
            "verdict_class outside Exp6563 enum",
        ),
        (
            lambda data: data.__setitem__("production_workload_canary_ready_score", 0.5),
            "production_workload_canary_ready_score must be 0.0 or 1.0",
        ),
        (
            lambda data: data.__setitem__("production_workload_promotion_candidate_score", 0.5),
            "production_workload_promotion_candidate_score must be 0.0 or 1.0",
        ),
        (
            lambda data: data.__setitem__("field_provenance", {}),
            "field_provenance must cover required fields",
        ),
        (
            lambda data: data["per_unit_rows"][0].__setitem__("candidate_preserved", False),
            "aggregate recomputation mismatch",
        ),
        (
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "disabled_identity_exact", False
            ),
            "aggregate recomputation mismatch",
        ),
        (
            lambda data: data["exact_output_and_candidate_receipt"].__setitem__(
                "all_exact_outputs_equal", False
            ),
            "exact output equality failed",
        ),
        (
            lambda data: data["shortcut_attack_matrix"].__setitem__(
                "all_attacks_fail_closed", False
            ),
            "shortcut attack false accept",
        ),
        (
            lambda data: data["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
            "protected files changed",
        ),
        (
            lambda data: data.__setitem__("production_workload_promotion_candidate_score", 1.0),
            "promotion score mismatch",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        _with_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)


def test_req_pipeline_6563_cli_write_validate_and_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PIPELINE-6563-ATOMIC: CLI writes atomically and validates output."""

    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad-json.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001

    result_path = tmp_path / "cli-exp6563.json"
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["status"] == "complete_production_safety_net_workload_canary_null"
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1

    original_build = mod.build_artifact
    try:
        monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": "artifact"})
        assert (
            mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad-build.json")]) == 1
        )
    finally:
        monkeypatch.setattr(mod, "build_artifact", original_build)
