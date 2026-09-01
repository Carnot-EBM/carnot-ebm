"""Tests for the frozen risk-sensitive memory opportunity fixture.

Spec refs: REQ-CL-6853 and SCENARIO-CL-6853-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6853_risk_sensitive_memory_opportunity_fixture as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def current_sources() -> dict[str, dict]:
    """Load the frozen inputs once because the chronological shards are large."""

    return exp.load_sources(exp.source_paths_for_root(REPO))


@pytest.fixture(scope="session")
def current_artifact(current_sources: dict[str, dict]) -> dict:
    """Build one in-memory artifact for contract checks without writing tracked state."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        sources=current_sources,
    )


def _minimal_row(decision_id: str = "decision-1") -> dict:
    """Return a valid small row so each failure test changes one condition."""

    return {
        "decision_id": decision_id,
        "decision_sequence_index": 1,
        "order_id": "order_1",
        "chronological_position": 2,
        "family": "family-a",
        "stratum": "family-a|ordinary|nonzero",
        "decision_context": {field: 0 for field in exp.DECISION_CONTEXT_FIELDS},
        "available_actions": list(exp.FIRST_CLASS_ACTIONS),
        "action_availability": {action: True for action in exp.FIRST_CLASS_ACTIONS},
        "observed_action": "verified_memory",
        "observed_potential_outcome_support": {
            "verified_memory": {"observed": True},
            "no_memory": {"observed": False},
            "abstain": {"observed": False},
        },
        "baseline_actions": {
            "random_admission": "no_memory",
            "always_memory": "verified_memory",
        },
        "exact_later_outcome": {
            "outcome_identity": "sha256:" + "1" * 64,
            "exact_outcome_hash": "sha256:" + "2" * 64,
            "signed_direction": 1,
            "revealed_after_decision": True,
        },
        "memory_effect_class": "helpful",
        "safe_selection_headroom": 1,
        "delayed_correction": False,
        "row_sha256": "sha256:" + "3" * 64,
    }


def test_req_cl_6853_builds_complete_frozen_fixture(current_artifact: dict) -> None:
    """REQ-CL-6853: the ready artifact has every required field and exact join."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(current_artifact)
    assert current_artifact["risk_sensitive_stream_ready_score"] == 1
    assert current_artifact["memory_headroom_nonzero_score"] == 1
    assert current_artifact["verifier_is_oracle"] is False
    assert current_artifact["verdict_class"] == "null"
    assert current_artifact["honest_verdict"].startswith("complete_")
    assert current_artifact["rows"]
    assert len(current_artifact["rows"]) == len(current_artifact["decision_headroom_rows"])
    assert all(row["exact_later_outcome"]["revealed_after_decision"] for row in current_artifact["rows"])
    assert all(row["observed_action"] == "verified_memory" for row in current_artifact["rows"])


def test_scenario_cl_6853_leakage_is_detected_and_current_rows_are_clean(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6853-LEAKAGE: outcome and learner fields stay post-boundary."""

    clean = current_artifact["leakage_attack_results"]
    assert clean["passed"] is True
    assert clean["clean_fixture_leakage_paths"] == []
    assert all(attack["detected"] for attack in clean["injection_attacks"])

    context = deepcopy(current_artifact["rows"][0]["decision_context"])
    context["nested"] = {"exact_outcome_hash": "leak"}
    assert exp.find_leakage_paths(context) == ["nested.exact_outcome_hash"]


def test_scenario_cl_6853_duplicate_decisions_block_readiness() -> None:
    """SCENARIO-CL-6853-DUPLICATES: one decision identity occurs once."""

    row = _minimal_row()
    checks = exp.validate_opportunity_rows([row, deepcopy(row)])
    duplicate = next(item for item in checks if item["check"] == "unique_decision_identities")
    assert duplicate["passed"] is False
    assert duplicate["observed"]["duplicate_count"] == 1


def test_scenario_cl_6853_missing_later_outcome_blocks_fixture() -> None:
    """SCENARIO-CL-6853-PRECONDITIONS: every decision needs exact later authority."""

    row = _minimal_row()
    row["exact_later_outcome"] = None
    checks = exp.validate_opportunity_rows([row])
    outcome = next(item for item in checks if item["check"] == "exact_later_outcomes_complete")
    assert outcome["passed"] is False
    assert outcome["observed"]["missing_count"] == 1


def test_scenario_cl_6853_invalid_action_availability_blocks_fixture() -> None:
    """SCENARIO-CL-6853-ACTIONS: each decision exposes all three actions."""

    row = _minimal_row()
    row["available_actions"].remove("abstain")
    row["action_availability"]["abstain"] = False
    checks = exp.validate_opportunity_rows([row])
    actions = next(item for item in checks if item["check"] == "first_class_actions_available")
    assert actions["passed"] is False
    assert actions["observed"]["invalid_count"] == 1


def test_scenario_cl_6853_zero_headroom_is_preserved() -> None:
    """SCENARIO-CL-6853-HEADROOM: ambiguous outcomes remain zero-headroom controls."""

    row = _minimal_row()
    row["exact_later_outcome"]["signed_direction"] = 0
    row["memory_effect_class"] = "ambiguous"
    row["safe_selection_headroom"] = 0
    summary = exp.summarize_headroom([row])
    assert summary["zero_headroom_count"] == 1
    assert summary["nonzero_headroom_count"] == 0


def test_scenario_cl_6853_delayed_correction_is_in_context_and_strata(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6853-HEADROOM: delayed corrections remain selectable strata."""

    delayed = [row for row in current_artifact["rows"] if row["delayed_correction"]]
    assert delayed
    assert all(
        row["decision_context"]["correction_status"] == "delayed_correction_pending"
        for row in delayed
    )
    assert any("delayed_correction" in row["stratum"] for row in delayed)


def test_scenario_cl_6853_family_imbalance_blocks_readiness() -> None:
    """SCENARIO-CL-6853-FAMILY-BALANCE: unequal family counts fail closed."""

    rows = [_minimal_row("a-1"), _minimal_row("a-2"), _minimal_row("b-1")]
    rows[-1]["family"] = "family-b"
    rows[-1]["decision_context"]["family"] = "family-b"
    check = exp.family_balance_check(rows)
    assert check["passed"] is False
    assert check["observed"] == {"family-a": 2, "family-b": 1}


def test_req_cl_6853_does_not_import_residual_decisions(current_artifact: dict) -> None:
    """REQ-CL-6853: the new rows retain exact outcomes but no residual learner fields."""

    forbidden = {
        "memory_dose",
        "predicted_direction",
        "admission_decision",
        "negative_transfer_delta",
        "residual_pressure",
    }
    assert forbidden.isdisjoint(set(exp.walk_keys(current_artifact["rows"])))
    manifest = current_artifact["outcome_authority_manifest"]
    assert manifest["producer_modules_imported"] == []
    assert manifest["counterfactual_outcomes_fabricated"] is False


def test_req_cl_6853_actions_and_baselines_are_first_class(current_artifact: dict) -> None:
    """REQ-CL-6853: actions and baseline policies have distinct bounded roles."""

    manifest = current_artifact["action_manifest"]
    assert set(manifest["first_class_actions"]) == set(exp.FIRST_CLASS_ACTIONS)
    assert set(manifest["comparison_baselines"]) == {"random_admission", "always_memory"}
    for row in current_artifact["rows"]:
        support = row["observed_potential_outcome_support"]
        assert support["verified_memory"]["observed"] is True
        assert support["no_memory"]["observed"] is False
        assert support["abstain"]["observed"] is False


def test_scenario_cl_6853_ready_requires_helpful_and_harmful_cases(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6853-READY: positive controls exist in both memory directions."""

    assert current_artifact["helpful_memory_count"] > 0
    assert current_artifact["harmful_memory_count"] > 0
    assert current_artifact["abstention_opportunity_count"] > 0
    summary = exp.summarize_headroom(current_artifact["rows"])
    assert summary["nonzero_headroom_count"] > 0
    assert summary["zero_headroom_count"] > 0
    assert all(stratum["decision_count"] > 0 for stratum in summary["by_stratum"].values())


def test_scenario_cl_6853_failed_v599_gate_writes_blocked_fixture(
    current_sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6853-PRECONDITIONS: a closed V599 gate blocks construction."""

    changed = deepcopy(current_sources)
    changed["evidence_contract"]["v599_evidence_contract_ready_score"] = 0
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        sources=changed,
    )
    assert artifact["risk_sensitive_stream_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_risk_sensitive_memory_opportunity_fixture"
    )
    assert artifact["gate_check_summary"]["failed_check"] == (
        "v599_evidence_contract_ready_score"
    )
    assert artifact["gate_check_summary"]["observed"] == 0


def test_req_cl_6853_checksum_is_stable_and_duration_independent(
    current_artifact: dict,
) -> None:
    """REQ-CL-6853: measured wall time cannot change stable fixture identity."""

    changed = deepcopy(current_artifact)
    changed["duration_s"] = 999.0
    assert exp.reproducibility_checksum(changed) == current_artifact["reproducibility_checksum"]


def test_req_cl_6853_artifact_matches_current_build(current_artifact: dict) -> None:
    """REQ-CL-6853: the checked-in deliverable matches deterministic construction."""

    path = REPO / exp.RESULT_RELATIVE_PATH
    if not path.is_file():
        pytest.skip("result is generated after the test-first implementation step")
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected = deepcopy(current_artifact)
    expected["duration_s"] = stored["duration_s"]
    expected["reproducibility_checksum"] = exp.reproducibility_checksum(expected)
    assert stored == expected


def test_req_cl_6853_source_failures_remain_machine_readable(tmp_path: Path) -> None:
    """SCENARIO-CL-6853-PRECONDITIONS: file failures do not become empty evidence."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    loaded = exp.load_sources(
        {"missing": tmp_path / "missing.json", "invalid": invalid, "array": array}
    )
    assert loaded["missing"] == {"_load_error": "FileNotFoundError"}
    assert loaded["invalid"] == {"_load_error": "JSONDecodeError"}
    assert loaded["array"] == {"_load_error": "not_object"}
    assert exp.sha256_file(tmp_path / "missing.json") is None


def test_req_cl_6853_hash_manifest_supports_external_checkouts(tmp_path: Path) -> None:
    """REQ-CL-6853: source paths stay auditable outside the canonical checkout."""

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    paths = {name: source for name in exp.SOURCE_RELATIVE_PATHS}
    manifest = exp.source_artifact_hashes(paths)
    assert all(row["path"] == str(source) for row in manifest.values())
    assert all(row["sha256"] == exp.sha256_file(source) for row in manifest.values())


def test_req_cl_6853_defensive_source_parsers_reject_malformed_rows() -> None:
    """REQ-CL-6853: malformed chronology and outcome rows cannot enter the fixture."""

    assert exp.select_chronological_rows({"rows": "not-a-list"}) == []
    assert exp._order_index("invalid") == 10**9
    index, conflicts = exp.build_outcome_index(
        {"rows": "not-a-list"},
        {
            "rows": [
                "not-an-object",
                {"source_event_row_id": "", "exact_outcome": {}},
                {
                    "source_event_row_id": "decision",
                    "exact_outcome": {
                        "outcome_identity": "one",
                        "exact_outcome_hash": "hash-one",
                        "signed_direction": 1,
                    },
                },
                {
                    "source_event_row_id": "decision",
                    "exact_outcome": {
                        "outcome_identity": "two",
                        "exact_outcome_hash": "hash-two",
                        "signed_direction": -1,
                    },
                },
            ]
        },
    )
    assert index["decision"]["signed_direction"] == 1
    assert len(conflicts) == 1


def test_req_cl_6853_unknown_context_values_are_conservative() -> None:
    """REQ-CL-6853: malformed visible metadata receives conservative fixed features."""

    context = exp.decision_context(
        {
            "event_id": "malformed",
            "chronological_position": 2,
            "write_operation_id": "malformed",
        }
    )
    assert context["relevance"] == 0.0
    assert context["uncertainty"] == 1.0
    assert context["correction_status"] == "correction_status_unknown"
    assert context["age"] == 2


def test_scenario_cl_6853_missing_join_and_nested_list_leakage_are_detected() -> None:
    """SCENARIO-CL-6853-LEAKAGE: missing joins and list injections fail visibly."""

    rows, missing = exp.build_rows([{"row_id": "missing"}], {})
    assert rows == []
    assert missing == ["missing"]
    assert exp.find_leakage_paths([{"signed_direction": 1}]) == ["[0].signed_direction"]
    attacks = exp.leakage_attack_results(
        [{"decision_context": {"nested": {"predicted_direction": 1}}}]
    )
    assert attacks["passed"] is False
    assert attacks["clean_fixture_leakage_paths"] == [
        "rows[0].decision_context.nested.predicted_direction"
    ]


def test_req_cl_6853_artifact_validation_rejects_each_contract_error(
    current_artifact: dict,
) -> None:
    """REQ-CL-6853: the writer refuses inconsistent ready artifacts."""

    changed = deepcopy(current_artifact)
    changed.pop("rows")
    changed["inference_substrate"] = "wrong"
    changed["verifier_is_oracle"] = True
    changed["honest_verdict"] = "pending"
    errors = exp.validate_artifact(changed)
    assert any("missing required fields" in error for error in errors)
    assert "invalid inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "reproducibility_checksum mismatch" in errors

    contradicted = deepcopy(current_artifact)
    contradicted["gate_check_summary"]["passed"] = False
    contradicted["rows"] = contradicted["rows"][:-1]
    contradicted["reproducibility_checksum"] = exp.reproducibility_checksum(contradicted)
    errors = exp.validate_artifact(contradicted)
    assert "ready score contradicts failed gates" in errors
    assert "ready fixture is family-imbalanced" in errors


def test_req_cl_6853_writer_and_cli_use_requested_temporary_path(
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6853: validated writes and CLI replay stay at the requested path."""

    invalid_path = tmp_path / "invalid.json"
    invalid = deepcopy(current_artifact)
    invalid["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        exp.write_artifact(invalid_path, invalid)

    direct_path = tmp_path / "nested" / "direct.json"
    exp.write_artifact(direct_path, current_artifact)
    assert json.loads(direct_path.read_text(encoding="utf-8")) == current_artifact

    cli_path = tmp_path / "cli.json"
    assert exp.main(["--date", "20260901", "--output", str(cli_path)]) == 0
    cli_artifact = json.loads(cli_path.read_text(encoding="utf-8"))
    assert cli_artifact["risk_sensitive_stream_ready_score"] == 1
