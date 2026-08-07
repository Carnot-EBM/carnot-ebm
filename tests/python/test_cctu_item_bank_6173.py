"""Tests for the Exp6173 CCTU item-bank preregistration.

Spec: REQ-VERIFY-6173, SCENARIO-VERIFY-6173-BANK-FREEZE,
SCENARIO-VERIFY-6173-VALIDATORS.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from carnot.verify import cctu_item_bank_6173 as exp


def test_req_verify_6173_freezes_120_cases_and_balanced_split() -> None:
    """REQ-VERIFY-6173: the frozen bank has 120 cases and a sealed 60/60 split."""

    bank = exp.build_item_bank()
    split = exp.build_split(bank)
    matrix = exp.constraint_taxonomy_balance_matrix(bank, split)

    assert len(bank) == 120
    assert len({case.case_id for case in bank}) == 120
    assert len(split["calibration_ids"]) == 60
    assert len(split["held_ids"]) == 60
    assert set(split["calibration_ids"]).isdisjoint(split["held_ids"])
    assert set(split["calibration_ids"]) | set(split["held_ids"]) == {case.case_id for case in bank}
    assert matrix["taxonomy_categories"] == list(exp.REQUIRED_TAXONOMY)
    for category in exp.REQUIRED_TAXONOMY:
        counts = matrix["by_primary_constraint"][category]
        assert counts["total"] == 15
        assert abs(counts["calibration"] - counts["held"]) <= 1
    assert all(case.input_bytes_sha256.startswith("sha256:") for case in bank)
    assert exp.audit_no_finite_choice_or_answer_position(bank)["passed"] is True


def test_scenario_verify_6173_exact_validators_accept_and_reject_controls() -> None:
    """SCENARIO-VERIFY-6173-VALIDATORS: controls exercise exact replay labels."""

    bank = exp.build_item_bank()
    controls = exp.run_validator_controls(bank)

    assert controls["known_valid"]["passed"] == 120
    assert controls["known_valid"]["failed"] == 0
    assert controls["single_violation"]["caught"] == controls["single_violation"]["total"]
    assert controls["multi_violation"]["caught"] == controls["multi_violation"]["total"]
    assert controls["parser_adversarial"]["caught"] == controls["parser_adversarial"]["total"]
    assert controls["metamorphic"]["passed"] == controls["metamorphic"]["total"]
    assert controls["independence_audit"]["candidate_provenance_invariant"] is True
    assert controls["independence_audit"]["surface_length_invariant"] is True
    assert controls["independence_audit"]["arbitrary_id_invariant"] is True

    cross_case = next(case for case in bank if case.primary_constraint == "cross_step_dependency")
    trace = exp.known_valid_trace(cross_case)
    bad_trace = exp.mutate_trace(trace, "break_dependency")
    result = exp.validate_candidate_trace(cross_case, bad_trace)
    assert result["terminal_passed"] is False
    assert "cross_step_dependency" in {v["category"] for v in result["violations"]}


def test_req_verify_6173_leakage_and_position_audits_are_negative_controls(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6173: leakage and finite-choice audits catch forbidden channels."""

    bank = exp.build_item_bank()
    bad_case = dataclasses.replace(
        bank[0],
        prompt=bank[0].prompt + "\nChoose exactly one option: A. left B. right",
    )
    assert exp.audit_no_finite_choice_or_answer_position([bad_case])["passed"] is False

    candidate_path = tmp_path / "experiment_6173_candidate_outcomes.jsonl"
    clean = exp.capture_preconditions(
        candidate_outcome_paths=(candidate_path,),
        model_cache_roots=(tmp_path / "model-cache",),
        result_root=tmp_path / "results",
        test_root=tmp_path / "tests",
    )
    assert clean["candidate_outcome_file_exists"] is False
    assert clean["model_cache_metadata"]["model_loader_invocations"] == 0
    assert clean["model_cache_metadata"]["content_hash_policy"] == "metadata_only"

    candidate_path.write_text('{"case_id": "leak"}\n', encoding="utf-8")
    leaked = exp.capture_preconditions(
        candidate_outcome_paths=(candidate_path,),
        model_cache_roots=(tmp_path / "model-cache",),
        result_root=tmp_path / "results",
        test_root=tmp_path / "tests",
    )
    assert leaked["candidate_outcome_file_exists"] is True
    assert leaked["candidate_outcome_paths_existing"] == [str(candidate_path)]


def test_req_verify_6173_writes_required_artifacts_and_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-6173: the result artifact exposes every required field."""

    config = exp.ExperimentConfig(
        artifact_path=tmp_path / exp.RESULT_FILENAME,
        bank_path=tmp_path / exp.BANK_FILENAME,
        split_path=tmp_path / exp.SPLIT_FILENAME,
        held_access_log_path=tmp_path / exp.HELD_ACCESS_LOG_FILENAME,
        started_at=10.0,
        clock=lambda: 15.25,
        test_commands=("focused pytest", "coverage new module"),
        test_exit_codes={"focused pytest": 0, "coverage new module": 0},
    )

    artifact = exp.write_frozen_artifacts(config)
    persisted = json.loads((tmp_path / exp.RESULT_FILENAME).read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["cctu_item_bank_ready_score"] == 1.0
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["duration_s"] == 5.25
    assert artifact["test_exit_codes"] == {"focused pytest": 0, "coverage new module": 0}
    assert artifact["cctu_item_bank_path_hash_count_and_schema"]["count"] == 120
    assert artifact["calibration_and_held_split_path_hash_counts"]["calibration_count"] == 60
    assert artifact["calibration_and_held_split_path_hash_counts"]["held_count"] == 60
    assert artifact["held_seal_and_access_log_path_hash"]["access_count"] == 0
    assert Path(artifact["cctu_item_bank_path_hash_count_and_schema"]["path"]).exists()
    assert Path(artifact["held_seal_and_access_log_path_hash"]["access_log_path"]).exists()


def test_req_verify_6173_reproducibility_checksum_is_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-6173: deterministic inputs produce identical freeze checksums."""

    first = exp.build_experiment_artifact(
        exp.ExperimentConfig(
            artifact_path=tmp_path / "first.json",
            bank_path=tmp_path / "first.jsonl",
            split_path=tmp_path / "first.split.json",
            held_access_log_path=tmp_path / "first.access.json",
            started_at=1.0,
            clock=lambda: 2.0,
            test_commands=("cmd",),
            test_exit_codes={"cmd": 0},
        )
    )
    second = exp.build_experiment_artifact(
        exp.ExperimentConfig(
            artifact_path=tmp_path / "second.json",
            bank_path=tmp_path / "second.jsonl",
            split_path=tmp_path / "second.split.json",
            held_access_log_path=tmp_path / "second.access.json",
            started_at=1.0,
            clock=lambda: 2.0,
            test_commands=("cmd",),
            test_exit_codes={"cmd": 0},
        )
    )

    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["field_provenance"]["verifier_is_oracle"] == [
        "REQ-VERIFY-6173 exact validator oracle declaration"
    ]


def test_req_verify_6173_defensive_branches_and_cli_are_covered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6173: defensive validator branches stay executable."""

    bank = exp.build_item_bank()
    case = bank[0]
    trace = exp.known_valid_trace(case)

    assert exp.ExperimentConfig(
        candidate_outcome_paths=(tmp_path / "candidate.jsonl",)
    ).resolved_candidate_outcome_paths() == (tmp_path / "candidate.jsonl",)
    with pytest.raises(ValueError, match="unknown trace mutation"):
        exp.mutate_trace(trace, "not-a-mutation")

    assert exp.validate_candidate_trace(case, json.dumps(trace))["terminal_passed"] is True
    assert exp.execute_tool(
        "math.aggregate",
        {"operation": "mod", "value": 137, "modulus": 11},
    ) == {"value": 5}
    with pytest.raises(ValueError, match="unknown math operation"):
        exp.execute_tool("math.aggregate", {"operation": "mystery"})
    with pytest.raises(ValueError, match="unknown text operation"):
        exp.execute_tool("text.transform", {"text": "x", "operations": [{"op": "mystery"}]})
    with pytest.raises(ValueError, match="unknown tool"):
        exp.execute_tool("missing.tool", {})

    assert exp.execute_tool(
        "table.filter",
        {"rows": [{"x": 1}, {"x": 2}], "where": {"x_max": 1}, "select": "__count__"},
    ) == {"rows": 1}
    assert exp.execute_tool(
        "table.filter",
        {"rows": [{"name": "night-a"}], "where": {"name_contains": "day"}, "select": "name"},
    ) == {"rows": []}
    assert exp.execute_tool("list.take", {"items": [1, 2, 3], "count": 2}) == {"items": [1, 2]}
    assert exp._answer_from_step({"result": {"rows": 2}}) == {"answer": "2", "abstain": False}
    assert exp._answer_from_step({"result": {"text": "done"}}) == {
        "answer": "done",
        "abstain": False,
    }
    with pytest.raises(ValueError, match="cannot build answer"):
        exp._answer_from_step({"result": {"unknown": True}})

    schema_cases = [
        {**trace, "case_id": "wrong"},
        {**trace, "steps": "not-list"},
        {**trace, "final": "not-dict"},
        {**trace, "verifier": "not-dict"},
        {**trace, "metadata": "not-dict"},
        {**trace, "steps": ["not-dict"]},
        {**trace, "steps": [{k: v for k, v in trace["steps"][0].items() if k != "result"}]},
    ]
    for candidate in schema_cases:
        result = exp.validate_candidate_trace(case, candidate)
        assert result["terminal_passed"] is False
        assert "response_schema" in {violation["category"] for violation in result["violations"]}

    bad_resource = exp.known_valid_trace(case)
    bad_resource["steps"][0]["resource_units"] = "not-an-int"
    assert exp.validate_candidate_trace(case, bad_resource)["terminal_passed"] is False
    assert exp._final_response_ok(case, None) is False
    assert exp._impossible_abstention_ok(case, None) is False
    assert exp._verifier_ok(None, True) is False
    assert exp._steps(None) == []
    assert exp._get_path({}, "not-a-path") is None

    dependency_case = next(
        item for item in bank if item.primary_constraint == "cross_step_dependency"
    )
    missing_source = exp.known_valid_trace(dependency_case)
    missing_source["steps"][1]["dependency_checks"][0]["from_step"] = "missing"
    assert exp.validate_candidate_trace(dependency_case, missing_source)["terminal_passed"] is False

    composition_case = next(item for item in bank if item.primary_constraint == "compositional")
    contains_miss = exp.known_valid_trace(composition_case)
    contains_miss["steps"][2]["arguments"]["text"] = "no total here"
    assert exp.validate_candidate_trace(composition_case, contains_miss)["terminal_passed"] is False

    no_dependency = exp.known_valid_trace(case)
    exp._break_first_dependency(no_dependency)
    assert no_dependency["steps"][0]["dependency_checks"][0]["from_step"] == "missing"

    monkeypatch.setattr(exp, "OPERATOR_CURATED_PATHS", ("definitely-missing-protected.md",))
    protected = exp.protected_file_hashes()
    assert any(row["path"] == "definitely-missing-protected.md" for row in protected["files"])

    monkeypatch.setattr(
        exp.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom"))
    )
    assert exp._git_status_short_hash()["available"] is False

    monkeypatch.setattr(exp, "write_frozen_artifacts", lambda: {"status": "complete_ready"})
    monkeypatch.setattr(exp, "repo_path", lambda *parts: tmp_path.joinpath(*parts))
    exp.main()
    assert "complete_ready" in capsys.readouterr().out
