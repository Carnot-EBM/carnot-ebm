"""Tests for Exp 3210 Context-CoT/CL-Bench parametric-shortcut fixtures.

Spec refs: REQ-VERIFY-3210, SCENARIO-VERIFY-3210.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import context_cot_clbench_parametric_shortcut_fixtures_v1 as mod


REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "reference_papers",
    "fixture_path",
    "fixture_count",
    "fixture_families",
    "exact_checker_types",
    "prior_bait_row_count",
    "context_following_score_available",
    "optional_llm_smoke",
    "ready_for_clean_verifier",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}

REQUIRED_ROW_FIELDS = {
    "fixture_id",
    "family",
    "context",
    "question",
    "expected_answer",
    "prior_bait_answer",
    "exact_checker_type",
    "minimal_counterexample",
}


def test_req_verify_3210_spec_anchor_and_paths_exist() -> None:
    """REQ-VERIFY-3210: OpenSpec declares the fixture and result artifacts."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3210" in spec
    assert "SCENARIO-VERIFY-3210" in spec
    assert mod.FIXTURE_REL_PATH.as_posix() in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_verify_3210_rows_cover_context_shortcut_families() -> None:
    """SCENARIO-VERIFY-3210: rows encode local context over prior shortcuts."""

    rows = mod.build_fixture_rows()
    summary = mod.fixture_bank_summary(rows)

    assert len(rows) == 30
    assert summary["fixture_count"] == 30
    assert summary["fixture_families"] == list(mod.FIXTURE_FAMILIES)
    assert summary["exact_checker_types"] == list(mod.EXACT_CHECKER_TYPES)
    assert summary["prior_bait_row_count"] == 30

    seen_ids: set[str] = set()
    for row in rows:
        assert REQUIRED_ROW_FIELDS <= set(row)
        assert row["fixture_id"] not in seen_ids
        seen_ids.add(row["fixture_id"])
        assert row["family"] in mod.FIXTURE_FAMILIES
        assert row["exact_checker_type"] in mod.EXACT_CHECKER_TYPES
        assert row["context"]
        assert row["question"]
        assert row["expected_answer"] != row["prior_bait_answer"]
        assert row["minimal_counterexample"]["candidate_answer"] == row["prior_bait_answer"]
        assert row["minimal_counterexample"]["failure_mode"] == "parametric_prior_shortcut"

        expected_check = mod.check_answer(row, str(row["expected_answer"]))
        prior_check = mod.check_answer(row, str(row["prior_bait_answer"]))
        assert expected_check["accepted"] is True
        assert prior_check["accepted"] is False
        assert prior_check["failure_reason"] == "answer_does_not_match_context_expected"

    assert mod.validate_fixture_bank(rows) == rows


def test_req_verify_3210_writer_materializes_jsonl_and_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-3210: writer persists a schema-valid fixture bank artifact."""

    result_path = mod.write_artifacts(
        tmp_path,
        tests_run=["REQ-VERIFY-3210 writer"],
    )
    fixture_path = tmp_path / mod.FIXTURE_REL_PATH
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result_path == tmp_path / mod.OUTPUT_REL_PATH
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3210"
    assert artifact["milestone"] == "2026.05.297"
    assert artifact["fixture_path"] == mod.FIXTURE_REL_PATH.as_posix()
    assert artifact["fixture_count"] == len(rows) == 30
    assert artifact["fixture_families"] == list(mod.FIXTURE_FAMILIES)
    assert artifact["exact_checker_types"] == list(mod.EXACT_CHECKER_TYPES)
    assert artifact["prior_bait_row_count"] == len(rows)
    assert artifact["context_following_score_available"] is True
    assert artifact["optional_llm_smoke"] is None
    assert artifact["ready_for_clean_verifier"] is True
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["REQ-VERIFY-3210 writer"]
    assert mod.validate_artifact(artifact, rows) == artifact


def test_req_verify_3210_validation_and_scoring_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-3210: exact scoring rejects prior-bait and malformed rows."""

    rows = mod.build_fixture_rows()
    expected_answers = {str(row["fixture_id"]): str(row["expected_answer"]) for row in rows}
    prior_answers = {str(row["fixture_id"]): str(row["prior_bait_answer"]) for row in rows}

    assert mod.context_following_score(rows, expected_answers) == 1.0
    assert mod.context_following_score(rows, prior_answers) == 0.0
    assert mod.context_following_score([], {}) is None
    assert mod.normalize_answer("  Banana. ") == "banana"

    with pytest.raises(ValueError, match="fixture bank count"):
        mod.validate_fixture_bank(rows[:19])

    missing_field = dict(rows[0])
    missing_field.pop("context")
    with pytest.raises(ValueError, match="missing required row fields"):
        mod.validate_fixture_bank([missing_field, *rows[1:]])

    bad_family = dict(rows[0], family="unsupported")
    with pytest.raises(ValueError, match="unsupported family"):
        mod.validate_fixture_bank([bad_family, *rows[1:]])

    bad_checker = dict(rows[0], exact_checker_type="unsupported")
    with pytest.raises(ValueError, match="unsupported checker"):
        mod.validate_fixture_bank([bad_checker, *rows[1:]])

    bad_prior = dict(rows[0], prior_bait_answer=rows[0]["expected_answer"])
    bad_prior["minimal_counterexample"] = {
        **dict(bad_prior["minimal_counterexample"]),
        "candidate_answer": rows[0]["expected_answer"],
    }
    with pytest.raises(ValueError, match="prior-bait answer must differ"):
        mod.validate_fixture_bank([bad_prior, *rows[1:]])

    bad_counterexample = dict(rows[0])
    bad_counterexample["minimal_counterexample"] = {
        **dict(bad_counterexample["minimal_counterexample"]),
        "candidate_answer": "not-the-prior",
    }
    with pytest.raises(ValueError, match="minimal counterexample"):
        mod.validate_fixture_bank([bad_counterexample, *rows[1:]])

    duplicate_id = dict(rows[1], fixture_id=rows[0]["fixture_id"])
    with pytest.raises(ValueError, match="duplicate fixture_id"):
        mod.validate_fixture_bank([rows[0], duplicate_id, *rows[2:]])

    with pytest.raises(ValueError, match="all required families"):
        mod.validate_fixture_bank(rows[:20])

    missing_checker_rows = [
        dict(row, exact_checker_type="exact_alias_string")
        if row["family"] == "context_defined_entity_facts"
        else row
        for row in rows
    ]
    with pytest.raises(ValueError, match="all exact checker types"):
        mod.validate_fixture_bank(missing_checker_rows)

    original_check = mod.check_answer

    def fail_expected(row: dict[str, Any], answer: Any) -> dict[str, Any]:
        if answer == row["expected_answer"]:
            return {"accepted": False}
        return original_check(row, answer)

    monkeypatch.setattr(mod, "check_answer", fail_expected)
    with pytest.raises(ValueError, match="expected answer fails checker"):
        mod.validate_fixture_bank(rows)

    def pass_prior(row: dict[str, Any], answer: Any) -> dict[str, Any]:
        if answer == row["prior_bait_answer"]:
            return {"accepted": True}
        return original_check(row, answer)

    monkeypatch.setattr(mod, "check_answer", pass_prior)
    with pytest.raises(ValueError, match="prior-bait answer passes checker"):
        mod.validate_fixture_bank(rows)
    monkeypatch.setattr(mod, "check_answer", original_check)

    valid_artifact = mod.build_artifact(tests_run=["REQ-VERIFY-3210 validation"])
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact({}, rows)
    with pytest.raises(ValueError, match="fixture_count"):
        mod.validate_artifact(valid_artifact | {"fixture_count": 29}, rows)
    with pytest.raises(ValueError, match="ready_for_clean_verifier"):
        mod.validate_artifact(valid_artifact | {"ready_for_clean_verifier": False}, rows)
    with pytest.raises(ValueError, match="mandated local SOTA GGUF"):
        mod.validate_artifact(valid_artifact | {"optional_llm_smoke": {"model_specs": []}}, rows)
    with pytest.raises(ValueError, match="optional_llm_smoke"):
        mod.validate_artifact(valid_artifact | {"optional_llm_smoke": "bad-smoke"}, rows)

    monkeypatch.setattr(mod, "write_artifacts", lambda: Path("results/fake-exp3210.json"))
    mod.main()
    assert capsys.readouterr().out.strip() == "results/fake-exp3210.json"


def test_req_verify_3210_llm_smoke_model_spec_gate() -> None:
    """REQ-VERIFY-3210: any optional smoke must name a mandated local SOTA GGUF."""

    rows = mod.build_fixture_rows()
    smoke = {"model_specs": [{"model_id": mod.MANDATED_LOCAL_SOTA_GGUF[0]}], "smoke_only": True}
    artifact = mod.build_artifact(optional_llm_smoke=smoke)

    assert artifact["optional_llm_smoke"] == smoke
    assert mod.validate_artifact(artifact, rows) == artifact

    bad_smoke = {"model_specs": [{"model_id": "legacy/small-smoke-only"}], "smoke_only": True}
    with pytest.raises(ValueError, match="mandated local SOTA GGUF"):
        mod.validate_artifact(artifact | {"optional_llm_smoke": bad_smoke}, rows)
