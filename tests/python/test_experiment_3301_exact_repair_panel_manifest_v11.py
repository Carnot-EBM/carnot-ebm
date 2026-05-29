"""Tests for Exp 3301 exact repair panel manifest v11.

Spec refs: REQ-VERIFY-3301, SCENARIO-VERIFY-3301.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import exact_repair_panel_manifest_v11 as mod


REQUIRED_FIELDS = {
    "repair_panel_manifest_ready",
    "panel_case_count",
    "case_family_counts",
    "exact_checker_types",
    "llm_judge_required_count",
    "panel_cases_path",
    "case_hashes",
    "localized_feedback_coverage",
    "known_failing_candidate_count",
    "validation_commands",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_verify_3301_spec_anchor_declares_manifest_schema() -> None:
    """REQ-VERIFY-3301: OpenSpec names the fixed exact manifest contract first."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3301" in spec
    assert "SCENARIO-VERIFY-3301" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.PANEL_CASES_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3301_writes_stratified_exact_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3301: the written JSONL panel is parseable and exact."""

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=10.0,
        now_s=12.75,
        tests_run=["SCENARIO-VERIFY-3301"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    panel_path = tmp_path / mod.PANEL_CASES_REL_PATH
    panel_cases = _read_jsonl(panel_path)

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_panel_manifest_ready"] is True
    assert artifact["panel_case_count"] == 30
    assert artifact["case_family_counts"] == {
        "arithmetic_exact_rows": 6,
        "bounded_logical_consistency": 6,
        "code_output_checks": 6,
        "context_shortcuts": 6,
        "symbolic_aliases": 6,
    }
    assert artifact["exact_checker_types"] == [
        "exact_alias_string",
        "exact_bool_string",
        "exact_context_string",
        "exact_integer_string",
        "exact_stdout_string",
    ]
    assert artifact["llm_judge_required_count"] == 0
    assert artifact["panel_cases_path"] == mod.PANEL_CASES_REL_PATH.as_posix()
    assert len(set(artifact["case_hashes"])) == 30
    assert artifact["localized_feedback_coverage"] == 1.0
    assert artifact["known_failing_candidate_count"] == 30
    assert artifact["inference_substrate"] == "deterministic_exact_manifest_no_live_inference"
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["validation_commands"] == ["SCENARIO-VERIFY-3301"]

    assert len(panel_cases) == 30
    assert [case["case_hash"] for case in panel_cases] == artifact["case_hashes"]
    for case in panel_cases:
        assert mod.REQUIRED_CASE_FIELDS <= set(case)
        assert case["localized_repair_feedback"]
        assert case["llm_judge_required"] is False
        assert mod.case_hash(case) == case["case_hash"]
        assert mod.exact_check(case, case["expected_answer"]) is True
        assert mod.exact_check(case, case["failing_candidate"]) is False
    mod.validate_cases(panel_cases)
    mod.validate_artifact(artifact)


def test_req_verify_3301_validation_rejects_overclaiming_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3301: exactness, feedback, hashes, and LLM-judge bans fail closed."""

    cases = mod.build_panel_cases()
    artifact = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        tests_run=["REQ-VERIFY-3301"],
    )

    assert mod.normalize_text("  North   Door ") == "north door"
    assert mod.parse_int_string("+12") == 12
    assert mod.parse_int_string("12.0") is None
    assert mod.normalize_bool_string("YES") == "true"
    assert mod.normalize_bool_string("0") == "false"
    assert mod.normalize_bool_string("maybe") == "maybe"
    assert mod.parse_int_string("-7") == -7
    assert mod.parse_int_string("-x") is None
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(3.0, 1.0) == 0.0
    assert mod.mapping_list("bad") == []

    duplicate_hash = [dict(case) for case in cases]
    duplicate_hash[1]["case_hash"] = duplicate_hash[0]["case_hash"]
    with pytest.raises(ValueError, match="case hashes"):
        mod.validate_cases(duplicate_hash)

    missing_feedback = [dict(case) for case in cases]
    missing_feedback[0]["localized_repair_feedback"] = ""
    missing_feedback[0]["case_hash"] = mod.case_hash(missing_feedback[0])
    with pytest.raises(ValueError, match="localized repair feedback"):
        mod.validate_cases(missing_feedback)

    candidate_not_failing = [dict(case) for case in cases]
    candidate_not_failing[0]["failing_candidate"] = candidate_not_failing[0]["expected_answer"]
    candidate_not_failing[0]["case_hash"] = mod.case_hash(candidate_not_failing[0])
    with pytest.raises(ValueError, match="known failing candidate"):
        mod.validate_cases(candidate_not_failing)

    expected_not_passing = [dict(case) for case in cases]
    expected_not_passing[6]["expected_answer"] = "not-the-answer"
    expected_not_passing[6]["case_hash"] = mod.case_hash(expected_not_passing[6])
    with pytest.raises(ValueError, match="expected answer"):
        mod.validate_cases(expected_not_passing)

    llm_judge_case = [dict(case) for case in cases]
    llm_judge_case[2]["llm_judge_required"] = True
    llm_judge_case[2]["case_hash"] = mod.case_hash(llm_judge_case[2])
    with pytest.raises(ValueError, match="LLM judge"):
        mod.validate_cases(llm_judge_case)

    too_few = cases[:-1]
    with pytest.raises(ValueError, match="at least 30"):
        mod.validate_cases(too_few)

    too_few_families = [dict(case, family="one_family") for case in cases]
    with pytest.raises(ValueError, match="five case families"):
        mod.validate_cases(too_few_families)

    missing_field = [dict(case) for case in cases]
    del missing_field[0]["context"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_cases(missing_field)

    unsupported_checker = [dict(case) for case in cases]
    unsupported_checker[0]["exact_checker_type"] = "llm_judge"
    unsupported_checker[0]["case_hash"] = mod.case_hash(unsupported_checker[0])
    with pytest.raises(ValueError, match="unsupported exact checker"):
        mod.validate_cases(unsupported_checker)

    stale_hash = [dict(case) for case in cases]
    stale_hash[0]["context"] += " Changed after hashing."
    with pytest.raises(ValueError, match="stale case_hash"):
        mod.validate_cases(stale_hash)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="repair_panel_manifest_ready"):
        mod.validate_artifact(artifact | {"repair_panel_manifest_ready": "true"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="panel_case_count"):
        mod.validate_artifact(artifact | {"panel_case_count": 29})
    with pytest.raises(ValueError, match="known_failing_candidate_count"):
        mod.validate_artifact(artifact | {"known_failing_candidate_count": 29})
    with pytest.raises(ValueError, match="localized_feedback_coverage"):
        mod.validate_artifact(artifact | {"localized_feedback_coverage": 0.99})
    with pytest.raises(ValueError, match="LLM judge"):
        mod.validate_artifact(artifact | {"llm_judge_required_count": 1})
    with pytest.raises(ValueError, match="case_family_counts"):
        mod.validate_artifact(artifact | {"case_family_counts": []})
    with pytest.raises(ValueError, match="at least five"):
        mod.validate_artifact(artifact | {"case_family_counts": {"one": 30}})
    with pytest.raises(ValueError, match="exact_checker_types"):
        mod.validate_artifact(artifact | {"exact_checker_types": "bad"})
    with pytest.raises(ValueError, match="one hash per case"):
        mod.validate_artifact(artifact | {"case_hashes": []})
    with pytest.raises(ValueError, match="case_hashes must be unique"):
        mod.validate_artifact(artifact | {"case_hashes": [artifact["case_hashes"][0]] * 30})
    with pytest.raises(ValueError, match="validation_commands"):
        mod.validate_artifact(artifact | {"validation_commands": "bad"})
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(artifact | {"duration_s": -1})
