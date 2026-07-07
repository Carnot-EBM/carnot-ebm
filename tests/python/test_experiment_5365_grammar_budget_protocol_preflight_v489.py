"""Tests for Exp 5365 grammar-budget structured protocol preflight.

Spec refs: REQ-VERIFY-5365, SCENARIO-VERIFY-5365.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5351_trigger_constrain_structured_protocol_v488 as exp5351
from carnot import experiment_5365_grammar_budget_protocol_preflight_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
EXP5351_PATH = REPO / exp5351.RESULT_RELATIVE_PATH
TEST_COMMAND = ".venv/bin/pytest tests/python/test_experiment_5365_grammar_budget_protocol_preflight_v489.py -q"


def test_req_verify_5365_spec_declares_preflight_contract() -> None:
    """REQ-VERIFY-5365: OpenSpec anchors the grammar-budget preflight gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5365") : spec.index("### REQ-VERIFY-5351")]

    for marker in (
        "REQ-VERIFY-5365",
        "SCENARIO-VERIFY-5365",
        str(mod.RESULT_RELATIVE_PATH),
        "schema reachability",
        "completion slack",
        "truncation risks",
        "schema risks",
        "tool/action marker reachability",
        "active_roadmap_modified=false",
        "conductor_modified=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in mod.field_provenance()[field]["principle"]


def test_req_verify_5365_schema_reachability_and_completion_slack() -> None:
    """REQ-VERIFY-5365: derived grammar reaches required fields with positive slack."""

    grammar = mod.derive_schema_grammar_summary(exp5351.DEFAULT_SCHEMA)
    cases = mod.build_schema_reachability_cases(
        exp5351.DEFAULT_CALIBRATION_PROMPTS,
        exp5351.DEFAULT_PROTOCOL_VARIANTS[0],
        grammar,
    )

    assert grammar["grammar_backend"] == "deterministic_json_schema_summary"
    assert grammar["required_fields"] == ["id", "answer", "facts"]
    assert grammar["property_types"] == {"id": "string", "answer": "string", "facts": "array"}
    assert mod.derive_schema_grammar_summary({"required": ["name"], "properties": []})[
        "property_types"
    ] == {"name": "any"}
    assert mod.minimal_valid_payload(
        {"target_final_object": {"items": "bad", "count": "3", "score": [], "flag": "no"}},
        {
            "required": ["items", "count", "score", "flag", "obj", "name"],
            "properties": {
                "items": {"type": "array"},
                "count": {"type": "integer"},
                "score": {"type": "number"},
                "flag": {"type": "boolean"},
                "obj": {"type": "object"},
                "name": {"type": "string"},
            },
        },
    ) == {"items": [], "count": 0, "score": 0.0, "flag": False, "obj": {}, "name": ""}
    assert mod.minimal_valid_payload(
        {"target_final_object": {}},
        {"required": ["name"], "properties": []},
    ) == {"name": ""}
    assert {case["prompt_id"] for case in cases} == {
        "battery_duration_probe",
        "code_word_probe",
        "route_probe",
        "count_probe",
    }
    assert all(case["schema_valid"] is True for case in cases)
    assert all(case["reachable_required_fields"] == ["id", "answer", "facts"] for case in cases)
    assert mod.required_field_reachability_rate(cases) == pytest.approx(1.0)
    assert mod.required_field_reachability_rate([]) == 0.0
    assert mod.completion_slack_min_tokens(cases) > 0
    assert mod.completion_slack_min_tokens([]) == -1
    assert mod.estimate_tokens("") == 0


def test_req_verify_5365_failure_classifier_separates_exp5351_shapes() -> None:
    """REQ-VERIFY-5365: .488 truncation and schema failures are counted separately."""

    exp5351_artifact = json.loads(EXP5351_PATH.read_text(encoding="utf-8"))
    receipts = mod.extract_exp5351_generation_receipts(exp5351_artifact)
    rows = mod.classify_failure_rows(receipts, exp5351.DEFAULT_PROTOCOL_VARIANTS[0])
    by_prompt = {row["prompt_id"]: row for row in rows}
    counts = mod.failure_counts(rows)

    assert by_prompt["battery_duration_probe"]["failure_class"] == "truncation"
    assert by_prompt["battery_duration_probe"]["schema_risk_independent_of_truncation"] is False
    assert by_prompt["code_word_probe"]["failure_class"] == "schema"
    assert by_prompt["count_probe"]["failure_class"] == "schema"
    assert by_prompt["route_probe"]["failure_class"] == "accepted"
    assert counts == {
        "accepted_count": 1,
        "parse_failure_count": 0,
        "schema_failure_count": 2,
        "truncation_failure_count": 1,
    }

    no_marker = mod.classify_protocol_failure(
        {"prompt_id": "no_marker", "stdout_tail": "plain text", "score": {"parse_success": False}},
        exp5351.DEFAULT_PROTOCOL_VARIANTS[0],
    )
    malformed_after_end = mod.classify_protocol_failure(
        {
            "prompt_id": "bad_json",
            "stdout_tail": "FINAL_JSON: {not json} END_FINAL_JSON",
            "score": {"parse_success": False},
        },
        exp5351.DEFAULT_PROTOCOL_VARIANTS[0],
    )
    timed_out = mod.classify_protocol_failure(
        {"prompt_id": "timeout", "timed_out": True, "score": {"parse_success": False}},
        exp5351.DEFAULT_PROTOCOL_VARIANTS[0],
    )
    assert no_marker == "parse"
    assert malformed_after_end == "parse"
    assert timed_out == "truncation"
    assert mod.extract_exp5351_generation_receipts({"protocol_variants": []}) == []
    assert mod._honest_verdict(ready=False, slack_min=-1).startswith("blocked_")


def test_req_verify_5365_tool_action_markers_are_reachable() -> None:
    """REQ-VERIFY-5365: deterministic fixture exposes all tool/action markers."""

    fixture = mod.build_tool_action_protocol_fixture(
        {"id": "route_probe", "answer": "north", "facts": ["ok"]},
        exp5351.DEFAULT_PROTOCOL_VARIANTS[0],
    )
    summary = mod.tool_action_token_reachability(fixture)
    missing = mod.tool_action_token_reachability("FINAL_JSON: {} END_FINAL_JSON")

    assert "TOOL_ACTION:" in fixture
    assert "END_TOOL_ACTION" in fixture
    assert summary["rate"] == pytest.approx(1.0)
    assert all(row["reachable"] for row in summary["rows"])
    assert missing["rate"] < 1.0


def test_scenario_verify_5365_run_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5365: preflight writes the .489 gate artifact without live GGUF."""

    output_path = tmp_path / "experiment_5365.json"
    artifact = mod.run(
        root=REPO,
        artifact_path=output_path,
        exp5351_path=EXP5351_PATH,
        tests_run=[TEST_COMMAND],
        started_s=10.0,
        now_s=12.25,
        write=True,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["grammar_budget_protocol_ready"] is True
    assert artifact["schema_reachability_cases"] == 4
    assert artifact["required_field_reachability_rate"] == pytest.approx(1.0)
    assert artifact["completion_slack_min_tokens"] > 0
    assert artifact["truncation_failure_count"] == 1
    assert artifact["schema_failure_count"] == 2
    assert artifact["tool_action_token_reachability_rate"] == pytest.approx(1.0)
    assert artifact["methodology_duration_s"] == pytest.approx(2.25)
    assert artifact["tests_run"] == [TEST_COMMAND]
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["live_llm_inference_run"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    relative = mod.run(
        root=tmp_path,
        artifact_path=Path("relative-exp5365.json"),
        exp5351_path=EXP5351_PATH,
        tests_run=[TEST_COMMAND],
        started_s=1.0,
        now_s=1.5,
        write=True,
    )
    assert (tmp_path / "relative-exp5365.json").is_file()
    assert relative["grammar_budget_protocol_ready"] is True


def test_req_verify_5365_artifact_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5365: artifact validation catches malformed gate fields."""

    artifact = mod.run(
        root=REPO,
        artifact_path=tmp_path / "unused.json",
        exp5351_path=EXP5351_PATH,
        tests_run=[TEST_COMMAND],
        started_s=1.0,
        now_s=2.0,
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("status"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("status", "running"), a)[1], "status must be complete or blocked"),
        (
            lambda a: (a.__setitem__("grammar_budget_protocol_ready", "yes"), a)[1],
            "grammar_budget_protocol_ready must be boolean",
        ),
        (
            lambda a: (a.__setitem__("schema_reachability_cases", "4"), a)[1],
            "schema_reachability_cases must be integer",
        ),
        (
            lambda a: (a.__setitem__("required_field_reachability_rate", 1.2), a)[1],
            "required_field_reachability_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("completion_slack_min_tokens", "900"), a)[1],
            "completion_slack_min_tokens must be integer",
        ),
        (
            lambda a: (a.__setitem__("truncation_failure_count", -1), a)[1],
            "truncation_failure_count must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("schema_failure_count", "2"), a)[1],
            "schema_failure_count must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("tool_action_token_reachability_rate", -0.1), a)[1],
            "tool_action_token_reachability_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", "1.0"), a)[1],
            "methodology_duration_s must be numeric",
        ),
        (lambda a: (a.__setitem__("tests_run", "pytest"), a)[1], "tests_run must be list"),
        (
            lambda a: (a.__setitem__("active_roadmap_modified", True), a)[1],
            "active_roadmap_modified must be false",
        ),
        (
            lambda a: (a.__setitem__("conductor_modified", True), a)[1],
            "conductor_modified must be false",
        ),
        (
            lambda a: (a.__setitem__("honest_verdict", "ready"), a)[1],
            "honest_verdict must start with complete: or blocked_",
        ),
        (
            lambda a: (
                a.__setitem__("grammar_budget_protocol_ready", True),
                a.__setitem__("completion_slack_min_tokens", -1),
                a,
            )[2],
            "ready preflight requires non-negative completion slack",
        ),
        (
            lambda a: (
                a.__setitem__("status", "complete"),
                a.__setitem__("grammar_budget_protocol_ready", False),
                a,
            )[2],
            "complete status requires grammar_budget_protocol_ready",
        ),
        (
            lambda a: (a["field_provenance"].pop("status"), a)[1],
            "field_provenance must cover required fields",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined

    with pytest.raises(AssertionError, match="complete status"):
        bad = clone()
        bad["grammar_budget_protocol_ready"] = False
        mod.validate_artifact(bad)
