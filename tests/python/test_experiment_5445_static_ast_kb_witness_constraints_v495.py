"""Tests for Exp5445 deterministic AST/KB witness constraints.

Spec refs: REQ-CODE-5445, SCENARIO-CODE-5445.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5445_static_ast_kb_witness_constraints_v495 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/code-verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5445_static_ast_kb_witness_constraints_v495.py -q"
)


def _row(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["row_id"] == row_id)


def test_req_code_5445_spec_declares_ast_kb_witness_contract() -> None:
    """REQ-CODE-5445: OpenSpec anchors all required witness artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-CODE-5445") : spec.index("### REQ-CODE-2946")]

    for marker in (
        "REQ-CODE-5445",
        "SCENARIO-CODE-5445",
        str(mod.RESULT_RELATIVE_PATH),
        "valid API calls",
        "nonexistent methods",
        "wrong-module aliases",
        "bare calls with missing imports",
        "argument-intent mismatches",
        "fixture_count",
        "api_family_counts",
        "ast_parse_success_rate",
        "kb_source_paths",
        "witness_field_names",
        "nonexistent_call_reject_rate",
        "valid_call_accept_rate",
        "unsafe_false_accepts",
        "row_provenance_checksum",
        'inference_substrate="deterministic_ast_kb_verifier_no_llm"',
    ):
        assert marker in section


def test_req_code_5445_alias_resolution_accepts_valid_and_rejects_wrong_module() -> None:
    """REQ-CODE-5445: alias maps drive fully qualified AST call-site witnesses."""

    rows = mod.evaluate_fixture_rows(kb=mod.ApiKnowledgeBase.from_fallback_metadata())
    valid = _row(rows, "fixture.valid_json_alias")
    wrong_module = _row(rows, "fixture.wrong_module_alias")

    assert valid["ast_parse_ok"] is True
    assert valid["alias_map"] == {"js": "json"}
    assert valid["fully_qualified_call_sites"][0]["fqn"] == "json.loads"
    assert valid["kb_lookup_results"][0]["exists"] is True
    assert valid["accepted"] is True
    assert valid["reject_reasons"] == []

    assert wrong_module["alias_map"] == {"json": "statistics"}
    assert wrong_module["fully_qualified_call_sites"][0]["fqn"] == "statistics.loads"
    assert wrong_module["kb_lookup_results"][0]["exists"] is False
    assert wrong_module["accepted"] is False
    assert "kb_missing_call:statistics.loads" in wrong_module["reject_reasons"]


def test_scenario_code_5445_rejects_nonexistent_imported_and_bare_calls() -> None:
    """SCENARIO-CODE-5445: nonexistent calls and unresolved bare calls are rejected."""

    rows = mod.evaluate_fixture_rows(kb=mod.ApiKnowledgeBase.from_fallback_metadata())
    nonexistent = _row(rows, "fixture.nonexistent_json_method")
    imported_missing = _row(rows, "fixture.imported_symbol_missing")
    bare_missing = _row(rows, "fixture.missing_bare_import")

    assert nonexistent["accepted"] is False
    assert nonexistent["fully_qualified_call_sites"][0]["fqn"] == "json.parse"
    assert nonexistent["kb_lookup_results"][0]["exists"] is False
    assert nonexistent["outcome"] == "reject"

    assert imported_missing["imported_symbol_checks"] == [
        {
            "module": "json",
            "symbol": "parse",
            "local_name": "parse",
            "fully_qualified_name": "json.parse",
            "exists": False,
            "kb_source": "fallback_metadata",
        }
    ]
    assert imported_missing["accepted"] is False
    assert "imported_symbol_missing:json.parse" in imported_missing["reject_reasons"]

    assert bare_missing["fully_qualified_call_sites"][0]["fqn"] == "<unresolved>.loads"
    assert bare_missing["kb_lookup_results"][0]["status"] == "unresolved_bare_call"
    assert bare_missing["accepted"] is False
    assert "missing_import_for_bare_call:loads" in bare_missing["reject_reasons"]


def test_scenario_code_5445_semantic_intent_mismatch_rejects_existing_api() -> None:
    """SCENARIO-CODE-5445: intent checks reject existing APIs used for the wrong purpose."""

    rows = mod.evaluate_fixture_rows(kb=mod.ApiKnowledgeBase.from_fallback_metadata())
    mismatch = _row(rows, "fixture.argument_intent_mismatch")

    assert mismatch["fully_qualified_call_sites"][0]["fqn"] == "json.dumps"
    assert mismatch["kb_lookup_results"][0]["exists"] is True
    assert mismatch["semantic_intent"]["intent"] == "parse_json_to_object"
    assert mismatch["semantic_intent"]["expected_call_fqns"] == ["json.loads"]
    assert mismatch["accepted"] is False
    assert "intent_mismatch:parse_json_to_object" in mismatch["reject_reasons"]


def test_req_code_5445_kb_falls_back_when_safe_import_is_unavailable() -> None:
    """REQ-CODE-5445: KB provenance records deterministic fallback metadata."""

    def unavailable(_module_name: str) -> Any:
        raise ImportError("forced unavailable")

    kb = mod.ApiKnowledgeBase.build(["json"], importer=unavailable)
    row = mod.evaluate_fixture(
        mod.fixture_by_id("fixture.valid_json_alias"),
        kb=kb,
    )

    assert kb.source_paths == ["fallback_metadata:json"]
    assert kb.has_module("json") is True
    assert kb.lookup("missing.module")["status"] == "unknown_module"
    assert row["accepted"] is True
    assert row["kb_lookup_results"][0]["kb_source"] == "fallback_metadata"


def test_req_code_5445_defensive_witness_paths_are_deterministic() -> None:
    """REQ-CODE-5445: parse and complex-call fallbacks emit stable rejects."""

    kb = mod.ApiKnowledgeBase.from_fallback_metadata()
    syntax_row = mod.evaluate_fixture(
        mod.AstKbFixture(
            row_id="fixture.syntax_probe",
            fixture_family="syntax_probe",
            api_family="no_api",
            source="def bad(:\n",
            expected_outcome="reject",
            intent="parse_probe",
            expected_call_fqns=(),
            metric_tags=("invalid_row",),
        ),
        kb=kb,
    )
    complex_row = mod.evaluate_fixture(
        mod.AstKbFixture(
            row_id="fixture.complex_call_probe",
            fixture_family="complex_call_probe",
            api_family="no_api",
            source="factory()[0]()\n",
            expected_outcome="reject",
            intent="complex_call_probe",
            expected_call_fqns=(),
            metric_tags=("invalid_row",),
        ),
        kb=kb,
    )

    assert syntax_row["ast_parse_ok"] is False
    assert syntax_row["outcome"] == "reject"
    assert syntax_row["reject_reasons"][0].startswith("ast_parse_error:")
    assert complex_row["fully_qualified_call_sites"][0]["fqn"] == "<unresolved>.Subscript"
    assert complex_row["outcome"] == "reject"


def test_scenario_code_5445_artifact_metrics_and_provenance_checksums(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-5445: artifact metrics derive from inspectable row witnesses."""

    output_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=output_path,
        tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}, TEST_COMMAND],
        write=True,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert artifact["fixture_count"] == len(mod.FIXTURES) == len(artifact["witness_rows"])
    assert artifact["api_family_counts"]["json"] >= 4
    assert artifact["ast_parse_success_rate"] == pytest.approx(1.0)
    assert artifact["nonexistent_call_reject_rate"] == pytest.approx(1.0)
    assert artifact["valid_call_accept_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["ast_kb_witness_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.WITNESS_FIELD_NAMES) <= set(artifact["witness_rows"][0])
    assert artifact["witness_field_names"] == list(mod.WITNESS_FIELD_NAMES)
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["witness_rows"]
    )
    for row in artifact["witness_rows"]:
        assert row["witness_checksum"] == mod.row_witness_checksum(row)

    tampered = deepcopy(artifact)
    tampered["witness_rows"][0]["accepted"] = not tampered["witness_rows"][0]["accepted"]
    assert "row_provenance_checksum mismatch" in mod.artifact_schema_errors(tampered)
    with pytest.raises(ValueError, match="row_provenance_checksum mismatch"):
        mod.validate_artifact(tampered)
