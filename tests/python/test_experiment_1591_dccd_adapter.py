"""Tests for Exp 1591 reusable DCCD structured verdict adapter.

Spec: REQ-VERIFY-1591, SCENARIO-VERIFY-1591.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.verdict_record import VerdictRecord
from carnot.verifiers import dccd_adapter as mod


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["case_id", "answer", "verdict", "confidence", "evidence"],
        "properties": {
            "case_id": {"type": "string"},
            "answer": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["sat", "unsat"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
    }


def _target() -> dict[str, Any]:
    return {
        "case_id": "case-1591-a",
        "answer": 4,
        "verdict": "sat",
        "confidence": 0.91,
        "evidence": ["2 + 2 = 4"],
    }


def _semantic_paths() -> dict[str, Any]:
    return {"case_id": "case-1591-a", "answer": 4, "verdict": "sat"}


class _FakeLLMatcher:
    calls: list[dict[str, Any]] = []

    @staticmethod
    def grammar_from_json_schema(
        schema: dict[str, Any],
        defaults: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        _FakeLLMatcher.calls.append(
            {"schema": schema, "defaults": defaults, "overrides": overrides}
        )
        return "LLG_GRAMMAR::structured_verdict"

    @staticmethod
    def validate_grammar(
        grammar: str,
        tokenizer: object | None = None,
        limits: object | None = None,
    ) -> str:
        _FakeLLMatcher.calls.append({"grammar": grammar, "tokenizer": tokenizer, "limits": limits})
        return ""


class _FakeLLGuidance:
    LLMatcher = _FakeLLMatcher

    @staticmethod
    def get_version() -> str:
        return "llguidance-test"


def test_req_verify_1591_compiles_llguidance_grammar_metadata() -> None:
    """REQ-VERIFY-1591: injected llguidance bindings compile JSON-schema grammar metadata."""

    _FakeLLMatcher.calls = []
    adapter = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload=_target(),
        llguidance_module=_FakeLLGuidance,
    )

    diagnostics = adapter.backend_diagnostics()
    metadata = adapter.build_generation_metadata("Solve 2 + 2.")

    assert diagnostics["llguidance_backend_available"] is True
    assert diagnostics["llguidance_version"] == "llguidance-test"
    assert diagnostics["grammar_compiled"] is True
    assert diagnostics["grammar_error"] is None
    assert metadata["grammar"] == "LLG_GRAMMAR::structured_verdict"
    assert metadata["json_schema"]["required"] == _schema()["required"]
    assert "Return JSON only" in metadata["prompt"]
    assert _FakeLLMatcher.calls[0]["overrides"] == {"whitespace_flexible": False}


def test_req_verify_1591_fallback_backend_is_available_without_llguidance() -> None:
    """REQ-VERIFY-1591: deterministic fallback remains available when llguidance is absent."""

    adapter = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload=_target(),
        probe_llguidance=False,
    )

    diagnostics = adapter.backend_diagnostics()

    assert diagnostics["llguidance_backend_available"] is False
    assert diagnostics["fallback_backend_available"] is True
    assert diagnostics["grammar_compiled"] is False
    assert diagnostics["backend_name"] == "post_decode_fallback"
    assert diagnostics["grammar_error"] == "llguidance probing disabled"


def test_req_verify_1591_records_schema_semantic_and_false_accept_failures() -> None:
    """REQ-VERIFY-1591: VerdictRecord extras distinguish schema errors and false accepts."""

    adapter = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload=_target(),
        probe_llguidance=False,
    )

    passed = adapter.evaluate(json.dumps(_target()), mode="dccd")
    invalid = adapter.evaluate(
        '{"case_id":"case-1591-a","answer":"4","verdict":"sat","confidence":1.2}',
        mode="unconstrained_draft",
    )
    false_accept = adapter.evaluate(
        json.dumps({**_target(), "answer": 5}),
        mode="schema_valid_semantic_false_accept",
    )

    assert isinstance(passed, VerdictRecord)
    assert passed.verdict == "pass"
    assert passed.extras["strict_schema_valid"] is True
    assert passed.extras["semantic_correct"] is True
    assert passed.extras["false_accept"] is False

    assert invalid.verdict == "fail"
    assert invalid.extras["parsed_payload"]["answer"] == "4"
    assert "$.answer expected integer" in invalid.extras["schema_errors"]
    assert "$.confidence expected <= 1.0" in invalid.extras["schema_errors"]
    assert invalid.extras["false_accept"] is False

    assert false_accept.verdict == "fail"
    assert false_accept.extras["strict_schema_valid"] is True
    assert false_accept.extras["semantic_errors"] == ["$.answer expected 4 observed 5"]
    assert false_accept.extras["false_accept"] is True
    assert false_accept.rationale == "semantic_mismatch_false_accept"


def test_req_verify_1591_dccd_projection_repairs_draft_shape_without_exec() -> None:
    """REQ-VERIFY-1591: DCCD projection uses schema/target data, not generated code."""

    adapter = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload=_target(),
        probe_llguidance=False,
    )
    draft = (
        'before __import__("os").system("bad") '
        '{"case_id":"case-1591-a","answer":"wrong","verdict":"sat","extra":true}'
    )

    projected = adapter.project_dccd_payload(draft)
    record = adapter.evaluate_projected(draft)

    assert projected == _target()
    assert record.verdict == "pass"
    assert record.extras["mode"] == "dccd"
    assert mod.compiler_uses_arbitrary_code_execution() is False


def test_req_verify_1591_schema_edges_and_backend_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1591: schema validator and llguidance error branches fail closed."""

    class MissingGrammarModule:
        LLMatcher = object()

    class RaisingMatcher:
        @staticmethod
        def grammar_from_json_schema(
            schema: dict[str, Any],
            defaults: dict[str, Any] | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> str:
            raise RuntimeError("compile failed")

    class BrokenMatcher:
        @staticmethod
        def grammar_from_json_schema(
            schema: dict[str, Any],
            defaults: dict[str, Any] | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> str:
            return "BROKEN"

        @staticmethod
        def validate_grammar(
            grammar: str,
            tokenizer: object | None = None,
            limits: object | None = None,
        ) -> str:
            return "bad grammar"

    broken_module = type("BrokenModule", (), {"LLMatcher": BrokenMatcher})
    raising_module = type("RaisingModule", (), {"LLMatcher": RaisingMatcher})
    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert (
        mod.DCCDStructuredVerdictAdapter(
            schema=_schema(),
            semantic_paths=_semantic_paths(),
            target_payload=_target(),
        ).backend_diagnostics()["grammar_error"]
        == "llguidance not installed"
    )
    adapter = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload=_target(),
        llguidance_module=broken_module,
    )

    assert adapter.backend_diagnostics()["llguidance_backend_available"] is False
    assert adapter.backend_diagnostics()["grammar_error"] == "bad grammar"
    assert (
        mod.DCCDStructuredVerdictAdapter(
            schema=_schema(),
            semantic_paths=_semantic_paths(),
            target_payload=_target(),
            llguidance_module=MissingGrammarModule,
        ).backend_diagnostics()["grammar_error"]
        == "llguidance LLMatcher.grammar_from_json_schema unavailable"
    )
    assert (
        mod.DCCDStructuredVerdictAdapter(
            schema=_schema(),
            semantic_paths=_semantic_paths(),
            target_payload=_target(),
            llguidance_module=raising_module,
        ).backend_diagnostics()["grammar_error"]
        == "RuntimeError: compile failed"
    )
    assert mod.extract_json_object("no json") is None
    assert mod.extract_json_object("") is None
    assert mod.extract_json_object("{bad") is None
    assert mod.extract_json_object('bad {"a": 1} {"longer": true, "b": 2}') == {
        "longer": True,
        "b": 2,
    }
    assert mod.validate_json_schema({"type": "array", "minItems": 2}, ["one"]) == [
        "$ expected at least 2 items"
    ]
    assert mod.validate_json_schema({"type": "boolean"}, 1) == ["$ expected boolean"]
    assert mod.validate_json_schema({"type": "number", "minimum": 2.0}, 1.5) == [
        "$ expected >= 2.0"
    ]
    assert mod.validate_json_schema({"type": "string", "enum": ["sat"]}, "unsat") == [
        "$ expected one of ['sat']"
    ]
    assert mod._validate_object({}, "not an object", "$") == []
    assert mod._matches_json_type(object(), "unknown") is True
    assert mod._path_value({"a": {}}, "a.missing") is None
    assert mod._claims_accept({"final_certificate": {"state": "SAT"}}) is True
    assert mod._claims_accept({"validator_metadata": {"expected_semantic_result": "SAT"}}) is True
    assert mod._claims_accept({"route": "reject"}) is False

    targetless = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        probe_llguidance=False,
    )
    assert targetless.project_dccd_payload(json.dumps(_target())) == _target()
    with pytest.raises(mod.DCCDAdapterError, match="target_payload required"):
        targetless.project_dccd_payload('{"not":"valid"}')

    invalid_target = mod.DCCDStructuredVerdictAdapter(
        schema=_schema(),
        semantic_paths=_semantic_paths(),
        target_payload={**_target(), "answer": "bad"},
        probe_llguidance=False,
    )
    with pytest.raises(mod.DCCDAdapterError, match="target_payload invalid"):
        invalid_target.project_dccd_payload("{}")


def test_scenario_verify_1591_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1591: artifact records complete adapter metrics and diagnostics."""

    output_path = tmp_path / "experiment_1591_dccd_adapter.json"
    artifact = mod.write_experiment_artifact(
        output_path=output_path,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1591_dccd_adapter.py -q"],
        llguidance_module=_FakeLLGuidance,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == "experiment_1591_dccd_adapter"
    assert artifact["adapter_module"] == "carnot.verifiers.dccd_adapter"
    assert artifact["llguidance_backend_available"] is True
    assert artifact["fallback_backend_available"] is True
    assert artifact["strict_schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["semantic_correctness_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_count"] == 0
    assert artifact["detected_false_accept_rejections"] == 1
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert artifact["honest_verdict"].startswith("complete:")
