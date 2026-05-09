"""Tests for Exp 1642 llguidance structured verdict adapter.

Spec: REQ-VERIFY-1642, SCENARIO-VERIFY-1642.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.verdict_record import VerdictRecord
from scripts import experiment_1642_llguidance as mod


def _record(verdict: str = "pass") -> VerdictRecord:
    energy = 0.0 if verdict == "pass" else 1.0
    return VerdictRecord(
        verdict=verdict,  # type: ignore[arg-type]
        energy=energy,
        calibrated_confidence=0.93,
        producing_tier=3,
        tier_reached=3,
        rationale="constraints_satisfied" if verdict == "pass" else "constraint_violation",
        budget_ms_consumed=2.5,
        repairs_applied=["none"],
        extras={"case_id": "case-1642", "source": "unit-test"},
    )


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
        return "LLG_GRAMMAR::verdict-record"

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


def test_req_verify_1642_compiles_llguidance_and_exposes_llama_cpp_metadata() -> None:
    """REQ-VERIFY-1642: injected llguidance emits grammar plus llama.cpp metadata."""

    _FakeLLMatcher.calls = []
    adapter = mod.LlGuidanceStructuredVerdictAdapter(llguidance_module=_FakeLLGuidance)
    metadata = adapter.build_llama_cpp_metadata("Verify a model answer.")
    diagnostics = adapter.backend_diagnostics()

    assert diagnostics["backend_name"] == "llguidance"
    assert diagnostics["llguidance_backend_available"] is True
    assert diagnostics["grammar_compiled"] is True
    assert diagnostics["llguidance_version"] == "llguidance-test"
    assert diagnostics["grammar_error"] is None
    assert metadata["grammar"] == "LLG_GRAMMAR::verdict-record"
    assert metadata["llama_cpp_kwargs"]["grammar"] == "LLG_GRAMMAR::verdict-record"
    assert metadata["json_schema"]["required"] == list(mod.REQUIRED_VERDICT_FIELDS)
    assert "Verify a model answer." in metadata["prompt"]
    assert _FakeLLMatcher.calls[0]["overrides"] == {"whitespace_flexible": False}


def test_req_verify_1642_roundtrips_verdict_record_and_abstains_on_invalid() -> None:
    """REQ-VERIFY-1642: good JSON round-trips while invalid generated JSON abstains."""

    adapter = mod.LlGuidanceStructuredVerdictAdapter(probe_llguidance=False)

    encoded = adapter.record_to_json(_record("pass"))
    decoded = adapter.parse_generated_verdict(encoded)
    malformed = adapter.parse_generated_verdict(
        '{"verdict":"pass","energy":"low","calibrated_confidence":1.4}'
    )
    no_json = adapter.parse_generated_verdict("model said yes")

    assert decoded.verdict == "pass"
    assert decoded.energy == pytest.approx(0.0)
    assert decoded.extras["case_id"] == "case-1642"
    assert decoded.repairs_applied == ["none"]
    assert malformed.verdict == "abstain"
    assert malformed.rationale == "structured_output_invalid"
    assert "$.energy expected number" in malformed.extras["schema_errors"]
    assert "$.calibrated_confidence expected <= 1.0" in malformed.extras["schema_errors"]
    assert no_json.verdict == "abstain"
    assert no_json.extras["schema_errors"] == ["$ is not a JSON object"]
    assert adapter.backend_diagnostics()["fallback_backend_available"] is True


def test_scenario_verify_1642_artifact_records_adapter_success(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1642: run_experiment writes required adapter_success JSON."""

    output_path = tmp_path / "results" / "experiment_1642_llguidance.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        tests_run=["focused"],
        llguidance_module=_FakeLLGuidance,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1642
    assert artifact["spec_traces"] == ["REQ-VERIFY-1642", "SCENARIO-VERIFY-1642"]
    assert artifact["adapter_success"] is True
    assert artifact["llama_cpp_adapter_ready"] is True
    assert artifact["structured_verdict_roundtrip"] is True
    assert artifact["invalid_output_abstains"] is True
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert artifact["tests_run"] == ["focused"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1642_backend_error_and_validation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1642: optional backend errors fail closed and validation catches drift."""

    class BadRecord:
        @staticmethod
        def to_dict() -> dict[str, Any]:
            return {"verdict": "pass"}

    class MissingMatcher:
        LLMatcher = object()

    class RaisingMatcher:
        @staticmethod
        def grammar_from_json_schema(
            schema: dict[str, Any],
            defaults: dict[str, Any] | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> str:
            raise RuntimeError("compile failed")

    class WarningMatcher:
        @staticmethod
        def grammar_from_json_schema(
            schema: dict[str, Any],
            defaults: dict[str, Any] | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> str:
            return "WARN_GRAMMAR"

        @staticmethod
        def validate_grammar(
            grammar: str,
            tokenizer: object | None = None,
            limits: object | None = None,
        ) -> str:
            return "WARNING: accepted with caution"

    class BrokenMatcher(WarningMatcher):
        @staticmethod
        def validate_grammar(
            grammar: str,
            tokenizer: object | None = None,
            limits: object | None = None,
        ) -> str:
            return "bad grammar"

    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    assert (
        mod.LlGuidanceStructuredVerdictAdapter().backend_diagnostics()["grammar_error"]
        == "llguidance not installed"
    )
    assert (
        mod.LlGuidanceStructuredVerdictAdapter(
            llguidance_module=MissingMatcher
        ).backend_diagnostics()["grammar_error"]
        == "llguidance LLMatcher.grammar_from_json_schema unavailable"
    )
    assert (
        mod.LlGuidanceStructuredVerdictAdapter(
            llguidance_module=type("RaisingModule", (), {"LLMatcher": RaisingMatcher})
        ).backend_diagnostics()["grammar_error"]
        == "RuntimeError: compile failed"
    )
    assert (
        mod.LlGuidanceStructuredVerdictAdapter(
            llguidance_module=type("BrokenModule", (), {"LLMatcher": BrokenMatcher})
        ).backend_diagnostics()["grammar_error"]
        == "bad grammar"
    )
    assert (
        mod.LlGuidanceStructuredVerdictAdapter(
            llguidance_module=type("WarningModule", (), {"LLMatcher": WarningMatcher})
        ).backend_diagnostics()["grammar_compiled"]
        is True
    )

    artifact = mod.build_artifact(probe_llguidance=False)
    missing = dict(artifact)
    del missing["adapter_success"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(AssertionError, match="adapter_success"):
        mod.validate_artifact(dict(artifact, adapter_success=False, status="complete"))
    with pytest.raises(AssertionError, match="invalid_output_abstains"):
        mod.validate_artifact(dict(artifact, invalid_output_abstains=False))
    with pytest.raises(ValueError, match="VerdictRecord payload invalid"):
        mod.LlGuidanceStructuredVerdictAdapter(probe_llguidance=False).record_to_json(
            BadRecord()  # type: ignore[arg-type]
        )
    assert mod.compiler_uses_arbitrary_code_execution() is False
