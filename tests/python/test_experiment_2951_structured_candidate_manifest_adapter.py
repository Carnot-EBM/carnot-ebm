"""Tests for Exp 2951 structured candidate manifest adapter.

Spec: REQ-CODE-2951, SCENARIO-CODE-2951.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import structured_candidate_manifest_adapter as exp


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 14.25,
        tests_run=("focused-pytest",),
    )


def _backend(name: str, available: bool, detail: str) -> dict[str, Any]:
    return {"backend_name": name, "available": available, "detail": detail}


def test_req_code_2951_spec_anchor_exists() -> None:
    """REQ-CODE-2951, SCENARIO-CODE-2951: Exp 2951 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2951" in spec
    assert "SCENARIO-CODE-2951" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_wiring"' in spec


def test_scenario_code_2951_builds_ready_artifact_with_three_schema_valid_fixtures(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2951: valid, syntax-failed, and hallucinated rows all validate."""

    backends = [
        _backend("jsonschema", False, "not installed"),
        _backend("llguidance", False, "not installed"),
        _backend("llama_cpp_grammar", True, "LlamaGrammar.from_json_schema available"),
    ]
    artifact = exp.write_artifact(_config(tmp_path), local_backends=backends)
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["structured_decode_manifest_ready"] is True
    assert artifact["schema_version"] == exp.SCHEMA_VERSION
    assert artifact["schema_fields"] == list(exp.CANDIDATE_SCHEMA_FIELDS)
    assert artifact["validation_fixture_count"] == 3
    assert artifact["validation_fixture_passed"] is True
    assert artifact["llguidance_available"] is False
    assert artifact["llama_cpp_grammar_available"] is True
    assert artifact["preferred_structured_output_backend"] == "llama_cpp_grammar"
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert [model["hf_id"] for model in artifact["model_specs_for_downstream_live_use"]] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]

    statuses = {row["fixture_id"]: row for row in artifact["validation_fixture_results"]}
    assert statuses["valid_candidate"]["schema_valid"] is True
    assert statuses["syntax_failure"]["parser_status"] == "syntax_error"
    assert statuses["unsupported_import_api_hallucination"]["failure_taxonomy"] == [
        "unsupported_import",
        "unsupported_api_hallucination",
    ]


def test_req_code_2951_schema_validation_rejects_malformed_manifest_rows() -> None:
    """REQ-CODE-2951: deterministic fallback rejects malformed manifest structure."""

    adapter = exp.StructuredCandidateManifestAdapter(
        local_backends=[_backend("llguidance", False, "disabled")]
    )
    valid = exp.synthetic_candidate_records()[0]
    broken = {
        **valid,
        "verifier_score": 1.2,
        "parser_status": "maybe",
        "provenance_checksums": {"raw_completion_sha256": "not-a-sha"},
        "unexpected": True,
    }

    assert adapter.validate_record(valid).ok is True
    result = adapter.validate_record(broken)
    assert result.ok is False
    assert "$ unexpected field unexpected" in result.errors
    assert "$.parser_status expected one of ['parsed', 'syntax_error']" in result.errors
    assert "$.verifier_score expected <= 1.0" in result.errors
    assert "$.provenance_checksums missing required field repaired_code_sha256" in result.errors
    assert "$.provenance_checksums.raw_completion_sha256 expected 64 lowercase hex characters" in (
        result.errors
    )

    type_errors = adapter.validate_record(
        {
            **valid,
            "task_id": "",
            "prompt_id": 123,
            "failure_taxonomy": "syntax_error",
            "provenance_checksums": "not-an-object",
            "verifier_score": "high",
        }
    )
    assert "$.task_id expected length >= 1" in type_errors.errors
    assert "$.prompt_id expected string" in type_errors.errors
    assert "$.failure_taxonomy expected array" in type_errors.errors
    assert "$.provenance_checksums expected object" in type_errors.errors
    assert "$.verifier_score expected number" in type_errors.errors

    empty_taxonomy = adapter.validate_record({**valid, "failure_taxonomy": []})
    negative_score = adapter.validate_record({**valid, "verifier_score": -0.1})
    assert "$.failure_taxonomy expected at least 1 item(s)" in empty_taxonomy.errors
    assert "$.verifier_score expected >= 0.0" in negative_score.errors
    assert exp._validate_schema("ignored", {"type": "boolean"}, "$") == []


def test_req_code_2951_backend_probe_covers_available_and_missing_paths() -> None:
    """REQ-CODE-2951: local backend probing records JSON, llguidance, and llama.cpp status."""

    class FakeMatcher:
        @staticmethod
        def grammar_from_json_schema(_schema: dict[str, Any]) -> str:
            return "grammar"

    class FakeLLGuidance:
        LLMatcher = FakeMatcher

    class FakeGrammar:
        @staticmethod
        def from_json_schema(_schema: str) -> object:
            return object()

    class FakeLlamaCpp:
        LlamaGrammar = FakeGrammar

    def importer(name: str) -> object:
        if name == "jsonschema":
            return object()
        if name == "llguidance":
            return FakeLLGuidance
        if name == "llama_cpp":
            return FakeLlamaCpp
        raise ImportError(name)

    available = exp.probe_local_backends(importer=importer)
    missing = exp.probe_local_backends(importer=lambda name: (_ for _ in ()).throw(ImportError(name)))

    assert {row["backend_name"]: row["available"] for row in available} == {
        "jsonschema": True,
        "llguidance": True,
        "llama_cpp_grammar": True,
    }
    assert exp.preferred_backend(available) == "llguidance"
    assert {row["backend_name"]: row["available"] for row in missing} == {
        "jsonschema": False,
        "llguidance": False,
        "llama_cpp_grammar": False,
    }
    assert exp.preferred_backend(missing) == "deterministic_schema_validation"
