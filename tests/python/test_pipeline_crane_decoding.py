"""Tests for Exp 1678 CRANE interleaved decoding.

Spec: REQ-PIPELINE-1678, SCENARIO-PIPELINE-1678.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import crane_decoding as mod


class RecordingBackend:
    """Small deterministic backend that records CRANE generation requests."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[mod.CRANEGenerationRequest] = []

    def generate(self, request: mod.CRANEGenerationRequest) -> str:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("backend exhausted")
        return self.outputs.pop(0)


def test_req_pipeline_1678_state_machine_alternates_free_then_constrained() -> None:
    """REQ-PIPELINE-1678: CRANE toggles from unconstrained reasoning to grammar enforcement."""

    backend = RecordingBackend(
        [
            "Think freely: six cases each week over three weeks means 6 * 3.",
            '{"answer": "18", "reasoning_summary": "six cases weekly for three weeks"}',
        ]
    )
    decoder = mod.CRANEDecoder(backend=backend)

    result = decoder.decode("A lab reviews 6 cases each week for 3 weeks. How many cases?")

    assert result.parseable is True
    assert result.structured == {
        "answer": "18",
        "reasoning_summary": "six cases weekly for three weeks",
    }
    assert result.phase_order == [
        mod.CRANEPhase.FREE_TEXT.value,
        mod.CRANEPhase.CONSTRAINED.value,
    ]
    assert [request.constraints_enforced for request in backend.requests] == [False, True]
    assert backend.requests[1].prior_reasoning.startswith("Think freely")
    assert backend.requests[1].grammar is decoder.grammar


def test_req_pipeline_1678_rejects_malformed_constrained_output_and_retries() -> None:
    """REQ-PIPELINE-1678: malformed structured output is rejected before a retry cycle."""

    backend = RecordingBackend(
        [
            "First free pass: track weekly quantity.",
            "answer: eighteen",
            "Repair free pass: preserve 6 * 3 = 18.",
            '{"answer": "18", "reasoning_summary": "weekly quantity multiplied by three"}',
        ]
    )
    decoder = mod.CRANEDecoder(
        backend=backend,
        config=mod.CRANEDecodingConfig(max_cycles=2),
    )

    result = decoder.decode("A lab reviews 6 cases each week for 3 weeks. How many cases?")

    assert result.parseable is True
    assert result.phase_order == [
        mod.CRANEPhase.FREE_TEXT.value,
        mod.CRANEPhase.CONSTRAINED.value,
        mod.CRANEPhase.FREE_TEXT.value,
        mod.CRANEPhase.CONSTRAINED.value,
    ]
    constrained = [segment for segment in result.trace if segment.phase == mod.CRANEPhase.CONSTRAINED]
    assert constrained[0].parse_error == "expected JSON object"
    assert constrained[0].parsed is None
    assert constrained[1].parsed == result.structured


def test_req_pipeline_1678_grammar_rejects_missing_keys_and_non_string_values() -> None:
    """REQ-PIPELINE-1678: strict phase parsing enforces the structured grammar."""

    grammar = mod.StructuredJSONGrammar()

    with pytest.raises(mod.CRANEParseError, match="missing required keys"):
        grammar.parse('{"answer": "18"}')

    with pytest.raises(mod.CRANEParseError, match="must be strings"):
        grammar.parse('{"answer": 18, "reasoning_summary": "math"}')

    with pytest.raises(mod.CRANEParseError, match="expected JSON object"):
        grammar.parse('["18", "math"]')

    assert grammar.instruction().startswith("Return exactly one JSON object")


def test_scenario_pipeline_1678_evaluation_improves_gemma_proxy_coherence() -> None:
    """SCENARIO-PIPELINE-1678: CRANE improves coherence versus strict grammar-only."""

    artifact = mod.build_artifact(tests_run=["focused pytest"])

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["model_specs"] == [mod.GEMMA_4_26B_A4B_GGUF]
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["parse_rate"] == pytest.approx(1.0)
    assert artifact["strict_baseline_parse_rate"] == pytest.approx(1.0)
    assert artifact["reasoning_quality_delta"] > 0.0
    assert artifact["crane_mean_reasoning_quality"] > artifact["strict_mean_reasoning_quality"]
    assert artifact["spec_traces"] == mod.SPEC_TRACES
    assert artifact["tests_run"] == ["focused pytest"]


def test_req_pipeline_1678_run_experiment_writes_json_deliverable(tmp_path: Path) -> None:
    """REQ-PIPELINE-1678: run_experiment writes the required Exp 1678 JSON."""

    output_path = tmp_path / "results" / "experiment_1678_crane.json"

    artifact = mod.run_experiment(output_path=output_path, tests_run=["focused"])
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert artifact["artifact_path"] == str(output_path)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert "reasoning_quality_delta" in persisted
    assert "parse_rate" in persisted


def test_req_pipeline_1678_validation_and_empty_cases_fail_closed() -> None:
    """REQ-PIPELINE-1678: validation catches schema drift and empty evaluations."""

    artifact = mod.build_artifact()

    missing = dict(artifact)
    del missing["parse_rate"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="parse_rate"):
        mod.validate_artifact(dict(artifact, parse_rate=0.5))

    with pytest.raises(AssertionError, match="reasoning_quality_delta"):
        mod.validate_artifact(dict(artifact, reasoning_quality_delta=-0.1))

    with pytest.raises(ValueError, match="at least one"):
        mod.evaluate_crane_decoding(cases=[])

    one_case = [
        mod.ReasoningCase(
            case_id="partial",
            prompt="A lab reviews 6 cases each week for 3 weeks. How many cases?",
            expected_answer="18",
            semantic_keywords=("weekly",),
        )
    ]
    partial_backend = RecordingBackend(
        [
            "free reasoning still runs",
            "not json",
            '{"answer": "18", "reasoning_summary": "weekly"}',
        ]
    )
    partial = mod.build_artifact(backend=partial_backend, cases=one_case)
    assert partial["status"] == "partial"
    assert partial["parse_rate"] == pytest.approx(0.0)
    assert partial["honest_verdict"].startswith("partial:")


def test_req_pipeline_1678_strict_baseline_parse_failure_is_recorded() -> None:
    """REQ-PIPELINE-1678: strict grammar-only baseline also fails closed on bad JSON."""

    backend = RecordingBackend(["not json"])
    decoder = mod.CRANEDecoder(backend=backend)

    result = decoder.strict_baseline("Return a structured answer.")

    assert result.parseable is False
    assert result.structured is None
    assert result.phase_order == [mod.CRANEPhase.STRICT_BASELINE.value]
    assert result.trace[0].parse_error == "expected JSON object"
    assert mod.semantic_coherence_score(None, mod.default_reasoning_cases()[0]) == 0.0
