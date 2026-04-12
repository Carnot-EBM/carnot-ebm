"""Tests for `carnot.pipeline.structured_reasoning`.

Spec: REQ-VERIFY-022, REQ-VERIFY-023, REQ-VERIFY-024,
SCENARIO-VERIFY-022, SCENARIO-VERIFY-023, SCENARIO-VERIFY-024
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import carnot.pipeline.structured_reasoning as structured_reasoning
import pytest
from carnot.pipeline.structured_reasoning import (
    StructuredReasoningController,
    build_gemma_structured_reasoning_prompt,
    build_qwen_structured_reasoning_prompt,
    load_monitorability_policy,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "structured_reasoning"


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def policy_with_modes() -> dict[str, object]:
    return {
        "per_task_slice": {
            "live_gsm8k_semantic_failure": {"recommended_mode": "structured_json"},
            "instruction_surface_only": {"recommended_mode": "answer_only_terse"},
            "code_typed_properties": {"recommended_mode": "answer_only_terse"},
        }
    }


def test_prompt_helpers_request_minimal_schema_for_qwen_and_gemma() -> None:
    """REQ-VERIFY-022: Supported models get schema-aware structured prompts."""
    qwen_prompt = build_qwen_structured_reasoning_prompt("What is 27 - 15, then split it?")
    gemma_prompt = build_gemma_structured_reasoning_prompt("What is 27 - 15, then split it?")

    assert '"constraints"' in qwen_prompt
    assert '"steps"' in qwen_prompt
    assert '"claims"' in qwen_prompt
    assert '"final_answer"' in qwen_prompt
    assert "strict JSON only" in qwen_prompt

    assert '"constraints"' in gemma_prompt
    assert '"steps"' in gemma_prompt
    assert '"claims"' in gemma_prompt
    assert '"final_answer"' in gemma_prompt
    assert "single JSON object" in gemma_prompt


def test_policy_helpers_cover_supported_models_and_override_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-024: Policy-gated structured mode uses task slice plus model support."""
    repo = tmp_path / "repo"
    policy_path = repo / "results" / "monitorability_policy_213.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        '{"per_task_slice":{"live_gsm8k_semantic_failure":{"recommended_mode":"structured_json"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    loaded = load_monitorability_policy()
    controller = StructuredReasoningController(policy=loaded)

    assert loaded["per_task_slice"]["live_gsm8k_semantic_failure"]["recommended_mode"] == (
        "structured_json"
    )
    assert (
        controller.should_use_structured_reasoning(
            "live_gsm8k_semantic_failure",
            "Qwen/Qwen3.5-0.8B",
        )
        is True
    )
    assert (
        controller.should_use_structured_reasoning(
            "code_typed_properties",
            "Qwen/Qwen3.5-0.8B",
        )
        is False
    )
    assert (
        controller.should_use_structured_reasoning(
            "live_gsm8k_semantic_failure",
            "unknown/model",
        )
        is False
    )
    assert load_monitorability_policy(repo / "results" / "missing.json") == {}


@pytest.mark.parametrize(
    ("fixture_name", "expected_step_count"),
    [
        ("clean_qwen.json", 3),
        ("clean_gemma.txt", 2),
    ],
)
def test_validate_response_accepts_clean_gold_outputs(
    fixture_name: str, expected_step_count: int
) -> None:
    """SCENARIO-VERIFY-022: Clean structured emissions validate into direct JSON IR."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    ir = controller.validate_response(
        question="Lana has 27 cups, 15 are cinnamon, split the rest evenly.",
        response=read_fixture(fixture_name),
    )

    assert ir.provenance.extraction_method == "direct_json"
    assert len(ir.reasoning_steps) == expected_step_count
    assert ir.final_answer is not None
    assert ir.final_answer.normalized == 2


@pytest.mark.parametrize(
    ("fixture_name", "expected_error"),
    [
        ("malformed_missing_final_answer.json", "missing required top-level key"),
        ("malformed_wrong_steps_type.json", "steps must be a list"),
        ("malformed_not_json.txt", "not valid JSON"),
    ],
)
def test_validate_response_rejects_malformed_gold_outputs(
    fixture_name: str, expected_error: str
) -> None:
    """REQ-VERIFY-023: Validation rejects malformed structured outputs deterministically."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    with pytest.raises(ValueError, match=expected_error):
        controller.validate_response(
            question="What is the mint count?",
            response=read_fixture(fixture_name),
        )


def test_emit_retries_until_a_clean_structured_response_validates() -> None:
    """SCENARIO-VERIFY-023: Malformed structured output triggers a retry before success."""
    controller = StructuredReasoningController(policy=policy_with_modes())
    model = MagicMock()
    tokenizer = MagicMock()

    with patch(
        "carnot.pipeline.structured_reasoning.generate",
        side_effect=[
            read_fixture("malformed_missing_final_answer.json"),
            read_fixture("clean_qwen.json"),
        ],
    ) as mock_generate:
        emission = controller.emit(
            question="Lana has 27 cups, 15 are cinnamon, split the rest evenly.",
            task_slice="live_gsm8k_semantic_failure",
            model_name="Qwen/Qwen3.5-0.8B",
            model=model,
            tokenizer=tokenizer,
            max_attempts=2,
        )

    assert mock_generate.call_count == 2
    assert emission.policy_mode == "structured_json"
    assert emission.response_mode == "structured_json"
    assert emission.fallback_used is False
    assert len(emission.attempts) == 2
    assert emission.attempts[0].valid is False
    assert "missing required top-level key" in str(emission.attempts[0].error)
    assert emission.attempts[1].valid is True
    assert emission.typed_reasoning is not None
    assert emission.typed_reasoning.provenance.extraction_method == "direct_json"
    assert "Issue:" in mock_generate.call_args_list[1].args[2]


def test_emit_falls_back_when_policy_does_not_request_structured_reasoning() -> None:
    """SCENARIO-VERIFY-024: Non-structured task slices skip structured generation entirely."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    with patch("carnot.pipeline.structured_reasoning.generate") as mock_generate:
        emission = controller.emit(
            question="Return two bullets naming the project owners.",
            task_slice="instruction_surface_only",
            model_name="google/gemma-4-E4B-it",
            model=MagicMock(),
            tokenizer=MagicMock(),
            fallback_generate=lambda prompt, max_new_tokens=256: "- Alex\n- Blair",
        )

    assert mock_generate.call_count == 0
    assert emission.policy_mode == "answer_only_terse"
    assert emission.response_mode == "fallback_text"
    assert emission.fallback_used is True
    assert emission.attempts == []
    assert emission.typed_reasoning is not None
    assert emission.typed_reasoning.provenance.extraction_method == "fallback_text"


def test_emit_falls_back_after_exhausting_structured_attempts_and_handles_missing_fallback() -> (
    None
):
    """REQ-VERIFY-023: Exhausted structured retries degrade to fallback or unavailable."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    with patch(
        "carnot.pipeline.structured_reasoning.generate",
        side_effect=[
            read_fixture("malformed_not_json.txt"),
            read_fixture("malformed_wrong_steps_type.json"),
        ],
    ):
        fallback_emission = controller.emit(
            question="What is the mint count?",
            task_slice="live_gsm8k_semantic_failure",
            model_name="google/gemma-4-E4B-it",
            model=MagicMock(),
            tokenizer=MagicMock(),
            fallback_generate=lambda prompt, max_new_tokens=256: "FINAL: 2",
            max_attempts=2,
        )

    assert fallback_emission.response_mode == "fallback_text"
    assert fallback_emission.fallback_used is True
    assert fallback_emission.typed_reasoning is not None
    assert fallback_emission.typed_reasoning.provenance.extraction_method == "fallback_text"
    assert len(fallback_emission.attempts) == 2

    unavailable_emission = controller._fallback(
        question="What is the mint count?",
        task_slice="live_gsm8k_semantic_failure",
        policy_mode="structured_json",
        attempts=[],
        fallback_generate=None,
        max_new_tokens=220,
    )
    assert unavailable_emission.response_mode == "unavailable"
    assert unavailable_emission.typed_reasoning is None


def test_helper_branches_cover_retry_prompt_invalid_attempts_and_model_aliases() -> None:
    """REQ-VERIFY-022: Helper branches cover aliases and invalid configuration."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    assert structured_reasoning._normalize_model_name("Qwen3.5-0.8B") == "qwen"
    assert structured_reasoning._normalize_model_name("Gemma4-E4B-it") == "gemma"
    assert structured_reasoning._normalize_model_name(None) is None
    assert structured_reasoning._normalize_model_name("other") is None
    assert structured_reasoning._extract_json_payload("   ") is None
    assert (
        StructuredReasoningController(policy={"per_task_slice": []}).recommended_mode("x") is None
    )
    assert "Issue: bad json" in controller.build_retry_prompt(
        question="What is the mint count?",
        model_name="Qwen3.5-0.8B",
        error="bad json",
    )

    with pytest.raises(ValueError, match="unsupported model"):
        controller.build_prompt("question", "unsupported/model")

    with pytest.raises(ValueError, match="max_attempts"):
        controller.emit(
            question="question",
            task_slice="live_gsm8k_semantic_failure",
            model_name="Qwen/Qwen3.5-0.8B",
            model=MagicMock(),
            tokenizer=MagicMock(),
            max_attempts=0,
        )


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            '{"constraints": "bad", "steps": [], "claims": [], "final_answer": {}}',
            "constraints must be a list",
        ),
        (
            '{"constraints": [], "steps": [], "claims": "bad", "final_answer": {}}',
            "claims must be a list",
        ),
        (
            '{"constraints": [], "steps": [], "claims": [], "final_answer": "bad"}',
            "final_answer must be an object",
        ),
    ],
)
def test_validate_response_covers_remaining_schema_type_guards(
    response: str, expected_error: str
) -> None:
    """REQ-VERIFY-023: Remaining schema type guards fail deterministically."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    with pytest.raises(ValueError, match=expected_error):
        controller.validate_response(question="q", response=response)


def test_validate_response_covers_direct_json_and_missing_final_answer_guards() -> None:
    """REQ-VERIFY-023: Patched extractor branches still reject invalid structured states."""
    controller = StructuredReasoningController(policy=policy_with_modes())
    response = read_fixture("clean_qwen.json")

    fallback_ir = MagicMock()
    fallback_ir.provenance.extraction_method = "fallback_text"
    fallback_ir.final_answer = object()
    with (
        patch.object(controller._extractor, "extract", return_value=fallback_ir),
        pytest.raises(ValueError, match="direct_json provenance"),
    ):
        controller.validate_response(question="q", response=response)

    missing_final_ir = MagicMock()
    missing_final_ir.provenance.extraction_method = "direct_json"
    missing_final_ir.final_answer = None
    with (
        patch.object(controller._extractor, "extract", return_value=missing_final_ir),
        pytest.raises(ValueError, match="did not produce a final_answer"),
    ):
        controller.validate_response(question="q", response=response)


def test_fallback_covers_typed_reasoning_extraction_failure() -> None:
    """REQ-VERIFY-023: Fallback degrades to raw text when typed extraction raises."""
    controller = StructuredReasoningController(policy=policy_with_modes())

    with patch(
        "carnot.pipeline.structured_reasoning.build_typed_reasoning_ir",
        side_effect=ValueError("boom"),
    ):
        emission = controller._fallback(
            question="What is the mint count?",
            task_slice="live_gsm8k_semantic_failure",
            policy_mode="structured_json",
            attempts=[],
            fallback_generate=lambda prompt, max_new_tokens=256: "plain text",
            max_new_tokens=220,
        )

    assert emission.response_mode == "fallback_text"
    assert emission.typed_reasoning is None


def test_verify_pipeline_exposes_additive_structured_generation_entrypoint() -> None:
    """REQ-VERIFY-024: VerifyRepairPipeline exposes a backward-compatible entry point."""
    pipeline = VerifyRepairPipeline()
    pipeline._model = MagicMock()
    pipeline._tokenizer = MagicMock()

    with patch(
        "carnot.pipeline.verify_repair.StructuredReasoningController.emit",
        return_value="structured-result",
    ) as mock_emit:
        result = pipeline.generate_structured_reasoning(
            question="What is the mint count?",
            task_slice="live_gsm8k_semantic_failure",
            model_name="Qwen/Qwen3.5-0.8B",
        )

    assert result == "structured-result"
    assert mock_emit.call_count == 1


def test_verify_pipeline_structured_generation_requires_a_loaded_model() -> None:
    """REQ-VERIFY-024: The additive pipeline entry point still enforces model loading."""
    pipeline = VerifyRepairPipeline()

    with pytest.raises(RuntimeError, match="No model loaded"):
        pipeline.generate_structured_reasoning(
            question="What is the mint count?",
            task_slice="live_gsm8k_semantic_failure",
        )
