"""Tests for Fast-Slow Training verify-repair integration.

Spec: REQ-FST-2240, SCENARIO-FST-2240
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline
from carnot.training.fast_slow import FastSlowTrainer, FastWeights, SlowWeights


class _FakeParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModule:
    def __init__(self) -> None:
        self.params = [_FakeParam(), _FakeParam()]
        self.eval_called = False

    def parameters(self) -> list[_FakeParam]:
        return self.params

    def eval(self) -> None:
        self.eval_called = True


def _failed_verification() -> VerificationResult:
    violation = ConstraintResult(
        constraint_type="arithmetic",
        description="10 + 5 = 16 is incorrect",
        metadata={"satisfied": False, "correct_result": 15},
    )
    return VerificationResult(
        verified=False,
        constraints=[violation],
        energy=1.0,
        violations=[violation],
        certificate={"n_constraints": 1, "n_violations": 1},
    )


def test_slow_weights_freeze_requires_grad_parameters() -> None:
    """REQ-FST-2240: slow weights freeze base LLM and verifier ensemble params."""
    base_llm = _FakeModule()
    verifier = _FakeModule()

    slow = SlowWeights(base_llm=base_llm, verifier_ensemble=[verifier])

    assert slow.assert_frozen() is True
    assert all(param.requires_grad is False for param in base_llm.params)
    assert all(param.requires_grad is False for param in verifier.params)
    assert base_llm.eval_called is True
    assert verifier.eval_called is True

    base_llm.params[0].requires_grad = True
    with pytest.raises(AssertionError, match="not frozen"):
        slow.assert_frozen()


def test_fast_weights_update_builds_terminal_context_prefix() -> None:
    """REQ-FST-2240: fast weights summarize verifier output for the next prompt."""
    fast = FastWeights()

    summary = fast.update_from_verification(_failed_verification(), iteration=1)

    assert summary.violation_count == 1
    assert fast.context_prefix.startswith("FST verifier-output summary:")
    assert "[arithmetic] 10 + 5 = 16 is incorrect" in fast.context_prefix
    assert "correct_result=15" in fast.context_prefix


def test_fast_slow_trainer_prepends_context_before_base_prompt() -> None:
    """SCENARIO-FST-2240: trainer injects the fast context at prompt start."""
    trainer = FastSlowTrainer(
        slow_weights=SlowWeights(base_llm=_FakeModule(), verifier_ensemble=[]),
        fast_weights=FastWeights(),
    )
    base_prompt = "Question: What is 10 + 5?"

    prompt = trainer.next_repair_prompt(
        verification_result=_failed_verification(),
        base_prompt=base_prompt,
        iteration=1,
    )

    assert prompt.startswith("FST verifier-output summary:")
    assert prompt.endswith(base_prompt)


def test_verify_repair_use_fst_injects_prefix_into_next_prompt_call() -> None:
    """SCENARIO-FST-2240: verify-repair passes FST prefix into repair generation."""
    pipeline = VerifyRepairPipeline(max_repairs=1)
    pipeline._model = MagicMock()
    pipeline._tokenizer = MagicMock()
    prompts: list[str] = []

    def mock_generate(prompt: str, max_new_tokens: int = 256) -> str:
        prompts.append(prompt)
        return "10 + 5 = 15."

    pipeline._generate = mock_generate  # type: ignore[assignment]

    result = pipeline.verify_and_repair(
        question="What is 10 + 5?",
        response="10 + 5 = 16.",
        domain="arithmetic",
        use_fst=True,
    )

    assert result.verified is True
    assert prompts
    assert prompts[0].startswith("FST verifier-output summary:")
    assert "\n\nQuestion: What is 10 + 5?" in prompts[0]
    assert result.history[0].certificate["fst"]["slow_weights_frozen"] is True
