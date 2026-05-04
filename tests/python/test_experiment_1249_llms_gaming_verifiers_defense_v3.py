"""Tests for Exp 1249 LLMs-gaming-verifiers k=5 defense measurement.

Spec: REQ-VERIFY-1249, SCENARIO-VERIFY-1249
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from carnot.eval import llms_gaming_verifiers_defense_v3 as exp


@dataclass
class _FakeK5Result:
    verified: bool
    per_verifier_scores: dict[str, float]


class _CountingZ3:
    """Alternate k=1 blocks so tests pin rate math without real verifiers."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return 1.0 if self.calls % 2 else 0.0


class _CountingK5:
    """Verify only every third sample so k=5 block rate is deterministic."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, question: str, response: str) -> _FakeK5Result:
        assert question
        assert response
        self.calls += 1
        verified = self.calls % 3 == 0
        return _FakeK5Result(
            verified=verified,
            per_verifier_scores={"FakeVerifier": 0.0 if verified else 1.0},
        )


def test_build_gaming_samples_has_required_req1249_shape() -> None:
    """REQ-VERIFY-1249-1/2: exactly 10 diverse gaming samples are defined."""

    samples = exp.build_gaming_samples()
    strategies = {sample.gaming_strategy for sample in samples}

    assert len(samples) == 10
    assert strategies == {
        "mixed_calculation",
        "off_by_one_confident_format",
        "wrong_problem_interpretation",
        "correct_answer_wrong_reasoning",
        "comparison_inversion",
        "percentage_confusion",
        "unit_mismatch",
        "distractor_anchoring",
        "contradictory_answer",
        "masked_arithmetic",
    }
    assert all(sample.question for sample in samples)
    assert all(sample.response for sample in samples)
    assert all(sample.correct_answer for sample in samples)


def test_score_gaming_defense_uses_req1249_blocking_rules() -> None:
    """REQ-VERIFY-1249-3/4: k=1 uses >0.5; k=5 uses not verified."""

    samples = exp.build_gaming_samples()
    result = exp.measure_gaming_defense(samples, z3_verifier=_CountingZ3(), k5_verifier=_CountingK5())

    assert result["n_gaming_samples"] == 10
    assert result["k1_block_rate"] == pytest.approx(0.5)
    assert result["k5_block_rate"] == pytest.approx(0.7)
    assert result["k5_improvement_over_k1"] == pytest.approx(0.2)
    assert result["gaming_defense_measured"] is True
    assert result["honest_verdict"] == "gaming_defense_k5_block_rate_0.70_vs_k1_0.50"


def test_run_experiment_writes_required_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1249: runner writes the required Exp 1249 artifact."""

    output_path = tmp_path / "experiment_1249_llms_gaming_verifiers_defense_v3.json"

    artifact = exp.run_experiment(
        output_path=output_path,
        z3_verifier=_CountingZ3(),
        k5_verifier=_CountingK5(),
    )
    persisted = json.loads(output_path.read_text())

    assert artifact == persisted
    assert persisted["experiment"] == "1249_llms_gaming_verifiers_defense_v3"
    assert persisted["run_date"] == "20260504"
    assert persisted["status"] == "complete"
    assert persisted["n_gaming_samples"] == 10
    assert isinstance(persisted["k1_block_rate"], float)
    assert isinstance(persisted["k5_block_rate"], float)
    assert isinstance(persisted["k5_improvement_over_k1"], float)
    assert persisted["gaming_defense_measured"] is True
    assert persisted["honest_verdict"] == "gaming_defense_k5_block_rate_0.70_vs_k1_0.50"
