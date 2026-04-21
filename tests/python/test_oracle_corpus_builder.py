"""Tests for OracleCorpusBuilder — 100% coverage on oracle_corpus_builder.py.

**Why these tests exist (RETRO-066):**
    ORACLE-style step-level labeling is the proposed fix for JEPA v13 ECE=0.207.
    We need high confidence that StepLabel, OracleChain, and OracleCorpusBuilder
    produce correct output before using them to generate JEPA v14 training data.

Spec: REQ-DATA-012, REQ-DATA-013,
      SCENARIO-DATA-019, SCENARIO-DATA-020
"""

from __future__ import annotations

import dataclasses

import pytest

from carnot.pipeline.oracle_corpus_builder import (
    OracleChain,
    OracleCorpusBuilder,
    StepLabel,
)
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(
    question: str = "What is 2+2?",
    response: str = "2 + 2 = 4. The answer is 4.",
    is_correct: bool = True,
    model: str = "test-model",
    question_id: str | None = None,
    inference_mode: str | None = None,
) -> dict:
    """Return a minimal live-pairs dict for testing."""
    row: dict = {
        "question": question,
        "response": response,
        "is_correct": is_correct,
        "model": model,
    }
    if question_id is not None:
        row["question_id"] = question_id
    if inference_mode is not None:
        row["inference_mode"] = inference_mode
    return row


# ---------------------------------------------------------------------------
# StepLabel dataclass
# ---------------------------------------------------------------------------


class TestStepLabel:
    """SCENARIO-DATA-019: StepLabel captures step verification outcome."""

    def test_fields_correct_label(self) -> None:
        # A step with no violation gets label='correct'
        sl = StepLabel(
            step_index=0,
            step_text="2 + 2 = 4",
            violation_detected=False,
            executed_result=4.0,
            stated_result=4.0,
            label="correct",
        )
        assert sl.step_index == 0
        assert sl.label == "correct"
        assert not sl.violation_detected

    def test_fields_violated_label(self) -> None:
        # A step with a violation gets label='violated'
        sl = StepLabel(
            step_index=1,
            step_text="47 + 28 = 65",
            violation_detected=True,
            executed_result=75.0,
            stated_result=65.0,
            label="violated",
        )
        assert sl.violation_detected
        assert sl.label == "violated"

    def test_none_results_allowed(self) -> None:
        # Steps with no arithmetic have None executed_result and stated_result
        sl = StepLabel(
            step_index=2,
            step_text="The answer is four.",
            violation_detected=False,
            executed_result=None,
            stated_result=None,
            label="correct",
        )
        assert sl.executed_result is None
        assert sl.stated_result is None

    def test_is_dataclass(self) -> None:
        sl = StepLabel(0, "text", False, None, None, "correct")
        assert dataclasses.is_dataclass(sl)
        d = dataclasses.asdict(sl)
        assert "step_index" in d
        assert "label" in d


# ---------------------------------------------------------------------------
# OracleChain dataclass
# ---------------------------------------------------------------------------


class TestOracleChain:
    """SCENARIO-DATA-019: OracleChain aggregates step labels correctly."""

    def _make_oracle_chain(self, has_violation: bool = False) -> OracleChain:
        sl = StepLabel(0, "step text", has_violation, None, None,
                       "violated" if has_violation else "correct")
        return OracleChain(
            question_id="q0",
            question="What is X?",
            model_response="step text.",
            is_correct=not has_violation,
            model_id="test-model",
            inference_mode="live_gpu",
            step_labels=[sl],
            has_violation=has_violation,
            n_violated_steps=1 if has_violation else 0,
        )

    def test_no_violation(self) -> None:
        oc = self._make_oracle_chain(has_violation=False)
        assert not oc.has_violation
        assert oc.n_violated_steps == 0
        assert oc.is_correct

    def test_with_violation(self) -> None:
        oc = self._make_oracle_chain(has_violation=True)
        assert oc.has_violation
        assert oc.n_violated_steps == 1

    def test_is_dataclass_serialisable(self) -> None:
        oc = self._make_oracle_chain()
        d = dataclasses.asdict(oc)
        assert "question_id" in d
        assert "step_labels" in d
        assert isinstance(d["step_labels"], list)


# ---------------------------------------------------------------------------
# OracleCorpusBuilder.label_chain
# ---------------------------------------------------------------------------


class TestLabelChain:
    """SCENARIO-DATA-019: label_chain produces correct OracleChain from live pair dict."""

    def setup_method(self) -> None:
        self.verifier = SymCodeVerifier(llm_caller=None)
        self.builder = OracleCorpusBuilder(self.verifier)

    def test_label_correct_response(self) -> None:
        # SCENARIO-DATA-019: a response with correct arithmetic produces no violated steps
        chain = _make_chain(
            response="Janet has 16 eggs per day. 16 - 3 - 4 = 9. She earns 9 * 2 = 18.",
            is_correct=True,
            question_id="q001",
        )
        result = self.builder.label_chain(chain)
        assert isinstance(result, OracleChain)
        assert result.question_id == "q001"
        assert result.is_correct
        assert result.model_id == "test-model"
        assert result.inference_mode == "live_gpu"
        assert isinstance(result.step_labels, list)
        assert len(result.step_labels) > 0

    def test_label_violation_detected(self) -> None:
        # SCENARIO-DATA-019: a response with an arithmetic error should detect a violation
        chain = _make_chain(
            response="47 + 28 = 65. The answer is 65.",
            is_correct=False,
            question_id="q002",
        )
        result = self.builder.label_chain(chain)
        # SymCodeVerifier in regex mode detects "47 + 28" vs stated "65"
        assert result.has_violation
        assert result.n_violated_steps >= 1
        violated = [sl for sl in result.step_labels if sl.violation_detected]
        assert all(sl.label == "violated" for sl in violated)

    def test_question_id_fallback_to_hash(self) -> None:
        # When question_id is absent, falls back to str(hash(question))
        chain = _make_chain(question="What is 1+1?", response="1+1=2.", is_correct=True)
        result = self.builder.label_chain(chain)
        assert result.question_id == str(hash("What is 1+1?"))

    def test_question_index_does_not_override_hash_fallback(self) -> None:
        # question_index is NOT used as question_id; only question_id or hash fallback
        chain = _make_chain(question="What is 1+1?", response="1+1=2.", is_correct=True)
        chain["question_index"] = 42  # present but not used
        result = self.builder.label_chain(chain)
        # falls back to str(hash(question)) when question_id absent
        assert result.question_id == str(hash("What is 1+1?"))

    def test_model_id_from_model_field(self) -> None:
        # 'model' key is used when 'model_id' is absent
        chain = _make_chain(model="google/gemma-4", response="2+2=4.")
        result = self.builder.label_chain(chain)
        assert result.model_id == "google/gemma-4"

    def test_model_id_field_takes_precedence(self) -> None:
        # 'model_id' takes precedence over 'model' when both present
        chain = _make_chain(model="fallback-model", response="2+2=4.")
        chain["model_id"] = "preferred-model"
        result = self.builder.label_chain(chain)
        assert result.model_id == "preferred-model"

    def test_inference_mode_default(self) -> None:
        # When inference_mode is absent, defaults to 'live_gpu'
        chain = _make_chain(response="2+2=4.")
        result = self.builder.label_chain(chain)
        assert result.inference_mode == "live_gpu"

    def test_inference_mode_explicit(self) -> None:
        chain = _make_chain(response="2+2=4.", inference_mode="cpu")
        result = self.builder.label_chain(chain)
        assert result.inference_mode == "cpu"

    def test_step_labels_are_steplabel_instances(self) -> None:
        chain = _make_chain(response="2+2=4. Then 5*3=15.")
        result = self.builder.label_chain(chain)
        for sl in result.step_labels:
            assert isinstance(sl, StepLabel)

    def test_step_label_indices_are_sequential(self) -> None:
        chain = _make_chain(response="Step one. Step two. Step three.")
        result = self.builder.label_chain(chain)
        for i, sl in enumerate(result.step_labels):
            assert sl.step_index == i

    def test_has_violation_consistent_with_step_labels(self) -> None:
        chain = _make_chain(response="47 + 28 = 65. The answer is 65.")
        result = self.builder.label_chain(chain)
        expected_has_violation = any(sl.violation_detected for sl in result.step_labels)
        assert result.has_violation == expected_has_violation

    def test_n_violated_steps_count(self) -> None:
        chain = _make_chain(response="47 + 28 = 65. The answer is 65.")
        result = self.builder.label_chain(chain)
        expected = sum(1 for sl in result.step_labels if sl.violation_detected)
        assert result.n_violated_steps == expected

    def test_empty_response(self) -> None:
        # An empty response produces an OracleChain with zero step labels
        chain = _make_chain(response="")
        result = self.builder.label_chain(chain)
        assert isinstance(result, OracleChain)
        assert result.step_labels == []
        assert not result.has_violation
        assert result.n_violated_steps == 0

    def test_is_correct_false(self) -> None:
        chain = _make_chain(is_correct=False, response="The answer is wrong.")
        result = self.builder.label_chain(chain)
        assert result.is_correct is False


# ---------------------------------------------------------------------------
# OracleCorpusBuilder.build_corpus
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    """SCENARIO-DATA-020: build_corpus processes a list of live-pair dicts."""

    def setup_method(self) -> None:
        self.verifier = SymCodeVerifier(llm_caller=None)
        self.builder = OracleCorpusBuilder(self.verifier)

    def test_empty_input(self) -> None:
        # REQ-DATA-012: build_corpus on empty list returns empty list
        result = self.builder.build_corpus([])
        assert result == []

    def test_single_chain(self) -> None:
        pairs = [_make_chain(response="2+2=4.", question_id="q1")]
        result = self.builder.build_corpus(pairs)
        assert len(result) == 1
        assert isinstance(result[0], OracleChain)

    def test_multiple_chains_order_preserved(self) -> None:
        # REQ-DATA-013: build_corpus preserves input order
        pairs = [
            _make_chain(question="Q1?", response="1+1=2.", question_id="q1"),
            _make_chain(question="Q2?", response="2+2=4.", question_id="q2"),
            _make_chain(question="Q3?", response="3+3=6.", question_id="q3"),
        ]
        result = self.builder.build_corpus(pairs)
        assert len(result) == 3
        assert result[0].question_id == "q1"
        assert result[1].question_id == "q2"
        assert result[2].question_id == "q3"

    def test_violation_chain_in_batch(self) -> None:
        # SCENARIO-DATA-020: violated chain is correctly labeled within a batch
        pairs = [
            _make_chain(response="2+2=4.", is_correct=True, question_id="q_ok"),
            _make_chain(response="47+28=65.", is_correct=False, question_id="q_bad"),
        ]
        result = self.builder.build_corpus(pairs)
        ok = next(c for c in result if c.question_id == "q_ok")
        bad = next(c for c in result if c.question_id == "q_bad")
        # Violation in the bad chain
        assert bad.has_violation

    def test_all_chains_are_oracle_chain(self) -> None:
        pairs = [_make_chain(response="x.", question_id=f"q{i}") for i in range(5)]
        result = self.builder.build_corpus(pairs)
        assert all(isinstance(c, OracleChain) for c in result)

    def test_serialisable_with_dataclasses_asdict(self) -> None:
        # REQ-DATA-012: corpus must be JSON-serialisable via dataclasses.asdict
        pairs = [_make_chain(response="2+2=4.", question_id="q1")]
        result = self.builder.build_corpus(pairs)
        dicts = [dataclasses.asdict(c) for c in result]
        import json
        serialised = json.dumps(dicts)  # must not raise
        loaded = json.loads(serialised)
        assert loaded[0]["question_id"] == "q1"
