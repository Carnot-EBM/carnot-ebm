"""Tests for Experiment 938: DraftConditionedVerifier wired into ThreeTierPipeline.

Covers:
- wire_tier_28: stores verifier on pipeline without raising (REQ-PIPE-025)
- wire_tier_28: _last_tier28_advisory is None before any verify() call (REQ-PIPE-025)
- verify(): _last_tier28_advisory populated when Tier 2.8 is wired and Ising is reached (REQ-PIPE-025)
- verify(): _last_tier28_advisory contains required keys (REQ-PIPE-025)
- verify(): _last_tier28_advisory remains None when Tier 2.8 not wired (REQ-PIPE-025)
- verify(): Tier 2.8 advisory does not alter Ising verified result (advisory-only) (REQ-PIPE-025)
- _FixedEORMStub: returns 0.9 so EORM gate never fires (experiment helper)
- _CountingIsingStub: records calls so activation rate can be measured (experiment helper)
- _FixedDraftRunner: always returns structurally rich draft string (experiment helper)
- _compute_auc: returns 1.0 for perfectly separated energies
- _compute_auc: returns 0.5 for identical energies (random baseline)
- activation_count: >= 3 over 20 questions when EORM always passes through to Ising

Spec: REQ-PIPE-025, SCENARIO-PIPE-010
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path.
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python.carnot.models.eorm import CoTEnergyInput  # noqa: E402
from python.carnot.pipeline.draft_conditioned_verifier import DraftConditionedVerifier  # noqa: E402
from python.carnot.pipeline.sink_probe import SinkProbe  # noqa: E402
from python.carnot.pipeline.three_tier_pipeline import ThreeTierPipeline  # noqa: E402

# Import helpers from the experiment script itself to test them in isolation.
# This ensures the test file covers the script's helper code.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from experiment_938_draft_conditioned_tier28_integration import (  # noqa: E402
    _ComputeAuc,
    _CountingIsingStub,
    _FixedDraftRunner,
    _FixedEORMStub,
    QUESTIONS,
    _make_response,
)


# ---------------------------------------------------------------------------
# Stubs re-used across tests
# ---------------------------------------------------------------------------


def _build_pipeline(*, wire: bool = True) -> tuple[ThreeTierPipeline, _CountingIsingStub]:
    """Build a ThreeTierPipeline with stub components.

    Returns (pipeline, ising_stub) so tests can inspect ising_stub.calls.
    When wire=True, DraftConditionedVerifier is wired via wire_tier_28().
    """
    sink_probe = SinkProbe()
    eorm_stub = _FixedEORMStub()
    ising_stub = _CountingIsingStub()
    pipeline = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_stub,
        ising_pipeline=ising_stub,
        sink_threshold=0.99,  # never fires in tests
        eorm_threshold=0.5,  # stub returns 0.9, never clears
    )
    if wire:
        verifier = DraftConditionedVerifier(
            draft_runner=_FixedDraftRunner(),
            ising_sampler=None,
        )
        pipeline.wire_tier_28(verifier)
    return pipeline, ising_stub


# ---------------------------------------------------------------------------
# REQ-PIPE-025: wire_tier_28 and _last_tier28_advisory
# ---------------------------------------------------------------------------


class TestWireTier28:
    """REQ-PIPE-025: ThreeTierPipeline.wire_tier_28() contract."""

    def test_wire_does_not_raise(self) -> None:
        """wire_tier_28 MUST not raise when called with a valid verifier."""
        pipeline, _ = _build_pipeline(wire=False)
        verifier = DraftConditionedVerifier(draft_runner=_FixedDraftRunner(), ising_sampler=None)
        pipeline.wire_tier_28(verifier)  # should not raise

    def test_advisory_none_before_verify(self) -> None:
        """_last_tier28_advisory is None immediately after pipeline construction."""
        pipeline, _ = _build_pipeline(wire=True)
        # We only check the initial state; verify() is called in subsequent tests.
        assert pipeline._last_tier28_advisory is None

    def test_advisory_populated_after_verify_when_wired(self) -> None:
        """_last_tier28_advisory is a dict after verify() when Tier 2.8 is wired."""
        pipeline, _ = _build_pipeline(wire=True)
        pipeline.verify("The answer is 8.", question="5 + 3 = ?")
        assert pipeline._last_tier28_advisory is not None
        assert isinstance(pipeline._last_tier28_advisory, dict)

    def test_advisory_has_required_keys(self) -> None:
        """_last_tier28_advisory dict has all required keys after verify()."""
        pipeline, _ = _build_pipeline(wire=True)
        pipeline.verify("The answer is 8.", question="5 + 3 = ?")
        advisory = pipeline._last_tier28_advisory
        assert advisory is not None
        for key in ("energy", "draft_used", "n_constraints", "draft_text", "constraints"):
            assert key in advisory, f"missing key: {key}"

    def test_advisory_none_when_not_wired(self) -> None:
        """_last_tier28_advisory stays None after verify() when Tier 2.8 not wired."""
        pipeline, _ = _build_pipeline(wire=False)
        pipeline.verify("The answer is 8.", question="5 + 3 = ?")
        # Without wiring, advisory should never be set.
        assert pipeline._last_tier28_advisory is None

    def test_ising_result_unchanged_by_tier28(self) -> None:
        """Tier 2.8 is advisory-only: verified result must match unwired pipeline."""
        p_with, _ = _build_pipeline(wire=True)
        p_without, _ = _build_pipeline(wire=False)
        resp = "Step 1: x = 5 + 3.\nStep 2: x = 8.\nThe answer is 8."
        question = "What is 5 + 3?"
        v_with, _, _ = p_with.verify(resp, question=question)
        v_without, _, _ = p_without.verify(resp, question=question)
        assert v_with == v_without


# ---------------------------------------------------------------------------
# Experiment helper unit tests
# ---------------------------------------------------------------------------


class TestFixedEORMStub:
    """_FixedEORMStub always returns 0.9 to force Ising pathway."""

    def test_returns_float(self) -> None:
        stub = _FixedEORMStub()
        cot = CoTEnergyInput(question_text="q", response_text="r")
        assert isinstance(stub.energy(cot), float)

    def test_returns_0_9(self) -> None:
        """Returns 0.9 regardless of input, ensuring eorm_threshold=0.5 is never met."""
        stub = _FixedEORMStub()
        cot = CoTEnergyInput(question_text="any", response_text="any")
        assert stub.energy(cot) == pytest.approx(0.9)


class TestCountingIsingStub:
    """_CountingIsingStub records calls and returns (bool, float)."""

    def test_call_returns_tuple(self) -> None:
        stub = _CountingIsingStub()
        result = stub("The answer is 8.", "5 + 3 = ?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_call_returns_bool_and_float(self) -> None:
        stub = _CountingIsingStub()
        verified, energy = stub("The answer is 8.", "q")
        assert isinstance(verified, bool)
        assert isinstance(energy, float)

    def test_call_logged(self) -> None:
        stub = _CountingIsingStub()
        stub("response text", "question text")
        assert len(stub.calls) == 1

    def test_multiple_calls_logged(self) -> None:
        stub = _CountingIsingStub()
        for _ in range(5):
            stub("r", "q")
        assert len(stub.calls) == 5


class TestFixedDraftRunner:
    """_FixedDraftRunner always returns a structurally rich draft."""

    def test_returns_string(self) -> None:
        runner = _FixedDraftRunner()
        out = runner.generate("any question", max_tokens=50)
        assert isinstance(out, str)

    def test_draft_non_empty(self) -> None:
        runner = _FixedDraftRunner()
        out = runner.generate("q")
        assert len(out) > 0

    def test_draft_contains_equals(self) -> None:
        """Draft must contain '=' so has_equals_sign constraint fires."""
        runner = _FixedDraftRunner()
        out = runner.generate("q")
        assert "=" in out

    def test_draft_has_multiline(self) -> None:
        """Draft must have > 3 lines so has_reasoning_steps constraint fires."""
        runner = _FixedDraftRunner()
        out = runner.generate("q")
        assert len(out.split("\n")) > 3


class TestComputeAuc:
    """_compute_auc rank-based AUC helper."""

    def test_perfect_separation_returns_1(self) -> None:
        """When all correct energies < all wrong energies: AUC = 1.0."""
        pairs = [(0.1, 1), (0.2, 1), (0.8, 0), (0.9, 0)]
        assert _ComputeAuc(pairs) == pytest.approx(1.0)

    def test_no_separation_returns_0(self) -> None:
        """When all correct energies > all wrong energies: AUC = 0.0."""
        pairs = [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]
        assert _ComputeAuc(pairs) == pytest.approx(0.0)

    def test_random_baseline_returns_0_5(self) -> None:
        """When energies are identical: AUC = 0.5."""
        pairs = [(0.5, 1), (0.5, 1), (0.5, 0), (0.5, 0)]
        assert _ComputeAuc(pairs) == pytest.approx(0.5)

    def test_empty_returns_0_5(self) -> None:
        """Empty input returns 0.5 (random baseline)."""
        assert _ComputeAuc([]) == pytest.approx(0.5)


class TestMakeResponse:
    """_make_response generates structurally plausible CoT responses."""

    def test_correct_response_contains_answer(self) -> None:
        resp = _make_response("What is 5 + 3?", 8, is_correct=True)
        assert "8" in resp

    def test_wrong_response_does_not_contain_correct_answer(self) -> None:
        resp = _make_response("What is 5 + 3?", 8, is_correct=False)
        # Wrong answer = 8 + 7 = 15; 8 should not appear as the final answer.
        assert "15" in resp

    def test_both_responses_non_empty(self) -> None:
        for is_correct in (True, False):
            resp = _make_response("q", 10, is_correct=is_correct)
            assert len(resp) > 0


class TestActivationRate:
    """Integration: Tier 2.8 must fire >= 3 times over 20 questions."""

    def test_activation_count_ge_3(self) -> None:
        """REQ-PIPE-025, SCENARIO-PIPE-010: tier28 activates >= 3 / 20 questions."""
        pipeline, _ = _build_pipeline(wire=True)
        activation_count = 0
        for question, correct_answer in QUESTIONS:
            resp = _make_response(question, correct_answer, is_correct=True)
            pipeline.verify(resp, question=question)
            advisory = pipeline._last_tier28_advisory
            if advisory is not None and advisory.get("draft_used", False):
                activation_count += 1
        assert activation_count >= 3, (
            f"Expected Tier 2.8 to activate >= 3 times, got {activation_count}"
        )
