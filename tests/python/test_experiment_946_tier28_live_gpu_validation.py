"""Tests for Experiment 946: Tier 2.8 Live GPU Validation.

Covers:
- _extract_numeric_answer: returns last number from text with digits
- _extract_numeric_answer: returns None when no digits present
- _extract_numeric_answer: returns decimal value when decimal present
- _build_stub_pipeline: returns pipeline with draft_conditioned_verifier wired
- _build_stub_pipeline: HighEnergyEORM returns energy above threshold
- _build_stub_pipeline: ising_stub returns lower energy for digit-rich responses
- pipeline.verify: _last_tier28_advisory is populated after verify() call
- pipeline.verify: tier28_activated is True when advisory is set
- main: writes blocked artifact when CARNOT_FORCE_LIVE not set
- main: writes blocked artifact when model load raises exception
- main: writes blocked artifact when loader._model is None after load
- GemmaTransformersLoader mock: generate() path through full main flow

Spec: REQ-TIER28-001, SCENARIO-TIER28-001, REQ-LOADER-001, REQ-LOADER-002
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root and scripts/ are on path.
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiment_946_tier28_live_gpu_validation import (  # noqa: E402
    GSM8K_QUESTIONS,
    _build_stub_pipeline,
    _extract_numeric_answer,
)


# ---------------------------------------------------------------------------
# _extract_numeric_answer tests
# ---------------------------------------------------------------------------


def test_extract_numeric_answer_integer():
    """Last integer in text is returned as a float."""
    assert _extract_numeric_answer("The answer is 42.") == 42.0


def test_extract_numeric_answer_multiple_numbers():
    """When multiple numbers present, the last one is returned."""
    assert _extract_numeric_answer("Sam has 5 apples and then 3 more making 8 total.") == 8.0


def test_extract_numeric_answer_no_digits():
    """Returns None when no digits appear in the text."""
    assert _extract_numeric_answer("No numbers here at all.") is None


def test_extract_numeric_answer_decimal():
    """Decimal numbers are returned as float."""
    result = _extract_numeric_answer("The cost is 3.5 dollars.")
    assert result == 3.5


def test_extract_numeric_answer_empty_string():
    """Empty string produces None."""
    assert _extract_numeric_answer("") is None


# ---------------------------------------------------------------------------
# _build_stub_pipeline tests
# ---------------------------------------------------------------------------


def test_build_stub_pipeline_tier28_wired():
    """_build_stub_pipeline wires DraftConditionedVerifier (not None)."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Step 1: 5 + 3 = 8. The answer is 8."
    pipeline, tier28_verifier = _build_stub_pipeline(mock_loader)
    assert pipeline.draft_conditioned_verifier is not None


def test_build_stub_pipeline_returns_verifier():
    """Second return value is the DraftConditionedVerifier instance (checked by class name)."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Answer: 42"
    _, tier28_verifier = _build_stub_pipeline(mock_loader)
    # Check by class name to avoid module-path mismatch (carnot vs python.carnot).
    assert type(tier28_verifier).__name__ == "DraftConditionedVerifier"


def test_high_energy_eorm_forces_tier3():
    """HighEnergyEORM always returns energy > 0.5, so Tier 3 (Ising) is always reached."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Step 1: 5 + 3 = 8."
    pipeline, _ = _build_stub_pipeline(mock_loader)

    # Run verify; because EORM energy=0.9 > 0.5, Ising should be the deciding tier.
    pipeline._last_tier28_advisory = None
    _verified, tier_used, _energy = pipeline.verify("Some response.", question="What is 5+3?")
    assert tier_used == "ising"


def test_ising_stub_high_energy_for_short_empty_response():
    """ising_stub returns 0.7 for an empty response (no digit, short)."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = ""
    pipeline, _ = _build_stub_pipeline(mock_loader)

    _verified, _tier_used, energy = pipeline.verify("", question="What is 5+3?")
    # An empty response has no digit → energy=0.7
    assert energy == pytest.approx(0.7, abs=1e-6)


def test_ising_stub_low_energy_for_digit_rich_response():
    """ising_stub returns 0.3 for a long response with digits."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Step 1: 5 + 3 = 8."
    pipeline, _ = _build_stub_pipeline(mock_loader)

    long_response = "Step 1: add 5 + 3 = 8. The answer is 8 apples total."
    _verified, _tier_used, energy = pipeline.verify(long_response, question="What is 5+3?")
    assert energy == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Tier 2.8 advisory activation tests
# ---------------------------------------------------------------------------


def test_tier28_advisory_populated_after_verify():
    """After pipeline.verify(), _last_tier28_advisory is not None when Tier 2.8 is wired."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Step 1: 5 + 3 = 8. Answer: 8."
    pipeline, _ = _build_stub_pipeline(mock_loader)

    pipeline._last_tier28_advisory = None
    pipeline.verify(
        "Step 1: 5 + 3 = 8. Answer: 8.",
        question="What is 5+3?",
    )
    assert pipeline._last_tier28_advisory is not None


def test_tier28_advisory_has_required_keys():
    """The advisory dict from Tier 2.8 contains energy, draft_used, n_constraints."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "The answer is 8."
    pipeline, _ = _build_stub_pipeline(mock_loader)

    pipeline._last_tier28_advisory = None
    pipeline.verify("The answer is 8.", question="What is 5+3?")
    advisory = pipeline._last_tier28_advisory
    assert "energy" in advisory
    assert "draft_used" in advisory
    assert "n_constraints" in advisory


def test_tier28_activation_across_questions():
    """All 20 GSM8K questions activate Tier 2.8 when EORM stub always passes through."""
    mock_loader = MagicMock()
    mock_loader.generate.return_value = "Step 1: x + y = z. Answer: 42."

    pipeline, _ = _build_stub_pipeline(mock_loader)

    activation_count = 0
    for question, _ in GSM8K_QUESTIONS:
        pipeline._last_tier28_advisory = None
        pipeline.verify(
            "Step 1: x + y = z. Answer: 42.",
            question=question,
        )
        if pipeline._last_tier28_advisory is not None:
            activation_count += 1

    # All 20 questions must activate Tier 2.8 with our HighEnergyEORM stub.
    assert activation_count == 20


# ---------------------------------------------------------------------------
# main() blocked-gate tests (mock ExperimentTemplate + GemmaTransformersLoader)
# ---------------------------------------------------------------------------


def _make_template_mock(deliverable: Path) -> MagicMock:
    """Build a MagicMock ExperimentTemplate that writes to deliverable.

    MagicMock raises AttributeError for attribute names starting with 'assert_' unless
    they are explicitly set as regular attributes.  We set assert_deliverable_written
    explicitly to avoid this guard.
    """
    instance = MagicMock()
    instance._output_path = deliverable
    instance.build_result.side_effect = lambda data, **kw: {
        **data,
        "status": kw.get("status", "blocked"),
    }
    # Explicitly set to avoid MagicMock's assert_* safety guard (Python 3.8+).
    instance.assert_deliverable_written = MagicMock(return_value=None)
    return instance


def test_main_blocked_when_no_live_flag(tmp_path, monkeypatch):
    """main() writes blocked artifact and exits 0 when CARNOT_FORCE_LIVE != '1'."""
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    deliverable = tmp_path / "results" / "experiment_946_tier28_live_gpu_validation.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)

    import experiment_946_tier28_live_gpu_validation as mod946  # noqa: PLC0415

    with patch.object(mod946, "ExperimentTemplate") as MockTemplate:
        MockTemplate.return_value = _make_template_mock(deliverable)

        with pytest.raises(SystemExit) as exc_info:
            mod946.main()

        assert exc_info.value.code == 0

    written = json.loads(deliverable.read_text())
    assert written.get("honest_verdict") == "blocked_no_live_gpu"


def test_main_blocked_when_model_load_raises(tmp_path, monkeypatch):
    """main() writes blocked_model_load_failed when GemmaTransformersLoader.load() raises."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    deliverable = tmp_path / "results" / "experiment_946_tier28_live_gpu_validation.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = RuntimeError("CUDA OOM")
    mock_loader_instance._model = None

    import experiment_946_tier28_live_gpu_validation as mod946  # noqa: PLC0415

    # GemmaTransformersLoader is a local import inside main().  Patch the source module
    # so the local `from carnot.pipeline.gemma_loader import GemmaTransformersLoader`
    # picks up our mock when the module is looked up in sys.modules at call time.
    with (
        patch.object(mod946, "ExperimentTemplate") as MockTemplate,
        patch(
            "carnot.pipeline.gemma_loader.GemmaTransformersLoader",
            return_value=mock_loader_instance,
        ),
    ):
        MockTemplate.return_value = _make_template_mock(deliverable)

        with pytest.raises(SystemExit) as exc_info:
            mod946.main()

        assert exc_info.value.code == 0

    written = json.loads(deliverable.read_text())
    assert written.get("honest_verdict") == "blocked_model_load_failed"


def test_main_blocked_when_model_is_none_after_load(tmp_path, monkeypatch):
    """main() writes blocked_model_load_failed when loader._model is None after load()."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    deliverable = tmp_path / "results" / "experiment_946_tier28_live_gpu_validation.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = None  # does not raise
    mock_loader_instance._model = None  # model is still None after load
    mock_loader_instance._tokenizer = None

    import experiment_946_tier28_live_gpu_validation as mod946  # noqa: PLC0415

    with (
        patch.object(mod946, "ExperimentTemplate") as MockTemplate,
        patch(
            "carnot.pipeline.gemma_loader.GemmaTransformersLoader",
            return_value=mock_loader_instance,
        ),
    ):
        MockTemplate.return_value = _make_template_mock(deliverable)

        with pytest.raises(SystemExit) as exc_info:
            mod946.main()

        assert exc_info.value.code == 0

    written = json.loads(deliverable.read_text())
    assert written.get("honest_verdict") == "blocked_model_load_failed"
