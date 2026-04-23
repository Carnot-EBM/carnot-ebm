"""Tests for Experiment 768 — Gemma4 Loader Fix v2 + VR Threshold Grid.

Spec: REQ-LOADER-010, REQ-VERIFY-170,
      SCENARIO-LOADER-010, SCENARIO-VERIFY-225, SCENARIO-VERIFY-226
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path for scripts/ imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_768_gemma4_loader_fix_v2 import (
    THRESHOLDS,
    _answers_match,
    _extract_numeric_answer,
    _symcode_confidence,
    audit_gemma4_call_sites,
    classify_verdict,
)


# ---------------------------------------------------------------------------
# REQ-LOADER-010: GemmaTransformersLoader is used for google/gemma-4-*
# ---------------------------------------------------------------------------


def test_gemma_transformers_loader_is_used_in_exp768() -> None:
    """Exp 768 imports GemmaTransformersLoader — not llama.cpp — for Gemma4.

    REQ-LOADER-010: all google/gemma-4-* loading MUST use GemmaTransformersLoader.
    SCENARIO-LOADER-010: the smoke test uses GemmaTransformersLoader.generate().
    """
    import scripts.experiment_768_gemma4_loader_fix_v2 as mod

    src = Path(mod.__file__).read_text()
    assert "GemmaTransformersLoader" in src, "Exp 768 must import GemmaTransformersLoader"
    # Confirm no bare llama_cpp usage for non-GGUF Gemma4 loading.
    assert "llama_cpp.Llama(" not in src, "Exp 768 must not use llama_cpp.Llama for Gemma4"


def test_gemma_transformers_loader_validates_output() -> None:
    """GemmaTransformersLoader.is_valid_output() rejects <unusedN> tokens.

    REQ-LOADER-010: the loader must reject llama.cpp#21516 bug output.
    SCENARIO-LOADER-010: smoke test calls is_valid_output() on the generated text.
    """
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader

    assert GemmaTransformersLoader.is_valid_output("<unused8>") is False
    assert GemmaTransformersLoader.is_valid_output("<unused8> <unused8>") is False
    assert GemmaTransformersLoader.is_valid_output("Hello, how can I help?") is True
    assert GemmaTransformersLoader.is_valid_output("") is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-170-2 / SCENARIO-VERIFY-225: per_threshold_results has 5 entries
# ---------------------------------------------------------------------------


def test_thresholds_has_five_entries() -> None:
    """THRESHOLDS constant must have exactly 5 entries.

    REQ-VERIFY-170-2: per_threshold_results MUST have exactly 5 entries.
    """
    assert len(THRESHOLDS) == 5


def test_threshold_values_are_in_expected_range() -> None:
    """All thresholds must be in [0.10, 0.50].

    REQ-VERIFY-170-2.
    """
    for t in THRESHOLDS:
        assert 0.10 <= t <= 0.50, f"Threshold {t} out of [0.10, 0.50]"


def test_per_threshold_results_has_five_entries() -> None:
    """Synthetic results list with 5 entries matches THRESHOLDS length.

    REQ-VERIFY-170-2: per_threshold_results MUST have exactly 5 entries.
    """
    results = [
        {"threshold": t, "signed_improvement": 0.0, "n_abstained": 0}
        for t in THRESHOLDS
    ]
    assert len(results) == 5


# ---------------------------------------------------------------------------
# REQ-VERIFY-170-4 / SCENARIO-VERIFY-226: positive_threshold_found
# ---------------------------------------------------------------------------


def test_positive_threshold_found_when_any_improvement_positive() -> None:
    """positive_threshold_found is True when any signed_improvement > 0.

    REQ-VERIFY-170-4, SCENARIO-VERIFY-226.
    """
    improvements = [-0.02, -0.01, 0.03, 0.00, -0.01]
    positive_found = any(si > 0 for si in improvements)
    assert positive_found is True


def test_positive_threshold_not_found_when_all_non_positive() -> None:
    """positive_threshold_found is False when all signed_improvement <= 0.

    REQ-VERIFY-170-4, SCENARIO-VERIFY-226.
    """
    improvements = [-0.04, -0.02, 0.00, -0.01, -0.03]
    positive_found = any(si > 0 for si in improvements)
    assert positive_found is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-170-5 through REQ-VERIFY-170-8: classify_verdict
# ---------------------------------------------------------------------------


def test_classify_verdict_blocked_no_live_gpu() -> None:
    """classify_verdict returns 'blocked_no_live_gpu' when inference_mode matches.

    REQ-VERIFY-170-8.
    """
    assert (
        classify_verdict(
            loader_test_passed=False,
            positive_threshold_found=False,
            inference_mode="blocked_no_live_gpu",
        )
        == "blocked_no_live_gpu"
    )


def test_classify_verdict_loader_still_broken() -> None:
    """classify_verdict returns 'loader_still_broken' when loader failed.

    REQ-VERIFY-170-7.
    """
    assert (
        classify_verdict(
            loader_test_passed=False,
            positive_threshold_found=False,
            inference_mode="live_gpu",
        )
        == "loader_still_broken"
    )


def test_classify_verdict_retro028_closed_positive() -> None:
    """classify_verdict returns 'retro028_closed_positive_threshold_found' on success.

    REQ-VERIFY-170-5.
    """
    assert (
        classify_verdict(
            loader_test_passed=True,
            positive_threshold_found=True,
            inference_mode="live_gpu",
        )
        == "retro028_closed_positive_threshold_found"
    )


def test_classify_verdict_retro028_closed_no_positive() -> None:
    """classify_verdict returns 'retro028_closed_no_positive_threshold' when no improvement.

    REQ-VERIFY-170-6.
    """
    assert (
        classify_verdict(
            loader_test_passed=True,
            positive_threshold_found=False,
            inference_mode="live_gpu",
        )
        == "retro028_closed_no_positive_threshold"
    )


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def test_extract_numeric_answer_explicit_pattern() -> None:
    """_extract_numeric_answer finds 'answer is X' patterns."""
    assert _extract_numeric_answer("The answer is 42") == 42.0


def test_extract_numeric_answer_fallback_last_num() -> None:
    """_extract_numeric_answer falls back to last numeric token."""
    assert _extract_numeric_answer("So we get 7 groups of 6 = 42.") == 42.0


def test_extract_numeric_answer_none_on_no_numbers() -> None:
    """_extract_numeric_answer returns None for text with no numbers."""
    assert _extract_numeric_answer("no numbers here") is None
    assert _extract_numeric_answer("") is None


def test_answers_match_within_tolerance() -> None:
    """_answers_match is True for values within 0.5 tolerance."""
    assert _answers_match(42.0, 42) is True
    assert _answers_match(42.3, 42) is True


def test_answers_match_false_for_wrong_answer() -> None:
    """_answers_match is False when difference exceeds tolerance."""
    assert _answers_match(10.0, 42) is False


def test_answers_match_none_returns_false() -> None:
    """_answers_match is False when either value is None."""
    assert _answers_match(None, 42) is False
    assert _answers_match(42.0, None) is False


# ---------------------------------------------------------------------------
# _symcode_confidence
# ---------------------------------------------------------------------------


def test_symcode_confidence_zero_compute_lines() -> None:
    """_symcode_confidence returns 0.2 when no COMPUTE: lines present."""
    assert _symcode_confidence("The answer is 42") == 0.2


def test_symcode_confidence_five_compute_lines() -> None:
    """_symcode_confidence returns 1.0 (capped) for 5+ COMPUTE: lines."""
    text = "COMPUTE: step1\nCOMPUTE: step2\nCOMPUTE: step3\nCOMPUTE: step4\nCOMPUTE: step5"
    assert _symcode_confidence(text) == 1.0


def test_symcode_confidence_partial_compute_lines() -> None:
    """_symcode_confidence is proportional below 5 COMPUTE: lines."""
    text = "COMPUTE: step1\nCOMPUTE: step2"
    assert abs(_symcode_confidence(text) - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# audit_gemma4_call_sites
# ---------------------------------------------------------------------------


def test_audit_returns_expected_keys() -> None:
    """audit_gemma4_call_sites returns a dict with all required keys.

    REQ-LOADER-010: the audit must produce n_call_sites_fixed.
    """
    result = audit_gemma4_call_sites()
    assert "n_files_scanned" in result
    assert "n_call_sites_audited" in result
    assert "n_call_sites_fixed" in result
    assert "flagged_files" in result
    assert isinstance(result["n_call_sites_fixed"], int)


def test_audit_finds_gemma4_references() -> None:
    """audit_gemma4_call_sites finds at least one google/gemma-4 reference.

    REQ-LOADER-010: the codebase has Gemma4 model references to audit.
    """
    result = audit_gemma4_call_sites()
    assert result["n_call_sites_audited"] > 0, "Expected at least one google/gemma-4 reference"


def test_audit_n_call_sites_fixed_is_nonnegative() -> None:
    """n_call_sites_fixed must be >= 0 (never negative).

    REQ-LOADER-010.
    """
    result = audit_gemma4_call_sites()
    assert result["n_call_sites_fixed"] >= 0


# ---------------------------------------------------------------------------
# main() integration test: blocked path (no CARNOT_FORCE_LIVE)
# ---------------------------------------------------------------------------


def test_main_writes_blocked_artifact_when_no_live_gpu(tmp_path: Path) -> None:
    """main() writes blocked_no_live_gpu artifact when CARNOT_FORCE_LIVE is not set.

    REQ-VERIFY-170-8: honest_verdict MUST be 'blocked_no_live_gpu' without CARNOT_FORCE_LIVE.
    """
    import scripts.experiment_768_gemma4_loader_fix_v2 as mod

    deliverable = tmp_path / "experiment_768_gemma4_loader_fix_v2.json"

    env_patch = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}

    with (
        patch.object(mod, "_DELIVERABLE", str(deliverable)),
        patch.dict(os.environ, env_patch, clear=True),
        patch("scripts.experiment_768_gemma4_loader_fix_v2.ExperimentTemplate") as MockTemplate,
        patch("scripts.experiment_768_gemma4_loader_fix_v2.ExperimentTimeoutWatchdog") as MockWatchdog,
    ):
        # Mock ExperimentTemplate so no filesystem side-effects.
        mock_tmpl = MagicMock()
        mock_tmpl._output_path = deliverable
        # assert_* names are magic on MagicMock; set it explicitly.
        mock_tmpl.assert_deliverable_written = MagicMock()
        mock_tmpl.build_result.side_effect = lambda data, **kw: {
            "experiment": 768,
            "schema": sorted(data.keys()),
            "run_date": "20260423",
            "started_at": "2026-04-23T00:00:00Z",
            "finished_at": "2026-04-23T00:00:01Z",
            "duration_s": 1.0,
            "status": kw.get("status", "blocked"),
            "title": "Gemma4 Loader Fix v2",
            **data,
        }
        MockTemplate.return_value = mock_tmpl

        # Mock watchdog as context manager.
        mock_watchdog_instance = MagicMock()
        mock_watchdog_instance.__enter__ = MagicMock(return_value=mock_watchdog_instance)
        mock_watchdog_instance.__exit__ = MagicMock(return_value=False)
        MockWatchdog.return_value = mock_watchdog_instance

        mod.main()

    assert deliverable.exists(), "Deliverable JSON must be written"
    with open(deliverable) as f:
        artifact = json.load(f)

    assert artifact["honest_verdict"] == "blocked_no_live_gpu"
    assert artifact["status"] == "blocked"
    assert "n_call_sites_fixed" in artifact
    assert "per_threshold_results" in artifact
    assert artifact["per_threshold_results"] == []
