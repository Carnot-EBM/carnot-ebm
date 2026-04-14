"""Tests for scripts/experiment_284_apple_analysis.py.

Validates the analysis functions and artifact schema for Exp 284: Apple
adversarial GSM8K result analysis.  All tests run without GPU hardware and
without requiring Exp 282/283 result files to be present.

Spec coverage:
  REQ-VERIFY-073  — Apple adversarial analysis and classification
  REQ-VERIFY-074  — compute_delta rounds to four decimal places
  REQ-VERIFY-075  — classify_result maps (primary_met, partial, stall) → label
  SCENARIO-VERIFY-088 — five key questions answered in artifact
  SCENARIO-VERIFY-089 — INCONCLUSIVE when result files are missing
  SCENARIO-VERIFY-090 — compare_vs_exp235 returns comparison dict
  SCENARIO-VERIFY-091 — compute_delta arithmetic
  SCENARIO-VERIFY-092 — artifact schema contains all required top-level keys
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_MODULE_NAME = "experiment_284_apple_analysis"
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "experiment_284_apple_analysis.py"
)


def _load_module() -> Any:
    """Load the experiment_284 module without invoking main()."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


MOD = _load_module()

compute_delta = MOD.compute_delta
classify_result = MOD.classify_result
compare_vs_exp235 = MOD.compare_vs_exp235
load_exp_results = MOD.load_exp_results
build_artifact = MOD.build_artifact
answer_five_questions = MOD.answer_five_questions
EXPERIMENT = MOD.EXPERIMENT
ARTIFACT_SCHEMA = MOD.ARTIFACT_SCHEMA
RUN_DATE = MOD.RUN_DATE


# ---------------------------------------------------------------------------
# Fixture: minimal Exp 282 artifact
# ---------------------------------------------------------------------------

_FAKE_282 = {
    "experiment": 282,
    "schema": "apple_adversarial_baseline.v1",
    "run_date": "20260414",
    "started_at": "2026-04-14T05:00:00Z",
    "finished_at": "2026-04-14T05:30:00Z",
    "inference_mode": "live_gpu",
    "partial": False,
    "stall_at": None,
    "model_results": {
        "Qwen3.5-0.8B": {
            "standard":             {"accuracy": 0.20, "correct": 40, "total": 200},
            "number_swap":          {"accuracy": 0.04, "correct":  8, "total": 200},
            "irrelevant_sentence":  {"accuracy": 0.19, "correct": 38, "total": 200},
        },
        "Gemma4-E4B-it": {
            "standard":             {"accuracy": 0.465, "correct": 93, "total": 200},
            "number_swap":          {"accuracy": 0.28,  "correct": 56, "total": 200},
            "irrelevant_sentence":  {"accuracy": 0.460, "correct": 92, "total": 200},
        },
    },
    "apple_2410_05229_check": {
        "Qwen3.5-0.8B": {"number_swap_drop_pp": 16.0, "drop_gte_15pp": True},
        "Gemma4-E4B-it": {"number_swap_drop_pp": 18.5, "drop_gte_15pp": True},
    },
    "logit_paths": [],
}

_FAKE_283 = {
    "experiment": 283,
    "schema": "apple_adversarial_verify_repair.v1",
    "run_date": "20260414",
    "started_at": "2026-04-14T05:31:00Z",
    "finished_at": "2026-04-14T06:15:00Z",
    "inference_mode": "live_gpu",
    "partial": False,
    "stall_at": None,
    "results": {
        "Qwen3.5-0.8B": {
            "number_swap": {
                "baseline":      {"accuracy": 0.04,  "correct":  8, "total": 200, "violation_detected_count": 0, "repaired_count": 0},
                "verify_only":   {"accuracy": 0.04,  "correct":  8, "total": 200, "violation_detected_count": 5, "repaired_count": 0},
                "verify_repair": {"accuracy": 0.065, "correct": 13, "total": 200, "violation_detected_count": 5, "repaired_count": 5},
            },
            "irrelevant_sentence": {
                "baseline":      {"accuracy": 0.19,  "correct": 38, "total": 200, "violation_detected_count": 0, "repaired_count": 0},
                "verify_only":   {"accuracy": 0.19,  "correct": 38, "total": 200, "violation_detected_count": 2, "repaired_count": 0},
                "verify_repair": {"accuracy": 0.195, "correct": 39, "total": 200, "violation_detected_count": 2, "repaired_count": 1},
            },
        },
        "Gemma4-E4B-it": {
            "number_swap": {
                "baseline":      {"accuracy": 0.28,  "correct": 56, "total": 200, "violation_detected_count":  0, "repaired_count":  0},
                "verify_only":   {"accuracy": 0.28,  "correct": 56, "total": 200, "violation_detected_count": 18, "repaired_count":  0},
                "verify_repair": {"accuracy": 0.33,  "correct": 66, "total": 200, "violation_detected_count": 18, "repaired_count": 10},
            },
            "irrelevant_sentence": {
                "baseline":      {"accuracy": 0.46,  "correct": 92, "total": 200, "violation_detected_count":  0, "repaired_count": 0},
                "verify_only":   {"accuracy": 0.46,  "correct": 92, "total": 200, "violation_detected_count":  3, "repaired_count": 0},
                "verify_repair": {"accuracy": 0.465, "correct": 93, "total": 200, "violation_detected_count":  3, "repaired_count": 1},
            },
        },
    },
    "improvement_deltas": {
        "Qwen3.5-0.8B": {
            "verify_repair_number_swap_delta":         0.025,
            "verify_repair_irrelevant_sentence_delta": 0.005,
        },
        "Gemma4-E4B-it": {
            "verify_repair_number_swap_delta":         0.05,
            "verify_repair_irrelevant_sentence_delta": 0.005,
        },
    },
    "primary_criterion_met": True,
    "extractor_summary": {
        "semantic_grounding": 23,
        "formal_claim": 8,
        "arithmetic": 5,
    },
    "logit_paths": [],
}


# ---------------------------------------------------------------------------
# REQ-VERIFY-074 / SCENARIO-VERIFY-091: compute_delta
# ---------------------------------------------------------------------------

class TestComputeDelta:
    """REQ-VERIFY-074, SCENARIO-VERIFY-091: compute_delta arithmetic."""

    def test_positive_delta(self) -> None:
        """SCENARIO-VERIFY-091: 0.475 - 0.465 = 0.01."""
        result = compute_delta(0.465, 0.475)
        assert result == pytest.approx(0.01, abs=1e-6)

    def test_zero_delta(self) -> None:
        """REQ-VERIFY-074: identical accuracies produce 0.0."""
        assert compute_delta(0.5, 0.5) == pytest.approx(0.0)

    def test_negative_delta(self) -> None:
        """REQ-VERIFY-074: regression produces negative delta."""
        assert compute_delta(0.5, 0.4) == pytest.approx(-0.1, abs=1e-6)

    def test_four_decimal_rounding(self) -> None:
        """REQ-VERIFY-074: result is rounded to four decimal places."""
        # 1/3 - 0 = 0.3333...
        result = compute_delta(0.0, 1 / 3)
        assert result == round(1 / 3, 4)


# ---------------------------------------------------------------------------
# REQ-VERIFY-075: classify_result
# ---------------------------------------------------------------------------

class TestClassifyResult:
    """REQ-VERIFY-075: classification logic."""

    def test_confirmed(self) -> None:
        """REQ-VERIFY-075: primary_met → CONFIRMED."""
        assert classify_result(primary_met=True, partial_improvement=False, stall_detected=False) == "CONFIRMED"

    def test_confirmed_with_partial(self) -> None:
        """REQ-VERIFY-075: primary_met takes precedence over partial."""
        assert classify_result(primary_met=True, partial_improvement=True, stall_detected=False) == "CONFIRMED"

    def test_partial(self) -> None:
        """REQ-VERIFY-075: partial_improvement without primary → PARTIAL."""
        assert classify_result(primary_met=False, partial_improvement=True, stall_detected=False) == "PARTIAL"

    def test_ruled_out(self) -> None:
        """REQ-VERIFY-075: no improvement at all → RULED_OUT."""
        assert classify_result(primary_met=False, partial_improvement=False, stall_detected=False) == "RULED_OUT"

    def test_inconclusive_on_stall(self) -> None:
        """REQ-VERIFY-075: stall detected → INCONCLUSIVE regardless of other flags."""
        assert classify_result(primary_met=True, partial_improvement=True, stall_detected=True) == "INCONCLUSIVE"

    def test_inconclusive_stall_only(self) -> None:
        """REQ-VERIFY-075: stall alone → INCONCLUSIVE."""
        assert classify_result(primary_met=False, partial_improvement=False, stall_detected=True) == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-090: compare_vs_exp235
# ---------------------------------------------------------------------------

class TestCompareVsExp235:
    """SCENARIO-VERIFY-090: compare_vs_exp235 returns a comparison dict."""

    def test_better_than_exp235(self) -> None:
        """SCENARIO-VERIFY-090: number_swap_acc > exp235 → better_than_exp235 is True."""
        result = compare_vs_exp235(number_swap_acc=0.50, exp235_acc=0.475)
        assert result["delta"] == pytest.approx(0.025, abs=1e-6)
        assert result["better_than_exp235"] is True
        assert result["exp235_reference_acc"] == pytest.approx(0.475)

    def test_worse_than_exp235(self) -> None:
        """SCENARIO-VERIFY-090: lower accuracy → better_than_exp235 is False."""
        result = compare_vs_exp235(number_swap_acc=0.30, exp235_acc=0.475)
        assert result["better_than_exp235"] is False
        assert result["delta"] < 0

    def test_equal_to_exp235(self) -> None:
        """SCENARIO-VERIFY-090: equal accuracy → better_than_exp235 is False (not strictly better)."""
        result = compare_vs_exp235(number_swap_acc=0.475, exp235_acc=0.475)
        assert result["better_than_exp235"] is False
        assert result["delta"] == pytest.approx(0.0)

    def test_required_keys_present(self) -> None:
        """SCENARIO-VERIFY-090: returned dict has delta, better_than_exp235, exp235_reference_acc."""
        result = compare_vs_exp235(number_swap_acc=0.35, exp235_acc=0.475)
        assert {"delta", "better_than_exp235", "exp235_reference_acc"} <= result.keys()


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-089: INCONCLUSIVE when result files are missing
# ---------------------------------------------------------------------------

class TestInconclusiveMissingFiles:
    """SCENARIO-VERIFY-089: INCONCLUSIVE classification when Exp 282/283 files absent."""

    def test_missing_both_files(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-089: both files missing → INCONCLUSIVE, missing_artifacts listed."""
        result = load_exp_results(
            exp282_path=tmp_path / "nope_282.json",
            exp283_path=tmp_path / "nope_283.json",
        )
        assert result["classification"] == "INCONCLUSIVE"
        assert len(result["missing_artifacts"]) == 2

    def test_missing_283_only(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-089: Exp 283 missing → INCONCLUSIVE."""
        p282 = tmp_path / "experiment_282_results.json"
        p282.write_text(json.dumps(_FAKE_282))
        result = load_exp_results(
            exp282_path=p282,
            exp283_path=tmp_path / "nope_283.json",
        )
        assert result["classification"] == "INCONCLUSIVE"
        # The function stores path.name, so the name of the missing file is reported.
        assert "nope_283.json" in result["missing_artifacts"]

    def test_missing_282_only(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-089: Exp 282 missing → INCONCLUSIVE (no baseline for delta)."""
        p283 = tmp_path / "experiment_283_results.json"
        p283.write_text(json.dumps(_FAKE_283))
        result = load_exp_results(
            exp282_path=tmp_path / "nope_282.json",
            exp283_path=p283,
        )
        assert result["classification"] == "INCONCLUSIVE"

    def test_stall_in_283(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-089: partial=True + stall_at in Exp 283 → INCONCLUSIVE."""
        p282 = tmp_path / "experiment_282_results.json"
        p282.write_text(json.dumps(_FAKE_282))
        stalled_283 = {**_FAKE_283, "partial": True, "stall_at": "Qwen3.5-0.8B:baseline:number_swap:gsm8k-178"}
        # Remove the full results so it looks like a real stall
        stalled_283.pop("results", None)
        stalled_283.pop("improvement_deltas", None)
        stalled_283.pop("primary_criterion_met", None)
        p283 = tmp_path / "experiment_283_results.json"
        p283.write_text(json.dumps(stalled_283))
        result = load_exp_results(exp282_path=p282, exp283_path=p283)
        assert result["classification"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-088: five key questions answered
# ---------------------------------------------------------------------------

class TestAnswerFiveQuestions:
    """SCENARIO-VERIFY-088: five key questions are answered when data is available."""

    def test_all_five_keys_present(self) -> None:
        """SCENARIO-VERIFY-088: answer_five_questions returns all five keys."""
        result = answer_five_questions(exp282=_FAKE_282, exp283=_FAKE_283)
        assert "apple_drop_replicated" in result
        assert "verify_repair_delta_larger_on_swap" in result
        assert "irrelevant_sentence_ignored" in result
        assert "extractor_firing_summary" in result
        assert "dual_model_consistent" in result

    def test_apple_drop_replicated_true(self) -> None:
        """SCENARIO-VERIFY-088, Q1: ≥15 pp drop on number_swap → apple_drop_replicated is True."""
        result = answer_five_questions(exp282=_FAKE_282, exp283=_FAKE_283)
        assert result["apple_drop_replicated"] is True

    def test_verify_repair_larger_on_swap_true(self) -> None:
        """SCENARIO-VERIFY-088, Q2: primary_criterion_met in Exp 283 → verify_repair_delta_larger_on_swap True."""
        result = answer_five_questions(exp282=_FAKE_282, exp283=_FAKE_283)
        assert result["verify_repair_delta_larger_on_swap"] is True

    def test_irrelevant_sentence_ignored(self) -> None:
        """SCENARIO-VERIFY-088, Q3: small drop on irrelevant_sentence → irrelevant_sentence_ignored True."""
        result = answer_five_questions(exp282=_FAKE_282, exp283=_FAKE_283)
        # Gemma drop: 0.465 - 0.460 = 0.005 pp < 5 pp — should be ignored
        assert result["irrelevant_sentence_ignored"] is True

    def test_extractor_firing_summary_is_dict(self) -> None:
        """SCENARIO-VERIFY-088, Q4: extractor_firing_summary is a dict."""
        result = answer_five_questions(exp282=_FAKE_282, exp283=_FAKE_283)
        assert isinstance(result["extractor_firing_summary"], dict)

    def test_apple_drop_not_replicated_when_small(self) -> None:
        """SCENARIO-VERIFY-088, Q1: drop < 15 pp → apple_drop_replicated is False."""
        small_drop_282 = {
            **_FAKE_282,
            "apple_2410_05229_check": {
                "Qwen3.5-0.8B": {"number_swap_drop_pp": 8.0, "drop_gte_15pp": False},
                "Gemma4-E4B-it": {"number_swap_drop_pp": 10.0, "drop_gte_15pp": False},
            },
        }
        result = answer_five_questions(exp282=small_drop_282, exp283=_FAKE_283)
        assert result["apple_drop_replicated"] is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-092: artifact schema
# ---------------------------------------------------------------------------

class TestBuildArtifact:
    """SCENARIO-VERIFY-092: build_artifact schema."""

    def test_required_top_level_keys(self) -> None:
        """SCENARIO-VERIFY-092: artifact has all required top-level fields."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=["experiment_282_results.json", "experiment_283_results.json"],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=["Exp 282 and Exp 283 results not found."],
        )
        for key in ARTIFACT_SCHEMA:
            assert key in artifact, f"Missing key: {key}"

    def test_experiment_number(self) -> None:
        """SCENARIO-VERIFY-092: experiment field matches EXPERIMENT constant."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=[],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=[],
        )
        assert artifact["experiment"] == EXPERIMENT

    def test_run_date_matches(self) -> None:
        """SCENARIO-VERIFY-092: run_date field matches RUN_DATE constant."""
        artifact = build_artifact(
            classification="CONFIRMED",
            missing_artifacts=[],
            five_questions={"apple_drop_replicated": True},
            exp235_comparison={"delta": 0.025},
            exp279_comparison={"detection_rate": 1.0},
            analysis_notes=[],
        )
        assert artifact["run_date"] == RUN_DATE

    def test_schema_key_present(self) -> None:
        """SCENARIO-VERIFY-092: schema field identifies artifact type."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=[],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=[],
        )
        assert "schema" in artifact
        assert artifact["schema"] != ""


# ---------------------------------------------------------------------------
# Integration: load_exp_results with valid data produces CONFIRMED
# ---------------------------------------------------------------------------

class TestFullAnalysisWithData:
    """REQ-VERIFY-073: end-to-end analysis with complete fake artifacts."""

    def test_confirmed_classification(self, tmp_path: Path) -> None:
        """REQ-VERIFY-073: when both files exist and primary criterion is met → CONFIRMED."""
        p282 = tmp_path / "experiment_282_results.json"
        p283 = tmp_path / "experiment_283_results.json"
        p282.write_text(json.dumps(_FAKE_282))
        p283.write_text(json.dumps(_FAKE_283))
        result = load_exp_results(exp282_path=p282, exp283_path=p283)
        assert result["classification"] == "CONFIRMED"
        assert result["missing_artifacts"] == []

    def test_ruled_out_when_no_improvement(self, tmp_path: Path) -> None:
        """REQ-VERIFY-073: when verify_repair delta ≤ 0 on number_swap → RULED_OUT."""
        no_improvement_283 = {
            **_FAKE_283,
            "primary_criterion_met": False,
            "improvement_deltas": {
                "Qwen3.5-0.8B": {
                    "verify_repair_number_swap_delta": 0.0,
                    "verify_repair_irrelevant_sentence_delta": 0.0,
                },
                "Gemma4-E4B-it": {
                    "verify_repair_number_swap_delta": 0.0,
                    "verify_repair_irrelevant_sentence_delta": 0.0,
                },
            },
        }
        p282 = tmp_path / "experiment_282_results.json"
        p283 = tmp_path / "experiment_283_results.json"
        p282.write_text(json.dumps(_FAKE_282))
        p283.write_text(json.dumps(no_improvement_283))
        result = load_exp_results(exp282_path=p282, exp283_path=p283)
        assert result["classification"] == "RULED_OUT"

    def test_partial_when_some_improvement(self, tmp_path: Path) -> None:
        """REQ-VERIFY-073: positive delta but primary criterion not met → PARTIAL."""
        partial_283 = {
            **_FAKE_283,
            "primary_criterion_met": False,
            "improvement_deltas": {
                "Qwen3.5-0.8B": {
                    "verify_repair_number_swap_delta": 0.01,
                    "verify_repair_irrelevant_sentence_delta": 0.0,
                },
                "Gemma4-E4B-it": {
                    "verify_repair_number_swap_delta": 0.0,
                    "verify_repair_irrelevant_sentence_delta": 0.0,
                },
            },
        }
        p282 = tmp_path / "experiment_282_results.json"
        p283 = tmp_path / "experiment_283_results.json"
        p282.write_text(json.dumps(_FAKE_282))
        p283.write_text(json.dumps(partial_283))
        result = load_exp_results(exp282_path=p282, exp283_path=p283)
        assert result["classification"] == "PARTIAL"
