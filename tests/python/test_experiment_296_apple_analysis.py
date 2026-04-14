"""Tests for scripts/experiment_296_apple_analysis.py.

Validates the analysis functions and artifact schema for Exp 296: Apple
adversarial GSM8K result analysis (Exp 294 baseline + Exp 295 verify-repair).
All tests run without GPU hardware and without requiring Exp 294/295 result
files to be present.

Spec coverage:
  REQ-VERIFY-080  — Apple adversarial analysis v2 and classification (Exps 294/295)
  REQ-VERIFY-081  — compute_delta rounds to four decimal places
  REQ-VERIFY-082  — classify_result maps (primary_met, partial, stall) → label
  SCENARIO-VERIFY-109 — five key questions answered in artifact
  SCENARIO-VERIFY-110 — INCONCLUSIVE when result files are missing
  SCENARIO-VERIFY-111 — compare_vs_exp235 returns comparison dict
  SCENARIO-VERIFY-112 — compute_delta arithmetic
  SCENARIO-VERIFY-113 — artifact schema contains all required top-level keys
  SCENARIO-VERIFY-114 — docs_updated is True only when Exp 295 fully completed
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

_MODULE_NAME = "experiment_296_apple_analysis"
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "experiment_296_apple_analysis.py"
)


def _load_module() -> Any:
    """Load the experiment_296 module without invoking main()."""
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
# Fixture: minimal Exp 294 artifact (baseline)
# ---------------------------------------------------------------------------

_FAKE_294 = {
    "experiment": 294,
    "schema": "apple_adversarial_baseline.v2",
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

# ---------------------------------------------------------------------------
# Fixture: minimal Exp 295 artifact (verify-repair)
# ---------------------------------------------------------------------------

_FAKE_295 = {
    "experiment": 295,
    "schema": "apple_adversarial_verify_repair.v2",
    "run_date": "20260414",
    "started_at": "2026-04-14T05:31:00Z",
    "finished_at": "2026-04-14T06:15:00Z",
    "inference_mode": "live_gpu",
    "partial": False,
    "stall_at": None,
    "results": {
        "Qwen3.5-0.8B": {
            "number_swap": {
                "baseline":      {"accuracy": 0.04,  "correct":  8, "total": 200},
                "verify_only":   {"accuracy": 0.04,  "correct":  8, "total": 200},
                "verify_repair": {"accuracy": 0.065, "correct": 13, "total": 200},
            },
            "irrelevant_sentence": {
                "baseline":      {"accuracy": 0.19,  "correct": 38, "total": 200},
                "verify_only":   {"accuracy": 0.19,  "correct": 38, "total": 200},
                "verify_repair": {"accuracy": 0.195, "correct": 39, "total": 200},
            },
        },
        "Gemma4-E4B-it": {
            "number_swap": {
                "baseline":      {"accuracy": 0.28, "correct": 56, "total": 200},
                "verify_only":   {"accuracy": 0.28, "correct": 56, "total": 200},
                "verify_repair": {"accuracy": 0.33, "correct": 66, "total": 200},
            },
            "irrelevant_sentence": {
                "baseline":      {"accuracy": 0.46,  "correct": 92, "total": 200},
                "verify_only":   {"accuracy": 0.46,  "correct": 92, "total": 200},
                "verify_repair": {"accuracy": 0.465, "correct": 93, "total": 200},
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
# REQ-VERIFY-077 / SCENARIO-VERIFY-096: compute_delta arithmetic
# ---------------------------------------------------------------------------

class TestComputeDelta:
    """REQ-VERIFY-081, SCENARIO-VERIFY-112: compute_delta arithmetic and precision."""

    def test_positive_improvement(self) -> None:
        """SCENARIO-VERIFY-112: mode_acc > baseline_acc → positive delta."""
        assert compute_delta(0.4, 0.5) == pytest.approx(0.1, abs=1e-9)

    def test_zero_delta(self) -> None:
        """SCENARIO-VERIFY-112: equal accuracies → delta is exactly 0."""
        assert compute_delta(0.3, 0.3) == 0.0

    def test_negative_delta(self) -> None:
        """SCENARIO-VERIFY-112: mode_acc < baseline_acc → negative delta."""
        assert compute_delta(0.5, 0.4) == pytest.approx(-0.1, abs=1e-9)

    def test_rounded_to_four_decimal_places(self) -> None:
        """REQ-VERIFY-081: result is rounded to four decimal places."""
        result = compute_delta(1 / 3, 2 / 3)
        assert result == round(result, 4)

    def test_zero_baseline(self) -> None:
        """SCENARIO-VERIFY-096: zero baseline with positive mode_acc."""
        assert compute_delta(0.0, 0.25) == pytest.approx(0.25, abs=1e-9)

    def test_one_minus_one(self) -> None:
        """SCENARIO-VERIFY-096: boundary values at 1.0 and 1.0 produce 0."""
        assert compute_delta(1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# REQ-VERIFY-078 / SCENARIO-VERIFY-097: classify_result branches
# ---------------------------------------------------------------------------

class TestClassifyResult:
    """REQ-VERIFY-078: classify_result maps flags to exactly one of four labels."""

    def test_stall_always_inconclusive(self) -> None:
        """REQ-VERIFY-078: stall_detected=True → INCONCLUSIVE regardless of other flags."""
        assert classify_result(True, True, True) == "INCONCLUSIVE"
        assert classify_result(False, False, True) == "INCONCLUSIVE"
        assert classify_result(True, False, True) == "INCONCLUSIVE"

    def test_primary_met_gives_confirmed(self) -> None:
        """REQ-VERIFY-078: primary_met=True, no stall → CONFIRMED."""
        assert classify_result(True, True, False) == "CONFIRMED"
        assert classify_result(True, False, False) == "CONFIRMED"

    def test_partial_improvement_gives_partial(self) -> None:
        """REQ-VERIFY-078: partial_improvement=True, primary not met, no stall → PARTIAL."""
        assert classify_result(False, True, False) == "PARTIAL"

    def test_no_improvement_gives_ruled_out(self) -> None:
        """REQ-VERIFY-078: no improvement, no stall → RULED_OUT."""
        assert classify_result(False, False, False) == "RULED_OUT"

    def test_return_values_are_strings(self) -> None:
        """classify_result always returns a str."""
        for primary in (True, False):
            for partial in (True, False):
                for stall in (True, False):
                    result = classify_result(primary, partial, stall)
                    assert isinstance(result, str)
                    assert result in {"CONFIRMED", "PARTIAL", "RULED_OUT", "INCONCLUSIVE"}


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-094: INCONCLUSIVE when result files are missing
# ---------------------------------------------------------------------------

class TestMissingArtifacts:
    """SCENARIO-VERIFY-094: INCONCLUSIVE when one or both result files are missing."""

    def test_both_missing(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-094: both files absent → INCONCLUSIVE."""
        result = load_exp_results(
            exp294_path=tmp_path / "nope_294.json",
            exp295_path=tmp_path / "nope_295.json",
        )
        assert result["classification"] == "INCONCLUSIVE"
        assert len(result["missing_artifacts"]) == 2

    def test_exp295_missing(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-094: Exp 295 absent → INCONCLUSIVE (can't compute verify-repair delta)."""
        p294 = tmp_path / "experiment_294_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        result = load_exp_results(
            exp294_path=p294,
            exp295_path=tmp_path / "nope_295.json",
        )
        assert result["classification"] == "INCONCLUSIVE"
        assert "nope_295.json" in result["missing_artifacts"]

    def test_exp294_missing(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-094: Exp 294 absent → INCONCLUSIVE (no baseline for delta)."""
        p295 = tmp_path / "experiment_295_results.json"
        p295.write_text(json.dumps(_FAKE_295))
        result = load_exp_results(
            exp294_path=tmp_path / "nope_294.json",
            exp295_path=p295,
        )
        assert result["classification"] == "INCONCLUSIVE"

    def test_stall_in_295(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-094: partial=True + stall_at in Exp 295 → INCONCLUSIVE."""
        p294 = tmp_path / "experiment_294_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        stalled_295 = {
            **_FAKE_295,
            "partial": True,
            "stall_at": "Qwen3.5-0.8B:baseline:number_swap:gsm8k-010",
        }
        stalled_295.pop("results", None)
        stalled_295.pop("improvement_deltas", None)
        stalled_295.pop("primary_criterion_met", None)
        p295 = tmp_path / "experiment_295_results.json"
        p295.write_text(json.dumps(stalled_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-093: five key questions answered
# ---------------------------------------------------------------------------

class TestAnswerFiveQuestions:
    """SCENARIO-VERIFY-093: five key questions are answered when data is available."""

    def test_all_five_keys_present(self) -> None:
        """SCENARIO-VERIFY-093: answer_five_questions returns all five keys."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        assert "apple_drop_replicated" in result
        assert "verify_repair_delta_larger_on_swap" in result
        assert "irrelevant_sentence_ignored" in result
        assert "extractor_firing_summary" in result
        assert "dual_model_consistent" in result

    def test_apple_drop_replicated_true(self) -> None:
        """SCENARIO-VERIFY-093, Q1: ≥15 pp drop → apple_drop_replicated is True."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        assert result["apple_drop_replicated"] is True

    def test_apple_drop_not_replicated_when_small(self) -> None:
        """SCENARIO-VERIFY-093, Q1: drop < 15 pp → apple_drop_replicated is False."""
        small_drop_294 = {
            **_FAKE_294,
            "apple_2410_05229_check": {
                "Qwen3.5-0.8B": {"number_swap_drop_pp": 8.0,  "drop_gte_15pp": False},
                "Gemma4-E4B-it": {"number_swap_drop_pp": 10.0, "drop_gte_15pp": False},
            },
        }
        result = answer_five_questions(exp294=small_drop_294, exp295=_FAKE_295)
        assert result["apple_drop_replicated"] is False

    def test_verify_repair_larger_on_swap_true(self) -> None:
        """SCENARIO-VERIFY-093, Q2: primary_criterion_met in Exp 295 → True."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        assert result["verify_repair_delta_larger_on_swap"] is True

    def test_verify_repair_larger_on_swap_false_when_absent(self) -> None:
        """SCENARIO-VERIFY-093, Q2: primary_criterion_met absent → False."""
        no_primary = {k: v for k, v in _FAKE_295.items() if k != "primary_criterion_met"}
        result = answer_five_questions(exp294=_FAKE_294, exp295=no_primary)
        assert result["verify_repair_delta_larger_on_swap"] is False

    def test_irrelevant_sentence_ignored_small_drop(self) -> None:
        """SCENARIO-VERIFY-093, Q3: < 5 pp drop on irrelevant_sentence → True."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        # Gemma drop: (0.465 - 0.460)*100 = 0.5 pp < 5 pp
        assert result["irrelevant_sentence_ignored"] is True

    def test_irrelevant_sentence_not_ignored_large_drop(self) -> None:
        """SCENARIO-VERIFY-093, Q3: ≥ 5 pp drop → irrelevant_sentence_ignored False."""
        large_drop_294 = {
            **_FAKE_294,
            "model_results": {
                "Qwen3.5-0.8B": {
                    "standard":            {"accuracy": 0.20,  "correct": 40, "total": 200},
                    "number_swap":         {"accuracy": 0.04,  "correct":  8, "total": 200},
                    "irrelevant_sentence": {"accuracy": 0.14,  "correct": 28, "total": 200},
                },
                "Gemma4-E4B-it": {
                    "standard":            {"accuracy": 0.465, "correct": 93, "total": 200},
                    "number_swap":         {"accuracy": 0.28,  "correct": 56, "total": 200},
                    "irrelevant_sentence": {"accuracy": 0.40,  "correct": 80, "total": 200},
                },
            },
        }
        result = answer_five_questions(exp294=large_drop_294, exp295=_FAKE_295)
        assert result["irrelevant_sentence_ignored"] is False

    def test_extractor_firing_summary_is_dict(self) -> None:
        """SCENARIO-VERIFY-093, Q4: extractor_firing_summary is a dict."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        assert isinstance(result["extractor_firing_summary"], dict)

    def test_extractor_firing_summary_empty_when_absent(self) -> None:
        """SCENARIO-VERIFY-093, Q4: missing extractor_summary → empty dict (not error)."""
        no_extractor = {k: v for k, v in _FAKE_295.items() if k != "extractor_summary"}
        result = answer_five_questions(exp294=_FAKE_294, exp295=no_extractor)
        assert result["extractor_firing_summary"] == {}

    def test_dual_model_consistent_true_when_flag_present(self) -> None:
        """SCENARIO-VERIFY-093, Q5: primary_criterion_met present → dual_model_consistent True."""
        result = answer_five_questions(exp294=_FAKE_294, exp295=_FAKE_295)
        assert result["dual_model_consistent"] is True

    def test_dual_model_consistent_false_when_flag_absent(self) -> None:
        """SCENARIO-VERIFY-093, Q5: primary_criterion_met absent → dual_model_consistent False."""
        no_primary = {k: v for k, v in _FAKE_295.items() if k != "primary_criterion_met"}
        result = answer_five_questions(exp294=_FAKE_294, exp295=no_primary)
        assert result["dual_model_consistent"] is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-095: compare_vs_exp235
# ---------------------------------------------------------------------------

class TestCompareVsExp235:
    """SCENARIO-VERIFY-095: compare_vs_exp235 returns comparison dict."""

    def test_returns_three_keys(self) -> None:
        """SCENARIO-VERIFY-095: returns delta, better_than_exp235, exp235_reference_acc."""
        result = compare_vs_exp235(number_swap_acc=0.5, exp235_acc=0.475)
        assert "delta" in result
        assert "better_than_exp235" in result
        assert "exp235_reference_acc" in result

    def test_positive_delta_is_better(self) -> None:
        """SCENARIO-VERIFY-095: number_swap_acc > exp235_acc → better_than_exp235 True."""
        result = compare_vs_exp235(number_swap_acc=0.5, exp235_acc=0.475)
        assert result["better_than_exp235"] is True
        assert result["delta"] > 0

    def test_negative_delta_is_not_better(self) -> None:
        """SCENARIO-VERIFY-095: number_swap_acc < exp235_acc → better_than_exp235 False."""
        result = compare_vs_exp235(number_swap_acc=0.3, exp235_acc=0.475)
        assert result["better_than_exp235"] is False
        assert result["delta"] < 0

    def test_equal_is_not_better(self) -> None:
        """SCENARIO-VERIFY-095: equal → delta is 0 and better_than_exp235 False."""
        result = compare_vs_exp235(number_swap_acc=0.475, exp235_acc=0.475)
        assert result["better_than_exp235"] is False
        assert result["delta"] == 0.0

    def test_reference_acc_stored(self) -> None:
        """SCENARIO-VERIFY-095: exp235_reference_acc matches input exp235_acc."""
        result = compare_vs_exp235(number_swap_acc=0.3, exp235_acc=0.475)
        assert result["exp235_reference_acc"] == pytest.approx(0.475)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-097: artifact schema
# ---------------------------------------------------------------------------

class TestBuildArtifact:
    """SCENARIO-VERIFY-097: build_artifact schema."""

    def test_required_top_level_keys(self) -> None:
        """SCENARIO-VERIFY-097: artifact has all required top-level fields."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=["experiment_294_results.json", "experiment_295_results.json"],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=["Exp 294 and Exp 295 results not found."],
            docs_updated=False,
        )
        for key in ARTIFACT_SCHEMA:
            assert key in artifact, f"Missing key: {key}"

    def test_experiment_number_is_296(self) -> None:
        """SCENARIO-VERIFY-097: experiment field is 296."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=[],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=[],
            docs_updated=False,
        )
        assert artifact["experiment"] == 296
        assert EXPERIMENT == 296

    def test_schema_is_v2(self) -> None:
        """SCENARIO-VERIFY-097: schema field identifies carnot.apple_analysis.v2."""
        artifact = build_artifact(
            classification="INCONCLUSIVE",
            missing_artifacts=[],
            five_questions=None,
            exp235_comparison=None,
            exp279_comparison=None,
            analysis_notes=[],
            docs_updated=False,
        )
        assert artifact["schema"] == "carnot.apple_analysis.v2"

    def test_run_date_matches(self) -> None:
        """SCENARIO-VERIFY-097: run_date field matches RUN_DATE constant."""
        artifact = build_artifact(
            classification="CONFIRMED",
            missing_artifacts=[],
            five_questions={"apple_drop_replicated": True},
            exp235_comparison={"delta": 0.025},
            exp279_comparison={"detection_rate": 1.0},
            analysis_notes=[],
            docs_updated=True,
        )
        assert artifact["run_date"] == RUN_DATE


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-098: docs_updated field
# ---------------------------------------------------------------------------

class TestDocsUpdated:
    """SCENARIO-VERIFY-098: docs_updated is True only when Exp 295 fully completed."""

    def test_docs_updated_false_when_stall(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-098: partial=True in Exp 295 → docs_updated False in artifact."""
        p294 = tmp_path / "experiment_294_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        stalled_295 = {**_FAKE_295, "partial": True, "stall_at": "Qwen:gsm8k-010"}
        stalled_295.pop("results", None)
        stalled_295.pop("improvement_deltas", None)
        stalled_295.pop("primary_criterion_met", None)
        p295 = tmp_path / "experiment_295_results.json"
        p295.write_text(json.dumps(stalled_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result.get("docs_updated") is False

    def test_docs_updated_false_when_files_missing(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-098: missing files → docs_updated False."""
        result = load_exp_results(
            exp294_path=tmp_path / "nope_294.json",
            exp295_path=tmp_path / "nope_295.json",
        )
        assert result.get("docs_updated") is False

    def test_docs_updated_true_when_complete_and_confirmed(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-098: partial=False on both files and CONFIRMED → docs_updated True."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        p295.write_text(json.dumps(_FAKE_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        # CONFIRMED because primary_criterion_met=True in _FAKE_295
        assert result["classification"] == "CONFIRMED"
        assert result.get("docs_updated") is True

    def test_docs_updated_true_when_complete_and_partial(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-098: fully-run (not stalled) experiment → docs_updated True even if PARTIAL."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        partial_295 = {
            **_FAKE_295,
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
        p295.write_text(json.dumps(partial_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "PARTIAL"
        assert result.get("docs_updated") is True

    def test_docs_updated_false_when_ruled_out(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-098: RULED_OUT → docs_updated False (no verify-repair improvement to report)."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        no_improvement_295 = {
            **_FAKE_295,
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
        p295.write_text(json.dumps(no_improvement_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "RULED_OUT"
        assert result.get("docs_updated") is False


# ---------------------------------------------------------------------------
# Integration: load_exp_results end-to-end with fake data
# ---------------------------------------------------------------------------

class TestFullAnalysisWithData:
    """REQ-VERIFY-076: end-to-end analysis with complete fake artifacts."""

    def test_confirmed_classification(self, tmp_path: Path) -> None:
        """REQ-VERIFY-076: both files present, primary criterion met → CONFIRMED."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        p295.write_text(json.dumps(_FAKE_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "CONFIRMED"
        assert result["missing_artifacts"] == []

    def test_ruled_out_when_no_improvement(self, tmp_path: Path) -> None:
        """REQ-VERIFY-076: verify_repair delta ≤ 0 on number_swap → RULED_OUT."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        no_improvement_295 = {
            **_FAKE_295,
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
        p295.write_text(json.dumps(no_improvement_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "RULED_OUT"

    def test_partial_when_some_improvement(self, tmp_path: Path) -> None:
        """REQ-VERIFY-076: positive delta but primary criterion not met → PARTIAL."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        partial_295 = {
            **_FAKE_295,
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
        p295.write_text(json.dumps(partial_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert result["classification"] == "PARTIAL"

    def test_five_questions_present_on_full_data(self, tmp_path: Path) -> None:
        """REQ-VERIFY-076: full data → five_questions dict with all five keys."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        p295.write_text(json.dumps(_FAKE_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        fq = result["five_questions"]
        assert fq is not None
        for key in (
            "apple_drop_replicated",
            "verify_repair_delta_larger_on_swap",
            "irrelevant_sentence_ignored",
            "extractor_firing_summary",
            "dual_model_consistent",
        ):
            assert key in fq, f"Missing five_questions key: {key}"

    def test_analysis_notes_non_empty_on_full_data(self, tmp_path: Path) -> None:
        """REQ-VERIFY-076: full data → analysis_notes is a non-empty list of strings."""
        p294 = tmp_path / "experiment_294_results.json"
        p295 = tmp_path / "experiment_295_results.json"
        p294.write_text(json.dumps(_FAKE_294))
        p295.write_text(json.dumps(_FAKE_295))
        result = load_exp_results(exp294_path=p294, exp295_path=p295)
        assert isinstance(result["analysis_notes"], list)
        assert len(result["analysis_notes"]) > 0
        for note in result["analysis_notes"]:
            assert isinstance(note, str)
