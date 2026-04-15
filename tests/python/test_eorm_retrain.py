"""Tests for python/carnot/models/eorm_retrain.py — 100% coverage required.

Spec coverage: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.eorm_retrain import (
    EORMRetrainResult,
    build_retrain_artifact,
    load_real_cot_pairs,
    make_synthetic_eorm_pairs,
    merge_cot_corpora,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as JSON to a temp file."""
    with open(path, "w") as f:
        json.dump(data, f)


def _gsm8k_result(n: int = 5) -> dict:
    """Build a minimal Exp 340/355-style GSM8K result dict with n responses."""
    return {
        "responses": [
            {
                "question_id": f"q{i:03d}",
                "model_id": "test_model",
                "response": f"The answer is {i}. Because {i}+{i}={i*2}.",
                "correct": i % 2 == 0,
            }
            for i in range(n)
        ]
    }


def _humaneval_result(n: int = 5) -> dict:
    """Build a minimal Exp 341-style HumanEval result dict with n entries."""
    return {
        "per_problem_results": [
            {
                "problem_id": f"HumanEval/{i}",
                "generated_code": f"def foo_{i}(x): return x + {i}",
                "passed_tests": i % 2 == 0,
            }
            for i in range(n)
        ]
    }


# ---------------------------------------------------------------------------
# load_real_cot_pairs
# ---------------------------------------------------------------------------


class TestLoadRealCotPairs:
    """Tests for REQ-LEARN-025-1, SCENARIO-LEARN-043."""

    def test_missing_file_skipped(self) -> None:
        """SCENARIO-LEARN-043: Missing file is skipped silently."""
        result = load_real_cot_pairs(["/nonexistent/path/exp999.json"])
        assert result == []

    def test_empty_list_returns_empty(self) -> None:
        """Empty file list returns empty list."""
        assert load_real_cot_pairs([]) == []

    def test_gsm8k_layout_loaded(self, tmp_path: Path) -> None:
        """REQ-LEARN-025-1: GSM8K-style 'responses' entries are extracted."""
        f = tmp_path / "exp340.json"
        _write_json(f, _gsm8k_result(n=6))
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 6
        # Correctness labeling: correct=True → has_violation=False
        for i, p in enumerate(pairs):
            assert p.has_violation == (i % 2 != 0)
        assert all(isinstance(p, ViolationPair) for p in pairs)

    def test_gsm8k_model_and_question_ids(self, tmp_path: Path) -> None:
        """model_id and question_id are preserved from GSM8K entries."""
        f = tmp_path / "exp340.json"
        _write_json(f, _gsm8k_result(n=3))
        pairs = load_real_cot_pairs([str(f)])
        assert pairs[0].model_id == "test_model"
        assert pairs[0].question_id == "q000"

    def test_humaneval_layout_loaded(self, tmp_path: Path) -> None:
        """REQ-LEARN-025-1: HumanEval-style 'per_problem_results' entries are extracted."""
        f = tmp_path / "exp341.json"
        _write_json(f, _humaneval_result(n=4))
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 4
        for i, p in enumerate(pairs):
            assert p.has_violation == (i % 2 != 0)
        assert pairs[0].model_id == "humaneval_unknown"
        assert pairs[0].question_id == "HumanEval/0"

    def test_humaneval_full_response_is_code(self, tmp_path: Path) -> None:
        """HumanEval: full_response and partial_response are the generated code."""
        f = tmp_path / "exp341.json"
        _write_json(f, _humaneval_result(n=2))
        pairs = load_real_cot_pairs([str(f)])
        assert "def foo_0" in pairs[0].full_response
        assert pairs[0].partial_response == pairs[0].full_response

    def test_missing_response_field_skipped(self, tmp_path: Path) -> None:
        """Entries without 'response' field are skipped."""
        f = tmp_path / "exp340.json"
        _write_json(f, {"responses": [{"question_id": "q001", "correct": True}]})
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_empty_response_field_skipped(self, tmp_path: Path) -> None:
        """Entries with empty response string are skipped."""
        f = tmp_path / "exp340.json"
        _write_json(
            f,
            {
                "responses": [
                    {"question_id": "q001", "response": "", "correct": True},
                    {"question_id": "q002", "response": "real answer", "correct": False},
                ]
            },
        )
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 1
        assert pairs[0].question_id == "q002"

    def test_none_response_field_skipped(self, tmp_path: Path) -> None:
        """Entries with response=None are skipped."""
        f = tmp_path / "exp340.json"
        _write_json(
            f,
            {
                "responses": [
                    {"question_id": "q001", "response": None, "correct": True},
                ]
            },
        )
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_missing_generated_code_skipped(self, tmp_path: Path) -> None:
        """HumanEval entries without generated_code are skipped."""
        f = tmp_path / "exp341.json"
        _write_json(f, {"per_problem_results": [{"problem_id": "HumanEval/0", "passed_tests": True}]})
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_empty_generated_code_skipped(self, tmp_path: Path) -> None:
        """HumanEval entries with empty generated_code are skipped."""
        f = tmp_path / "exp341.json"
        _write_json(
            f,
            {
                "per_problem_results": [
                    {"problem_id": "HumanEval/0", "generated_code": "", "passed_tests": True},
                ]
            },
        )
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_multiple_files_concatenated(self, tmp_path: Path) -> None:
        """Pairs from multiple files are concatenated into one list."""
        f1 = tmp_path / "exp340.json"
        f2 = tmp_path / "exp341.json"
        _write_json(f1, _gsm8k_result(n=3))
        _write_json(f2, _humaneval_result(n=2))
        pairs = load_real_cot_pairs([str(f1), str(f2)])
        assert len(pairs) == 5

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        """Files with invalid JSON content are skipped."""
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_non_dict_responses_entry_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries in responses list are skipped."""
        f = tmp_path / "exp340.json"
        _write_json(f, {"responses": ["not a dict", None, 42]})
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_non_dict_per_problem_entry_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries in per_problem_results list are skipped."""
        f = tmp_path / "exp341.json"
        _write_json(f, {"per_problem_results": ["bad", None]})
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_missing_keys_use_defaults(self, tmp_path: Path) -> None:
        """Missing question_id and model_id fall back to 'unknown'."""
        f = tmp_path / "exp340.json"
        _write_json(f, {"responses": [{"response": "some answer", "correct": True}]})
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 1
        assert pairs[0].model_id == "unknown"
        assert pairs[0].question_id == "unknown"

    def test_missing_correct_defaults_false(self, tmp_path: Path) -> None:
        """Missing 'correct' field defaults to False → has_violation=True."""
        f = tmp_path / "exp340.json"
        _write_json(f, {"responses": [{"question_id": "q1", "response": "answer"}]})
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 1
        assert pairs[0].has_violation is True  # not False = True

    def test_missing_passed_tests_defaults_false(self, tmp_path: Path) -> None:
        """Missing 'passed_tests' defaults to False → has_violation=True."""
        f = tmp_path / "exp341.json"
        _write_json(f, {"per_problem_results": [{"problem_id": "HE/0", "generated_code": "code"}]})
        pairs = load_real_cot_pairs([str(f)])
        assert len(pairs) == 1
        assert pairs[0].has_violation is True

    def test_schema_without_responses_or_per_problem_returns_empty(self, tmp_path: Path) -> None:
        """Files with no recognised key return empty list."""
        f = tmp_path / "other.json"
        _write_json(f, {"status": "simulated", "inference_mode": "simulated"})
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_simulated_mode_file_returns_empty(self, tmp_path: Path) -> None:
        """A real-world simulated experiment file (status=simulated) returns empty list."""
        f = tmp_path / "exp355.json"
        _write_json(
            f,
            {
                "status": "simulated",
                "inference_mode": "simulated",
                "per_model_results": [
                    {
                        "model_id": "gemma4",
                        "n_questions": 100,
                        "standard_accuracy": 0.5,
                    }
                ],
            },
        )
        # per_model_results does not match either schema; should return empty
        pairs = load_real_cot_pairs([str(f)])
        assert pairs == []

    def test_full_response_equals_partial_for_gsm8k(self, tmp_path: Path) -> None:
        """For GSM8K pairs, partial_response equals full_response (EORM uses full text)."""
        f = tmp_path / "exp340.json"
        _write_json(f, _gsm8k_result(n=1))
        pairs = load_real_cot_pairs([str(f)])
        assert pairs[0].partial_response == pairs[0].full_response

    def test_os_error_on_read_skipped(self, tmp_path: Path) -> None:
        """A path that exists but causes OSError during open is skipped."""
        # Use a directory path instead of a file — open() will raise IsADirectoryError (subclass of OSError)
        d = tmp_path / "a_dir"
        d.mkdir()
        pairs = load_real_cot_pairs([str(d)])
        assert pairs == []


# ---------------------------------------------------------------------------
# merge_cot_corpora
# ---------------------------------------------------------------------------


class TestMergeCotCorpora:
    """Tests for REQ-LEARN-025-2, SCENARIO-LEARN-044."""

    def _make_pairs(self, n: int, prefix: str = "real") -> list[ViolationPair]:
        return [
            ViolationPair(
                partial_response=f"{prefix}_response_{i}",
                full_response=f"{prefix}_full_{i}",
                has_violation=i % 2 == 0,
                model_id=f"{prefix}_model",
                question_id=f"{prefix}_q{i}",
            )
            for i in range(n)
        ]

    def test_real_pairs_come_first(self) -> None:
        """SCENARIO-LEARN-044: Real pairs appear before synthetic in the merged list."""
        real = self._make_pairs(5, "real")
        synth = self._make_pairs(5, "synth")
        merged = merge_cot_corpora(real, synth, max_real=10, max_synthetic=10)
        assert [p.model_id for p in merged[:5]] == ["real_model"] * 5
        assert [p.model_id for p in merged[5:]] == ["synth_model"] * 5

    def test_total_length_with_real_below_max(self) -> None:
        """SCENARIO-LEARN-044: 30 real + 100 synth = 130 total."""
        real = self._make_pairs(30, "real")
        synth = self._make_pairs(100, "synth")
        merged = merge_cot_corpora(real, synth, max_real=300, max_synthetic=100)
        assert len(merged) == 130

    def test_real_capped_at_max_real(self) -> None:
        """Real pairs are capped at max_real."""
        real = self._make_pairs(500, "real")
        synth = self._make_pairs(10, "synth")
        merged = merge_cot_corpora(real, synth, max_real=300, max_synthetic=100)
        assert len([p for p in merged if p.model_id == "real_model"]) == 300

    def test_synthetic_capped_at_max_synthetic(self) -> None:
        """Synthetic pairs are capped at max_synthetic."""
        real = self._make_pairs(5, "real")
        synth = self._make_pairs(200, "synth")
        merged = merge_cot_corpora(real, synth, max_real=300, max_synthetic=100)
        assert len([p for p in merged if p.model_id == "synth_model"]) == 100
        assert len(merged) == 105  # 5 real + 100 synth

    def test_empty_real_uses_synthetic_only(self) -> None:
        """When no real pairs exist, synthetic fills the corpus up to max_synthetic."""
        synth = self._make_pairs(50, "synth")
        merged = merge_cot_corpora([], synth, max_real=300, max_synthetic=100)
        assert len(merged) == 50
        assert all(p.model_id == "synth_model" for p in merged)

    def test_empty_synthetic_uses_real_only(self) -> None:
        """When no synthetic pairs exist, only real pairs are returned."""
        real = self._make_pairs(10, "real")
        merged = merge_cot_corpora(real, [], max_real=300, max_synthetic=100)
        assert len(merged) == 10
        assert all(p.model_id == "real_model" for p in merged)

    def test_both_empty_returns_empty(self) -> None:
        """Both empty inputs yield an empty merged list."""
        assert merge_cot_corpora([], [], max_real=300, max_synthetic=100) == []

    def test_default_caps(self) -> None:
        """Default max_real=300, max_synthetic=100 are applied."""
        real = self._make_pairs(400, "real")
        synth = self._make_pairs(200, "synth")
        merged = merge_cot_corpora(real, synth)
        assert len(merged) == 400  # 300 real + 100 synth

    def test_real_pairs_unchanged_content(self) -> None:
        """Real pairs' content is not modified by the merge."""
        real = self._make_pairs(3, "real")
        synth = self._make_pairs(2, "synth")
        merged = merge_cot_corpora(real, synth, max_real=10, max_synthetic=10)
        for orig, merged_p in zip(real, merged[:3]):
            assert orig.full_response == merged_p.full_response
            assert orig.has_violation == merged_p.has_violation


# ---------------------------------------------------------------------------
# EORMRetrainResult
# ---------------------------------------------------------------------------


class TestEORMRetrainResult:
    """Tests for REQ-LEARN-025-3: EORMRetrainResult dataclass."""

    def test_fields_stored(self) -> None:
        """All fields are stored and accessible."""
        r = EORMRetrainResult(
            n_real_pairs=10,
            n_synthetic_pairs=5,
            before_auc=0.6,
            after_auc=0.75,
            auc_improvement=0.15,
            retrain_mode="real_data",
            model_path="/tmp/model.safetensors",
        )
        assert r.n_real_pairs == 10
        assert r.n_synthetic_pairs == 5
        assert r.before_auc == pytest.approx(0.6)
        assert r.after_auc == pytest.approx(0.75)
        assert r.auc_improvement == pytest.approx(0.15)
        assert r.retrain_mode == "real_data"
        assert r.model_path == "/tmp/model.safetensors"

    def test_synthetic_only_mode(self) -> None:
        """retrain_mode='synthetic_only' is valid."""
        r = EORMRetrainResult(
            n_real_pairs=0,
            n_synthetic_pairs=100,
            before_auc=0.5,
            after_auc=0.55,
            auc_improvement=0.05,
            retrain_mode="synthetic_only",
            model_path="/tmp/synth.safetensors",
        )
        assert r.retrain_mode == "synthetic_only"
        assert r.n_real_pairs == 0

    def test_negative_improvement_stored(self) -> None:
        """Negative auc_improvement (retrain degraded the model) is stored honestly."""
        r = EORMRetrainResult(
            n_real_pairs=0,
            n_synthetic_pairs=50,
            before_auc=0.7,
            after_auc=0.65,
            auc_improvement=-0.05,
            retrain_mode="synthetic_only",
            model_path="",
        )
        assert r.auc_improvement < 0


# ---------------------------------------------------------------------------
# build_retrain_artifact
# ---------------------------------------------------------------------------


class TestBuildRetrainArtifact:
    """Tests for REQ-LEARN-025-4: build_retrain_artifact."""

    def test_schema_key(self) -> None:
        """Artifact has schema='carnot.eorm_retrain.v1'."""
        r = EORMRetrainResult(0, 50, 0.5, 0.55, 0.05, "synthetic_only", "")
        art = build_retrain_artifact(r)
        assert art["schema"] == "carnot.eorm_retrain.v1"

    def test_synthetic_only_verdict(self) -> None:
        """REQ-LEARN-025-4: synthetic_only retrain_mode → honest_verdict='synthetic_only'."""
        r = EORMRetrainResult(0, 100, 0.5, 0.6, 0.1, "synthetic_only", "")
        art = build_retrain_artifact(r)
        assert art["honest_verdict"] == "synthetic_only"

    def test_real_data_improvement_verdict(self) -> None:
        """REQ-LEARN-025-4: real_data + positive improvement → 'real_data_improvement'."""
        r = EORMRetrainResult(80, 20, 0.6, 0.75, 0.15, "real_data", "")
        art = build_retrain_artifact(r)
        assert art["honest_verdict"] == "real_data_improvement"

    def test_real_data_no_improvement_verdict(self) -> None:
        """REQ-LEARN-025-4: real_data + zero/negative improvement → 'real_data_no_improvement'."""
        r_zero = EORMRetrainResult(80, 20, 0.7, 0.7, 0.0, "real_data", "")
        r_neg = EORMRetrainResult(80, 20, 0.7, 0.65, -0.05, "real_data", "")
        assert build_retrain_artifact(r_zero)["honest_verdict"] == "real_data_no_improvement"
        assert build_retrain_artifact(r_neg)["honest_verdict"] == "real_data_no_improvement"

    def test_all_required_keys_present(self) -> None:
        """All required keys are present in the artifact."""
        r = EORMRetrainResult(10, 90, 0.6, 0.65, 0.05, "real_data", "/tmp/model.safetensors")
        art = build_retrain_artifact(r)
        for key in [
            "schema",
            "retrain_mode",
            "n_real_pairs",
            "n_synthetic_pairs",
            "before_auc",
            "after_auc",
            "auc_improvement",
            "honest_verdict",
            "model_path",
        ]:
            assert key in art, f"Missing key: {key}"

    def test_auc_values_rounded_to_6dp(self) -> None:
        """AUC values are rounded to 6 decimal places."""
        r = EORMRetrainResult(0, 50, 0.123456789, 0.987654321, 0.864197532, "synthetic_only", "")
        art = build_retrain_artifact(r)
        assert art["before_auc"] == 0.123457
        assert art["after_auc"] == 0.987654
        assert art["auc_improvement"] == 0.864198

    def test_n_pairs_are_integers(self) -> None:
        """n_real_pairs and n_synthetic_pairs are stored as int."""
        r = EORMRetrainResult(10, 90, 0.5, 0.5, 0.0, "synthetic_only", "")
        art = build_retrain_artifact(r)
        assert isinstance(art["n_real_pairs"], int)
        assert isinstance(art["n_synthetic_pairs"], int)

    def test_retrain_mode_propagated(self) -> None:
        """retrain_mode from result is propagated to artifact unchanged."""
        for mode in ("real_data", "synthetic_only"):
            auc_imp = 0.1 if mode == "real_data" else 0.05
            r = EORMRetrainResult(50 if mode == "real_data" else 0, 50, 0.5, 0.6, auc_imp, mode, "")
            art = build_retrain_artifact(r)
            assert art["retrain_mode"] == mode

    def test_model_path_propagated(self) -> None:
        """model_path is stored as string in artifact."""
        r = EORMRetrainResult(0, 50, 0.5, 0.5, 0.0, "synthetic_only", "/some/path.safetensors")
        art = build_retrain_artifact(r)
        assert art["model_path"] == "/some/path.safetensors"


# ---------------------------------------------------------------------------
# make_synthetic_eorm_pairs
# ---------------------------------------------------------------------------


class TestMakeSyntheticEormPairs:
    """Tests for make_synthetic_eorm_pairs (re-exported helper)."""

    def test_returns_n_pairs(self) -> None:
        """Returns exactly n pairs."""
        pairs = make_synthetic_eorm_pairs(n=20, seed=359)
        assert len(pairs) == 20

    def test_deterministic(self) -> None:
        """Same seed produces same output."""
        p1 = make_synthetic_eorm_pairs(n=10, seed=359)
        p2 = make_synthetic_eorm_pairs(n=10, seed=359)
        assert [p.full_response for p in p1] == [p.full_response for p in p2]

    def test_different_seed_different_output(self) -> None:
        """Different seed produces different pairs (with overwhelming probability)."""
        p1 = make_synthetic_eorm_pairs(n=10, seed=359)
        p2 = make_synthetic_eorm_pairs(n=10, seed=42)
        assert [p.full_response for p in p1] != [p.full_response for p in p2]

    def test_all_are_violation_pairs(self) -> None:
        """All returned objects are ViolationPair instances."""
        pairs = make_synthetic_eorm_pairs(n=10)
        assert all(isinstance(p, ViolationPair) for p in pairs)

    def test_default_n_is_100(self) -> None:
        """Default n=100."""
        pairs = make_synthetic_eorm_pairs()
        assert len(pairs) == 100
