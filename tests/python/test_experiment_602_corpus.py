"""Tests for Exp 602: Live Corpus Expansion v2 — corpus merging and diversity metrics.

100% targeted coverage on functions added in scripts/experiment_602_live_corpus_expansion_v2.py.
Tests exercise all branches of _merge_live_corpora, _compute_diversity_metrics,
and _build_corpus_artifact without requiring GPU hardware or live model inference.

Spec: REQ-DATA-010, SCENARIO-DATA-019, SCENARIO-DATA-020
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The module-level CARNOT_FORCE_LIVE gate fires on import.
# Set the env var before importing the module so the gate passes.
os.environ["CARNOT_FORCE_LIVE"] = "1"

import scripts.experiment_602_live_corpus_expansion_v2 as exp602  # noqa: E402


# ---------------------------------------------------------------------------
# _build_corpus_artifact
# ---------------------------------------------------------------------------


class TestBuildCorpusArtifact:
    """REQ-DATA-010: artifact must have all required schema fields on every exit path."""

    def _diversity(self) -> dict:
        return {
            "n_unique_questions": 10,
            "n_correct_pairs": 8,
            "n_incorrect_pairs": 2,
            "model_accuracy_qwen": 0.8,
            "model_accuracy_gemma": 0.9,
        }

    def test_schema_field(self):
        # SCENARIO-DATA-020: schema must be carnot.live_corpus_v2.v1
        art = exp602._build_corpus_artifact(100, 300, self._diversity(), "results/fover_corpus_v4.json", "live_gpu")
        assert art["schema"] == "carnot.live_corpus_v2.v1"

    def test_honest_verdict_corpus_expanded_at_80(self):
        # SCENARIO-DATA-020: n_new_pairs >= 80 -> corpus_expanded
        art = exp602._build_corpus_artifact(80, 300, {}, "f.json", "live_gpu")
        assert art["honest_verdict"] == "corpus_expanded"

    def test_honest_verdict_corpus_expanded_above_80(self):
        art = exp602._build_corpus_artifact(100, 300, {}, "f.json", "live_gpu")
        assert art["honest_verdict"] == "corpus_expanded"

    def test_honest_verdict_corpus_partial_below_80(self):
        # SCENARIO-DATA-020: n_new_pairs < 80 -> corpus_partial
        art = exp602._build_corpus_artifact(79, 300, {}, "f.json", "live_gpu")
        assert art["honest_verdict"] == "corpus_partial"

    def test_honest_verdict_corpus_partial_at_zero(self):
        art = exp602._build_corpus_artifact(0, 0, {}, None, "gpu_required")
        assert art["honest_verdict"] == "corpus_partial"

    def test_n_new_pairs_field(self):
        art = exp602._build_corpus_artifact(123, 500, {}, "f.json", "live_gpu")
        assert art["n_new_pairs"] == 123

    def test_n_total_corpus_v4_field(self):
        art = exp602._build_corpus_artifact(100, 456, {}, "f.json", "live_gpu")
        assert art["n_total_corpus_v4"] == 456

    def test_inference_mode_field(self):
        art = exp602._build_corpus_artifact(100, 300, {}, "f.json", "live_gpu")
        assert art["inference_mode"] == "live_gpu"

    def test_fover_corpus_v4_path_field(self):
        art = exp602._build_corpus_artifact(100, 300, {}, "results/fover_corpus_v4.json", "live_gpu")
        assert art["fover_corpus_v4_path"] == "results/fover_corpus_v4.json"

    def test_diversity_metrics_included(self):
        # Diversity metrics are merged into the artifact dict
        diversity = self._diversity()
        art = exp602._build_corpus_artifact(100, 300, diversity, "f.json", "live_gpu")
        assert art["n_unique_questions"] == 10
        assert art["model_accuracy_qwen"] == pytest.approx(0.8)
        assert art["model_accuracy_gemma"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _compute_diversity_metrics
# ---------------------------------------------------------------------------


class TestComputeDiversityMetrics:
    """REQ-DATA-010-3: diversity metrics must be accurate over any corpus."""

    def _make_pair(self, q_idx: int, model: str, is_correct: bool) -> dict:
        return {
            "question_index": q_idx,
            "model": model,
            "is_correct": is_correct,
        }

    def test_empty_corpus(self):
        # Edge case: empty corpus returns zeroed metrics
        m = exp602._compute_diversity_metrics([])
        assert m["n_unique_questions"] == 0
        assert m["n_correct_pairs"] == 0
        assert m["n_incorrect_pairs"] == 0
        assert m["model_accuracy_qwen"] == pytest.approx(0.0)
        assert m["model_accuracy_gemma"] == pytest.approx(0.0)

    def test_unique_question_count(self):
        corpus = [
            self._make_pair(1, "Qwen/Qwen3.5-0.8B", True),
            self._make_pair(1, "google/gemma-4-E4B-it", False),
            self._make_pair(2, "Qwen/Qwen3.5-0.8B", True),
        ]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["n_unique_questions"] == 2

    def test_correct_incorrect_counts(self):
        corpus = [
            self._make_pair(1, "Qwen/Qwen3.5-0.8B", True),
            self._make_pair(1, "google/gemma-4-E4B-it", False),
            self._make_pair(2, "Qwen/Qwen3.5-0.8B", False),
        ]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["n_correct_pairs"] == 1
        assert m["n_incorrect_pairs"] == 2

    def test_qwen_accuracy(self):
        corpus = [
            self._make_pair(1, "Qwen/Qwen3.5-0.8B", True),
            self._make_pair(2, "Qwen/Qwen3.5-0.8B", True),
            self._make_pair(3, "Qwen/Qwen3.5-0.8B", False),
        ]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_qwen"] == pytest.approx(2 / 3)

    def test_gemma_accuracy(self):
        corpus = [
            self._make_pair(1, "google/gemma-4-E4B-it", True),
            self._make_pair(2, "google/gemma-4-E4B-it", False),
        ]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_gemma"] == pytest.approx(0.5)

    def test_qwen_accuracy_zero_when_no_qwen(self):
        corpus = [self._make_pair(1, "google/gemma-4-E4B-it", True)]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_qwen"] == pytest.approx(0.0)

    def test_gemma_accuracy_zero_when_no_gemma(self):
        corpus = [self._make_pair(1, "Qwen/Qwen3.5-0.8B", True)]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_gemma"] == pytest.approx(0.0)

    def test_all_correct_qwen(self):
        corpus = [self._make_pair(i, "Qwen/Qwen3.5-0.8B", True) for i in range(5)]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_qwen"] == pytest.approx(1.0)

    def test_all_incorrect_gemma(self):
        corpus = [self._make_pair(i, "google/gemma-4-E4B-it", False) for i in range(3)]
        m = exp602._compute_diversity_metrics(corpus)
        assert m["model_accuracy_gemma"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _merge_live_corpora
# ---------------------------------------------------------------------------


class TestMergeLiveCorpora:
    """REQ-DATA-010-2: merged corpus must deduplicate by (question_index, model)."""

    def _make_pair(self, q_idx: int, model: str, response: str = "r") -> dict:
        return {
            "question_index": q_idx,
            "model": model,
            "response": response,
            "is_correct": False,
            "inference_mode": "live_gpu",
        }

    def test_no_prior_files_returns_new_pairs(self, tmp_path: Path):
        new_pairs = [
            self._make_pair(250, "Qwen/Qwen3.5-0.8B"),
            self._make_pair(250, "google/gemma-4-E4B-it"),
        ]
        # Use tmp_path as repo root — no prior files exist there
        with patch.object(exp602, "PRIOR_LIVE_PAIR_PATHS", []):
            result = exp602._merge_live_corpora(tmp_path, new_pairs)
        assert len(result) == 2

    def test_deduplication_new_wins(self, tmp_path: Path):
        # When the same (question_index, model) appears in new_pairs and a prior file,
        # the new_pairs entry (higher priority) must win.
        prior_pair = self._make_pair(250, "Qwen/Qwen3.5-0.8B", response="old_response")
        new_pair = self._make_pair(250, "Qwen/Qwen3.5-0.8B", response="new_response")

        prior_path = tmp_path / "results/live_pairs_578.json"
        prior_path.parent.mkdir(parents=True)
        prior_path.write_text(json.dumps([prior_pair]))

        with patch.object(exp602, "PRIOR_LIVE_PAIR_PATHS", ["results/live_pairs_578.json"]):
            result = exp602._merge_live_corpora(tmp_path, [new_pair])

        assert len(result) == 1
        assert result[0]["response"] == "new_response"

    def test_prior_pairs_added_when_no_conflict(self, tmp_path: Path):
        # Prior pairs with different question indices are all included
        prior_pair = self._make_pair(10, "Qwen/Qwen3.5-0.8B")
        new_pair = self._make_pair(250, "Qwen/Qwen3.5-0.8B")

        prior_path = tmp_path / "results/live_pairs_578.json"
        prior_path.parent.mkdir(parents=True)
        prior_path.write_text(json.dumps([prior_pair]))

        with patch.object(exp602, "PRIOR_LIVE_PAIR_PATHS", ["results/live_pairs_578.json"]):
            result = exp602._merge_live_corpora(tmp_path, [new_pair])

        assert len(result) == 2

    def test_missing_prior_file_skipped_gracefully(self, tmp_path: Path):
        # Missing prior file should be skipped without raising an exception
        new_pair = self._make_pair(250, "Qwen/Qwen3.5-0.8B")
        with patch.object(exp602, "PRIOR_LIVE_PAIR_PATHS", ["results/nonexistent.json"]):
            result = exp602._merge_live_corpora(tmp_path, [new_pair])
        assert len(result) == 1

    def test_corrupt_prior_file_skipped_gracefully(self, tmp_path: Path):
        # A corrupt (non-JSON) prior file should be skipped without raising
        prior_path = tmp_path / "results/live_pairs_bad.json"
        prior_path.parent.mkdir(parents=True)
        prior_path.write_text("not valid json {{{")

        new_pair = self._make_pair(250, "Qwen/Qwen3.5-0.8B")
        with patch.object(exp602, "PRIOR_LIVE_PAIR_PATHS", ["results/live_pairs_bad.json"]):
            result = exp602._merge_live_corpora(tmp_path, [new_pair])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    """REQ-DATA-010-2: atomic write must produce a valid, complete JSON file."""

    def test_writes_valid_json(self, tmp_path: Path):
        path = tmp_path / "out.json"
        data = {"key": "value", "n": 42}
        exp602._write_json_atomic(path, data)
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_no_tmp_file_left_after_write(self, tmp_path: Path):
        path = tmp_path / "out.json"
        exp602._write_json_atomic(path, {"x": 1})
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep/nested/dir/out.json"
        exp602._write_json_atomic(path, [1, 2, 3])
        assert path.exists()


# ---------------------------------------------------------------------------
# _load_gsm8k_questions — synthetic fallback
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """REQ-DATA-010-1: question loader must return index-correct synthetic fallback."""

    def test_synthetic_fallback_on_import_error(self):
        # When datasets is unavailable, synthetic questions must cover the requested range
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp602._load_gsm8k_questions(250, 254)
        indices = [q["index"] for q in questions]
        assert indices == [250, 251, 252, 253, 254]

    def test_synthetic_fallback_question_count(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp602._load_gsm8k_questions(300, 349)
        assert len(questions) == 50

    def test_synthetic_fallback_has_answer_field(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp602._load_gsm8k_questions(250, 250)
        assert "answer" in questions[0]
        assert "####" in questions[0]["answer"]


# ---------------------------------------------------------------------------
# _collect_pairs_for_question
# ---------------------------------------------------------------------------


class TestCollectPairsForQuestion:
    """SCENARIO-DATA-019: each question must produce exactly 2 pairs (one per model)."""

    def _make_q(self, idx: int = 250) -> dict:
        return {"index": idx, "question": "What is 2+2?", "answer": "#### 4"}

    def test_returns_two_pairs(self):
        mock_gemma4 = MagicMock()
        mock_gemma4.generate.return_value = "4"
        mock_qwen = MagicMock()
        mock_qwen.return_value = [{"generated_text": "4"}]

        with patch.object(exp602, "_qwen_generate", return_value="4"):
            pairs = exp602._collect_pairs_for_question(self._make_q(), mock_gemma4, mock_qwen)

        assert len(pairs) == 2

    def test_model_ids_in_pairs(self):
        mock_gemma4 = MagicMock()
        mock_gemma4.generate.return_value = "4"

        with patch.object(exp602, "_qwen_generate", return_value="4"):
            pairs = exp602._collect_pairs_for_question(self._make_q(), mock_gemma4, MagicMock())

        models = {p["model"] for p in pairs}
        assert exp602.GEMMA4_MODEL_ID in models
        assert exp602.QWEN_MODEL_ID in models

    def test_inference_mode_is_live_gpu(self):
        mock_gemma4 = MagicMock()
        mock_gemma4.generate.return_value = "18"

        with patch.object(exp602, "_qwen_generate", return_value="18"):
            pairs = exp602._collect_pairs_for_question(self._make_q(), mock_gemma4, MagicMock())

        for p in pairs:
            assert p["inference_mode"] == "live_gpu"

    def test_stub_when_gemma4_none(self):
        # When gemma4 is None, a stub response is recorded rather than raising
        with patch.object(exp602, "_qwen_generate", return_value="4"):
            pairs = exp602._collect_pairs_for_question(self._make_q(), None, MagicMock())
        gemma_pair = next(p for p in pairs if p["model"] == exp602.GEMMA4_MODEL_ID)
        assert gemma_pair["response"] == "[gemma4_not_loaded]"

    def test_stub_when_qwen_none(self):
        # When qwen_pipeline is None, a stub response is recorded
        mock_gemma4 = MagicMock()
        mock_gemma4.generate.return_value = "4"
        pairs = exp602._collect_pairs_for_question(self._make_q(), mock_gemma4, None)
        qwen_pair = next(p for p in pairs if p["model"] == exp602.QWEN_MODEL_ID)
        assert qwen_pair["response"] == "[qwen_not_loaded]"

    def test_question_index_recorded(self):
        mock_gemma4 = MagicMock()
        mock_gemma4.generate.return_value = "4"
        with patch.object(exp602, "_qwen_generate", return_value="4"):
            pairs = exp602._collect_pairs_for_question(self._make_q(idx=275), mock_gemma4, MagicMock())
        for p in pairs:
            assert p["question_index"] == 275
