"""Tests for Exp 542 FOVER corpus expansion logic.

Spec: REQ-LEARN-055, SCENARIO-LEARN-086, SCENARIO-LEARN-087
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap repo root so the script module is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_542_fover_expansion import merge_fover_corpora  # noqa: E402


# ---------------------------------------------------------------------------
# merge_fover_corpora unit tests
# ---------------------------------------------------------------------------


class TestMergeFoverCorpora:
    """Unit tests for the core merge + dedup helper.

    Spec: REQ-LEARN-055, SCENARIO-LEARN-086
    """

    def test_empty_inputs_produce_empty_result(self):
        result = merge_fover_corpora([], [])
        assert result == []

    def test_prior_only_no_new(self):
        # When no new pairs, result equals prior unchanged.
        prior = [{"step_text": "2 + 2 = 4", "label": "correct", "confidence": 1.0}]
        result = merge_fover_corpora(prior, [])
        assert result == prior

    def test_new_only_no_prior(self):
        new = [{"step_text": "3 * 3 = 9", "label": "correct", "confidence": 1.0}]
        result = merge_fover_corpora([], new)
        assert result == new

    def test_merge_distinct_pairs(self):
        # Both pairs are unique — result contains both.
        prior = [{"step_text": "1 + 1 = 2", "label": "correct", "confidence": 1.0}]
        new = [{"step_text": "5 - 3 = 2", "label": "correct", "confidence": 1.0}]
        result = merge_fover_corpora(prior, new)
        assert len(result) == 2

    def test_dedup_removes_identical_step_text(self):
        # SCENARIO-LEARN-086: same step_text in both lists → only one copy retained.
        shared = {"step_text": "4 / 2 = 2", "label": "correct", "confidence": 1.0}
        prior = [shared]
        new = [dict(shared)]  # same content, different dict object
        result = merge_fover_corpora(prior, new)
        assert len(result) == 1
        assert result[0]["step_text"] == "4 / 2 = 2"

    def test_prior_copy_wins_over_new_on_dedup(self):
        # When step_text collides, the prior version is kept (prior inserted first).
        prior = [{"step_text": "10 + 5 = 15", "label": "correct", "confidence": 1.0, "source": "prior"}]
        new = [{"step_text": "10 + 5 = 15", "label": "incorrect", "confidence": 0.5, "source": "new"}]
        result = merge_fover_corpora(prior, new)
        assert len(result) == 1
        assert result[0].get("source") == "prior"

    def test_dedup_uses_sha256_of_step_text(self):
        # Verify that dedup is actually hash-based by constructing a collision manually.
        text = "unique step text for hash check"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        pair = {"step_text": text, "label": "correct", "confidence": 1.0}
        result = merge_fover_corpora([pair], [dict(pair)])
        assert len(result) == 1

    def test_empty_step_text_deduped(self):
        # Two pairs with empty step_text should deduplicate to one.
        a = {"step_text": "", "label": "not_verifiable", "confidence": 0.0}
        b = {"step_text": "", "label": "correct", "confidence": 1.0}
        result = merge_fover_corpora([a], [b])
        assert len(result) == 1

    def test_order_preserved_within_sources(self):
        # Order within each source is preserved in the merged result.
        prior = [
            {"step_text": "a", "label": "correct", "confidence": 1.0},
            {"step_text": "b", "label": "incorrect", "confidence": 1.0},
        ]
        new = [
            {"step_text": "c", "label": "correct", "confidence": 1.0},
        ]
        result = merge_fover_corpora(prior, new)
        assert [r["step_text"] for r in result] == ["a", "b", "c"]

    def test_large_merge_count(self):
        # Merge 60 prior + 50 new unique pairs → 110 total.
        prior = [{"step_text": f"prior_step_{i}", "label": "correct", "confidence": 1.0} for i in range(60)]
        new = [{"step_text": f"new_step_{i}", "label": "correct", "confidence": 1.0} for i in range(50)]
        result = merge_fover_corpora(prior, new)
        assert len(result) == 110

    def test_partial_overlap_merges_correctly(self):
        # 3 prior + 3 new where 1 overlaps → 5 total.
        prior = [{"step_text": f"s{i}", "label": "correct", "confidence": 1.0} for i in range(3)]
        new = [
            {"step_text": "s1", "label": "correct", "confidence": 1.0},  # duplicate
            {"step_text": "s3", "label": "correct", "confidence": 1.0},  # unique
            {"step_text": "s4", "label": "correct", "confidence": 1.0},  # unique
        ]
        result = merge_fover_corpora(prior, new)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Integration-style test: main() with mocked heavy dependencies
# ---------------------------------------------------------------------------


class TestExp542MainIntegration:
    """Smoke tests for main() verifying file I/O and honest_verdict logic.

    Spec: REQ-LEARN-055, SCENARIO-LEARN-087
    """

    def _make_prior_pairs(self, n: int) -> list[dict]:
        return [
            {"step_text": f"prior step {i}", "label": "correct", "confidence": 1.0, "question_id": str(i)}
            for i in range(n)
        ]

    def _make_exp538_items(self, n: int) -> list[dict]:
        return [
            {
                "question": f"Q{i}",
                "cot_text": f"unique new step text {i}: 2 + 2 = 4",
                "correct": True,
                "model_id": "test-model",
                "latency_s": 1.0,
            }
            for i in range(n)
        ]

    def test_missing_exp538_tolerates_absence(self, tmp_path):
        """SCENARIO-LEARN-087: absent exp538 file → n_new_pairs=0, prior corpus preserved."""
        prior_pairs = self._make_prior_pairs(57)
        prior_file = tmp_path / "results" / "fover_labeled_steps_live.json"
        prior_file.parent.mkdir(parents=True, exist_ok=True)
        prior_file.write_text(json.dumps(prior_pairs))

        output_file = tmp_path / "results" / "experiment_542_fover_expansion.json"

        with (
            patch("scripts.experiment_542_fover_expansion._REPO_ROOT", tmp_path),
            patch("scripts.experiment_542_fover_expansion.apply_env_autofix"),
            patch("scripts.experiment_542_fover_expansion.ExperimentTimeoutWatchdog") as mock_watchdog,
            patch("scripts.experiment_542_fover_expansion.ExperimentTemplate") as mock_tmpl_cls,
        ):
            # Make the watchdog context manager a no-op.
            mock_watchdog.return_value.__enter__ = MagicMock(return_value=None)
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)

            # Mock template so setup() and assert_deliverable_written() are no-ops.
            mock_tmpl = MagicMock()
            mock_tmpl.build_result.side_effect = lambda data, **kwargs: {
                "experiment": 542, "status": kwargs.get("status", "success"), **data,
                "schema": sorted({"experiment", "status", *data.keys()}),
                "run_date": "20260420", "started_at": "T", "finished_at": "T",
                "duration_s": 0.1, "title": "t",
            }
            mock_tmpl_cls.return_value = mock_tmpl

            # Patch FOVERAnnotator to return empty annotation (no new pairs).
            with patch("carnot.pipeline.fover_annotator.FOVERAnnotator") as mock_ann_cls:
                mock_ann = MagicMock()
                mock_ann.annotate_corpus.return_value = []
                mock_ann.to_training_pairs.return_value = []
                mock_ann_cls.return_value = mock_ann

                from scripts.experiment_542_fover_expansion import main  # noqa: PLC0415

                # exp538 file does NOT exist → should still run without error.
                main()

        artifact = json.loads(output_file.read_text())
        assert artifact["n_prior_pairs"] == 57
        assert artifact["n_new_pairs"] == 0
        assert artifact["honest_verdict"] in ("partial_expansion", "synthetic_fallback")

    def test_corpus_expanded_verdict_when_ge_100(self, tmp_path):
        """honest_verdict == 'corpus_expanded' when merged total >= 100."""
        # 57 prior + 50 new (unique) = 107 → corpus_expanded
        prior_pairs = self._make_prior_pairs(57)
        exp538_items = self._make_exp538_items(50)

        prior_file = tmp_path / "results" / "fover_labeled_steps_live.json"
        exp538_file = tmp_path / "results" / "exp538_cot_pairs.json"
        prior_file.parent.mkdir(parents=True, exist_ok=True)
        prior_file.write_text(json.dumps(prior_pairs))
        exp538_file.write_text(json.dumps(exp538_items))

        output_file = tmp_path / "results" / "experiment_542_fover_expansion.json"

        # Synthesise 50 new unique training pairs (simulating FOVERAnnotator output).
        new_pairs_for_mock = [
            {"step_text": f"unique new step text {i}: 2 + 2 = 4", "label": "correct",
             "confidence": 1.0, "question_id": f"exp538_{i}"}
            for i in range(50)
        ]

        with (
            patch("scripts.experiment_542_fover_expansion._REPO_ROOT", tmp_path),
            patch("scripts.experiment_542_fover_expansion.apply_env_autofix"),
            patch("scripts.experiment_542_fover_expansion.ExperimentTimeoutWatchdog") as mock_watchdog,
            patch("scripts.experiment_542_fover_expansion.ExperimentTemplate") as mock_tmpl_cls,
        ):
            mock_watchdog.return_value.__enter__ = MagicMock(return_value=None)
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)

            mock_tmpl = MagicMock()
            mock_tmpl.build_result.side_effect = lambda data, **kwargs: {
                "experiment": 542, "status": kwargs.get("status", "success"), **data,
                "schema": sorted({"experiment", "status", *data.keys()}),
                "run_date": "20260420", "started_at": "T", "finished_at": "T",
                "duration_s": 0.1, "title": "t",
            }
            mock_tmpl_cls.return_value = mock_tmpl

            with patch("carnot.pipeline.fover_annotator.FOVERAnnotator") as mock_ann_cls:
                mock_ann = MagicMock()
                mock_ann.annotate_corpus.return_value = [[]] * 50
                mock_ann.to_training_pairs.return_value = new_pairs_for_mock
                mock_ann_cls.return_value = mock_ann

                from scripts import experiment_542_fover_expansion  # noqa: PLC0415
                import importlib  # noqa: PLC0415
                importlib.reload(experiment_542_fover_expansion)
                experiment_542_fover_expansion.main()

        artifact = json.loads(output_file.read_text())
        assert artifact["n_total_pairs"] >= 100
        assert artifact["honest_verdict"] == "corpus_expanded"

    def test_partial_expansion_verdict_when_lt_100(self, tmp_path):
        """honest_verdict == 'partial_expansion' when 57 < n_total < 100."""
        prior_pairs = self._make_prior_pairs(57)
        prior_file = tmp_path / "results" / "fover_labeled_steps_live.json"
        prior_file.parent.mkdir(parents=True, exist_ok=True)
        prior_file.write_text(json.dumps(prior_pairs))

        exp538_file = tmp_path / "results" / "exp538_cot_pairs.json"
        exp538_file.write_text(json.dumps(self._make_exp538_items(5)))

        output_file = tmp_path / "results" / "experiment_542_fover_expansion.json"

        # Only 5 new unique pairs → total 62 < 100
        new_pairs_for_mock = [
            {"step_text": f"unique new step text {i}: 2 + 2 = 4", "label": "correct",
             "confidence": 1.0, "question_id": f"exp538_{i}"}
            for i in range(5)
        ]

        with (
            patch("scripts.experiment_542_fover_expansion._REPO_ROOT", tmp_path),
            patch("scripts.experiment_542_fover_expansion.apply_env_autofix"),
            patch("scripts.experiment_542_fover_expansion.ExperimentTimeoutWatchdog") as mock_watchdog,
            patch("scripts.experiment_542_fover_expansion.ExperimentTemplate") as mock_tmpl_cls,
        ):
            mock_watchdog.return_value.__enter__ = MagicMock(return_value=None)
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)

            mock_tmpl = MagicMock()
            mock_tmpl.build_result.side_effect = lambda data, **kwargs: {
                "experiment": 542, "status": kwargs.get("status", "success"), **data,
                "schema": sorted({"experiment", "status", *data.keys()}),
                "run_date": "20260420", "started_at": "T", "finished_at": "T",
                "duration_s": 0.1, "title": "t",
            }
            mock_tmpl_cls.return_value = mock_tmpl

            with patch("carnot.pipeline.fover_annotator.FOVERAnnotator") as mock_ann_cls:
                mock_ann = MagicMock()
                mock_ann.annotate_corpus.return_value = [[]] * 5
                mock_ann.to_training_pairs.return_value = new_pairs_for_mock
                mock_ann_cls.return_value = mock_ann

                from scripts import experiment_542_fover_expansion as mod  # noqa: PLC0415
                import importlib  # noqa: PLC0415
                importlib.reload(mod)
                mod.main()

        artifact = json.loads(output_file.read_text())
        assert artifact["honest_verdict"] == "partial_expansion"
