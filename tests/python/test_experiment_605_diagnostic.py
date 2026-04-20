"""Tests for Exp 605: Live Extractor Diagnostic v4 — gate logic and artifact schema.

100% targeted coverage on functions added in scripts/experiment_605_extractor_diagnostic_v4.py.
Tests exercise _load_test_sets, _load_training_pairs, _build_artifact, _run_coace_v4,
_run_dsvd, and _build_dsvd_adapter without requiring GPU hardware or live model inference.

Spec: REQ-BENCH-058, SCENARIO-BENCH-050, SCENARIO-BENCH-051, SCENARIO-BENCH-052
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
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Prevent GPU assertion from firing in CI (no live GPU in test environment).
os.environ["CARNOT_IS_CI"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_605_extractor_diagnostic_v4 as exp605  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pair(q_idx: int, model: str, is_correct: bool, response: str = "") -> dict:
    """Build a minimal live pair dict for tests."""
    return {
        "question_index": q_idx,
        "model": model,
        "is_correct": is_correct,
        "response": response or ("correct answer" if is_correct else "wrong answer"),
    }


def _write_corpus(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs))


# ---------------------------------------------------------------------------
# _load_test_sets
# ---------------------------------------------------------------------------


class TestLoadTestSets:
    """REQ-BENCH-058-1/2: load_test_sets must return exactly n_incorrect and n_correct."""

    def test_returns_correct_counts(self, tmp_path: Path) -> None:
        # SCENARIO-BENCH-050: first 25 incorrect entries loaded
        pairs = (
            [_make_pair(i, "ModelA", False) for i in range(30)]
            + [_make_pair(i, "ModelB", True) for i in range(15)]
        )
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, pairs)
        incorrect, correct = exp605._load_test_sets(p, n_incorrect=25, n_correct=10)
        assert len(incorrect) == 25
        assert len(correct) == 10

    def test_all_incorrect(self, tmp_path: Path) -> None:
        # Edge case: fewer incorrect entries than n_incorrect requested
        pairs = [_make_pair(i, "M", False) for i in range(5)]
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, pairs)
        incorrect, correct = exp605._load_test_sets(p, n_incorrect=25, n_correct=10)
        assert len(incorrect) == 5
        assert len(correct) == 0

    def test_entries_have_is_correct_false(self, tmp_path: Path) -> None:
        # Every entry in incorrect set must have is_correct=False
        pairs = [_make_pair(i, "M", False) for i in range(30)] + [_make_pair(99, "M", True)]
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, pairs)
        incorrect, _ = exp605._load_test_sets(p, n_incorrect=10, n_correct=1)
        assert all(not e["is_correct"] for e in incorrect)

    def test_entries_have_is_correct_true(self, tmp_path: Path) -> None:
        # Every entry in correct set must have is_correct=True
        pairs = [_make_pair(i, "M", False) for i in range(30)] + [
            _make_pair(i + 30, "M", True) for i in range(15)
        ]
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, pairs)
        _, correct = exp605._load_test_sets(p, n_incorrect=5, n_correct=10)
        assert all(e["is_correct"] for e in correct)

    def test_stops_early_when_both_filled(self, tmp_path: Path) -> None:
        # Should stop as soon as both quotas are met (not scan the whole file)
        pairs = (
            [_make_pair(i, "M", False) for i in range(30)]
            + [_make_pair(i + 30, "M", True) for i in range(20)]
        )
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, pairs)
        incorrect, correct = exp605._load_test_sets(p, n_incorrect=3, n_correct=3)
        assert len(incorrect) == 3
        assert len(correct) == 3


# ---------------------------------------------------------------------------
# _load_training_pairs
# ---------------------------------------------------------------------------


class TestLoadTrainingPairs:
    """REQ-BENCH-058-4: training pairs must exclude test entries to prevent data leakage."""

    def test_excludes_test_pairs(self, tmp_path: Path) -> None:
        # Test pairs must not appear in training output
        test_pair = _make_pair(0, "M", False, response="test_response_unique")
        other_pair = _make_pair(1, "M", False, response="train_response")
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, [test_pair, other_pair])

        texts, labels = exp605._load_training_pairs(p, [test_pair], [])
        assert "test_response_unique" not in texts
        assert "train_response" in texts

    def test_labels_1_for_incorrect(self, tmp_path: Path) -> None:
        # Incorrect pairs get label 1.0 (violation)
        pair = _make_pair(5, "M", False, response="bad_resp")
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, [pair])
        texts, labels = exp605._load_training_pairs(p, [], [])
        assert labels[texts.index("bad_resp")] == 1.0

    def test_labels_0_for_correct(self, tmp_path: Path) -> None:
        # Correct pairs get label 0.0 (no violation)
        pair = _make_pair(5, "M", True, response="good_resp")
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, [pair])
        texts, labels = exp605._load_training_pairs(p, [], [])
        assert labels[texts.index("good_resp")] == 0.0

    def test_empty_corpus_returns_empty_lists(self, tmp_path: Path) -> None:
        p = tmp_path / "live_pairs.json"
        _write_corpus(p, [])
        texts, labels = exp605._load_training_pairs(p, [], [])
        assert texts == []
        assert labels == []


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """SCENARIO-BENCH-052: gate decision and schema must be correct in all cases."""

    def _make_tmpl(self, tmp_path: Path):
        """Create a minimal ExperimentTemplate stub."""
        from experiment_template import ExperimentTemplate  # noqa: PLC0415

        return ExperimentTemplate(
            605,
            "Live Extractor Diagnostic v4",
            "results/experiment_605_extractor_diagnostic_v4.json",
            requires_gpu=False,
            repo_root=tmp_path,
        )

    def test_schema_field(self, tmp_path: Path) -> None:
        # build_result() stores the schema identifier in result_schema; schema key is sorted key list
        tmpl = self._make_tmpl(tmp_path)
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert art["result_schema"] == "carnot.extractor_diagnostic_v4.v1"
        assert isinstance(art["schema"], list)

    def test_gate_closed_when_both_zero_recall(self, tmp_path: Path) -> None:
        # SCENARIO-BENCH-052: gate_open=False when both recalls are below 0.20
        tmpl = self._make_tmpl(tmp_path)
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert art["gate_open"] is False
        assert art["honest_verdict"] == "gate_closed_recall_below_threshold"

    def test_gate_open_when_coace_v4_above_threshold(self, tmp_path: Path) -> None:
        # 6/25 = 0.24 >= 0.20 — gate should open
        tmpl = self._make_tmpl(tmp_path)
        coace_flags = [True] * 6 + [False] * 19
        art = exp605._build_artifact(tmpl, coace_flags, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert art["gate_open"] is True
        assert art["honest_verdict"] == "gate_open_proceed_to_vr"

    def test_gate_open_when_dsvd_above_threshold(self, tmp_path: Path) -> None:
        # 5/25 = 0.20 — exactly at threshold — gate should open
        tmpl = self._make_tmpl(tmp_path)
        dsvd_flags = [True] * 5 + [False] * 20
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, dsvd_flags, [False] * 10, 25, 10)
        assert art["gate_open"] is True

    def test_gate_note_contains_exp609_when_open(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        flags = [True] * 6 + [False] * 19
        art = exp605._build_artifact(tmpl, flags, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert "Exp 609" in art["gate_note"]

    def test_gate_note_contains_do_not_schedule_when_closed(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert "DO NOT schedule Exp 609" in art["gate_note"]

    def test_winning_extractor_coace_v4_when_higher(self, tmp_path: Path) -> None:
        # coace_v4_recall=0.08 > dsvd_recall=0.04
        tmpl = self._make_tmpl(tmp_path)
        coace_flags = [True] * 2 + [False] * 23
        dsvd_flags = [True] * 1 + [False] * 24
        art = exp605._build_artifact(tmpl, coace_flags, [False] * 10, dsvd_flags, [False] * 10, 25, 10)
        assert art["winning_extractor"] == "coace_v4"

    def test_winning_extractor_dsvd_when_higher(self, tmp_path: Path) -> None:
        # dsvd_recall=0.12 > coace_v4_recall=0.04
        tmpl = self._make_tmpl(tmp_path)
        coace_flags = [True] * 1 + [False] * 24
        dsvd_flags = [True] * 3 + [False] * 22
        art = exp605._build_artifact(tmpl, coace_flags, [False] * 10, dsvd_flags, [False] * 10, 25, 10)
        assert art["winning_extractor"] == "dsvd"

    def test_coace_v4_recall_computed_correctly(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        coace_flags = [True] * 3 + [False] * 22  # 3/25 = 0.12
        art = exp605._build_artifact(tmpl, coace_flags, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert art["coace_v4_recall"] == pytest.approx(3 / 25)

    def test_coace_v4_fp_rate_computed_correctly(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        fp_flags = [True] * 2 + [False] * 8  # 2/10 = 0.20
        art = exp605._build_artifact(tmpl, [False] * 25, fp_flags, [False] * 25, [False] * 10, 25, 10)
        assert art["coace_v4_fp_rate"] == pytest.approx(2 / 10)

    def test_dsvd_recall_computed_correctly(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        dsvd_flags = [True] * 4 + [False] * 21  # 4/25 = 0.16
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, dsvd_flags, [False] * 10, 25, 10)
        assert art["dsvd_recall"] == pytest.approx(4 / 25)

    def test_n_incorrect_and_n_correct_in_artifact(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        art = exp605._build_artifact(tmpl, [False] * 25, [False] * 10, [False] * 25, [False] * 10, 25, 10)
        assert art["n_incorrect"] == 25
        assert art["n_correct"] == 10

    def test_best_recall_is_max(self, tmp_path: Path) -> None:
        tmpl = self._make_tmpl(tmp_path)
        coace_flags = [True] * 2 + [False] * 23  # 0.08
        dsvd_flags = [True] * 4 + [False] * 21   # 0.16
        art = exp605._build_artifact(tmpl, coace_flags, [False] * 10, dsvd_flags, [False] * 10, 25, 10)
        assert art["best_recall"] == pytest.approx(4 / 25)


# ---------------------------------------------------------------------------
# _run_coace_v4
# ---------------------------------------------------------------------------


class TestRunCoaceV4:
    """SCENARIO-BENCH-050: CoACEExtractorV4 flags responses with arithmetic violations."""

    def test_returns_list_of_bools(self) -> None:
        entries = [{"response": "The answer is 42."}]
        flags = exp605._run_coace_v4(entries)
        assert isinstance(flags, list)
        assert all(isinstance(f, bool) for f in flags)

    def test_length_matches_entries(self) -> None:
        entries = [{"response": f"text {i}"} for i in range(5)]
        flags = exp605._run_coace_v4(entries)
        assert len(flags) == 5

    def test_incorrect_arithmetic_detected(self) -> None:
        # 3 * 4 = 13 is wrong — V4 should detect this
        entries = [{"response": "3 * 4 = 13"}]
        flags = exp605._run_coace_v4(entries)
        assert flags[0] is True

    def test_placeholder_not_flagged(self) -> None:
        # Placeholder responses with no arithmetic should not be flagged
        entries = [{"response": "The answer is 42."}]
        flags = exp605._run_coace_v4(entries)
        assert flags[0] is False

    def test_correct_arithmetic_not_flagged(self) -> None:
        # 3 * 4 = 12 is correct — should NOT be flagged
        entries = [{"response": "3 * 4 = 12"}]
        flags = exp605._run_coace_v4(entries)
        assert flags[0] is False

    def test_exception_in_extractor_returns_false(self) -> None:
        # If extract() raises, the flag should be False (non-fatal)
        entries = [{"response": "any text"}]
        with patch(
            "carnot.extraction.coace_extractor_v4.CoACEExtractorV4.extract",
            side_effect=RuntimeError("boom"),
        ):
            flags = exp605._run_coace_v4(entries)
        assert flags == [False]


# ---------------------------------------------------------------------------
# _build_dsvd_adapter
# ---------------------------------------------------------------------------


class TestBuildDsvdAdapter:
    """SCENARIO-BENCH-051: DSVDAdapter creation and fitting."""

    def test_returns_dsvd_adapter(self) -> None:
        from carnot.pipeline.dsvd_adapter import DSVDAdapter  # noqa: PLC0415

        adapter = exp605._build_dsvd_adapter([], [])
        assert isinstance(adapter, DSVDAdapter)

    def test_unfitted_adapter_has_threshold_05(self) -> None:
        adapter = exp605._build_dsvd_adapter([], [])
        assert adapter.violation_threshold == 0.5

    def test_fits_without_error_on_training_data(self) -> None:
        texts = ["step one arithmetic 3 * 4 = 12", "step two logic wrong"]
        labels = [0.0, 1.0]
        # Should not raise
        adapter = exp605._build_dsvd_adapter(texts, labels)
        assert adapter is not None


# ---------------------------------------------------------------------------
# _run_dsvd
# ---------------------------------------------------------------------------


class TestRunDsvd:
    """SCENARIO-BENCH-051: DSVDAdapter.verify_step() scoring and violation flagging."""

    def test_returns_list_of_bools(self) -> None:
        adapter = exp605._build_dsvd_adapter([], [])
        entries = [{"response": "some text"}]
        flags = exp605._run_dsvd(adapter, entries)
        assert isinstance(flags, list)
        assert all(isinstance(f, bool) for f in flags)

    def test_length_matches_entries(self) -> None:
        adapter = exp605._build_dsvd_adapter([], [])
        entries = [{"response": f"text {i}"} for i in range(7)]
        flags = exp605._run_dsvd(adapter, entries)
        assert len(flags) == 7

    def test_exception_returns_false(self) -> None:
        adapter = MagicMock()
        adapter.violation_threshold = 0.5
        adapter.verify_step.side_effect = RuntimeError("probe failed")
        entries = [{"response": "anything"}]
        flags = exp605._run_dsvd(adapter, entries)
        assert flags == [False]

    def test_probability_above_threshold_returns_true(self) -> None:
        adapter = MagicMock()
        adapter.violation_threshold = 0.5
        result = MagicMock()
        result.violation_probability = 0.8
        adapter.verify_step.return_value = result
        entries = [{"response": "test"}]
        flags = exp605._run_dsvd(adapter, entries)
        assert flags == [True]

    def test_probability_at_threshold_returns_false(self) -> None:
        # Boundary: 0.5 > 0.5 is False — threshold is strict greater-than
        adapter = MagicMock()
        adapter.violation_threshold = 0.5
        result = MagicMock()
        result.violation_probability = 0.5
        adapter.verify_step.return_value = result
        entries = [{"response": "test"}]
        flags = exp605._run_dsvd(adapter, entries)
        assert flags == [False]
