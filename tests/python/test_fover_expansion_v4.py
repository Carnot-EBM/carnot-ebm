"""Tests for Exp 1055 FoVer Corpus Expansion v4 helper functions.

Covers arithmetic verification, GGUF snapshot detection, corpus merge,
and stratified splitting without hitting the network or GPU.

Spec: REQ-FOVER-004 (n_total_pairs >= 500), REQ-FOVER-005 (n_metamorphic_validated > 0)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_1055_fover_expansion_v4 import (  # noqa: E402
    _safe_eval_arithmetic,
    verify_step_with_z3,
    merge_corpora,
    stratified_split,
    _find_gguf_in_all_snapshots,
)


# ---------------------------------------------------------------------------
# _safe_eval_arithmetic
# ---------------------------------------------------------------------------


class TestSafeEvalArithmetic:
    """Spec: REQ-FOVER-004 (Z3 labeling correctness)."""

    def test_integer_addition(self):
        assert _safe_eval_arithmetic("48+24") == 72.0

    def test_integer_division(self):
        assert _safe_eval_arithmetic("48/2") == 24.0

    def test_multiplication(self):
        assert _safe_eval_arithmetic("3*4") == 12.0

    def test_compound_expression(self):
        assert _safe_eval_arithmetic("10+15+25") == 50.0

    def test_rejects_identifiers(self):
        # Expressions with variable names must return None to prevent code injection.
        assert _safe_eval_arithmetic("x+2") is None

    def test_rejects_invalid_syntax(self):
        assert _safe_eval_arithmetic("not_a_number") is None

    def test_empty_string(self):
        assert _safe_eval_arithmetic("") is None

    def test_subtraction(self):
        assert _safe_eval_arithmetic("10-3") == 7.0


# ---------------------------------------------------------------------------
# verify_step_with_z3
# ---------------------------------------------------------------------------


class TestVerifyStepWithZ3:
    """Spec: REQ-FOVER-004 (correct vs incorrect labeling)."""

    def test_correct_division(self):
        assert verify_step_with_z3("48/2", "24") is True

    def test_correct_addition(self):
        assert verify_step_with_z3("48+24", "72") is True

    def test_incorrect_result(self):
        # 48/2 = 24, not 25.
        assert verify_step_with_z3("48/2", "25") is False

    def test_unparseable_expr(self):
        assert verify_step_with_z3("x*y", "10") is None

    def test_unparseable_result(self):
        assert verify_step_with_z3("2+2", "four") is None

    def test_off_by_one_is_incorrect(self):
        # GSM8K step with wrong embedded answer.
        assert verify_step_with_z3("3*4", "13") is False

    def test_float_tolerance(self):
        # Small floating-point discrepancies (< 0.5) are treated as correct.
        assert verify_step_with_z3("1/3", "0.3333") is True


# ---------------------------------------------------------------------------
# merge_corpora
# ---------------------------------------------------------------------------


class TestMergeCorpora:
    """Spec: REQ-FOVER-004 (deduplication preserves prior labels)."""

    def _make_pair(self, text: str, label: str = "correct") -> dict:
        return {"question_id": "q1", "step_text": text, "label": label, "confidence": 1.0}

    def test_empty_inputs(self):
        assert merge_corpora([], []) == []

    def test_prior_only(self):
        prior = [self._make_pair("2+2=4")]
        result = merge_corpora(prior, [])
        assert len(result) == 1

    def test_new_only(self):
        new = [self._make_pair("3*3=9")]
        result = merge_corpora([], new)
        assert len(result) == 1

    def test_dedup_preserves_prior_label(self):
        # When prior has "correct" and new has "incorrect" for the SAME text,
        # prior label wins.
        text = "2 + 2 = 4"
        prior = [self._make_pair(text, label="correct")]
        new = [self._make_pair(text, label="incorrect")]
        result = merge_corpora(prior, new)
        assert len(result) == 1
        assert result[0]["label"] == "correct"

    def test_distinct_steps_both_kept(self):
        prior = [self._make_pair("a step")]
        new = [self._make_pair("different step")]
        result = merge_corpora(prior, new)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    """Spec: REQ-FOVER-004 (balanced train/test split)."""

    def _corpus(self, n_correct: int, n_incorrect: int) -> list[dict]:
        items = []
        for i in range(n_correct):
            items.append(
                {
                    "step_text": f"correct step {i}",
                    "label": "correct",
                    "confidence": 1.0,
                    "question_id": f"c{i}",
                }
            )
        for i in range(n_incorrect):
            items.append(
                {
                    "step_text": f"incorrect step {i}",
                    "label": "incorrect",
                    "confidence": 1.0,
                    "question_id": f"i{i}",
                }
            )
        return items

    def test_split_sizes_sum_to_corpus(self):
        corpus = self._corpus(80, 20)
        train, test = stratified_split(corpus)
        assert len(train) + len(test) == len(corpus)

    def test_test_fraction_roughly_correct(self):
        corpus = self._corpus(100, 100)
        _, test = stratified_split(corpus, test_fraction=0.2)
        # Should be ~40 (20% of 200).
        assert 30 <= len(test) <= 50

    def test_both_labels_appear_in_train_and_test(self):
        corpus = self._corpus(50, 50)
        train, test = stratified_split(corpus)
        train_labels = {x["label"] for x in train}
        test_labels = {x["label"] for x in test}
        assert "correct" in train_labels
        assert "incorrect" in train_labels
        assert "correct" in test_labels
        assert "incorrect" in test_labels

    def test_deterministic_with_same_seed(self):
        corpus = self._corpus(40, 40)
        train1, test1 = stratified_split(corpus, seed=7)
        train2, test2 = stratified_split(corpus, seed=7)
        assert [x["step_text"] for x in train1] == [x["step_text"] for x in train2]


# ---------------------------------------------------------------------------
# _find_gguf_in_all_snapshots
# ---------------------------------------------------------------------------


class TestFindGgufInAllSnapshots:
    """Spec: REQ-FOVER-005 (GGUF detection across all snapshots)."""

    def test_returns_none_when_model_dir_absent(self, tmp_path):
        result = _find_gguf_in_all_snapshots(
            "nonexistent/model-GGUF",
            cache_root=str(tmp_path),
        )
        assert result is None

    def test_finds_gguf_in_older_snapshot(self, tmp_path):
        # Simulate the real HF cache layout: metadata-only snapshot is newest,
        # GGUF lives in an older snapshot.
        model_dir = tmp_path / "models--org--model-GGUF" / "snapshots"
        old_snap = model_dir / "abc123"
        old_snap.mkdir(parents=True)
        gguf = old_snap / "model-Q4_K_M.gguf"
        gguf.write_bytes(b"fake gguf")

        # Newer snapshot with only config.json (metadata).
        new_snap = model_dir / "def456"
        new_snap.mkdir(parents=True)
        (new_snap / "config.json").write_text("{}")
        # Make new_snap appear newer.
        import os
        import time

        os.utime(new_snap, (time.time() + 100, time.time() + 100))

        result = _find_gguf_in_all_snapshots("org/model-GGUF", cache_root=str(tmp_path))
        assert result is not None
        assert result.endswith(".gguf")

    def test_prefers_q4_k_m(self, tmp_path):
        model_dir = tmp_path / "models--org--model-GGUF" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model-Q8_0.gguf").write_bytes(b"q8")
        (model_dir / "model-Q4_K_M.gguf").write_bytes(b"q4km")

        result = _find_gguf_in_all_snapshots("org/model-GGUF", cache_root=str(tmp_path))
        assert result is not None
        assert "Q4_K_M" in result

    def test_returns_none_when_no_ggufs_anywhere(self, tmp_path):
        model_dir = tmp_path / "models--org--model-GGUF" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        result = _find_gguf_in_all_snapshots("org/model-GGUF", cache_root=str(tmp_path))
        assert result is None
