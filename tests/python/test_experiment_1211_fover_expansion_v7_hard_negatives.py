"""Tests for Exp 1211 FoVer Expansion v7 — hard negatives helpers.

Spec: REQ-VERIFY-1211, SCENARIO-VERIFY-1211
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
for _d in [str(_PYTHON_ROOT), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.eval.fover_expansion_v7 import (  # noqa: E402
    ALLOWED_VERDICTS,
    HARD_NEG_HI,
    HARD_NEG_LO,
    K5_AUROC_BASELINE,
    REQUIRED_ARTIFACT_FIELDS,
    Z3_CORRECT_THRESHOLD,
    Z3_INCORRECT_THRESHOLD,
    _answer_matches,
    append_rows_to_jsonl,
    build_artifact,
    compute_hard_negative_fraction,
    determine_verdict,
    evaluate_k5_auroc_on_rows,
    label_response,
    load_eval_rows,
    load_fover_jsonl,
    tie_aware_auroc,
)


# ---------------------------------------------------------------------------
# _answer_matches tests
# ---------------------------------------------------------------------------


class TestAnswerMatches:
    """Spec: REQ-VERIFY-1211"""

    def test_exact_integer_match(self):
        # REQ-VERIFY-1211: final answer correct when it matches expected
        assert _answer_matches("The answer is 42.", "42") is True

    def test_decimal_match_within_tolerance(self):
        assert _answer_matches("Total is 3.99", "4") is True

    def test_mismatch_returns_false(self):
        assert _answer_matches("The answer is 10.", "42") is False

    def test_empty_expected_always_true(self):
        assert _answer_matches("some text", "") is True

    def test_comma_number_match(self):
        assert _answer_matches("Grand total: 1,000 dollars.", "1000") is True

    def test_no_numbers_in_response(self):
        assert _answer_matches("No numbers here.", "42") is False


# ---------------------------------------------------------------------------
# label_response tests
# ---------------------------------------------------------------------------


class TestLabelResponse:
    """Spec: REQ-VERIFY-1211, SCENARIO-VERIFY-1211"""

    def _make_verifier(self, score: float) -> MagicMock:
        verifier = MagicMock()
        verifier.score.return_value = score
        return verifier

    def test_low_energy_labeled_correct(self):
        # REQ-VERIFY-1211: z3_score < Z3_CORRECT_THRESHOLD → label='correct'
        row = label_response(
            "Step 1: 2 + 2 = 4. Answer: 4.",
            "What is 2+2?",
            "4",
            "TestModel",
            "q001",
            z3_verifier=self._make_verifier(0.10),
        )
        assert row["label"] == "correct"
        assert row["z3_score"] == pytest.approx(0.10)
        assert row["hard_negative"] is False

    def test_high_energy_labeled_incorrect(self):
        # SCENARIO-VERIFY-1211: z3_score > Z3_INCORRECT_THRESHOLD → label='incorrect'
        row = label_response(
            "Step 1: 2 + 2 = 5. Answer: 5.",
            "What is 2+2?",
            "4",
            "TestModel",
            "q002",
            z3_verifier=self._make_verifier(0.85),
        )
        assert row["label"] == "incorrect"
        assert row["hard_negative"] is False

    def test_middle_energy_is_hard_negative(self):
        # REQ-VERIFY-1211: HARD_NEG_LO <= z3_score <= HARD_NEG_HI → hard_negative=True
        row = label_response(
            "Mixed arithmetic",
            "Question",
            "10",
            "TestModel",
            "q003",
            z3_verifier=self._make_verifier(0.50),
        )
        assert row["hard_negative"] is True
        # Conservative labeling: middle range → "incorrect" (not "correct")
        assert row["label"] == "incorrect"

    def test_hard_negative_boundary_lo(self):
        # Edge: exactly at HARD_NEG_LO boundary → hard_negative
        row = label_response(
            "text",
            "q",
            "1",
            "m",
            "q004",
            z3_verifier=self._make_verifier(HARD_NEG_LO),
        )
        assert row["hard_negative"] is True

    def test_hard_negative_boundary_hi(self):
        # Edge: exactly at HARD_NEG_HI boundary → hard_negative
        row = label_response(
            "text",
            "q",
            "1",
            "m",
            "q005",
            z3_verifier=self._make_verifier(HARD_NEG_HI),
        )
        assert row["hard_negative"] is True

    def test_required_fields_present(self):
        # SCENARIO-VERIFY-1211: all corpus schema fields must be present
        row = label_response(
            "Some response",
            "Question",
            "42",
            "TestModel",
            "q006",
            z3_verifier=self._make_verifier(0.2),
        )
        for field in (
            "question_id",
            "question",
            "step_text",
            "label",
            "confidence",
            "z3_score",
            "sc_energy_score",
            "model",
            "source",
            "verifier",
            "hard_negative",
            "answer_matches_expected",
            "expected_answer",
        ):
            assert field in row, f"missing field: {field}"

    def test_question_id_prefixed_with_exp(self):
        row = label_response(
            "text",
            "q",
            "1",
            "m",
            "xyz",
            z3_verifier=self._make_verifier(0.1),
        )
        assert row["question_id"].startswith("exp1211-")

    def test_confidence_inverse_of_z3_score(self):
        row = label_response(
            "text",
            "q",
            "1",
            "m",
            "q007",
            z3_verifier=self._make_verifier(0.4),
        )
        assert row["confidence"] == pytest.approx(1.0 - 0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_hard_negative_fraction tests
# ---------------------------------------------------------------------------


class TestComputeHardNegativeFraction:
    """Spec: REQ-VERIFY-1211"""

    def test_empty_returns_zero(self):
        assert compute_hard_negative_fraction([]) == pytest.approx(0.0)

    def test_all_hard_negatives(self):
        rows = [{"hard_negative": True}] * 5
        assert compute_hard_negative_fraction(rows) == pytest.approx(1.0)

    def test_no_hard_negatives(self):
        rows = [{"hard_negative": False}] * 5
        assert compute_hard_negative_fraction(rows) == pytest.approx(0.0)

    def test_twenty_percent(self):
        rows = [{"hard_negative": True}] + [{"hard_negative": False}] * 4
        assert compute_hard_negative_fraction(rows) == pytest.approx(0.20)

    def test_missing_field_treated_as_false(self):
        rows = [{"label": "correct"}] * 5
        assert compute_hard_negative_fraction(rows) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# tie_aware_auroc tests
# ---------------------------------------------------------------------------


class TestTieAwareAuroc:
    """Spec: REQ-VERIFY-1211"""

    def test_perfect_separation(self):
        # All positives score higher than all negatives → AUROC = 1.0
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert tie_aware_auroc(labels, scores) == pytest.approx(1.0)

    def test_complete_reversal(self):
        # All positives score lower → AUROC = 0.0
        labels = [1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9]
        assert tie_aware_auroc(labels, scores) == pytest.approx(0.0)

    def test_all_tied_returns_half(self):
        # Every score identical → random performance
        labels = [1, 1, 0, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        assert tie_aware_auroc(labels, scores) == pytest.approx(0.5)

    def test_missing_class_returns_half(self):
        # Single class → cannot compute AUROC
        assert tie_aware_auroc([1, 1, 1], [0.8, 0.9, 0.7]) == pytest.approx(0.5)
        assert tie_aware_auroc([0, 0, 0], [0.2, 0.3, 0.1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluate_k5_auroc_on_rows tests
# ---------------------------------------------------------------------------


class TestEvaluateK5Auroc:
    """Spec: REQ-VERIFY-1211"""

    def test_returns_float_in_unit_interval(self):
        rows = [
            {"label": "correct", "step_text": "2 + 2 = 4"},
            {"label": "incorrect", "step_text": "2 + 2 = 5"},
        ]
        result = evaluate_k5_auroc_on_rows(rows)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_rows_returns_half(self):
        result = evaluate_k5_auroc_on_rows([])
        assert result == pytest.approx(0.5)

    def test_blank_step_text_rows_skipped(self):
        rows = [{"label": "correct", "step_text": ""}]
        result = evaluate_k5_auroc_on_rows(rows)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# determine_verdict tests
# ---------------------------------------------------------------------------


class TestDetermineVerdict:
    """Spec: REQ-VERIFY-1211"""

    def test_below_500_pairs_is_expansion_below_target(self):
        assert determine_verdict(499, 0.05, 0.25) == "expansion_below_target"

    def test_above_500_with_improvement(self):
        assert determine_verdict(500, 0.01, 0.25) == "fover_expanded_k5_improved"

    def test_above_500_flat(self):
        assert determine_verdict(500, 0.001, 0.25) == "fover_expanded_k5_flat"

    def test_above_500_regressed(self):
        assert determine_verdict(500, -0.01, 0.25) == "fover_expanded_k5_regressed"

    def test_exact_threshold_is_flat(self):
        # Exactly at the improvement threshold → flat (not improved)
        assert determine_verdict(500, 0.002, 0.25) == "fover_expanded_k5_flat"

    def test_all_verdicts_in_allowed_set(self):
        test_cases = [
            (499, 0.0, 0.2),
            (500, 0.01, 0.2),
            (500, 0.001, 0.2),
            (500, -0.01, 0.2),
        ]
        for n, delta, hn in test_cases:
            verdict = determine_verdict(n, delta, hn)
            assert verdict in ALLOWED_VERDICTS, f"verdict {verdict!r} not in ALLOWED_VERDICTS"


# ---------------------------------------------------------------------------
# build_artifact tests
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """Spec: REQ-VERIFY-1211, SCENARIO-VERIFY-1211"""

    def _make_rows(self, n: int, n_hard: int = 0, n_incorrect: int = 0) -> list[dict]:
        rows: list[dict] = []
        for i in range(n):
            is_hard = i < n_hard
            is_inc = i < n_incorrect or is_hard
            rows.append(
                {
                    "label": "incorrect" if is_inc else "correct",
                    "hard_negative": is_hard,
                    "sc_energy_score": 0.50 if is_hard else (0.80 if is_inc else 0.10),
                }
            )
        return rows

    def test_all_required_fields_present(self):
        rows = self._make_rows(500, n_hard=110, n_incorrect=200)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.928,
            models_used=["ModelA", "ModelB"],
            fover_corpus_total_before=7329,
            duration_s=1800.0,
            started_at="2026-05-03T00:00:00Z",
        )
        for field in REQUIRED_ARTIFACT_FIELDS:
            assert field in artifact, f"missing field: {field}"

    def test_counts_match_rows(self):
        rows = self._make_rows(500, n_incorrect=200)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.927,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["n_new_pairs_generated"] == 500
        assert artifact["n_new_pairs_incorrect"] == 200
        assert artifact["n_new_pairs_correct"] == 300

    def test_fover_v7_pairs_above_500_true_for_500(self):
        rows = self._make_rows(500)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.924,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["fover_v7_pairs_above_500"] is True

    def test_fover_v7_pairs_above_500_false_for_499(self):
        rows = self._make_rows(499)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.924,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["fover_v7_pairs_above_500"] is False

    def test_corpus_total_after_is_sum(self):
        rows = self._make_rows(500)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.924,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["fover_corpus_total_after"] == 7329 + 500

    def test_auroc_delta_computed_correctly(self):
        rows = self._make_rows(500)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.900,
            k5_auroc_post=0.920,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["k5_auroc_delta"] == pytest.approx(0.020, abs=1e-6)

    def test_honest_verdict_in_allowed_set(self):
        rows = self._make_rows(500)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.930,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["honest_verdict"] in ALLOWED_VERDICTS

    def test_missing_rows_gives_expansion_below_target(self):
        rows = self._make_rows(100)
        artifact = build_artifact(
            rows,
            k5_auroc_pre=0.924,
            k5_auroc_post=0.930,
            models_used=["M"],
            fover_corpus_total_before=7329,
            duration_s=10.0,
            started_at="2026-05-03T00:00:00Z",
        )
        assert artifact["honest_verdict"] == "expansion_below_target"
        assert artifact["status"] == "partial"


# ---------------------------------------------------------------------------
# K5_AUROC_BASELINE sanity check
# ---------------------------------------------------------------------------


def test_k5_auroc_baseline_matches_exp1185():
    """Confirm the baseline AUROC is consistent with Exp 1185 measurement.

    Spec: REQ-VERIFY-1211
    """
    # 0.92403 from exp1185 artifact field 'k5_auroc_on_eval'
    assert K5_AUROC_BASELINE == pytest.approx(0.92403, abs=1e-5)


# ---------------------------------------------------------------------------
# _answer_matches — non-numeric expected path (lines 163-164, 172-173)
# ---------------------------------------------------------------------------


class TestAnswerMatchesNonNumeric:
    """Spec: REQ-VERIFY-1211 — covers the non-numeric expected value branch."""

    def test_non_numeric_expected_substring_match(self):
        # expected cannot be parsed as float → fall back to substring check
        assert _answer_matches("The answer is yes.", "yes") is True

    def test_non_numeric_expected_substring_mismatch(self):
        assert _answer_matches("The answer is no.", "yes") is False

    def test_regex_fragment_in_response_not_matching(self):
        # Numbers found but contain garbage that causes float conversion failure.
        # We create a response whose only "number" token is invalid → nums_clean is empty.
        # _answer_matches should return False (no clean numbers).
        assert _answer_matches("value is +-.", "5") is False


# ---------------------------------------------------------------------------
# load_fover_jsonl / append_rows_to_jsonl (lines 283-304)
# ---------------------------------------------------------------------------


class TestCorpusIO:
    """Spec: REQ-VERIFY-1211 — I/O helpers for FoVer JSONL corpus files."""

    def test_load_nonexistent_file_returns_empty(self, tmp_path):
        result = load_fover_jsonl(tmp_path / "missing.jsonl")
        assert result == []

    def test_load_valid_jsonl(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        rows = [
            {"label": "correct", "step_text": "2+2=4"},
            {"label": "incorrect", "step_text": "2+2=5"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        loaded = load_fover_jsonl(p)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "correct"

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        p.write_text('\n{"label":"correct"}\n\n{"label":"incorrect"}\n')
        loaded = load_fover_jsonl(p)
        assert len(loaded) == 2

    def test_load_jsonl_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        p.write_text('{"label":"correct"}\nnot-json\n{"label":"incorrect"}\n')
        loaded = load_fover_jsonl(p)
        assert len(loaded) == 2

    def test_append_rows_creates_file(self, tmp_path):
        p = tmp_path / "sub" / "out.jsonl"
        rows = [{"label": "correct"}]
        n = append_rows_to_jsonl(p, rows)
        assert n == 1
        assert p.exists()
        reloaded = load_fover_jsonl(p)
        assert len(reloaded) == 1

    def test_append_rows_accumulates(self, tmp_path):
        p = tmp_path / "out.jsonl"
        append_rows_to_jsonl(p, [{"label": "correct"}])
        append_rows_to_jsonl(p, [{"label": "incorrect"}])
        reloaded = load_fover_jsonl(p)
        assert len(reloaded) == 2

    def test_append_returns_count(self, tmp_path):
        p = tmp_path / "out.jsonl"
        n = append_rows_to_jsonl(p, [{"x": 1}, {"x": 2}, {"x": 3}])
        assert n == 3


# ---------------------------------------------------------------------------
# load_eval_rows (lines 322-349)
# ---------------------------------------------------------------------------


class TestLoadEvalRows:
    """Spec: REQ-VERIFY-1211 — balanced eval set loader."""

    def _write_json_corpus(self, path: Path, n_correct: int, n_incorrect: int) -> None:
        rows = [{"label": "correct", "step_text": "c"}] * n_correct
        rows += [{"label": "incorrect", "step_text": "i"}] * n_incorrect
        path.write_text(json.dumps(rows))

    def _write_jsonl_corpus(self, path: Path, n_correct: int, n_incorrect: int) -> None:
        rows = [{"label": "correct", "step_text": "c"}] * n_correct
        rows += [{"label": "incorrect", "step_text": "i"}] * n_incorrect
        path.write_text("\n".join(json.dumps(r) for r in rows))

    def test_raises_if_too_few_rows_json(self, tmp_path):
        p = tmp_path / "corpus.json"
        self._write_json_corpus(p, n_correct=10, n_incorrect=10)
        with pytest.raises(ValueError, match="rows, need at least"):
            load_eval_rows(p, n_examples=200)

    def test_raises_if_only_one_class(self, tmp_path):
        p = tmp_path / "corpus.json"
        self._write_json_corpus(p, n_correct=300, n_incorrect=0)
        with pytest.raises(ValueError, match="must contain both"):
            load_eval_rows(p, n_examples=200)

    def test_loads_from_jsonl(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        self._write_jsonl_corpus(p, n_correct=150, n_incorrect=150)
        result = load_eval_rows(p, n_examples=100)
        assert len(result) == 100

    def test_loads_from_json(self, tmp_path):
        p = tmp_path / "corpus.json"
        self._write_json_corpus(p, n_correct=150, n_incorrect=150)
        result = load_eval_rows(p, n_examples=100)
        assert len(result) == 100

    def test_imbalanced_corpus_clips_to_available(self, tmp_path):
        # More incorrect than correct — should not crash, just use what's there.
        p = tmp_path / "corpus.json"
        self._write_json_corpus(p, n_correct=20, n_incorrect=300)
        result = load_eval_rows(p, n_examples=40)
        assert len(result) == 40

    def test_corpus_with_non_list_json_gives_empty_path(self, tmp_path):
        # If JSON file holds a dict instead of list, all_rows becomes [].
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps({"label": "correct"}))
        with pytest.raises(ValueError):
            load_eval_rows(p, n_examples=10)


# ---------------------------------------------------------------------------
# build_artifact error paths (lines 450, 454)
# ---------------------------------------------------------------------------


class TestBuildArtifactErrorPaths:
    """Spec: REQ-VERIFY-1211 — artifact validation raises on bad data."""

    def test_invalid_verdict_raises(self, monkeypatch):
        # Monkeypatch determine_verdict to return a bogus string so the
        # invalid_verdict branch (line 452-454) is exercised.
        import carnot.eval.fover_expansion_v7 as mod

        monkeypatch.setattr(mod, "determine_verdict", lambda *a, **kw: "not_a_valid_verdict")
        rows = [{"label": "correct", "hard_negative": False}] * 500
        with pytest.raises(ValueError, match="invalid honest_verdict"):
            build_artifact(
                rows,
                k5_auroc_pre=0.924,
                k5_auroc_post=0.930,
                models_used=["M"],
                fover_corpus_total_before=7329,
                duration_s=10.0,
                started_at="2026-05-03T00:00:00Z",
            )

    def test_missing_required_fields_raises(self, monkeypatch):
        # Monkeypatch REQUIRED_ARTIFACT_FIELDS to include a field that build_artifact
        # does not populate, so the missing-fields guard (line 450) fires.
        import carnot.eval.fover_expansion_v7 as mod

        extended = mod.REQUIRED_ARTIFACT_FIELDS | frozenset({"__phantom_field__"})
        monkeypatch.setattr(mod, "REQUIRED_ARTIFACT_FIELDS", extended)
        rows = [{"label": "correct", "hard_negative": False}] * 500
        with pytest.raises(ValueError, match="missing required artifact fields"):
            build_artifact(
                rows,
                k5_auroc_pre=0.924,
                k5_auroc_post=0.930,
                models_used=["M"],
                fover_corpus_total_before=7329,
                duration_s=10.0,
                started_at="2026-05-03T00:00:00Z",
            )


# ---------------------------------------------------------------------------
# evaluate_k5_auroc_on_rows — ImportError fallback (lines 259-260)
# ---------------------------------------------------------------------------


class TestEvaluateK5AurocImportError:
    """Spec: REQ-VERIFY-1211 — graceful degradation when Z3MathVerifier is absent."""

    def test_import_error_returns_half(self, monkeypatch):
        import sys
        import carnot.eval.fover_expansion_v7 as mod

        # Temporarily hide Z3MathVerifier so the ImportError branch fires.
        original = sys.modules.get("carnot.verify.z3_math_verifier")
        sys.modules["carnot.verify.z3_math_verifier"] = None  # type: ignore[assignment]
        try:
            result = mod.evaluate_k5_auroc_on_rows([{"label": "correct", "step_text": "2+2=4"}])
        finally:
            if original is None:
                del sys.modules["carnot.verify.z3_math_verifier"]
            else:
                sys.modules["carnot.verify.z3_math_verifier"] = original
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# load_eval_rows — n_cor clipping (lines 344-345)
# ---------------------------------------------------------------------------


class TestLoadEvalRowsCorClip:
    """Spec: REQ-VERIFY-1211 — handles imbalanced correct/incorrect class sizes."""

    def test_clips_n_cor_when_fewer_correct_than_half(self, tmp_path):
        # 200 incorrect, 20 correct, n_examples=100.
        # n_inc = min(200, 50) = 50; n_cor = 50 > len(correct)=20 → clips to 20,
        # then n_inc = 80.  Total selected = 100, but only 100 rows available total
        # so we need >=100 rows in the corpus.
        rows = [{"label": "incorrect", "step_text": "i"}] * 200
        rows += [{"label": "correct", "step_text": "c"}] * 20
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps(rows))
        result = load_eval_rows(p, n_examples=100)
        # Should not raise; should return up to 100 rows.
        assert len(result) == 100
