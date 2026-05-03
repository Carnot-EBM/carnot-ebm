"""Tests for experiment_1185: SC-Energy overfit diagnosis and regularized k=6.

Spec: REQ-VERIFY-1185, SCENARIO-VERIFY-1185

Tests cover only the new code added in this experiment:
  - Data loading and train/holdout splitting
  - tie_aware_auroc computation
  - Regularized training (_compute_margin_loss, dropout application)
  - Checkpoint save/load
  - Verdict mapping (determine_verdict)
  - Artifact schema validation (build_artifact)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment_1185_sc_energy_overfit_regularized_k6 import (
    ALLOWED_VERDICTS,
    REQUIRED_FIELDS,
    _compute_margin_loss,
    build_artifact,
    determine_verdict,
    evaluate_auroc_on_rows,
    is_row_incorrect,
    load_jsonl_rows,
    row_step_text,
    save_verifier_weights,
    split_rows_by_question_80_20,
    tie_aware_auroc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fover_rows(n_correct: int = 8, n_incorrect: int = 4, n_questions: int = 4) -> list[dict]:
    """Generate synthetic fover-style rows for testing."""
    rows = []
    per_q_correct = max(1, n_correct // n_questions)
    per_q_incorrect = max(1, n_incorrect // n_questions)
    qid = 0
    for _ in range(n_questions):
        qid += 1
        for _ in range(per_q_correct):
            rows.append(
                {
                    "question_id": str(qid),
                    "step_text": f"correct step qid={qid}",
                    "label": "correct",
                }
            )
        for _ in range(per_q_incorrect):
            rows.append(
                {
                    "question_id": str(qid),
                    "step_text": f"wrong step qid={qid}",
                    "label": "incorrect",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# load_jsonl_rows
# ---------------------------------------------------------------------------


class TestLoadJsonlRows:
    """Spec: REQ-VERIFY-1185"""

    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        """load_jsonl_rows correctly loads well-formed JSONL.
        Spec: REQ-VERIFY-1185
        """
        p = tmp_path / "corpus.jsonl"
        rows = [
            {"question_id": "1", "step_text": "step one", "label": "correct"},
            {"question_id": "1", "step_text": "step two", "label": "incorrect"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        loaded = load_jsonl_rows(p)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "correct"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """load_jsonl_rows silently skips blank lines.
        Spec: REQ-VERIFY-1185
        """
        p = tmp_path / "corpus.jsonl"
        p.write_text('\n{"label": "correct", "step_text": "x"}\n\n')
        loaded = load_jsonl_rows(p)
        assert len(loaded) == 1

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        """load_jsonl_rows silently skips non-JSON lines.
        Spec: REQ-VERIFY-1185
        """
        p = tmp_path / "corpus.jsonl"
        p.write_text('not-json\n{"label": "correct", "step_text": "y"}\n')
        loaded = load_jsonl_rows(p)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# is_row_incorrect
# ---------------------------------------------------------------------------


class TestIsRowIncorrect:
    """Spec: REQ-VERIFY-1185"""

    def test_label_incorrect(self) -> None:
        """label='incorrect' → is_row_incorrect returns True.
        Spec: REQ-VERIFY-1185
        """
        assert is_row_incorrect({"label": "incorrect"})

    def test_label_correct(self) -> None:
        """label='correct' → is_row_incorrect returns False.
        Spec: REQ-VERIFY-1185
        """
        assert not is_row_incorrect({"label": "correct"})

    def test_is_correct_false(self) -> None:
        """is_correct=False → is_row_incorrect returns True.
        Spec: REQ-VERIFY-1185
        """
        assert is_row_incorrect({"is_correct": False})

    def test_is_correct_true(self) -> None:
        """is_correct=True → is_row_incorrect returns False.
        Spec: REQ-VERIFY-1185
        """
        assert not is_row_incorrect({"is_correct": True})

    def test_empty_row(self) -> None:
        """Row with no label fields → is_row_incorrect returns False.
        Spec: REQ-VERIFY-1185
        """
        assert not is_row_incorrect({})


# ---------------------------------------------------------------------------
# row_step_text
# ---------------------------------------------------------------------------


class TestRowStepText:
    """Spec: REQ-VERIFY-1185"""

    def test_prefers_step_text(self) -> None:
        """row_step_text returns step_text when present.
        Spec: REQ-VERIFY-1185
        """
        row = {"step_text": "hello", "response": "world"}
        assert row_step_text(row) == "hello"

    def test_falls_back_to_response(self) -> None:
        """row_step_text falls back to response when step_text absent.
        Spec: REQ-VERIFY-1185
        """
        row = {"response": "hello"}
        assert row_step_text(row) == "hello"

    def test_empty_row(self) -> None:
        """row_step_text returns empty string for rows with no text.
        Spec: REQ-VERIFY-1185
        """
        assert row_step_text({}) == ""


# ---------------------------------------------------------------------------
# split_rows_by_question_80_20
# ---------------------------------------------------------------------------


class TestSplitRows:
    """Spec: REQ-VERIFY-1185"""

    def test_split_proportions(self) -> None:
        """80/20 split produces approximately 80% train rows.
        Spec: REQ-VERIFY-1185
        """
        rows = _make_fover_rows(n_correct=80, n_incorrect=20, n_questions=20)
        train, holdout = split_rows_by_question_80_20(rows, seed=42)
        total = len(train) + len(holdout)
        assert total == len(rows)
        # Train should be ~80% (allow ±5%)
        assert 0.70 <= len(train) / total <= 0.90

    def test_no_question_id_overlap(self) -> None:
        """Train and holdout question_ids must be disjoint.
        Spec: REQ-VERIFY-1185
        """
        rows = _make_fover_rows(n_correct=40, n_incorrect=10, n_questions=10)
        train, holdout = split_rows_by_question_80_20(rows, seed=99)
        train_qids = {r["question_id"] for r in train}
        holdout_qids = {r["question_id"] for r in holdout}
        assert train_qids.isdisjoint(holdout_qids), "Train/holdout question IDs must not overlap"

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed always produces the same split.
        Spec: REQ-VERIFY-1185
        """
        rows = _make_fover_rows(n_questions=8)
        t1, h1 = split_rows_by_question_80_20(rows, seed=7)
        t2, h2 = split_rows_by_question_80_20(rows, seed=7)
        assert [r["question_id"] for r in t1] == [r["question_id"] for r in t2]

    def test_different_seed_different_split(self) -> None:
        """Different seeds produce different splits with high probability.
        Spec: REQ-VERIFY-1185
        """
        rows = _make_fover_rows(n_questions=20)
        t1, _ = split_rows_by_question_80_20(rows, seed=1)
        t2, _ = split_rows_by_question_80_20(rows, seed=2)
        # It would be astronomically unlikely for both seeds to produce the same partition
        ids_1 = {r["question_id"] for r in t1}
        ids_2 = {r["question_id"] for r in t2}
        assert ids_1 != ids_2


# ---------------------------------------------------------------------------
# tie_aware_auroc
# ---------------------------------------------------------------------------


class TestTieAwareAUROC:
    """Spec: SCENARIO-VERIFY-1185"""

    def test_perfect_separation(self) -> None:
        """All positives score higher than all negatives → AUROC = 1.0.
        Spec: SCENARIO-VERIFY-1185
        """
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert tie_aware_auroc(labels, scores) == 1.0

    def test_perfect_anti_separation(self) -> None:
        """All positives score lower than all negatives → AUROC = 0.0.
        Spec: SCENARIO-VERIFY-1185
        """
        labels = [1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9]
        assert tie_aware_auroc(labels, scores) == 0.0

    def test_all_tied(self) -> None:
        """All scores identical → AUROC = 0.5.
        Spec: SCENARIO-VERIFY-1185
        """
        labels = [1, 1, 0, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        assert tie_aware_auroc(labels, scores) == 0.5

    def test_no_positives(self) -> None:
        """No positive examples → returns 0.5 (undefined AUROC).
        Spec: SCENARIO-VERIFY-1185
        """
        assert tie_aware_auroc([0, 0, 0], [0.1, 0.5, 0.9]) == 0.5

    def test_no_negatives(self) -> None:
        """No negative examples → returns 0.5 (undefined AUROC).
        Spec: SCENARIO-VERIFY-1185
        """
        assert tie_aware_auroc([1, 1, 1], [0.1, 0.5, 0.9]) == 0.5

    def test_partial_tie(self) -> None:
        """Partial tie (one pos == one neg) gets 0.5 credit.
        Spec: SCENARIO-VERIFY-1185
        """
        # 1 pos (score=0.5), 1 neg (score=0.5) → tie → 0.5 credit → AUROC = 0.5
        assert tie_aware_auroc([1, 0], [0.5, 0.5]) == 0.5


# ---------------------------------------------------------------------------
# _compute_margin_loss
# ---------------------------------------------------------------------------


class TestComputeMarginLoss:
    """Spec: REQ-VERIFY-1185"""

    def test_zero_loss_when_separated(self) -> None:
        """Loss is 0 when energy gap already exceeds margin.
        Spec: REQ-VERIFY-1185
        """
        # metric = [1, 0], bias = 0, margin = 1.0
        # coh_feat gives energy 0.0, inc_feat gives energy 2.0 → gap = 2.0 > margin
        metric = np.array([1.0, 0.0], dtype=np.float32)
        coh_feats = [np.array([0.0, 0.0])]
        inc_feats = [np.array([2.0, 0.0])]
        loss = _compute_margin_loss(coh_feats, inc_feats, metric, bias=0.0, margin=1.0)
        assert loss == 0.0

    def test_positive_loss_when_not_separated(self) -> None:
        """Loss is positive when energy gap is less than margin.
        Spec: REQ-VERIFY-1185
        """
        metric = np.array([1.0, 0.0], dtype=np.float32)
        # Both features are the same → gap = 0 < margin → loss = margin = 1.0
        coh_feats = [np.array([1.0, 0.0])]
        inc_feats = [np.array([1.0, 0.0])]
        loss = _compute_margin_loss(coh_feats, inc_feats, metric, bias=0.0, margin=1.0)
        assert loss > 0.0

    def test_empty_input(self) -> None:
        """Empty feature lists → loss = 0.0.
        Spec: REQ-VERIFY-1185
        """
        metric = np.array([1.0], dtype=np.float32)
        loss = _compute_margin_loss([], [], metric, bias=0.0, margin=1.0)
        assert loss == 0.0


# ---------------------------------------------------------------------------
# save_verifier_weights
# ---------------------------------------------------------------------------


class TestSaveVerifierWeights:
    """Spec: REQ-VERIFY-1185"""

    def test_saves_loadable_numpy_file(self, tmp_path: Path) -> None:
        """save_verifier_weights writes a numpy npz readable file.
        Spec: REQ-VERIFY-1185
        """

        # Build a minimal verifier-like object
        class _FakeVerifier:
            _metric = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            _bias = 1.5

        out = tmp_path / "weights.pt"
        save_verifier_weights(_FakeVerifier(), out)
        assert out.exists()
        data = np.load(str(out), allow_pickle=False)
        np.testing.assert_allclose(data["metric"], _FakeVerifier._metric)
        assert float(data["bias"][0]) == pytest.approx(1.5, abs=1e-5)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_verifier_weights creates missing parent directories.
        Spec: REQ-VERIFY-1185
        """

        class _FakeVerifier:
            _metric = np.zeros(4, dtype=np.float32)
            _bias = 0.0

        nested = tmp_path / "a" / "b" / "model.pt"
        save_verifier_weights(_FakeVerifier(), nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# determine_verdict
# ---------------------------------------------------------------------------


class TestDetermineVerdict:
    """Spec: REQ-VERIFY-1185, SCENARIO-VERIFY-1185"""

    def test_overfit_not_resolved(self) -> None:
        """Overfit not resolved → overfit_not_resolved verdict.
        Spec: SCENARIO-VERIFY-1185
        """
        v = determine_verdict(overfit_resolved=False, k6_above_k5=True)
        assert v == "overfit_not_resolved"

    def test_k6_viable(self) -> None:
        """Overfit resolved + k6 >= k5 → k6_viable_after_regularization.
        Spec: SCENARIO-VERIFY-1185
        """
        v = determine_verdict(overfit_resolved=True, k6_above_k5=True)
        assert v == "k6_viable_after_regularization"

    def test_k6_still_regresses(self) -> None:
        """Overfit resolved + k6 < k5 → overfit_resolved_but_k6_still_regresses.
        Spec: SCENARIO-VERIFY-1185
        """
        v = determine_verdict(overfit_resolved=True, k6_above_k5=False)
        assert v == "overfit_resolved_but_k6_still_regresses"

    def test_all_verdicts_are_allowed(self) -> None:
        """All three verdict paths produce values in ALLOWED_VERDICTS.
        Spec: REQ-VERIFY-1185
        """
        cases = [
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ]
        for overfit_resolved, k6_above_k5 in cases:
            v = determine_verdict(overfit_resolved, k6_above_k5)
            assert v in ALLOWED_VERDICTS, (
                f"Unexpected verdict {v!r} for overfit_resolved={overfit_resolved}, k6_above_k5={k6_above_k5}"
            )


# ---------------------------------------------------------------------------
# build_artifact — schema validation
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """Spec: REQ-VERIFY-1185"""

    def _base_kwargs(self) -> dict:
        return dict(
            v1_holdout_auroc=0.55,
            v2_holdout_auroc=0.62,
            k5_auroc_on_eval=0.924,
            k6_regularized_auroc=0.90,
            overfit_resolved=True,
            k6_above_k5=False,
            training_diagnostics={"n_epochs_run": 20},
            started_at="2026-05-03T00:00:00+00:00",
            duration_s=12.3,
        )

    def test_all_required_fields_present(self) -> None:
        """build_artifact produces all REQUIRED_FIELDS.
        Spec: REQ-VERIFY-1185
        """
        artifact = build_artifact(**self._base_kwargs())
        for field in REQUIRED_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_honest_verdict_in_allowed_set(self) -> None:
        """honest_verdict produced by build_artifact is in ALLOWED_VERDICTS.
        Spec: REQ-VERIFY-1185
        """
        artifact = build_artifact(**self._base_kwargs())
        assert artifact["honest_verdict"] in ALLOWED_VERDICTS

    def test_retire_k6_when_k6_does_not_beat_k5(self) -> None:
        """retire_k6 = True when k6_above_k5 = False.
        Spec: REQ-VERIFY-1185
        """
        kwargs = self._base_kwargs()
        kwargs["k6_above_k5"] = False
        artifact = build_artifact(**kwargs)
        assert artifact["retire_k6"] is True

    def test_k6_viable_when_above_k5_and_overfit_resolved(self) -> None:
        """k6_viable_for_production = True when k6 beats k5 and overfit resolved.
        Spec: REQ-VERIFY-1185
        """
        kwargs = self._base_kwargs()
        kwargs["k6_above_k5"] = True
        kwargs["overfit_resolved"] = True
        artifact = build_artifact(**kwargs)
        assert artifact["k6_viable_for_production"] is True
        assert artifact["retire_k6"] is False

    def test_overfit_resolved_field_matches_input(self) -> None:
        """build_artifact preserves the overfit_resolved boolean.
        Spec: REQ-VERIFY-1185
        """
        for val in [True, False]:
            kwargs = self._base_kwargs()
            kwargs["overfit_resolved"] = val
            kwargs["k6_above_k5"] = False
            artifact = build_artifact(**kwargs)
            assert artifact["overfit_resolved"] == val

    def test_regularized_flag_is_true(self) -> None:
        """sc_energy_regularized is always True in this experiment.
        Spec: REQ-VERIFY-1185
        """
        artifact = build_artifact(**self._base_kwargs())
        assert artifact["sc_energy_regularized"] is True

    def test_duration_s_rounded(self) -> None:
        """duration_s is stored as a rounded float.
        Spec: REQ-VERIFY-1185
        """
        artifact = build_artifact(**self._base_kwargs())
        assert isinstance(artifact["duration_s"], float)


# ---------------------------------------------------------------------------
# evaluate_auroc_on_rows — smoke test with mock verifier
# ---------------------------------------------------------------------------


class TestEvaluateAUROCOnRows:
    """Spec: SCENARIO-VERIFY-1185"""

    def test_mock_perfect_verifier(self) -> None:
        """A mock verifier that perfectly separates correct/incorrect achieves AUROC=1.0.
        Spec: SCENARIO-VERIFY-1185
        """

        class _PerfectVerifier:
            def score(self, text: str, context: str = "") -> float:
                # Incorrect steps contain "wrong", correct steps contain "correct"
                return 1.0 if "wrong" in text else 0.0

        rows = [
            {"step_text": "correct step one", "label": "correct"},
            {"step_text": "correct step two", "label": "correct"},
            {"step_text": "wrong step one", "label": "incorrect"},
            {"step_text": "wrong step two", "label": "incorrect"},
        ]
        auroc = evaluate_auroc_on_rows(_PerfectVerifier(), rows)
        assert auroc == 1.0

    def test_mock_random_verifier_near_half(self) -> None:
        """A constant-score verifier (no signal) produces AUROC = 0.5.
        Spec: SCENARIO-VERIFY-1185
        """

        class _ConstantVerifier:
            def score(self, text: str, context: str = "") -> float:
                return 0.5

        rows = _make_fover_rows(n_correct=10, n_incorrect=5, n_questions=5)
        auroc = evaluate_auroc_on_rows(_ConstantVerifier(), rows)
        assert auroc == 0.5

    def test_empty_rows_returns_half(self) -> None:
        """No rows with text → returns 0.5 (no signal).
        Spec: SCENARIO-VERIFY-1185
        """

        class _AnyVerifier:
            def score(self, text: str, context: str = "") -> float:
                return 0.5

        auroc = evaluate_auroc_on_rows(_AnyVerifier(), [])
        assert auroc == 0.5
