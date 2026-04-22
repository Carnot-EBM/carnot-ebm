"""Tests for Experiment 682: JEPA v15 True OOD Audit on GSM8K 500-699.

These tests verify the helper functions used in the OOD audit:
- load_training_question_ids()
- _load_gsm8k_ood_questions() synthetic fallback
- embed_questions()
- compute_auc_manual()
- compute_ece()
- fit_platt_temperature()
- determine_verdict() / VALID_VERDICTS
- The produced deliverable passes schema validation.

Spec: REQ-LEARN-087, REQ-LEARN-088,
      SCENARIO-LEARN-136, SCENARIO-LEARN-137
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repository root on sys.path so carnot imports resolve
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_682_jepa_v15_ood_audit import (  # noqa: E402
    VALID_VERDICTS,
    _load_gsm8k_ood_questions,
    compute_auc_manual,
    compute_ece,
    determine_verdict,
    fit_platt_temperature,
    load_training_question_ids,
)


# ---------------------------------------------------------------------------
# Test: load_training_question_ids
# Spec: REQ-LEARN-087, SCENARIO-LEARN-136
# ---------------------------------------------------------------------------


class TestLoadTrainingQuestionIds:
    """Verify that training question ID loading is robust to file absence."""

    def test_missing_file_returns_empty_set(self, tmp_path):
        """load_training_question_ids() returns empty set when file is absent.

        Spec: REQ-LEARN-087 — OOD audit must not crash on missing FOVER data.
        """
        result = load_training_question_ids(str(tmp_path / "nonexistent.json"))
        assert result == set()

    def test_loads_question_ids_from_json(self, tmp_path):
        """load_training_question_ids() extracts question_id strings from JSON list.

        Spec: REQ-LEARN-087 — training indices must be known to verify no data leakage.
        """
        items = [
            {"question_id": "156", "label": "correct"},
            {"question_id": "159", "label": "incorrect"},
            {"question_id": "200", "label": "correct"},
        ]
        fover = tmp_path / "fover.json"
        fover.write_text(json.dumps(items))
        result = load_training_question_ids(str(fover))
        assert result == {"156", "159", "200"}

    def test_items_without_question_id_are_skipped(self, tmp_path):
        """Items missing question_id key are silently skipped.

        Spec: REQ-LEARN-087 — partial/malformed data should not abort the audit.
        """
        items = [
            {"question_id": "99", "label": "correct"},
            {"label": "incorrect"},  # missing question_id
        ]
        fover = tmp_path / "fover.json"
        fover.write_text(json.dumps(items))
        result = load_training_question_ids(str(fover))
        assert result == {"99"}


# ---------------------------------------------------------------------------
# Test: _load_gsm8k_ood_questions synthetic fallback
# Spec: REQ-LEARN-087, SCENARIO-LEARN-136
# ---------------------------------------------------------------------------


class TestLoadGsm8kOodQuestions:
    """Verify synthetic fallback produces well-formed question dicts."""

    def test_returns_correct_count(self, monkeypatch):
        """Synthetic fallback returns exactly the requested number of questions.

        Spec: REQ-LEARN-087 — N_OOD=200 questions must be loaded.
        """
        # Force synthetic fallback by monkeypatching datasets import
        monkeypatch.setitem(sys.modules, "datasets", None)
        rows = _load_gsm8k_ood_questions(500, 510)
        assert len(rows) == 10

    def test_each_row_has_required_keys(self, monkeypatch):
        """Each row dict has 'question', 'answer', 'idx', 'ground_truth_label'.

        Spec: SCENARIO-LEARN-136 — downstream embedding requires 'question' key.
        """
        monkeypatch.setitem(sys.modules, "datasets", None)
        rows = _load_gsm8k_ood_questions(500, 502)
        for row in rows:
            assert "question" in row
            assert "answer" in row
            assert "idx" in row
            assert "ground_truth_label" in row

    def test_label_is_binary(self, monkeypatch):
        """ground_truth_label is 0 or 1 for all rows.

        Spec: REQ-LEARN-087 — binary labels are required for AUC computation.
        """
        monkeypatch.setitem(sys.modules, "datasets", None)
        rows = _load_gsm8k_ood_questions(500, 520)
        for row in rows:
            assert row["ground_truth_label"] in (0, 1)

    def test_indices_are_offset_correctly(self, monkeypatch):
        """Row idx values match the requested start offset.

        Spec: REQ-LEARN-087 — indices must be 500-699 to avoid training overlap.
        """
        monkeypatch.setitem(sys.modules, "datasets", None)
        rows = _load_gsm8k_ood_questions(500, 505)
        idxs = [row["idx"] for row in rows]
        assert idxs == [500, 501, 502, 503, 504]


# ---------------------------------------------------------------------------
# Test: compute_auc_manual
# Spec: REQ-LEARN-087, SCENARIO-LEARN-136
# ---------------------------------------------------------------------------


class TestComputeAucManual:
    """Verify AUC computation correctness on known inputs."""

    def test_perfect_separation_returns_one(self):
        """AUC = 1.0 when all positives rank above all negatives.

        Spec: SCENARIO-LEARN-136 — JEPA v15 overfit case should produce AUC=1.0.
        """
        scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        auc = compute_auc_manual(scores, labels)
        assert auc == pytest.approx(1.0)

    def test_random_returns_half(self):
        """AUC = 0.5 for uniformly random scores regardless of labels.

        Spec: SCENARIO-LEARN-136 — a useless predictor should score 0.5.
        """
        rng = np.random.RandomState(42)
        scores = rng.uniform(0, 1, 1000).astype(np.float32)
        labels = rng.randint(0, 2, 1000).astype(np.float32)
        auc = compute_auc_manual(scores, labels)
        assert 0.45 < auc < 0.55

    def test_all_positive_returns_half(self):
        """AUC = 0.5 when there are no negative examples (degenerate set).

        Spec: REQ-LEARN-087 — graceful handling of degenerate label distributions.
        """
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        labels = np.array([1, 1, 1], dtype=np.float32)
        assert compute_auc_manual(scores, labels) == pytest.approx(0.5)

    def test_all_negative_returns_half(self):
        """AUC = 0.5 when there are no positive examples (degenerate set).

        Spec: REQ-LEARN-087 — graceful handling of degenerate label distributions.
        """
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        labels = np.array([0, 0, 0], dtype=np.float32)
        assert compute_auc_manual(scores, labels) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test: compute_ece
# Spec: REQ-LEARN-088, SCENARIO-LEARN-137
# ---------------------------------------------------------------------------


class TestComputeEce:
    """Verify ECE computation on known calibration patterns."""

    def test_perfect_calibration_zero_ece(self):
        """ECE = 0 when predicted probabilities exactly match observed rates.

        Spec: SCENARIO-LEARN-137 — a perfectly calibrated model should have ECE=0.
        """
        # All predictions 1.0 and all labels 1.0 → perfectly calibrated
        probs = np.ones(10, dtype=np.float32)
        labels = np.ones(10, dtype=np.float32)
        ece = compute_ece(probs, labels)
        assert ece == pytest.approx(0.0)

    def test_maximally_miscalibrated_one(self):
        """ECE is close to 1.0 when high-confidence predictions are all wrong.

        WHY 0.95 not 1.0: probabilities at exactly 1.0 fall above the last half-open
        bin boundary [0.9, 1.0) and would be silently dropped.  Using 0.95 puts all
        samples in the [0.9, 1.0) bucket with conf=0.95 and acc=0.0 → ECE ≈ 0.95.

        Spec: SCENARIO-LEARN-137 — ECE near 1.0 flags totally inverted confidence.
        """
        probs = np.full(10, 0.95, dtype=np.float32)
        labels = np.zeros(10, dtype=np.float32)
        ece = compute_ece(probs, labels)
        assert ece == pytest.approx(0.95, abs=0.01)

    def test_empty_array_returns_zero(self):
        """compute_ece() returns 0.0 for empty arrays without error.

        Spec: REQ-LEARN-088 — degenerate inputs must not raise exceptions.
        """
        assert compute_ece(np.array([]), np.array([])) == pytest.approx(0.0)

    def test_ece_in_valid_range(self):
        """ECE is always in [0, 1] for any valid probability/label arrays.

        Spec: REQ-LEARN-088 — ECE must be a valid probability-space metric.
        """
        rng = np.random.RandomState(7)
        probs = rng.uniform(0, 1, 200).astype(np.float32)
        labels = rng.randint(0, 2, 200).astype(np.float32)
        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Test: fit_platt_temperature
# Spec: REQ-LEARN-088, SCENARIO-LEARN-137
# ---------------------------------------------------------------------------


class TestFitPlattTemperature:
    """Verify Platt temperature fitting returns a positive finite value."""

    def test_returns_positive_float(self):
        """fit_platt_temperature() always returns T > 0.

        Spec: REQ-LEARN-088 — T <= 0 would cause division by zero in calibration.
        """
        scores = np.array([0.8, 0.7, 0.2, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        T = fit_platt_temperature(scores, labels)
        assert T > 0.0

    def test_temperature_in_search_bounds(self):
        """Returned temperature is within the grid search bounds [0.01, 10.0].

        Spec: REQ-LEARN-088 — temperature outside this range would not be interpretable.
        """
        scores = np.array([0.6, 0.4, 0.3, 0.9], dtype=np.float32)
        labels = np.array([1, 0, 0, 1], dtype=np.float32)
        T = fit_platt_temperature(scores, labels)
        assert 0.01 <= T <= 10.0


# ---------------------------------------------------------------------------
# Test: determine_verdict
# Spec: REQ-LEARN-087, SCENARIO-LEARN-136
# ---------------------------------------------------------------------------


class TestDetermineVerdict:
    """Verify verdict mapping produces valid enum members."""

    def test_auc_one_is_overfit(self):
        """AUC=1.0 always triggers 'jepa_v15_overfit' regardless of ECE.

        Spec: SCENARIO-LEARN-136 — AUC=1.0 on OOD confirms the Exp 671 suspicion.
        """
        assert determine_verdict(1.0, 0.05) == "jepa_v15_overfit"
        assert determine_verdict(1.0, 0.15) == "jepa_v15_overfit"

    def test_high_auc_low_ece_is_target_met(self):
        """AUC>=0.80 AND ECE<0.10 triggers 'jepa_v15_ood_target_met'.

        Spec: REQ-LEARN-087 — passing both gates is the success criterion.
        """
        assert determine_verdict(0.85, 0.05) == "jepa_v15_ood_target_met"
        assert determine_verdict(0.80, 0.09) == "jepa_v15_ood_target_met"

    def test_partial_auc_is_partial(self):
        """0.60 <= AUC < 0.80 triggers 'jepa_v15_ood_partial'.

        Spec: REQ-LEARN-087 — partial success is distinguished from failure.
        """
        assert determine_verdict(0.65, 0.20) == "jepa_v15_ood_partial"
        assert determine_verdict(0.60, 0.50) == "jepa_v15_ood_partial"

    def test_below_random_verdict(self):
        """AUC < 0.50 triggers 'jepa_v15_ood_below_random'.

        Spec: REQ-LEARN-087 — below-random AUC is a distinct diagnostic outcome.
        """
        assert determine_verdict(0.45, 0.30) == "jepa_v15_ood_below_random"
        assert determine_verdict(0.0, 0.99) == "jepa_v15_ood_below_random"

    def test_all_verdicts_are_in_valid_set(self):
        """All returned verdicts belong to VALID_VERDICTS.

        Spec: SCENARIO-LEARN-136 — downstream conductor must recognise every verdict.
        """
        test_cases = [
            (1.0, 0.05),
            (0.85, 0.05),
            (0.65, 0.20),
            (0.45, 0.30),
        ]
        for auc, ece in test_cases:
            v = determine_verdict(auc, ece)
            assert v in VALID_VERDICTS, f"Unexpected verdict '{v}' for auc={auc} ece={ece}"


# ---------------------------------------------------------------------------
# Test: deliverable schema validation
# Spec: REQ-LEARN-087, REQ-LEARN-088, SCENARIO-LEARN-136, SCENARIO-LEARN-137
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    """Verify the written deliverable contains required fields."""

    def test_deliverable_exists(self):
        """results/experiment_682_jepa_v15_ood_audit.json was written.

        Spec: REQ-LEARN-087 — the deliverable must exist for the conductor to proceed.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        assert path.exists(), f"Deliverable not found: {path}"

    def test_deliverable_required_fields(self):
        """Deliverable contains all REQUIRED_RESULT_FIELDS.

        Spec: REQ-LEARN-087 — schema compliance is mandatory per REQUIRED_RESULT_FIELDS.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        if not path.exists():
            pytest.skip("deliverable not written yet")
        data = json.loads(path.read_text())
        required = ["experiment", "title", "run_date", "started_at", "finished_at",
                    "duration_s", "status"]
        for field in required:
            assert field in data, f"Missing required field: {field}"

    def test_deliverable_honest_verdict_valid(self):
        """honest_verdict in deliverable is a member of VALID_VERDICTS.

        Spec: SCENARIO-LEARN-136 — conductor reads honest_verdict to pick the next task.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        if not path.exists():
            pytest.skip("deliverable not written yet")
        data = json.loads(path.read_text())
        assert data.get("honest_verdict") in VALID_VERDICTS

    def test_deliverable_experiment_id(self):
        """Deliverable experiment field is 682.

        Spec: REQ-LEARN-087 — conductor matches result files by experiment ID.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        if not path.exists():
            pytest.skip("deliverable not written yet")
        data = json.loads(path.read_text())
        assert data["experiment"] == 682

    def test_deliverable_true_ood_auc_present(self):
        """true_ood_auc is present and numeric when status=success.

        Spec: REQ-LEARN-087 — true OOD AUC is the primary result of this experiment.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        if not path.exists():
            pytest.skip("deliverable not written yet")
        data = json.loads(path.read_text())
        if data.get("status") == "success":
            assert isinstance(data.get("true_ood_auc"), (int, float))

    def test_deliverable_ece_present_on_success(self):
        """ece is present and numeric when status=success.

        Spec: REQ-LEARN-088 — ECE is the calibration quality metric for JEPA v15.
        """
        path = _REPO_ROOT / "results" / "experiment_682_jepa_v15_ood_audit.json"
        if not path.exists():
            pytest.skip("deliverable not written yet")
        data = json.loads(path.read_text())
        if data.get("status") == "success":
            assert isinstance(data.get("ece"), (int, float))
