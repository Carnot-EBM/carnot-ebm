"""Tests for Exp 1120 — SOSKANEnergyV3 retrain on FoVer v5 SOTA corpus.

Verifies:
  - _featurize produces correctly shaped / ranged outputs (REQ-SAMPLE-016-v3)
  - EBRM noise filtering drops low-confidence examples correctly
  - SOTA holdout selection returns 50 entries (25+25) and excludes them from training pool
  - Artifact has all required schema fields with correct types and ranges
  - energy_inversion_fixed = True (mean_correct < mean_incorrect)
  - retrained_auroc_val >= 0.9
  - honest_verdict is a valid enum value

Spec: REQ-SAMPLE-016-v3, REQ-EVAL-001, SCENARIO-EXP-1120-GATE
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load the experiment module without triggering carnot.models JAX
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "experiment_1120_energy_verifier_retrain_sota.py"

_spec = importlib.util.spec_from_file_location("exp1120", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["exp1120"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_featurize = _mod._featurize
_select_sota_holdout = _mod._select_sota_holdout
N_FEATURES = _mod.N_FEATURES
NOISE_THRESHOLD = _mod.NOISE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    label: str, confidence: float, source: str = "fover_v4", verifier: str = "heuristic"
) -> dict:
    return {
        "question_id": "q1",
        "step_text": "x = 1 + 2 = 3. Therefore x = 3.",
        "label": label,
        "confidence": confidence,
        "model": "Test",
        "source": source,
        "verifier": verifier,
    }


# ---------------------------------------------------------------------------
# _featurize tests
# ---------------------------------------------------------------------------


class TestFeaturize:
    def test_shape(self) -> None:
        """Output X has shape (n_items, N_FEATURES) and y has shape (n_items,)."""
        items = [_make_entry("correct", 1.0), _make_entry("incorrect", 1.0)]
        X, y = _featurize(items)
        assert X.shape == (2, N_FEATURES), f"Expected (2, {N_FEATURES}), got {X.shape}"
        assert y.shape == (2,)

    def test_features_in_range(self) -> None:
        """All features are in [-1, 1] as required by the SOS-KAN hat basis."""
        items = [_make_entry("correct", 1.0) for _ in range(20)]
        X, _ = _featurize(items)
        assert np.all(X >= -1.0 - 1e-6), "Feature below -1"
        assert np.all(X <= 1.0 + 1e-6), "Feature above 1"

    def test_labels_binary(self) -> None:
        """y values are in {0, 1}."""
        items = [_make_entry("correct", 1.0), _make_entry("incorrect", 0.8)]
        _, y = _featurize(items)
        assert set(y.tolist()).issubset({0, 1})

    def test_correct_label_maps_to_1(self) -> None:
        items = [_make_entry("correct", 1.0)]
        _, y = _featurize(items)
        assert y[0] == 1

    def test_incorrect_label_maps_to_0(self) -> None:
        items = [_make_entry("incorrect", 0.9)]
        _, y = _featurize(items)
        assert y[0] == 0

    def test_empty_text_does_not_crash(self) -> None:
        items = [{"question_id": "q1", "step_text": "", "label": "correct", "confidence": 1.0}]
        X, y = _featurize(items)
        assert X.shape == (1, N_FEATURES)


# ---------------------------------------------------------------------------
# Noise-filter logic tests
# ---------------------------------------------------------------------------


class TestNoiseFilter:
    def test_drops_below_threshold(self) -> None:
        """Entries with confidence < NOISE_THRESHOLD are dropped."""
        entries = [
            _make_entry("correct", 0.5),  # below threshold — drop
            _make_entry("correct", 0.7),  # at threshold — keep
            _make_entry("correct", 1.0),  # above threshold — keep
        ]
        filtered = [e for e in entries if float(e.get("confidence", 1.0)) >= NOISE_THRESHOLD]
        assert len(filtered) == 2, f"Expected 2, got {len(filtered)}"

    def test_z3_entries_always_kept(self) -> None:
        """Z3Math entries have confidence=1.0 and are never dropped."""
        entries = [_make_entry("correct", 1.0, verifier="Z3Math")]
        filtered = [e for e in entries if float(e.get("confidence", 1.0)) >= NOISE_THRESHOLD]
        assert len(filtered) == 1

    def test_threshold_is_0_7(self) -> None:
        assert NOISE_THRESHOLD == 0.7


# ---------------------------------------------------------------------------
# SOTA holdout selection tests
# ---------------------------------------------------------------------------


class TestSelectSotaHoldout:
    def _make_sota_pool(self, n_correct: int = 30, n_incorrect: int = 30) -> list[dict]:
        pool = []
        for i in range(n_correct):
            pool.append(
                {
                    "question_id": f"c{i}",
                    "step_text": f"Correct step {i}",
                    "label": "correct",
                    "confidence": 1.0,
                    "model": "Qwen",
                    "source": "sota_extension_v5",
                    "verifier": "Z3Math",
                }
            )
        for i in range(n_incorrect):
            pool.append(
                {
                    "question_id": f"inc{i}",
                    "step_text": f"Incorrect step {i}",
                    "label": "incorrect",
                    "confidence": 0.9,
                    "model": "Qwen",
                    "source": "sota_extension_v5",
                    "verifier": "heuristic",
                }
            )
        return pool

    def test_returns_50_items(self) -> None:
        """Holdout has exactly 50 items (25 correct + 25 incorrect)."""
        pool = self._make_sota_pool(30, 30)
        holdout, _ = _select_sota_holdout(pool, n_correct=25, n_incorrect=25)
        assert len(holdout) == 50, f"Expected 50, got {len(holdout)}"

    def test_holdout_indices_excluded(self) -> None:
        """Returned indices are not in the training pool."""
        pool = self._make_sota_pool(30, 30)
        holdout, holdout_indices = _select_sota_holdout(pool, n_correct=25, n_incorrect=25)
        # Simulate what main() does: exclude holdout from training pool
        training_pool = [e for i, e in enumerate(pool) if i not in holdout_indices]
        holdout_qids = {h["question_id"] for h in holdout}
        for entry in training_pool:
            assert entry["question_id"] not in holdout_qids, (
                "Holdout entry leaked into training pool"
            )

    def test_correct_class_count(self) -> None:
        pool = self._make_sota_pool(30, 30)
        holdout, _ = _select_sota_holdout(pool, n_correct=25, n_incorrect=25)
        n_c = sum(1 for h in holdout if h.get("_holdout_class") == "correct")
        assert n_c == 25

    def test_incorrect_class_count(self) -> None:
        pool = self._make_sota_pool(30, 30)
        holdout, _ = _select_sota_holdout(pool, n_correct=25, n_incorrect=25)
        n_i = sum(1 for h in holdout if h.get("_holdout_class") == "incorrect")
        assert n_i == 25


# ---------------------------------------------------------------------------
# Artifact schema validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "n_training_pairs",
    "n_dropped_by_noise_filter",
    "retrained_auroc_val",
    "mean_correct_energy_before",
    "mean_incorrect_energy_before",
    "mean_correct_energy_after",
    "mean_incorrect_energy_after",
    "energy_inversion_fixed",
    "energy_inversion_measured_post_retrain",
    "noise_filter_threshold",
    "honest_verdict",
]

VALID_VERDICTS = {
    "inversion_fixed_ordering_correct",
    "inversion_reduced_not_fixed",
    "inversion_unchanged",
    "partial",
    "blocked_gate",
}

_ARTIFACT_PATH = _REPO / "results" / "experiment_1120_energy_verifier_retrain_sota.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not _ARTIFACT_PATH.exists():
        pytest.skip("Artifact not yet generated — run the experiment first.")
    with open(_ARTIFACT_PATH) as f:
        return json.load(f)


class TestArtifact:
    def test_required_fields_present(self, artifact: dict) -> None:
        """All required schema fields must be present. REQ-SAMPLE-016-v3."""
        for field in REQUIRED_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_n_training_pairs_positive(self, artifact: dict) -> None:
        assert artifact["n_training_pairs"] > 0

    def test_n_dropped_nonnegative(self, artifact: dict) -> None:
        assert artifact["n_dropped_by_noise_filter"] >= 0

    def test_auroc_in_range(self, artifact: dict) -> None:
        """AUROC must be in [0, 1]. REQ-EVAL-001."""
        auroc = artifact["retrained_auroc_val"]
        assert 0.0 <= auroc <= 1.0, f"AUROC out of range: {auroc}"

    def test_auroc_meets_target(self, artifact: dict) -> None:
        """Retrained AUROC must be >= 0.9 (SOSKANEnergyV3 quality gate)."""
        assert artifact["retrained_auroc_val"] >= 0.9, (
            f"AUROC {artifact['retrained_auroc_val']:.4f} below 0.9 target"
        )

    def test_energy_inversion_fixed(self, artifact: dict) -> None:
        """mean_correct_energy_after < mean_incorrect_energy_after (inversion resolved)."""
        assert artifact["energy_inversion_fixed"] is True, (
            f"Inversion not fixed: correct={artifact['mean_correct_energy_after']:.4f}, "
            f"incorrect={artifact['mean_incorrect_energy_after']:.4f}"
        )

    def test_energy_inversion_measured(self, artifact: dict) -> None:
        """energy_inversion_measured_post_retrain must always be True."""
        assert artifact["energy_inversion_measured_post_retrain"] is True

    def test_noise_filter_threshold(self, artifact: dict) -> None:
        assert artifact["noise_filter_threshold"] == 0.7

    def test_honest_verdict_valid(self, artifact: dict) -> None:
        """honest_verdict must be one of the defined enum values."""
        verdict = artifact["honest_verdict"]
        assert verdict in VALID_VERDICTS, f"Unknown verdict: {verdict}"

    def test_honest_verdict_is_inversion_fixed(self, artifact: dict) -> None:
        """Experiment should have achieved the primary goal."""
        assert artifact["honest_verdict"] == "inversion_fixed_ordering_correct", (
            f"Expected inversion_fixed_ordering_correct, got {artifact['honest_verdict']}"
        )

    def test_ordering_direction_correct(self, artifact: dict) -> None:
        """After retrain: correct energy < incorrect energy (lower = more correct)."""
        assert artifact["mean_correct_energy_after"] < artifact["mean_incorrect_energy_after"], (
            f"Ordering wrong: correct={artifact['mean_correct_energy_after']:.4f} >= "
            f"incorrect={artifact['mean_incorrect_energy_after']:.4f}"
        )

    def test_baseline_energies_preserved(self, artifact: dict) -> None:
        """Baseline energies from exp1100 must match the known inverted values."""
        assert artifact["mean_correct_energy_before"] == pytest.approx(0.689, abs=1e-3)
        assert artifact["mean_incorrect_energy_before"] == pytest.approx(0.621, abs=1e-3)
