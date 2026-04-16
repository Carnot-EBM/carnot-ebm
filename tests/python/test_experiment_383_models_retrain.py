"""Tests for scripts/experiment_383_models_retrain.py — 100% coverage.

Covers:
  - _evaluate_eorm_auc: empty pairs, degenerate labels, normal AUC computation
  - _pairs_to_contrastive_triples: synthetic pool, real questions, empty result
  - _load_jepa_pairs_from_files: missing files, empty responses, valid layout A
  - _combined_honest_verdict: all 5 outcome combinations
  - run_experiment: insufficient_pairs path (default with empty live files)
  - run_experiment: real data path (injected pairs via mocked load functions)
  - main(): full artifact written to disk

All tests run on CPU (JAX_PLATFORMS=cpu).  No live GPU required.

Spec: REQ-LEARN-025, SCENARIO-LEARN-048
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_383_models_retrain.py"

# Ensure source paths are available
for _d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# Module loader (avoids running main() at import time)
# ---------------------------------------------------------------------------


def _load_module() -> Any:
    """Load experiment_383 script as a Python module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_383", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_383"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_module()

_evaluate_eorm_auc = _mod._evaluate_eorm_auc
_pairs_to_contrastive_triples = _mod._pairs_to_contrastive_triples
_load_jepa_pairs_from_files = _mod._load_jepa_pairs_from_files
_combined_honest_verdict = _mod._combined_honest_verdict
run_experiment = _mod.run_experiment
main = _mod.main

# Import helpers from the library
from carnot.embeddings.jepa_retrain import ViolationPair, _make_synthetic_pairs
from carnot.models.eorm import EORMModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_violation_pairs(n: int = 20, seed: int = 1) -> list[ViolationPair]:
    """Return n synthetic ViolationPairs balanced 50/50 violation/non-violation."""
    return _make_synthetic_pairs(n=n, seed=seed)


def _make_fresh_eorm() -> EORMModel:
    """Return a tiny EORMModel for fast CPU tests."""
    import jax.random as jr
    return EORMModel(embed_dim=32, n_heads=2, n_layers=1, key=jr.PRNGKey(0))


# ---------------------------------------------------------------------------
# Tests: _evaluate_eorm_auc
# ---------------------------------------------------------------------------


class TestEvaluateEormAuc:
    def test_empty_pairs_returns_half(self) -> None:
        """Spec: SCENARIO-LEARN-048 — empty set gives 0.5 baseline."""
        model = _make_fresh_eorm()
        assert _evaluate_eorm_auc(model, []) == 0.5

    def test_all_same_label_returns_half(self) -> None:
        """Degenerate: all violations → no discrimination possible → 0.5."""
        model = _make_fresh_eorm()
        pairs = [
            ViolationPair(
                partial_response="foo",
                full_response="foo bar",
                has_violation=True,
                model_id="m",
                question_id=f"q{i}",
            )
            for i in range(5)
        ]
        assert _evaluate_eorm_auc(model, pairs) == 0.5

    def test_returns_float_in_unit_interval(self) -> None:
        """Normal path: AUC must be in [0, 1]."""
        model = _make_fresh_eorm()
        pairs = _make_violation_pairs(10)
        auc = _evaluate_eorm_auc(model, pairs)
        assert 0.0 <= auc <= 1.0

    def test_uses_negated_energy_as_score(self) -> None:
        """score = -energy; AUC=1.0 requires violation to have LOWEST energy.

        The scoring convention in _evaluate_eorm_auc is score = -energy.
        For AUC=1.0 (violation=positive class has highest score), the violation
        response must have the LOWEST energy (score = -energy is then highest).
        This is the correct EORM convention: low energy = "model considers this
        a violation" path when the sign is flipped for AUC computation.
        """
        model = _make_fresh_eorm()

        def patched_energy(cot_input):  # type: ignore[override]
            # violation response gets LOW energy → score = -energy = HIGH
            # correct response gets HIGH energy → score = -energy = LOW
            return -10.0 if cot_input.question_text == "q_viol" else 10.0

        model.energy = patched_energy  # type: ignore[method-assign]
        pairs = [
            ViolationPair("p", "f", True, "m", "q_viol"),    # violation, low energy
            ViolationPair("p", "f", False, "m", "q_correct"), # correct, high energy
        ]
        auc = _evaluate_eorm_auc(model, pairs)
        # Perfect discrimination: violation gets highest score (-(-10)=10 > -(10)=-10)
        assert auc == 1.0


# ---------------------------------------------------------------------------
# Tests: _pairs_to_contrastive_triples
# ---------------------------------------------------------------------------


class TestPairsToContrastiveTriples:
    def test_empty_returns_empty(self) -> None:
        assert _pairs_to_contrastive_triples([]) == []

    def test_only_violations_returns_empty(self) -> None:
        """Cannot form a triple without at least one correct entry."""
        pairs = [
            ViolationPair("p", "wrong response", True, "m", "q1"),
            ViolationPair("p", "also wrong", True, "m", "q1"),
        ]
        assert _pairs_to_contrastive_triples(pairs) == []

    def test_only_correct_returns_empty(self) -> None:
        pairs = [
            ViolationPair("p", "correct response", False, "m", "q1"),
        ]
        assert _pairs_to_contrastive_triples(pairs) == []

    def test_real_question_forms_triple(self) -> None:
        """One correct + one incorrect for the same real question_id → one triple."""
        pairs = [
            ViolationPair("p", "correct", False, "m", "gsm8k_q001"),
            ViolationPair("p", "wrong", True, "m", "gsm8k_q001"),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 1
        correct_resp, incorrect_resp, q_id = triples[0]
        assert correct_resp == "correct"
        assert incorrect_resp == "wrong"
        assert q_id == "gsm8k_q001"

    def test_synthetic_pairs_pooled(self) -> None:
        """synthetic_* question IDs share a pool so cross-product triples form."""
        pairs = [
            ViolationPair("p", "correct1", False, "m", "synthetic_q000"),
            ViolationPair("p", "correct2", False, "m", "synthetic_q001"),
            ViolationPair("p", "wrong1", True, "m", "synthetic_q002"),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        # 2 corrects × 1 incorrect = 2 triples (round-robin)
        assert len(triples) == 2

    def test_unknown_question_pooled(self) -> None:
        """question_id='unknown' also goes to the synthetic pool."""
        pairs = [
            ViolationPair("p", "correct", False, "m", "unknown"),
            ViolationPair("p", "wrong", True, "m", "unknown"),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 1

    def test_round_robin_with_unequal_counts(self) -> None:
        """When corrects > incorrects, triples = max(len(corrects), len(incorrects))."""
        pairs = [
            ViolationPair("p", "c1", False, "m", "gsm_q1"),
            ViolationPair("p", "c2", False, "m", "gsm_q1"),
            ViolationPair("p", "c3", False, "m", "gsm_q1"),
            ViolationPair("p", "w1", True, "m", "gsm_q1"),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        # max(3, 1) = 3
        assert len(triples) == 3


# ---------------------------------------------------------------------------
# Tests: _load_jepa_pairs_from_files
# ---------------------------------------------------------------------------


class TestLoadJepaPairsFromFiles:
    def test_missing_files_return_empty(self) -> None:
        assert _load_jepa_pairs_from_files(["/nonexistent/path.json"]) == []

    def test_empty_responses_key_returns_empty(self, tmp_path: Path) -> None:
        """An empty responses list → no pairs (no synthetic fallback from this fn)."""
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"responses": []}))
        result = _load_jepa_pairs_from_files([str(f)])
        assert result == []

    def test_missing_responses_key_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"status": "blocked"}))
        result = _load_jepa_pairs_from_files([str(f)])
        assert result == []

    def test_valid_layout_a_extracts_pairs(self, tmp_path: Path) -> None:
        """Layout A with real responses → pairs extracted with prefix_fraction=0.5."""
        payload = {
            "responses": [
                {
                    "question_id": "q1",
                    "model_id": "gemma4",
                    "response": "word1 word2 word3 word4",
                    "correct": True,
                },
                {
                    "question_id": "q2",
                    "model_id": "gemma4",
                    "response": "step one step two bad answer",
                    "correct": False,
                },
            ]
        }
        f = tmp_path / "exp_real.json"
        f.write_text(json.dumps(payload))
        pairs = _load_jepa_pairs_from_files([str(f)])
        assert len(pairs) == 2
        # First pair: correct → has_violation=False
        assert pairs[0].has_violation is False
        # Second pair: incorrect → has_violation=True
        assert pairs[1].has_violation is True
        # Partial prefix should be first half of words (≈2/5 words, at least 1)
        assert len(pairs[0].partial_response.split()) >= 1

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{")
        assert _load_jepa_pairs_from_files([str(f)]) == []

    def test_multiple_files_concatenated(self, tmp_path: Path) -> None:
        """Pairs from multiple files are concatenated."""
        def _make_file(name: str, q_id: str) -> Path:
            p = tmp_path / name
            p.write_text(json.dumps({
                "responses": [
                    {"question_id": q_id, "model_id": "m", "response": "a b c d", "correct": True},
                    {"question_id": q_id, "model_id": "m", "response": "x y z w", "correct": False},
                ]
            }))
            return p

        f1 = _make_file("exp1.json", "q1")
        f2 = _make_file("exp2.json", "q2")
        pairs = _load_jepa_pairs_from_files([str(f1), str(f2)])
        assert len(pairs) == 4


# ---------------------------------------------------------------------------
# Tests: _combined_honest_verdict
# ---------------------------------------------------------------------------


class TestCombinedHonestVerdict:
    def test_both_improved(self) -> None:
        assert _combined_honest_verdict("improved", "improved") == "both_improved"

    def test_eorm_only(self) -> None:
        assert _combined_honest_verdict("improved", "no_improvement") == "eorm_only"

    def test_jepa_only(self) -> None:
        assert _combined_honest_verdict("no_improvement", "improved") == "jepa_only"

    def test_neither_improved(self) -> None:
        assert _combined_honest_verdict("no_improvement", "no_improvement") == "neither_improved"

    def test_eorm_insufficient(self) -> None:
        assert _combined_honest_verdict("insufficient_real_pairs", "improved") == "insufficient_pairs"

    def test_jepa_insufficient(self) -> None:
        assert _combined_honest_verdict("improved", "insufficient_real_pairs") == "insufficient_pairs"

    def test_both_insufficient(self) -> None:
        assert _combined_honest_verdict(
            "insufficient_real_pairs", "insufficient_real_pairs"
        ) == "insufficient_pairs"


# ---------------------------------------------------------------------------
# Tests: run_experiment — insufficient pairs path
# ---------------------------------------------------------------------------


class TestRunExperimentInsufficientPairs:
    """When live files have no real pairs, both models skip retrain."""

    def test_artifact_schema_and_required_fields(self, tmp_path: Path) -> None:
        """Spec: SCENARIO-LEARN-048 — artifact has all required keys."""
        artifact = run_experiment(repo_root=tmp_path)
        required = [
            "experiment", "schema", "run_date", "started_at", "finished_at",
            "duration_s", "status",
            "n_eorm_pairs", "eorm_before_auc", "eorm_after_auc",
            "eorm_improvement", "eorm_verdict",
            "n_jepa_pairs", "jepa_before_auc", "jepa_after_auc",
            "jepa_improvement", "jepa_verdict",
            "retrain_mode", "honest_verdict",
        ]
        for key in required:
            assert key in artifact, f"Missing key: {key}"

    def test_experiment_id_is_383(self, tmp_path: Path) -> None:
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["experiment"] == 383

    def test_insufficient_pairs_verdicts(self, tmp_path: Path) -> None:
        """With no live files, both models return insufficient_real_pairs."""
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["eorm_verdict"] == "insufficient_real_pairs"
        assert artifact["jepa_verdict"] == "insufficient_real_pairs"
        assert artifact["honest_verdict"] == "insufficient_pairs"

    def test_auc_values_default_to_half(self, tmp_path: Path) -> None:
        """Spec: no retrain → before/after AUC stay at 0.5 baseline."""
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["eorm_before_auc"] == 0.5
        assert artifact["eorm_after_auc"] == 0.5
        assert artifact["jepa_before_auc"] == 0.5
        assert artifact["jepa_after_auc"] == 0.5

    def test_retrain_mode_synthetic_only(self, tmp_path: Path) -> None:
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["retrain_mode"] == "synthetic_only"

    def test_status_is_success(self, tmp_path: Path) -> None:
        """Even when insufficient pairs, the experiment completes successfully."""
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["status"] == "success"


# ---------------------------------------------------------------------------
# Tests: run_experiment — real data path (mocked pairs injection)
# ---------------------------------------------------------------------------


class TestRunExperimentRealData:
    """Inject sufficient synthetic pairs to exercise the real-data training paths.

    Training epochs are patched to 2 (EORM) and 2 (JEPA) so tests finish quickly
    on CPU.  AUC evaluators are also patched for the verdict-specific tests so
    those don't depend on random initialisation outcomes.
    """

    def _make_enough_eorm_pairs(self) -> list[ViolationPair]:
        """60 balanced pairs — exceeds EORM_MIN_PAIRS=50."""
        return _make_synthetic_pairs(n=60, seed=999)

    def _make_enough_jepa_pairs(self) -> list[ViolationPair]:
        """40 balanced pairs — exceeds JEPA_MIN_PAIRS=30."""
        return _make_synthetic_pairs(n=40, seed=888)

    def test_eorm_retrain_runs_when_enough_pairs(self, tmp_path: Path) -> None:
        """Spec: SCENARIO-LEARN-048 — with ≥50 pairs, EORM retrain runs."""
        eorm_pairs = self._make_enough_eorm_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=[]),
            patch.object(_mod, "EORM_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["n_eorm_pairs"] == 60
        assert artifact["eorm_verdict"] in {"improved", "no_improvement"}
        assert "eorm_model_path" in artifact

    def test_jepa_retrain_runs_when_enough_pairs(self, tmp_path: Path) -> None:
        """With ≥30 JEPA pairs, retrain runs."""
        jepa_pairs = self._make_enough_jepa_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=[]),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["n_jepa_pairs"] == 40
        assert artifact["jepa_verdict"] in {"improved", "no_improvement"}

    def test_retrain_mode_real_data_when_eorm_sufficient(self, tmp_path: Path) -> None:
        """retrain_mode='real_data' when EORM has enough pairs."""
        eorm_pairs = self._make_enough_eorm_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=[]),
            patch.object(_mod, "EORM_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["retrain_mode"] == "real_data"

    def test_honest_verdict_both_when_both_improve(self, tmp_path: Path) -> None:
        """When both verdicts='improved', honest_verdict='both_improved'."""
        eorm_pairs = self._make_enough_eorm_pairs()
        jepa_pairs = self._make_enough_jepa_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod, "_evaluate_eorm_auc", side_effect=[0.5, 0.7]),
            patch.object(_mod.JEPARetrainer, "evaluate_auc_roc", side_effect=[0.5, 0.7]),
            patch.object(_mod.EORMTrainer, "train_epoch", return_value=0.0),
            patch.object(_mod.JEPARetrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "EORM_EPOCHS", 2),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "both_improved"

    def test_honest_verdict_eorm_only(self, tmp_path: Path) -> None:
        """EORM improves, JEPA does not → eorm_only."""
        eorm_pairs = self._make_enough_eorm_pairs()
        jepa_pairs = self._make_enough_jepa_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod, "_evaluate_eorm_auc", side_effect=[0.5, 0.7]),
            patch.object(_mod.JEPARetrainer, "evaluate_auc_roc", side_effect=[0.6, 0.5]),
            patch.object(_mod.EORMTrainer, "train_epoch", return_value=0.0),
            patch.object(_mod.JEPARetrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "EORM_EPOCHS", 2),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "eorm_only"

    def test_honest_verdict_jepa_only(self, tmp_path: Path) -> None:
        """JEPA improves, EORM does not → jepa_only."""
        eorm_pairs = self._make_enough_eorm_pairs()
        jepa_pairs = self._make_enough_jepa_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod, "_evaluate_eorm_auc", side_effect=[0.6, 0.5]),
            patch.object(_mod.JEPARetrainer, "evaluate_auc_roc", side_effect=[0.5, 0.7]),
            patch.object(_mod.EORMTrainer, "train_epoch", return_value=0.0),
            patch.object(_mod.JEPARetrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "EORM_EPOCHS", 2),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "jepa_only"

    def test_honest_verdict_neither_improved(self, tmp_path: Path) -> None:
        """Both run but neither AUC improves → neither_improved."""
        eorm_pairs = self._make_enough_eorm_pairs()
        jepa_pairs = self._make_enough_jepa_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod, "_evaluate_eorm_auc", side_effect=[0.6, 0.5]),
            patch.object(_mod.JEPARetrainer, "evaluate_auc_roc", side_effect=[0.6, 0.5]),
            patch.object(_mod.EORMTrainer, "train_epoch", return_value=0.0),
            patch.object(_mod.JEPARetrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "EORM_EPOCHS", 2),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "neither_improved"

    def test_eorm_model_save_failure_graceful(self, tmp_path: Path) -> None:
        """If EORM model save raises, model_path is empty string and run succeeds."""
        eorm_pairs = self._make_enough_eorm_pairs()

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=eorm_pairs),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=[]),
            patch.object(_mod.EORMModel, "save", side_effect=OSError("disk full")),
            patch.object(_mod.EORMTrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "EORM_EPOCHS", 2),
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["status"] == "success"
        assert artifact["eorm_model_path"] == ""

    def test_jepa_model_save_failure_graceful(self, tmp_path: Path) -> None:
        """If JEPA safetensors save raises, model_path is empty and run succeeds."""
        jepa_pairs = self._make_enough_jepa_pairs()

        import safetensors.numpy as st_numpy
        original_save = st_numpy.save_file

        def _raise(*a: Any, **k: Any) -> None:
            raise OSError("disk full")

        with (
            patch.object(_mod, "load_real_cot_pairs", return_value=[]),
            patch.object(_mod, "_load_jepa_pairs_from_files", return_value=jepa_pairs),
            patch.object(_mod.JEPARetrainer, "train_epoch", return_value=0.0),
            patch.object(_mod, "JEPA_EPOCHS", 2),
        ):
            st_numpy.save_file = _raise  # type: ignore[assignment]
            try:
                artifact = run_experiment(repo_root=tmp_path)
            finally:
                st_numpy.save_file = original_save

        assert artifact["status"] == "success"
        assert artifact["jepa_model_path"] == ""


# ---------------------------------------------------------------------------
# Tests: main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_artifact_json(self, tmp_path: Path) -> None:
        """main() writes a valid JSON file to the deliverable path."""
        with (
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "run_experiment", return_value={
                "experiment": 383,
                "schema": "carnot.combined_retrain.v1",
                "status": "success",
                "retrain_mode": "synthetic_only",
                "honest_verdict": "insufficient_pairs",
                "eorm_verdict": "insufficient_real_pairs",
                "eorm_before_auc": 0.5,
                "eorm_after_auc": 0.5,
                "jepa_verdict": "insufficient_real_pairs",
                "jepa_before_auc": 0.5,
                "jepa_after_auc": 0.5,
            }),
        ):
            main()

        out_path = tmp_path / "results" / "experiment_383_models_retrain.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["experiment"] == 383
        assert data["status"] == "success"

    def test_main_respects_force_live_env(self, tmp_path: Path) -> None:
        """main() passes force_live=True when CARNOT_FORCE_LIVE=1."""
        captured: dict[str, Any] = {}

        def mock_run(*, force_live: bool = False, repo_root: Any = None) -> dict:
            captured["force_live"] = force_live
            return {
                "experiment": 383,
                "schema": [],
                "status": "success",
                "retrain_mode": "synthetic_only",
                "honest_verdict": "insufficient_pairs",
                "eorm_verdict": "insufficient_real_pairs",
                "eorm_before_auc": 0.5,
                "eorm_after_auc": 0.5,
                "jepa_verdict": "insufficient_real_pairs",
                "jepa_before_auc": 0.5,
                "jepa_after_auc": 0.5,
            }

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "run_experiment", side_effect=mock_run),
        ):
            main()

        assert captured["force_live"] is True
