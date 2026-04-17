"""Tests for Exp 431 — EORM + JEPA retrain on FOVER-labeled real pairs.

Coverage target: 100% of:
  - python/carnot/pipeline/fover_eorm_retrain.py
      load_fover_pairs, fover_pairs_to_contrastive, compute_retrain_verdict
  - scripts/experiment_431_eorm_jepa_real_retrain.py
      _evaluate_eorm_auc, _fover_pairs_to_violation_pairs, _save_jepa_model,
      _load_or_build_eorm_model, _build_eorm_triples, run_experiment, main

All tests run without a live GPU (JAX_PLATFORMS=cpu, no HuggingFace calls).
EORM and JEPA model saves/loads are tested via temp directories.

Spec: REQ-LEARN-032, REQ-LEARN-033,
      SCENARIO-LEARN-057, SCENARIO-LEARN-058, SCENARIO-LEARN-059
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

_SCRIPT_PATH = _REPO_ROOT / "scripts" / "experiment_431_eorm_jepa_real_retrain.py"


# ---------------------------------------------------------------------------
# Module loader — imports the script without executing main()
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_431 as a module without running main()."""
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_431", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("experiment_431", None)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

# ---------------------------------------------------------------------------
# Import tested module directly
# ---------------------------------------------------------------------------

from carnot.pipeline.fover_eorm_retrain import (  # noqa: E402
    compute_retrain_verdict,
    fover_pairs_to_contrastive,
    load_fover_pairs,
)


# ===========================================================================
# Tests for load_fover_pairs
# ===========================================================================


class TestLoadFoverPairs:
    """Tests for load_fover_pairs(path) — SCENARIO-LEARN-057."""

    def test_loads_valid_pairs_schema_v1(self, tmp_path: Path) -> None:
        """Pairs in carnot.fover_labels.v1 schema are loaded correctly."""
        data = {
            "schema": "carnot.fover_labels.v1",
            "pairs": [
                {"question_id": "q1", "step_text": "2 + 3 = 5", "label": "correct", "confidence": 1.0},
                {"question_id": "q1", "step_text": "2 + 3 = 6", "label": "incorrect", "confidence": 1.0},
                {"question_id": "q2", "step_text": "fuzzy", "label": "not_verifiable", "confidence": 0.0},
            ],
        }
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))
        result = load_fover_pairs(str(p))
        # not_verifiable filtered out
        assert len(result) == 2
        labels = {r["label"] for r in result}
        assert labels == {"correct", "incorrect"}

    def test_confidence_filter(self, tmp_path: Path) -> None:
        """Pairs with confidence < 0.3 are excluded (SCENARIO-LEARN-057)."""
        data = {
            "pairs": [
                {"question_id": "q1", "step_text": "a", "label": "correct", "confidence": 0.1},
                {"question_id": "q1", "step_text": "b", "label": "incorrect", "confidence": 0.5},
                {"question_id": "q1", "step_text": "c", "label": "correct", "confidence": 0.3},
            ]
        }
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))
        result = load_fover_pairs(str(p))
        assert len(result) == 2  # confidence 0.1 filtered out

    def test_bare_list_format(self, tmp_path: Path) -> None:
        """Bare list format (forward compatibility) is also supported."""
        data = [
            {"question_id": "q1", "step_text": "x", "label": "correct", "confidence": 1.0},
        ]
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))
        result = load_fover_pairs(str(p))
        assert len(result) == 1

    def test_missing_file_returns_empty(self) -> None:
        """Missing file returns empty list rather than raising."""
        result = load_fover_pairs("/tmp/nonexistent_carnot_431_test.json")
        assert result == []

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        """Malformed JSON returns empty list."""
        p = tmp_path / "bad.json"
        p.write_text("{broken json[")
        result = load_fover_pairs(str(p))
        assert result == []

    def test_wrong_schema_type_returns_empty(self, tmp_path: Path) -> None:
        """JSON root that is neither dict nor list returns empty."""
        p = tmp_path / "weird.json"
        p.write_text('"just a string"')
        result = load_fover_pairs(str(p))
        assert result == []

    def test_pairs_key_not_list_returns_empty(self, tmp_path: Path) -> None:
        """If 'pairs' is not a list, returns empty."""
        p = tmp_path / "bad2.json"
        p.write_text('{"pairs": "not a list"}')
        result = load_fover_pairs(str(p))
        assert result == []

    def test_non_dict_entries_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries inside the pairs list are skipped."""
        data = {
            "pairs": [
                "not_a_dict",
                {"question_id": "q1", "step_text": "x", "label": "correct", "confidence": 1.0},
            ]
        }
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))
        result = load_fover_pairs(str(p))
        assert len(result) == 1

    def test_output_fields_present(self, tmp_path: Path) -> None:
        """Every returned dict has the four required fields."""
        data = {
            "pairs": [
                {"question_id": "q1", "step_text": "y", "label": "correct", "confidence": 1.0},
            ]
        }
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))
        result = load_fover_pairs(str(p))
        assert set(result[0].keys()) == {"question_id", "step_text", "label", "confidence"}

    def test_empty_pairs_list(self, tmp_path: Path) -> None:
        """Empty pairs list returns empty."""
        p = tmp_path / "empty.json"
        p.write_text('{"pairs": []}')
        result = load_fover_pairs(str(p))
        assert result == []

    def test_oserror_returns_empty(self, tmp_path: Path) -> None:
        """OSError during file open returns empty list."""
        with patch("builtins.open", side_effect=OSError("permission denied")):
            # Need a file that exists so we get past the path.exists() check
            p = tmp_path / "fover.json"
            p.write_text("{}")
            result = load_fover_pairs(str(p))
        assert result == []


# ===========================================================================
# Tests for fover_pairs_to_contrastive
# ===========================================================================


class TestFoverPairsToContrastive:
    """Tests for fover_pairs_to_contrastive(pairs) — SCENARIO-LEARN-058."""

    def _make_pair(
        self, question_id: str, label: str, step_text: str = "text"
    ) -> dict:
        return {"question_id": question_id, "step_text": step_text, "label": label, "confidence": 1.0}

    def test_basic_contrastive_pair(self) -> None:
        """Two steps on same question (correct + incorrect) produce one contrastive tuple."""
        pairs = [
            self._make_pair("q1", "correct", "2 + 3 = 5"),
            self._make_pair("q1", "incorrect", "2 + 3 = 6"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 1
        pos, neg = result[0]
        assert pos.shape == (64,)
        assert neg.shape == (64,)

    def test_no_cross_question_matching(self) -> None:
        """Steps from different questions are NOT matched (SCENARIO-LEARN-058)."""
        pairs = [
            self._make_pair("q1", "correct", "correct step q1"),
            self._make_pair("q2", "incorrect", "incorrect step q2"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        # Each question has only one label type, so no pairs can be formed
        assert len(result) == 0

    def test_multiple_pairs_same_question(self) -> None:
        """Multiple correct and incorrect steps on same question: round-robin."""
        pairs = [
            self._make_pair("q1", "correct", "correct 1"),
            self._make_pair("q1", "correct", "correct 2"),
            self._make_pair("q1", "incorrect", "incorrect 1"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        # max(2, 1) = 2 round-robin pairs
        assert len(result) == 2

    def test_only_correct_steps_produces_no_pairs(self) -> None:
        """Question with only correct steps (no incorrect) produces no pairs."""
        pairs = [
            self._make_pair("q1", "correct", "a"),
            self._make_pair("q1", "correct", "b"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 0

    def test_only_incorrect_steps_produces_no_pairs(self) -> None:
        """Question with only incorrect steps (no correct) produces no pairs."""
        pairs = [
            self._make_pair("q1", "incorrect", "a"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 0

    def test_synthetic_question_ids_pooled(self) -> None:
        """synthetic_* question IDs are pooled together so they form contrastive pairs."""
        pairs = [
            self._make_pair("synthetic_correct_0", "correct", "correct step"),
            self._make_pair("synthetic_incorrect_0", "incorrect", "incorrect step"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 1

    def test_unknown_question_id_pooled(self) -> None:
        """question_id='unknown' is pooled with other unknowns."""
        pairs = [
            self._make_pair("unknown", "correct", "correct step"),
            self._make_pair("unknown", "incorrect", "incorrect step"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 1

    def test_empty_input_returns_empty(self) -> None:
        """Empty input list returns empty list."""
        assert fover_pairs_to_contrastive([]) == []

    def test_returns_jnp_arrays(self) -> None:
        """Each element of the result is a tuple of jnp.ndarray."""
        pairs = [
            self._make_pair("q1", "correct", "good step"),
            self._make_pair("q1", "incorrect", "bad step"),
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 1
        pos, neg = result[0]
        assert isinstance(pos, jnp.ndarray)
        assert isinstance(neg, jnp.ndarray)

    def test_empty_step_text_produces_zero_embedding(self) -> None:
        """Empty step_text maps to all-zero embedding (does not crash)."""
        pairs = [
            {"question_id": "q1", "step_text": "", "label": "correct", "confidence": 1.0},
            {"question_id": "q1", "step_text": "", "label": "incorrect", "confidence": 1.0},
        ]
        result = fover_pairs_to_contrastive(pairs)
        assert len(result) == 1
        pos, neg = result[0]
        # Both empty texts → zero embeddings
        assert float(jnp.sum(jnp.abs(pos))) == pytest.approx(0.0)
        assert float(jnp.sum(jnp.abs(neg))) == pytest.approx(0.0)


# ===========================================================================
# Tests for compute_retrain_verdict
# ===========================================================================


class TestComputeRetrainVerdict:
    """Tests for compute_retrain_verdict — SCENARIO-LEARN-059."""

    def test_real_data_improvement(self) -> None:
        """after_auc > before_auc AND n_real >= 10 -> 'real_data_improvement'."""
        assert compute_retrain_verdict(0.5, 0.62, 25) == "real_data_improvement"

    def test_synthetic_only_below_threshold(self) -> None:
        """n_real < 10 -> 'synthetic_only' regardless of AUC (SCENARIO-LEARN-059)."""
        assert compute_retrain_verdict(0.5, 0.8, 5) == "synthetic_only"
        assert compute_retrain_verdict(0.5, 0.5, 0) == "synthetic_only"
        assert compute_retrain_verdict(0.9, 0.1, 9) == "synthetic_only"

    def test_real_data_no_improvement(self) -> None:
        """after_auc <= before_auc AND n_real >= 10 -> 'real_data_no_improvement'."""
        assert compute_retrain_verdict(0.62, 0.55, 25) == "real_data_no_improvement"
        assert compute_retrain_verdict(0.6, 0.6, 10) == "real_data_no_improvement"

    def test_threshold_boundary_exactly_10(self) -> None:
        """n_real == 10 is above threshold; 9 is below."""
        assert compute_retrain_verdict(0.5, 0.6, 10) == "real_data_improvement"
        assert compute_retrain_verdict(0.5, 0.6, 9) == "synthetic_only"

    def test_auc_equal_not_improvement(self) -> None:
        """after_auc == before_auc is NOT an improvement."""
        assert compute_retrain_verdict(0.5, 0.5, 50) == "real_data_no_improvement"

    def test_tiny_improvement_counts(self) -> None:
        """Very small AUC improvement still counts as real_data_improvement."""
        assert compute_retrain_verdict(0.5, 0.5 + 1e-9, 20) == "real_data_improvement"


# ===========================================================================
# Tests for experiment_431 script helpers
# ===========================================================================


class TestEvaluateEormAuc:
    """Tests for _evaluate_eorm_auc in the experiment script."""

    def test_empty_pairs_returns_half(self) -> None:
        """Empty pair list returns AUC = 0.5 (random baseline)."""
        model = MagicMock()
        assert _mod._evaluate_eorm_auc(model, []) == pytest.approx(0.5)

    def test_all_same_label_returns_half(self) -> None:
        """All pairs with the same label returns AUC = 0.5."""
        from carnot.embeddings.jepa_retrain import ViolationPair

        model = MagicMock()
        model.energy.return_value = 1.0

        pairs = [
            ViolationPair(
                partial_response="x", full_response="x",
                has_violation=True, model_id="m", question_id="q1",
            )
        ]
        assert _mod._evaluate_eorm_auc(model, pairs) == pytest.approx(0.5)

    def test_perfect_discrimination(self) -> None:
        """Model with perfect discrimination (high energy for violations) -> AUC = 1.0."""
        from carnot.embeddings.jepa_retrain import ViolationPair

        model = MagicMock()
        # Return energy as a simple float; the mock's side_effect uses cot argument
        call_count = [0]

        def mock_energy(cot: Any) -> float:
            # violations get high energy (score = -energy -> most negative for correct,
            # most positive for violations).  With correct=False first and violation=True
            # second, we need violation to have higher energy.
            # Use call order: first call is correct (low energy), second is violation (high).
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return -10.0  # correct: low energy
            return 10.0  # incorrect/violation: high energy

        model.energy.side_effect = mock_energy

        pairs = [
            ViolationPair("p", "correct_response", False, "m", "q1"),
            ViolationPair("p", "incorrect_response", True, "m", "q2"),
        ]
        auc = _mod._evaluate_eorm_auc(model, pairs)
        assert auc == pytest.approx(1.0)


class TestFoverPairsToViolationPairs:
    """Tests for _fover_pairs_to_violation_pairs."""

    def test_correct_maps_to_no_violation(self) -> None:
        """label='correct' -> has_violation=False."""
        fover = [{"question_id": "q1", "step_text": "good", "label": "correct", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert len(vps) == 1
        assert vps[0].has_violation is False

    def test_incorrect_maps_to_violation(self) -> None:
        """label='incorrect' -> has_violation=True."""
        fover = [{"question_id": "q1", "step_text": "bad", "label": "incorrect", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert len(vps) == 1
        assert vps[0].has_violation is True

    def test_step_text_used_for_both_fields(self) -> None:
        """step_text is used as both partial_response and full_response."""
        fover = [{"question_id": "q1", "step_text": "step text here", "label": "correct", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert vps[0].partial_response == "step text here"
        assert vps[0].full_response == "step text here"

    def test_empty_input_returns_empty(self) -> None:
        assert _mod._fover_pairs_to_violation_pairs([]) == []


class TestSaveJepaModel:
    """Tests for _save_jepa_model."""

    def test_save_creates_safetensors_file(self, tmp_path: Path) -> None:
        """Saving a model creates a .safetensors file."""
        from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig

        config = JEPAEnergyConfig(embed_dim=4, hidden_dims=[4])
        model = ContextPredictionEnergy(config=config)
        save_path = str(tmp_path / "test_jepa.safetensors")
        _mod._save_jepa_model(model, save_path)
        assert Path(save_path).exists()

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        """save creates parent directory if it does not exist."""
        from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig

        config = JEPAEnergyConfig(embed_dim=4, hidden_dims=[4])
        model = ContextPredictionEnergy(config=config)
        nested = tmp_path / "nested" / "dir" / "jepa.safetensors"
        _mod._save_jepa_model(model, str(nested))
        assert nested.exists()


class TestLoadOrBuildEormModel:
    """Tests for _load_or_build_eorm_model."""

    def test_builds_fresh_when_no_saved_model(self, tmp_path: Path) -> None:
        """Returns a valid EORMModel when no saved model files exist."""
        from carnot.models.eorm import EORMModel
        model = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(model, EORMModel)

    def test_loads_exp359_model_when_present(self, tmp_path: Path) -> None:
        """Loads eorm_model_359_real.safetensors when it exists."""
        from carnot.models.eorm import EORMModel
        import jax.random as jrandom

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        path_359 = results_dir / "eorm_model_359_real.safetensors"

        # Build and save a model
        m = EORMModel(embed_dim=128, n_heads=4, n_layers=2, key=jrandom.PRNGKey(0))
        m.save(str(path_359))

        loaded = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(loaded, EORMModel)

    def test_falls_back_to_exp346_when_exp359_missing(self, tmp_path: Path) -> None:
        """Falls back to eorm_model_346.safetensors when 359 model is absent."""
        from carnot.models.eorm import EORMModel
        import jax.random as jrandom

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        path_346 = results_dir / "eorm_model_346.safetensors"

        m = EORMModel(embed_dim=128, n_heads=4, n_layers=2, key=jrandom.PRNGKey(0))
        m.save(str(path_346))

        loaded = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(loaded, EORMModel)


class TestBuildEormTriples:
    """Tests for _build_eorm_triples."""

    def _vp(self, q: str, has_violation: bool, text: str = "text") -> Any:
        from carnot.embeddings.jepa_retrain import ViolationPair
        return ViolationPair(
            partial_response=text, full_response=text,
            has_violation=has_violation, model_id="m", question_id=q,
        )

    def test_basic_triple_formation(self) -> None:
        fover = [
            {"question_id": "q1", "step_text": "c", "label": "correct", "confidence": 1.0},
            {"question_id": "q1", "step_text": "i", "label": "incorrect", "confidence": 1.0},
        ]
        vps = [self._vp("q1", False, "c"), self._vp("q1", True, "i")]
        triples = _mod._build_eorm_triples(fover, vps)
        assert len(triples) == 1
        correct_resp, incorrect_resp, q_id = triples[0]
        assert correct_resp == "c"
        assert incorrect_resp == "i"

    def test_no_triples_when_same_label(self) -> None:
        fover: list[dict] = []
        vps = [self._vp("q1", False, "a"), self._vp("q1", False, "b")]
        triples = _mod._build_eorm_triples(fover, vps)
        assert len(triples) == 0

    def test_synthetic_ids_pooled(self) -> None:
        fover: list[dict] = []
        vps = [
            self._vp("synthetic_0", False, "correct"),
            self._vp("synthetic_1", True, "incorrect"),
        ]
        triples = _mod._build_eorm_triples(fover, vps)
        assert len(triples) == 1


# ===========================================================================
# Integration test: run_experiment end-to-end
# ===========================================================================


class TestRunExperiment:
    """Integration tests for run_experiment() — tests the full pipeline in CI mode."""

    def test_run_experiment_synthetic_fallback(self, tmp_path: Path) -> None:
        """run_experiment falls back to synthetic data when fover file is absent."""
        artifact = _mod.run_experiment(repo_root=tmp_path)

        assert artifact["schema"] == "carnot.eorm_jepa_retrain.v2"
        assert artifact["honest_verdict"] == "synthetic_only"
        assert artifact["n_real_pairs"] == 0
        assert artifact["retro_024_closed"] is False
        assert "before_auc" in artifact
        assert "after_auc" in artifact
        assert artifact["status"] == "success"

    def test_run_experiment_with_real_pairs(self, tmp_path: Path) -> None:
        """run_experiment uses real pairs when fover file has >= 10 qualifying pairs."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Build a FOVER file with 20 real pairs (10 correct + 10 incorrect, same question)
        pairs = []
        for i in range(10):
            pairs.append({
                "question_id": f"q{i}",
                "step_text": f"correct step {i}: 1 + {i} = {1 + i}",
                "label": "correct",
                "confidence": 1.0,
            })
            pairs.append({
                "question_id": f"q{i}",
                "step_text": f"incorrect step {i}: 1 + {i} = {2 + i}",
                "label": "incorrect",
                "confidence": 1.0,
            })

        fover_file = results_dir / "fover_labeled_steps.json"
        fover_file.write_text(json.dumps({
            "schema": "carnot.fover_labels.v1",
            "pairs": pairs,
        }))

        artifact = _mod.run_experiment(repo_root=tmp_path)

        assert artifact["schema"] == "carnot.eorm_jepa_retrain.v2"
        assert artifact["retrain_mode"] == "real_data"
        assert artifact["n_real_pairs"] == 20
        # honest_verdict is one of the three valid values
        assert artifact["honest_verdict"] in (
            "real_data_improvement", "real_data_no_improvement"
        )
        assert "retro_024_closed" in artifact

    def test_run_experiment_artifact_fields_complete(self, tmp_path: Path) -> None:
        """Artifact contains all required fields for schema='carnot.eorm_jepa_retrain.v2'."""
        artifact = _mod.run_experiment(repo_root=tmp_path)

        required_fields = {
            "schema", "retrain_mode", "n_real_pairs",
            "before_auc", "after_auc", "auc_improvement", "honest_verdict",
            "retro_024_closed", "eorm_model_path", "jepa_before_auc",
            "jepa_after_auc", "jepa_model_path", "n_contrastive_triples",
            "n_train_pairs", "n_test_pairs", "n_eorm_epochs", "status",
        }
        for field in required_fields:
            assert field in artifact, f"missing required field: {field}"

    def test_retro_024_closed_when_verdict_is_improvement(self, tmp_path: Path) -> None:
        """retro_024_closed is True iff honest_verdict == 'real_data_improvement'."""
        artifact = _mod.run_experiment(repo_root=tmp_path)
        expected = artifact["honest_verdict"] == "real_data_improvement"
        assert artifact["retro_024_closed"] == expected


# ===========================================================================
# Tests for main() entry point
# ===========================================================================


class TestMain:
    """Tests for the main() function — writes artifact to disk."""

    def test_main_writes_deliverable_json(self, tmp_path: Path) -> None:
        """main() writes the result JSON to the deliverable path."""
        deliverable = tmp_path / "results" / "experiment_431_eorm_jepa_real_retrain.json"

        def mock_run_experiment(**kwargs: Any) -> dict:
            return {
                "schema": "carnot.eorm_jepa_retrain.v2",
                "honest_verdict": "synthetic_only",
                "retro_024_closed": False,
                "status": "success",
            }

        with (
            patch.object(_mod, "run_experiment", side_effect=mock_run_experiment),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
        ):
            _mod.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["honest_verdict"] == "synthetic_only"

    def test_main_uses_watchdog(self, tmp_path: Path) -> None:
        """main() wraps run_experiment in an ExperimentTimeoutWatchdog."""
        watchdog_mock = MagicMock()
        watchdog_mock.__enter__ = MagicMock(return_value=watchdog_mock)
        watchdog_mock.__exit__ = MagicMock(return_value=False)

        def mock_run(**kwargs: Any) -> dict:
            return {
                "schema": "carnot.eorm_jepa_retrain.v2",
                "honest_verdict": "synthetic_only",
                "retro_024_closed": False,
                "status": "success",
            }

        with (
            patch.object(_mod, "ExperimentTimeoutWatchdog", return_value=watchdog_mock),
            patch.object(_mod, "run_experiment", side_effect=mock_run),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
        ):
            _mod.main()

        watchdog_mock.__enter__.assert_called_once()
        watchdog_mock.__exit__.assert_called_once()
