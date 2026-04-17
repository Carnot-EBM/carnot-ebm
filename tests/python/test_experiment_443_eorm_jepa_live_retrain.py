"""Tests for Exp 443 — EORM + JEPA retrain on live FOVER pairs (RETRO-024 milestone 8).

Coverage targets:
  scripts/experiment_443_eorm_jepa_live_retrain.py:
    _evaluate_eorm_auc, _fover_pairs_to_violation_pairs, _save_jepa_model,
    _load_or_build_eorm_model, _build_eorm_triples, run_experiment, main

All tests run without a live GPU (JAX_PLATFORMS=cpu, no HuggingFace calls).

Spec: REQ-LEARN-036, SCENARIO-LEARN-064, SCENARIO-LEARN-065
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

_SCRIPT_PATH = _REPO_ROOT / "scripts" / "experiment_443_eorm_jepa_live_retrain.py"


# ---------------------------------------------------------------------------
# Module loader — imports the script without executing main()
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_443 as a module without running main()."""
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_443", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("experiment_443", None)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vp(q: str, has_violation: bool, text: str = "text") -> Any:
    from carnot.embeddings.jepa_retrain import ViolationPair
    return ViolationPair(
        partial_response=text, full_response=text,
        has_violation=has_violation, model_id="m", question_id=q,
    )


def _make_fover_file(tmp_path: Path, n_correct: int = 30, n_incorrect: int = 27) -> Path:
    """Write a fover_labeled_steps_live.json with real-looking pairs."""
    pairs = []
    for i in range(n_correct):
        pairs.append({
            "question_id": str(i),
            "step_text": f"correct step {i}",
            "label": "correct",
            "confidence": 1.0,
        })
    for i in range(n_incorrect):
        pairs.append({
            "question_id": str(i),  # same question_id allows contrastive matching
            "step_text": f"incorrect step {i}",
            "label": "incorrect",
            "confidence": 1.0,
        })
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / "fover_labeled_steps_live.json"
    p.write_text(json.dumps(pairs))
    return p


def _make_ann_file(tmp_path: Path, source: str = "live", n_labeled: int = 57) -> Path:
    """Write a minimal Exp 442 annotation result file."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ann = {
        "experiment": 442,
        "source": source,
        "n_labeled": n_labeled,
        "honest_verdict": "real_data_labeled",
    }
    p = results_dir / "experiment_442_fover_live_annotation.json"
    p.write_text(json.dumps(ann))
    return p


# ===========================================================================
# Tests for _evaluate_eorm_auc
# ===========================================================================


class TestEvaluateEormAuc:
    """Tests for _evaluate_eorm_auc in experiment_443."""

    def test_empty_pairs_returns_half(self) -> None:
        model = MagicMock()
        assert _mod._evaluate_eorm_auc(model, []) == pytest.approx(0.5)

    def test_all_same_label_returns_half(self) -> None:
        model = MagicMock()
        model.energy.return_value = 1.0
        pairs = [_vp("q1", True), _vp("q2", True)]
        assert _mod._evaluate_eorm_auc(model, pairs) == pytest.approx(0.5)

    def test_mixed_labels_returns_valid_auc(self) -> None:
        """Mixed labels and energies produce AUC in [0, 1]."""
        model = MagicMock()
        call_count = [0]

        def energy_fn(cot: Any) -> float:
            call_count[0] += 1
            return float(call_count[0])  # increasing energy

        model.energy.side_effect = energy_fn
        pairs = [_vp("q1", False, "a"), _vp("q2", True, "b"), _vp("q3", False, "c")]
        auc = _mod._evaluate_eorm_auc(model, pairs)
        assert 0.0 <= auc <= 1.0


# ===========================================================================
# Tests for _fover_pairs_to_violation_pairs
# ===========================================================================


class TestFoverPairsToViolationPairs:
    """Tests for _fover_pairs_to_violation_pairs in experiment_443."""

    def test_correct_maps_to_no_violation(self) -> None:
        fover = [{"question_id": "q1", "step_text": "good", "label": "correct", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert len(vps) == 1
        assert vps[0].has_violation is False

    def test_incorrect_maps_to_violation(self) -> None:
        fover = [{"question_id": "q1", "step_text": "bad", "label": "incorrect", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert len(vps) == 1
        assert vps[0].has_violation is True

    def test_step_text_in_both_fields(self) -> None:
        fover = [{"question_id": "q1", "step_text": "abc", "label": "correct", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert vps[0].partial_response == "abc"
        assert vps[0].full_response == "abc"

    def test_model_id_is_fover_live_443(self) -> None:
        """model_id is tagged as 'fover_live_443' to distinguish from Exp 431 pairs."""
        fover = [{"question_id": "q1", "step_text": "x", "label": "correct", "confidence": 1.0}]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert vps[0].model_id == "fover_live_443"

    def test_empty_input_returns_empty(self) -> None:
        assert _mod._fover_pairs_to_violation_pairs([]) == []

    def test_multiple_pairs(self) -> None:
        fover = [
            {"question_id": "q1", "step_text": "c", "label": "correct", "confidence": 1.0},
            {"question_id": "q2", "step_text": "i", "label": "incorrect", "confidence": 1.0},
        ]
        vps = _mod._fover_pairs_to_violation_pairs(fover)
        assert len(vps) == 2


# ===========================================================================
# Tests for _save_jepa_model
# ===========================================================================


class TestSaveJepaModel:
    """Tests for _save_jepa_model in experiment_443."""

    def test_creates_safetensors_file(self, tmp_path: Path) -> None:
        from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig
        config = JEPAEnergyConfig(embed_dim=4, hidden_dims=[4])
        model = ContextPredictionEnergy(config=config)
        path = str(tmp_path / "jepa_443.safetensors")
        _mod._save_jepa_model(model, path)
        assert Path(path).exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig
        config = JEPAEnergyConfig(embed_dim=4, hidden_dims=[4])
        model = ContextPredictionEnergy(config=config)
        nested = tmp_path / "a" / "b" / "jepa.safetensors"
        _mod._save_jepa_model(model, str(nested))
        assert nested.exists()


# ===========================================================================
# Tests for _load_or_build_eorm_model
# ===========================================================================


class TestLoadOrBuildEormModel:
    """Tests for _load_or_build_eorm_model in experiment_443."""

    def test_builds_fresh_when_no_saved(self, tmp_path: Path) -> None:
        from carnot.models.eorm import EORMModel
        model = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(model, EORMModel)

    def test_loads_431_model_first(self, tmp_path: Path) -> None:
        """Prefers eorm_431_real.safetensors over older checkpoints."""
        import jax.random as jrandom
        from carnot.models.eorm import EORMModel

        results = tmp_path / "results"
        results.mkdir()
        m = EORMModel(embed_dim=128, n_heads=4, n_layers=2, key=jrandom.PRNGKey(0))
        m.save(str(results / "eorm_431_real.safetensors"))

        loaded = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(loaded, EORMModel)

    def test_falls_back_to_359(self, tmp_path: Path) -> None:
        """Falls back to eorm_model_359_real.safetensors when 431 is absent."""
        import jax.random as jrandom
        from carnot.models.eorm import EORMModel

        results = tmp_path / "results"
        results.mkdir()
        m = EORMModel(embed_dim=128, n_heads=4, n_layers=2, key=jrandom.PRNGKey(0))
        m.save(str(results / "eorm_model_359_real.safetensors"))

        loaded = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(loaded, EORMModel)

    def test_falls_back_to_346(self, tmp_path: Path) -> None:
        """Falls back to eorm_model_346.safetensors when 431/359 are absent."""
        import jax.random as jrandom
        from carnot.models.eorm import EORMModel

        results = tmp_path / "results"
        results.mkdir()
        m = EORMModel(embed_dim=128, n_heads=4, n_layers=2, key=jrandom.PRNGKey(0))
        m.save(str(results / "eorm_model_346.safetensors"))

        loaded = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(loaded, EORMModel)

    def test_skips_corrupt_model_file(self, tmp_path: Path) -> None:
        """Corrupted safetensors file is skipped; falls back to fresh init."""
        from carnot.models.eorm import EORMModel

        results = tmp_path / "results"
        results.mkdir()
        corrupt = results / "eorm_431_real.safetensors"
        corrupt.write_bytes(b"not a valid safetensors file")

        model = _mod._load_or_build_eorm_model(tmp_path)
        assert isinstance(model, EORMModel)


# ===========================================================================
# Tests for _build_eorm_triples
# ===========================================================================


class TestBuildEormTriples:
    """Tests for _build_eorm_triples in experiment_443 (takes only violation_pairs)."""

    def test_basic_triple_formation(self) -> None:
        vps = [_vp("q1", False, "correct"), _vp("q1", True, "incorrect")]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 1
        correct_resp, incorrect_resp, q_id = triples[0]
        assert correct_resp == "correct"
        assert incorrect_resp == "incorrect"

    def test_no_cross_question_pairing(self) -> None:
        """correct from q1 and incorrect from q2 must NOT be paired."""
        vps = [_vp("q1", False, "c"), _vp("q2", True, "i")]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 0

    def test_synthetic_ids_pooled(self) -> None:
        vps = [
            _vp("synthetic_c", False, "correct"),
            _vp("synthetic_i", True, "incorrect"),
        ]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 1

    def test_unknown_id_pooled(self) -> None:
        vps = [_vp("unknown", False, "c"), _vp("unknown", True, "i")]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 1

    def test_empty_returns_empty(self) -> None:
        assert _mod._build_eorm_triples([]) == []

    def test_only_correct_no_triples(self) -> None:
        vps = [_vp("q1", False, "c1"), _vp("q1", False, "c2")]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 0

    def test_round_robin_when_unequal(self) -> None:
        """When n_correct != n_incorrect, round-robin produces max(n, m) triples."""
        vps = [
            _vp("q1", False, "c1"), _vp("q1", False, "c2"),
            _vp("q1", True, "i1"),
        ]
        triples = _mod._build_eorm_triples(vps)
        assert len(triples) == 2  # max(2, 1) = 2


# ===========================================================================
# Tests for run_experiment
# ===========================================================================


class TestRunExperiment:
    """Integration tests for run_experiment() in experiment_443."""

    def test_live_source_with_sufficient_pairs_runs(self, tmp_path: Path) -> None:
        """With 57 live pairs, run_experiment produces a valid artifact."""
        _make_ann_file(tmp_path, source="live", n_labeled=57)
        _make_fover_file(tmp_path, n_correct=30, n_incorrect=27)

        artifact = _mod.run_experiment(repo_root=tmp_path)

        assert artifact["schema"] == "carnot.eorm_jepa_retrain.v3"
        assert artifact["n_real_pairs"] == 57
        assert artifact["source"] == "live"
        assert artifact["honest_verdict"] in {
            "real_data_improvement",
            "real_data_no_improvement",
        }
        assert isinstance(artifact["retro_024_closed"], bool)
        assert artifact["retro_024_closed"] == (artifact["honest_verdict"] == "real_data_improvement")

    def test_synthetic_fallback_when_pairs_insufficient(self, tmp_path: Path) -> None:
        """With <20 pairs, honest_verdict='real_data_insufficient'."""
        _make_ann_file(tmp_path, source="live", n_labeled=5)
        # Write a fover file with only 5 pairs
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        pairs = [
            {"question_id": "q1", "step_text": "c", "label": "correct", "confidence": 1.0},
            {"question_id": "q1", "step_text": "i", "label": "incorrect", "confidence": 1.0},
        ]
        (results / "fover_labeled_steps_live.json").write_text(json.dumps(pairs))

        artifact = _mod.run_experiment(repo_root=tmp_path)

        # With only 2 real pairs (< 20), should be insufficient
        assert artifact["honest_verdict"] == "real_data_insufficient"
        assert artifact["retro_024_closed"] is False

    def test_synthetic_source_returns_synthetic_only(self, tmp_path: Path) -> None:
        """When source='synthetic', verdict is 'synthetic_only'."""
        _make_ann_file(tmp_path, source="synthetic", n_labeled=0)
        # No fover live file

        artifact = _mod.run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "synthetic_only"
        assert artifact["retro_024_closed"] is False

    def test_missing_ann_file_defaults_to_synthetic(self, tmp_path: Path) -> None:
        """If Exp 442 annotation file is missing, source defaults to 'synthetic'."""
        # No annotation file, no fover live file
        artifact = _mod.run_experiment(repo_root=tmp_path)
        assert artifact["honest_verdict"] == "synthetic_only"

    def test_artifact_has_required_fields(self, tmp_path: Path) -> None:
        """All required artifact fields are present."""
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS
        _make_ann_file(tmp_path, source="synthetic", n_labeled=0)

        artifact = _mod.run_experiment(repo_root=tmp_path)

        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_n_labeled_from_442_in_artifact(self, tmp_path: Path) -> None:
        """n_labeled_from_442 field reflects the Exp 442 annotation count."""
        _make_ann_file(tmp_path, source="live", n_labeled=57)
        _make_fover_file(tmp_path, n_correct=30, n_incorrect=27)

        artifact = _mod.run_experiment(repo_root=tmp_path)
        assert artifact["n_labeled_from_442"] == 57

    def test_schema_is_v3(self, tmp_path: Path) -> None:
        """Schema is carnot.eorm_jepa_retrain.v3 (upgraded from v2 in Exp 431)."""
        _make_ann_file(tmp_path, source="synthetic", n_labeled=0)
        artifact = _mod.run_experiment(repo_root=tmp_path)
        assert artifact["schema"] == "carnot.eorm_jepa_retrain.v3"

    def test_malformed_ann_file_defaults_to_synthetic(self, tmp_path: Path) -> None:
        """Malformed Exp 442 annotation JSON defaults gracefully to synthetic."""
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "experiment_442_fover_live_annotation.json").write_text("{not valid json")

        artifact = _mod.run_experiment(repo_root=tmp_path)
        assert artifact["honest_verdict"] == "synthetic_only"

    def test_auc_fields_are_floats(self, tmp_path: Path) -> None:
        """before_auc, after_auc, auc_improvement are all floats."""
        _make_ann_file(tmp_path, source="synthetic", n_labeled=0)
        artifact = _mod.run_experiment(repo_root=tmp_path)
        for field in ("before_auc", "after_auc", "auc_improvement"):
            assert isinstance(artifact[field], float), f"{field} must be float"


# ===========================================================================
# Tests for main()
# ===========================================================================


class TestMain:
    """Tests for main() entry point of experiment_443."""

    def test_main_creates_deliverable(self, tmp_path: Path) -> None:
        """main() writes the deliverable JSON and it is parseable."""
        deliverable = tmp_path / "results" / "experiment_443_eorm_jepa_live_retrain.json"
        deliverable.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "DELIVERABLE", str(deliverable.relative_to(tmp_path))),
            patch.object(_mod, "run_experiment", return_value={"honest_verdict": "synthetic_only", "retro_024_closed": False}),
        ):
            _mod.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert "honest_verdict" in data

    def test_main_uses_watchdog_with_correct_experiment_id(self) -> None:
        """main() creates ExperimentTimeoutWatchdog with experiment_id=443."""
        watchdog_calls: list[int] = []

        class FakeWatchdog:
            def __init__(self, exp_id: int, **kwargs: Any) -> None:
                watchdog_calls.append(exp_id)

            def __enter__(self) -> "FakeWatchdog":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        with (
            patch("carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog", FakeWatchdog),
            patch.object(_mod, "ExperimentTimeoutWatchdog", FakeWatchdog),
            patch.object(_mod, "run_experiment", return_value={"honest_verdict": "synthetic_only", "retro_024_closed": False}),
            patch("builtins.open", MagicMock()),
            patch("json.dump"),
        ):
            _mod.main()

        assert 443 in watchdog_calls
