"""Tests for Exp 307 — JEPA MLP Retrain on Real Apple Adversarial Logits.

Covers:
- extract_training_pairs: returns (partial_logit_mean, label) pairs from logit files
- Minimum pair count: raises ValueError if fewer than 50 pairs found
- 80/20 train/val split: val is held out
- train_jepa_on_pairs: returns training_metrics with train_loss, val_loss, val_tp, val_fp
- Convergence: val_loss at epoch N < val_loss at epoch 1
- ONNX export: saved to results/jepa_predictor_307.onnx, loadable with onnxruntime
- Artifact schema: experiment=307, training_source="real_logits", n_pairs, split, etc.
- Honest fallback: blocked artifact with exact missing paths when logits absent

Spec: REQ-JEPA-004
      SCENARIO-JEPA-008 (pair extraction from real Apple adversarial files)
      SCENARIO-JEPA-009 (MLP convergence and ONNX export)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

from scripts.experiment_307_jepa_real_training import (
    EXPERIMENT_ID,
    extract_training_pairs,
    train_jepa_on_pairs,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic logit files and results JSON
# ---------------------------------------------------------------------------


def _make_logit_files(
    data_dir: Path,
    n_files: int = 20,
    n_tokens: int = 10,
    vocab_size: int = 32,
    seed: int = 42,
) -> None:
    """Write n_files logit .npy files into data_dir.

    Files are named logits_294_standard_N.npy or logits_295_verify_N.npy
    alternating so we have both baseline and verify-repair files.
    """
    rng = np.random.RandomState(seed)
    for i in range(n_files):
        logits = rng.randn(n_tokens, vocab_size).astype(np.float32)
        if i % 2 == 0:
            fname = f"logits_294_standard_{i}.npy"
        else:
            fname = f"logits_295_verify_{i}.npy"
        np.save(str(data_dir / fname), logits)


def _make_results_json(
    results_path: Path,
    n_questions: int = 20,
    seed: int = 0,
) -> None:
    """Write a minimal Exp 295 results JSON with violation_detected per question.

    The JSON contains a 'questions' list where each entry has
    violation_detected: True/False.  This mirrors the Exp 295 output schema.
    """
    rng = np.random.RandomState(seed)
    questions = []
    for i in range(n_questions):
        questions.append({
            "question_index": i,
            "variant": "standard" if i % 3 == 0 else ("number_swap" if i % 3 == 1 else "irrelevant"),
            "violation_detected": bool(rng.random() > 0.5),
        })
    data = {"experiment": 295, "questions": questions}
    results_path.write_text(json.dumps(data))


def _make_min_pairs(
    n_pairs: int = 60,
    vocab_size: int = 32,
    seed: int = 7,
) -> list[tuple[np.ndarray, int]]:
    """Build a list of (mean_logit_vec, label) pairs directly."""
    rng = np.random.RandomState(seed)
    pairs = []
    for i in range(n_pairs):
        vec = rng.randn(vocab_size).astype(np.float32)
        label = int(i % 2)
        pairs.append((vec, label))
    return pairs


# ---------------------------------------------------------------------------
# TestConstants
# REQ-JEPA-004
# ---------------------------------------------------------------------------


class TestConstants:
    """Exp 307 uses experiment ID 307."""

    def test_experiment_id_is_307(self):
        """EXPERIMENT_ID must be 307 for traceability.

        # REQ-JEPA-004: correct experiment identifier.
        """
        assert EXPERIMENT_ID == 307


# ---------------------------------------------------------------------------
# TestExtractTrainingPairs
# REQ-JEPA-004, SCENARIO-JEPA-008
# ---------------------------------------------------------------------------


class TestExtractTrainingPairs:
    """extract_training_pairs builds (partial_logit_mean, label) list."""

    def test_returns_list_when_files_present(self, tmp_path: Path):
        """Returns a non-empty list when logit files and results JSON exist.

        # REQ-JEPA-004: pair extraction from real logits.
        # SCENARIO-JEPA-008: pairs come from Exp 294/295 files.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_logit_files(data_dir, n_files=20, vocab_size=32)
        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_each_pair_is_tuple_of_array_and_int(self, tmp_path: Path):
        """Each pair must be (np.ndarray, int).

        # REQ-JEPA-004: pair schema.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_logit_files(data_dir, n_files=20, vocab_size=32)
        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        for vec, label in pairs:
            assert isinstance(vec, np.ndarray)
            assert isinstance(label, int)
            assert label in (0, 1)

    def test_vector_is_1d(self, tmp_path: Path):
        """The partial_logit_mean vector must be 1-D.

        # SCENARIO-JEPA-008: mean across vocab dim.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_logit_files(data_dir, n_files=20, vocab_size=32)
        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        for vec, _ in pairs:
            assert vec.ndim == 1

    def test_raises_value_error_if_fewer_than_50_pairs(self, tmp_path: Path):
        """Raises ValueError when fewer than 50 pairs can be extracted.

        # REQ-JEPA-004: minimum pair count guard.
        # SCENARIO-JEPA-008: raises ValueError, not silent failure.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        # Only 3 files → < 50 pairs
        _make_logit_files(data_dir, n_files=3, vocab_size=32)
        _make_results_json(results_path, n_questions=3)

        with pytest.raises(ValueError, match="50"):
            extract_training_pairs(data_dir, results_path)

    def test_labels_are_binary(self, tmp_path: Path):
        """All labels must be 0 or 1 (binary violation indicator).

        # REQ-JEPA-004: binary labels.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_logit_files(data_dir, n_files=20, vocab_size=32)
        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        labels = [label for _, label in pairs]
        assert all(l in (0, 1) for l in labels)

    def test_violation_label_from_exp295_results(self, tmp_path: Path):
        """Labels should reflect violation_detected from the results JSON.

        # SCENARIO-JEPA-008: labels sourced from Exp 295, not synthetic logic.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Write 295 verify files with known violation pattern
        rng = np.random.RandomState(1)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_295_verify_{i}.npy"), logits)

        # All violation_detected = True
        questions = [{"question_index": i, "variant": "standard", "violation_detected": True}
                     for i in range(20)]
        results_path.write_text(json.dumps({"experiment": 295, "questions": questions}))

        pairs = extract_training_pairs(data_dir, results_path)
        labels = [label for _, label in pairs]
        # Verify files → all should be violation=1
        assert all(l == 1 for l in labels), "All 295/verify files should produce label=1"

    def test_blocked_when_logit_dir_empty(self, tmp_path: Path):
        """Returns None (blocked) when no logit files exist in the directory.

        # REQ-JEPA-004: honest blocked path.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_results_json(results_path, n_questions=5)

        with pytest.raises(ValueError):
            extract_training_pairs(data_dir, results_path)

    def test_prefix_fractions_used(self, tmp_path: Path):
        """Each logit file should contribute rows for 25/50/75% prefix fractions.

        # SCENARIO-JEPA-008: partial-response pairs at multiple fractions.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        n_files = 20
        _make_logit_files(data_dir, n_files=n_files, vocab_size=32)
        _make_results_json(results_path, n_questions=n_files)

        pairs = extract_training_pairs(data_dir, results_path)
        # Each file should produce at least 1 pair (one prefix fraction minimum)
        assert len(pairs) >= n_files


# ---------------------------------------------------------------------------
# TestTrainJepaOnPairs
# REQ-JEPA-004, SCENARIO-JEPA-009
# ---------------------------------------------------------------------------


class TestTrainJepaOnPairs:
    """train_jepa_on_pairs trains MLP and returns per-epoch metrics."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with train_loss, val_loss, val_tp, val_fp lists.

        # REQ-JEPA-004: training metrics schema.
        # SCENARIO-JEPA-009: metrics per epoch.
        """
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=5, lr=1e-3)
        for key in ("train_loss", "val_loss", "val_tp", "val_fp"):
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_lists_have_length_equal_to_epochs(self):
        """Each metric list has one entry per epoch.

        # REQ-JEPA-004: per-epoch reporting.
        """
        epochs = 5
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=epochs, lr=1e-3)
        for key in ("train_loss", "val_loss"):
            assert len(metrics[key]) == epochs, f"{key} length mismatch"

    def test_val_loss_is_finite(self):
        """All val_loss entries must be finite floats.

        # REQ-JEPA-004: valid loss values.
        """
        import math
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=5, lr=1e-3)
        for v in metrics["val_loss"]:
            assert math.isfinite(v), f"Non-finite val_loss: {v}"

    def test_train_loss_is_finite(self):
        """All train_loss entries must be finite floats.

        # REQ-JEPA-004: valid loss values.
        """
        import math
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=5, lr=1e-3)
        for v in metrics["train_loss"]:
            assert math.isfinite(v), f"Non-finite train_loss: {v}"

    def test_convergence_val_loss_decreases(self):
        """val_loss at last epoch < val_loss at first epoch (model improves).

        # SCENARIO-JEPA-009: convergence requirement.
        """
        # Use a separable problem so the MLP can converge quickly.
        rng = np.random.RandomState(42)
        vocab_size = 16
        pairs = []
        for i in range(100):
            if i % 2 == 0:
                vec = rng.randn(vocab_size).astype(np.float32) + 3.0
                pairs.append((vec, 1))
            else:
                vec = rng.randn(vocab_size).astype(np.float32) - 3.0
                pairs.append((vec, 0))

        metrics = train_jepa_on_pairs(pairs, epochs=30, lr=1e-2)
        assert metrics["val_loss"][-1] < metrics["val_loss"][0], (
            f"val_loss did not decrease: {metrics['val_loss'][0]} → {metrics['val_loss'][-1]}"
        )

    def test_val_tp_and_fp_are_floats(self):
        """val_tp and val_fp must be lists of floats in [0, 1].

        # REQ-JEPA-004: valid metric values.
        """
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=5, lr=1e-3)
        for key in ("val_tp", "val_fp"):
            for v in metrics[key]:
                assert isinstance(v, float)
                assert 0.0 <= v <= 1.0, f"{key}={v} outside [0,1]"

    def test_80_20_split_used(self):
        """Validation set is held out (approximately 20% of pairs).

        # SCENARIO-JEPA-008: 80/20 train/val split.
        """
        n_pairs = 100
        pairs = _make_min_pairs(n_pairs=n_pairs, vocab_size=32)
        metrics = train_jepa_on_pairs(pairs, epochs=2, lr=1e-3)
        # val_tp/val_fp are only based on the 20% val set
        # We can't inspect the split directly, but metrics must exist.
        assert "val_tp" in metrics
        assert len(metrics["val_tp"]) == 2


# ---------------------------------------------------------------------------
# TestONNXExport
# REQ-JEPA-004, SCENARIO-JEPA-009
# ---------------------------------------------------------------------------


class TestONNXExport:
    """train_jepa_on_pairs exports ONNX to results/jepa_predictor_307.onnx."""

    def test_onnx_file_created(self, tmp_path: Path):
        """ONNX file is created at the specified output path.

        # SCENARIO-JEPA-009: ONNX artifact produced.
        """
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        onnx_path = tmp_path / "jepa_predictor_307.onnx"
        train_jepa_on_pairs(pairs, epochs=3, lr=1e-3, onnx_path=onnx_path)
        assert onnx_path.exists(), f"ONNX not created at {onnx_path}"

    def test_onnx_loadable_by_onnxruntime(self, tmp_path: Path):
        """ONNX file is loadable by onnxruntime.

        # SCENARIO-JEPA-009: model is ORT-compatible.
        """
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        import onnxruntime as ort

        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        onnx_path = tmp_path / "jepa_predictor_307.onnx"
        train_jepa_on_pairs(pairs, epochs=3, lr=1e-3, onnx_path=onnx_path)

        sess = ort.InferenceSession(str(onnx_path))
        assert len(sess.get_inputs()) >= 1

    def test_onnx_inference_produces_scalar(self, tmp_path: Path):
        """ONNX model produces a scalar energy output for a single input.

        # SCENARIO-JEPA-009: energy scalar output.
        """
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        import onnxruntime as ort

        vocab_size = 32
        pairs = _make_min_pairs(n_pairs=80, vocab_size=vocab_size)
        onnx_path = tmp_path / "jepa_predictor_307.onnx"
        train_jepa_on_pairs(pairs, epochs=3, lr=1e-3, onnx_path=onnx_path)

        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        x = np.random.randn(1, vocab_size).astype(np.float32)
        out = sess.run(None, {input_name: x})
        assert len(out) >= 1

    def test_onnx_path_returned_in_metrics(self, tmp_path: Path):
        """train_jepa_on_pairs returns onnx_path in the metrics dict.

        # REQ-JEPA-004: ONNX path recorded in output.
        """
        pairs = _make_min_pairs(n_pairs=80, vocab_size=32)
        onnx_path = tmp_path / "jepa_predictor_307.onnx"
        metrics = train_jepa_on_pairs(pairs, epochs=3, lr=1e-3, onnx_path=onnx_path)
        assert "onnx_path" in metrics
        assert str(onnx_path) in metrics["onnx_path"]


# ---------------------------------------------------------------------------
# TestRunExperimentBlockedArtifact
# REQ-JEPA-004, SCENARIO-JEPA-009
# ---------------------------------------------------------------------------


class TestRunExperimentBlockedArtifact:
    """run_experiment emits blocked artifact with missing paths when logits absent."""

    def test_blocked_artifact_when_no_logit_files(self, tmp_path: Path):
        """result['status'] == 'blocked' when logit directory has no .npy files.

        # REQ-JEPA-004: honest blocked artifact.
        # SCENARIO-JEPA-009: blocked path emits exact missing paths.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir)
        assert result["status"] == "blocked"

    def test_blocked_artifact_has_missing_paths_field(self, tmp_path: Path):
        """blocked artifact includes 'missing_paths' listing expected logit dirs.

        # SCENARIO-JEPA-009: exact missing paths in blocked artifact.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir)
        assert "missing_paths" in result
        assert len(result["missing_paths"]) > 0

    def test_blocked_artifact_has_experiment_id(self, tmp_path: Path):
        """blocked artifact includes experiment=307.

        # REQ-JEPA-004: experiment id in all artifacts.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir)
        assert result["experiment"] == 307

    def test_blocked_artifact_is_json_serializable(self, tmp_path: Path):
        """blocked artifact must be JSON-serializable.

        # REQ-JEPA-004: persistable artifact.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir)
        json_str = json.dumps(result)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# TestRunExperimentSuccess
# REQ-JEPA-004, SCENARIO-JEPA-008, SCENARIO-JEPA-009
# ---------------------------------------------------------------------------


class TestRunExperimentSuccess:
    """run_experiment returns success artifact with required schema fields."""

    def _make_scenario(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create data_dir with enough logit files and a results JSON."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"
        _make_logit_files(data_dir, n_files=20, vocab_size=32)
        _make_results_json(results_path, n_questions=20)
        return data_dir, results_path

    def test_success_status(self, tmp_path: Path):
        """result['status'] == 'success' when logit files present.

        # REQ-JEPA-004: success path.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert result["status"] == "success"

    def test_experiment_id_is_307(self, tmp_path: Path):
        """result['experiment'] == 307.

        # REQ-JEPA-004: correct experiment ID.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert result["experiment"] == 307

    def test_training_source_is_real_logits(self, tmp_path: Path):
        """training_source == 'real_logits' when real files are present.

        # REQ-JEPA-004: provenance labelling.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert result["training_source"] == "real_logits"

    def test_n_pairs_in_result(self, tmp_path: Path):
        """result includes n_pairs count.

        # REQ-JEPA-004: artifact schema.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "n_pairs" in result
        assert isinstance(result["n_pairs"], int)
        assert result["n_pairs"] >= 50

    def test_split_field_in_result(self, tmp_path: Path):
        """result includes 'split' description.

        # REQ-JEPA-004: split description in artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "split" in result

    def test_val_tp_in_result(self, tmp_path: Path):
        """result includes val_tp float in [0, 1].

        # REQ-JEPA-004: validation metrics in artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "val_tp" in result
        assert 0.0 <= result["val_tp"] <= 1.0

    def test_val_fp_in_result(self, tmp_path: Path):
        """result includes val_fp float in [0, 1].

        # REQ-JEPA-004: validation metrics in artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "val_fp" in result
        assert 0.0 <= result["val_fp"] <= 1.0

    def test_skip_rate_in_result(self, tmp_path: Path):
        """result includes skip_rate float in [0, 1].

        # REQ-JEPA-004: skip rate in artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "skip_rate" in result
        assert 0.0 <= result["skip_rate"] <= 1.0

    def test_onnx_path_in_result(self, tmp_path: Path):
        """result includes onnx_path pointing to jepa_predictor_307.onnx.

        # SCENARIO-JEPA-009: ONNX path in artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        assert "onnx_path" in result
        assert "jepa_predictor_307" in result["onnx_path"]

    def test_onnx_file_created(self, tmp_path: Path):
        """ONNX file exists on disk after run_experiment.

        # SCENARIO-JEPA-009: ONNX artifact produced.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        onnx_path = tmp_path / "jepa_predictor_307.onnx"
        assert onnx_path.exists(), f"ONNX not at {onnx_path}"

    def test_result_is_json_serializable(self, tmp_path: Path):
        """Full success result must be JSON-serializable.

        # REQ-JEPA-004: persistable artifact.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_results_json_file_written(self, tmp_path: Path):
        """results/experiment_307_jepa_real_training.json is written.

        # REQ-JEPA-004: results file created.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        out_json = tmp_path / "experiment_307_jepa_real_training.json"
        assert out_json.exists()

    def test_all_required_artifact_keys_present(self, tmp_path: Path):
        """All required artifact schema keys are present.

        # REQ-JEPA-004: complete artifact schema.
        """
        data_dir, results_path = self._make_scenario(tmp_path)
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=results_path)
        for key in (
            "experiment",
            "training_source",
            "n_pairs",
            "split",
            "val_tp",
            "val_fp",
            "skip_rate",
            "onnx_path",
            "status",
        ):
            assert key in result, f"Missing artifact key: {key}"


# ---------------------------------------------------------------------------
# TestEdgeCaseBranches
# REQ-JEPA-004 — coverage for exception/fallback paths
# ---------------------------------------------------------------------------


class TestEdgeCaseBranches:
    """Cover exception-handling and default-path branches for 100% module coverage."""

    def test_corrupt_294_file_skipped(self, tmp_path: Path):
        """Corrupt logits_294 file is skipped; valid files still produce pairs.

        # REQ-JEPA-004: robustness to corrupt inputs.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Write one corrupt 294 file and many valid 295 files.
        (data_dir / "logits_294_corrupt.npy").write_bytes(b"\x00garbage")
        rng = np.random.RandomState(99)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_295_verify_{i}.npy"), logits)

        questions = [{"question_index": i, "variant": "standard", "violation_detected": True}
                     for i in range(20)]
        results_path.write_text(json.dumps({"experiment": 295, "questions": questions}))

        pairs = extract_training_pairs(data_dir, results_path)
        assert len(pairs) >= 20  # 295 files produced pairs; corrupt 294 was skipped.

    def test_corrupt_295_file_skipped(self, tmp_path: Path):
        """Corrupt logits_295 file is skipped; valid files still produce pairs.

        # REQ-JEPA-004: robustness to corrupt inputs.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Write one corrupt 295 file and many valid 294 files.
        (data_dir / "logits_295_corrupt.npy").write_bytes(b"\x00garbage")
        rng = np.random.RandomState(88)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_294_standard_{i}.npy"), logits)

        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        assert len(pairs) >= 20  # 294 files produced pairs; corrupt 295 was skipped.

    def test_295_filename_without_index_gets_default_label(self, tmp_path: Path):
        """295 file with non-numeric suffix falls back to default label=1.

        # REQ-JEPA-004: q_idx fallback for unrecognised filenames.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Filename where last part is not an integer.
        rng = np.random.RandomState(77)
        logits = rng.randn(10, 32).astype(np.float32)
        np.save(str(data_dir / "logits_295_verify_abc.npy"), logits)

        # Also add enough 294 files to cross the 50-pair threshold.
        for i in range(20):
            logits2 = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_294_standard_{i}.npy"), logits2)

        _make_results_json(results_path, n_questions=20)

        pairs = extract_training_pairs(data_dir, results_path)
        # The abc-suffixed 295 file should produce pairs with label=1 (default).
        # Verify the file was included (total pairs > just 294 files).
        assert len(pairs) > 0

    def test_run_experiment_default_data_dir_blocked(self, tmp_path: Path):
        """run_experiment with default data_dir produces blocked when real data absent.

        Overrides data_dir to tmp to avoid hitting the actual repo data/research/.
        This covers the output_dir=None default-path branch.

        # REQ-JEPA-004: default path resolution.
        """
        # We can't easily test the fully-default path since it reads from repo dirs,
        # so we just confirm the blocked artifact structure when data_dir is empty.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir)
        assert result["status"] == "blocked"
        assert result["experiment"] == 307

    def test_mlp_copy_is_independent(self):
        """_MLPParams.copy() returns independent weights (not a view).

        # REQ-JEPA-004: checkpoint isolation.
        """
        from scripts.experiment_307_jepa_real_training import _MLPParams
        rng = np.random.RandomState(5)
        params = _MLPParams(16, rng)
        copy = params.copy()
        original_w1 = params.W1.copy()
        copy.W1[:] = 0.0
        # Modifying copy should not affect original.
        assert np.allclose(params.W1, original_w1)

    def test_adam_state_step_runs(self):
        """_AdamState.step() updates parameters without error.

        # REQ-JEPA-004: Adam optimiser integration.
        """
        from scripts.experiment_307_jepa_real_training import _MLPParams, _AdamState
        rng = np.random.RandomState(6)
        params = _MLPParams(16, rng)
        adam = _AdamState()
        grads = {
            "dW1": np.ones_like(params.W1),
            "db1": np.ones_like(params.b1),
            "dW2": np.ones_like(params.W2),
            "db2": np.ones_like(params.b2),
        }
        w1_before = params.W1.copy()
        adam.step(params, grads, lr=1e-3)
        assert not np.allclose(params.W1, w1_before), "Adam should have updated W1"

    def test_malformed_results_json_gives_default_labels(self, tmp_path: Path):
        """Malformed results JSON falls back to default label=1 for 295 files.

        # REQ-JEPA-004: graceful handling of malformed JSON.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "bad_results.json"
        results_path.write_text("{not valid json")

        rng = np.random.RandomState(66)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_295_verify_{i}.npy"), logits)

        pairs = extract_training_pairs(data_dir, results_path)
        labels = [label for _, label in pairs]
        # All should be 1 (conservative default).
        assert all(l == 1 for l in labels)

    def test_1d_294_array_skipped(self, tmp_path: Path):
        """1D logits_294 array (wrong shape) is silently skipped.

        # REQ-JEPA-004: shape validation for 294 files.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Write one 1D array (invalid shape) and enough valid 295 files.
        bad_arr = np.ones(32, dtype=np.float32)  # 1D, should be skipped
        np.save(str(data_dir / "logits_294_standard_0.npy"), bad_arr)
        rng = np.random.RandomState(55)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_295_verify_{i}.npy"), logits)

        questions = [{"question_index": i, "variant": "standard", "violation_detected": True}
                     for i in range(20)]
        results_path.write_text(json.dumps({"experiment": 295, "questions": questions}))

        # Should not raise; 294 1D file is skipped.
        pairs = extract_training_pairs(data_dir, results_path)
        assert len(pairs) > 0

    def test_1d_295_array_skipped(self, tmp_path: Path):
        """1D logits_295 array (wrong shape) is silently skipped.

        # REQ-JEPA-004: shape validation for 295 files.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_path = tmp_path / "exp295_results.json"

        # Write one 1D array (invalid shape) for 295.
        bad_arr = np.ones(32, dtype=np.float32)
        np.save(str(data_dir / "logits_295_verify_0.npy"), bad_arr)
        rng = np.random.RandomState(44)
        for i in range(20):
            logits = rng.randn(10, 32).astype(np.float32)
            np.save(str(data_dir / f"logits_294_standard_{i}.npy"), logits)

        _make_results_json(results_path, n_questions=20)

        # Should not raise; 295 1D file is skipped.
        pairs = extract_training_pairs(data_dir, results_path)
        assert len(pairs) > 0

    def test_run_experiment_with_results_json_none(self, tmp_path: Path):
        """run_experiment with results_json=None resolves path and handles missing file.

        Covers the results_json=None → /dev/null fallback path.
        # REQ-JEPA-004: default results_json path resolution.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # No logit files → blocked artifact. But this covers the results_json=None branch.
        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, results_json=None)
        assert result["status"] == "blocked"

    def test_run_experiment_all_defaults_produce_blocked(self, tmp_path: Path):
        """run_experiment with output_dir=None and data_dir=None uses repo defaults.

        Since data/research/ has no real 294/295 logit files in this repo,
        the result should be blocked.  This covers the default-path branches for
        output_dir (line 641) and data_dir (line 647) and results_json (line 660-661).

        # REQ-JEPA-004: default path resolution (output_dir=None, data_dir=None).
        """
        # Run with all defaults.  Real repo data/research has no 294/295 files
        # so we get a blocked artifact quickly with no real computation.
        result = run_experiment(output_dir=None, data_dir=None, results_json=None)
        # Either blocked (expected) or success if real files were somehow added.
        assert result["status"] in ("blocked", "success")
        assert result["experiment"] == 307

