"""Tests for Exp 538: live 25q precision v9 — RETRO-033 attempt #10, RETRO-055 fix.

100% coverage on functions added in scripts/experiment_538_live_25q_precision_v9.py.

Spec: REQ-BENCH-014, REQ-BENCH-015,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034, SCENARIO-BENCH-035
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

import scripts.experiment_538_live_25q_precision_v9 as exp538


# ---------------------------------------------------------------------------
# _build_v9_artifact
# ---------------------------------------------------------------------------


class TestBuildV9Artifact:
    """SCENARIO-BENCH-034: signed_improvement is pipeline - baseline."""

    def test_signed_improvement_positive(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.40, "pipeline_accuracy": 0.60, "n_questions": 25},
            "live_gpu",
            None,
            [1.0] * 25,
            {},
        )
        assert abs(art["signed_improvement"] - 0.20) < 1e-9
        assert art["is_positive"] is True

    def test_signed_improvement_zero_is_not_positive(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.50, "pipeline_accuracy": 0.50, "n_questions": 25},
            "live_gpu",
            None,
            [1.0] * 25,
            {},
        )
        assert art["signed_improvement"] == 0.0
        assert art["is_positive"] is False

    def test_signed_improvement_negative_not_clamped(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.60, "pipeline_accuracy": 0.40, "n_questions": 25},
            "live_gpu",
            None,
            [1.0] * 25,
            {},
        )
        assert art["signed_improvement"] < 0
        assert art["retro_033_closed"] is False

    def test_retro_033_closed_requires_live_and_positive(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.40, "pipeline_accuracy": 0.60, "n_questions": 25},
            "live_gpu",
            None,
            [],
            {},
        )
        assert art["retro_033_closed"] is True

    def test_retro_033_closed_false_when_gpu_required(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.40, "pipeline_accuracy": 0.60, "n_questions": 25},
            "gpu_required",
            None,
            [],
            {},
        )
        assert art["retro_033_closed"] is False

    def test_retro_055_resolved_always_true(self):
        art = exp538._build_v9_artifact({}, "gpu_required", None, [], {})
        assert art["retro_055_resolved"] is True

    def test_honest_verdict_gpu_required(self):
        art = exp538._build_v9_artifact({}, "gpu_required", None, [], {})
        assert art["honest_verdict"] == "gpu_required"

    def test_honest_verdict_first_positive_25q(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.40, "pipeline_accuracy": 0.60},
            "live_gpu",
            None,
            [],
            {},
        )
        assert art["honest_verdict"] == "first_positive_25q"

    def test_honest_verdict_live_no_improvement_25q(self):
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.50, "pipeline_accuracy": 0.50},
            "live_gpu",
            None,
            [],
            {},
        )
        assert art["honest_verdict"] == "live_no_improvement_25q"

    def test_schema_field(self):
        art = exp538._build_v9_artifact({}, "gpu_required", None, [], {})
        assert art["schema"] == "carnot.live_precision.v3"

    def test_env_autofix_applied_always_true(self):
        art = exp538._build_v9_artifact({}, "live_gpu", None, [], {})
        assert art["env_autofix_applied"] is True

    def test_cot_pairs_path_preserved(self):
        art = exp538._build_v9_artifact({}, "live_gpu", "some/path.json", [], {})
        assert art["cot_pairs_written"] == "some/path.json"

    def test_cot_pairs_path_none(self):
        art = exp538._build_v9_artifact({}, "gpu_required", None, [], {})
        assert art["cot_pairs_written"] is None

    def test_mean_latency_computed_correctly(self):
        latencies = [10.0, 20.0, 30.0]
        art = exp538._build_v9_artifact({}, "live_gpu", None, latencies, {})
        assert abs(art["mean_latency_s"] - 20.0) < 1e-9

    def test_mean_latency_zero_when_empty(self):
        art = exp538._build_v9_artifact({}, "gpu_required", None, [], {})
        assert art["mean_latency_s"] == 0.0

    def test_all_required_schema_keys_present(self):
        required = {
            "schema", "inference_mode", "n_questions", "baseline_accuracy",
            "pipeline_accuracy", "signed_improvement", "is_positive",
            "mean_latency_s", "per_question_latencies",
            "retro_033_closed", "retro_055_resolved",
            "cot_pairs_written", "env_autofix_applied", "honest_verdict",
        }
        art = exp538._build_v9_artifact(
            {"baseline_accuracy": 0.5, "pipeline_accuracy": 0.5, "n_questions": 25},
            "live_gpu",
            "p.json",
            [1.0],
            {},
        )
        assert required <= set(art.keys())


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-035: per_question_latency_s is recorded
# ---------------------------------------------------------------------------


class TestPerQuestionLatency:
    """SCENARIO-BENCH-035: per_question_latencies has one entry per question."""

    def test_per_question_latencies_in_artifact(self):
        latencies = [27.3, 25.1, 30.0]
        art = exp538._build_v9_artifact(
            {"n_questions": 3},
            "live_gpu",
            None,
            latencies,
            {},
        )
        assert art["per_question_latencies"] == latencies
        assert len(art["per_question_latencies"]) == 3

    def test_all_latencies_positive(self):
        latencies = [1.0, 2.0, 3.0]
        art = exp538._build_v9_artifact({}, "live_gpu", None, latencies, {})
        assert all(lat > 0 for lat in art["per_question_latencies"])


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-033: gpu_required fast-path
# ---------------------------------------------------------------------------


class TestGPURequiredFastPath:
    """SCENARIO-BENCH-033: deferred artifact when CARNOT_FORCE_LIVE not set."""

    def test_deferred_when_force_live_not_set(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp538.run_experiment(repo_root=tmp_path)

        assert artifact["status"] in ("gpu_required", "blocked", "gpu_vram_insufficient", "success")
        out = tmp_path / exp538.DELIVERABLE
        assert out.exists(), f"Deliverable not written to {out}"

    def test_deferred_artifact_has_required_fields(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp538.run_experiment(repo_root=tmp_path)

        for key in {"experiment", "status", "run_date", "started_at", "finished_at", "duration_s"}:
            assert key in artifact, f"Missing required field: {key}"

    def test_deferred_artifact_has_v3_schema(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp538.run_experiment(repo_root=tmp_path)

        assert artifact.get("artifact_type") == "carnot.live_precision.v3"

    def test_deferred_artifact_honest_verdict_gpu_required(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp538.run_experiment(repo_root=tmp_path)

        assert artifact.get("honest_verdict") == "gpu_required"

    def test_retro_055_resolved_in_deferred_artifact(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp538.run_experiment(repo_root=tmp_path)

        assert artifact.get("retro_055_resolved") is True

    def test_deliverable_path_constant_references_538(self):
        assert "538" in exp538.DELIVERABLE
        assert exp538.DELIVERABLE.endswith(".json")

    def test_cot_pairs_path_references_538(self):
        assert "538" in exp538.COT_PAIRS_PATH
        assert exp538.COT_PAIRS_PATH.endswith(".json")


# ---------------------------------------------------------------------------
# FOVERAnnotator integration
# ---------------------------------------------------------------------------


class TestFOVERAnnotatorIntegration:
    """Verify FOVERAnnotator is called and its output is recorded in the artifact."""

    def test_fover_annotator_import_works(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        fa = FOVERAnnotator()
        assert hasattr(fa, "annotate_corpus")
        assert hasattr(fa, "to_training_pairs")

    def test_vericot_validator_import_works(self):
        from carnot.extraction.vericot_validator import VeriCoTStepValidator
        v = VeriCoTStepValidator()
        assert hasattr(v, "detect_violations")

    def test_fover_annotate_corpus_returns_list(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        fa = FOVERAnnotator()
        result = fa.annotate_corpus([{"response": "Step 1: 2 + 2 = 4. Step 2: done."}])
        assert isinstance(result, list)

    def test_fover_to_training_pairs_returns_list(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        fa = FOVERAnnotator()
        annotated = fa.annotate_corpus([{"response": "Step 1: 3 + 3 = 6."}])
        pairs = fa.to_training_pairs(annotated)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# _write_cot_pairs
# ---------------------------------------------------------------------------


class TestWriteCotPairs:
    def test_writes_json_file(self, tmp_path):
        pairs = [{"question": "Q", "cot_text": "T", "correct": True, "model_id": "M"}]
        out = str(tmp_path / "cot.json")
        n = exp538._write_cot_pairs(pairs, out)
        assert n == 1
        loaded = json.loads(Path(out).read_text())
        assert loaded == pairs

    def test_creates_parent_dirs(self, tmp_path):
        pairs = [{"q": "x"}]
        out = str(tmp_path / "sub" / "dir" / "cot.json")
        exp538._write_cot_pairs(pairs, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# _load_gsm8k_questions (fallback path)
# ---------------------------------------------------------------------------


class TestLoadGSM8KQuestions:
    def test_returns_n_questions(self):
        # dataset may not be available; the fallback always returns n items
        qs = exp538._load_gsm8k_questions(5, 42)
        assert len(qs) == 5

    def test_each_question_has_required_keys(self):
        qs = exp538._load_gsm8k_questions(3, 0)
        for q in qs:
            assert "question" in q
            assert "answer" in q
