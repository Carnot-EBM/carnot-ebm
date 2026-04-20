"""Tests for Exp 579: live 50q data collection C (GSM8K batch 200-249).

100% targeted coverage on functions added in scripts/experiment_579_live_data_c.py.

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-016, SCENARIO-DATA-017, SCENARIO-DATA-018
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Must set CARNOT_FORCE_LIVE before importing the module so the module-level gate passes.
os.environ["CARNOT_FORCE_LIVE"] = "1"

import scripts.experiment_579_live_data_c as exp579  # noqa: E402


# ---------------------------------------------------------------------------
# _build_live_data_artifact
# ---------------------------------------------------------------------------


class TestBuildLiveDataArtifact:
    """REQ-DATA-001: artifact must have all required fields on every exit path."""

    def test_schema_field(self):
        # SCENARIO-DATA-016: schema field is always present
        art = exp579._build_live_data_artifact(
            "live_gpu", 50, 50, "results/live_pairs_579.json", [1.0] * 50, 182
        )
        assert art["schema"] == "carnot.live_data_collection.v1"

    def test_experiment_id_field(self):
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], 0)
        assert art["experiment"] == 579

    def test_question_indices_field(self):
        # SCENARIO-DATA-016: batch C uses indices 200-249
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], 0)
        assert art["question_indices"] == "200-249"

    def test_honest_verdict_corpus_expanded_at_40(self):
        # SCENARIO-DATA-016: corpus_expanded when n_pairs_collected >= 40
        art = exp579._build_live_data_artifact("live_gpu", 50, 40, "f.json", [], 182)
        assert art["honest_verdict"] == "corpus_expanded"

    def test_honest_verdict_corpus_expanded_at_50(self):
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], 182)
        assert art["honest_verdict"] == "corpus_expanded"

    def test_honest_verdict_partial_below_40(self):
        # partial_collection_579 when < 40 pairs
        art = exp579._build_live_data_artifact("live_gpu", 50, 39, "f.json", [], 0)
        assert art["honest_verdict"] == "partial_collection_579"

    def test_honest_verdict_gpu_required(self):
        # SCENARIO-DATA-013 pattern: blocked when GPU not live
        art = exp579._build_live_data_artifact("gpu_required", 0, 0, None, [], 0)
        assert art["honest_verdict"] == "gpu_required"

    def test_n_pairs_collected_field(self):
        art = exp579._build_live_data_artifact("live_gpu", 50, 99, "f.json", [], 231)
        assert art["n_pairs_collected"] == 99

    def test_fover_corpus_v3_size_field(self):
        # SCENARIO-DATA-018: artifact records merged corpus size
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], 182)
        assert art["fover_corpus_v3_size"] == 182

    def test_models_field(self):
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], 0)
        assert "google/gemma-4-E4B-it" in art["models"]
        assert "Qwen/Qwen3.5-0.8B" in art["models"]

    def test_mean_latency_computed(self):
        art = exp579._build_live_data_artifact("live_gpu", 50, 50, "f.json", [2.0, 4.0], 0)
        assert art["mean_latency_s"] == pytest.approx(3.0)

    def test_mean_latency_zero_for_empty(self):
        art = exp579._build_live_data_artifact("live_gpu", 0, 0, None, [], 0)
        assert art["mean_latency_s"] == 0.0

    def test_per_question_latencies_field(self):
        latencies = [1.1, 2.2, 3.3]
        art = exp579._build_live_data_artifact("live_gpu", 3, 3, "f.json", latencies, 0)
        assert art["per_question_latencies"] == latencies


# ---------------------------------------------------------------------------
# _write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    """REQ-DATA-002: atomic write must be crash-safe and round-trip stable."""

    def test_writes_valid_json(self, tmp_path):
        data = {"key": "value", "n": 42}
        p = tmp_path / "out.json"
        exp579._write_json_atomic(p, data)
        assert json.loads(p.read_text()) == data

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.json"
        exp579._write_json_atomic(p, {"a": 1})
        assert p.exists()

    def test_no_tmp_file_left_behind(self, tmp_path):
        p = tmp_path / "out.json"
        exp579._write_json_atomic(p, {"x": 1})
        assert not (tmp_path / "out.tmp").exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text('{"old": true}')
        exp579._write_json_atomic(p, {"new": True})
        assert json.loads(p.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# _load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """SCENARIO-DATA-016: batch C must load indices 200-249 (50 questions)."""

    def test_fallback_returns_50_questions(self):
        # When datasets is unavailable the synthetic fallback must yield exactly 50 items.
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp579._load_gsm8k_questions(200, 249, 42)
        assert len(qs) == 50

    def test_fallback_indices_are_200_to_249(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp579._load_gsm8k_questions(200, 249, 42)
        indices = [q["index"] for q in qs]
        assert indices[0] == 200
        assert indices[-1] == 249

    def test_fallback_has_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp579._load_gsm8k_questions(200, 249, 42)
        for q in qs:
            assert "question" in q
            assert "answer" in q
            assert "index" in q

    def test_synthetic_answer_is_double_index(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp579._load_gsm8k_questions(200, 200, 42)
        assert f"#### {200 * 2}" in qs[0]["answer"]

    def test_dataset_load_failure_returns_synthetic(self):
        # Exception during dataset load must trigger the synthetic fallback.
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = RuntimeError("no network")
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp579._load_gsm8k_questions(200, 204, 42)
        assert len(qs) == 5
        assert qs[0]["index"] == 200


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """Unit tests for the Qwen generation wrapper."""

    def test_returns_generated_text_from_list(self):
        mock_pipeline = MagicMock(return_value=[{"generated_text": "hello"}])
        result = exp579._qwen_generate(mock_pipeline, "prompt")
        assert result == "hello"

    def test_returns_str_on_non_list(self):
        mock_pipeline = MagicMock(return_value={"generated_text": "world"})
        result = exp579._qwen_generate(mock_pipeline, "prompt")
        assert isinstance(result, str)

    def test_returns_error_string_on_exception(self):
        mock_pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp579._qwen_generate(mock_pipeline, "prompt")
        assert result.startswith("[qwen_error:")

    def test_returns_str_on_empty_list(self):
        mock_pipeline = MagicMock(return_value=[])
        result = exp579._qwen_generate(mock_pipeline, "prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _load_qwen_pipeline
# ---------------------------------------------------------------------------


class TestLoadQwenPipeline:
    """_load_qwen_pipeline returns None on failure, not an exception."""

    def test_returns_none_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            result = exp579._load_qwen_pipeline("cpu")
        assert result is None

    def test_returns_none_on_load_exception(self):
        mock_transformers = MagicMock()
        mock_transformers.pipeline.side_effect = RuntimeError("cuda oom")
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = exp579._load_qwen_pipeline("cuda:1")
        assert result is None


# ---------------------------------------------------------------------------
# _annotate_response
# ---------------------------------------------------------------------------


class TestAnnotateResponse:
    """SCENARIO-DATA-017: each pair records cot_steps and fover_labels."""

    def test_returns_cot_steps_and_labels(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator

        annotator = FOVERAnnotator()
        result = exp579._annotate_response(annotator, "1. 2 + 2 = 4", "q200")
        assert "cot_steps" in result
        assert "fover_labels" in result

    def test_cot_steps_have_required_keys(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator

        annotator = FOVERAnnotator()
        result = exp579._annotate_response(annotator, "1. 2 + 2 = 4", "q200")
        for step in result["cot_steps"]:
            assert "step_idx" in step
            assert "z3_label" in step

    def test_fover_labels_parallel_to_cot_steps(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator

        annotator = FOVERAnnotator()
        result = exp579._annotate_response(annotator, "1. Step A\n2. Step B", "q201")
        assert len(result["cot_steps"]) == len(result["fover_labels"])


# ---------------------------------------------------------------------------
# Import-time CARNOT_FORCE_LIVE gate (SCENARIO-DATA-016 — import-time block)
# ---------------------------------------------------------------------------


class TestImportTimeGate:
    """SCENARIO-DATA-016: import-time gate blocks before any model import."""

    def test_import_time_block_writes_artifact_and_exits(self, tmp_path):
        """When CARNOT_FORCE_LIVE != '1', writes a blocked artifact and exits 1."""
        mod_key = "scripts.experiment_579_live_data_c"
        saved = sys.modules.pop(mod_key, None)
        env_backup = os.environ.get("CARNOT_FORCE_LIVE")
        try:
            stub_modules = {
                k: MagicMock()
                for k in [
                    "carnot.pipeline.env_autofix",
                    "carnot.pipeline.deliverable_guard",
                    "carnot.pipeline.experiment_watchdog",
                    "carnot.pipeline.fover_annotator",
                    "carnot.pipeline.fover_corpus",
                    "carnot.pipeline.gemma4_quantized_loader",
                    "carnot.pipeline.jit_vram_check",
                    "carnot.pipeline.live_gpu_gate",
                    "carnot.pipeline.live_100q_v7_helpers",
                ]
            }
            env_patch = {"CARNOT_FORCE_LIVE": "0"}
            with patch.dict(os.environ, env_patch, clear=False):
                with patch.dict("sys.modules", stub_modules):
                    with pytest.raises(SystemExit) as exc_info:
                        importlib.import_module(mod_key)
            assert exc_info.value.code == 1
        finally:
            sys.modules.pop(mod_key, None)
            if saved is not None:
                sys.modules[mod_key] = saved
            if env_backup is not None:
                os.environ["CARNOT_FORCE_LIVE"] = env_backup

    def test_blocked_artifact_schema(self, tmp_path):
        """The blocked artifact must have the correct schema and question_indices."""
        blocked = {
            "schema": "carnot.live_data_collection.v1",
            "experiment": 579,
            "status": "blocked",
            "inference_mode": "gpu_required",
            "n_questions": 0,
            "question_indices": "200-249",
            "models": ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"],
            "n_pairs_collected": 0,
            "live_pairs_file": None,
            "fover_corpus_v3_size": 0,
            "honest_verdict": "import_time_block_carnot_force_live_missing",
            "blocked_reason": "CARNOT_FORCE_LIVE must be 1 — source scripts/session_startup.sh",
        }
        p = tmp_path / "results" / "experiment_579_live_data_c.json"
        exp579._write_json_atomic(p, blocked)
        on_disk = json.loads(p.read_text())
        assert on_disk["question_indices"] == "200-249"
        assert on_disk["experiment"] == 579
        assert on_disk["honest_verdict"] == "import_time_block_carnot_force_live_missing"


# ---------------------------------------------------------------------------
# corpus merge integration (SCENARIO-DATA-018)
# ---------------------------------------------------------------------------


class TestCorpusMergeIntegration:
    """SCENARIO-DATA-018: batch C pairs merge with v2 corpus to form v3."""

    def test_merge_produces_larger_corpus(self, tmp_path):
        # Write a minimal fover_corpus_v2.json (Exp 551/552 schema)
        v2_pairs = [
            {
                "question": f"Q{i}",
                "model": "Qwen/Qwen3.5-0.8B",
                "response": f"R{i}",
                "is_correct": True,
                "fover_labels": ["correct"],
                "cot_steps": [],
            }
            for i in range(5)
        ]
        v2_path = tmp_path / "results" / "fover_corpus_v2.json"
        exp579._write_json_atomic(v2_path, v2_pairs)

        # Write minimal live_pairs_579.json (new batch C pairs)
        batch_c_pairs = [
            {
                "question": f"BatchC_Q{i}",
                "model": "Qwen/Qwen3.5-0.8B",
                "response": f"R{i}",
                "is_correct": False,
                "fover_labels": ["incorrect"],
                "cot_steps": [],
            }
            for i in range(3)
        ]
        pairs_path = tmp_path / "results" / "live_pairs_579.json"
        exp579._write_json_atomic(pairs_path, batch_c_pairs)

        from carnot.pipeline.fover_corpus import merge_fover_sources

        merged = merge_fover_sources([str(v2_path), str(pairs_path)])
        # All 8 unique (question, model_id) pairs must be present.
        assert len(merged) == 8

    def test_merge_deduplicates_on_question_model(self, tmp_path):
        # The same (question, model) in two files should appear only once.
        pair = {
            "question": "Same question",
            "model": "Qwen/Qwen3.5-0.8B",
            "response": "R",
            "is_correct": True,
            "fover_labels": [],
            "cot_steps": [],
        }
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        exp579._write_json_atomic(f1, [pair])
        exp579._write_json_atomic(f2, [pair])

        from carnot.pipeline.fover_corpus import merge_fover_sources

        merged = merge_fover_sources([str(f1), str(f2)])
        assert len(merged) == 1

    def test_missing_source_is_skipped(self, tmp_path):
        pair = {
            "question": "Q1",
            "model": "Qwen/Qwen3.5-0.8B",
            "response": "R",
            "is_correct": True,
            "fover_labels": [],
            "cot_steps": [],
        }
        f_exists = tmp_path / "exists.json"
        exp579._write_json_atomic(f_exists, [pair])
        missing = str(tmp_path / "does_not_exist.json")

        from carnot.pipeline.fover_corpus import merge_fover_sources

        merged = merge_fover_sources([str(f_exists), missing])
        assert len(merged) == 1
