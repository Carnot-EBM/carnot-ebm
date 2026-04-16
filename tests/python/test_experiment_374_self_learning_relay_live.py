"""Tests for scripts/experiment_374_self_learning_relay_live.py.

Covers 100% of new helper functions:
  - diagnose_live_gpu: blocks when CARNOT_FORCE_LIVE != "1"
  - load_eorm_model: exp371 path, exp359 fallback, synthetic fallback
  - load_gsm8k_questions: success, ImportError, RuntimeError (too few rows)
  - extract_numeric_answer: boxed, "the answer is", ####, last-number, empty
  - is_correct_answer: numeric match, numeric mismatch, fallback string match
  - build_components: returns pipeline, library, tracker with correct types
  - run_live_batch: delegates to BatchedInferenceRunner, evaluates correctness
  - _load_model_pipeline: ImportError path; wrapper smoke test via mock
  - main(): blocked (no live flag), blocked (GPU unhealthy), blocked (gsm8k fail),
            blocked (model load fail), success path

All tests run without a live GPU via mocks.

Spec: REQ-LEARN-026, REQ-LEARN-027, SCENARIO-LEARN-050
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_374_self_learning_relay_live.py"

for _d in [str(REPO_ROOT), str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_374 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_374", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_374"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

diagnose_live_gpu = _mod.diagnose_live_gpu
load_eorm_model = _mod.load_eorm_model
load_gsm8k_questions = _mod.load_gsm8k_questions
extract_numeric_answer = _mod.extract_numeric_answer
is_correct_answer = _mod.is_correct_answer
build_components = _mod.build_components
run_live_batch = _mod.run_live_batch
_load_model_pipeline = _mod._load_model_pipeline


# ---------------------------------------------------------------------------
# diagnose_live_gpu
# ---------------------------------------------------------------------------


class TestDiagnoseGpu:
    """Tests for diagnose_live_gpu() guard."""

    def test_raises_when_not_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Spec: SCENARIO-LEARN-050 — blocked when CARNOT_FORCE_LIVE != 1."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE"):
            diagnose_live_gpu()

    def test_raises_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE"):
            diagnose_live_gpu()

    def test_passes_when_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        # Should not raise.
        diagnose_live_gpu()


# ---------------------------------------------------------------------------
# load_eorm_model
# ---------------------------------------------------------------------------


class TestLoadEormModel:
    """Tests for load_eorm_model() priority chain."""

    def test_exp371_loaded_when_present(self, tmp_path: Path) -> None:
        """Spec: SCENARIO-LEARN-050 — prefer Exp 371 model."""
        from carnot.models.eorm import EORMModel
        import jax.random as jr

        # Create and save a minimal model to a temp path.
        key = jr.PRNGKey(0)
        model = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)
        ckpt_path = tmp_path / "eorm_model_371_real.safetensors"
        model.save(str(ckpt_path))

        with patch.object(_mod, "_EORM_371_PATH", ckpt_path):
            loaded, source = load_eorm_model()

        assert source == "exp371_real"
        assert isinstance(loaded, EORMModel)

    def test_exp359_fallback(self, tmp_path: Path) -> None:
        """Spec: SCENARIO-LEARN-050 — fall back to Exp 359 when 371 absent."""
        from carnot.models.eorm import EORMModel
        import jax.random as jr

        key = jr.PRNGKey(1)
        model = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)
        ckpt_path = tmp_path / "eorm_model_359_real.safetensors"
        model.save(str(ckpt_path))

        absent = tmp_path / "eorm_model_371_real.safetensors"
        with patch.object(_mod, "_EORM_371_PATH", absent):
            with patch.object(_mod, "_EORM_359_PATH", ckpt_path):
                loaded, source = load_eorm_model()

        assert source == "exp359_real"
        assert isinstance(loaded, EORMModel)

    def test_synthetic_fallback(self, tmp_path: Path) -> None:
        """Spec: SCENARIO-LEARN-050 — fresh model when no checkpoint found."""
        from carnot.models.eorm import EORMModel

        absent_371 = tmp_path / "eorm_model_371_real.safetensors"
        absent_359 = tmp_path / "eorm_model_359_real.safetensors"

        with patch.object(_mod, "_EORM_371_PATH", absent_371):
            with patch.object(_mod, "_EORM_359_PATH", absent_359):
                loaded, source = load_eorm_model()

        assert source == "synthetic_fallback"
        assert isinstance(loaded, EORMModel)


# ---------------------------------------------------------------------------
# load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """Tests for load_gsm8k_questions()."""

    def test_import_error_propagated(self) -> None:
        """ImportError when datasets is not available."""
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                load_gsm8k_questions(n=5)

    def test_too_few_rows(self) -> None:
        """RuntimeError when split has fewer rows than requested."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=3)

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            with pytest.raises(RuntimeError, match="only 3 questions"):
                load_gsm8k_questions(n=10)

    def test_success(self) -> None:
        """Returns (questions, answers) lists of length n."""
        rows = [
            {"question": f"Q{i}", "answer": f"Step 1. #### {i}"}
            for i in range(10)
        ]

        class _FakeDataset:
            def __len__(self) -> int:
                return 10

            def select(self, idxs: Any) -> list[dict]:
                return [rows[i] for i in idxs]

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = _FakeDataset()

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            questions, answers = load_gsm8k_questions(n=5)

        assert len(questions) == 5
        assert len(answers) == 5
        assert questions[0] == "Q0"
        assert answers[0] == "0"


# ---------------------------------------------------------------------------
# extract_numeric_answer
# ---------------------------------------------------------------------------


class TestExtractNumericAnswer:
    """Tests for extract_numeric_answer() heuristics."""

    def test_latex_boxed(self) -> None:
        assert extract_numeric_answer(r"Therefore \boxed{42}") == "42"

    def test_latex_boxed_with_commas(self) -> None:
        assert extract_numeric_answer(r"\boxed{1,234}") == "1234"

    def test_the_answer_is(self) -> None:
        assert extract_numeric_answer("The answer is 99.") == "99"

    def test_the_answer_is_case_insensitive(self) -> None:
        assert extract_numeric_answer("THE ANSWER IS: 7") == "7"

    def test_gsm8k_marker(self) -> None:
        assert extract_numeric_answer("...calculations... #### 15") == "15"

    def test_last_number_fallback(self) -> None:
        assert extract_numeric_answer("foo 3 bar 7 baz") == "7"

    def test_negative_number(self) -> None:
        assert extract_numeric_answer("Result is -5") == "-5"

    def test_empty_response(self) -> None:
        assert extract_numeric_answer("no numbers here!") == ""

    def test_empty_string(self) -> None:
        assert extract_numeric_answer("") == ""


# ---------------------------------------------------------------------------
# is_correct_answer
# ---------------------------------------------------------------------------


class TestIsCorrectAnswer:
    """Tests for is_correct_answer()."""

    def test_numeric_match(self) -> None:
        assert is_correct_answer(r"\boxed{42}", "42")

    def test_numeric_float_match(self) -> None:
        assert is_correct_answer("The answer is 3.5", "3.50")

    def test_numeric_mismatch(self) -> None:
        assert not is_correct_answer("The answer is 41", "42")

    def test_comma_normalisation(self) -> None:
        assert is_correct_answer(r"\boxed{1,234}", "1234")

    def test_string_fallback_mismatch(self) -> None:
        # Neither response nor reference has a parseable number — strings differ.
        assert not is_correct_answer("word", "other")

    def test_empty_response_is_wrong(self) -> None:
        assert not is_correct_answer("no answer here!", "42")


# ---------------------------------------------------------------------------
# build_components
# ---------------------------------------------------------------------------


class TestBuildComponents:
    """Tests for build_components()."""

    def test_returns_correct_types(self) -> None:
        """Spec: SCENARIO-LEARN-050 — pipeline, library, tracker returned."""
        from carnot.models.eorm import EORMModel
        from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
        from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
        import jax.random as jr

        key = jr.PRNGKey(0)
        eorm = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)
        pipeline, library, tracker = build_components(eorm, seed=42)

        assert isinstance(pipeline, ThreeTierPipeline)
        assert isinstance(library, ConstraintTemplateLibrary)
        assert isinstance(tracker, PerModelFPTracker)

    def test_library_has_builtin_templates(self) -> None:
        from carnot.models.eorm import EORMModel
        import jax.random as jr

        key = jr.PRNGKey(1)
        eorm = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)
        _, library, _ = build_components(eorm, seed=1)
        # Should have 4 built-in templates registered.
        assert len(library._templates) >= 4


# ---------------------------------------------------------------------------
# run_live_batch
# ---------------------------------------------------------------------------


class TestRunLiveBatch:
    """Tests for run_live_batch()."""

    def test_correct_and_incorrect(self) -> None:
        """Correctness labels match is_correct_answer() for each response.

        pipeline_fn is called as list[str]->list[str]; the wrapper in run_live_batch
        calls it with a single-element list per question.
        """
        responses_map = {"Q1": "42", "Q2": "99"}

        def _mock_infer(prompts: list[str]) -> list[str]:
            return [responses_map.get(p, "0") for p in prompts]

        questions = ["Q1", "Q2"]
        references = ["42", "100"]  # Q1 correct, Q2 wrong

        responses, ground_truth = run_live_batch(questions, references, "ci-test", _mock_infer)

        assert responses == ["42", "99"]
        assert ground_truth == [True, False]

    def test_all_correct(self) -> None:
        def _mock_infer(prompts: list[str]) -> list[str]:
            return [r"The answer is 7"] * len(prompts)

        questions = ["Q"] * 3
        references = ["7"] * 3
        _, gt = run_live_batch(questions, references, "ci-test", _mock_infer)
        assert all(gt)

    def test_all_incorrect(self) -> None:
        def _mock_infer(prompts: list[str]) -> list[str]:
            return ["999"] * len(prompts)

        questions = ["Q"] * 3
        references = ["1"] * 3
        _, gt = run_live_batch(questions, references, "ci-test", _mock_infer)
        assert not any(gt)


# ---------------------------------------------------------------------------
# _load_model_pipeline
# ---------------------------------------------------------------------------


class TestLoadModelPipeline:
    """Tests for _load_model_pipeline()."""

    def test_import_error_when_transformers_missing(self) -> None:
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                _load_model_pipeline("gemma4-e4b-it")

    def test_wrapper_returns_strings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrapper converts pipeline outputs to list[str]."""
        mock_output = [[{"generated_text": "The answer is 3"}]]

        mock_pipeline_fn = MagicMock(return_value=mock_output)

        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = mock_pipeline_fn

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            fn = _load_model_pipeline("gemma4-e4b-it")

        result = fn(["What is 1+2?"])
        assert result == ["The answer is 3"]

    def test_wrapper_handles_dict_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Handles when pipeline returns a list of dicts (not list of lists)."""
        mock_output = [{"generated_text": "42"}]

        mock_pipeline_fn = MagicMock(return_value=mock_output)

        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = mock_pipeline_fn

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            fn = _load_model_pipeline("gemma4-e4b-it")

        result = fn(["Q"])
        assert result == ["42"]

    def test_wrapper_handles_other_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Handles when pipeline returns unexpected output type."""
        mock_output = ["plain string"]

        mock_pipeline_fn = MagicMock(return_value=mock_output)

        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = mock_pipeline_fn

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            fn = _load_model_pipeline("gemma4-e4b-it")

        result = fn(["Q"])
        assert result == ["plain string"]

    def test_custom_model_id_used_as_hf_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-mapped model_id is passed directly to hf_pipeline."""
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = MagicMock(return_value=[])

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            _load_model_pipeline("some/custom-model")

        call_kwargs = mock_transformers.pipeline.call_args
        assert "some/custom-model" in call_kwargs[0] or call_kwargs[1].get("model") == "some/custom-model"


# ---------------------------------------------------------------------------
# main() — integration tests via heavy mocking
# ---------------------------------------------------------------------------


class TestMain:
    """Integration tests for main() exercising all blocked/success paths."""

    def _write_output_path(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "experiment_374_self_learning_relay_live.json"

    @pytest.fixture()
    def _patch_repo_root(self, tmp_path: Path) -> None:
        """Redirect deliverable output to tmp_path."""
        pass  # Used in individual tests via patch

    def test_blocked_no_live_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes blocked artifact when CARNOT_FORCE_LIVE != 1."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.build_result.return_value = {"honest_verdict": "blocked"}

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["honest_verdict"] == "blocked"

    def test_blocked_gpu_unhealthy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes blocked artifact when GPU pre-warm fails."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.setup_gpu.return_value = {
            "all_healthy": False,
            "models": [{"name": "Gemma4-E4B-it", "health_ok": False}],
        }
        mock_tmpl.build_result.return_value = {"honest_verdict": "blocked"}

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["honest_verdict"] == "blocked"

    def test_blocked_gsm8k_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes blocked artifact when GSM8K load fails."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.setup_gpu.return_value = {"all_healthy": True, "models": []}
        mock_tmpl.build_result.return_value = {"honest_verdict": "blocked"}

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            with patch.object(_mod, "load_eorm_model", return_value=(MagicMock(), "synthetic_fallback")):
                with patch.object(_mod, "load_gsm8k_questions", side_effect=ImportError("datasets")):
                    _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()

    def test_blocked_model_load_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes blocked artifact when model pipeline load fails."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.setup_gpu.return_value = {"all_healthy": True, "models": []}
        mock_tmpl.build_result.return_value = {"honest_verdict": "blocked"}

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            with patch.object(_mod, "load_eorm_model", return_value=(MagicMock(), "synthetic_fallback")):
                with patch.object(_mod, "load_gsm8k_questions", return_value=(["Q"] * 100, ["1"] * 100)):
                    with patch.object(_mod, "_load_model_pipeline", side_effect=ImportError("transformers")):
                        _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()

    def test_success_learning_confirmed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes learning_confirmed artifact on successful live run with improvement."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        from carnot.models.eorm import EORMModel
        import jax.random as jr

        key = jr.PRNGKey(42)
        eorm = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)

        # Ground truth engineered for improvement: batch 0 lower, batch 3 higher.
        # 100 questions across 4 batches of 25.
        # Batch 0: 10/25 correct; Batch 3: 20/25 correct → improved=True.
        gt_all: list[bool] = (
            [True] * 10 + [False] * 15  # batch 0: 0.40
            + [True] * 13 + [False] * 12  # batch 1: 0.52
            + [True] * 16 + [False] * 9   # batch 2: 0.64
            + [True] * 20 + [False] * 5   # batch 3: 0.80
        )
        assert len(gt_all) == 100

        # refs: "1" for correct positions, "99" for wrong positions.
        # pipeline_fn is called as list[str]->list[str] (single-element list per question).
        questions_all = [f"Q{i}" for i in range(100)]
        answers_all = [
            "1" if gt_all[i] else "99"
            for i in range(100)
        ]

        def _mock_infer(prompts: list[str]) -> list[str]:
            # Return "1" for every prompt; is_correct_answer("1", ref) depends on ref.
            return ["1"] * len(prompts)

        # Mock ExperimentTemplate to avoid disk I/O
        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.setup_gpu.return_value = {"all_healthy": True, "models": []}

        captured: dict[str, Any] = {}

        def _capture_build_result(payload: dict, status: str = "success") -> dict:
            captured.update(payload)
            captured["status"] = status
            return dict(captured)

        mock_tmpl.build_result.side_effect = _capture_build_result

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            with patch.object(_mod, "load_eorm_model", return_value=(eorm, "synthetic_fallback")):
                with patch.object(_mod, "load_gsm8k_questions", return_value=(questions_all, answers_all)):
                    with patch.object(_mod, "_load_model_pipeline", return_value=_mock_infer):
                        _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()
        data = json.loads(out.read_text())

        # Schema upgraded to v2.
        assert data["schema"] == "carnot.self_learning_relay.v2"
        assert data["inference_mode"] == "live_gpu"
        assert data["eorm_source"] == "synthetic_fallback"
        assert len(data["learning_trajectory"]) == 4

        # Verify trajectory structure.
        for entry in data["learning_trajectory"]:
            assert "batch_id" in entry
            assert entry["n_questions"] == 25
            assert "accuracy" in entry
            assert entry["n_tier1_updates"] == 25
            assert "n_tier2_templates_active" in entry
            assert "tier3_gate_auc" in entry
            assert "cumulative_accuracy" in entry

        # Improvement detected.
        assert data["batch1_accuracy"] == pytest.approx(0.40, abs=0.01)
        assert data["batch4_accuracy"] == pytest.approx(0.80, abs=0.01)
        assert data["improved"] is True
        assert data["honest_verdict"] == "learning_confirmed"

    def test_success_no_improvement(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Artifact has honest_verdict=no_improvement when accuracy does not rise."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        from carnot.models.eorm import EORMModel
        import jax.random as jr

        key = jr.PRNGKey(7)
        eorm = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64, key=key)

        # Batch 0: 20/25 correct; Batch 3: 10/25 correct → improved=False.
        gt_all: list[bool] = (
            [True] * 20 + [False] * 5   # batch 0: 0.80
            + [True] * 15 + [False] * 10  # batch 1: 0.60
            + [True] * 12 + [False] * 13  # batch 2: 0.48
            + [True] * 10 + [False] * 15  # batch 3: 0.40
        )
        assert len(gt_all) == 100

        questions_all = [f"Q{i}" for i in range(100)]
        answers_all = ["1" if gt_all[i] else "99" for i in range(100)]

        mock_tmpl = MagicMock()
        mock_tmpl._repo_root = tmp_path
        mock_tmpl.setup_gpu.return_value = {"all_healthy": True, "models": []}

        def _capture_build_result(payload: dict, status: str = "success") -> dict:
            return dict(payload)

        mock_tmpl.build_result.side_effect = _capture_build_result

        # pipeline_fn called as list[str]->list[str] (single-element list per question).
        def _mock_infer(prompts: list[str]) -> list[str]:
            return ["1"] * len(prompts)

        with patch.object(_mod, "ExperimentTemplate", return_value=mock_tmpl):
            with patch.object(_mod, "load_eorm_model", return_value=(eorm, "exp359_real")):
                with patch.object(_mod, "load_gsm8k_questions", return_value=(questions_all, answers_all)):
                    with patch.object(_mod, "_load_model_pipeline", return_value=_mock_infer):
                        _mod.main()

        out = tmp_path / "results" / "experiment_374_self_learning_relay_live.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["honest_verdict"] == "no_improvement"
        assert data["improved"] is False
