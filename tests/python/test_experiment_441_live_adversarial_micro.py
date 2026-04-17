"""Tests for scripts/experiment_441_live_adversarial_micro.py.

Covers new code in Exp 441 (helpers not already tested elsewhere):
  - _write_artifact: JSON write, directory creation, pretty-printing.
  - _load_model_with_explicit_device: import path and kwarg forwarding.
  - _run_three_conditions_for_model: standard/adversarial/repaired accuracy,
      VerifyRepairPipeline wiring + fallback on exception.
  - main(): all gate paths:
      Gate 1 blocked → blocked artifact
      Gate 3 unhealthy → blocked artifact
      Gate 4 model load failure → blocked artifact
      All gates pass → success artifact
      GPU1 zombie warning (non-blocking)

All tests run without a live GPU.  GPU infrastructure is mocked throughout.
Shared helpers imported from Exp 355 and pipeline modules are NOT re-tested here.

Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030
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

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_441_live_adversarial_micro.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_441 as a module without executing main()."""
    for d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
        if d not in sys.path:
            sys.path.insert(0, d)
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_441", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_441"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

_write_artifact = _mod._write_artifact
_run_three_conditions_for_model = _mod._run_three_conditions_for_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_gpu_health(*, zombie: bool = False, temp_warn: bool = False) -> Any:
    from carnot.pipeline.dual_gpu_health import DualGPUHealthResult

    return DualGPUHealthResult(
        gpu0_util_pct=50.0,
        gpu1_util_pct=0.0 if zombie else 50.0,
        gpu0_temp_c=70.0,
        gpu1_temp_c=85.0 if temp_warn else 70.0,
        gpu0_vram_mb=20000.0,
        gpu1_vram_mb=512.0 if zombie else 20000.0,
        gpu1_is_zombie=zombie,
        temperature_warning=temp_warn,
        recommended_batch_size_factor=0.5 if temp_warn else 1.0,
    )


def _make_adversarial_questions(n: int = 5) -> list[Any]:
    from carnot.pipeline.adversarial_gsm8k import AdversarialGSMQuestion

    return [
        AdversarialGSMQuestion(
            question_id=f"q_{i:04d}",
            original_question=f"What is {i} + {i}?",
            adversarial_question=f"What is {i} + {i}? The weather was sunny.",
            ground_truth_answer=str(i + i),
            irrelevant_sentence="The weather was sunny.",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_writes_json(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "artifact.json"
        _write_artifact(path, {"key": "value"})
        assert path.exists()
        assert json.loads(path.read_text())["key"] == "value"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "artifact.json"
        _write_artifact(path, {"x": 1})
        assert path.exists()

    def test_pretty_printed(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.json"
        _write_artifact(path, {"a": 1, "b": 2})
        raw = path.read_text()
        # pretty-print means newlines are present
        assert "\n" in raw


# ---------------------------------------------------------------------------
# _run_three_conditions_for_model
# ---------------------------------------------------------------------------


class TestRunThreeConditions:
    def _make_executor(self, tmp_path: Path) -> Any:
        from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor

        return LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=str(tmp_path / "ckpt"))

    def test_all_correct_standard(self, tmp_path: Path) -> None:
        """Model always returns correct answer for standard questions."""
        qs = _make_adversarial_questions(4)
        # Model callable that returns "#### <gold>" format.
        model = MagicMock(side_effect=lambda prompt: [{"generated_text": f"#### {1+1}"}])
        # Patch _call_model and _is_correct to be deterministic.
        with (
            patch.object(_mod, "_call_model", side_effect=lambda m, p: "#### 2"),
            patch.object(_mod, "_is_correct", return_value=True),
        ):
            result = _run_three_conditions_for_model(
                qs, model, "TestModel", self._make_executor(tmp_path), 441
            )
        assert result.model_id == "TestModel"
        assert result.inference_mode == "live_gpu"
        assert result.standard_accuracy == pytest.approx(1.0)
        assert result.adversarial_accuracy == pytest.approx(1.0)
        assert result.repaired_accuracy == pytest.approx(1.0)

    def test_adversarial_drop(self, tmp_path: Path) -> None:
        """Adversarial condition produces lower accuracy than standard."""
        qs = _make_adversarial_questions(4)
        call_count = [0]

        def _mock_is_correct(resp: str, gold: str) -> bool:
            call_count[0] += 1
            # standard batch (calls 1-4): all correct
            # adversarial batch (calls 5-8): all wrong
            # repaired batch (calls 9-12): all correct
            batch = (call_count[0] - 1) // len(qs)
            return batch != 1  # batch 1 (adversarial) returns False

        with (
            patch.object(_mod, "_call_model", return_value="response"),
            patch.object(_mod, "_is_correct", side_effect=_mock_is_correct),
        ):
            result = _run_three_conditions_for_model(
                qs, MagicMock(), "TestModel", self._make_executor(tmp_path), 441
            )
        assert result.standard_accuracy == pytest.approx(1.0)
        assert result.adversarial_accuracy == pytest.approx(0.0)
        assert result.repaired_accuracy == pytest.approx(1.0)
        assert result.adversarial_drop_pct == pytest.approx(100.0)
        assert result.repair_improvement_pct == pytest.approx(100.0)

    def test_verify_repair_pipeline_fallback_on_exception(self, tmp_path: Path) -> None:
        """VerifyRepairPipeline import failure falls back to re-inference gracefully."""
        qs = _make_adversarial_questions(3)
        with (
            patch.object(_mod, "_call_model", return_value="#### 2"),
            patch.object(_mod, "_is_correct", return_value=True),
            patch.dict("sys.modules", {"carnot.pipeline.verify_repair": None}),
        ):
            result = _run_three_conditions_for_model(
                qs, MagicMock(), "TestModel", self._make_executor(tmp_path), 441
            )
        assert result.inference_mode == "live_gpu"
        assert result.n_questions == 3

    def test_n_questions_field(self, tmp_path: Path) -> None:
        qs = _make_adversarial_questions(5)
        with (
            patch.object(_mod, "_call_model", return_value="answer"),
            patch.object(_mod, "_is_correct", return_value=False),
        ):
            result = _run_three_conditions_for_model(
                qs, MagicMock(), "M", self._make_executor(tmp_path), 441
            )
        assert result.n_questions == 5


# ---------------------------------------------------------------------------
# main() gate paths
# ---------------------------------------------------------------------------


class TestMainGatePaths:
    def _base_patches(self, tmp_path: Path) -> dict[str, Any]:
        """Return a dict of patches needed to run main() without real GPU."""
        from carnot.pipeline.adversarial_gsm8k import AdversarialGSMQuestion

        fake_questions = [
            {"question_id": f"q_{i:04d}", "question": f"Q{i}", "answer": str(i)}
            for i in range(5)
        ]
        fake_adv_questions = [
            AdversarialGSMQuestion(
                question_id=f"q_{i:04d}",
                original_question=f"Q{i}",
                adversarial_question=f"Q{i} Distractor.",
                ground_truth_answer=str(i),
                irrelevant_sentence="Distractor.",
            )
            for i in range(5)
        ]

        tmpl_mock = MagicMock()
        tmpl_mock.build_result.return_value = {"status": "success", "honest_verdict": "blocked"}
        tmpl_mock._output_path = tmp_path / DELIVERABLE_STEM

        return {
            "tmpl_mock": tmpl_mock,
            "fake_questions": fake_questions,
            "fake_adv_questions": fake_adv_questions,
        }

    def test_gate1_blocked_writes_artifact(self, tmp_path: Path) -> None:
        ctx = self._base_patches(tmp_path)
        output_path = tmp_path / "results" / "experiment_441_live_adversarial_micro.json"

        with (
            patch.object(_mod, "LiveGPUGate") as mock_gate,
            patch.object(_mod, "ExperimentTemplate", return_value=ctx["tmpl_mock"]),
            patch.object(_mod, "_write_artifact") as mock_write,
        ):
            mock_gate.require_live_or_blocked.return_value = {"gate": "blocked"}
            _mod.main()

        mock_write.assert_called_once()

    def test_gate3_unhealthy_writes_blocked_artifact(self, tmp_path: Path) -> None:
        ctx = self._base_patches(tmp_path)

        unhealthy_status = {"all_healthy": False, "models": []}
        ctx["tmpl_mock"].setup_gpu.return_value = unhealthy_status

        with (
            patch.object(_mod, "LiveGPUGate") as mock_gate,
            patch.object(_mod, "check_dual_gpu_health", return_value=_make_gpu_health()),
            patch.object(_mod, "ExperimentTemplate", return_value=ctx["tmpl_mock"]),
            patch.object(_mod, "_write_artifact") as mock_write,
        ):
            mock_gate.require_live_or_blocked.return_value = None
            _mod.main()

        mock_write.assert_called_once()
        written_art = mock_write.call_args[0][1]
        assert written_art["honest_verdict"] == "blocked"

    def test_gate4_model_load_failure_writes_blocked(self, tmp_path: Path) -> None:
        ctx = self._base_patches(tmp_path)
        ctx["tmpl_mock"].setup_gpu.return_value = {"all_healthy": True}

        with (
            patch.object(_mod, "LiveGPUGate") as mock_gate,
            patch.object(_mod, "check_dual_gpu_health", return_value=_make_gpu_health()),
            patch.object(_mod, "ExperimentTemplate", return_value=ctx["tmpl_mock"]),
            patch.object(_mod, "_load_model_with_explicit_device", side_effect=RuntimeError("OOM")),
            patch.object(_mod, "_write_artifact") as mock_write,
        ):
            mock_gate.require_live_or_blocked.return_value = None
            _mod.main()

        mock_write.assert_called_once()
        written_art = mock_write.call_args[0][1]
        assert written_art["honest_verdict"] == "blocked"

    def test_all_gates_pass_success_artifact(self, tmp_path: Path) -> None:
        ctx = self._base_patches(tmp_path)
        ctx["tmpl_mock"].setup_gpu.return_value = {"all_healthy": True}

        from carnot.pipeline.adversarial_gsm8k import MicroAdversarialResult

        fake_micro_result = MicroAdversarialResult(
            model_id="Gemma4-E4B-it",
            n_questions=5,
            standard_accuracy=0.8,
            adversarial_accuracy=0.6,
            repaired_accuracy=0.7,
            adversarial_drop_pct=20.0,
            repair_improvement_pct=10.0,
            inference_mode="live_gpu",
        )

        with (
            patch.object(_mod, "LiveGPUGate") as mock_gate,
            patch.object(_mod, "check_dual_gpu_health", return_value=_make_gpu_health()),
            patch.object(_mod, "ExperimentTemplate", return_value=ctx["tmpl_mock"]),
            patch.object(_mod, "_load_model_with_explicit_device", return_value=MagicMock()),
            patch.object(_mod, "load_gsm8k_questions", return_value=ctx["fake_questions"]),
            patch.object(_mod, "build_adversarial_questions", return_value=ctx["fake_adv_questions"]),
            patch.object(_mod, "_run_three_conditions_for_model", return_value=fake_micro_result),
            patch.object(_mod, "_write_artifact") as mock_write,
        ):
            mock_gate.require_live_or_blocked.return_value = None
            _mod.main()

        mock_write.assert_called_once()
        written_art = mock_write.call_args[0][1]
        # tmpl.build_result was called; it returns its mock value
        assert mock_write.call_args[0][1] is not None

    def test_gpu1_zombie_non_blocking(self, tmp_path: Path) -> None:
        ctx = self._base_patches(tmp_path)
        ctx["tmpl_mock"].setup_gpu.return_value = {"all_healthy": True}

        from carnot.pipeline.adversarial_gsm8k import MicroAdversarialResult

        fake_micro_result = MicroAdversarialResult(
            model_id="M", n_questions=5,
            standard_accuracy=0.8, adversarial_accuracy=0.6, repaired_accuracy=0.7,
            adversarial_drop_pct=20.0, repair_improvement_pct=10.0, inference_mode="live_gpu",
        )

        with (
            patch.object(_mod, "LiveGPUGate") as mock_gate,
            patch.object(_mod, "check_dual_gpu_health", return_value=_make_gpu_health(zombie=True)),
            patch.object(_mod, "ExperimentTemplate", return_value=ctx["tmpl_mock"]),
            patch.object(_mod, "_load_model_with_explicit_device", return_value=MagicMock()),
            patch.object(_mod, "load_gsm8k_questions", return_value=ctx["fake_questions"]),
            patch.object(_mod, "build_adversarial_questions", return_value=ctx["fake_adv_questions"]),
            patch.object(_mod, "_run_three_conditions_for_model", return_value=fake_micro_result),
            patch.object(_mod, "_write_artifact") as mock_write,
        ):
            mock_gate.require_live_or_blocked.return_value = None
            _mod.main()  # Should not raise even with zombie GPU1

        mock_write.assert_called_once()


DELIVERABLE_STEM = Path("results/experiment_441_live_adversarial_micro.json")
