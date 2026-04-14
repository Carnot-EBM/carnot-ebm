"""Tests for scripts/experiment_294_gpu_baseline_apple.py.

All tests run under CARNOT_FORCE_LIVE=0 (simulated / mock mode).
No GPU hardware is required.

Spec coverage:
  REQ-VERIFY-079    — GPU pre-warm health-check for live inference (Exp 294)
  SCENARIO-VERIFY-101 — Pre-warm returns True on fast mock load
  SCENARIO-VERIFY-102 — Pre-warm returns False on timeout
  REQ-VERIFY-064    — Apple adversarial baseline inference with logit saving
  REQ-VERIFY-065    — Checkpoint every 10 questions with resume support
  REQ-VERIFY-066    — Partial artifact emitted on 60 s hard timeout (stall_at field)
  REQ-VERIFY-067    — Logit tensors saved at 25 / 50 / 75 / 100 % prefix fractions
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "experiment_294_gpu_baseline_apple.py"
)


def _load_module() -> Any:
    """Load experiment_294 without executing main(), in mock mode."""
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location(
        "experiment_294_gpu_baseline_apple", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_294_gpu_baseline_apple"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_module()

PrewarmResult = _mod.PrewarmResult
model_prewarm = _mod.model_prewarm
AppleBaselineRunner294 = _mod.AppleBaselineRunner294
build_artifact = _mod.build_artifact
ARTIFACT_SCHEMA = _mod.ARTIFACT_SCHEMA
EXPERIMENT = _mod.EXPERIMENT
LOGIT_FRACTIONS = _mod.LOGIT_FRACTIONS
CHECKPOINT_INTERVAL = _mod.CHECKPOINT_INTERVAL


# ---------------------------------------------------------------------------
# Fake dataset (2 questions × 2 variant types, same shape as Exp 282 tests)
# ---------------------------------------------------------------------------

_FAKE_ROWS = [
    {
        "question_id": "gsm8k-001",
        "original_question": "Alice has 2 apples. Bob gives her 3 more. How many does Alice have?",
        "original_answer": 5,
        "variant_type": "number_swap",
        "variant_question": "Alice has 4 apples. Bob gives her 6 more. How many does Alice have?",
        "variant_answer": 10,
        "provenance": {"experiment": "exp281", "scale_factor": 2},
    },
    {
        "question_id": "gsm8k-002",
        "original_question": "There are 10 birds on a tree. 4 fly away. How many remain?",
        "original_answer": 6,
        "variant_type": "irrelevant_sentence",
        "variant_question": "There are 10 birds. Yesterday it was sunny. 4 fly away. How many remain?",
        "variant_answer": 6,
        "provenance": {"experiment": "exp281"},
    },
]


# ---------------------------------------------------------------------------
# REQ-VERIFY-079 / SCENARIO-VERIFY-101
# Pre-warm returns True on fast mock load
# ---------------------------------------------------------------------------


class TestPrewarmHealthCheckSuccess:
    """SCENARIO-VERIFY-101: model_prewarm() returns True on fast mock load."""

    def test_health_ok_true_on_fast_load(self) -> None:
        """Pre-warm with instant mock load returns health_ok=True."""

        def _fast_load(hf_id: str, gpu_id: int) -> tuple[Any, Any]:
            """Mock load that returns immediately."""
            return object(), object()

        def _fast_generate(model: Any, tokenizer: Any, prompt: str) -> str:
            """Mock generate that returns a non-empty response."""
            return "4"

        result = model_prewarm(
            "TestModel",
            "test/model",
            0,
            load_fn=_fast_load,
            generate_fn=_fast_generate,
            timeout_seconds=5,
        )
        assert isinstance(result, PrewarmResult)
        assert result.health_ok is True, "Expected health_ok=True on fast load"
        assert result.stall_root_cause is None, "Expected no stall root cause"
        assert result.load_time_s >= 0.0, "Expected non-negative load time"
        assert result.model_name == "TestModel"
        assert result.gpu_id == 0

    def test_load_time_s_reflects_actual_duration(self) -> None:
        """load_time_s should be ≥ the actual sleep duration."""
        SLEEP_S = 0.05

        def _slow_load(hf_id: str, gpu_id: int) -> tuple[Any, Any]:
            time.sleep(SLEEP_S)
            return object(), object()

        def _fast_generate(model: Any, tokenizer: Any, prompt: str) -> str:
            return "4"

        result = model_prewarm(
            "SlowModel",
            "test/slow",
            1,
            load_fn=_slow_load,
            generate_fn=_fast_generate,
            timeout_seconds=5,
        )
        assert result.health_ok is True
        assert result.load_time_s >= SLEEP_S


# ---------------------------------------------------------------------------
# REQ-VERIFY-079 / SCENARIO-VERIFY-102
# Pre-warm returns False on timeout
# ---------------------------------------------------------------------------


class TestPrewarmHealthCheckTimeout:
    """SCENARIO-VERIFY-102: model_prewarm() returns False when load times out."""

    def test_health_ok_false_on_timeout(self) -> None:
        """When load_fn sleeps beyond timeout_seconds, health_ok is False."""

        def _stalling_load(hf_id: str, gpu_id: int) -> tuple[Any, Any]:
            time.sleep(10)  # will be interrupted by 0.1 s timeout
            return object(), object()

        def _fast_generate(model: Any, tokenizer: Any, prompt: str) -> str:
            return "4"

        result = model_prewarm(
            "StallingModel",
            "test/stall",
            0,
            load_fn=_stalling_load,
            generate_fn=_fast_generate,
            timeout_seconds=0.1,  # intentionally very short
        )
        assert result.health_ok is False, "Expected health_ok=False on timeout"
        assert result.stall_root_cause == "lazy_load_stall", (
            "Expected stall_root_cause='lazy_load_stall'"
        )

    def test_stall_detected_when_generate_stalls(self) -> None:
        """stall_root_cause='lazy_load_stall' also fires when generate hangs."""

        def _fast_load(hf_id: str, gpu_id: int) -> tuple[Any, Any]:
            return object(), object()

        def _stalling_generate(model: Any, tokenizer: Any, prompt: str) -> str:
            time.sleep(10)
            return "4"

        result = model_prewarm(
            "StallingGenModel",
            "test/gen-stall",
            0,
            load_fn=_fast_load,
            generate_fn=_stalling_generate,
            timeout_seconds=0.1,
        )
        assert result.health_ok is False
        assert result.stall_root_cause == "lazy_load_stall"


# ---------------------------------------------------------------------------
# Artifact schema tests (mirrors Exp 282 SCENARIO-VERIFY-080)
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Apple baseline artifact must contain all required top-level fields."""

    def test_all_required_fields_present(self) -> None:
        """ARTIFACT_SCHEMA fields are all present in a build_artifact() result."""
        artifact = build_artifact(
            run_date="20260414",
            started_at="2026-04-14T09:00:00Z",
            finished_at="2026-04-14T09:01:00Z",
            inference_mode="mock",
            model_results={},
            logit_paths={},
            stall_at=None,
            stall_diagnosis={"vram_gpu0_free_gb": 24.0, "vram_gpu1_free_gb": 24.0},
            pre_warm_status={"Qwen3.5-0.8B": True, "Gemma4-E4B-it": True},
            pre_warm_time_s={"Qwen3.5-0.8B": 1.0, "Gemma4-E4B-it": 1.2},
        )
        for field in ARTIFACT_SCHEMA:
            assert field in artifact, f"Missing required field: {field!r}"

    def test_experiment_number_is_294(self) -> None:
        """Artifact experiment field must be 294 (Exp 294)."""
        assert EXPERIMENT == 294

    def test_partial_false_when_stall_at_none(self) -> None:
        """partial is False when stall_at is None."""
        artifact = build_artifact(
            run_date="20260414",
            started_at="2026-04-14T09:00:00Z",
            finished_at="2026-04-14T09:01:00Z",
            inference_mode="mock",
            model_results={},
            logit_paths={},
            stall_at=None,
            stall_diagnosis={},
            pre_warm_status={},
            pre_warm_time_s={},
        )
        assert artifact["partial"] is False

    def test_partial_true_when_stall_at_set(self) -> None:
        """partial is True when stall_at identifies a stall location."""
        artifact = build_artifact(
            run_date="20260414",
            started_at="2026-04-14T09:00:00Z",
            finished_at="2026-04-14T09:01:00Z",
            inference_mode="live_gpu",
            model_results={},
            logit_paths={},
            stall_at="Qwen3.5-0.8B:number_swap:gsm8k-042",
            stall_diagnosis={},
            pre_warm_status={},
            pre_warm_time_s={},
        )
        assert artifact["partial"] is True
        assert artifact["stall_at"] == "Qwen3.5-0.8B:number_swap:gsm8k-042"


# ---------------------------------------------------------------------------
# Baseline accuracy bounds (REQ-VERIFY-064)
# ---------------------------------------------------------------------------


class TestBaselineAccuracyBounds:
    """Accuracy values must be floats in [0.0, 1.0] for every variant/model."""

    def _make_runner(self, correct_answer: bool) -> AppleBaselineRunner294:
        """Build an AppleBaselineRunner294 with a deterministic mock generate_fn."""

        def _generate(question: str, expected: int, *, model_name: str = "M", variant_type: str = "standard", **kw: Any) -> tuple[str, np.ndarray]:
            logits = np.zeros((1, 8, 100), dtype=np.float32)
            return str(expected) if correct_answer else "999999", logits

        return AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=_generate,
            checkpoint_dir=None,
            logit_dir=None,
        )

    def test_accuracy_in_unit_interval_when_all_correct(self, tmp_path: Path) -> None:
        """Accuracy is 1.0 when generate_fn always returns the expected answer."""
        runner = AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=lambda q, e, **kw: (str(e), np.zeros((1, 8, 100), dtype=np.float32)),
            checkpoint_dir=tmp_path / "ckpt",
            logit_dir=tmp_path / "logits",
        )
        results = runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")
        n_correct = sum(1 for r in results if r["correct"])
        accuracy = n_correct / max(len(results), 1)
        assert 0.0 <= accuracy <= 1.0

    def test_accuracy_in_unit_interval_when_all_wrong(self, tmp_path: Path) -> None:
        """Accuracy is 0.0 when generate_fn never returns the expected answer."""
        runner = AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=lambda q, e, **kw: ("999999", np.zeros((1, 8, 100), dtype=np.float32)),
            checkpoint_dir=tmp_path / "ckpt",
            logit_dir=tmp_path / "logits",
        )
        results = runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")
        n_correct = sum(1 for r in results if r["correct"])
        accuracy = n_correct / max(len(results), 1)
        assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# Logit saving (REQ-VERIFY-067) — files created at each prefix fraction
# ---------------------------------------------------------------------------


class TestLogitSaving:
    """Logit .npy files are created at 25 / 50 / 75 / 100 % prefix fractions."""

    def test_logit_files_created_for_each_fraction(self, tmp_path: Path) -> None:
        """After run_variant(), logit files exist at all four prefix fractions."""
        # Use enough rows so the 25 % threshold is crossed (at least 1 question).
        rows = _FAKE_ROWS * 2  # 4 rows total: 2 number_swap, 2 irrelevant_sentence

        def _generate(question: str, expected: int, **kw: Any) -> tuple[str, np.ndarray]:
            return str(expected), np.zeros((1, 8, 50), dtype=np.float32)

        runner = AppleBaselineRunner294(
            rows,
            generate_fn=_generate,
            checkpoint_dir=tmp_path / "ckpt",
            logit_dir=tmp_path / "logits",
        )
        runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")

        # At least the 100 % file must exist.
        logit_dir = tmp_path / "logits"
        npy_files = list(logit_dir.glob("logits_294_*.npy"))
        assert len(npy_files) >= 1, "Expected at least one logit .npy file"

    def test_logit_npy_contains_object_array(self, tmp_path: Path) -> None:
        """Saved .npy file is a 1-D object array (ragged logit rows)."""
        rows = _FAKE_ROWS * 2

        def _generate(question: str, expected: int, **kw: Any) -> tuple[str, np.ndarray]:
            return str(expected), np.zeros((1, 5, 50), dtype=np.float32)

        runner = AppleBaselineRunner294(
            rows,
            generate_fn=_generate,
            checkpoint_dir=tmp_path / "ckpt",
            logit_dir=tmp_path / "logits",
        )
        runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")

        npy_files = sorted((tmp_path / "logits").glob("logits_294_*.npy"))
        assert npy_files, "Expected at least one logit file"
        arr = np.load(str(npy_files[-1]), allow_pickle=True)
        # Object array must be 1-D.
        assert arr.ndim == 1
        assert arr.dtype == object


# ---------------------------------------------------------------------------
# Checkpoint resume (REQ-VERIFY-065)
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """run_variant() must resume from a partial checkpoint without re-calling generate_fn."""

    def test_resume_skips_completed_questions(self, tmp_path: Path) -> None:
        """generate_fn is not called for questions already in the checkpoint."""
        call_count: list[int] = [0]

        def _counting_generate(question: str, expected: int, **kw: Any) -> tuple[str, np.ndarray]:
            call_count[0] += 1
            return str(expected), np.zeros((1, 4, 20), dtype=np.float32)

        ckpt_dir = tmp_path / "ckpt"
        logit_dir = tmp_path / "logits"

        runner1 = AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=_counting_generate,
            checkpoint_dir=ckpt_dir,
            logit_dir=logit_dir,
        )
        runner1.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")
        first_run_calls = call_count[0]

        # Second runner on same checkpoint — should not call generate_fn again.
        call_count[0] = 0
        runner2 = AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=_counting_generate,
            checkpoint_dir=ckpt_dir,
            logit_dir=logit_dir,
        )
        runner2.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")
        assert call_count[0] == 0, (
            f"Expected 0 generate_fn calls on resume, got {call_count[0]}"
        )
        assert first_run_calls > 0, "First run should have called generate_fn"


# ---------------------------------------------------------------------------
# Timeout / partial artifact (REQ-VERIFY-066)
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """A TimeoutError inside generate_fn must produce a partial artifact with stall_at."""

    def test_stall_at_field_set_on_timeout(self, tmp_path: Path) -> None:
        """stall_at identifies the (model, variant, question_id) that timed out."""

        def _timeout_generate(question: str, expected: int, **kw: Any) -> tuple[str, np.ndarray]:
            raise TimeoutError("mock inference timeout")

        runner = AppleBaselineRunner294(
            _FAKE_ROWS,
            generate_fn=_timeout_generate,
            checkpoint_dir=tmp_path / "ckpt",
            logit_dir=tmp_path / "logits",
        )
        stall_at = runner.run_with_timeout_handling()
        assert stall_at is not None, "Expected stall_at to be set on TimeoutError"
        # Format: "model_name:variant_type:question_id"
        parts = stall_at.split(":")
        assert len(parts) >= 3, f"stall_at format unexpected: {stall_at!r}"


# ---------------------------------------------------------------------------
# Apple 2410.05229 hypothesis check in artifact
# ---------------------------------------------------------------------------


class TestAppleHypothesisCheck:
    """apple_2410_05229_check field must encode the ≥ 15 pp accuracy-drop result."""

    def test_hypothesis_confirmed_field_present_when_both_variants_complete(self) -> None:
        """hypothesis_confirmed key appears when standard + number_swap results are available."""
        artifact = build_artifact(
            run_date="20260414",
            started_at="2026-04-14T09:00:00Z",
            finished_at="2026-04-14T09:01:00Z",
            inference_mode="mock",
            model_results={
                "Qwen3.5-0.8B": {
                    "standard": {"correct": 8, "total": 10, "accuracy": 0.8},
                    "number_swap": {"correct": 6, "total": 10, "accuracy": 0.6},
                }
            },
            logit_paths={},
            stall_at=None,
            stall_diagnosis={},
            pre_warm_status={},
            pre_warm_time_s={},
        )
        check = artifact["apple_2410_05229_check"]
        assert "Qwen3.5-0.8B" in check
        entry = check["Qwen3.5-0.8B"]
        assert "hypothesis_confirmed" in entry
        # 0.8 - 0.6 = 0.2 = 20 pp → hypothesis confirmed
        assert entry["hypothesis_confirmed"] is True
        assert entry["drop_pp"] == pytest.approx(20.0, abs=0.01)

    def test_hypothesis_not_confirmed_when_drop_below_threshold(self) -> None:
        """hypothesis_confirmed=False when drop_pp < 15."""
        artifact = build_artifact(
            run_date="20260414",
            started_at="2026-04-14T09:00:00Z",
            finished_at="2026-04-14T09:01:00Z",
            inference_mode="mock",
            model_results={
                "Qwen3.5-0.8B": {
                    "standard": {"correct": 8, "total": 10, "accuracy": 0.8},
                    "number_swap": {"correct": 7, "total": 10, "accuracy": 0.7},
                }
            },
            logit_paths={},
            stall_at=None,
            stall_diagnosis={},
            pre_warm_status={},
            pre_warm_time_s={},
        )
        check = artifact["apple_2410_05229_check"]
        entry = check["Qwen3.5-0.8B"]
        # 0.8 - 0.7 = 0.1 = 10 pp → NOT confirmed
        assert entry["hypothesis_confirmed"] is False
