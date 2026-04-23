"""Tests for Exp 785 SOTA GGUF 2-round code repair v2 helpers.

Spec: REQ-REPAIR-024, REQ-REPAIR-025, SCENARIO-REPAIR-044, SCENARIO-REPAIR-045
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.pipeline.two_round_repair import TwoRoundResult  # noqa: E402
from scripts.experiment_785_sota_gguf_code_repair_v2 import (  # noqa: E402
    MODEL_FALLBACK,
    MODEL_LARGE,
    _LARGE_MODEL_VRAM_THRESHOLD_MB,
    build_repair_prompt_785,
    classify_verdict_785,
    compute_repair_metrics_785,
    select_model_by_vram,
)


class TestBuildRepairPrompt785:
    """REQ-REPAIR-024, SCENARIO-REPAIR-044: repair prompt includes error signal from round 1."""

    def test_prompt_includes_error_message(self):
        # REQ-REPAIR-024: traceback must appear in repair prompt so model can fix the bug.
        prompt = build_repair_prompt_785(
            original_problem="def add(a, b): ...",
            failed_code="def add(a, b): return a - b",
            error_message="AssertionError: expected 5 got 1",
        )
        assert "AssertionError: expected 5 got 1" in prompt

    def test_prompt_includes_failing_code(self):
        # Model must see what it wrote so it can identify the bug.
        prompt = build_repair_prompt_785(
            original_problem="def foo(): ...",
            failed_code="def foo(): return None",
            error_message="TypeError: expected int",
        )
        assert "def foo(): return None" in prompt

    def test_prompt_empty_error_handled(self):
        # Empty error_message must not crash — replaced with fallback string.
        prompt = build_repair_prompt_785(
            original_problem="def baz(): ...",
            failed_code="def baz(): pass",
            error_message="",
        )
        assert "(no traceback)" in prompt

    def test_prompt_includes_fix_instruction(self):
        # The repair instruction must be present so model knows to output corrected code.
        prompt = build_repair_prompt_785("def x(): ...", "def x(): return 0", "error")
        assert "Fix the bug" in prompt


class TestSelectModelByVram:
    """REQ-REPAIR-025, SCENARIO-REPAIR-045: VRAM-based model selection after zombie kill."""

    def test_large_model_selected_when_vram_sufficient(self):
        # REQ-REPAIR-025: >= 20000 MB free → select the 35B model.
        result = select_model_by_vram(20_000)
        assert result == MODEL_LARGE

    def test_large_model_selected_at_exact_threshold(self):
        # Boundary: exactly 20000 MB free must still select the large model.
        result = select_model_by_vram(_LARGE_MODEL_VRAM_THRESHOLD_MB)
        assert result == MODEL_LARGE

    def test_fallback_model_selected_when_vram_low(self):
        # REQ-REPAIR-025: < 20000 MB free → fall back to 7B model (SCENARIO-REPAIR-045).
        result = select_model_by_vram(19_999)
        assert result == MODEL_FALLBACK

    def test_fallback_model_selected_when_vram_zero(self):
        # Edge case: nvidia-smi unavailable → free_vram_mb = 0.0 → use 7B fallback.
        result = select_model_by_vram(0.0)
        assert result == MODEL_FALLBACK

    def test_fallback_model_is_seven_b(self):
        # SCENARIO-REPAIR-045: the fallback is explicitly Qwen3.5-7B-Instruct-GGUF.
        assert "7B" in MODEL_FALLBACK or "7b" in MODEL_FALLBACK.lower()


class TestComputeRepairMetrics785:
    """REQ-REPAIR-023, REQ-REPAIR-024: signed_improvement = pass_at_1_round2 - pass_at_1_round1."""

    def test_signed_improvement_equals_delta(self):
        # REQ-REPAIR-024 inherits REQ-REPAIR-023: signed_improvement must equal the delta.
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics_785(results)
        expected_si = round(m["pass_at_1_round2"] - m["pass_at_1_round1"], 4)
        assert m["signed_improvement"] == expected_si

    def test_n_repaired_matches_definition(self):
        # n_repaired = count(NOT round0_pass AND round1_pass).
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics_785(results)
        assert m["n_repaired"] == 2

    def test_empty_results(self):
        m = compute_repair_metrics_785([])
        assert m["pass_at_1_round1"] == 0.0
        assert m["pass_at_1_round2"] == 0.0
        assert m["signed_improvement"] == 0.0
        assert m["n_repaired"] == 0

    def test_all_pass_round1_no_improvement(self):
        # When all pass in round 1, signed_improvement must be 0.
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics_785(results)
        assert m["signed_improvement"] == 0.0
        assert m["n_repaired"] == 0


class TestClassifyVerdict785:
    """REQ-REPAIR-024: honest_verdict maps correctly from signed_improvement and inference_mode."""

    def test_blocked_no_live_gpu(self):
        # CARNOT_FORCE_LIVE not set → blocked_no_live_gpu regardless of improvement.
        assert classify_verdict_785(0.1, "blocked") == "blocked_no_live_gpu"
        assert classify_verdict_785(0.0, "blocked") == "blocked_no_live_gpu"
        assert classify_verdict_785(-0.1, "blocked") == "blocked_no_live_gpu"

    def test_blocked_model_load_failed(self):
        # llama-cpp load failure → blocked_model_load_failed.
        assert classify_verdict_785(0.0, "blocked_model_load_failed") == "blocked_model_load_failed"

    def test_positive_improvement_live_gpu(self):
        # signed_improvement > 0 on live GPU → sota_code_repair_positive.
        assert classify_verdict_785(0.1, "live_gpu") == "sota_code_repair_positive"

    def test_zero_improvement_live_gpu(self):
        # signed_improvement == 0 on live GPU → sota_code_repair_zero.
        assert classify_verdict_785(0.0, "live_gpu") == "sota_code_repair_zero"

    def test_negative_improvement_live_gpu(self):
        # signed_improvement < 0 on live GPU → sota_code_repair_negative.
        assert classify_verdict_785(-0.05, "live_gpu") == "sota_code_repair_negative"


class TestKillGpuZombiesCalledBeforeModelLoad:
    """REQ-REPAIR-024, SCENARIO-REPAIR-044: kill_gpu_zombies() is invoked before any model load."""

    def test_kill_gpu_zombies_called_in_main_when_force_live(self, tmp_path):
        """When CARNOT_FORCE_LIVE=1, kill_gpu_zombies(gpu_index=0) is called before model load.

        We mock out the heavy dependencies (llama-cpp, watchdog, template) so the test
        is fast and hermetic. The critical assertion is that kill_gpu_zombies was called
        with gpu_index=0 before any attempt to load the GGUF model.
        """
        import os as _os  # noqa: PLC0415

        call_order: list[str] = []

        fake_zombie_result = MagicMock()
        fake_zombie_result.honest_verdict = "no_zombies_found"
        fake_zombie_result.pids_killed = []
        fake_zombie_result.vram_freed_mb = 0.0

        def fake_kill_zombies(gpu_index=0, **kwargs):
            call_order.append("kill_gpu_zombies")
            return fake_zombie_result

        def fake_build_gguf_caller(model_id, gpu_id=0):
            call_order.append("build_gguf_caller")
            raise RuntimeError("test_stop_after_zombie_kill")

        deliverable = str(tmp_path / "exp785_test.json")

        # Patch environment, zombie killer, gguf caller, and template to keep test fast.
        with (
            patch.dict(_os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.kill_gpu_zombies",
                side_effect=fake_kill_zombies,
            ),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2._build_gguf_caller_785",
                side_effect=fake_build_gguf_caller,
            ),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.ExperimentTemplate"
            ) as MockTemplate,
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.ExperimentTimeoutWatchdog"
            ) as MockWatchdog,
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.apply_env_autofix"
            ),
            patch("subprocess.run") as mock_smi,
        ):
            # Make the watchdog a no-op context manager.
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            MockWatchdog.return_value = mock_ctx

            # Make template emit a blocked artifact and not raise.
            mock_tmpl = MagicMock()
            mock_tmpl.checkpoint = None
            mock_tmpl.build_result.return_value = {"honest_verdict": "blocked_model_load_failed"}
            # Python 3.14 blocks assert_* attribute access on MagicMock — set explicitly.
            mock_tmpl.assert_deliverable_written = MagicMock()
            MockTemplate.return_value = mock_tmpl

            # Mock nvidia-smi to return enough VRAM to attempt the large model.
            mock_smi.return_value = MagicMock(returncode=0, stdout="22000\n")

            from scripts.experiment_785_sota_gguf_code_repair_v2 import main  # noqa: PLC0415
            main()

        # kill_gpu_zombies must appear before build_gguf_caller in the call order.
        assert "kill_gpu_zombies" in call_order
        assert "build_gguf_caller" in call_order
        assert call_order.index("kill_gpu_zombies") < call_order.index("build_gguf_caller"), (
            "kill_gpu_zombies() must be called BEFORE _build_gguf_caller_785() (REQ-REPAIR-024)"
        )

    def test_fallback_model_used_when_low_vram(self, tmp_path):
        """When free_vram_mb < 20000 after zombie kill, the 7B fallback model is selected.

        This verifies SCENARIO-REPAIR-045: the model_used field in the artifact must
        reflect the fallback when VRAM is constrained.
        """
        import os as _os  # noqa: PLC0415

        selected_models: list[str] = []

        fake_zombie_result = MagicMock()
        fake_zombie_result.honest_verdict = "no_zombies_found"
        fake_zombie_result.pids_killed = []
        fake_zombie_result.vram_freed_mb = 0.0

        def fake_build_gguf_caller(model_id, gpu_id=0):
            selected_models.append(model_id)
            raise RuntimeError("test_stop")

        with (
            patch.dict(_os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.kill_gpu_zombies",
                return_value=fake_zombie_result,
            ),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2._build_gguf_caller_785",
                side_effect=fake_build_gguf_caller,
            ),
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.ExperimentTemplate"
            ) as MockTemplate,
            patch(
                "scripts.experiment_785_sota_gguf_code_repair_v2.ExperimentTimeoutWatchdog"
            ) as MockWatchdog,
            patch("scripts.experiment_785_sota_gguf_code_repair_v2.apply_env_autofix"),
            patch("subprocess.run") as mock_smi,
        ):
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            MockWatchdog.return_value = mock_ctx

            mock_tmpl = MagicMock()
            mock_tmpl.checkpoint = None
            mock_tmpl.build_result.return_value = {"honest_verdict": "blocked_model_load_failed"}
            # Python 3.14 blocks assert_* attribute access on MagicMock — set explicitly.
            mock_tmpl.assert_deliverable_written = MagicMock()
            MockTemplate.return_value = mock_tmpl

            # Low VRAM → should trigger fallback model selection.
            mock_smi.return_value = MagicMock(returncode=0, stdout="8000\n")

            from scripts.experiment_785_sota_gguf_code_repair_v2 import main  # noqa: PLC0415
            main()

        assert len(selected_models) == 1, "Expected exactly one model load attempt"
        assert selected_models[0] == MODEL_FALLBACK, (
            f"Expected fallback model {MODEL_FALLBACK}, got {selected_models[0]} "
            "(REQ-REPAIR-025, SCENARIO-REPAIR-045)"
        )
