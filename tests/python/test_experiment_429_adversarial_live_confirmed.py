"""Tests for scripts/experiment_429_adversarial_live_confirmed.py.

Covers 100% of new code in Exp 429:
  - _load_preflight_verdict: happy path, missing file, malformed JSON.
  - _write_artifact: JSON write, directory creation, pretty-printing.
  - _build_exp429_artifact: all verdict paths, pct field computation, gate fields.
  - _run_three_conditions: standard/adversarial/repaired correctness lists,
      checkpointing cadence, VerifyRepairPipeline fallback on exception.
  - main(): all gate paths:
      Exp 421 confirm path (status='success', inference_mode='live_gpu')
      Gate 1 blocked → blocked artifact
      Gate 3 unhealthy → blocked artifact
      Gate 4 model load failure → blocked artifact
      All gates pass → success artifact
      GPU1 zombie warning (non-blocking)

All tests run without a live GPU.  GPU infrastructure is mocked throughout.
Shared helpers imported from Exp 355 and Exp 368 are NOT re-tested here
(already covered in their own test suites).

Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-014-019, REQ-INFRA-021/023
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
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_429_adversarial_live_confirmed.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_429 as a module without executing main()."""
    for d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
        if d not in sys.path:
            sys.path.insert(0, d)
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_429", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_429"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

_load_preflight_verdict = _mod._load_preflight_verdict
_write_artifact = _mod._write_artifact
_build_exp429_artifact = _mod._build_exp429_artifact
_run_three_conditions = _mod._run_three_conditions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_gpu_health(*, zombie: bool = False, temp_warn: bool = False) -> Any:
    """Return a DualGPUHealthResult mock."""
    from carnot.pipeline.dual_gpu_health import DualGPUHealthResult

    return DualGPUHealthResult(
        gpu0_util_pct=50.0,
        gpu1_util_pct=0.0 if zombie else 50.0,
        gpu0_temp_c=70.0,
        gpu1_temp_c=85.0 if temp_warn else 70.0,
        gpu0_vram_mb=8000.0,
        gpu1_vram_mb=600.0 if zombie else 8000.0,
        gpu1_is_zombie=zombie,
        temperature_warning=temp_warn,
        recommended_batch_size_factor=0.75 if temp_warn else 1.0,
    )


def _make_paired_questions(n: int = 5) -> list[Any]:
    """Return n minimal AdversarialGSMQuestion objects."""
    from carnot.pipeline.adversarial_gsm8k import AdversarialGSMQuestion

    return [
        AdversarialGSMQuestion(
            question_id=f"q_{i:04d}",
            original_question=f"What is {i} + {i}?  The answer is {i + i}.",
            adversarial_question=f"What is {i} + {i}?  The answer is {i + i}. Five of them were smaller.",
            ground_truth_answer=str(i + i),
            irrelevant_sentence="Five of them were smaller.",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# _load_preflight_verdict
# ---------------------------------------------------------------------------


class TestLoadPreflightVerdict:
    """Gate 0 preflight loader tests."""

    def test_loads_valid_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "exp413.json"
        f.write_text(json.dumps({"honest_verdict": "auto_fix_applied", "retro_022_resolved": True}))
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", f)
        result = _load_preflight_verdict()
        assert result["honest_verdict"] == "auto_fix_applied"
        assert result["retro_022_resolved"] is True

    def test_missing_file_returns_sentinel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", tmp_path / "nonexistent.json")
        result = _load_preflight_verdict()
        assert result["honest_verdict"] == "preflight_file_missing"
        assert result["retro_022_resolved"] is False
        assert "error" in result

    def test_malformed_json_returns_sentinel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", f)
        result = _load_preflight_verdict()
        assert result["honest_verdict"] == "preflight_file_missing"
        assert "error" in result


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    """Tests for the _write_artifact helper."""

    def test_writes_json(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out = tmp_path / "results" / "exp429.json"
        tmpl._output_path = out
        _write_artifact(tmpl, {"key": "value"})
        assert out.exists()
        assert json.loads(out.read_text())["key"] == "value"

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        tmpl._output_path = tmp_path / "a" / "b" / "result.json"
        _write_artifact(tmpl, {"x": 1})
        assert tmpl._output_path.exists()

    def test_pretty_printed(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        tmpl._output_path = tmp_path / "pretty.json"
        _write_artifact(tmpl, {"a": 1})
        assert "\n" in tmpl._output_path.read_text()


# ---------------------------------------------------------------------------
# _build_exp429_artifact
# ---------------------------------------------------------------------------


class TestBuildExp429Artifact:
    """Tests for the artifact builder."""

    def _make_per_model(self, drop: float, improvement: float) -> dict[str, Any]:
        return {
            "model_id": "TestModel",
            "n_questions": 50,
            "standard_accuracy": 0.80,
            "adversarial_accuracy": 0.80 - drop,
            "accuracy_drop": drop,
            "repaired_adversarial_accuracy": 0.80 - drop + improvement,
            "repair_improvement": improvement,
            "inference_mode": "live_gpu",
        }

    def test_schema_v2(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.05)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert art["adversarial_schema"] == "carnot.adversarial_gsm8k.v2"

    def test_improvement_positive_verdict(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.15, 0.06)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert art["honest_verdict"] == "improvement_positive"
        assert art["headline_result"]["improvement_positive"] is True

    def test_degradation_positive_verdict(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.0)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert art["honest_verdict"] == "degradation_positive"

    def test_neutral_verdict(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.0, 0.0)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert art["honest_verdict"] == "neutral"

    def test_blocked_simulated_verdict(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.05)], "blocked", 0,
            False, "auto_fix_applied", False, False,
        )
        assert art["honest_verdict"] == "blocked_simulated"

    def test_adversarial_drop_pct_is_percentage(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.15, 0.0)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert abs(art["adversarial_drop_pct"] - 15.0) < 1e-3

    def test_repair_improvement_pct_is_percentage(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.08)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        assert abs(art["repair_improvement_pct"] - 8.0) < 1e-3

    def test_gate_fields_propagated(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.05)], "live_gpu", 50,
            True, "auto_fix_applied", True, False,
        )
        assert art["gate0_autofix_applied"] is True
        assert art["gate0_preflight_verdict"] == "auto_fix_applied"
        assert art["gate2_gpu1_zombie"] is True
        assert art["gate2_temperature_warning"] is False

    def test_confirmed_from_and_rerun(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.05)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
            confirmed_from=421, rerun=True,
        )
        assert art["confirmed_from"] == 421
        assert art["rerun"] is True

    def test_empty_per_model_results(self) -> None:
        art = _build_exp429_artifact(
            [], "blocked", 0, False, "preflight_file_missing", False, False,
        )
        assert art["adversarial_drop_pct"] == 0.0
        assert art["repair_improvement_pct"] == 0.0

    def test_headline_result_keys_present(self) -> None:
        art = _build_exp429_artifact(
            [self._make_per_model(0.10, 0.05)], "live_gpu", 50,
            False, "auto_fix_applied", False, False,
        )
        hr = art["headline_result"]
        for key in [
            "honest_verdict", "inference_mode", "avg_adversarial_drop",
            "avg_repair_improvement", "adversarial_drop_pct", "repair_improvement_pct",
            "improvement_positive", "n_questions_per_model", "n_models",
        ]:
            assert key in hr, f"Missing headline key: {key}"

    def test_two_model_average(self) -> None:
        models = [self._make_per_model(0.20, 0.0), self._make_per_model(0.10, 0.10)]
        art = _build_exp429_artifact(models, "live_gpu", 50, False, "auto_fix_applied", False, False)
        # avg drop = 0.15, avg improvement = 0.05
        assert abs(art["adversarial_drop_pct"] - 15.0) < 1e-3
        assert abs(art["repair_improvement_pct"] - 5.0) < 1e-3


# ---------------------------------------------------------------------------
# _run_three_conditions
# ---------------------------------------------------------------------------


class TestRunThreeConditions:
    """Tests for the per-model inference loop."""

    def _make_tmpl(self, tmp_path: Path) -> MagicMock:
        tmpl = MagicMock()
        tmpl._output_path = tmp_path / "result.json"
        tmpl.checkpoint_save = MagicMock()
        return tmpl

    def test_returns_per_model_dict_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        questions = _make_paired_questions(5)

        def _fake_call_model(obj: Any, prompt: str) -> str:
            # answer correctly for standard, wrong for adversarial
            if "Five of them" in prompt:
                return "The answer is 999"
            num = prompt.split(" + ")[0].split()[-1]
            return f"#### {int(num) * 2}"

        monkeypatch.setattr(_mod, "_call_model", _fake_call_model)
        monkeypatch.setattr(_mod, "_is_correct", lambda r, g: (g in r))

        tmpl = self._make_tmpl(tmp_path)
        result = _run_three_conditions(questions, MagicMock(), "TestModel", tmpl, 0)

        for key in [
            "model_id", "n_questions", "standard_accuracy", "adversarial_accuracy",
            "accuracy_drop", "repaired_adversarial_accuracy", "repair_improvement",
            "inference_mode",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_inference_mode_is_live_gpu(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        questions = _make_paired_questions(3)
        monkeypatch.setattr(_mod, "_call_model", lambda *a: "#### 0")
        monkeypatch.setattr(_mod, "_is_correct", lambda *a: False)
        tmpl = self._make_tmpl(tmp_path)
        result = _run_three_conditions(questions, MagicMock(), "M", tmpl, 0)
        assert result["inference_mode"] == "live_gpu"

    def test_checkpoint_fires_at_10(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        questions = _make_paired_questions(10)
        monkeypatch.setattr(_mod, "_call_model", lambda *a: "#### 0")
        monkeypatch.setattr(_mod, "_is_correct", lambda *a: False)
        tmpl = self._make_tmpl(tmp_path)
        _run_three_conditions(questions, MagicMock(), "M", tmpl, 0)
        # checkpoint_save should be called once (at q=10)
        assert tmpl.checkpoint_save.call_count >= 1
        last_call_kwargs = tmpl.checkpoint_save.call_args_list[-1]
        assert last_call_kwargs.kwargs["step"] == 10  # model_idx=0, so step = 0*10 + 10

    def test_repair_pipeline_exception_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When repair_pipeline.verify_and_repair raises, fall back to plain re-inference."""
        questions = _make_paired_questions(3)
        monkeypatch.setattr(_mod, "_call_model", lambda *a: "#### 2")
        monkeypatch.setattr(_mod, "_is_correct", lambda *a: True)

        # Mock VerifyRepairPipeline to raise on verify_and_repair
        bad_pipeline = MagicMock()
        bad_pipeline.verify_and_repair.side_effect = RuntimeError("simulated repair failure")

        import carnot.pipeline.verify_repair as vrm
        import carnot.pipeline.extract as exm

        monkeypatch.setattr(vrm, "VerifyRepairPipeline", lambda **kw: bad_pipeline)
        monkeypatch.setattr(exm, "AutoExtractor", MagicMock)

        tmpl = self._make_tmpl(tmp_path)
        # Should NOT raise — exception must be caught and logged
        result = _run_three_conditions(questions, MagicMock(), "M", tmpl, 0)
        assert result["n_questions"] == 3


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() — all GPU interactions mocked.

    Gate paths covered:
        1. Exp 421 confirmable → copy artifact (confirmed_from=421, rerun=False).
        2. Gate 1 (LiveGPUGate) blocks → blocked artifact.
        3. Gate 3 (setup_gpu) unhealthy → blocked artifact.
        4. Gate 4 (model load) fails → blocked artifact.
        5. All gates pass → success artifact with schema v2.
        6. GPU1 zombie (Gate 2 warning, non-blocking).
    """

    def _preflight_file(self, tmp_path: Path) -> Path:
        f = tmp_path / "results" / "experiment_413_env_autofix.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"honest_verdict": "auto_fix_applied", "retro_022_resolved": True}))
        return f

    def _run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        exp421_data: dict[str, Any] | None = None,
        gate_blocked: bool = True,
        gpu_healthy: bool = True,
        model_ok: bool = True,
        gpu_zombie: bool = False,
    ) -> dict[str, Any]:
        """Run main() with mocked infrastructure; return artifact dict."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        # Set up preflight file
        pf = self._preflight_file(tmp_path)
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", pf)

        # Set up Exp 421 result
        exp421_path = tmp_path / "results" / "experiment_421_adversarial_live.json"
        exp421_path.parent.mkdir(parents=True, exist_ok=True)
        if exp421_data is not None:
            exp421_path.write_text(json.dumps(exp421_data))
        else:
            exp421_path.write_text(json.dumps({"experiment": 421, "status": "partial"}))
        monkeypatch.setattr(_mod, "_EXP421_RESULT_PATH", exp421_path)

        # Gate 1: LiveGPUGate
        if gate_blocked:
            def _fake_gate(tmpl_obj: Any, ids: Any) -> dict[str, Any]:
                return tmpl_obj.build_result(
                    {"inference_mode": "blocked", "honest_verdict": "blocked",
                     "failure_reason": "test: gate blocked", "n_questions": 0,
                     "n_models": 0, "per_model_results": [],
                     "adversarial_drop_pct": 0.0, "repair_improvement_pct": 0.0},
                    status="blocked",
                )
        else:
            _fake_gate = lambda tmpl_obj, ids: None

        monkeypatch.setattr(_mod.LiveGPUGate, "require_live_or_blocked", staticmethod(_fake_gate))

        # Gate 2: check_dual_gpu_health
        monkeypatch.setattr(_mod, "check_dual_gpu_health", lambda: _make_gpu_health(zombie=gpu_zombie))

        # Gate 3: setup_gpu
        def _fake_setup_gpu(self_obj: Any, specs: Any, **kw: Any) -> dict[str, Any]:
            return {
                "all_healthy": gpu_healthy, "models": [], "prewarm_time_s": 0.0,
                "dual_gpu_auto_assigned": False, "model_server_active": False,
                "gpu_runner_active": False, "cpu_fallback": True,
                "gpu_monitor_results": {
                    "n_gpus_detected": 0, "n_zombies": 0, "idle_gpus": [],
                    "all_healthy": True, "error": "cpu_fallback",
                },
            }

        monkeypatch.setattr(_mod.ExperimentTemplate, "setup_gpu", _fake_setup_gpu)

        # Gate 4: _load_model_pipeline
        if model_ok:
            monkeypatch.setattr(_mod, "_load_model_pipeline", lambda *a: MagicMock())
        else:
            monkeypatch.setattr(
                _mod, "_load_model_pipeline",
                lambda *a: (_ for _ in ()).throw(RuntimeError("load failed"))
            )

        # Questions and three-condition runner
        monkeypatch.setattr(
            _mod, "load_gsm8k_questions",
            lambda n: [{"question_id": f"q_{i}", "question": f"Q {i}", "answer": str(i)}
                       for i in range(3)],
        )
        monkeypatch.setattr(
            _mod, "build_adversarial_questions",
            lambda qs, **kw: _make_paired_questions(len(qs)),
        )
        monkeypatch.setattr(
            _mod, "_run_three_conditions",
            lambda questions, model_obj, model_name, tmpl, idx: {
                "model_id": model_name,
                "n_questions": len(questions),
                "standard_accuracy": 0.80,
                "adversarial_accuracy": 0.65,
                "accuracy_drop": 0.15,
                "repaired_adversarial_accuracy": 0.72,
                "repair_improvement": 0.07,
                "inference_mode": "live_gpu",
            },
        )

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_429_adversarial_live.json"
        assert artifact_path.exists(), f"Artifact not written to {artifact_path}"
        return json.loads(artifact_path.read_text())

    # -- Exp 421 confirm path --

    def test_confirm_path_sets_confirmed_from(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exp421 = {
            "status": "success", "inference_mode": "live_gpu",
            "honest_verdict": "improvement_positive",
            "experiment": 421,
        }
        artifact = self._run_main(tmp_path, monkeypatch, exp421_data=exp421)
        assert artifact["confirmed_from"] == 421
        assert artifact["rerun"] is False
        assert artifact["experiment"] == 429

    def test_confirm_path_schema_v2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exp421 = {
            "status": "success", "inference_mode": "live_gpu",
            "honest_verdict": "improvement_positive", "experiment": 421,
        }
        artifact = self._run_main(tmp_path, monkeypatch, exp421_data=exp421)
        assert artifact["adversarial_schema"] == "carnot.adversarial_gsm8k.v2"

    # -- Gate 1 blocked --

    def test_gate1_blocked_produces_blocked_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate1_blocked_records_gate0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert "gate0_preflight_verdict" in artifact

    def test_gate1_blocked_n_questions_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["n_questions"] == 0

    # -- Gate 3 unhealthy --

    def test_gate3_unhealthy_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False)
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate3_unhealthy_has_failure_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False)
        assert "failure_reason" in artifact

    # -- Gate 4 model load failure --

    def test_gate4_model_load_failure_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"

    def test_gate4_model_load_failure_n_questions_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["n_questions"] == 0

    # -- Gate 2 GPU1 zombie warning (non-blocking) --

    def test_gate2_zombie_does_not_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True,
            gpu_zombie=True,
        )
        assert artifact["status"] == "success"
        assert artifact["gate2_gpu1_zombie"] is True

    # -- Success path --

    def test_success_inference_mode_live_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["inference_mode"] == "live_gpu"

    def test_success_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["status"] == "success"

    def test_success_schema_v2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["adversarial_schema"] == "carnot.adversarial_gsm8k.v2"

    def test_success_experiment_id_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["experiment"] == 429

    def test_success_per_model_results_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "per_model_results" in artifact
        assert len(artifact["per_model_results"]) == 2  # 2 models

    def test_success_adversarial_drop_pct_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "adversarial_drop_pct" in artifact
        assert "repair_improvement_pct" in artifact

    def test_success_improvement_positive_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _run_three_conditions mock returns repair_improvement=0.07 > 0 → improvement_positive
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["honest_verdict"] == "improvement_positive"

    def test_success_gate_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in [
            "gate0_autofix_applied", "gate0_preflight_verdict",
            "gate2_gpu1_zombie", "gate2_temperature_warning",
        ]:
            assert field in artifact, f"Missing gate field: {field}"

    def test_success_required_result_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiment_template import REQUIRED_RESULT_FIELDS

        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_rerun_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["rerun"] is True
        assert artifact["confirmed_from"] == 421
