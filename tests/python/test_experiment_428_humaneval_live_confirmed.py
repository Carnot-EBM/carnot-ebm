"""Tests for scripts/experiment_428_humaneval_live_confirmed.py.

Covers 100% of new code in Exp 428:
  - _load_preflight_verdict: happy path, missing file, malformed JSON
  - _write_artifact: JSON write via ExperimentTemplate (same helper as Exp 380)
  - main(): all gate paths:
      Gate 1 blocked → blocked artifact
      Gate 3 unhealthy → blocked artifact
      Gate 4 model load failure → blocked artifact
      All gates pass → success artifact

All tests run without a live GPU.  GPU infrastructure is mocked throughout.
The shared HumanEval helpers from Exp 369 are NOT re-tested here (already at
100% in test_experiment_369_humaneval_live.py).

Spec: REQ-BENCH-004, SCENARIO-BENCH-021, REQ-INFRA-021
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
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_428_humaneval_live_confirmed.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_428 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_428", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_428"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    for d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
        if d not in sys.path:
            sys.path.insert(0, d)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

_load_preflight_verdict = _mod._load_preflight_verdict
_write_artifact = _mod._write_artifact


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MINI_PROBLEMS = [
    {
        "task_id": f"HumanEval/{i}",
        "entry_point": "f",
        "prompt": "def f():\n",
        "canonical_solution": "    return 1\n",
        "test_cases": [],
        "test": "",
    }
    for i in range(3)
]


def _make_he_result(*, problem_id: str = "HumanEval/0") -> Any:
    """Return a HumanEvalResult369 with all-pass values."""
    from experiment_369_humaneval_live import HumanEvalResult369

    return HumanEvalResult369(
        problem_id=problem_id,
        generated_code="def f(): return 1",
        passed_tests=True,
        violations_found=0,
        repair_attempted=False,
        final_code="def f(): return 1",
        final_passed_tests=True,
        pbt_bug_found=False,
    )


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


# ---------------------------------------------------------------------------
# _load_preflight_verdict tests
# ---------------------------------------------------------------------------


class TestLoadPreflightVerdict:
    """Tests for the Gate 0 preflight loader."""

    def test_loads_valid_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        preflight_file = tmp_path / "results" / "experiment_413_env_autofix.json"
        preflight_file.parent.mkdir(parents=True)
        preflight_file.write_text(json.dumps({"honest_verdict": "auto_fix_applied", "retro_022_resolved": True}))
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", preflight_file)
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
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", bad_file)
        result = _load_preflight_verdict()
        assert result["honest_verdict"] == "preflight_file_missing"
        assert "error" in result


# ---------------------------------------------------------------------------
# _write_artifact tests
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    """Tests for the _write_artifact helper."""

    def test_writes_json(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out_path = tmp_path / "results" / "exp428.json"
        tmpl._output_path = out_path
        _write_artifact(tmpl, {"key": "value"})
        assert out_path.exists()
        assert json.loads(out_path.read_text())["key"] == "value"

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
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() — all GPU interactions mocked.

    Gate paths covered:
      1. Gate 1 (LiveGPUGate) blocks → blocked artifact
      2. Gate 3 (setup_gpu) unhealthy → blocked artifact
      3. Gate 4 (_load_model_pipeline) fails → blocked artifact
      4. All gates pass → success artifact
      5. GPU1 zombie warning (non-blocking)
    """

    def _run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        gate_blocked: bool = True,
        gpu_healthy: bool = True,
        model_ok: bool = True,
        gpu_zombie: bool = False,
        problems_override: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run main() with mocked infrastructure; return artifact dict."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        # Gate 0: mock preflight file path so it uses the real test fixture
        preflight_file = tmp_path / "results" / "experiment_413_env_autofix.json"
        preflight_file.parent.mkdir(parents=True, exist_ok=True)
        preflight_file.write_text(json.dumps({
            "honest_verdict": "auto_fix_applied",
            "retro_022_resolved": True,
        }))
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", preflight_file)

        # Gate 1: LiveGPUGate.require_live_or_blocked
        if gate_blocked:
            def _fake_gate(tmpl: Any, model_ids: Any) -> dict[str, Any]:
                return tmpl.build_result(
                    {
                        "humaneval_schema": "carnot.humaneval_benchmark.v2",
                        "inference_mode": "blocked",
                        "honest_verdict": "blocked",
                        "failure_reason": "test: gate mocked as blocked",
                        "n_problems": 0,
                        "pass_at_1_before": 0.0,
                        "pass_at_1_after": 0.0,
                        "signed_improvement": 0.0,
                        "pbt_bugs_found": 0,
                    },
                    status="blocked",
                )
        else:
            _fake_gate = lambda tmpl, model_ids: None  # gate passes

        monkeypatch.setattr(_mod.LiveGPUGate, "require_live_or_blocked", staticmethod(_fake_gate))

        # Gate 2: check_dual_gpu_health
        monkeypatch.setattr(
            _mod, "check_dual_gpu_health", lambda: _make_gpu_health(zombie=gpu_zombie)
        )

        # Gate 3: setup_gpu
        def _fake_setup_gpu(self_obj: Any, model_specs: Any, **kw: Any) -> dict[str, Any]:
            return {
                "all_healthy": gpu_healthy,
                "models": [{"name": "Gemma4-E4B-it", "health_ok": gpu_healthy}],
                "prewarm_time_s": 0.0,
                "dual_gpu_auto_assigned": False,
                "model_server_active": False,
                "gpu_runner_active": False,
                "cpu_fallback": True,
                "gpu_monitor_results": {
                    "n_gpus_detected": 0,
                    "n_zombies": 0,
                    "idle_gpus": [],
                    "all_healthy": True,
                    "error": "cpu_fallback",
                },
            }

        monkeypatch.setattr(_mod.ExperimentTemplate, "setup_gpu", _fake_setup_gpu)

        # Gate 4: _load_model_pipeline
        monkeypatch.setattr(
            _mod, "_load_model_pipeline",
            lambda *a, **kw: (MagicMock(), MagicMock(), "cpu", model_ok),
        )

        # Problems and processing
        monkeypatch.setattr(
            _mod, "_load_problems",
            lambda: (problems_override if problems_override is not None else _MINI_PROBLEMS),
        )
        monkeypatch.setattr(
            _mod, "_process_problem",
            lambda p, tok, mod, dev: _make_he_result(problem_id=p["task_id"]),
        )

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_428_humaneval_live_confirmed.json"
        assert artifact_path.exists(), f"Artifact not written to {artifact_path}"
        return json.loads(artifact_path.read_text())

    # -- Gate 1: LiveGPUGate blocks --

    def test_gate1_blocked_produces_blocked_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate1_blocked_n_problems_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["n_problems"] == 0

    def test_gate1_blocked_records_gate0_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert "gate0_preflight_verdict" in artifact
        assert artifact["gate0_preflight_verdict"] == "auto_fix_applied"

    # -- Gate 3: setup_gpu unhealthy --

    def test_gate3_unhealthy_produces_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate3_unhealthy_failure_reason_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert "failure_reason" in artifact

    # -- Gate 4: model load failure --

    def test_gate4_model_load_failure_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate4_model_load_failure_n_problems_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["n_problems"] == 0

    # -- Gate 2: GPU1 zombie warning (non-blocking) --

    def test_gate2_zombie_does_not_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch,
            gate_blocked=False, gpu_healthy=True, model_ok=True,
            gpu_zombie=True,
        )
        # Zombie is a warning, not a block — experiment still runs
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
        assert artifact["humaneval_schema"] == "carnot.humaneval_benchmark.v2"

    def test_success_experiment_id_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["experiment"] == 428

    def test_success_n_problems_matches_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["n_problems"] == len(_MINI_PROBLEMS)

    def test_success_exp226_baseline_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "exp226_baseline_pass_at_1" in artifact
        assert "exp226_target_pass_at_1" in artifact
        assert artifact["exp226_baseline_pass_at_1"] == pytest.approx(0.116)
        assert artifact["exp226_target_pass_at_1"] == pytest.approx(0.146)

    def test_success_gate_fields_in_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in ["gate0_autofix_applied", "gate0_preflight_verdict",
                      "gate2_gpu1_zombie", "gate2_temperature_warning"]:
            assert field in artifact, f"Missing field: {field}"

    def test_success_required_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiment_template import REQUIRED_RESULT_FIELDS
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_positive_verdict_when_improvement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When repair fixes some problems, honest_verdict = code_verification_positive."""
        from experiment_369_humaneval_live import HumanEvalResult369

        def _failing_then_repaired(p: Any, tok: Any, mod: Any, dev: Any) -> HumanEvalResult369:
            return HumanEvalResult369(
                problem_id=p["task_id"],
                generated_code="def f(): return 0",
                passed_tests=False,
                violations_found=1,
                repair_attempted=True,
                final_code="def f(): return 1",
                final_passed_tests=True,
                pbt_bug_found=False,
            )

        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        preflight_file = tmp_path / "results" / "experiment_413_env_autofix.json"
        preflight_file.parent.mkdir(parents=True, exist_ok=True)
        preflight_file.write_text(json.dumps({"honest_verdict": "auto_fix_applied", "retro_022_resolved": True}))
        monkeypatch.setattr(_mod, "_EXP413_PREFLIGHT_PATH", preflight_file)
        monkeypatch.setattr(_mod.LiveGPUGate, "require_live_or_blocked", staticmethod(lambda tmpl, ids: None))
        monkeypatch.setattr(_mod, "check_dual_gpu_health", lambda: _make_gpu_health())

        def _fake_setup_gpu(self_obj: Any, model_specs: Any, **kw: Any) -> dict[str, Any]:
            return {
                "all_healthy": True, "models": [], "prewarm_time_s": 0.0,
                "dual_gpu_auto_assigned": False, "model_server_active": False,
                "gpu_runner_active": False, "cpu_fallback": True,
                "gpu_monitor_results": {"n_gpus_detected": 0, "n_zombies": 0,
                                        "idle_gpus": [], "all_healthy": True, "error": "cpu_fallback"},
            }

        monkeypatch.setattr(_mod.ExperimentTemplate, "setup_gpu", _fake_setup_gpu)
        monkeypatch.setattr(_mod, "_load_model_pipeline", lambda *a, **kw: (MagicMock(), MagicMock(), "cpu", True))
        monkeypatch.setattr(_mod, "_load_problems", lambda: _MINI_PROBLEMS)
        monkeypatch.setattr(_mod, "_process_problem", _failing_then_repaired)

        _mod.main()

        artifact = json.loads(
            (tmp_path / "results" / "experiment_428_humaneval_live_confirmed.json").read_text()
        )
        assert artifact["signed_improvement"] > 0
        assert artifact["honest_verdict"] == "code_verification_positive"

    def test_success_checkpointing_at_10(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With 10 problems, checkpointing fires exactly once at step 10."""
        ten_problems = [
            {
                "task_id": f"HumanEval/{i}",
                "entry_point": "f",
                "prompt": "def f():\n",
                "canonical_solution": "    return 1\n",
                "test_cases": [],
                "test": "",
            }
            for i in range(10)
        ]
        checkpoint_steps: list[int] = []

        original = _mod.ExperimentTemplate.checkpoint_save

        def _tracking(self_obj: Any, partial: Any, *, step: int) -> None:
            checkpoint_steps.append(step)
            original(self_obj, partial, step=step)

        monkeypatch.setattr(_mod.ExperimentTemplate, "checkpoint_save", _tracking)

        self._run_main(
            tmp_path, monkeypatch,
            gate_blocked=False, gpu_healthy=True, model_ok=True,
            problems_override=ten_problems,
        )
        assert checkpoint_steps == [10]
