"""Tests for scripts/experiment_380_humaneval_execute.py.

Covers 100% of new code in Exp 380:
  - _write_artifact: JSON file write via ExperimentTemplate
  - main(): all gate paths (LiveGPUGate blocked, setup_gpu unhealthy,
    model load failure, live success)

All tests run without a live GPU.  Live model loading and GPU infrastructure
are mocked throughout.

Note: the core HumanEval helpers (HumanEvalResult369, compute_pass_at_1,
_run_tests, etc.) are NOT re-tested here — they are already covered at 100%
in test_experiment_369_humaneval_live.py.  This file tests ONLY the new code
introduced in Exp 380.

Spec: REQ-BENCH-004, SCENARIO-BENCH-021
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
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_380_humaneval_execute.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_380 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_380", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_380"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    for d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
        if d not in sys.path:
            sys.path.insert(0, d)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

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
    """Build a HumanEvalResult369-compatible mock with all pass=True."""
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


# ---------------------------------------------------------------------------
# _write_artifact tests
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    """Tests for the _write_artifact helper (only new function in Exp 380)."""

    def test_writes_json(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out_path = tmp_path / "results" / "exp380.json"
        tmpl._output_path = out_path
        _write_artifact(tmpl, {"key": "value"})
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["key"] == "value"

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out_path = tmp_path / "a" / "b" / "c" / "result.json"
        tmpl._output_path = out_path
        _write_artifact(tmpl, {"x": 1})
        assert out_path.exists()

    def test_pretty_printed(self, tmp_path: Path) -> None:
        """Artifact must be pretty-printed (indent=2) for human readability."""
        tmpl = MagicMock()
        out_path = tmp_path / "pretty.json"
        tmpl._output_path = out_path
        _write_artifact(tmpl, {"a": 1, "b": 2})
        raw = out_path.read_text()
        # Pretty-printed JSON has newlines
        assert "\n" in raw

    def test_existing_file_overwritten(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out_path = tmp_path / "overwrite.json"
        out_path.write_text('{"old": true}')
        tmpl._output_path = out_path
        _write_artifact(tmpl, {"new": True})
        loaded = json.loads(out_path.read_text())
        assert "new" in loaded
        assert "old" not in loaded


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() — all GPU interactions mocked.

    Covers every gate path:
      1. LiveGPUGate returns blocked → write blocked artifact, return
      2. setup_gpu not all_healthy → write blocked artifact, return
      3. _load_model_pipeline fails → write blocked artifact, return
      4. All gates pass → run benchmark → write success artifact
    """

    def _run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        gate_blocked: bool = True,
        gpu_healthy: bool = True,
        model_ok: bool = True,
        problems_override: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run main() with mocked infrastructure; return artifact dict."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")  # always off; gate is mocked

        # ---- LiveGPUGate.require_live_or_blocked ----
        if gate_blocked:
            # Return a blocked artifact dict (gate failure)
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

        # ---- setup_gpu ----
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

        monkeypatch.setattr(
            _mod.ExperimentTemplate, "setup_gpu", _fake_setup_gpu
        )

        # ---- _load_model_pipeline ----
        monkeypatch.setattr(
            _mod,
            "_load_model_pipeline",
            lambda *a, **kw: (MagicMock(), MagicMock(), "cpu", model_ok),
        )

        # ---- _load_problems ----
        monkeypatch.setattr(
            _mod,
            "_load_problems",
            lambda: (problems_override if problems_override is not None else _MINI_PROBLEMS),
        )

        # ---- _process_problem ----
        monkeypatch.setattr(
            _mod,
            "_process_problem",
            lambda p, tok, mod, dev: _make_he_result(problem_id=p["task_id"]),
        )

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_380_humaneval_execute.json"
        assert artifact_path.exists(), f"Artifact not written to {artifact_path}"
        return json.loads(artifact_path.read_text())

    # -- Gate 1: LiveGPUGate blocks --

    def test_gate_blocked_produces_blocked_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate_blocked_n_problems_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["n_problems"] == 0

    def test_gate_blocked_signed_improvement_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["signed_improvement"] == 0.0

    # -- Gate 2: setup_gpu unhealthy --

    def test_unhealthy_gpu_produces_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_unhealthy_gpu_failure_reason_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert "failure_reason" in artifact

    # -- Gate 3: model load failure --

    def test_model_load_failure_produces_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["inference_mode"] == "blocked"
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_model_load_failure_n_problems_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        assert artifact["n_problems"] == 0

    # -- Success path --

    def test_success_inference_mode_live_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["inference_mode"] == "live_gpu"

    def test_success_status_success(
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

    def test_success_n_problems_matches_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["n_problems"] == len(_MINI_PROBLEMS)

    def test_success_pbt_bugs_found_key_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "pbt_bugs_found" in artifact

    def test_success_pass_at_1_before_and_after_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "pass_at_1_before" in artifact
        assert "pass_at_1_after" in artifact

    def test_success_all_problems_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all problems pass on first generation, pass@1 = 1.0."""
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["pass_at_1_before"] == 1.0
        assert artifact["pass_at_1_after"] == 1.0

    def test_success_no_improvement_verdict_when_all_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All pass on first gen → signed_improvement = 0 → no_improvement verdict."""
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        # All mocked results pass on first gen so signed_improvement == 0
        assert artifact["signed_improvement"] == 0.0
        assert artifact["honest_verdict"] == "no_improvement"

    def test_success_positive_verdict_when_improvement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When repair fixes some problems, honest_verdict = code_verification_positive."""
        from experiment_369_humaneval_live import HumanEvalResult369

        # Patch _process_problem to return a failing-then-repaired result
        def _failing_then_repaired(
            p: Any, tok: Any, mod: Any, dev: Any
        ) -> HumanEvalResult369:
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
        monkeypatch.setattr(
            _mod.LiveGPUGate,
            "require_live_or_blocked",
            staticmethod(lambda tmpl, model_ids: None),
        )

        def _fake_setup_gpu(self_obj: Any, model_specs: Any, **kw: Any) -> dict[str, Any]:
            return {
                "all_healthy": True,
                "models": [],
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
        monkeypatch.setattr(
            _mod,
            "_load_model_pipeline",
            lambda *a, **kw: (MagicMock(), MagicMock(), "cpu", True),
        )
        monkeypatch.setattr(_mod, "_load_problems", lambda: _MINI_PROBLEMS)
        monkeypatch.setattr(_mod, "_process_problem", _failing_then_repaired)

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_380_humaneval_execute.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["signed_improvement"] > 0
        assert artifact["honest_verdict"] == "code_verification_positive"

    def test_success_required_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All REQUIRED_RESULT_FIELDS must appear in the artifact."""
        from experiment_template import REQUIRED_RESULT_FIELDS

        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_experiment_id_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["experiment"] == 380

    def test_success_checkpointing_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With 10 problems, checkpointing should be invoked once (at step 10)."""
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
        checkpoint_calls: list[int] = []

        original_checkpoint_save = _mod.ExperimentTemplate.checkpoint_save

        def _tracking_checkpoint(self_obj: Any, partial: Any, *, step: int) -> None:
            checkpoint_calls.append(step)
            original_checkpoint_save(self_obj, partial, step=step)

        monkeypatch.setattr(
            _mod.ExperimentTemplate, "checkpoint_save", _tracking_checkpoint
        )

        self._run_main(
            tmp_path,
            monkeypatch,
            gate_blocked=False,
            gpu_healthy=True,
            model_ok=True,
            problems_override=ten_problems,
        )
        assert checkpoint_calls == [10]

    def test_gate_blocked_process_problem_not_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When gate blocks, _process_problem must never be called."""
        calls: list[int] = []

        def _tracking_process(p: Any, tok: Any, mod: Any, dev: Any) -> Any:
            calls.append(1)
            return _make_he_result()

        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        def _fake_gate(tmpl: Any, model_ids: Any) -> dict[str, Any]:
            return tmpl.build_result(
                {
                    "humaneval_schema": "carnot.humaneval_benchmark.v2",
                    "inference_mode": "blocked",
                    "honest_verdict": "blocked",
                    "failure_reason": "test",
                    "n_problems": 0,
                    "pass_at_1_before": 0.0,
                    "pass_at_1_after": 0.0,
                    "signed_improvement": 0.0,
                    "pbt_bugs_found": 0,
                },
                status="blocked",
            )

        monkeypatch.setattr(_mod.LiveGPUGate, "require_live_or_blocked", staticmethod(_fake_gate))
        monkeypatch.setattr(_mod, "_process_problem", _tracking_process)

        _mod.main()

        assert calls == [], "_process_problem must not be called when gate blocks"
