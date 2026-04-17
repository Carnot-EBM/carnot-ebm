"""Tests for python/carnot/pipeline/humaneval_micro.py and
scripts/experiment_440_live_humaneval_micro.py.

Coverage targets (100%):

humaneval_micro module:
  - MicroHumanEvalResult: construction, field values, dataclass behavior
  - _result_to_dict: all fields serialized correctly as floats/ints/strings
  - build_micro_humaneval_artifact: all verdict paths
      * empty results → blocked
      * any non-live result → blocked (includes blocked results in per_model_results)
      * all live + best signed_improvement > 0 → code_verification_positive
      * all live + best signed_improvement <= 0 → code_no_improvement
      * single result with positive improvement → positive verdict
      * single result with zero improvement → no_improvement verdict
      * schema field is always carnot.humaneval_micro.v1

experiment_440 script (main() paths — all GPU mocked):
  - Gate 2 (LiveGPUGate) blocks → blocked artifact written
  - Gate 4 (setup_gpu) unhealthy → blocked artifact written
  - Gate 5 (_load_model_pipeline) fails for one model → blocked result appended,
    rest of models still attempted
  - All gates pass → success artifact with correct schema and fields
  - GPU1 zombie warning is non-blocking
  - Checkpoint called after each model completes
  - _write_artifact: JSON write via ExperimentTemplate (same as Exp 428 pattern)
  - _run_model_benchmark: integration via mock inference_fn

All tests run without a live GPU.  GPU infrastructure is mocked throughout.
Shared HumanEval helpers from Exp 369 are NOT re-tested here (100% in
test_experiment_369_humaneval_live.py).

Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_440_live_humaneval_micro.py"

# Ensure python/ and scripts/ are on the path.
for _d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_440 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_440", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_440"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

# ---------------------------------------------------------------------------
# Import module under test directly
# ---------------------------------------------------------------------------

from carnot.pipeline.humaneval_micro import (  # noqa: E402
    MicroHumanEvalResult,
    _result_to_dict,
    build_micro_humaneval_artifact,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
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
    for i in range(4)
]


def _make_result(
    *,
    model_id: str = "google/gemma-4-E4B-it",
    n_problems: int = 50,
    pass_before: float = 0.4,
    pass_after: float = 0.5,
    signed_improvement: float = 0.1,
    pbt_bugs: int = 2,
    inference_mode: str = "live_gpu",
) -> MicroHumanEvalResult:
    return MicroHumanEvalResult(
        model_id=model_id,
        n_problems=n_problems,
        pass_at_1_before=pass_before,
        pass_at_1_after=pass_after,
        signed_improvement=signed_improvement,
        pbt_bugs_found=pbt_bugs,
        inference_mode=inference_mode,
    )


def _make_he_result369(*, problem_id: str = "HumanEval/0", passed: bool = True) -> Any:
    """Return a HumanEvalResult369 for use as a mock inference result."""
    from experiment_369_humaneval_live import HumanEvalResult369
    return HumanEvalResult369(
        problem_id=problem_id,
        generated_code="def f(): return 1",
        passed_tests=passed,
        violations_found=0,
        repair_attempted=not passed,
        final_code="def f(): return 1",
        final_passed_tests=passed,
        pbt_bug_found=False,
    )


def _make_gpu_health(*, zombie: bool = False, temp_warn: bool = False) -> Any:
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


# ===========================================================================
# Tests for MicroHumanEvalResult
# ===========================================================================


class TestMicroHumanEvalResult:
    """Spec: REQ-BENCH-010, SCENARIO-BENCH-027"""

    def test_construction(self) -> None:
        r = _make_result()
        assert r.model_id == "google/gemma-4-E4B-it"
        assert r.n_problems == 50
        assert r.pass_at_1_before == pytest.approx(0.4)
        assert r.pass_at_1_after == pytest.approx(0.5)
        assert r.signed_improvement == pytest.approx(0.1)
        assert r.pbt_bugs_found == 2
        assert r.inference_mode == "live_gpu"

    def test_blocked_inference_mode(self) -> None:
        r = _make_result(inference_mode="blocked")
        assert r.inference_mode == "blocked"

    def test_zero_improvement(self) -> None:
        r = _make_result(signed_improvement=0.0)
        assert r.signed_improvement == pytest.approx(0.0)

    def test_negative_improvement(self) -> None:
        r = _make_result(signed_improvement=-0.05, pass_before=0.5, pass_after=0.45)
        assert r.signed_improvement < 0.0

    def test_pbt_bugs_zero(self) -> None:
        r = _make_result(pbt_bugs=0)
        assert r.pbt_bugs_found == 0


# ===========================================================================
# Tests for _result_to_dict
# ===========================================================================


class TestResultToDict:
    """Spec: REQ-BENCH-010"""

    def test_all_fields_present(self) -> None:
        r = _make_result()
        d = _result_to_dict(r)
        for field in ["model_id", "n_problems", "pass_at_1_before", "pass_at_1_after",
                       "signed_improvement", "pbt_bugs_found", "inference_mode"]:
            assert field in d

    def test_float_fields_are_float(self) -> None:
        r = _make_result()
        d = _result_to_dict(r)
        assert isinstance(d["pass_at_1_before"], float)
        assert isinstance(d["pass_at_1_after"], float)
        assert isinstance(d["signed_improvement"], float)

    def test_int_field(self) -> None:
        r = _make_result(pbt_bugs=3)
        d = _result_to_dict(r)
        assert d["pbt_bugs_found"] == 3

    def test_round_trip_values(self) -> None:
        r = _make_result(model_id="TestModel", n_problems=25, pbt_bugs=1, inference_mode="blocked")
        d = _result_to_dict(r)
        assert d["model_id"] == "TestModel"
        assert d["n_problems"] == 25
        assert d["pbt_bugs_found"] == 1
        assert d["inference_mode"] == "blocked"


# ===========================================================================
# Tests for build_micro_humaneval_artifact
# ===========================================================================


class TestBuildMicroHumanevalArtifact:
    """Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028"""

    def test_empty_results_blocked(self) -> None:
        artifact = build_micro_humaneval_artifact([])
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["inference_mode"] == "blocked"
        assert artifact["headline_result"] is None
        assert artifact["per_model_results"] == []

    def test_empty_results_schema(self) -> None:
        artifact = build_micro_humaneval_artifact([])
        assert artifact["humaneval_micro_schema"] == "carnot.humaneval_micro.v1"

    def test_non_live_result_blocked(self) -> None:
        results = [_make_result(inference_mode="blocked")]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["inference_mode"] == "blocked"
        assert artifact["headline_result"] is None

    def test_non_live_result_preserves_per_model(self) -> None:
        results = [_make_result(inference_mode="blocked")]
        artifact = build_micro_humaneval_artifact(results)
        # blocked due to non-live, but per_model_results should contain the result
        assert len(artifact["per_model_results"]) == 1

    def test_mixed_live_non_live_blocked(self) -> None:
        results = [
            _make_result(model_id="A", inference_mode="live_gpu", signed_improvement=0.1),
            _make_result(model_id="B", inference_mode="blocked", signed_improvement=0.0),
        ]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "blocked"

    def test_positive_improvement_verdict(self) -> None:
        results = [_make_result(signed_improvement=0.05)]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "code_verification_positive"

    def test_positive_improvement_inference_mode(self) -> None:
        results = [_make_result(signed_improvement=0.05)]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["inference_mode"] == "live_gpu"

    def test_positive_improvement_headline_present(self) -> None:
        results = [_make_result(model_id="GoodModel", signed_improvement=0.08)]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["headline_result"] is not None
        assert artifact["headline_result"]["model_id"] == "GoodModel"

    def test_zero_improvement_no_improvement_verdict(self) -> None:
        results = [_make_result(signed_improvement=0.0)]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "code_no_improvement"

    def test_negative_improvement_no_improvement_verdict(self) -> None:
        results = [_make_result(signed_improvement=-0.03)]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "code_no_improvement"

    def test_headline_is_best_model(self) -> None:
        results = [
            _make_result(model_id="ModelA", signed_improvement=0.02),
            _make_result(model_id="ModelB", signed_improvement=0.08),
        ]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["headline_result"]["model_id"] == "ModelB"

    def test_per_model_results_count(self) -> None:
        results = [
            _make_result(model_id="A"),
            _make_result(model_id="B"),
        ]
        artifact = build_micro_humaneval_artifact(results)
        assert len(artifact["per_model_results"]) == 2

    def test_schema_always_correct(self) -> None:
        for results in [
            [],
            [_make_result(inference_mode="blocked")],
            [_make_result(signed_improvement=0.1)],
            [_make_result(signed_improvement=0.0)],
        ]:
            artifact = build_micro_humaneval_artifact(results)
            assert artifact["humaneval_micro_schema"] == "carnot.humaneval_micro.v1"

    def test_one_positive_one_negative_is_positive(self) -> None:
        results = [
            _make_result(model_id="A", signed_improvement=-0.02),
            _make_result(model_id="B", signed_improvement=0.05),
        ]
        artifact = build_micro_humaneval_artifact(results)
        assert artifact["honest_verdict"] == "code_verification_positive"
        assert artifact["headline_result"]["model_id"] == "B"


# ===========================================================================
# Tests for _write_artifact helper (from experiment_440 module)
# ===========================================================================


class TestWriteArtifact:
    """Spec: REQ-BENCH-010"""

    def test_writes_json(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        out_path = tmp_path / "results" / "exp440.json"
        tmpl._output_path = out_path
        _mod._write_artifact(tmpl, {"key": "value"})
        assert out_path.exists()
        assert json.loads(out_path.read_text())["key"] == "value"

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        tmpl._output_path = tmp_path / "a" / "b" / "result.json"
        _mod._write_artifact(tmpl, {"x": 1})
        assert tmpl._output_path.exists()

    def test_pretty_printed(self, tmp_path: Path) -> None:
        tmpl = MagicMock()
        tmpl._output_path = tmp_path / "pretty.json"
        _mod._write_artifact(tmpl, {"a": 1})
        assert "\n" in tmpl._output_path.read_text()


# ===========================================================================
# Tests for main() — all GPU infrastructure mocked
# ===========================================================================


class TestMain:
    """Gate paths: blocked, gpu unhealthy, model load failure, success.

    Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
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
        signed_improvement_override: float = 0.1,
    ) -> dict[str, Any]:
        """Run main() with mocked infrastructure; return the artifact dict."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")

        # Gate 2: LiveGPUGate.require_live_or_blocked
        if gate_blocked:
            def _fake_gate(tmpl: Any, model_ids: Any) -> dict[str, Any]:
                return tmpl.build_result(
                    {
                        "schema": "carnot.humaneval_micro.v1",
                        "inference_mode": "blocked",
                        "honest_verdict": "blocked",
                        "failure_reason": "test: gate mocked as blocked",
                        "n_problems": 0,
                        "per_model_results": [],
                    },
                    status="blocked",
                )
        else:
            _fake_gate = lambda tmpl, model_ids: None

        monkeypatch.setattr(_mod.LiveGPUGate, "require_live_or_blocked", staticmethod(_fake_gate))

        # Gate 3: check_dual_gpu_health
        monkeypatch.setattr(
            _mod, "check_dual_gpu_health",
            lambda: _make_gpu_health(zombie=gpu_zombie),
        )

        # Gate 4: setup_gpu
        def _fake_setup_gpu(self_obj: Any, model_specs: Any, **kw: Any) -> dict[str, Any]:
            return {
                "all_healthy": gpu_healthy,
                "models": [{"name": s["name"], "health_ok": gpu_healthy} for s in model_specs],
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

        # Gate 5: _load_model_pipeline
        monkeypatch.setattr(
            _mod, "_load_model_pipeline",
            lambda *a, **kw: (MagicMock(), MagicMock(), "cpu", model_ok),
        )

        # Problem loading
        monkeypatch.setattr(
            _mod, "_load_problems",
            lambda: (problems_override if problems_override is not None else _MINI_PROBLEMS),
        )

        # _process_problem: always returns a passing result so signed_improvement can be controlled
        # We override _run_model_benchmark to return a fixed MicroHumanEvalResult.
        def _fake_run_model_benchmark(
            model_spec: Any, problems: Any, tokenizer: Any, model: Any,
            device: Any, executor: Any, exp_prefix: Any,
        ) -> MicroHumanEvalResult:
            return MicroHumanEvalResult(
                model_id=model_spec["hf_id"],
                n_problems=len(problems),
                pass_at_1_before=0.4,
                pass_at_1_after=0.4 + signed_improvement_override,
                signed_improvement=signed_improvement_override,
                pbt_bugs_found=0,
                inference_mode="live_gpu",
            )

        monkeypatch.setattr(_mod, "_run_model_benchmark", _fake_run_model_benchmark)

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_440_live_humaneval_micro.json"
        assert artifact_path.exists(), f"Artifact not written to {artifact_path}"
        return json.loads(artifact_path.read_text())

    # -- Gate 2 blocked --

    def test_gate2_blocked_produces_blocked_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["inference_mode"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate2_blocked_n_problems_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert artifact["n_problems"] == 0

    def test_gate2_blocked_records_autofix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(tmp_path, monkeypatch, gate_blocked=True)
        assert "gate0_autofix_applied" in artifact

    # -- Gate 4 unhealthy --

    def test_gate4_unhealthy_blocked_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "blocked"

    def test_gate4_unhealthy_failure_reason_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=False
        )
        assert "failure_reason" in artifact

    # -- Gate 5 model load failure (one model fails) --

    def test_gate5_model_load_failure_produces_success_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When model_ok=False, both models get blocked results → blocked verdict
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=False
        )
        # All model results have inference_mode='blocked' → build_micro_humaneval_artifact blocked
        assert artifact["honest_verdict"] == "blocked"
        assert artifact["status"] == "success"

    # -- GPU1 zombie warning (non-blocking) --

    def test_gate3_zombie_does_not_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch,
            gate_blocked=False, gpu_healthy=True, model_ok=True,
            gpu_zombie=True,
        )
        assert artifact["status"] == "success"
        assert artifact["gate3_gpu1_zombie"] is True

    # -- Success path with positive improvement --

    def test_success_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["humaneval_micro_schema"] == "carnot.humaneval_micro.v1"

    def test_success_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["status"] == "success"

    def test_success_positive_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch,
            gate_blocked=False, gpu_healthy=True, model_ok=True,
            signed_improvement_override=0.05,
        )
        assert artifact["honest_verdict"] == "code_verification_positive"

    def test_success_no_improvement_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch,
            gate_blocked=False, gpu_healthy=True, model_ok=True,
            signed_improvement_override=0.0,
        )
        assert artifact["honest_verdict"] == "code_no_improvement"

    def test_success_inference_mode_live_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["inference_mode"] == "live_gpu"

    def test_success_experiment_id_correct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert artifact["experiment"] == 440

    def test_success_per_model_results_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        # Two models in MODEL_SPECS
        assert len(artifact["per_model_results"]) == 2

    def test_success_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiment_template import REQUIRED_RESULT_FIELDS
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_gate3_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        assert "gate0_autofix_applied" in artifact
        assert "gate3_gpu1_zombie" in artifact
        assert "gate3_temperature_warning" in artifact

    def test_success_checkpointing_per_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpoint called once after each model completes (2 models → 2 checkpoints)."""
        checkpoint_steps: list[int] = []
        original = _mod.ExperimentTemplate.checkpoint_save

        def _tracking(self_obj: Any, partial: Any, *, step: int) -> None:
            checkpoint_steps.append(step)
            original(self_obj, partial, step=step)

        monkeypatch.setattr(_mod.ExperimentTemplate, "checkpoint_save", _tracking)

        self._run_main(
            tmp_path, monkeypatch, gate_blocked=False, gpu_healthy=True, model_ok=True
        )
        # 2 models → 2 checkpoint calls (step=1, step=2)
        assert checkpoint_steps == [1, 2]


# ===========================================================================
# Tests for _run_model_benchmark (integration via mocked LongRunBenchmarkExecutor)
# ===========================================================================


class TestRunModelBenchmark:
    """Spec: REQ-BENCH-010"""

    def test_returns_micro_humaneval_result(self) -> None:
        """_run_model_benchmark returns a MicroHumanEvalResult with live_gpu mode."""
        from experiment_369_humaneval_live import HumanEvalResult369
        from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
        from dataclasses import asdict
        import tempfile

        with tempfile.TemporaryDirectory() as ckpt_dir:
            executor = LongRunBenchmarkExecutor(batch_size=2, checkpoint_dir=ckpt_dir)

            problems = _MINI_PROBLEMS[:2]
            model_spec = {"name": "TestModel", "hf_id": "test/model", "gpu": 0}

            # _process_problem is called inside _inference_fn which wraps with asdict()
            # We mock _process_problem to return a HumanEvalResult369 (as normal).
            original_process = _mod._process_problem

            def _fake_process(p: Any, tok: Any, mod: Any, dev: Any) -> HumanEvalResult369:
                return _make_he_result369(problem_id=p["task_id"], passed=True)

            _mod._process_problem = _fake_process
            try:
                result = _mod._run_model_benchmark(
                    model_spec=model_spec,
                    problems=problems,
                    tokenizer=MagicMock(),
                    model=MagicMock(),
                    device="cpu",
                    executor=executor,
                    exp_prefix="test",
                )
            finally:
                _mod._process_problem = original_process

        assert isinstance(result, MicroHumanEvalResult)
        assert result.inference_mode == "live_gpu"
        assert result.model_id == "test/model"
        assert result.n_problems == 2

    def test_pbt_bugs_aggregated(self) -> None:
        """pbt_bugs_found is the sum of pbt_bug_found=True across all results."""
        from experiment_369_humaneval_live import HumanEvalResult369
        from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
        import tempfile

        with tempfile.TemporaryDirectory() as ckpt_dir:
            executor = LongRunBenchmarkExecutor(batch_size=4, checkpoint_dir=ckpt_dir)
            problems = _MINI_PROBLEMS[:4]
            model_spec = {"name": "T", "hf_id": "t/m", "gpu": 0}

            call_count = [0]

            def _fake_process(p: Any, tok: Any, mod: Any, dev: Any) -> HumanEvalResult369:
                call_count[0] += 1
                return HumanEvalResult369(
                    problem_id=p["task_id"],
                    generated_code="def f(): return 1",
                    passed_tests=True,
                    violations_found=0,
                    repair_attempted=False,
                    final_code="def f(): return 1",
                    final_passed_tests=True,
                    pbt_bug_found=(call_count[0] % 2 == 0),  # bugs on even calls
                )

            original_process = _mod._process_problem
            _mod._process_problem = _fake_process
            try:
                result = _mod._run_model_benchmark(
                    model_spec=model_spec,
                    problems=problems,
                    tokenizer=MagicMock(),
                    model=MagicMock(),
                    device="cpu",
                    executor=executor,
                    exp_prefix="test",
                )
            finally:
                _mod._process_problem = original_process

        # 4 problems, even calls (2nd, 4th) have pbt_bug_found=True → 2 bugs
        assert result.pbt_bugs_found == 2

    def test_exception_in_process_problem_handled(self) -> None:
        """If _process_problem raises, a failed result is appended and run continues."""
        from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
        import tempfile

        with tempfile.TemporaryDirectory() as ckpt_dir:
            executor = LongRunBenchmarkExecutor(batch_size=2, checkpoint_dir=ckpt_dir)
            problems = _MINI_PROBLEMS[:2]
            model_spec = {"name": "T", "hf_id": "t/m", "gpu": 0}

            def _failing_process(p: Any, tok: Any, mod: Any, dev: Any) -> None:
                raise RuntimeError("simulated failure")

            original_process = _mod._process_problem
            _mod._process_problem = _failing_process
            try:
                result = _mod._run_model_benchmark(
                    model_spec=model_spec,
                    problems=problems,
                    tokenizer=MagicMock(),
                    model=MagicMock(),
                    device="cpu",
                    executor=executor,
                    exp_prefix="test",
                )
            finally:
                _mod._process_problem = original_process

        # Failures produce results with passed_tests=False → pass@1 = 0.0
        assert result.pass_at_1_before == pytest.approx(0.0)
        assert result.pass_at_1_after == pytest.approx(0.0)
        assert result.signed_improvement == pytest.approx(0.0)
