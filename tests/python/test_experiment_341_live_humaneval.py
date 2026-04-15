"""Tests for scripts/experiment_341_live_humaneval.py.

Covers:
  - HumanEvalResult dataclass: construction, asdict serialization
  - compute_pass_at_1: fraction with passed_tests=True before repair
  - compute_pass_at_1_after_repair: fraction with final_passed_tests=True
  - build_humaneval_artifact: schema, headline fields, inference_mode
  - _run_tests: test execution logic
  - _extract_code: markdown fence stripping
  - _simulated_solution: CI-mode deterministic code generation
  - _manual_problems: 50 problems returned
  - _process_problem: simulated mode pipeline without live LLM
  - main(): CI-mode (CARNOT_FORCE_LIVE=0) full run produces artifact

All tests run under CARNOT_FORCE_LIVE=0 (no GPU, no model loading).

Spec: REQ-BENCH-004, SCENARIO-BENCH-010, SCENARIO-BENCH-011
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_341_live_humaneval.py"


# ---------------------------------------------------------------------------
# Module loader — no live GPU, no real model loading
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_341 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_341", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_341"] = mod
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    python_dir = str(REPO_ROOT / "python")
    scripts_dir = str(REPO_ROOT / "scripts")
    for d in [python_dir, scripts_dir]:
        if d not in sys.path:
            sys.path.insert(0, d)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

HumanEvalResult = _mod.HumanEvalResult
compute_pass_at_1 = _mod.compute_pass_at_1
compute_pass_at_1_after_repair = _mod.compute_pass_at_1_after_repair
build_humaneval_artifact = _mod.build_humaneval_artifact
_run_tests = _mod._run_tests
_extract_code = _mod._extract_code
_simulated_solution = _mod._simulated_solution
_manual_problems = _mod._manual_problems
_process_problem = _mod._process_problem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    *,
    passed_tests: bool = True,
    final_passed_tests: bool = True,
    violations_found: int = 0,
    repair_attempted: bool = False,
    problem_id: str = "HumanEval/0",
    generated_code: str = "def foo(): return 1",
    final_code: str = "def foo(): return 1",
) -> HumanEvalResult:
    return HumanEvalResult(
        problem_id=problem_id,
        generated_code=generated_code,
        passed_tests=passed_tests,
        violations_found=violations_found,
        repair_attempted=repair_attempted,
        final_code=final_code,
        final_passed_tests=final_passed_tests,
    )


def _simple_problem() -> dict[str, Any]:
    """Minimal problem dict for testing _process_problem in simulated mode."""
    return {
        "task_id": "HumanEval/0",
        "entry_point": "add",
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return a + b.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "test_cases": [([1, 2], 3), ([0, 0], 0)],
        "test": "",
    }


# ---------------------------------------------------------------------------
# HumanEvalResult dataclass tests (SCENARIO-BENCH-010)
# ---------------------------------------------------------------------------


class TestHumanEvalResult:
    """Tests for the HumanEvalResult dataclass."""

    def test_construction_defaults(self):
        """Verify all fields can be set and retrieved."""
        r = HumanEvalResult(
            problem_id="HumanEval/0",
            generated_code="def f(): pass",
            passed_tests=True,
            violations_found=0,
            repair_attempted=False,
            final_code="def f(): pass",
            final_passed_tests=True,
        )
        assert r.problem_id == "HumanEval/0"
        assert r.generated_code == "def f(): pass"
        assert r.passed_tests is True
        assert r.violations_found == 0
        assert r.repair_attempted is False
        assert r.final_code == "def f(): pass"
        assert r.final_passed_tests is True

    def test_asdict_serializable(self):
        """Verify asdict() produces a JSON-serializable dict."""
        r = _make_result()
        d = asdict(r)
        assert "problem_id" in d
        assert "generated_code" in d
        assert "passed_tests" in d
        assert "violations_found" in d
        assert "repair_attempted" in d
        assert "final_code" in d
        assert "final_passed_tests" in d
        # Must be JSON-serializable
        json.dumps(d)

    def test_failed_result_fields(self):
        """Verify a failed result has expected field values."""
        r = HumanEvalResult(
            problem_id="HumanEval/5",
            generated_code="def f(): return None",
            passed_tests=False,
            violations_found=3,
            repair_attempted=True,
            final_code="def f(): return 1",
            final_passed_tests=True,
        )
        assert r.passed_tests is False
        assert r.violations_found == 3
        assert r.repair_attempted is True
        assert r.final_passed_tests is True


# ---------------------------------------------------------------------------
# compute_pass_at_1 tests (SCENARIO-BENCH-010)
# ---------------------------------------------------------------------------


class TestComputePassAt1:
    """Tests for compute_pass_at_1."""

    def test_all_pass(self):
        """All passed_tests=True → 1.0."""
        results = [_make_result(passed_tests=True) for _ in range(5)]
        assert compute_pass_at_1(results) == 1.0

    def test_none_pass(self):
        """All passed_tests=False → 0.0."""
        results = [_make_result(passed_tests=False) for _ in range(5)]
        assert compute_pass_at_1(results) == 0.0

    def test_half_pass(self):
        """2/4 passed → 0.5."""
        results = [
            _make_result(passed_tests=True),
            _make_result(passed_tests=False),
            _make_result(passed_tests=True),
            _make_result(passed_tests=False),
        ]
        assert compute_pass_at_1(results) == 0.5

    def test_empty_list(self):
        """Empty list → 0.0 (no division by zero)."""
        assert compute_pass_at_1([]) == 0.0

    def test_single_pass(self):
        """Single passing result → 1.0."""
        assert compute_pass_at_1([_make_result(passed_tests=True)]) == 1.0

    def test_single_fail(self):
        """Single failing result → 0.0."""
        assert compute_pass_at_1([_make_result(passed_tests=False)]) == 0.0


# ---------------------------------------------------------------------------
# compute_pass_at_1_after_repair tests (SCENARIO-BENCH-010)
# ---------------------------------------------------------------------------


class TestComputePassAt1AfterRepair:
    """Tests for compute_pass_at_1_after_repair."""

    def test_repair_recovers(self):
        """failed + repaired_to_pass → 1.0 after repair."""
        r = _make_result(passed_tests=False, final_passed_tests=True)
        assert compute_pass_at_1_after_repair([r]) == 1.0

    def test_repair_fails(self):
        """failed + repair_failed → 0.0 after repair."""
        r = _make_result(passed_tests=False, final_passed_tests=False)
        assert compute_pass_at_1_after_repair([r]) == 0.0

    def test_mixed(self):
        """2 pass, 1 repair-success, 1 repair-fail out of 4 → 0.75."""
        results = [
            _make_result(passed_tests=True, final_passed_tests=True),
            _make_result(passed_tests=True, final_passed_tests=True),
            _make_result(passed_tests=False, final_passed_tests=True),
            _make_result(passed_tests=False, final_passed_tests=False),
        ]
        assert compute_pass_at_1_after_repair(results) == 0.75

    def test_empty_list(self):
        """Empty list → 0.0."""
        assert compute_pass_at_1_after_repair([]) == 0.0

    def test_all_final_pass(self):
        """All final_passed_tests=True → 1.0."""
        results = [_make_result(final_passed_tests=True) for _ in range(3)]
        assert compute_pass_at_1_after_repair(results) == 1.0


# ---------------------------------------------------------------------------
# build_humaneval_artifact tests (SCENARIO-BENCH-011)
# ---------------------------------------------------------------------------


class TestBuildHumanevalArtifact:
    """Tests for build_humaneval_artifact."""

    def test_schema_present(self):
        """Artifact must have humaneval_schema='carnot.humaneval_benchmark.v1'."""
        r = _make_result(passed_tests=True, final_passed_tests=True)
        artifact = build_humaneval_artifact([r], "simulated")
        assert artifact["humaneval_schema"] == "carnot.humaneval_benchmark.v1"

    def test_inference_mode_embedded(self):
        """inference_mode is passed through to artifact."""
        r = _make_result()
        art_live = build_humaneval_artifact([r], "live_gpu")
        art_sim = build_humaneval_artifact([r], "simulated")
        assert art_live["inference_mode"] == "live_gpu"
        assert art_sim["inference_mode"] == "simulated"

    def test_positive_improvement_label(self):
        """Improvement > 0 → headline_label='code_verification_positive'."""
        # before=0.5, after=0.75 → improvement=0.25
        results = [
            _make_result(passed_tests=True, final_passed_tests=True),
            _make_result(passed_tests=True, final_passed_tests=True),
            _make_result(passed_tests=False, final_passed_tests=True),
            _make_result(passed_tests=False, final_passed_tests=False),
        ]
        art = build_humaneval_artifact(results, "live_gpu")
        assert art["headline_label"] == "code_verification_positive"
        assert art["headline_improvement"] > 0

    def test_no_improvement_label(self):
        """Improvement <= 0 → headline_label='no_improvement'."""
        # before=1.0, after=0.5 (impossible in practice but valid for spec)
        results = [
            _make_result(passed_tests=True, final_passed_tests=True),
            _make_result(passed_tests=True, final_passed_tests=False),
        ]
        art = build_humaneval_artifact(results, "simulated")
        assert art["headline_label"] == "no_improvement"

    def test_zero_improvement_no_label(self):
        """Improvement=0 → headline_label='no_improvement'."""
        results = [_make_result(passed_tests=True, final_passed_tests=True)]
        art = build_humaneval_artifact(results, "simulated")
        assert art["headline_improvement"] == 0.0
        assert art["headline_label"] == "no_improvement"

    def test_n_problems(self):
        """n_problems matches input list length."""
        results = [_make_result() for _ in range(10)]
        art = build_humaneval_artifact(results, "simulated")
        assert art["n_problems"] == 10

    def test_per_problem_results_serializable(self):
        """per_problem_results is JSON-serializable."""
        results = [_make_result() for _ in range(3)]
        art = build_humaneval_artifact(results, "simulated")
        json.dumps(art)  # must not raise

    def test_repair_counts(self):
        """n_repair_attempted and n_repair_succeeded are correct."""
        results = [
            _make_result(passed_tests=False, final_passed_tests=True, repair_attempted=True),
            _make_result(passed_tests=False, final_passed_tests=False, repair_attempted=True),
            _make_result(passed_tests=True, final_passed_tests=True, repair_attempted=False),
        ]
        art = build_humaneval_artifact(results, "simulated")
        assert art["n_repair_attempted"] == 2
        assert art["n_repair_succeeded"] == 1

    def test_total_violations(self):
        """total_violations_found sums violations_found across all results."""
        results = [
            _make_result(violations_found=3),
            _make_result(violations_found=1),
            _make_result(violations_found=0),
        ]
        art = build_humaneval_artifact(results, "simulated")
        assert art["total_violations_found"] == 4


# ---------------------------------------------------------------------------
# _run_tests helper tests
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests execution helper."""

    def test_correct_code_passes(self):
        """A correct implementation passes all test cases."""
        code = "def add(a, b):\n    return a + b\n"
        assert _run_tests(code, "add", [([1, 2], 3), ([0, 0], 0)]) is True

    def test_wrong_code_fails(self):
        """Wrong output → False."""
        code = "def add(a, b):\n    return a - b\n"
        assert _run_tests(code, "add", [([1, 2], 3)]) is False

    def test_syntax_error_fails(self):
        """Syntax error in code → False."""
        code = "def add(a, b) return a + b"
        assert _run_tests(code, "add", [([1, 2], 3)]) is False

    def test_missing_function_fails(self):
        """Missing entry_point function → False."""
        code = "def wrong_name(a, b):\n    return a + b\n"
        assert _run_tests(code, "add", [([1, 2], 3)]) is False

    def test_runtime_exception_fails(self):
        """RuntimeError during execution → False."""
        code = "def add(a, b):\n    raise ValueError('oops')\n"
        assert _run_tests(code, "add", [([1, 2], 3)]) is False

    def test_empty_test_cases_passes(self):
        """Empty test cases → True (vacuously)."""
        code = "def add(a, b):\n    return a + b\n"
        assert _run_tests(code, "add", []) is True


# ---------------------------------------------------------------------------
# _extract_code tests
# ---------------------------------------------------------------------------


class TestExtractCode:
    """Tests for the markdown fence stripper."""

    def test_python_fence(self):
        """```python ... ``` fence is stripped."""
        raw = "```python\ndef f(): pass\n```"
        assert _extract_code(raw) == "def f(): pass"

    def test_plain_fence(self):
        """``` ... ``` fence is stripped."""
        raw = "```\ndef f(): pass\n```"
        assert _extract_code(raw) == "def f(): pass"

    def test_no_fence(self):
        """No fence → returned as-is (stripped)."""
        raw = "def f(): pass"
        assert _extract_code(raw) == "def f(): pass"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        raw = "  def f(): pass  "
        assert _extract_code(raw) == "def f(): pass"


# ---------------------------------------------------------------------------
# _simulated_solution tests
# ---------------------------------------------------------------------------


class TestSimulatedSolution:
    """Tests for the CI-mode simulated solution generator."""

    def test_returns_string(self):
        """Always returns a string."""
        rng = random.Random(1)
        problem = _simple_problem()
        sol = _simulated_solution(problem, rng=rng)
        assert isinstance(sol, str)

    def test_contains_prompt(self):
        """Output always includes the problem prompt."""
        rng = random.Random(42)
        problem = _simple_problem()
        sol = _simulated_solution(problem, rng=rng)
        assert problem["prompt"] in sol

    def test_canonical_path_used_sometimes(self):
        """Over many RNG seeds, at least some outputs use the canonical solution."""
        problem = _simple_problem()
        canonical_seen = False
        for seed in range(100):
            rng = random.Random(seed)
            sol = _simulated_solution(problem, rng=rng)
            if problem["canonical_solution"] in sol:
                canonical_seen = True
                break
        assert canonical_seen

    def test_buggy_path_used_sometimes(self):
        """Over many RNG seeds, at least some outputs are deliberately different."""
        problem = _simple_problem()
        buggy_seen = False
        for seed in range(100):
            rng = random.Random(seed)
            sol = _simulated_solution(problem, rng=rng)
            if problem["canonical_solution"] not in sol:
                buggy_seen = True
                break
        assert buggy_seen


# ---------------------------------------------------------------------------
# _manual_problems tests
# ---------------------------------------------------------------------------


class TestManualProblems:
    """Tests for the 50 manually-crafted HumanEval-style problems."""

    def test_count(self):
        """Exactly 50 problems are returned."""
        problems = _manual_problems()
        assert len(problems) == 50

    def test_required_keys(self):
        """Every problem has required keys."""
        for p in _manual_problems():
            assert "task_id" in p
            assert "entry_point" in p
            assert "prompt" in p
            assert "canonical_solution" in p
            assert "test_cases" in p

    def test_test_cases_non_empty(self):
        """Every problem has at least one test case."""
        for p in _manual_problems():
            assert len(p["test_cases"]) >= 1, f"{p['task_id']} has no test cases"

    def test_canonical_solutions_pass(self):
        """Canonical solutions pass their own test cases."""
        failures = []
        for p in _manual_problems():
            code = p["prompt"] + p["canonical_solution"]
            ok = _run_tests(code, p["entry_point"], p["test_cases"])
            if not ok:
                failures.append(p["task_id"])
        assert failures == [], f"Canonical solutions failed: {failures}"


# ---------------------------------------------------------------------------
# _process_problem tests (simulated mode)
# ---------------------------------------------------------------------------


class TestProcessProblem:
    """Tests for the per-problem pipeline in simulated mode."""

    def test_returns_humaneval_result(self):
        """_process_problem returns a HumanEvalResult."""
        rng = random.Random(0)
        problem = _simple_problem()
        result = _process_problem(problem, live_model_state={"live": False}, rng=rng)
        assert isinstance(result, HumanEvalResult)

    def test_problem_id_preserved(self):
        """result.problem_id matches input task_id."""
        rng = random.Random(0)
        problem = _simple_problem()
        result = _process_problem(problem, live_model_state={"live": False}, rng=rng)
        assert result.problem_id == problem["task_id"]

    def test_passing_code_no_repair(self):
        """When generated code passes tests, repair_attempted=False."""
        # Force canonical path by patching _simulated_solution
        def canonical_rng(*args, **kwargs):
            return _simulated_solution(args[0], rng=random.Random(7))

        problem = _simple_problem()
        # Use a rng seed known to produce canonical code
        for seed in range(200):
            rng = random.Random(seed)
            sol = _simulated_solution(problem, rng=rng)
            if _run_tests(sol, problem["entry_point"], problem["test_cases"]):
                break

        # Now find a seed that gives a passing solution and use it
        for seed in range(200):
            rng_test = random.Random(seed)
            result = _process_problem(problem, live_model_state={"live": False}, rng=rng_test)
            if result.passed_tests:
                assert result.repair_attempted is False
                assert result.final_passed_tests is True
                break

    def test_failing_code_triggers_repair(self):
        """When generated code fails, repair_attempted=True."""
        for seed in range(200):
            rng = random.Random(seed)
            problem = _simple_problem()
            result = _process_problem(problem, live_model_state={"live": False}, rng=rng)
            if not result.passed_tests:
                assert result.repair_attempted is True
                break

    def test_result_serializable(self):
        """Result is JSON-serializable via asdict."""
        rng = random.Random(42)
        problem = _simple_problem()
        result = _process_problem(problem, live_model_state={"live": False}, rng=rng)
        json.dumps(asdict(result))


# ---------------------------------------------------------------------------
# Main integration test (SCENARIO-BENCH-011 CI mode)
# ---------------------------------------------------------------------------


class TestMainCIMode:
    """Tests for main() in CI mode (CARNOT_FORCE_LIVE=0)."""

    def test_main_produces_artifact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """main() writes artifact with correct schema in simulated mode."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))

        # Limit to 5 problems for speed
        mini_problems = _manual_problems()[:5]
        monkeypatch.setattr(_mod, "_load_problems", lambda: mini_problems)

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_341_live_humaneval.json"
        assert artifact_path.exists(), "Artifact file not written"

        artifact = json.loads(artifact_path.read_text())
        assert artifact["humaneval_schema"] == "carnot.humaneval_benchmark.v1"
        assert artifact["inference_mode"] == "simulated"
        assert artifact["n_problems"] == 5
        assert "pass_at_1_before_repair" in artifact
        assert "pass_at_1_after_repair" in artifact
        assert "headline_improvement" in artifact
        assert "headline_label" in artifact

    def test_artifact_has_required_experiment_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """main() artifact includes ExperimentTemplate required fields."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        mini_problems = _manual_problems()[:5]
        monkeypatch.setattr(_mod, "_load_problems", lambda: mini_problems)

        _mod.main()

        artifact_path = tmp_path / "results" / "experiment_341_live_humaneval.json"
        artifact = json.loads(artifact_path.read_text())

        for key in ["experiment", "status", "run_date", "started_at", "finished_at"]:
            assert key in artifact, f"Missing required field: {key}"

        assert artifact["experiment"] == 341
        assert artifact["status"] == "success"

    def test_simulated_inference_mode_in_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """In CI mode, artifact must have inference_mode='simulated'."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        monkeypatch.setattr(_mod, "_load_problems", lambda: _manual_problems()[:3])

        _mod.main()

        artifact = json.loads(
            (tmp_path / "results" / "experiment_341_live_humaneval.json").read_text()
        )
        assert artifact["inference_mode"] == "simulated"
