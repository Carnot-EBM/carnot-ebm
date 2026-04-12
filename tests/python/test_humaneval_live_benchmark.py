"""Tests for the Exp 208 HumanEval live benchmark helpers.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import pytest
from carnot.pipeline.humaneval_live_benchmark import (
    HarnessResult,
    bootstrap_ci,
    bootstrap_delta_ci,
    build_candidate_code,
    build_repair_prompt,
    build_results_payload,
    execute_humaneval,
    generate_probe_inputs,
    run_instrumentation,
    sample_problems,
    summarize_cases,
)


def _problem(*, prompt: str, test: str, entry_point: str) -> dict[str, str]:
    return {
        "task_id": "HumanEval/0",
        "prompt": prompt,
        "test": test,
        "entry_point": entry_point,
    }


def test_sample_problems_uses_seeded_shuffle_first_n() -> None:
    """SCENARIO-VERIFY-006: problem sampling is deterministic for a fixed seed."""
    problems = [{"dataset_idx": idx} for idx in range(10)]

    sample = sample_problems(problems, sample_size=4, sample_seed=5)

    assert [problem["dataset_idx"] for problem in sample] == [2, 3, 1, 0]


def test_build_candidate_code_strips_wrappers_and_normalizes_indentation() -> None:
    """REQ-VERIFY-001: generated bodies become executable candidate code."""
    prompt = 'def add(a, b):\n    """Return a + b."""\n'
    raw_body = "```python\ndef add(a, b):\n        return a + b\n```"

    code = build_candidate_code(prompt, raw_body)

    assert code == ('def add(a, b):\n    """Return a + b."""\n    return a + b\n')


def test_build_candidate_code_uses_pass_for_empty_body() -> None:
    """REQ-VERIFY-001: empty model output still yields syntactically valid code."""
    prompt = "def noop():\n    pass\n"

    code = build_candidate_code(prompt, "```python\n```")

    assert code.endswith("    pass\n")


def test_execute_humaneval_reports_passing_harness() -> None:
    """REQ-VERIFY-003: passing code returns a successful harness result."""
    problem = _problem(
        prompt='def add(a, b):\n    """Return a + b."""\n',
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 4) == 3\n"
        ),
        entry_point="add",
    )
    code = build_candidate_code(problem["prompt"], "return a + b")

    result = execute_humaneval(code, problem, timeout=1.0)

    assert result.passed is True
    assert result.error_type == "none"
    assert result.error_message == ""


def test_execute_humaneval_reports_failed_harness() -> None:
    """REQ-VERIFY-003: failing code returns the captured harness error."""
    problem = _problem(
        prompt='def add(a, b):\n    """Return a + b."""\n',
        test=("def check(candidate):\n    assert candidate(1, 2) == 3\n"),
        entry_point="add",
    )
    code = build_candidate_code(problem["prompt"], "return a - b")

    result = execute_humaneval(code, problem, timeout=1.0)

    assert result.passed is False
    assert result.error_type == "failure"
    assert "AssertionError" in result.error_message


def test_execute_humaneval_reports_timeout() -> None:
    """REQ-VERIFY-003: non-terminating code is cut off by the subprocess timeout."""
    problem = _problem(
        prompt='def hang():\n    """Never return."""\n',
        test=("def check(candidate):\n    candidate()\n"),
        entry_point="hang",
    )
    code = build_candidate_code(problem["prompt"], "while True:\n    pass")

    result = execute_humaneval(code, problem, timeout=0.1)

    assert result.passed is False
    assert result.error_type == "timeout"
    assert "timeout" in result.error_message


def test_execute_humaneval_ignores_cleanup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-003: tempfile cleanup failure does not mask the harness result."""
    problem = _problem(
        prompt='def add(a, b):\n    """Return a + b."""\n',
        test=("def check(candidate):\n    assert candidate(1, 2) == 3\n"),
        entry_point="add",
    )
    code = build_candidate_code(problem["prompt"], "return a + b")

    def raise_unlink(_: str) -> None:
        raise OSError("already removed")

    monkeypatch.setattr(
        "carnot.pipeline.humaneval_live_benchmark.os.unlink",
        raise_unlink,
    )

    result = execute_humaneval(code, problem, timeout=1.0)

    assert result.passed is True


def test_generate_probe_inputs_uses_prompt_annotations() -> None:
    """REQ-VERIFY-002: instrumentation probes reflect the prompt's parameter types."""
    prompt = (
        "def demo(a: int, items: list[int], name: str, flag: bool, meta: dict) -> int:\n"
        '    """Exercise mixed argument types."""\n'
    )

    probes = generate_probe_inputs(prompt, "demo")

    assert probes == [
        {"a": 1, "items": [1, 2], "name": "x", "flag": True, "meta": {"key": 1}},
        {"a": 0, "items": [], "name": "", "flag": False, "meta": {}},
        {"a": -1, "items": [0], "name": " ", "flag": True, "meta": {"key": 0}},
    ]


def test_generate_probe_inputs_falls_back_to_empty_kwargs_for_zero_arg_function() -> None:
    """REQ-VERIFY-002: zero-argument functions still get a probe for instrumentation."""
    prompt = 'def demo() -> int:\n    """No args."""\n'

    assert generate_probe_inputs(prompt, "demo") == [{}]


def test_generate_probe_inputs_supports_float_and_tuple_and_ignores_self() -> None:
    """REQ-VERIFY-002: method-style prompts skip self and preserve float/tuple probes."""
    prompt = (
        "def demo(self, ratio: float, bounds: tuple[int, int]) -> int:\n"
        '    """Method-style signature."""\n'
    )

    probes = generate_probe_inputs(prompt, "demo")

    assert probes == [
        {"ratio": 1.0, "bounds": (1,)},
        {"ratio": 0.0, "bounds": ()},
        {"ratio": -1.0, "bounds": (0, 1)},
    ]


def test_run_instrumentation_reports_static_and_dynamic_violations() -> None:
    """REQ-VERIFY-002: CodeExtractor and Exp 53 instrumentation both feed repair feedback."""
    prompt = 'def broken(x: int) -> int:\n    """Return x plus a missing name."""\n'
    code = build_candidate_code(prompt, "return missing + x")

    result = run_instrumentation(code, prompt, "broken")

    assert result["detected"] is True
    assert result["n_constraints"] >= 2
    assert result["n_static_violations"] == 1
    assert result["n_dynamic_violations"] == 3
    assert "never assigned or passed as parameter" in result["static_violations"][0]
    assert "NameError" in result["dynamic_violations"][0]
    assert result["probe_inputs"][0] == {"x": 1}


def test_run_instrumentation_handles_syntax_error_code() -> None:
    """REQ-VERIFY-002: malformed code still yields bounded instrumentation feedback."""
    prompt = 'def broken(x: int) -> int:\n    """Broken code."""\n'
    code = "def broken(x: int) -> int:\n    return (\n"

    result = run_instrumentation(code, prompt, "broken")

    assert result["detected"] is True
    assert result["n_constraints"] == 0
    assert result["n_static_violations"] == 0
    assert result["n_dynamic_violations"] == 1
    assert "Code execution failed" in result["dynamic_violations"][0]


def test_build_repair_prompt_includes_test_and_instrumentation_feedback() -> None:
    """REQ-VERIFY-003: repair prompts include both harness and instrumentation signals."""
    prompt = 'def add(a, b):\n    """Return a + b."""\n'
    previous_body = "return a - b"
    harness = HarnessResult(
        passed=False,
        error_type="failure",
        error_message="AssertionError",
        stdout="trace",
    )
    instrumentation = {
        "constraint_feedback": ["add(): variable 'x' used but never assigned"],
        "dynamic_violations": ["NameError: name 'x' is not defined"],
    }

    repair_prompt = build_repair_prompt(
        prompt,
        previous_body,
        harness,
        instrumentation,
        repair_idx=1,
    )

    assert "repair attempt 2" in repair_prompt.lower()
    assert "AssertionError" in repair_prompt
    assert "variable 'x' used but never assigned" in repair_prompt
    assert "NameError" in repair_prompt
    assert "Write ONLY the corrected function body" in repair_prompt


def test_bootstrap_helpers_compute_point_estimates() -> None:
    """SCENARIO-VERIFY-006: summary statistics are deterministic for a fixed seed."""
    point, lo, hi = bootstrap_ci([True, True, False, False], n_bootstrap=256, seed=9)
    delta, delta_lo, delta_hi = bootstrap_delta_ci(
        [False, False, True, True],
        [True, False, True, True],
        n_bootstrap=256,
        seed=11,
    )

    assert point == 0.5
    assert lo <= point <= hi
    assert delta == 0.25
    assert delta_lo <= delta <= delta_hi


def test_summarize_cases_and_payload_report_pass_rates_and_metadata() -> None:
    """REQ-VERIFY-003: final payload reports paired baseline and verify-repair outcomes."""
    cases = [
        {
            "task_id": "HumanEval/1",
            "dataset_idx": 1,
            "baseline": {
                "passed": True,
                "n_static_violations": 0,
                "n_dynamic_violations": 0,
            },
            "verify_repair": {
                "passed": True,
                "repaired": False,
                "n_repairs": 0,
            },
        },
        {
            "task_id": "HumanEval/2",
            "dataset_idx": 2,
            "baseline": {
                "passed": False,
                "n_static_violations": 1,
                "n_dynamic_violations": 1,
            },
            "verify_repair": {
                "passed": True,
                "repaired": True,
                "n_repairs": 1,
            },
        },
        {
            "task_id": "HumanEval/3",
            "dataset_idx": 3,
            "baseline": {
                "passed": False,
                "n_static_violations": 0,
                "n_dynamic_violations": 2,
            },
            "verify_repair": {
                "passed": False,
                "repaired": False,
                "n_repairs": 2,
            },
        },
    ]

    summary = summarize_cases(cases, n_bootstrap=256, seed=13)

    assert summary["n_problems"] == 3
    assert summary["baseline"]["pass_at_1"] == pytest.approx(1 / 3)
    assert summary["verify_repair"]["pass_at_1"] == pytest.approx(2 / 3)
    assert summary["improvement"]["delta"] == pytest.approx(1 / 3)
    assert summary["repair_stats"]["n_problems_needing_repair"] == 2
    assert summary["repair_stats"]["n_repaired"] == 1
    assert summary["instrumentation"]["problems_with_static_violations"] == 1
    assert summary["instrumentation"]["problems_with_dynamic_violations"] == 2

    payload = build_results_payload(
        experiment=208,
        title="Exp 208",
        timestamp="2026-04-12T00:00:00Z",
        model_name="Gemma4-E4B-it",
        hf_id="google/gemma-4-E4B-it",
        device="cuda:0",
        inference_mode="live_gpu",
        sample_seed=208,
        sample_size=3,
        sample_dataset_indices=[1, 2, 3],
        sample_task_ids=["HumanEval/1", "HumanEval/2", "HumanEval/3"],
        max_new_tokens=512,
        max_repairs=3,
        runtime_seconds=12.5,
        statistics=summary,
        cases=cases,
    )

    assert payload["metadata"]["inference_mode"] == "live_gpu"
    assert payload["metadata"]["sample_size"] == 3
    assert payload["metadata"]["sample_task_ids"] == ["HumanEval/1", "HumanEval/2", "HumanEval/3"]
    assert payload["statistics"] == summary
    assert payload["per_problem_results"] == cases


def test_summarize_cases_rejects_empty_cohort() -> None:
    """REQ-VERIFY-003: final summary refuses an empty benchmark cohort."""
    with pytest.raises(ValueError, match="empty"):
        summarize_cases([])
