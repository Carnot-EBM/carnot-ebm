"""Tests for the Exp 1742 SWE-Bench Lite harness helpers.

Spec: REQ-BENCH-1742, SCENARIO-BENCH-1742
"""

from __future__ import annotations

import json

from carnot.pipeline.swebench_harness import (
    DEFAULT_TARGET_INSTANCE_IDS,
    PatchEvaluation,
    build_results_payload,
    build_swebench_prompt,
    extract_unified_diff,
    load_swebench_lite_problems,
    run_model_on_problems,
    run_verify_repair_case,
    summarize_model_results,
    verify_patch_candidate,
)


def _row(instance_id: str, repo: str = "django/django") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "patch": "diff --git a/pkg/mod.py b/pkg/mod.py\n",
        "test_patch": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
        "problem_statement": f"Fix issue for {instance_id}",
        "hints_text": "Use the smallest fix.",
        "created_at": "2026-01-01T00:00:00Z",
        "version": "1.0",
        "FAIL_TO_PASS": json.dumps(["tests/test_bug.py::test_regression"]),
        "PASS_TO_PASS": json.dumps(["tests/test_existing.py::test_still_passes"]),
        "environment_setup_commit": "def456",
    }


def _passing_diff(return_value: int = 2) -> str:
    return (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def value():\n"
        "-    return 1\n"
        "+def value():\n"
        f"+    return {return_value}\n"
    )


def test_load_swebench_lite_problems_selects_target_ids_in_order() -> None:
    """REQ-BENCH-1742: targeted SWE-Bench Lite rows are normalized deterministically."""
    rows = [
        _row(DEFAULT_TARGET_INSTANCE_IDS[2], "matplotlib/matplotlib"),
        _row("django__django-99999", "django/django"),
        _row(DEFAULT_TARGET_INSTANCE_IDS[0], "django/django"),
        _row(DEFAULT_TARGET_INSTANCE_IDS[1], "sympy/sympy"),
    ]

    problems = load_swebench_lite_problems(rows=rows, limit=3)

    assert [problem.instance_id for problem in problems] == list(DEFAULT_TARGET_INSTANCE_IDS[:3])
    assert [problem.dataset_idx for problem in problems] == [2, 3, 0]
    assert problems[0].fail_to_pass == ["tests/test_bug.py::test_regression"]
    assert problems[0].pass_to_pass == ["tests/test_existing.py::test_still_passes"]


def test_load_swebench_lite_problems_falls_back_to_viewer_fetcher() -> None:
    """SCENARIO-BENCH-1742: fetching is injectable when datasets is unavailable."""
    rows = [_row(DEFAULT_TARGET_INSTANCE_IDS[0])]

    def missing_dataset_loader(_split: str) -> list[dict[str, object]]:
        raise ImportError("datasets unavailable")

    def viewer_fetcher(split: str) -> list[dict[str, object]]:
        assert split == "test"
        return rows

    problems = load_swebench_lite_problems(
        dataset_loader=missing_dataset_loader,
        viewer_fetcher=viewer_fetcher,
        limit=1,
    )

    assert [problem.instance_id for problem in problems] == [DEFAULT_TARGET_INSTANCE_IDS[0]]


def test_load_swebench_lite_problems_handles_nonstandard_test_lists() -> None:
    """REQ-BENCH-1742: dataset test-list fields are normalized defensively."""
    row = _row(DEFAULT_TARGET_INSTANCE_IDS[0])
    row["FAIL_TO_PASS"] = ["tests/test_bug.py::test_from_list"]
    row["PASS_TO_PASS"] = "{}"
    row["hints_text"] = None
    row_without_tests = _row(DEFAULT_TARGET_INSTANCE_IDS[1])
    row_without_tests["FAIL_TO_PASS"] = None

    problems = load_swebench_lite_problems(rows=[row, row_without_tests], limit=2)
    problem = problems[0]

    assert problem.fail_to_pass == ["tests/test_bug.py::test_from_list"]
    assert problem.pass_to_pass == []
    assert problem.hints_text == "None"
    assert problems[1].fail_to_pass == []


def test_prompt_and_patch_extraction_keep_swebench_contract() -> None:
    """REQ-BENCH-1742: model prompts request unified diffs with EqM disabled."""
    problem = load_swebench_lite_problems(rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])], limit=1)[0]

    prompt = build_swebench_prompt(problem, eqm_decoding_enabled=False)
    patch = extract_unified_diff(
        "Here is the patch:\n```diff\n" + _passing_diff() + "```\nExtra commentary"
    )

    assert "SWE-Bench Lite" in prompt
    assert "EqM decoding enabled: false" in prompt
    assert "tests/test_bug.py::test_regression" in prompt
    assert patch.startswith("diff --git")
    assert "Extra commentary" not in patch


def test_verify_patch_candidate_rejects_empty_malformed_and_test_only_patches() -> None:
    """REQ-BENCH-1742: bounded Carnot patch checks block invalid candidates."""
    problem = load_swebench_lite_problems(rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])], limit=1)[0]

    empty = verify_patch_candidate(problem, "")
    malformed = verify_patch_candidate(problem, "return 2")
    test_only = verify_patch_candidate(
        problem,
        "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
        "--- a/tests/test_bug.py\n"
        "+++ b/tests/test_bug.py\n"
        "@@ -1 +1 @@\n"
        "-assert False\n"
        "+assert True\n",
    )
    accepted = verify_patch_candidate(problem, _passing_diff())

    assert empty.accepted is False
    assert empty.violations == ["empty_patch"]
    assert malformed.accepted is False
    assert "missing_unified_diff_header" in malformed.violations
    assert test_only.accepted is False
    assert "test_only_patch" in test_only.violations
    assert accepted.accepted is True
    assert accepted.violations == []
    assert accepted.n_constraints == 4


def test_run_verify_repair_case_uses_evaluator_feedback_for_repair() -> None:
    """SCENARIO-BENCH-1742: verify-repair records baseline failure and repaired success."""
    problem = load_swebench_lite_problems(rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])], limit=1)[0]
    outputs = iter([_passing_diff(return_value=1), _passing_diff(return_value=2)])
    prompts: list[str] = []

    def generator(prompt: str, *, model_name: str, eqm_decoding_enabled: bool) -> str:
        assert model_name == "Qwen3.6-35B-A3B"
        assert eqm_decoding_enabled is False
        prompts.append(prompt)
        return next(outputs)

    def evaluator(problem_arg, patch: str, model_name: str) -> PatchEvaluation:
        assert problem_arg == problem
        assert model_name == "Qwen3.6-35B-A3B"
        resolved = "return 2" in patch
        return PatchEvaluation(
            resolved=resolved,
            status="complete",
            fail_to_pass_passed=resolved,
            pass_to_pass_passed=True,
            error_type="none" if resolved else "test_failure",
            error_message="" if resolved else "tests/test_bug.py::test_regression failed",
            stdout="ok" if resolved else "failed",
            report_path="reports/case.json",
        )

    case = run_verify_repair_case(
        problem,
        model_name="Qwen3.6-35B-A3B",
        generator=generator,
        evaluator=evaluator,
        max_repairs=1,
        eqm_decoding_enabled=False,
    )

    assert case["baseline"]["resolved"] is False
    assert case["verify_repair"]["resolved"] is True
    assert case["verify_repair"]["repaired"] is True
    assert case["verify_repair"]["n_repairs"] == 1
    assert len(case["attempts"]) == 2
    assert "tests/test_bug.py::test_regression failed" in prompts[1]


def test_run_verify_repair_case_blocks_malformed_generation_without_evaluation() -> None:
    """REQ-BENCH-1742: malformed baseline output is recorded honestly."""
    problem = load_swebench_lite_problems(rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])], limit=1)[0]
    eval_calls = 0

    def generator(prompt: str, *, model_name: str, eqm_decoding_enabled: bool) -> str:
        del prompt, model_name, eqm_decoding_enabled
        return "not a diff"

    def evaluator(problem_arg, patch: str, model_name: str) -> PatchEvaluation:
        del problem_arg, patch, model_name
        nonlocal eval_calls
        eval_calls += 1
        return PatchEvaluation(resolved=True, status="complete")

    case = run_verify_repair_case(
        problem,
        model_name="Gemma4-31B-it",
        generator=generator,
        evaluator=evaluator,
        max_repairs=0,
        eqm_decoding_enabled=False,
    )

    assert eval_calls == 0
    assert case["baseline"]["evaluation_status"] == "blocked"
    assert case["baseline"]["verification"]["accepted"] is False
    assert case["verify_repair"]["resolved"] is False


def test_model_summary_and_payload_schema() -> None:
    """REQ-BENCH-1742: result payload reports baseline and verify-repair metrics."""
    problems = load_swebench_lite_problems(rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])], limit=1)

    def generator(prompt: str, *, model_name: str, eqm_decoding_enabled: bool) -> str:
        del prompt, model_name, eqm_decoding_enabled
        return _passing_diff(return_value=2)

    def evaluator(problem_arg, patch: str, model_name: str) -> PatchEvaluation:
        del problem_arg, patch, model_name
        return PatchEvaluation(resolved=True, status="complete", report_path="report.json")

    model_result = run_model_on_problems(
        problems,
        model_spec={"name": "Qwen3.6-35B-A3B", "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        generator=generator,
        evaluator=evaluator,
        max_repairs=1,
        eqm_decoding_enabled=False,
    )
    metrics = summarize_model_results([model_result])
    payload = build_results_payload(
        status="complete",
        honest_verdict="baseline_complete",
        timestamp="2026-05-10T00:00:00Z",
        runtime_seconds=1.25,
        selected_problems=problems,
        model_results=[model_result],
        metrics=metrics,
        blockers=[],
        evaluator_backend="injected",
        eqm_decoding_enabled=False,
    )

    assert metrics["n_models"] == 1
    assert metrics["n_instances"] == 1
    assert metrics["baseline_resolve_rate"] == 1.0
    assert metrics["verify_repair_resolve_rate"] == 1.0
    assert payload["experiment_id"] == "1742"
    assert payload["spec_refs"] == ["REQ-BENCH-1742", "SCENARIO-BENCH-1742"]
    assert payload["config"]["eqm_decoding_enabled"] is False
    assert payload["dataset"]["selected_instance_ids"] == [DEFAULT_TARGET_INSTANCE_IDS[0]]
