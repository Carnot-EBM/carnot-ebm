"""Tests for the packaged code-verification surfaces.

Spec coverage: REQ-CODE-019, REQ-CODE-022,
SCENARIO-CODE-016, SCENARIO-CODE-019
"""

from __future__ import annotations

from pathlib import Path

from carnot.pipeline import VerifyRepairPipeline, verify_code
from carnot.pipeline.humaneval_live_benchmark import (
    build_candidate_code,
    execute_humaneval,
)


def test_verify_code_api_uses_source_as_prompt_fallback() -> None:
    """SCENARIO-CODE-016: verify_code runs signature-derived PBT checks from source alone."""
    code = (
        "def increment_all(nums: list[int]) -> list[int]:\n"
        "    for index, value in enumerate(nums):\n"
        "        nums[index] = value + 1\n"
        "    return nums\n"
    )

    result = verify_code(code, entry_point="increment_all")

    assert result.verified is False
    assert result.certificate["pbt_summary"]["enabled"] is True
    assert result.certificate["pbt_summary"]["n_failures"] >= 1
    assert any(v.metadata.get("property_name") == "input_immutability" for v in result.violations)


def test_generate_verify_repair_workflow_reverifies_cleanly() -> None:
    """SCENARIO-CODE-019: generated code is verified, repaired, and re-verified."""
    problem = {
        "task_id": "HumanEval/0",
        "prompt": (
            "def sort_numbers(nums: list[int]) -> list[int]:\n"
            '    """Return numbers sorted in ascending order."""\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == []\n"
            "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
        ),
        "entry_point": "sort_numbers",
    }
    buggy_code = build_candidate_code(problem["prompt"], "return nums")
    repaired_code = build_candidate_code(problem["prompt"], "return sorted(nums)")

    buggy_harness = execute_humaneval(buggy_code, problem, timeout=1.0)
    buggy_result = verify_code(
        buggy_code,
        prompt=problem["prompt"],
        entry_point=problem["entry_point"],
        official_tests=problem["test"],
    )
    repair_feedback = VerifyRepairPipeline()._format_violations(buggy_result.violations)
    repaired_result = verify_code(
        repaired_code,
        prompt=problem["prompt"],
        entry_point=problem["entry_point"],
        official_tests=problem["test"],
    )
    repaired_harness = execute_humaneval(repaired_code, problem, timeout=1.0)

    assert buggy_harness.passed is True
    assert buggy_result.verified is False
    assert "sorted_output" in repair_feedback
    assert repaired_result.verified is True
    assert repaired_harness.passed is True


def test_docs_include_packaged_code_verification_examples() -> None:
    """REQ-CODE-022: docs include CLI, MCP, Python API, and repair workflow examples."""
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    usage = (docs_dir / "usage-guide.md").read_text()
    api_reference = (docs_dir / "api-reference.md").read_text()
    getting_started = (docs_dir / "getting-started.md").read_text()

    assert "carnot verify-code" in usage
    assert "verify_code_with_pbt" in usage
    assert "from carnot.pipeline import verify_code" in api_reference
    assert "generate-verify-repair" in getting_started.lower()
