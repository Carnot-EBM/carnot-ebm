"""Tests for `carnot.pipeline.spec_code_verifier`.

Spec: REQ-CODE-025, REQ-CODE-026, REQ-CODE-027,
SCENARIO-CODE-022, SCENARIO-CODE-023, SCENARIO-CODE-024, SCENARIO-CODE-025
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from carnot.pipeline.pbt_code_verifier import PBTCodeVerificationResult
from carnot.pipeline.spec_code_verifier import (
    CodeSpecRow,
    ExplicitSpecClause,
    RepairHint,
    SpecClauseResult,
    SpecCodeVerificationResult,
    SpecCodeVerifier,
    _build_spec_harness,
    _build_spec_prompt,
    _default_harness,
    _default_pbt,
    _example_clause_result,
    _example_consistency_result,
    _parse_example_hint,
    _parse_family,
    _property_clause_result,
    _sample_inputs,
    _unique_examples,
    load_code_spec_row,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline

if TYPE_CHECKING:
    from pathlib import Path


def _write_corpus(path: Path) -> Path:
    row = {
        "row_id": "exp236-humaneval-999",
        "schema_version": "carnot.code_spec_corpus.v1",
        "run_date": "20260413",
        "task_id": "HumanEval/999",
        "case_id": "humaneval-999",
        "entry_point": "sort_numbers",
        "signature": "sort_numbers(nums: list[int]) -> list[int]",
        "preconditions": [
            {
                "kind": "declared_arity",
                "text": "call uses the declared 1 positional input(s)",
                "sources": ["signature"],
                "trace_refs": [],
            }
        ],
        "postconditions": [
            {
                "kind": "example_consistency",
                "text": "behavior remains consistent with the explicit examples",
                "sources": ["docstring_example", "official_test"],
                "trace_refs": [],
            },
            {
                "kind": "sorted_output",
                "text": "output is an ordered permutation of the primary sequence input",
                "sources": ["prompt_intent"],
                "trace_refs": ["exp226:humaneval-999"],
            },
        ],
        "invariants": [
            {
                "kind": "deterministic",
                "text": "repeated calls on the same input stay stable",
                "sources": ["verifier_default"],
                "trace_refs": ["exp226:humaneval-999"],
            },
            {
                "kind": "no_exception",
                "text": "admitted inputs execute without raising exceptions",
                "sources": ["verifier_default"],
                "trace_refs": ["exp226:humaneval-999"],
            },
        ],
        "mutation_constraints": [],
        "oracle_hints": [
            {
                "kind": "prompt_example",
                "text": "sort_numbers([3, 1, 2],) -> [1, 2, 3]",
                "sources": ["docstring_example"],
                "trace_refs": [],
            },
            {
                "kind": "official_test_example",
                "text": "sort_numbers([1, 2, 3],) -> [1, 2, 3]",
                "sources": ["official_test"],
                "trace_refs": [],
            },
            {
                "kind": "official_test_miss_trace",
                "text": "checked-in additive verification surfaced a harness-passing bug",
                "sources": ["trace_outcome"],
                "trace_refs": ["exp226:humaneval-999"],
            },
        ],
        "source_traces": [
            {
                "artifact": "results/experiment_226_results.json",
                "case_id": "humaneval-999",
                "experiment": 226,
                "failure_properties": ["sorted_output"],
                "model_name": "Gemma4-E4B-it",
                "official_test_miss": True,
                "repair_iterations": 1,
                "repaired": True,
                "source_ref": "exp226:humaneval-999",
            }
        ],
        "trace_summary": {
            "artifacts": ["results/experiment_226_results.json"],
            "failure_properties": ["sorted_output"],
            "official_test_miss_trace_count": 1,
            "repaired_trace_count": 1,
            "source_refs": ["exp226:humaneval-999"],
            "source_trace_count": 1,
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _write_learning_artifact(path: Path) -> Path:
    payload = {
        "experiment": 999,
        "cohort": {
            "cases": [
                {
                    "case_id": "humaneval-999",
                    "task_id": "HumanEval/999",
                    "prompt": "prompt",
                    "entry_point": "sort_numbers",
                }
            ]
        },
        "per_problem_results": [
            {
                "case_id": "humaneval-999",
                "task_id": "HumanEval/999",
                "entry_point": "sort_numbers",
                "baseline": {
                    "passed": True,
                    "accepted": False,
                    "official_test_miss_caught_by_pbt": True,
                    "pbt_derived_properties": [
                        {
                            "name": "sorted_output",
                            "source": "prompt_intent",
                            "description": "sorted output",
                        }
                    ],
                    "pbt_failure_records": [
                        {
                            "property_name": "sorted_output",
                            "source": "prompt_intent",
                            "description": "sorted output",
                            "input_args": [[2, 1]],
                            "actual": "[2, 1]",
                            "expected": "[1, 2]",
                            "error": None,
                        }
                    ],
                },
                "verify_repair": {
                    "repaired": True,
                    "n_repairs": 1,
                },
                "history": [
                    {
                        "iteration": 0,
                        "detected": True,
                        "accepted": False,
                        "harness": {
                            "passed": True,
                            "error_message": "",
                        },
                        "pbt": {
                            "failure_records": [
                                {
                                    "property_name": "sorted_output",
                                    "source": "prompt_intent",
                                    "error": None,
                                }
                            ]
                        },
                    },
                    {
                        "iteration": 1,
                        "detected": False,
                        "accepted": True,
                        "harness": {
                            "passed": True,
                            "error_message": "",
                        },
                        "pbt": {
                            "failure_records": [],
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_PROMPT = (
    "def sort_numbers(nums: list[int]) -> list[int]:\n"
    '    """Return numbers sorted in ascending order."""\n'
)
_OFFICIAL_TESTS = (
    "def check(candidate):\n"
    "    assert candidate([]) == []\n"
    "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
)
_BUGGY_CODE = "def sort_numbers(nums: list[int]) -> list[int]:\n    return nums\n"


def test_load_code_spec_row_preserves_grouped_families_and_run_date(tmp_path: Path) -> None:
    """SCENARIO-CODE-022: task lookup returns the matching explicit spec row."""
    corpus_path = _write_corpus(tmp_path / "code_spec_corpus.jsonl")

    spec = load_code_spec_row(task_id="HumanEval/999", corpus_path=corpus_path)

    assert spec is not None
    assert spec.task_id == "HumanEval/999"
    assert spec.case_id == "humaneval-999"
    assert spec.run_date == "20260413"
    assert spec.postconditions[0].kind == "example_consistency"
    assert spec.postconditions[1].kind == "sorted_output"
    assert spec.oracle_hints[0].kind == "prompt_example"


def test_spec_verifier_aggregates_harness_pbt_and_explicit_spec_statuses(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-023: one result carries harness, PBT, and explicit spec sections."""
    corpus_path = _write_corpus(tmp_path / "code_spec_corpus.jsonl")
    learning_path = _write_learning_artifact(tmp_path / "learning.json")
    verifier = SpecCodeVerifier(
        spec_corpus_path=corpus_path,
        learning_artifact_paths=(learning_path,),
    )

    result = verifier.verify(
        _BUGGY_CODE,
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
        task_id="HumanEval/999",
    )

    status_by_clause = {(item.family, item.kind): item for item in result.spec_clause_results}
    constraint_types = [constraint.constraint_type for constraint in result.to_constraint_results()]

    assert result.verified is False
    assert result.harness.passed is True
    assert result.pbt.verified is False
    assert status_by_clause[("postconditions", "example_consistency")].status == "violated"
    assert status_by_clause[("postconditions", "sorted_output")].status == "violated"
    assert status_by_clause[("oracle_hints", "prompt_example")].status == "violated"
    assert status_by_clause[("oracle_hints", "official_test_example")].status == "satisfied"
    assert status_by_clause[("oracle_hints", "official_test_miss_trace")].status == "not_checked"
    assert "pbt_code" in constraint_types
    assert "spec_code" in constraint_types
    assert result.to_certificate()["spec_summary"]["corpus_run_date"] == "20260413"


def test_spec_verifier_ranks_repair_hints_from_checked_in_learning(tmp_path: Path) -> None:
    """SCENARIO-CODE-024: learned repair hints prefer the trace-backed ordering fix."""
    corpus_path = _write_corpus(tmp_path / "code_spec_corpus.jsonl")
    learning_path = _write_learning_artifact(tmp_path / "learning.json")
    verifier = SpecCodeVerifier(
        spec_corpus_path=corpus_path,
        learning_artifact_paths=(learning_path,),
    )

    result = verifier.verify(
        _BUGGY_CODE,
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
        task_id="HumanEval/999",
    )

    assert result.repair_hints
    assert result.repair_hints[0].strategy_name == "ordering_fix"
    assert "sorted_output" in result.repair_hints[0].supporting_properties
    assert result.repair_hints[0].support_case_ids == ("humaneval-999",)
    assert result.repair_hints[0].score >= result.repair_hints[-1].score


def test_pipeline_opt_in_spec_verifier_preserves_legacy_generated_code_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-025: spec-aware metadata only appears on the opt-in pipeline path."""
    corpus_path = _write_corpus(tmp_path / "code_spec_corpus.jsonl")
    learning_path = _write_learning_artifact(tmp_path / "learning.json")
    pipeline = VerifyRepairPipeline()

    legacy = pipeline.verify_generated_code(
        _BUGGY_CODE,
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
    )
    opted_in = pipeline.verify_generated_code_with_specs(
        _BUGGY_CODE,
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
        task_id="HumanEval/999",
        spec_corpus_path=corpus_path,
        learning_artifact_paths=(learning_path,),
    )

    assert "spec_summary" not in legacy.certificate
    assert opted_in.certificate["spec_summary"]["task_id"] == "HumanEval/999"
    assert opted_in.certificate["repair_ranking"]["hints"][0]["strategy_name"] == "ordering_fix"
    assert opted_in.certificate["pbt_summary"]["enabled"] is True
    assert any(violation.constraint_type == "spec_code" for violation in opted_in.violations)


def test_helper_branches_cover_lookup_example_and_clause_fallbacks(tmp_path: Path) -> None:
    """REQ-CODE-025: helper fallbacks stay deterministic on sparse or invalid inputs."""
    corpus_path = _write_corpus(tmp_path / "code_spec_corpus.jsonl")

    by_case = load_code_spec_row(case_id="humaneval-999", corpus_path=corpus_path)
    by_entry = load_code_spec_row(entry_point="sort_numbers", corpus_path=corpus_path)
    missing = load_code_spec_row(
        task_id="HumanEval/missing",
        corpus_path=tmp_path / "missing.jsonl",
    )
    invalid_family = _parse_family({"preconditions": "bad"}, "preconditions")
    missing_arrow = _parse_example_hint("sort_numbers([1, 2, 3])", "sort_numbers")
    bad_syntax = _parse_example_hint("sort_numbers([1, 2,)", "sort_numbers")
    syntax_with_arrow = _parse_example_hint("sort_numbers([1, 2,) -> [1, 2]", "sort_numbers")
    wrong_name = _parse_example_hint("wrong_name([1]) -> [1]", "sort_numbers")
    bad_literal = _parse_example_hint("sort_numbers([1]) -> object()", "sort_numbers")
    parsed_family = _parse_family(
        {
            "preconditions": [
                None,
                {"kind": "declared_arity", "text": "one arg", "sources": [], "trace_refs": []},
            ]
        },
        "preconditions",
    )

    prompt_clause = ExplicitSpecClause(
        family="oracle_hints",
        kind="prompt_example",
        text="sort_numbers([1, 2, 3])",
        sources=("docstring_example",),
        trace_refs=(),
    )
    error_clause = ExplicitSpecClause(
        family="oracle_hints",
        kind="prompt_example",
        text="sort_numbers([3, 2, 1],) -> [1, 2, 3]",
        sources=("docstring_example",),
        trace_refs=(),
    )
    unknown_property = ExplicitSpecClause(
        family="postconditions",
        kind="mystery_property",
        text="unknown property",
        sources=("prompt_intent",),
        trace_refs=(),
    )
    sorted_property = ExplicitSpecClause(
        family="postconditions",
        kind="sorted_output",
        text="sorted output",
        sources=("prompt_intent",),
        trace_refs=(),
    )
    consistency_clause = ExplicitSpecClause(
        family="postconditions",
        kind="example_consistency",
        text="examples remain consistent",
        sources=("docstring_example",),
        trace_refs=(),
    )

    unparseable = _example_clause_result(
        prompt_clause,
        code=_BUGGY_CODE,
        entry_point="sort_numbers",
    )
    runtime_error = _example_clause_result(
        error_clause,
        code="def sort_numbers(nums: list[int]) -> list[int]:\n    raise RuntimeError('boom')\n",
        entry_point="sort_numbers",
    )
    property_unknown = _property_clause_result(
        unknown_property,
        checked_properties=set(),
        failed_properties=(),
    )
    property_satisfied = _property_clause_result(
        sorted_property,
        checked_properties={"sorted_output"},
        failed_properties=(),
    )
    property_unchecked = _property_clause_result(
        sorted_property,
        checked_properties=set(),
        failed_properties=(),
    )
    no_examples = _example_consistency_result(consistency_clause, example_results=[])
    satisfied_examples = _example_consistency_result(
        consistency_clause,
        example_results=(
            SpecClauseResult(
                family="oracle_hints",
                kind="prompt_example",
                text="sort_numbers([1, 2, 3],) -> [1, 2, 3]",
                status="satisfied",
                checked_by="example",
                detail="",
                sources=("docstring_example",),
                trace_refs=(),
                matched_properties=("example_consistency",),
            ),
        ),
    )

    assert by_case is not None
    assert by_entry is not None
    assert by_case.clause_for_kind("missing") is None
    assert missing is None
    assert invalid_family == ()
    assert missing_arrow is None
    assert bad_syntax is None
    assert syntax_with_arrow is None
    assert wrong_name is None
    assert bad_literal is None
    assert len(parsed_family) == 1
    assert unparseable.status == "not_checked"
    assert runtime_error.status == "violated"
    assert "boom" in runtime_error.detail
    assert property_unknown.status == "not_checked"
    assert property_satisfied.status == "satisfied"
    assert property_unchecked.status == "not_checked"
    assert no_examples.status == "not_checked"
    assert satisfied_examples.status == "satisfied"


def test_disabled_paths_and_manual_results_cover_fallback_and_official_failures(
    tmp_path: Path,
) -> None:
    """REQ-CODE-026, REQ-CODE-027: disabled modes and generic fallback hints stay structured."""
    verifier = SpecCodeVerifier(
        spec_corpus_path=tmp_path / "missing.jsonl",
        learning_artifact_paths=(tmp_path / "missing-learning.json",),
        include_official_tests=False,
        include_pbt=False,
    )

    result = verifier.verify(
        _BUGGY_CODE,
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
        task_id="HumanEval/999",
    )
    violated_clause = SpecClauseResult(
        family="postconditions",
        kind="sorted_output",
        text="sorted output",
        status="violated",
        checked_by="pbt",
        detail="sorted_output failed",
        sources=("prompt_intent",),
        trace_refs=(),
        matched_properties=("sorted_output",),
    )
    official_failure = SpecCodeVerificationResult(
        harness=type(_default_harness())(
            passed=False,
            error_type="failure",
            error_message="AssertionError: failed",
            stdout="traceback",
        ),
        pbt=_default_pbt(),
        spec=None,
        spec_clause_results=(),
        repair_hints=(),
    )
    spec_only_failure = SpecCodeVerificationResult(
        harness=_default_harness(),
        pbt=PBTCodeVerificationResult(),
        spec=None,
        spec_clause_results=(violated_clause,),
        repair_hints=(
            RepairHint(
                strategy_name="generic_repair",
                error_family="unknown",
                score=0.0,
                success_rate=0.0,
                attempts=0,
                partial_recoveries=0,
                supporting_properties=(),
                support_case_ids=(),
                rationale="fallback",
            ),
        ),
    )

    assert result.verified is True
    assert result.harness.error_type == "disabled"
    assert result.pbt.verified is True
    assert result.repair_hints[0].strategy_name == "generic_repair"
    assert official_failure.verified is False
    assert official_failure.to_constraint_results()[0].constraint_type == "official_tests"
    assert official_failure.to_certificate()["official_test_summary"]["passed"] is False
    assert spec_only_failure.verified is False
    assert spec_only_failure.to_constraint_results()[0].constraint_type == "spec_code"


def test_sparse_helpers_cover_blank_lines_duplicates_and_signature_fallbacks(
    tmp_path: Path,
) -> None:
    """REQ-CODE-025: sparse helper branches stay deterministic and parseable."""
    corpus_path = tmp_path / "sparse.jsonl"
    corpus_path.write_text(
        "\n"
        + json.dumps(
            {
                "row_id": "exp236-empty",
                "schema_version": "carnot.code_spec_corpus.v1",
                "run_date": "20260413",
                "task_id": "HumanEval/empty",
                "case_id": "humaneval-empty",
                "entry_point": "identity",
                "signature": "identity(nums)",
                "preconditions": [123],
                "postconditions": [],
                "invariants": [],
                "mutation_constraints": [],
                "oracle_hints": [],
                "source_traces": [],
                "trace_summary": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = load_code_spec_row(task_id="HumanEval/empty", corpus_path=corpus_path)
    assert spec is not None
    assert spec.clause_for_kind("missing") is None
    assert _parse_family({"preconditions": [123]}, "preconditions") == ()
    assert _parse_example_hint("identity([1, 2,) -> [1, 2]", "identity") is None
    assert _unique_examples([(([1],), [1]), (([1],), [1])]) == [(([1],), [1])]

    empty_spec = CodeSpecRow(
        task_id="HumanEval/blank",
        case_id="humaneval-blank",
        row_id="exp236-blank",
        run_date="20260413",
        schema_version="carnot.code_spec_corpus.v1",
        entry_point="identity",
        signature="identity(nums)",
        preconditions=(),
        postconditions=(),
        invariants=(),
        mutation_constraints=(),
        oracle_hints=(),
        source_traces=(),
        trace_summary={},
    )
    assert _build_spec_prompt(empty_spec, []) == "def identity(nums):\n    pass\n"
    assert _build_spec_harness(empty_spec, []) == "def check(candidate):\n    return None\n"
    assert _sample_inputs(
        code="def identity(nums: list[int]) -> list[int]:\n    return nums\n",
        prompt="not python",
        entry_point="identity",
        spec_prompt="not python",
        spec_harness="def check(candidate):\n    return None\n",
    ) == [([1, 2, 3],), ([],)]


def test_example_consistency_and_no_exception_branches_report_explicit_violations(
    tmp_path: Path,
) -> None:
    """REQ-CODE-026: explicit example consistency and no-exception clauses can both violate."""
    corpus_path = tmp_path / "explicit.jsonl"
    row = json.loads((_write_corpus(tmp_path / "base.jsonl")).read_text(encoding="utf-8").strip())
    row["postconditions"].append(
        {
            "kind": "example_consistency",
            "text": "examples remain consistent",
            "sources": ["docstring_example"],
            "trace_refs": [],
        }
    )
    corpus_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    verifier = SpecCodeVerifier(
        spec_corpus_path=corpus_path,
        learning_artifact_paths=(tmp_path / "missing-learning.json",),
        include_official_tests=False,
        include_pbt=False,
    )

    result = verifier.verify(
        "def sort_numbers(nums: list[int]) -> list[int]:\n    raise RuntimeError('boom')\n",
        _PROMPT,
        "sort_numbers",
        _OFFICIAL_TESTS,
        task_id="HumanEval/999",
    )
    status_by_clause = {(item.family, item.kind): item for item in result.spec_clause_results}

    assert status_by_clause[("postconditions", "example_consistency")].status == "violated"
    assert "boom" in status_by_clause[("postconditions", "example_consistency")].detail
    assert status_by_clause[("invariants", "no_exception")].status == "violated"
