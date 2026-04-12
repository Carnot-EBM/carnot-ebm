"""Tests for `carnot.pipeline.code_learning`.

Spec: REQ-CODE-016, REQ-CODE-017, REQ-CODE-018,
SCENARIO-CODE-014, SCENARIO-CODE-015
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline.code_learning import (
    PropertyRanker,
    RepairStrategy,
    TraceAnalyzer,
)


def _case(
    *,
    case_id: str,
    task_id: str,
    derived_properties: list[str],
    failure_records: list[dict],
    history: list[dict] | None = None,
    official_test_miss: bool = False,
    repaired: bool = False,
    accepted: bool = False,
) -> dict:
    baseline = {
        "passed": accepted,
        "error_type": "none" if accepted else "failure",
        "error_message": "" if accepted else "AssertionError: failed",
        "body": "",
        "candidate_code": "",
        "detected": bool(failure_records) or not accepted,
        "accepted": accepted,
        "official_test_miss_caught_by_pbt": official_test_miss,
        "n_static_violations": 0,
        "n_dynamic_violations": 0,
        "constraint_feedback": "",
        "dynamic_violations": [],
        "probe_inputs": [],
        "n_pbt_failures": len(failure_records),
        "pbt_violations": [],
        "pbt_derived_properties": [
            {"name": name, "source": "signature", "description": name}
            for name in derived_properties
        ],
        "pbt_failure_records": failure_records,
        "pbt_verified": not failure_records,
        "latency_seconds": 0.1,
    }
    return {
        "case_id": case_id,
        "dataset_idx": 0,
        "task_id": task_id,
        "entry_point": task_id.replace("/", "_"),
        "baseline": baseline,
        "verify_only": {
            "detected": baseline["detected"],
            "accepted": accepted,
            "official_test_miss_caught_by_pbt": official_test_miss,
            "n_pbt_failures": len(failure_records),
            "pbt_violations": [],
            "latency_seconds": 0.1,
        },
        "verify_repair": {
            "passed": accepted or repaired,
            "repaired": repaired,
            "n_repairs": max(0, len(history or []) - 1),
            "final_body": "",
            "final_code": "",
            "final_detected": bool(history and history[-1].get("detected", False)),
            "final_accepted": bool(history and history[-1].get("accepted", accepted or repaired)),
            "final_error_type": "none" if repaired else baseline["error_type"],
            "final_error_message": "" if repaired else baseline["error_message"],
        },
        "history": history or [],
    }


def _history_step(
    *,
    iteration: int,
    harness_error: str = "",
    accepted: bool = False,
    property_failures: list[dict] | None = None,
) -> dict:
    failures = property_failures or []
    return {
        "iteration": iteration,
        "body": "",
        "candidate_code": "",
        "harness": {
            "passed": accepted and not harness_error,
            "error_type": "none" if accepted and not harness_error else "failure",
            "error_message": harness_error,
            "stdout": "",
        },
        "instrumentation": {
            "n_constraints": 0,
            "constraint_feedback": "",
            "n_static_violations": 0,
            "static_violations": [],
            "n_dynamic_violations": 0,
            "dynamic_violations": [],
            "n_property_violations": len(failures),
            "property_violations": [],
            "probe_inputs": [],
            "detected": bool(harness_error) or bool(failures),
        },
        "pbt": {
            "verified": not failures,
            "derived_properties": [],
            "failure_records": failures,
            "n_failures": len(failures),
            "violations": [],
            "repair_feedback": "",
            "wall_clock_seconds": 0.1,
            "max_examples": 16,
        },
        "detected": bool(harness_error) or bool(failures),
        "accepted": accepted,
    }


def _payload(*cases: dict) -> dict:
    cohort_cases = []
    for index, case in enumerate(cases):
        cohort_cases.append(
            {
                "case_id": case["case_id"],
                "dataset_idx": index,
                "task_id": case["task_id"],
                "prompt": f"prompt for {case['task_id']}",
                "test": "def check(candidate): pass",
                "entry_point": case["entry_point"],
                "sample_position": index + 1,
                "prompt_seeds": {
                    "baseline": index + 1,
                    "verify_only": index + 1,
                    "verify_repair": index + 1,
                },
            }
        )
    return {
        "experiment": 999,
        "benchmark": "humaneval_pbt_full",
        "title": "synthetic trace payload",
        "cohort": {
            "case_count": len(cases),
            "case_ids": [case["case_id"] for case in cases],
            "cases": cohort_cases,
        },
        "per_problem_results": list(cases),
    }


class TestTraceAnalyzer:
    """REQ-CODE-016: Artifact ingestion and high-level trace analysis."""

    def test_real_artifacts_skip_exp225_and_learn_from_exp226(self) -> None:
        """SCENARIO-CODE-014: Exp 225 is metadata-only while Exp 226 yields learnable traces."""
        repo_root = Path(__file__).resolve().parents[2]
        analyzer = TraceAnalyzer.from_paths(
            [
                repo_root / "results/experiment_225_results.json",
                repo_root / "results/experiment_226_results.json",
            ]
        )

        analysis = analyzer.analyze()
        by_property = {item.property_name: item for item in analysis.property_rankings}
        by_problem = {item.problem_type: item for item in analysis.problem_type_rankings}

        assert analysis.trace_artifact_count == 1
        assert analysis.skipped_artifact_count == 1
        assert analysis.case_count == 164
        assert analysis.skipped_artifacts == ("results/experiment_225_results.json",)
        assert by_property["no_exception"].failure_count == 144
        assert by_property["input_immutability"].repaired_cases == 3
        assert by_property["sorted_output"].official_test_misses == 2
        assert by_problem["signature_robustness"].official_test_misses == 6
        assert by_problem["sequence_intent"].official_test_misses == 2


class TestPropertyRanker:
    """REQ-CODE-017: Property effectiveness ranking over accumulated traces."""

    def test_ranks_properties_by_failures_misses_and_repairs(self) -> None:
        """REQ-CODE-017: High-value signals outrank raw low-value support."""
        syntax_failure = {
            "property_name": "no_exception",
            "source": "signature",
            "description": "no exception",
            "input_args": [0],
            "actual": "",
            "expected": "",
            "error": "IndentationError",
        }
        ordering_failure = {
            "property_name": "sorted_output",
            "source": "prompt_intent",
            "description": "sorted output",
            "input_args": [[2, 1]],
            "actual": "[2, 1]",
            "expected": "[1, 2]",
            "error": None,
        }

        traces = TraceAnalyzer.from_payloads(
            [
                _payload(
                    _case(
                        case_id="case-1",
                        task_id="HumanEval/1",
                        derived_properties=["no_exception", "deterministic"],
                        failure_records=[syntax_failure],
                        official_test_miss=True,
                        repaired=True,
                        history=[
                            _history_step(
                                iteration=0,
                                harness_error="IndentationError: unexpected indent",
                                property_failures=[syntax_failure],
                            ),
                            _history_step(iteration=1, accepted=True),
                        ],
                    ),
                    _case(
                        case_id="case-2",
                        task_id="HumanEval/2",
                        derived_properties=["sorted_output"],
                        failure_records=[ordering_failure],
                        official_test_miss=False,
                        repaired=False,
                        history=[_history_step(iteration=0, property_failures=[ordering_failure])],
                    ),
                )
            ],
            artifact_names=("synthetic.json",),
        ).cases

        ranker = PropertyRanker().fit(traces)
        ranked = ranker.rank()

        assert [item.property_name for item in ranked] == [
            "no_exception",
            "sorted_output",
        ]
        assert ranked[0].official_test_misses == 1
        assert ranked[0].repaired_cases == 1
        assert ranked[0].score > ranked[1].score


class TestRepairStrategy:
    """REQ-CODE-018: Repair strategy learning from verification histories."""

    def test_learns_repair_strategy_success_rates_and_recommendations(self) -> None:
        """REQ-CODE-018: Syntax recovery outranks unsuccessful ordering fixes on matching traces."""
        syntax_failure = {
            "property_name": "no_exception",
            "source": "signature",
            "description": "no exception",
            "input_args": [0],
            "actual": "",
            "expected": "",
            "error": "IndentationError",
        }
        ordering_failure = {
            "property_name": "sorted_output",
            "source": "prompt_intent",
            "description": "sorted output",
            "input_args": [[2, 1]],
            "actual": "[2, 1]",
            "expected": "[1, 2]",
            "error": None,
        }
        return_failure = {
            "property_name": "annotated_return_type",
            "source": "signature",
            "description": "return type",
            "input_args": [0],
            "actual": "None",
            "expected": "int",
            "error": None,
        }

        traces = TraceAnalyzer.from_payloads(
            [
                _payload(
                    _case(
                        case_id="case-syntax",
                        task_id="HumanEval/10",
                        derived_properties=["no_exception", "deterministic"],
                        failure_records=[syntax_failure],
                        repaired=True,
                        history=[
                            _history_step(
                                iteration=0,
                                harness_error="IndentationError: unexpected indent",
                                property_failures=[syntax_failure],
                            ),
                            _history_step(iteration=1, accepted=True),
                        ],
                    ),
                    _case(
                        case_id="case-order",
                        task_id="HumanEval/11",
                        derived_properties=["sorted_output"],
                        failure_records=[ordering_failure],
                        repaired=False,
                        history=[
                            _history_step(iteration=0, property_failures=[ordering_failure]),
                            _history_step(iteration=1, property_failures=[ordering_failure]),
                        ],
                    ),
                    _case(
                        case_id="case-return",
                        task_id="HumanEval/12",
                        derived_properties=["annotated_return_type"],
                        failure_records=[return_failure],
                        repaired=False,
                        history=[
                            _history_step(iteration=0, property_failures=[return_failure]),
                            _history_step(iteration=1, accepted=False),
                        ],
                    ),
                )
            ],
            artifact_names=("synthetic.json",),
        ).cases

        learner = RepairStrategy().fit(traces)
        ranked = learner.rank()
        syntax_recommendations = learner.recommend(
            harness_error_message="IndentationError: unexpected indent",
            property_names=("no_exception",),
        )
        fallback = learner.recommend(
            harness_error_message="UnknownError: nope",
            property_names=(),
        )

        assert ranked[0].strategy_name == "syntax_recovery"
        assert ranked[0].error_family == "syntax"
        assert ranked[0].attempts == 1
        assert ranked[0].successes == 1
        assert syntax_recommendations[0].strategy_name == "syntax_recovery"
        assert syntax_recommendations[0].success_rate == 1.0
        assert fallback


class TestLearningCurve:
    """SCENARIO-CODE-015: Cumulative trace learning should sharpen recommendations."""

    def test_demonstrates_improvement_over_accumulated_traces(self) -> None:
        """SCENARIO-CODE-015: Later prefixes favor the empirically successful syntax repair path."""
        ordering_failure = {
            "property_name": "sorted_output",
            "source": "prompt_intent",
            "description": "sorted output",
            "input_args": [[2, 1]],
            "actual": "[2, 1]",
            "expected": "[1, 2]",
            "error": None,
        }
        syntax_failure = {
            "property_name": "no_exception",
            "source": "signature",
            "description": "no exception",
            "input_args": [0],
            "actual": "",
            "expected": "",
            "error": "IndentationError",
        }

        analyzer = TraceAnalyzer.from_payloads(
            [
                _payload(
                    _case(
                        case_id="case-1",
                        task_id="HumanEval/20",
                        derived_properties=["sorted_output"],
                        failure_records=[ordering_failure],
                        repaired=False,
                        history=[
                            _history_step(iteration=0, property_failures=[ordering_failure]),
                            _history_step(iteration=1, property_failures=[ordering_failure]),
                        ],
                    ),
                    _case(
                        case_id="case-2",
                        task_id="HumanEval/21",
                        derived_properties=["no_exception", "deterministic"],
                        failure_records=[syntax_failure],
                        repaired=True,
                        history=[
                            _history_step(
                                iteration=0,
                                harness_error="IndentationError: unexpected indent",
                                property_failures=[syntax_failure],
                            ),
                            _history_step(iteration=1, accepted=True),
                        ],
                    ),
                    _case(
                        case_id="case-3",
                        task_id="HumanEval/22",
                        derived_properties=["no_exception", "deterministic"],
                        failure_records=[syntax_failure],
                        repaired=True,
                        history=[
                            _history_step(
                                iteration=0,
                                harness_error="IndentationError: unexpected indent",
                                property_failures=[syntax_failure],
                            ),
                            _history_step(iteration=1, accepted=True),
                        ],
                    ),
                )
            ],
            artifact_names=("synthetic.json",),
        )

        improvement = analyzer.demonstrate_improvement(prefix_sizes=(1, 3))

        assert len(improvement.points) == 2
        assert improvement.points[0].top_strategy == "ordering_fix"
        assert improvement.points[-1].top_strategy == "syntax_recovery"
        assert improvement.points[-1].top_strategy_success_rate > (
            improvement.points[0].top_strategy_success_rate
        )
        assert improvement.improved is True


class TestParserAndFallbacks:
    """REQ-CODE-016, REQ-CODE-018: Sparse parser branches and empty fallbacks."""

    def test_handles_sparse_payloads_default_names_and_empty_inputs(
        self,
        tmp_path: Path,
    ) -> None:
        """REQ-CODE-016: Sparse artifacts are skipped or normalized without crashing."""
        external_artifact = tmp_path / "outside.json"
        external_artifact.write_text(json.dumps([]))

        empty = TraceAnalyzer.from_paths([external_artifact])
        empty_analysis = empty.analyze()
        ordering_failure = {
            "property_name": "sorted_output",
            "source": "prompt_intent",
            "description": "sorted output",
            "input_args": [[2, 1]],
            "actual": "[2, 1]",
            "expected": "[1, 2]",
            "error": None,
        }

        sparse = TraceAnalyzer.from_payloads(
            [
                {},
                {
                    "cohort": {"cases": None},
                    "per_problem_results": [
                        None,
                        {"baseline": {}, "task_id": ""},
                        {
                            "case_id": "case-a",
                            "task_id": "HumanEval/A",
                            "entry_point": "entry_a",
                            "baseline": None,
                            "verify_repair": {},
                            "history": None,
                        },
                    ],
                },
                {
                    "experiment_id": 321,
                    "cohort": {
                        "cases": [
                            None,
                            {"case_id": "", "prompt": "ignored"},
                            {
                                "case_id": "case-b",
                                "prompt": "prompt b",
                                "entry_point": "entry_b",
                            },
                        ]
                    },
                    "per_problem_results": [
                        {
                            "case_id": "case-b",
                            "task_id": "HumanEval/B",
                            "entry_point": "entry_b",
                            "baseline": {
                                "passed": False,
                                "accepted": False,
                                "official_test_miss_caught_by_pbt": True,
                                "pbt_derived_properties": [
                                    {"name": "sorted_output"},
                                ],
                                "pbt_failure_records": [
                                    "skip-me",
                                    {},
                                    {"property_name": ""},
                                    ordering_failure,
                                ],
                            },
                            "verify_repair": {},
                            "history": [
                                {
                                    "iteration": 0,
                                    "detected": True,
                                    "accepted": False,
                                    "harness": {},
                                    "pbt": {"failure_records": None},
                                }
                            ],
                        },
                        {
                            "case_id": "case-c",
                            "task_id": "HumanEval/C",
                            "entry_point": "entry_c",
                            "baseline": {
                                "passed": False,
                                "accepted": False,
                                "official_test_miss_caught_by_pbt": True,
                                "pbt_derived_properties": [],
                                "pbt_failure_records": None,
                            },
                            "verify_repair": {},
                            "history": [
                                {
                                    "iteration": 0,
                                    "detected": True,
                                    "accepted": False,
                                    "harness": {},
                                    "pbt": {"failure_records": None},
                                },
                                {
                                    "iteration": 1,
                                    "detected": False,
                                    "accepted": False,
                                    "harness": {},
                                    "pbt": {},
                                },
                            ],
                        },
                    ],
                },
            ]
        )

        sparse_analysis = sparse.analyze()
        default_curve = sparse.learning_curve()
        ordering_only = RepairStrategy().fit(
            TraceAnalyzer.from_payloads(
                [
                    _payload(
                        _case(
                            case_id="case-order-only",
                            task_id="HumanEval/Z",
                            derived_properties=["sorted_output"],
                            failure_records=[ordering_failure],
                            repaired=False,
                            history=[
                                _history_step(
                                    iteration=0,
                                    property_failures=[ordering_failure],
                                ),
                                _history_step(
                                    iteration=1,
                                    property_failures=[ordering_failure],
                                ),
                            ],
                        )
                    )
                ]
            ).cases
        )
        family_fallback = ordering_only.recommend(
            harness_error_message="",
            property_names=("annotated_return_type",),
        )

        assert empty_analysis.skipped_artifact_count == 1
        assert empty_analysis.case_count == 0
        assert empty.learning_curve() == ()
        assert empty.demonstrate_improvement().improved is False

        assert sparse_analysis.skipped_artifacts == ("artifact-1.json",)
        assert sparse_analysis.trace_artifact_count == 2
        assert sparse_analysis.case_count == 3
        assert sparse.cases[0].experiment_id == "artifact-2.json"
        assert sparse.cases[0].steps[0].property_failures == ()
        assert sparse.cases[1].experiment_id == "321"
        assert sparse.cases[1].prompt == "prompt b"
        assert sparse.cases[1].steps[0].property_failures == ()
        assert sparse.cases[2].official_test_miss is True
        assert default_curve[-1].prefix_size == len(sparse.cases)
        assert family_fallback[0].strategy_name == "ordering_fix"
