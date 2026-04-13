"""Spec: REQ-VERIFY-054, REQ-VERIFY-055,
SCENARIO-VERIFY-060, SCENARIO-VERIFY-061, SCENARIO-VERIFY-062.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def load_pipeline_module():
    return importlib.import_module("carnot.pipeline.self_learning_replay")


def load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_241_self_learning_replay_v2.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_241_self_learning_replay_v2",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def exp223_reference_fixture() -> dict[str, object]:
    return {
        "experiment": 223,
        "run_date": "20260412",
        "strategies": {
            "no_learning": {
                "overall": {
                    "success_rate": 0.5,
                    "false_positives": 2,
                    "retrieval_hit_rate": 0.0,
                    "retrieval_precision": 0.0,
                }
            },
            "tracker_only": {
                "overall": {
                    "success_rate": 0.25,
                    "false_positives": 1,
                    "retrieval_hit_rate": 0.0,
                    "retrieval_precision": 0.0,
                }
            },
            "tracker_plus_memory": {
                "overall": {
                    "success_rate": 0.25,
                    "false_positives": 1,
                    "retrieval_hit_rate": 0.1,
                    "retrieval_precision": 0.2,
                }
            },
        },
        "summary": {"held_out_cases": 8},
    }


def exp235_fixture() -> dict[str, object]:
    def baseline_case(case_id: str, *, correct: bool, latency_seconds: float) -> dict[str, object]:
        return {
            "case_id": case_id,
            "correct": correct,
            "latency_seconds": latency_seconds,
        }

    def verify_only_case(
        case_id: str,
        *,
        correct: bool,
        flagged: bool,
        error_type: str | None,
    ) -> dict[str, object]:
        violations: list[dict[str, object]] = []
        if error_type is not None:
            taxonomy_hint, violation_type = error_type.split(":", 1)
            violations.append(
                {
                    "constraint_type": "semantic_grounding",
                    "description": violation_type.replace("_", " "),
                    "metadata": {
                        "taxonomy_hint": taxonomy_hint,
                        "violation_type": violation_type,
                    },
                }
            )
        return {
            "case_id": case_id,
            "correct": correct,
            "flagged": flagged,
            "latency_seconds": 0.05,
            "response_mode": "grammar_gated_json",
            "verification": {"violations": violations},
        }

    def verify_repair_case(
        case_id: str,
        *,
        correct: bool,
        repaired: bool,
        latency_seconds: float,
    ) -> dict[str, object]:
        return {
            "case_id": case_id,
            "correct": correct,
            "repaired": repaired,
            "n_repairs": 1 if repaired else 0,
            "latency_seconds": latency_seconds,
        }

    return {
        "experiment": 235,
        "benchmark": "gsm8k_semantic",
        "run_date": "20260413",
        "cohort": {
            "case_count": 4,
            "case_ids": ["gsm8k-1", "gsm8k-2", "gsm8k-3", "gsm8k-4"],
            "cases": [
                {
                    "case_id": "gsm8k-1",
                    "question": "semantic learning one",
                    "task_slice": "live_gsm8k_semantic_failure",
                    "sample_position": 1,
                    "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1},
                },
                {
                    "case_id": "gsm8k-2",
                    "question": "semantic learning two",
                    "task_slice": "live_gsm8k_semantic_failure",
                    "sample_position": 2,
                    "prompt_seeds": {"baseline": 2, "verify_only": 2, "verify_repair": 2},
                },
                {
                    "case_id": "gsm8k-3",
                    "question": "semantic heldout",
                    "task_slice": "live_gsm8k_semantic_failure",
                    "sample_position": 3,
                    "prompt_seeds": {"baseline": 3, "verify_only": 3, "verify_repair": 3},
                },
                {
                    "case_id": "gsm8k-4",
                    "question": "semantic noisy heldout",
                    "task_slice": "live_gsm8k_semantic_failure",
                    "sample_position": 4,
                    "prompt_seeds": {"baseline": 4, "verify_only": 4, "verify_repair": 4},
                },
            ],
        },
        "paired_runs": [
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "baseline",
                "cases": [
                    baseline_case("gsm8k-1", correct=False, latency_seconds=0.21),
                    baseline_case("gsm8k-2", correct=False, latency_seconds=0.22),
                    baseline_case("gsm8k-3", correct=False, latency_seconds=0.23),
                    baseline_case("gsm8k-4", correct=True, latency_seconds=0.24),
                ],
            },
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "verify_only",
                "cases": [
                    verify_only_case(
                        "gsm8k-1",
                        correct=False,
                        flagged=True,
                        error_type="question_grounding_failures:answer_target_mismatch",
                    ),
                    verify_only_case(
                        "gsm8k-2",
                        correct=False,
                        flagged=True,
                        error_type="question_grounding_failures:answer_target_mismatch",
                    ),
                    verify_only_case(
                        "gsm8k-3",
                        correct=False,
                        flagged=True,
                        error_type="question_grounding_failures:answer_target_mismatch",
                    ),
                    verify_only_case(
                        "gsm8k-4",
                        correct=True,
                        flagged=True,
                        error_type="unsupported_assumption:extra_claim",
                    ),
                ],
            },
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "verify_repair",
                "cases": [
                    verify_repair_case(
                        "gsm8k-1",
                        correct=True,
                        repaired=True,
                        latency_seconds=0.81,
                    ),
                    verify_repair_case(
                        "gsm8k-2",
                        correct=True,
                        repaired=True,
                        latency_seconds=0.82,
                    ),
                    verify_repair_case(
                        "gsm8k-3",
                        correct=True,
                        repaired=True,
                        latency_seconds=0.83,
                    ),
                    verify_repair_case(
                        "gsm8k-4",
                        correct=False,
                        repaired=False,
                        latency_seconds=0.84,
                    ),
                ],
            },
        ],
    }


def exp238_fixture() -> dict[str, object]:
    def problem_result(
        case_id: str,
        *,
        sample_position: int,
        baseline_passed: bool,
        spec_accepted: bool,
        repair_accepted: bool,
        error_message: str,
        repair_hint_family: str,
    ) -> dict[str, object]:
        return {
            "case_id": case_id,
            "dataset_idx": sample_position,
            "task_id": f"HumanEval/{sample_position}",
            "entry_point": "solve",
            "baseline": {"official_passed": baseline_passed},
            "spec_aware_verify_only": {"accepted": spec_accepted},
            "verify_repair": {
                "accepted": repair_accepted,
                "official_passed": repair_accepted,
                "repaired": repair_accepted and not baseline_passed,
                "n_repairs": 1 if repair_accepted and not baseline_passed else 0,
            },
            "history": [
                {
                    "iteration": 0,
                    "latency_seconds": 0.4 + (sample_position / 100),
                    "evaluation": {
                        "official_tests": {
                            "passed": baseline_passed,
                            "error_message": error_message,
                        },
                        "pbt": {
                            "violations": [f"no_exception failed: {error_message}"],
                            "derived_properties": [{"name": "no_exception"}],
                        },
                        "explicit_specs": {
                            "violations": [f"deterministic failed: {error_message}"],
                            "repair_hints": [{"error_family": repair_hint_family}],
                        },
                    },
                },
                {
                    "iteration": 1,
                    "latency_seconds": 0.8 + (sample_position / 100),
                    "evaluation": {
                        "official_tests": {"passed": repair_accepted, "error_message": ""},
                        "pbt": {"violations": [], "derived_properties": [{"name": "no_exception"}]},
                        "explicit_specs": {
                            "violations": [],
                            "repair_hints": [{"error_family": repair_hint_family}],
                        },
                    },
                },
            ],
        }

    return {
        "experiment": 238,
        "benchmark": "humaneval_dual_model_spec",
        "run_date": "20260413",
        "cohort": {
            "case_count": 4,
            "case_ids": ["humaneval-1", "humaneval-2", "humaneval-3", "humaneval-4"],
            "task_ids": ["HumanEval/1", "HumanEval/2", "HumanEval/3", "HumanEval/4"],
            "cases": [
                {
                    "case_id": "humaneval-1",
                    "task_id": "HumanEval/1",
                    "prompt": "code learning one",
                    "sample_position": 1,
                    "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1},
                },
                {
                    "case_id": "humaneval-2",
                    "task_id": "HumanEval/2",
                    "prompt": "code learning two",
                    "sample_position": 2,
                    "prompt_seeds": {"baseline": 2, "verify_only": 2, "verify_repair": 2},
                },
                {
                    "case_id": "humaneval-3",
                    "task_id": "HumanEval/3",
                    "prompt": "code heldout",
                    "sample_position": 3,
                    "prompt_seeds": {"baseline": 3, "verify_only": 3, "verify_repair": 3},
                },
                {
                    "case_id": "humaneval-4",
                    "task_id": "HumanEval/4",
                    "prompt": "code clean heldout",
                    "sample_position": 4,
                    "prompt_seeds": {"baseline": 4, "verify_only": 4, "verify_repair": 4},
                },
            ],
        },
        "model_runs": {
            "Qwen3.5-0.8B": {
                "model_name": "Qwen3.5-0.8B",
                "model_hf_id": "Qwen/Qwen3.5-0.8B",
                "run_status": "complete",
                "per_problem_results": [
                    problem_result(
                        "humaneval-1",
                        sample_position=1,
                        baseline_passed=False,
                        spec_accepted=False,
                        repair_accepted=True,
                        error_message="NameError: helper is not defined",
                        repair_hint_family="syntax",
                    ),
                    problem_result(
                        "humaneval-2",
                        sample_position=2,
                        baseline_passed=False,
                        spec_accepted=False,
                        repair_accepted=True,
                        error_message="AssertionError: wrong ordering",
                        repair_hint_family="syntax",
                    ),
                    problem_result(
                        "humaneval-3",
                        sample_position=3,
                        baseline_passed=False,
                        spec_accepted=False,
                        repair_accepted=True,
                        error_message="AssertionError: wrong ordering",
                        repair_hint_family="syntax",
                    ),
                    problem_result(
                        "humaneval-4",
                        sample_position=4,
                        baseline_passed=True,
                        spec_accepted=True,
                        repair_accepted=True,
                        error_message="",
                        repair_hint_family="logic",
                    ),
                ],
            }
        },
    }


def write_fixture_repo(repo: Path) -> None:
    results = repo / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_223_results.json").write_text(
        json.dumps(exp223_reference_fixture()), encoding="utf-8"
    )
    (results / "experiment_235_results.json").write_text(
        json.dumps(exp235_fixture()), encoding="utf-8"
    )
    (results / "experiment_238_results.json").write_text(
        json.dumps(exp238_fixture()), encoding="utf-8"
    )


# REQ-VERIFY-054, REQ-VERIFY-055
def test_run_replay_cases_v2_compares_case_memory_and_policy_conditions():
    """SCENARIO-VERIFY-061 and SCENARIO-VERIFY-062: policy updates can change held-out decisions."""
    module = load_pipeline_module()

    cases = [
        module.ReplayCase(
            source_experiment=235,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Gemma4-E4B-it",
            case_id="semantic-learn-1",
            sample_position=1,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("target mismatch",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.2,
            repair_latency_seconds=0.8,
        ),
        module.ReplayCase(
            source_experiment=235,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Gemma4-E4B-it",
            case_id="semantic-learn-2",
            sample_position=2,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("target mismatch",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.21,
            repair_latency_seconds=0.81,
        ),
        module.ReplayCase(
            source_experiment=235,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Gemma4-E4B-it",
            case_id="semantic-learn-3",
            sample_position=3,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("target mismatch",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.22,
            repair_latency_seconds=0.82,
        ),
        module.ReplayCase(
            source_experiment=238,
            benchmark="humaneval_dual_model_spec",
            metric_name="pass_rate",
            domain="code_spec_properties",
            model_name="Qwen3.5-0.8B",
            case_id="code-learn-1",
            sample_position=1,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("no_exception failed on official_tests",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.4,
            repair_latency_seconds=1.2,
        ),
        module.ReplayCase(
            source_experiment=238,
            benchmark="humaneval_dual_model_spec",
            metric_name="pass_rate",
            domain="code_spec_properties",
            model_name="Qwen3.5-0.8B",
            case_id="code-learn-2",
            sample_position=2,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("no_exception failed on official_tests",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.41,
            repair_latency_seconds=1.21,
        ),
        module.ReplayCase(
            source_experiment=235,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Gemma4-E4B-it",
            case_id="heldout-semantic",
            sample_position=4,
            held_out=True,
            actual_error=True,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("target mismatch",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.23,
            repair_latency_seconds=0.83,
        ),
        module.ReplayCase(
            source_experiment=238,
            benchmark="humaneval_dual_model_spec",
            metric_name="pass_rate",
            domain="code_spec_properties",
            model_name="Qwen3.5-0.8B",
            case_id="heldout-code-policy",
            sample_position=3,
            held_out=True,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("no_exception failed on official_tests",),
            baseline_success=False,
            repair_success=True,
            baseline_latency_seconds=0.42,
            repair_latency_seconds=1.22,
        ),
        module.ReplayCase(
            source_experiment=235,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="heldout-noisy",
            sample_position=5,
            held_out=True,
            actual_error=False,
            detected=True,
            error_types=("unsupported_assumption:extra_claim",),
            descriptions=("noisy extra claim",),
            baseline_success=True,
            repair_success=False,
            baseline_latency_seconds=0.24,
            repair_latency_seconds=0.84,
        ),
    ]

    payload = module.run_replay_cases_v2(
        cases,
        tracker_min_support=4,
        tracker_min_precision=0.75,
        memory_min_support=3,
        policy_min_case_support=2,
    )

    decisions = {item["case_id"]: item["strategies"] for item in payload["held_out_decisions"]}
    assert decisions["heldout-semantic"]["tracker_only"]["use_repair"] is False
    assert decisions["heldout-semantic"]["case_memory"]["use_repair"] is True
    assert decisions["heldout-code-policy"]["case_memory"]["use_repair"] is False
    assert decisions["heldout-code-policy"]["case_memory_plus_policy"]["use_repair"] is True
    assert decisions["heldout-noisy"]["no_learning"]["use_repair"] is True
    assert decisions["heldout-noisy"]["case_memory_plus_policy"]["use_repair"] is False
    assert (
        decisions["heldout-code-policy"]["case_memory_plus_policy"]["policy_context"][
            "routing_hints"
        ]
        != []
    )

    strategies = payload["strategies"]
    assert strategies["no_learning"]["overall"]["false_positives"] == 1
    assert strategies["tracker_only"]["overall"]["false_positives"] == 0
    assert (
        strategies["case_memory"]["overall"]["success_rate"]
        > (strategies["tracker_only"]["overall"]["success_rate"])
    )
    assert (
        strategies["case_memory_plus_policy"]["overall"]["success_rate"]
        > (strategies["case_memory"]["overall"]["success_rate"])
    )
    assert strategies["case_memory"]["overall"]["retrieval_hit_rate"] > 0.0
    assert (
        strategies["case_memory_plus_policy"]["overall"]["latency_overhead_seconds"]
        > (strategies["case_memory"]["overall"]["latency_overhead_seconds"])
    )
    assert (
        payload["summary"]["false_positive_regression_budget"]["case_memory_plus_policy"][
            "within_budget"
        ]
        is True
    )
    assert payload["summary"]["primary_success_condition"]["met"] is True


# REQ-VERIFY-054
def test_build_exp241_replay_cases_supports_exp235_and_exp238_artifact_shapes():
    """SCENARIO-VERIFY-060: Exp 235 and Exp 238 normalize into one held-out replay stream."""
    module = load_pipeline_module()

    cases = module.build_exp241_replay_cases(
        exp235=exp235_fixture(),
        exp238=exp238_fixture(),
        holdout_fraction=0.25,
    )

    by_case = {(case.source_experiment, case.case_id): case for case in cases}
    assert by_case[(235, "gsm8k-4")].held_out is True
    assert by_case[(238, "humaneval-4")].held_out is True
    assert by_case[(235, "gsm8k-2")].metric_name == "accuracy"
    assert by_case[(238, "humaneval-2")].metric_name == "pass_rate"
    assert by_case[(238, "humaneval-2")].error_types == ("syntax", "official_test_failure")
    assert by_case[(238, "humaneval-2")].baseline_latency_seconds > 0.0
    assert (
        by_case[(238, "humaneval-2")].repair_latency_seconds
        > by_case[(238, "humaneval-2")].baseline_latency_seconds
    )


# REQ-VERIFY-055
def test_build_self_learning_replay_v2_payload_reports_comparison_to_exp223():
    """SCENARIO-VERIFY-062: Exp 241 payload includes the honest Exp 223 comparison block."""
    module = load_pipeline_module()

    payload = module.build_self_learning_replay_v2_payload(
        exp235=exp235_fixture(),
        exp238=exp238_fixture(),
        exp223_reference=exp223_reference_fixture(),
        holdout_fraction=0.25,
    )

    assert payload["experiment"] == 241
    assert payload["run_date"] == "20260413"
    assert payload["metadata"]["held_out_policy"]["name"] == "final_slice_per_source_artifact"
    assert payload["summary"]["held_out_cases"] == 2
    assert payload["summary"]["primary_success_condition"]["metric"] == (
        "real_held_out_task_gain_with_no_extra_false_positives"
    )
    assert payload["comparison_to_experiment_223"]["reference_experiment"] == 223
    assert (
        payload["comparison_to_experiment_223"]["strategy_deltas"]["case_memory"][
            "reference_strategy"
        ]
        == "tracker_plus_memory"
    )
    assert (
        payload["comparison_to_experiment_223"]["strategy_deltas"]["case_memory_plus_policy"][
            "reference_strategy"
        ]
        == "tracker_plus_memory"
    )
    assert "case_memory_plus_policy" in payload["strategies"]


# REQ-VERIFY-054, REQ-VERIFY-055
def test_run_experiment_v2_and_script_write_exp241_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """REQ-VERIFY-054 and REQ-VERIFY-055: the module and script refresh Exp 241 in place."""
    pipeline_module = load_pipeline_module()
    script_module = load_script_module()
    repo = make_repo(tmp_path)
    write_fixture_repo(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    payload = pipeline_module.run_experiment_v2(repo)
    result_path = repo / "results" / "experiment_241_results.json"
    assert result_path.exists()
    assert payload["metadata"]["output_path"] == "results/experiment_241_results.json"

    parser = script_module.build_parser()
    args = parser.parse_args([])
    assert args.output == "results/experiment_241_results.json"
    assert script_module.get_repo_root() == repo.resolve()

    empty = pipeline_module.run_replay_cases_v2([])
    assert empty["summary"]["held_out_cases"] == 0
    assert empty["strategies"]["case_memory_plus_policy"]["overall"]["n_cases"] == 0

    argv = sys.argv
    try:
        sys.argv = ["experiment_241_self_learning_replay_v2.py"]
        runpy.run_path(str(Path(script_module.__file__)), run_name="__main__")
    finally:
        sys.argv = argv

    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["title"] == "Held-out live self-learning replay benchmark v2"
    assert written["comparison_to_experiment_223"]["reference_experiment"] == 223


# REQ-VERIFY-054, REQ-VERIFY-055
def test_v2_internal_helpers_cover_sparse_artifacts_and_policy_fallback_paths():
    """REQ-VERIFY-054 and REQ-VERIFY-055: helper branches stay deterministic on sparse inputs."""
    module = load_pipeline_module()

    exp235_events = module._extract_exp235_events(
        {
            "paired_runs": [
                "bad",
                {
                    "mode": "verify_only",
                    "model_name": "Model",
                    "cases": [
                        "bad",
                        {
                            "case_id": "legacy-metadata",
                            "correct": False,
                            "flagged": True,
                            "verification": {
                                "violations": [
                                    "bad",
                                    {
                                        "description": "legacy metadata",
                                        "metadata": {
                                            "legacy_violation_types": [
                                                "question_grounding_failures:legacy"
                                            ]
                                        },
                                    },
                                ]
                            },
                        },
                        {
                            "case_id": "legacy-certificate",
                            "correct": False,
                            "flagged": True,
                            "verification": {
                                "certificate": {
                                    "semantic_verifier_v2": {
                                        "legacy_violation_types": [
                                            "omitted_premises:missing_quantity_coverage"
                                        ]
                                    }
                                }
                            },
                        },
                    ],
                },
            ]
        }
    )
    assert exp235_events[0].error_types == ("question_grounding_failures:legacy",)
    assert exp235_events[1].error_types == ("omitted_premises:missing_quantity_coverage",)

    assert module._classify_exp238_text("SyntaxError: bad indent") == "syntax_error"
    assert module._classify_exp238_text("mystery failure") == "humaneval_failure"
    assert module._extract_exp238_errors({}) == ((), ())
    assert module._extract_exp238_errors(
        {
            "history": [
                {
                    "evaluation": {
                        "explicit_specs": {"repair_hints": ["bad"], "violations": []},
                        "pbt": {"violations": []},
                        "official_tests": {},
                    }
                }
            ]
        }
    ) == ((), ())

    assert module._build_exp238_cases({"model_runs": []}, holdout_fraction=0.25) == []

    sparse_code_cases = module._build_exp238_cases(
        {
            "cohort": {
                "case_count": 2,
                "cases": [
                    {"case_id": "code-1", "sample_position": 1},
                    {"case_id": "code-2", "sample_position": 2},
                ],
            },
            "model_runs": {
                "bad-run": {"model_name": "Bad", "per_problem_results": "bad"},
                "good-run": {
                    "model_name": "Good",
                    "per_problem_results": [
                        {},
                        {
                            "case_id": "code-2",
                            "baseline": {"official_passed": False},
                            "spec_aware_verify_only": {"accepted": False},
                            "verify_repair": {"accepted": True},
                            "history": ["bad"],
                        },
                    ],
                },
            },
        },
        holdout_fraction=0.25,
    )
    assert len(sparse_code_cases) == 1
    assert sparse_code_cases[0].repair_latency_seconds == 0.0

    policy_case = module.ReplayCase(
        source_experiment=238,
        benchmark="humaneval_dual_model_spec",
        metric_name="pass_rate",
        domain="code_spec_properties",
        model_name="Qwen3.5-0.8B",
        case_id="policy-branches",
        sample_position=1,
        held_out=True,
        actual_error=True,
        detected=True,
        error_types=("official_test_failure",),
        descriptions=("no_exception failed",),
        baseline_success=False,
        repair_success=True,
    )
    base_decision = module._Decision(use_repair=False, reason="no_memory_match")
    assert (
        module._policy_decision(
            policy_case,
            base_decision=base_decision,
            policy_context={
                "case_matches": (),
                "threshold_overrides": ("threshold:1",),
                "property_budget_updates": (),
                "repair_prompt_patches": (),
                "routing_hints": (),
                "routing_targets": (),
            },
        ).reason
        == "policy_threshold_override"
    )
    assert (
        module._policy_decision(
            policy_case,
            base_decision=base_decision,
            policy_context={
                "case_matches": (),
                "threshold_overrides": (),
                "property_budget_updates": (),
                "repair_prompt_patches": ("patch:1",),
                "routing_hints": (),
                "routing_targets": (),
            },
        ).reason
        == "policy_repair_patch"
    )
    assert (
        module._policy_decision(
            policy_case,
            base_decision=base_decision,
            policy_context={
                "case_matches": (),
                "threshold_overrides": (),
                "property_budget_updates": ("budget:1",),
                "repair_prompt_patches": (),
                "routing_hints": (),
                "routing_targets": (),
            },
        ).reason
        == "policy_property_budget"
    )
