"""Spec: REQ-VERIFY-033, REQ-VERIFY-034, REQ-VERIFY-035,
SCENARIO-VERIFY-033, SCENARIO-VERIFY-034, SCENARIO-VERIFY-035.
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
    module_path = repo_root / "scripts" / "experiment_223_self_learning_replay.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_223_self_learning_replay",
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


def _baseline_case(case_id: str, success_key: str, success: bool) -> dict[str, object]:
    return {"case_id": case_id, success_key: success}


def exp219_fixture() -> dict[str, object]:
    def verify_only_case(
        case_id: str,
        *,
        correct: bool,
        flagged: bool,
        error_type: str | None,
        description: str = "semantic violation",
    ) -> dict[str, object]:
        violations: list[dict[str, object]] = []
        if error_type is not None:
            taxonomy_hint, violation_type = error_type.split(":", 1)
            violations.append(
                {
                    "violation_type": violation_type,
                    "description": description,
                    "metadata": {"taxonomy_hint": taxonomy_hint},
                }
            )
        return {
            "case_id": case_id,
            "correct": correct,
            "flagged": flagged,
            "response_mode": "structured_json",
            "verification": {
                "semantic_grounding": {
                    "violations": violations,
                    "refinement_applied": False,
                },
                "violations": [
                    {
                        "constraint_type": "semantic_grounding",
                        "description": description,
                        "metadata": {
                            "taxonomy_hint": error_type.split(":", 1)[0],
                            "violation_type": error_type.split(":", 1)[1],
                        },
                    }
                ]
                if error_type is not None
                else [],
            },
        }

    return {
        "experiment": 219,
        "benchmark": "gsm8k_semantic",
        "cohort": {
            "case_count": 4,
            "case_ids": ["gsm8k-1", "gsm8k-2", "gsm8k-3", "gsm8k-4"],
        },
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "baseline",
                "cases": [
                    _baseline_case("gsm8k-1", "correct", True),
                    _baseline_case("gsm8k-2", "correct", False),
                    _baseline_case("gsm8k-3", "correct", False),
                    _baseline_case("gsm8k-4", "correct", True),
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_only",
                "cases": [
                    verify_only_case("gsm8k-1", correct=True, flagged=False, error_type=None),
                    verify_only_case(
                        "gsm8k-2",
                        correct=False,
                        flagged=True,
                        error_type="arithmetic:wrong_sum",
                    ),
                    verify_only_case(
                        "gsm8k-3",
                        correct=False,
                        flagged=True,
                        error_type="arithmetic:wrong_sum",
                    ),
                    verify_only_case(
                        "gsm8k-4",
                        correct=True,
                        flagged=True,
                        error_type="question_grounding_failures:answer_target_mismatch",
                    ),
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_repair",
                "cases": [
                    _baseline_case("gsm8k-1", "correct", True) | {"repaired": False},
                    _baseline_case("gsm8k-2", "correct", True) | {"repaired": True},
                    _baseline_case("gsm8k-3", "correct", True) | {"repaired": True},
                    _baseline_case("gsm8k-4", "correct", False) | {"repaired": False},
                ],
            },
        ],
    }


def exp220_fixture() -> dict[str, object]:
    def verify_only_case(
        case_id: str,
        *,
        passed: bool,
        detected: bool,
        property_violations: list[str],
    ) -> dict[str, object]:
        return {
            "case_id": case_id,
            "passed": passed,
            "response_mode": "answer_only_terse",
            "execution_plus_property": {
                "detected": detected,
                "property_violations": property_violations,
                "static_violations": [],
                "dynamic_violations": [],
                "constraint_feedback": [],
            },
        }

    return {
        "experiment": 220,
        "benchmark": "humaneval_property",
        "cohort": {
            "case_count": 4,
            "case_ids": ["humaneval-1", "humaneval-2", "humaneval-3", "humaneval-4"],
        },
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "baseline",
                "cases": [
                    _baseline_case("humaneval-1", "passed", False),
                    _baseline_case("humaneval-2", "passed", False),
                    _baseline_case("humaneval-3", "passed", False),
                    _baseline_case("humaneval-4", "passed", False),
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_only",
                "cases": [
                    verify_only_case(
                        "humaneval-1",
                        passed=False,
                        detected=True,
                        property_violations=[
                            "deterministic (official_tests) failed for input=(1,): AssertionError"
                        ],
                    ),
                    verify_only_case(
                        "humaneval-2",
                        passed=False,
                        detected=True,
                        property_violations=[
                            "deterministic (official_tests) failed for input=(2,): AssertionError"
                        ],
                    ),
                    verify_only_case(
                        "humaneval-3",
                        passed=False,
                        detected=True,
                        property_violations=[
                            "deterministic (official_tests) failed for input=(3,): AssertionError"
                        ],
                    ),
                    verify_only_case(
                        "humaneval-4",
                        passed=False,
                        detected=False,
                        property_violations=[
                            "deterministic (official_tests) failed for input=(4,): AssertionError"
                        ],
                    ),
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_repair",
                "cases": [
                    _baseline_case("humaneval-1", "passed", True) | {"repaired": True},
                    _baseline_case("humaneval-2", "passed", True) | {"repaired": True},
                    _baseline_case("humaneval-3", "passed", True) | {"repaired": True},
                    _baseline_case("humaneval-4", "passed", True) | {"repaired": True},
                ],
            },
        ],
    }


def exp221_fixture() -> dict[str, object]:
    def verify_only_case(
        case_id: str,
        *,
        exact_satisfaction: bool,
        flagged: bool,
        error_type: str | None,
        task_slice: str = "code_typed_properties",
    ) -> dict[str, object]:
        results: list[dict[str, object]] = []
        if error_type is not None:
            family, kind = error_type.split(":", 1)
            results.append(
                {
                    "status": "violated",
                    "family": family,
                    "type": kind,
                }
            )
        return {
            "case_id": case_id,
            "exact_satisfaction": exact_satisfaction,
            "flagged": flagged,
            "response_mode": "answer_only_terse",
            "evaluation": {
                "task_slice": task_slice,
                "constraint_results": results,
                "judging_summary": {
                    "deterministic": 1,
                    "heuristic_rule": 0,
                    "model_assisted": 0,
                },
            },
        }

    return {
        "experiment": 221,
        "benchmark": "constraint_ir",
        "cohort": {
            "case_count": 4,
            "case_ids": ["constraint-1", "constraint-2", "constraint-3", "constraint-4"],
        },
        "paired_runs": [
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "baseline",
                "cases": [
                    _baseline_case("constraint-1", "exact_satisfaction", False),
                    _baseline_case("constraint-2", "exact_satisfaction", False),
                    _baseline_case("constraint-3", "exact_satisfaction", False),
                    _baseline_case("constraint-4", "exact_satisfaction", False),
                ],
            },
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "verify_only",
                "cases": [
                    verify_only_case(
                        "constraint-1",
                        exact_satisfaction=False,
                        flagged=True,
                        error_type="search_optimization_limited:semantic_property",
                    ),
                    verify_only_case(
                        "constraint-2",
                        exact_satisfaction=False,
                        flagged=True,
                        error_type="search_optimization_limited:semantic_property",
                    ),
                    verify_only_case(
                        "constraint-3",
                        exact_satisfaction=False,
                        flagged=True,
                        error_type="search_optimization_limited:semantic_property",
                    ),
                    verify_only_case(
                        "constraint-4",
                        exact_satisfaction=False,
                        flagged=False,
                        error_type="search_optimization_limited:semantic_property",
                    ),
                ],
            },
            {
                "model_name": "Gemma4-E4B-it",
                "mode": "verify_repair",
                "cases": [
                    _baseline_case("constraint-1", "exact_satisfaction", False)
                    | {"repaired": False},
                    _baseline_case("constraint-2", "exact_satisfaction", True) | {"repaired": True},
                    _baseline_case("constraint-3", "exact_satisfaction", True) | {"repaired": True},
                    _baseline_case("constraint-4", "exact_satisfaction", True) | {"repaired": True},
                ],
            },
        ],
    }


def write_fixture_repo(repo: Path) -> None:
    results = repo / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_219_results.json").write_text(
        json.dumps(exp219_fixture()), encoding="utf-8"
    )
    (results / "experiment_220_results.json").write_text(
        json.dumps(exp220_fixture()), encoding="utf-8"
    )
    (results / "experiment_221_results.json").write_text(
        json.dumps(exp221_fixture()), encoding="utf-8"
    )
    (results / "constraint_memory_live_222.json").write_text(
        json.dumps({"experiment": 222, "run_date": "20260412"}), encoding="utf-8"
    )


# REQ-VERIFY-033, REQ-VERIFY-034, REQ-VERIFY-035
def test_run_replay_cases_uses_only_non_heldout_updates_and_tracks_cross_model_memory():
    """SCENARIO-VERIFY-033 through SCENARIO-VERIFY-035: replay stays provenance-aware."""
    module = load_pipeline_module()

    cases = [
        module.ReplayCase(
            source_experiment=219,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="noise-1",
            sample_position=1,
            held_out=False,
            actual_error=False,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("noisy signal",),
            baseline_success=True,
            repair_success=False,
        ),
        module.ReplayCase(
            source_experiment=219,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="noise-2",
            sample_position=2,
            held_out=False,
            actual_error=False,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("noisy signal",),
            baseline_success=True,
            repair_success=False,
        ),
        module.ReplayCase(
            source_experiment=219,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="noise-3",
            sample_position=3,
            held_out=False,
            actual_error=False,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("noisy signal",),
            baseline_success=True,
            repair_success=False,
        ),
        module.ReplayCase(
            source_experiment=219,
            benchmark="gsm8k_semantic",
            metric_name="accuracy",
            domain="live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="heldout-noisy",
            sample_position=4,
            held_out=True,
            actual_error=False,
            detected=True,
            error_types=("question_grounding_failures:answer_target_mismatch",),
            descriptions=("noisy signal",),
            baseline_success=True,
            repair_success=False,
        ),
        module.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="memory-1",
            sample_position=1,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("official failure",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="memory-2",
            sample_position=2,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("official failure",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="memory-3",
            sample_position=3,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=("official failure",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Gemma4-E4B-it",
            case_id="heldout-memory-cross-model",
            sample_position=4,
            held_out=True,
            actual_error=True,
            detected=False,
            error_types=("official_test_failure",),
            descriptions=("official failure",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=221,
            benchmark="constraint_ir",
            metric_name="constraint_satisfaction",
            domain="instruction_surface_only",
            model_name="Gemma4-E4B-it",
            case_id="partial-memory-1",
            sample_position=1,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("literal:word_count_range",),
            descriptions=("word count",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=221,
            benchmark="constraint_ir",
            metric_name="constraint_satisfaction",
            domain="instruction_surface_only",
            model_name="Gemma4-E4B-it",
            case_id="partial-memory-2",
            sample_position=2,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("literal:word_count_range",),
            descriptions=("word count",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=221,
            benchmark="constraint_ir",
            metric_name="constraint_satisfaction",
            domain="instruction_surface_only",
            model_name="Gemma4-E4B-it",
            case_id="heldout-no-update-a",
            sample_position=4,
            held_out=True,
            actual_error=True,
            detected=False,
            error_types=("literal:word_count_range",),
            descriptions=("word count",),
            baseline_success=False,
            repair_success=True,
        ),
        module.ReplayCase(
            source_experiment=221,
            benchmark="constraint_ir",
            metric_name="constraint_satisfaction",
            domain="instruction_surface_only",
            model_name="Gemma4-E4B-it",
            case_id="heldout-no-update-b",
            sample_position=5,
            held_out=True,
            actual_error=True,
            detected=False,
            error_types=("literal:word_count_range",),
            descriptions=("word count",),
            baseline_success=False,
            repair_success=True,
        ),
    ]

    payload = module.run_replay_cases(
        cases,
        tracker_min_support=3,
        tracker_min_precision=0.75,
        memory_min_support=3,
    )

    decisions = {item["case_id"]: item["strategies"] for item in payload["held_out_decisions"]}
    assert decisions["heldout-noisy"]["no_learning"]["use_repair"] is True
    assert decisions["heldout-noisy"]["tracker_only"]["use_repair"] is False
    assert decisions["heldout-memory-cross-model"]["tracker_only"]["use_repair"] is False
    assert decisions["heldout-memory-cross-model"]["tracker_plus_memory"]["use_repair"] is True
    assert decisions["heldout-no-update-a"]["tracker_plus_memory"]["use_repair"] is False
    assert decisions["heldout-no-update-b"]["tracker_plus_memory"]["use_repair"] is False

    strategies = payload["strategies"]
    assert strategies["tracker_only"]["overall"]["false_positives"] == 0
    assert (
        strategies["tracker_plus_memory"]["overall"]["success_rate"]
        > (strategies["tracker_only"]["overall"]["success_rate"])
    )

    transfer = payload["transfer_effects"]["tracker_plus_memory"]["Gemma4-E4B-it"]
    assert transfer["cross_model_helpful_events"] == 1
    assert transfer["same_model_helpful_events"] == 0


# REQ-VERIFY-033, REQ-VERIFY-034
def test_build_replay_cases_marks_final_quarter_holdout_and_metric_names():
    """REQ-VERIFY-033 and REQ-VERIFY-034: Exp 219-221 payloads become held-out replay cases."""
    module = load_pipeline_module()

    cases = module.build_replay_cases(
        exp219=exp219_fixture(),
        exp220=exp220_fixture(),
        exp221=exp221_fixture(),
    )

    by_case = {(case.source_experiment, case.case_id): case for case in cases}
    assert by_case[(219, "gsm8k-4")].held_out is True
    assert by_case[(220, "humaneval-4")].held_out is True
    assert by_case[(221, "constraint-4")].held_out is True
    assert by_case[(219, "gsm8k-2")].metric_name == "accuracy"
    assert by_case[(220, "humaneval-2")].metric_name == "pass_rate"
    assert by_case[(221, "constraint-2")].metric_name == "constraint_satisfaction"
    assert by_case[(220, "humaneval-4")].error_types == ("official_test_failure",)


# REQ-VERIFY-034, REQ-VERIFY-035
def test_build_self_learning_replay_payload_summarizes_heldout_metrics():
    """SCENARIO-VERIFY-034 and SCENARIO-VERIFY-035: payload includes key summaries."""
    module = load_pipeline_module()

    payload = module.build_self_learning_replay_payload(
        exp219=exp219_fixture(),
        exp220=exp220_fixture(),
        exp221=exp221_fixture(),
    )

    assert payload["experiment"] == 223
    assert payload["run_date"] == "20260412"
    assert payload["metadata"]["held_out_policy"]["name"] == "final_quarter_per_experiment"
    assert payload["summary"]["held_out_cases"] == 3
    assert (
        payload["summary"]["false_positive_regression_budget"]["tracker_only"]["within_budget"]
        is True
    )
    assert payload["strategies"]["tracker_plus_memory"]["overall"]["retrieval_precision"] == (2 / 3)


# REQ-VERIFY-034, REQ-VERIFY-035
def test_run_experiment_and_script_write_exp223_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """REQ-VERIFY-034 and REQ-VERIFY-035: the module and script refresh Exp 223 in place."""
    pipeline_module = load_pipeline_module()
    script_module = load_script_module()
    repo = make_repo(tmp_path)
    write_fixture_repo(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    payload = pipeline_module.run_experiment(repo)
    result_path = repo / "results" / "experiment_223_results.json"
    assert result_path.exists()
    assert payload["metadata"]["output_path"] == "results/experiment_223_results.json"

    parser = script_module.build_parser()
    args = parser.parse_args([])
    assert args.output == "results/experiment_223_results.json"
    assert script_module.get_repo_root() == repo.resolve()

    argv = sys.argv
    try:
        sys.argv = ["experiment_223_self_learning_replay.py"]
        runpy.run_path(str(Path(script_module.__file__)), run_name="__main__")
    finally:
        sys.argv = argv

    written = json.loads(result_path.read_text())
    assert written["title"] == "Held-out live self-learning replay benchmark"
    assert written["summary"]["held_out_cases"] == 3


# REQ-VERIFY-033
def test_helper_branches_cover_repo_override_and_empty_replay(monkeypatch: pytest.MonkeyPatch):
    """REQ-VERIFY-033: helper defaults stay deterministic on empty input."""
    module = load_pipeline_module()
    monkeypatch.delenv("CARNOT_REPO_ROOT", raising=False)

    assert module.get_repo_root().name == "carnot"
    empty = module.run_replay_cases([])
    assert empty["held_out_decisions"] == []
    assert empty["strategies"]["no_learning"]["overall"]["n_cases"] == 0
    assert empty["summary"]["held_out_cases"] == 0


# REQ-VERIFY-033, REQ-VERIFY-034, REQ-VERIFY-035
def test_internal_helpers_cover_filtering_and_harmful_transfer_paths(tmp_path: Path):
    """REQ-VERIFY-033 through REQ-VERIFY-035: helper branches stay deterministic and traceable."""
    module = load_pipeline_module()

    assert module._ObservedTypeStats().precision == 0.0
    assert module._ObservedTypeStats(fired=4, true_positives=3).precision == 0.75
    assert module._relative_path(Path("/tmp/outside.json"), tmp_path) == "/tmp/outside.json"

    assert module._sample_positions({"cohort": None}) == ({}, 0)
    assert module._sample_positions(
        {"cohort": {"case_count": 2, "cases": ["bad", {"case_id": "x", "sample_position": 0}]}}
    ) == ({}, 2)
    assert module._sample_positions(
        {
            "cohort": {
                "case_count": 2,
                "cases": [{"case_id": "x", "sample_position": 1}],
            }
        }
    ) == ({"x": 1}, 2)

    lookup = module._paired_case_lookup(
        {
            "paired_runs": [
                "bad",
                {
                    "model_name": "Model",
                    "mode": "baseline",
                    "cases": ["bad", {"case_id": "case-1", "correct": True}],
                },
            ]
        }
    )
    assert lookup[("Model", "baseline")]["case-1"]["correct"] is True

    tracker = module.ConstraintTracker()
    tracker.record("trusted", fired=True, caught_error=True, any_error_in_batch=True)
    tracker.record("trusted", fired=True, caught_error=True, any_error_in_batch=True)
    tracker.record("trusted", fired=True, caught_error=True, any_error_in_batch=True)
    observed_types = {
        "trusted": module._ObservedTypeStats(
            fired=3,
            true_positives=3,
            source_models={"Model"},
        )
    }
    supported_case = module.ReplayCase(
        source_experiment=1,
        benchmark="constraint_ir",
        metric_name="constraint_satisfaction",
        domain="code_typed_properties",
        model_name="Model",
        case_id="supported",
        sample_position=1,
        held_out=True,
        actual_error=True,
        detected=True,
        error_types=("trusted",),
        descriptions=("trusted description",),
        baseline_success=False,
        repair_success=True,
    )
    assert (
        module._tracker_decision(
            supported_case,
            tracker=tracker,
            observed_types=observed_types,
            tracker_min_support=3,
            tracker_min_precision=0.75,
        ).reason
        == "tracker_supported"
    )

    no_type_case = module.ReplayCase(
        source_experiment=1,
        benchmark="constraint_ir",
        metric_name="constraint_satisfaction",
        domain="code_typed_properties",
        model_name="Model",
        case_id="no-type",
        sample_position=1,
        held_out=True,
        actual_error=True,
        detected=True,
        error_types=(),
        descriptions=(),
        baseline_success=False,
        repair_success=True,
    )
    assert (
        module._tracker_decision(
            no_type_case,
            tracker=tracker,
            observed_types={},
            tracker_min_support=3,
            tracker_min_precision=0.75,
        ).reason
        == "detected_without_error_type"
    )

    observed_types = {
        "supported": module._ObservedTypeStats(fired=3, true_positives=3, source_models={"A"}),
        "low_support": module._ObservedTypeStats(fired=2, true_positives=2, source_models={"A"}),
        "mixed_precision": module._ObservedTypeStats(
            fired=3, true_positives=2, source_models={"A"}
        ),
        "no_gain": module._ObservedTypeStats(fired=3, true_positives=3, source_models={"A"}),
    }
    observed_patterns = {
        ("code_typed_properties", "missing_type"): module._ObservedPatternStats(
            support=3,
            repair_improvements=1,
            source_models={"A"},
        ),
        ("code_typed_properties", "low_support"): module._ObservedPatternStats(
            support=3,
            repair_improvements=1,
            source_models={"A"},
        ),
        ("code_typed_properties", "mixed_precision"): module._ObservedPatternStats(
            support=3,
            repair_improvements=1,
            source_models={"A"},
        ),
        ("code_typed_properties", "no_gain"): module._ObservedPatternStats(
            support=3,
            repair_improvements=0,
            repair_harms=0,
            source_models={"A"},
        ),
        ("code_typed_properties", "supported"): module._ObservedPatternStats(
            support=3,
            repair_improvements=2,
            source_models={"A"},
        ),
    }
    candidates = module._memory_candidates(
        domain="code_typed_properties",
        tracker=tracker,
        observed_types=observed_types,
        observed_patterns=observed_patterns,
        memory_min_support=3,
    )
    assert [error_type for error_type, _ in candidates] == ["supported"]

    tracker_decision = module._Decision(
        use_repair=True,
        reason="tracker_supported",
        support_models=("A",),
    )
    memory_case = module.ReplayCase(
        source_experiment=1,
        benchmark="humaneval_property",
        metric_name="pass_rate",
        domain="code_typed_properties",
        model_name="B",
        case_id="memory",
        sample_position=1,
        held_out=True,
        actual_error=True,
        detected=True,
        error_types=("supported",),
        descriptions=("supported description",),
        baseline_success=False,
        repair_success=True,
    )
    memory_decision = module._memory_decision(
        memory_case,
        tracker_decision=tracker_decision,
        tracker=tracker,
        observed_types=observed_types,
        observed_patterns=observed_patterns,
        memory_min_support=3,
    )
    assert memory_decision.candidate_error_types == ("supported",)
    assert memory_decision.matched_error_types == ("supported",)

    transfer_effects = {"tracker_only": {}}
    harmful_case = module.ReplayCase(
        source_experiment=1,
        benchmark="gsm8k_semantic",
        metric_name="accuracy",
        domain="live_gsm8k_semantic_failure",
        model_name="ModelA",
        case_id="harmful",
        sample_position=1,
        held_out=True,
        actual_error=False,
        detected=True,
        error_types=("noise",),
        descriptions=("noise",),
        baseline_success=True,
        repair_success=False,
    )
    module._update_transfer_effects(
        transfer_effects,
        strategy_name="tracker_only",
        case=harmful_case,
        decision=module._Decision(
            use_repair=True,
            reason="tracker_supported",
            support_models=("ModelA", "ModelB"),
        ),
        reference_use_repair=False,
    )
    assert transfer_effects["tracker_only"]["ModelA"]["same_model_harmful_events"] == 1
    assert transfer_effects["tracker_only"]["ModelA"]["cross_model_harmful_events"] == 1

    short_desc_case = module.ReplayCase(
        source_experiment=1,
        benchmark="constraint_ir",
        metric_name="constraint_satisfaction",
        domain="code_typed_properties",
        model_name="Model",
        case_id="short-desc",
        sample_position=1,
        held_out=True,
        actual_error=True,
        detected=True,
        error_types=("a", "b"),
        descriptions=("only first",),
        baseline_success=False,
        repair_success=True,
    )
    assert module._description_for_error_type(short_desc_case, "b") == "only first"
    assert module._description_for_error_type(no_type_case, "fallback") == "fallback"
