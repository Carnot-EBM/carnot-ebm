"""Spec: REQ-VERIFY-030, REQ-VERIFY-031, REQ-VERIFY-032,
SCENARIO-VERIFY-030, SCENARIO-VERIFY-031, SCENARIO-VERIFY-032.
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
    return importlib.import_module("carnot.pipeline.live_trace_memory")


def load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_222_live_trace_memory.py"
    spec = importlib.util.spec_from_file_location("experiment_222_live_trace_memory", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def sample_policy() -> dict[str, object]:
    return {
        "experiment": "Exp 213",
        "run_date": "20260412",
        "per_task_slice": {
            "live_gsm8k_semantic_failure": {
                "recommended_mode": "structured_json",
            },
            "code_typed_properties": {
                "recommended_mode": "answer_only_terse",
            },
        },
        "model_guidance": {
            "Qwen3.5-0.8B": "Use terse or structured outputs based on task slice.",
        },
    }


def exp219_fixture() -> dict[str, object]:
    violation = {
        "violation_type": "answer_target_mismatch",
        "description": "Computed the wrong target quantity.",
        "metadata": {
            "taxonomy_hint": "question_grounding_failures",
        },
    }
    return {
        "experiment": 219,
        "benchmark": "gsm8k_semantic",
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_only",
                "cases": [
                    {
                        "case_id": "gsm8k-fp",
                        "response_mode": "structured_json",
                        "correct": True,
                        "flagged": True,
                        "typed_reasoning_parse_status": "fallback_text",
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": violation["description"],
                                    "metadata": {
                                        "violation_type": violation["violation_type"],
                                        "taxonomy_hint": "question_grounding_failures",
                                    },
                                }
                            ],
                            "semantic_grounding": {
                                "violations": [violation],
                                "refinement_applied": False,
                            },
                        },
                    },
                    {
                        "case_id": "gsm8k-tp-1",
                        "response_mode": "structured_json",
                        "correct": False,
                        "flagged": True,
                        "typed_reasoning_parse_status": "fallback_text",
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": violation["description"],
                                    "metadata": {
                                        "violation_type": violation["violation_type"],
                                        "taxonomy_hint": "question_grounding_failures",
                                    },
                                }
                            ],
                            "semantic_grounding": {
                                "violations": [violation],
                                "refinement_applied": False,
                            },
                        },
                    },
                    {
                        "case_id": "gsm8k-tp-2",
                        "response_mode": "structured_json",
                        "correct": False,
                        "flagged": True,
                        "typed_reasoning_parse_status": "fallback_text",
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": violation["description"],
                                    "metadata": {
                                        "violation_type": violation["violation_type"],
                                        "taxonomy_hint": "question_grounding_failures",
                                    },
                                }
                            ],
                            "semantic_grounding": {
                                "violations": [violation],
                                "refinement_applied": False,
                            },
                        },
                    },
                    {
                        "case_id": "gsm8k-tp-3",
                        "response_mode": "structured_json",
                        "correct": False,
                        "flagged": True,
                        "typed_reasoning_parse_status": "fallback_text",
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": violation["description"],
                                    "metadata": {
                                        "violation_type": violation["violation_type"],
                                        "taxonomy_hint": "question_grounding_failures",
                                    },
                                }
                            ],
                            "semantic_grounding": {
                                "violations": [violation],
                                "refinement_applied": False,
                            },
                        },
                    },
                    {
                        "case_id": "gsm8k-tp-4",
                        "response_mode": "structured_json",
                        "correct": False,
                        "flagged": True,
                        "typed_reasoning_parse_status": "fallback_text",
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": violation["description"],
                                    "metadata": {
                                        "violation_type": violation["violation_type"],
                                        "taxonomy_hint": "question_grounding_failures",
                                    },
                                }
                            ],
                            "semantic_grounding": {
                                "violations": [violation],
                                "refinement_applied": False,
                            },
                        },
                    },
                ],
            }
        ],
    }


def exp220_fixture() -> dict[str, object]:
    return {
        "experiment": 220,
        "benchmark": "humaneval_property",
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_only",
                "cases": [
                    {
                        "case_id": "humaneval-tp",
                        "response_mode": "answer_only_terse",
                        "passed": False,
                        "error_type": "failure",
                        "error_message": "AssertionError: bad answer",
                        "execution_plus_property": {
                            "detected": True,
                            "property_violations": [
                                "deterministic (official_tests) failed for input=(1,): "
                                "AssertionError: bad answer"
                            ],
                            "static_violations": [],
                            "dynamic_violations": [],
                            "constraint_feedback": [],
                        },
                    },
                    {
                        "case_id": "humaneval-fn",
                        "response_mode": "answer_only_terse",
                        "passed": False,
                        "error_type": "failure",
                        "error_message": "IndexError",
                        "execution_plus_property": {
                            "detected": False,
                            "property_violations": [],
                            "static_violations": [],
                            "dynamic_violations": [],
                            "constraint_feedback": [],
                        },
                    },
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_repair",
                "cases": [
                    {
                        "case_id": "humaneval-tp",
                        "repaired": False,
                        "n_repairs": 1,
                        "history": [
                            {"iteration": 0},
                            {
                                "iteration": 1,
                                "repair_prompt": (
                                    "You are fixing a Python function (repair attempt 1).\n\n"
                                    "Previous function body:\n"
                                    "    return 0\n\n"
                                    "HumanEval test failure:\n"
                                    "  - AssertionError: bad answer\n\n"
                                    "Write ONLY the corrected function body. No markdown fences.\n"
                                    "Indent with 4 spaces."
                                ),
                            },
                        ],
                    }
                ],
            },
        ],
    }


def exp221_fixture() -> dict[str, object]:
    return {
        "experiment": 221,
        "benchmark": "constraint_ir",
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_only",
                "cases": [
                    {
                        "case_id": "constraint-tp",
                        "response_mode": "answer_only_terse",
                        "exact_satisfaction": False,
                        "flagged": True,
                        "evaluation": {
                            "task_slice": "code_typed_properties",
                            "parseable": True,
                            "constraint_results": [
                                {
                                    "status": "violated",
                                    "type": "semantic_property",
                                    "family": "search_optimization_limited",
                                    "judge": "deterministic",
                                    "details": {},
                                }
                            ],
                            "judging_summary": {
                                "deterministic": 1,
                                "heuristic_rule": 0,
                                "model_assisted": 0,
                            },
                        },
                    },
                    {
                        "case_id": "constraint-fp",
                        "response_mode": "answer_only_terse",
                        "exact_satisfaction": True,
                        "flagged": True,
                        "evaluation": {
                            "task_slice": "code_typed_properties",
                            "parseable": True,
                            "constraint_results": [],
                            "judging_summary": {
                                "deterministic": 1,
                                "heuristic_rule": 0,
                                "model_assisted": 0,
                            },
                        },
                    },
                    {
                        "case_id": "constraint-ambiguous",
                        "response_mode": "answer_only_terse",
                        "exact_satisfaction": False,
                        "flagged": True,
                        "evaluation": {
                            "task_slice": "code_typed_properties",
                            "parseable": True,
                            "constraint_results": [
                                {
                                    "status": "violated",
                                    "type": "semantic_property",
                                    "family": "search_optimization_limited",
                                    "judge": "model_assisted",
                                    "details": {},
                                }
                            ],
                            "judging_summary": {
                                "deterministic": 0,
                                "heuristic_rule": 0,
                                "model_assisted": 1,
                            },
                        },
                    },
                ],
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_repair",
                "cases": [
                    {
                        "case_id": "constraint-tp",
                        "repaired": False,
                        "n_repairs": 1,
                        "history": [
                            {"iteration": 0, "response": "bad code"},
                            {
                                "iteration": 1,
                                "response": "bad code",
                                "repair_prompt": (
                                    "Audit Example ID: constraint-tp\n"
                                    "Audit Mode: answer_only_terse\n"
                                    "Your previous response did not satisfy "
                                    "the required contract.\n"
                                    "Previous response:\n"
                                    "bad code\n\n"
                                    "Issues:\n"
                                    "- parseable=True\n"
                                    "- answer_quality=0.75\n"
                                    "- constraint_coverage=1.0\n\n"
                                    "Answer again using the same response contract.\n"
                                ),
                            },
                        ],
                    }
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
    (results / "monitorability_policy_213.json").write_text(
        json.dumps(sample_policy()), encoding="utf-8"
    )


# REQ-VERIFY-030, REQ-VERIFY-031, REQ-VERIFY-032
def test_build_live_trace_memory_bundle_quarantines_false_signals_and_reuses_patterns():
    """SCENARIO-VERIFY-030 and SCENARIO-VERIFY-031: quarantine bad traces, reuse mature ones."""
    module = load_pipeline_module()

    bundle = module.build_live_trace_memory_bundle(
        exp219=exp219_fixture(),
        exp220=exp220_fixture(),
        exp221=exp221_fixture(),
        base_policy=sample_policy(),
    )

    summary = bundle["result_payload"]["summary"]
    assert summary["memory_growth"]["accepted_trace_events"] == 6
    assert summary["memory_growth"]["quarantined_trace_events"] == 4
    assert summary["memory_growth"]["total_patterns"] == 3
    assert summary["memory_growth"]["mature_patterns"] == 1
    assert summary["retrieval_usefulness"]["helpful_retrieval_events"] == 1
    assert summary["retrieval_usefulness"]["reused_pattern_precision"] == 1.0

    patterns = bundle["memory_payload"]["patterns"]["live_gsm8k_semantic_failure"]
    learned = patterns["question_grounding_failures:answer_target_mismatch"]
    assert learned["frequency"] == 4
    assert learned["auto_generated"] is True

    quarantined = bundle["memory_payload"]["quarantined_events"]
    reasons = {event["exclusion_reason"] for event in quarantined}
    assert reasons == {"ambiguous_trace", "false_negative", "false_positive"}


# REQ-VERIFY-031, REQ-VERIFY-032
def test_bundle_derives_repair_snippets_and_policy_updates():
    """SCENARIO-VERIFY-032: live repair histories become reusable prompt patches."""
    module = load_pipeline_module()

    bundle = module.build_live_trace_memory_bundle(
        exp219=exp219_fixture(),
        exp220=exp220_fixture(),
        exp221=exp221_fixture(),
        base_policy=sample_policy(),
    )

    snippets = {item["snippet_id"]: item for item in bundle["result_payload"]["repair_snippets"]}
    assert "humaneval_property:official_test_failure" in snippets
    assert (
        "HumanEval test failure:"
        in snippets["humaneval_property:official_test_failure"]["template"]
    )
    assert "constraint_ir:search_optimization_limited:semantic_property" in snippets
    assert (
        "Answer again using the same response contract."
        in snippets["constraint_ir:search_optimization_limited:semantic_property"]["template"]
    )

    updates = {
        (item["model_name"], item["domain"]): item
        for item in bundle["result_payload"]["monitorability_policy_updates"]
    }
    assert updates[("Qwen3.5-0.8B", "live_gsm8k_semantic_failure")]["recommended_action"] == (
        "guarded_memory_only"
    )
    assert updates[("Qwen3.5-0.8B", "code_typed_properties")]["recommended_mode"] == (
        "answer_only_terse"
    )


# REQ-VERIFY-031, REQ-VERIFY-032
def test_run_experiment_writes_exp222_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """REQ-VERIFY-031 and REQ-VERIFY-032: writing the artifacts refreshes both outputs."""
    module = load_pipeline_module()
    repo = make_repo(tmp_path)
    write_fixture_repo(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    result_payload, memory_payload = module.run_experiment(repo)

    result_path = repo / "results" / "experiment_222_results.json"
    memory_path = repo / "results" / "constraint_memory_live_222.json"
    assert result_path.exists()
    assert memory_path.exists()
    assert result_payload["experiment"] == 222
    assert result_payload["run_date"] == "20260412"
    assert memory_payload["version"] == 1
    assert memory_payload["experiment"] == 222


# REQ-VERIFY-031, REQ-VERIFY-032
def test_experiment_222_script_defaults_and_main_write_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """REQ-VERIFY-031 and REQ-VERIFY-032: the script refreshes both Exp 222 artifacts."""
    module = load_script_module()
    repo = make_repo(tmp_path)
    write_fixture_repo(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.output == "results/experiment_222_results.json"
    assert args.memory_output == "results/constraint_memory_live_222.json"
    assert module.get_repo_root() == repo.resolve()

    argv = sys.argv
    try:
        sys.argv = ["experiment_222_live_trace_memory.py"]
        runpy.run_path(str(Path(module.__file__)), run_name="__main__")
    finally:
        sys.argv = argv

    result_payload = json.loads((repo / "results" / "experiment_222_results.json").read_text())
    memory_payload = json.loads((repo / "results" / "constraint_memory_live_222.json").read_text())
    assert result_payload["title"] == "Live trace memory and repair guidance"
    assert memory_payload["result_source"] == "results/experiment_222_results.json"


# REQ-VERIFY-030, REQ-VERIFY-031
def test_trace_event_and_helper_branches_cover_defaults_and_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """REQ-VERIFY-030: helper branches preserve provenance defaults and exclusions."""
    module = load_pipeline_module()
    monkeypatch.delenv("CARNOT_REPO_ROOT", raising=False)

    assert module.get_repo_root().name == "carnot"

    eligible = module.TraceEvent(
        source_experiment=1,
        benchmark="b",
        domain="d",
        model_name="m",
        case_id="c",
        response_mode="r",
        verifier_path="v",
        actual_error=True,
        detected=True,
        error_types=("e",),
        descriptions=("desc",),
        confidence=0.9,
    )
    assert eligible.exclusion_reason is None

    low_confidence = module.TraceEvent(
        source_experiment=1,
        benchmark="b",
        domain="d",
        model_name="m",
        case_id="c",
        response_mode="r",
        verifier_path="v",
        actual_error=True,
        detected=True,
        error_types=("e",),
        descriptions=("desc",),
        confidence=0.1,
    )
    assert low_confidence.exclusion_reason == "low_confidence"

    missing_taxonomy = module.TraceEvent(
        source_experiment=1,
        benchmark="b",
        domain="d",
        model_name="m",
        case_id="c",
        response_mode="r",
        verifier_path="v",
        actual_error=True,
        detected=True,
        error_types=(),
        descriptions=(),
        confidence=0.9,
    )
    assert missing_taxonomy.exclusion_reason == "missing_error_taxonomy"

    true_negative = module.TraceEvent(
        source_experiment=1,
        benchmark="b",
        domain="d",
        model_name="m",
        case_id="c",
        response_mode="r",
        verifier_path="v",
        actual_error=False,
        detected=False,
        error_types=(),
        descriptions=(),
        confidence=0.9,
    )
    assert true_negative.outcome == "true_negative"
    assert true_negative.exclusion_reason is None

    assert module._relative_path(Path("/tmp/outside.json"), tmp_path) == "/tmp/outside.json"
    assert module._dedupe_preserve(["", "dup", "dup", "solo"]) == ("dup", "solo")
    assert module._combine_error_type(None, None) == "unknown_violation"

    reliability = module._build_reliability_stats([true_negative])
    assert reliability[0]["true_negatives"] == 1


# REQ-VERIFY-030
def test_internal_extractors_cover_skip_and_fallback_branches():
    """REQ-VERIFY-030: fallback extraction uses verifier-visible artifacts."""
    module = load_pipeline_module()

    exp219 = {
        "paired_runs": [
            {"mode": "baseline", "model_name": "skip-me", "cases": []},
            {
                "mode": "verify_only",
                "model_name": "Qwen3.5-0.8B",
                "cases": [
                    {
                        "case_id": "fallback",
                        "response_mode": "structured_json",
                        "correct": False,
                        "flagged": True,
                        "verification": {
                            "semantic_grounding": {
                                "violations": ["not-a-dict"],
                                "refinement_applied": False,
                            },
                            "violations": [
                                "not-a-dict",
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": "fallback violation",
                                    "metadata": {
                                        "taxonomy_hint": "omitted_premises",
                                        "violation_type": "missing_quantity_coverage",
                                    },
                                },
                            ],
                        },
                    }
                ],
            },
        ]
    }
    events = module._extract_exp219_events(exp219)
    assert len(events) == 1
    assert events[0].error_types == ("omitted_premises:missing_quantity_coverage",)

    assert module._classify_humaneval_text("unterminated triple-quoted string") == "syntax_error"
    assert module._classify_humaneval_text("parameter 'x' annotated as int") == (
        "annotation_feedback"
    )
    assert module._classify_humaneval_text("prompt_examples check failed") == "property_violation"
    assert module._classify_humaneval_text("plain failure") == "humaneval_failure"

    assert module._extract_humaneval_errors(
        {"execution_plus_property": {}, "error_type": "failure"}
    ) == (("humaneval_failure",), ("failure",))

    constraint_errors = module._extract_constraint_ir_errors(
        {
            "evaluation": {
                "constraint_results": [
                    {"status": "satisfied", "type": "x", "family": "literal"},
                    "not-a-dict",
                    {"status": "violated", "type": "semantic_property", "family": "semantic"},
                ]
            }
        }
    )
    assert constraint_errors == (
        ("semantic:semantic_property",),
        ("semantic:semantic_property",),
    )

    assert module._sanitize_repair_prompt("other", "  untouched prompt  ") == "untouched prompt"


# REQ-VERIFY-031, REQ-VERIFY-032
def test_snippet_fallback_and_policy_update_branches_cover_success_paths():
    """REQ-VERIFY-031 and REQ-VERIFY-032: repair snippets and policy updates."""
    module = load_pipeline_module()

    exp220 = {
        "experiment": 220,
        "benchmark": "humaneval_property",
        "paired_runs": [
            {
                "mode": "verify_repair",
                "model_name": "Qwen3.5-0.8B",
                "cases": [
                    {
                        "case_id": "success-case",
                        "repaired": True,
                        "history": [
                            {
                                "iteration": 1,
                                "repair_prompt": (
                                    "HumanEval test failure:\n"
                                    "  - AssertionError: fixed now\n\n"
                                    "Write ONLY the corrected function body. No markdown fences.\n"
                                    "Indent with 4 spaces."
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
    }
    exp221 = {"experiment": 221, "benchmark": "constraint_ir", "paired_runs": []}
    snippets = module._collect_repair_snippets(exp220, exp221)
    assert snippets[0]["successful_cases"] == 1
    assert snippets[0]["snippet_id"] == "humaneval_property:official_test_failure"

    updates = module._build_policy_updates(
        [
            {
                "model_name": "Qwen3.5-0.8B",
                "benchmark": "constraint_ir",
                "domain": "instruction_surface_only",
                "n_cases": 1,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
                "precision": 1.0,
                "recall": 1.0,
            },
            {
                "model_name": "Qwen3.5-0.8B",
                "benchmark": "gsm8k_semantic",
                "domain": "live_gsm8k_semantic_failure",
                "n_cases": 1,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
                "precision": 1.0,
                "recall": 1.0,
            },
        ],
        sample_policy(),
    )
    by_domain = {item["domain"]: item["recommended_action"] for item in updates}
    assert by_domain["instruction_surface_only"] == "reuse_contract_patch"
    assert by_domain["live_gsm8k_semantic_failure"] == "promote_memory_reuse"
