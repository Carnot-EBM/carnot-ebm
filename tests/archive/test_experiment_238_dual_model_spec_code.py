"""Spec: REQ-CODE-028, REQ-CODE-029, REQ-CODE-030,
SCENARIO-CODE-026, SCENARIO-CODE-027, SCENARIO-CODE-028.
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_238_dual_model_spec_code.py"
    python_dir = str(repo_root / "python")
    with_removed = False
    if python_dir in sys.path:
        sys.path.remove(python_dir)
        with_removed = True
    spec = importlib.util.spec_from_file_location(
        "experiment_238_dual_model_spec_code",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if with_removed:
            sys.path.insert(0, python_dir)
    return module


def make_case(case_id: str, *, dataset_idx: int = 0) -> dict[str, object]:
    prompt_seed = 3000 + dataset_idx
    return {
        "case_id": case_id,
        "dataset_idx": dataset_idx,
        "task_id": f"HumanEval/{dataset_idx}",
        "prompt": f"def fn_{dataset_idx}(x: int) -> int:\n    pass\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        "entry_point": f"fn_{dataset_idx}",
        "sample_position": dataset_idx + 1,
        "prompt_seeds": {
            "baseline": prompt_seed,
            "verify_only": prompt_seed,
            "verify_repair": prompt_seed,
        },
    }


def make_case_result(
    case: dict[str, object],
    *,
    baseline: bool,
    pbt: bool,
    spec: bool,
    repair: bool,
    repaired: bool = False,
    n_repairs: int = 0,
) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "dataset_idx": case["dataset_idx"],
        "task_id": case["task_id"],
        "entry_point": case["entry_point"],
        "baseline": {
            "official_passed": baseline,
            "body": "return x",
            "candidate_code": "def fn(x):\n    return x\n",
        },
        "official_tests_verify_only": {"accepted": baseline},
        "pbt_verify_only": {
            "accepted": pbt,
            "harness_passing_rejected_by_pbt": bool(baseline and not pbt),
        },
        "spec_aware_verify_only": {
            "accepted": spec,
            "harness_passing_rejected_by_specs": bool(pbt and not spec),
        },
        "verify_repair": {
            "accepted": repair,
            "official_passed": repair,
            "repaired": repaired,
            "n_repairs": n_repairs,
            "final_body": "return x + 1",
            "final_code": "def fn(x):\n    return x + 1\n",
        },
        "history": [],
    }


def test_build_parser_defaults_and_load_shared_cohort_from_exp227_artifact(
    tmp_path: Path,
) -> None:
    """REQ-CODE-028, SCENARIO-CODE-026: both models reuse one checked-in cohort."""
    module = load_module()
    reference_path = tmp_path / "experiment_227_results.json"
    cases = [make_case("humaneval-7", dataset_idx=7), make_case("humaneval-3", dataset_idx=3)]
    reference_path.write_text(
        json.dumps(
            {
                "experiment": 227,
                "run_date": "20260412",
                "metadata": {
                    "model_name": "Qwen3.5-0.8B",
                    "reference_experiment": 208,
                },
                "cohort": {
                    "case_count": len(cases),
                    "case_ids": [case["case_id"] for case in cases],
                    "task_ids": [case["task_id"] for case in cases],
                    "cases": cases,
                },
            }
        ),
        encoding="utf-8",
    )

    parser = module.build_parser()
    args = parser.parse_args([])
    loaded_cases, cohort_meta = module.load_shared_cohort(reference_path)

    assert "T" in module.utc_now()
    assert args.reference_artifact == module.default_reference_artifact_path()
    assert args.output == module.default_output_path()
    assert args.checkpoint_dir == module.default_checkpoint_dir()
    assert args.max_repairs == 3
    assert len(loaded_cases) == 2
    assert loaded_cases[0]["case_id"] == "humaneval-7"
    assert loaded_cases[1]["task_id"] == "HumanEval/3"
    assert (
        loaded_cases[0]["prompt_seeds"]["baseline"]
        == loaded_cases[0]["prompt_seeds"]["verify_only"]
    )
    assert (
        loaded_cases[0]["prompt_seeds"]["verify_only"]
        == loaded_cases[0]["prompt_seeds"]["verify_repair"]
    )
    assert cohort_meta == {
        "source_artifact": str(reference_path),
        "source_experiment": 227,
        "reference_experiment": 208,
        "reference_run_date": "20260412",
        "case_count": 2,
    }

    missing_cohort_path = tmp_path / "missing_cohort.json"
    missing_cohort_path.write_text(json.dumps({"experiment": 227, "cohort": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort block"):
        module.load_shared_cohort(missing_cohort_path)

    missing_cases_path = tmp_path / "missing_cases.json"
    missing_cases_path.write_text(json.dumps({"experiment": 227, "cohort": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort.cases"):
        module.load_shared_cohort(missing_cases_path)

    bad_seed_path = tmp_path / "bad_seed.json"
    bad_case = make_case("humaneval-8", dataset_idx=8)
    bad_case["prompt_seeds"]["verify_repair"] = 9999
    bad_seed_path.write_text(
        json.dumps({"experiment": 227, "cohort": {"cases": [bad_case]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt seeds"):
        module.load_shared_cohort(bad_seed_path)

    verifier = module.build_spec_verifier(17)
    assert verifier._pbt_verifier._max_examples == 17
    assert module._display_path(tmp_path / "outside.json") == str(tmp_path / "outside.json")


def test_build_spec_repair_prompt_includes_official_pbt_spec_and_ranked_hints() -> None:
    """REQ-CODE-028: repair prompts keep one format while adding spec-aware feedback."""
    module = load_module()
    prompt = module.build_spec_repair_prompt(
        make_case("humaneval-1", dataset_idx=1),
        previous_body="return xs",
        evaluation={
            "official_tests": {
                "passed": False,
                "error_type": "failure",
                "error_message": "AssertionError: expected sorted output",
                "stdout": "",
            },
            "instrumentation": {
                "constraint_feedback": ["missing defensive branch"],
                "dynamic_violations": ["NameError: xs is not defined"],
            },
            "pbt": {
                "violations": ["sorted_output failed for input=([2, 1],)"],
                "repair_feedback": "Return the items in sorted order.",
            },
            "explicit_specs": {
                "violations": ["sorted_output (postconditions) failed: returned [2, 1]"],
                "repair_hints": [
                    {
                        "strategy_name": "ordering_fix",
                        "rationale": (
                            "Ordering fixes are the top historical win on this trace family."
                        ),
                    },
                    {
                        "strategy_name": "syntax_cleanup",
                        "rationale": "",
                    },
                ],
            },
        },
        repair_idx=1,
    )

    assert "repair attempt 2" in prompt.lower()
    assert "AssertionError: expected sorted output" in prompt
    assert "missing defensive branch" in prompt
    assert "NameError: xs is not defined" in prompt
    assert "sorted_output failed for input=([2, 1],)" in prompt
    assert "Explicit spec findings:" in prompt
    assert "ordering_fix" in prompt
    assert "syntax_cleanup" in prompt
    assert "Write ONLY the corrected function body" in prompt


def test_evaluate_candidate_combines_official_pbt_and_explicit_spec_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-028: one evaluation captures official, PBT, and explicit-spec traces."""
    module = load_module()

    monkeypatch.setattr(
        module,
        "execute_humaneval",
        lambda code, problem, timeout=5.0: module.HarnessResult(
            passed=True,
            error_type="none",
            error_message="",
            stdout="ok",
        ),
    )
    monkeypatch.setattr(
        module,
        "run_instrumentation",
        lambda code, prompt, entry_point, official_tests=None: {
            "constraint_feedback": [],
            "dynamic_violations": [],
            "n_static_violations": 0,
            "n_dynamic_violations": 0,
            "probe_inputs": [{"x": 1}],
        },
    )

    class FakeConstraint:
        def __init__(self, description: str) -> None:
            self.description = description

    class FakePBTResult:
        verified = False
        derived_properties = [SimpleNamespace(name="sorted_output", source="prompt_intent")]
        failures = [SimpleNamespace(property_name="sorted_output", source="prompt_intent")]
        wall_clock_seconds = 0.25
        max_examples = 32

        def to_constraint_results(self):
            return [FakeConstraint("sorted_output failed for input=([2, 1],)")]

        def repair_feedback(self) -> str:
            return "sorted_output failed for input=([2, 1],)"

    class FakePBTVerifier:
        def __init__(self, max_examples: int) -> None:
            assert max_examples == 32

        def verify(self, code: str, prompt: str, entry_point: str, official_tests: str):
            return FakePBTResult()

    class FakeClauseResult:
        def __init__(self, description: str, *, status: str = "violated") -> None:
            self.kind = "sorted_output"
            self.family = "postconditions"
            self.text = "return the items in sorted order"
            self.status = status
            self.checked_by = "explicit_spec"
            self.detail = description
            self.sources = ("prompt_intent",)
            self.trace_refs = ("exp226:humaneval-1",)
            self.matched_properties = ("sorted_output",)

        def to_constraint_result(self):
            return SimpleNamespace(description=f"spec_code: {self.detail}")

    class FakeSpecResult:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(
                task_id="HumanEval/2",
                case_id="humaneval-2",
                entry_point="fn_2",
                run_date="20260413",
            )
            self.spec_clause_results = (
                FakeClauseResult("sorted_output failed: returned [2, 1]"),
                FakeClauseResult("sorted_output satisfied", status="satisfied"),
            )
            self.repair_hints = (
                SimpleNamespace(
                    to_dict=lambda: {
                        "strategy_name": "ordering_fix",
                        "rationale": "Use sorted(...) instead of returning the input unchanged.",
                    }
                ),
            )

    class FakeSpecVerifier:
        def verify(
            self,
            code: str,
            prompt: str,
            entry_point: str,
            official_tests: str,
            *,
            task_id: str | None = None,
            case_id: str | None = None,
        ):
            assert task_id == "HumanEval/2"
            assert case_id == "humaneval-2"
            return FakeSpecResult()

    monkeypatch.setattr(module, "PBTCodeVerifier", FakePBTVerifier)
    monkeypatch.setattr(module, "build_spec_verifier", lambda pbt_max_examples: FakeSpecVerifier())

    result = module.evaluate_candidate(
        make_case("humaneval-2", dataset_idx=2),
        "def fn_2(x: int) -> int:\n    return x\n",
        pbt_max_examples=32,
    )

    assert result["official_tests"]["passed"] is True
    assert result["pbt"]["n_failures"] == 1
    assert result["pbt"]["verified"] is False
    assert result["explicit_specs"]["matched"] is True
    assert result["explicit_specs"]["n_violations"] == 1
    assert result["explicit_specs"]["violations"] == [
        "spec_code: sorted_output failed: returned [2, 1]"
    ]
    assert result["explicit_specs"]["repair_hints"][0]["strategy_name"] == "ordering_fix"
    assert result["stage_acceptance"] == {
        "official_tests_verify_only": True,
        "pbt_verify_only": False,
        "spec_aware_verify_only": False,
    }


def test_run_case_repairs_even_when_only_the_spec_layer_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-028: spec-aware verify-repair can fire on harness-passing baselines."""
    module = load_module()
    case = make_case("humaneval-4", dataset_idx=4)
    generated_prompts: list[str] = []

    monkeypatch.setattr(
        module,
        "_generate_text",
        lambda *, model, tokenizer, prompt, prompt_seed, max_new_tokens: (
            generated_prompts.append(prompt)
            or ["return xs", "return sorted(xs)"][len(generated_prompts) - 1]
        ),
    )
    monkeypatch.setattr(
        module,
        "build_candidate_code",
        lambda prompt, body: f"{prompt}    {body}\n",
    )

    evaluations = iter(
        [
            {
                "official_tests": {
                    "passed": True,
                    "error_type": "none",
                    "error_message": "",
                    "stdout": "",
                },
                "instrumentation": {
                    "constraint_feedback": [],
                    "dynamic_violations": [],
                    "n_static_violations": 0,
                    "n_dynamic_violations": 0,
                    "probe_inputs": [{"xs": [2, 1]}],
                },
                "pbt": {
                    "verified": True,
                    "n_failures": 0,
                    "violations": [],
                    "repair_feedback": "",
                    "derived_properties": [],
                    "failure_records": [],
                },
                "explicit_specs": {
                    "matched": True,
                    "n_violations": 1,
                    "violations": ["spec_code: sorted_output failed"],
                    "repair_hints": [{"strategy_name": "ordering_fix", "rationale": "sort it"}],
                },
                "stage_acceptance": {
                    "official_tests_verify_only": True,
                    "pbt_verify_only": True,
                    "spec_aware_verify_only": False,
                },
                "latency_seconds": 0.1,
            },
            {
                "official_tests": {
                    "passed": True,
                    "error_type": "none",
                    "error_message": "",
                    "stdout": "",
                },
                "instrumentation": {
                    "constraint_feedback": [],
                    "dynamic_violations": [],
                    "n_static_violations": 0,
                    "n_dynamic_violations": 0,
                    "probe_inputs": [{"xs": [2, 1]}],
                },
                "pbt": {
                    "verified": True,
                    "n_failures": 0,
                    "violations": [],
                    "repair_feedback": "",
                    "derived_properties": [],
                    "failure_records": [],
                },
                "explicit_specs": {
                    "matched": True,
                    "n_violations": 0,
                    "violations": [],
                    "repair_hints": [],
                },
                "stage_acceptance": {
                    "official_tests_verify_only": True,
                    "pbt_verify_only": True,
                    "spec_aware_verify_only": True,
                },
                "latency_seconds": 0.1,
            },
        ]
    )
    monkeypatch.setattr(
        module, "evaluate_candidate", lambda case, code, pbt_max_examples: next(evaluations)
    )
    monkeypatch.setattr(
        module,
        "build_spec_repair_prompt",
        lambda case, previous_body, evaluation, repair_idx: f"repair prompt {repair_idx + 1}",
    )

    result = module.run_case(
        case,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=48,
    )

    assert len(generated_prompts) == 2
    assert result["baseline"]["official_passed"] is True
    assert result["official_tests_verify_only"]["accepted"] is True
    assert result["pbt_verify_only"]["accepted"] is True
    assert result["spec_aware_verify_only"]["accepted"] is False
    assert result["verify_repair"]["accepted"] is True
    assert result["verify_repair"]["repaired"] is True
    assert result["verify_repair"]["n_repairs"] == 1
    assert result["history"][1]["repair_prompt"] == "repair prompt 1"


def test_run_case_returns_early_when_spec_aware_baseline_is_already_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-028: clean baselines do not spend repair budget."""
    module = load_module()
    case = make_case("humaneval-5", dataset_idx=5)
    monkeypatch.setattr(
        module,
        "_generate_text",
        lambda *, model, tokenizer, prompt, prompt_seed, max_new_tokens: "return x + 1",
    )
    monkeypatch.setattr(
        module,
        "build_candidate_code",
        lambda prompt, body: f"{prompt}    {body}\n",
    )
    monkeypatch.setattr(
        module,
        "evaluate_candidate",
        lambda case, code, pbt_max_examples: {
            "official_tests": {
                "passed": True,
                "error_type": "none",
                "error_message": "",
                "stdout": "",
            },
            "instrumentation": {
                "constraint_feedback": [],
                "dynamic_violations": [],
                "n_static_violations": 0,
                "n_dynamic_violations": 0,
                "probe_inputs": [],
            },
            "pbt": {
                "verified": True,
                "n_failures": 0,
                "violations": [],
                "repair_feedback": "",
                "derived_properties": [],
                "failure_records": [],
            },
            "explicit_specs": {
                "matched": True,
                "n_violations": 0,
                "violations": [],
                "repair_hints": [],
            },
            "stage_acceptance": {
                "official_tests_verify_only": True,
                "pbt_verify_only": True,
                "spec_aware_verify_only": True,
            },
            "latency_seconds": 0.1,
        },
    )

    result = module.run_case(
        case,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=64,
    )

    assert result["verify_repair"]["accepted"] is True
    assert result["verify_repair"]["n_repairs"] == 0
    assert len(result["history"]) == 1


def test_load_checkpoint_edges_and_run_benchmark_resumes_after_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-030, SCENARIO-CODE-028: partial checkpoints preserve completed traces."""
    module = load_module()
    checkpoint_path = tmp_path / "exp238_model.json"
    cases = [make_case("humaneval-a", dataset_idx=10), make_case("humaneval-b", dataset_idx=11)]
    case_ids = [str(case["case_id"]) for case in cases]

    assert module.load_checkpoint(checkpoint_path, case_ids) == {
        "case_ids": case_ids,
        "results_by_case": {},
    }

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(
        json.dumps({"case_ids": ["other"], "results_by_case": {"other": {}}}),
        encoding="utf-8",
    )
    assert module.load_checkpoint(mismatch_path, case_ids) == {
        "case_ids": case_ids,
        "results_by_case": {},
    }

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"case_ids": case_ids, "results_by_case": []}),
        encoding="utf-8",
    )
    assert module.load_checkpoint(invalid_path, case_ids) == {
        "case_ids": case_ids,
        "results_by_case": {},
    }

    call_count = {"n": 0}

    def fake_run_case(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return make_case_result(
                cases[0],
                baseline=True,
                pbt=True,
                spec=False,
                repair=True,
                repaired=True,
                n_repairs=1,
            )
        raise RuntimeError("GPU disappeared mid-run")

    monkeypatch.setattr(module, "run_case", fake_run_case)

    with pytest.raises(RuntimeError, match="GPU disappeared"):
        module.run_benchmark(
            cases,
            model=object(),
            tokenizer=object(),
            device_str="cuda:0",
            checkpoint_path=checkpoint_path,
            checkpoint_interval=10,
            max_repairs=3,
            pbt_max_examples=64,
            max_new_tokens=64,
        )

    checkpoint_payload = module.load_checkpoint(checkpoint_path, case_ids)
    assert list(checkpoint_payload["results_by_case"]) == ["humaneval-a"]

    monkeypatch.setattr(
        module,
        "run_case",
        lambda *args, **kwargs: make_case_result(
            cases[1],
            baseline=False,
            pbt=False,
            spec=False,
            repair=False,
        ),
    )
    resumed = module.run_benchmark(
        cases,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=64,
    )

    assert [row["case_id"] for row in resumed] == ["humaneval-a", "humaneval-b"]

    interval_checkpoint = tmp_path / "interval.json"
    monkeypatch.setattr(
        module,
        "run_case",
        lambda *args, **kwargs: (
            make_case_result(
                cases[0],
                baseline=True,
                pbt=True,
                spec=True,
                repair=True,
            )
            if kwargs["device_str"] == "cuda:0"
            else make_case_result(
                cases[1],
                baseline=True,
                pbt=True,
                spec=True,
                repair=True,
            )
        ),
    )
    module.run_benchmark(
        cases,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        checkpoint_path=interval_checkpoint,
        checkpoint_interval=1,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=64,
    )
    assert interval_checkpoint.exists()


def test_summaries_and_comparison_report_stagewise_spec_increment() -> None:
    """REQ-CODE-029, SCENARIO-CODE-027: summaries isolate the explicit-spec layer."""
    module = load_module()
    cases = [make_case("humaneval-0", dataset_idx=0), make_case("humaneval-1", dataset_idx=1)]
    qwen_results = [
        make_case_result(
            cases[0], baseline=True, pbt=False, spec=False, repair=True, repaired=True, n_repairs=1
        ),
        make_case_result(cases[1], baseline=False, pbt=False, spec=False, repair=False),
    ]
    gemma_results = [
        make_case_result(
            cases[0], baseline=True, pbt=True, spec=False, repair=True, repaired=True, n_repairs=1
        ),
        make_case_result(cases[1], baseline=True, pbt=True, spec=True, repair=True),
    ]

    qwen_summary = module.summarize_model_results(qwen_results, n_bootstrap=20, seed=7)
    gemma_summary = module.summarize_model_results(gemma_results, n_bootstrap=20, seed=11)

    assert qwen_summary["stages"]["baseline"]["accepted_pass_at_1"] == 0.5
    assert qwen_summary["stages"]["pbt_verify_only"]["accepted_pass_at_1"] == 0.0
    assert qwen_summary["stages"]["spec_aware_verify_only"]["added_rejections_over_pbt"] == 0
    assert gemma_summary["stages"]["spec_aware_verify_only"]["accepted_pass_at_1"] == 0.5
    assert gemma_summary["paired_deltas"]["spec_over_pbt"]["delta"] == -0.5
    assert gemma_summary["paired_deltas"]["spec_over_pbt"]["ci_lower"] <= -0.5
    assert gemma_summary["paired_deltas"]["spec_over_pbt"]["ci_upper"] >= -0.5
    assert gemma_summary["verify_repair"]["n_repaired"] == 1

    comparison = module.build_comparison_summary(
        {
            "Qwen3.5-0.8B": {
                "model_name": "Qwen3.5-0.8B",
                "run_status": "complete",
                "per_problem_results": qwen_results,
                "statistics": qwen_summary,
            },
            "Gemma4-E4B-it": {
                "model_name": "Gemma4-E4B-it",
                "run_status": "complete",
                "per_problem_results": gemma_results,
                "statistics": gemma_summary,
            },
        },
        n_bootstrap=20,
        seed=13,
        repair_budget=3,
    )

    assert comparison["paired_case_count"] == 2
    assert comparison["shared_repair_budget"] == 3
    assert comparison["shared_verifier_stack"] == [
        "official_tests",
        "pbt",
        "explicit_specs",
    ]
    assert comparison["stage_deltas"]["baseline"]["gemma_minus_qwen"] == 0.5
    assert comparison["stage_deltas"]["baseline"]["ci_lower"] <= 0.5
    assert comparison["stage_deltas"]["baseline"]["ci_upper"] >= 0.5
    assert comparison["stage_outcomes"]["spec_aware_verify_only"]["gemma_only"] == 1
    assert "explicit spec layer" in comparison["technical_report_summary"]["paragraph"].lower()
    assert module._model_checkpoint_path(
        Path("results/checkpoints/experiment_238"), "Gemma4-E4B-it"
    ) == (Path("results/checkpoints/experiment_238") / "humaneval_spec_dual__gemma4_e4b_it.json")
    empty_summary = module.summarize_model_results([], n_bootstrap=8, seed=1)
    assert empty_summary["stages"]["baseline"]["accepted_pass_at_1"] == 0.0
    assert empty_summary["paired_deltas"]["spec_over_pbt"]["ci_upper"] == 0.0
    assert empty_summary["verify_repair"]["n_repaired"] == 0

    missing_model = module.build_comparison_summary({}, n_bootstrap=8, seed=1, repair_budget=3)
    assert missing_model["paired_case_count"] == 0

    qwen_non_list = module.build_comparison_summary(
        {
            "Qwen3.5-0.8B": {"per_problem_results": {}, "statistics": {}, "run_status": "blocked"},
            "Gemma4-E4B-it": {"per_problem_results": [], "statistics": {}, "run_status": "blocked"},
        },
        n_bootstrap=8,
        seed=1,
        repair_budget=3,
    )
    assert qwen_non_list["paired_case_count"] == 0

    no_pairs = module.build_comparison_summary(
        {
            "Qwen3.5-0.8B": {
                "model_name": "Qwen3.5-0.8B",
                "run_status": "complete",
                "per_problem_results": [
                    make_case_result(cases[0], baseline=True, pbt=True, spec=True, repair=True)
                ],
                "statistics": qwen_summary,
            },
            "Gemma4-E4B-it": {
                "model_name": "Gemma4-E4B-it",
                "run_status": "complete",
                "per_problem_results": [
                    make_case_result(
                        make_case("other-case", dataset_idx=9),
                        baseline=True,
                        pbt=True,
                        spec=True,
                        repair=True,
                    )
                ],
                "statistics": gemma_summary,
            },
        },
        n_bootstrap=8,
        seed=1,
        repair_budget=3,
    )
    assert no_pairs["paired_case_count"] == 0
    assert no_pairs["stage_deltas"]["baseline"]["ci_upper"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"cohort": []},
        {"cohort": {"cases": []}},
        {"cohort": {"cases": [1]}},
        {"cohort": {"cases": [{"case_id": "x", "prompt_seeds": None}]}},
        {
            "cohort": {
                "cases": [
                    {
                        "case_id": "x",
                        "prompt_seeds": {"baseline": 1, "verify_only": 2, "verify_repair": 1},
                    }
                ]
            }
        },
    ],
)
def test_load_shared_cohort_validation_errors_and_display_path_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    """REQ-CODE-028: malformed cohort references fail fast without silent fallback."""
    module = load_module()
    path = tmp_path / "bad_reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.load_shared_cohort(path)

    monkeypatch.setattr(module, "get_repo_root", lambda: tmp_path / "elsewhere")
    assert module._display_path(path) == str(path)


def test_runtime_helpers_cover_seed_cuda_load_unload_generate_and_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-028: runtime helpers stay deterministic and fail honestly."""
    module = load_module()
    seed_calls: list[tuple[str, int]] = []
    empty_cache_calls: list[str] = []
    mem_queries: list[int] = []
    load_calls: list[tuple[str, str]] = []
    generate_calls: list[tuple[str, int]] = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def manual_seed_all(self, seed: int) -> None:
            seed_calls.append(("torch.cuda", seed))

        def device_count(self) -> int:
            return 2

        def mem_get_info(self, index: int) -> tuple[int, int]:
            mem_queries.append(index)
            if index == 0:
                raise RuntimeError("device query failed")
            return (200, 400)

        def empty_cache(self) -> None:
            empty_cache_calls.append("empty")

    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: seed_calls.append(("torch", seed)),
        cuda=FakeCuda(),
    )
    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: seed_calls.append(("numpy", seed)))
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    import carnot.inference.model_loader as model_loader

    monkeypatch.setattr(
        model_loader,
        "load_model",
        lambda model_name, device="cpu": (
            load_calls.append((model_name, device)) or ("model", "tok")
        ),
    )
    monkeypatch.setattr(
        model_loader,
        "generate",
        lambda model, tokenizer, prompt, max_new_tokens=256: (
            generate_calls.append((prompt, max_new_tokens)) or "return 1"
        ),
    )

    module._seed_runtime(123)
    assert ("numpy", 123) in seed_calls
    assert ("torch", 123) in seed_calls
    assert ("torch.cuda", 123) in seed_calls
    assert module._best_cuda_device() == "cuda:1"
    assert mem_queries == [0, 1]
    assert (
        module.checkpoint_path(tmp_path / "checkpoints", "Gemma4-E4B-it").name
        == "gemma4_e4b_it.json"
    )
    assert (
        module._checkpoint_path(tmp_path / "checkpoints", "Gemma4-E4B-it").name
        == "gemma4_e4b_it.json"
    )
    assert module.build_generation_prompt(make_case("humaneval-0", dataset_idx=0)).startswith(
        "You are an expert Python programmer."
    )
    assert module._serialize_repair_hints([{"strategy_name": "ordering_fix"}]) == [
        {"strategy_name": "ordering_fix"}
    ]

    class BareClause:
        status = "violated"
        kind = "sorted_output"
        family = "postconditions"
        detail = "returned [2, 1]"

    assert module._spec_violation_texts(SimpleNamespace(spec_clause_results=(BareClause(),))) == [
        "sorted_output (postconditions) failed: returned [2, 1]"
    ]

    verifier = module.build_spec_verifier(55)
    assert verifier._pbt_verifier._max_examples == 55
    assert module.checkpoint_path(tmp_path / "checkpoints", "Gemma4-E4B-it") == (
        tmp_path / "checkpoints" / "gemma4_e4b_it.json"
    )

    model, tokenizer, device = module._load_live_model("hf/test")
    assert (model, tokenizer, device) == ("model", "tok", "cuda:1")
    assert load_calls == [("hf/test", "cuda:1")]
    assert os.environ["CARNOT_FORCE_LIVE"] == "1"
    assert os.environ["CARNOT_FORCE_CPU"] == "0"

    generated = module._generate_text(
        model="model",
        tokenizer="tok",
        prompt="prompt",
        prompt_seed=9,
        max_new_tokens=33,
    )
    assert generated == "return 1"
    assert generate_calls == [("prompt", 33)]

    module._unload_live_model("model", "tok")
    assert empty_cache_calls == ["empty"]

    monkeypatch.setattr(model_loader, "load_model", lambda model_name, device="cpu": (None, None))
    with pytest.raises(RuntimeError, match="Failed to load live model"):
        module._load_live_model("hf/test")

    fake_torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        module._best_cuda_device()

    assert module._serialize_repair_hints([{"strategy_name": "dict_hint"}]) == [
        {"strategy_name": "dict_hint"}
    ]

    class FallbackClause:
        status = "violated"
        kind = "typed_output"
        family = "postconditions"
        detail = "returned None"

        def to_constraint_result(self):
            return None

    assert module._spec_violation_texts(
        SimpleNamespace(spec_clause_results=(FallbackClause(),))
    ) == ["typed_output (postconditions) failed: returned None"]


def test_run_case_skips_repairs_when_baseline_is_already_spec_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-028: full-stack clean baselines do not enter the repair loop."""
    module = load_module()
    case = make_case("humaneval-5", dataset_idx=5)
    generated_prompts: list[str] = []

    monkeypatch.setattr(
        module,
        "_generate_text",
        lambda *, model, tokenizer, prompt, prompt_seed, max_new_tokens: (
            generated_prompts.append(prompt) or "return x + 1"
        ),
    )
    monkeypatch.setattr(
        module,
        "build_candidate_code",
        lambda prompt, body: f"{prompt}    {body}\n",
    )
    monkeypatch.setattr(
        module,
        "evaluate_candidate",
        lambda case, code, pbt_max_examples: {
            "official_tests": {
                "passed": True,
                "error_type": "none",
                "error_message": "",
                "stdout": "",
            },
            "instrumentation": {
                "constraint_feedback": [],
                "dynamic_violations": [],
                "n_static_violations": 0,
                "n_dynamic_violations": 0,
                "probe_inputs": [{}],
            },
            "pbt": {
                "verified": True,
                "n_failures": 0,
                "violations": [],
                "repair_feedback": "",
                "derived_properties": [],
                "failure_records": [],
            },
            "explicit_specs": {
                "matched": True,
                "n_violations": 0,
                "violations": [],
                "repair_hints": [],
            },
            "stage_acceptance": {
                "official_tests_verify_only": True,
                "pbt_verify_only": True,
                "spec_aware_verify_only": True,
            },
            "latency_seconds": 0.1,
        },
    )

    result = module.run_case(
        case,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
    )

    assert generated_prompts == [module.build_generation_prompt(case)]
    assert result["verify_repair"]["accepted"] is True
    assert result["verify_repair"]["n_repairs"] == 0
    assert len(result["history"]) == 1


def test_run_benchmark_checkpoint_interval_save_and_empty_comparison_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-029, REQ-CODE-030: empty paired subsets degrade cleanly and still checkpoint."""
    module = load_module()
    cases = [make_case("humaneval-6", dataset_idx=6), make_case("humaneval-7", dataset_idx=7)]
    checkpoint_path = tmp_path / "interval.json"
    save_calls: list[Path] = []
    real_save = module.save_checkpoint

    monkeypatch.setattr(
        module,
        "run_case",
        lambda case, **kwargs: make_case_result(
            case,
            baseline=True,
            pbt=True,
            spec=True,
            repair=True,
        ),
    )

    def tracking_save(path: Path, payload: dict[str, object]) -> None:
        save_calls.append(path)
        real_save(path, payload)

    monkeypatch.setattr(module, "save_checkpoint", tracking_save)

    results = module.run_benchmark(
        cases,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
    )

    assert len(results) == 2
    assert save_calls == [checkpoint_path, checkpoint_path]

    comparison = module.build_comparison_summary(
        {
            "Qwen3.5-0.8B": {"per_problem_results": {}, "statistics": {}, "run_status": "blocked"},
            "Gemma4-E4B-it": {"per_problem_results": (), "statistics": {}, "run_status": "blocked"},
        },
        n_bootstrap=8,
        seed=1,
        repair_budget=3,
    )

    assert comparison["paired_case_count"] == 0
    assert comparison["stage_deltas"]["baseline"]["gemma_minus_qwen"] == 0.0
    assert comparison["stage_outcomes"]["verify_repair"] == {
        "gemma_only": 0,
        "qwen_only": 0,
        "both": 0,
        "neither": 0,
    }


def test_model_runner_and_live_benchmark_cover_blocked_partial_complete_and_main_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CODE-028: runner orchestration preserves blockers and remains executable."""
    module = load_module()
    cohort = [make_case("humaneval-8", dataset_idx=8)]

    monkeypatch.setattr(
        module, "_load_live_model", lambda hf_id: (_ for _ in ()).throw(RuntimeError("no gpu"))
    )
    blocked = module.run_model_benchmark_cell(
        model_spec={"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
        cohort=cohort,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
        bootstrap_samples=8,
    )
    assert blocked["run_status"] == "blocked"
    assert blocked["blockers"][0]["error"] == "no gpu"

    checkpoint_dir = tmp_path / "cell-checkpoints"
    partial_checkpoint = module._checkpoint_path(checkpoint_dir, "Qwen3.5-0.8B")
    unload_calls: list[str] = []
    monkeypatch.setattr(module, "_load_live_model", lambda hf_id: ("model", "tok", "cuda:0"))
    monkeypatch.setattr(
        module, "_unload_live_model", lambda model, tokenizer: unload_calls.append("unloaded")
    )

    def fake_run_benchmark(
        cases,
        *,
        model,
        tokenizer,
        device_str,
        checkpoint_path,
        checkpoint_interval,
        max_repairs,
        pbt_max_examples,
        max_new_tokens,
    ):
        module.save_checkpoint(
            checkpoint_path,
            {
                "case_ids": [str(case["case_id"]) for case in cases],
                "results_by_case": {
                    str(cases[0]["case_id"]): make_case_result(
                        cases[0],
                        baseline=True,
                        pbt=True,
                        spec=False,
                        repair=True,
                        repaired=True,
                        n_repairs=1,
                    )
                },
            },
        )
        raise RuntimeError("mid-run failure")

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    partial = module.run_model_benchmark_cell(
        model_spec={"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
        cohort=cohort,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
        bootstrap_samples=8,
    )
    assert partial["run_status"] == "partial"
    assert partial["completed_case_count"] == 1
    assert partial["blockers"][0]["error"] == "mid-run failure"
    assert unload_calls == ["unloaded"]
    assert partial["checkpoint_path"] == str(partial_checkpoint)

    args = SimpleNamespace(
        reference_artifact=tmp_path / "reference.json",
        output=tmp_path / "out.json",
        checkpoint_dir=tmp_path / "run-checkpoints",
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
        bootstrap_samples=8,
    )
    monkeypatch.setattr(
        module,
        "load_shared_cohort",
        lambda path: (
            cohort,
            {
                "source_artifact": "results/experiment_227_results.json",
                "source_experiment": 227,
                "reference_experiment": 208,
                "reference_run_date": "20260412",
                "case_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "run_model_benchmark_cell",
        lambda **kwargs: {
            "model_name": kwargs["model_spec"]["name"],
            "model_hf_id": kwargs["model_spec"]["hf_id"],
            "device": "cuda:0",
            "run_status": "complete",
            "completed_case_count": 1,
            "pending_case_count": 0,
            "blockers": [],
            "statistics": module.summarize_model_results(
                [make_case_result(cohort[0], baseline=True, pbt=True, spec=True, repair=True)],
                n_bootstrap=8,
                seed=1,
            ),
            "per_problem_results": [
                make_case_result(cohort[0], baseline=True, pbt=True, spec=True, repair=True)
            ],
        },
    )
    complete_payload = module._run_live_benchmark(args)
    assert complete_payload["run_status"] == "complete"

    monkeypatch.setattr(
        module,
        "run_model_benchmark_cell",
        lambda **kwargs: {
            "model_name": kwargs["model_spec"]["name"],
            "model_hf_id": kwargs["model_spec"]["hf_id"],
            "device": "",
            "run_status": "blocked",
            "completed_case_count": 0,
            "pending_case_count": 1,
            "blockers": [
                {
                    "model_name": kwargs["model_spec"]["name"],
                    "stage": "model_load",
                    "error": "blocked",
                }
            ],
            "statistics": module.summarize_model_results([], n_bootstrap=8, seed=1),
            "per_problem_results": [],
        },
    )
    blocked_payload = module._run_live_benchmark(args)
    assert blocked_payload["run_status"] == "blocked"

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_238_dual_model_spec_code.py"
    guard_output = tmp_path / "guard_results.json"
    guard_reference = tmp_path / "guard_reference.json"
    guard_reference.write_text(
        json.dumps(
            {
                "experiment": 227,
                "run_date": "20260412",
                "metadata": {"reference_experiment": 208},
                "cohort": {"case_count": 1, "cases": cohort},
            }
        ),
        encoding="utf-8",
    )

    import carnot.inference.model_loader as model_loader

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                manual_seed_all=lambda seed: None,
                device_count=lambda: 1,
                mem_get_info=lambda index: (100, 100),
                empty_cache=lambda: None,
            ),
        ),
    )
    monkeypatch.setattr(
        model_loader, "load_model", lambda model_name, device="cpu": ("model", "tok")
    )
    monkeypatch.setattr(
        model_loader,
        "generate",
        lambda model, tokenizer, prompt, max_new_tokens=256: "return x + 1",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module_path),
            "--reference-artifact",
            str(guard_reference),
            "--output",
            str(guard_output),
            "--checkpoint-dir",
            str(tmp_path / "guard-checkpoints"),
            "--bootstrap-samples",
            "8",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")

    assert exit_info.value.code == 0
    assert guard_output.exists()


def test_build_artifact_payload_and_main_write_blocker_aware_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-029, REQ-CODE-030: the final artifact keeps schema and blocker metadata."""
    module = load_module()
    payload = module.build_artifact_payload(
        output_path=Path("results/experiment_238_results.json"),
        cohort=[make_case("humaneval-9", dataset_idx=9)],
        cohort_meta={
            "source_artifact": "results/experiment_227_results.json",
            "source_experiment": 227,
            "reference_experiment": 208,
            "reference_run_date": "20260412",
            "case_count": 1,
        },
        model_runs={
            "Qwen3.5-0.8B": {
                "model_name": "Qwen3.5-0.8B",
                "model_hf_id": "Qwen/Qwen3.5-0.8B",
                "device": "cuda:0",
                "run_status": "complete",
                "completed_case_count": 1,
                "pending_case_count": 0,
                "blockers": [],
                "statistics": {"stages": {"baseline": {"accepted_pass_at_1": 1.0}}},
                "per_problem_results": [{"case_id": "humaneval-9"}],
            },
            "Gemma4-E4B-it": {
                "model_name": "Gemma4-E4B-it",
                "model_hf_id": "google/gemma-4-E4B-it",
                "device": "",
                "run_status": "blocked",
                "completed_case_count": 0,
                "pending_case_count": 1,
                "blockers": [
                    {
                        "model_name": "Gemma4-E4B-it",
                        "stage": "model_load",
                        "error": "CUDA unavailable",
                    }
                ],
                "statistics": {"stages": {}},
                "per_problem_results": [],
            },
        },
        comparison={
            "paired_case_count": 0,
            "stage_deltas": {},
            "stage_outcomes": {},
            "technical_report_summary": {"paragraph": "No paired cases completed."},
        },
        blockers=[
            {
                "model_name": "Gemma4-E4B-it",
                "stage": "model_load",
                "error": "CUDA unavailable",
            }
        ],
        started_at="2026-04-13T10:00:00Z",
        finished_at="2026-04-13T10:05:00Z",
        runtime_seconds=300.0,
        checkpoint_dir=Path("results/checkpoints/experiment_238"),
        max_repairs=3,
        pbt_max_examples=64,
        bootstrap_samples=10000,
        run_status="partial",
    )

    assert payload["experiment"] == 238
    assert payload["benchmark"] == "humaneval_dual_model_spec"
    assert payload["run_date"] == "20260413"
    assert payload["schema"] == {
        "artifact": "carnot.humaneval_dual_model_spec.v1",
        "benchmark_case_schema": "humaneval_dual_model_spec.v1",
    }
    assert payload["metadata"]["source_artifacts"] == [
        "results/experiment_227_results.json",
        "data/research/code_spec_corpus_236.jsonl",
        "results/experiment_225_results.json",
        "results/experiment_226_results.json",
        "results/experiment_227_results.json",
    ]
    assert payload["metadata"]["checkpoint_dir"] == "results/checkpoints/experiment_238"
    assert payload["cohort"]["shared_with_reference_artifact"] is True
    assert payload["model_runs"]["Gemma4-E4B-it"]["run_status"] == "blocked"
    assert payload["blockers"][0]["error"] == "CUDA unavailable"
    assert payload["run_status"] == "partial"

    repo = tmp_path / "repo"
    repo.mkdir()
    results_dir = repo / "results"
    results_dir.mkdir()
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr(module, "_run_live_benchmark", lambda args: payload)

    exit_code = module.main([])

    assert exit_code == 0
    assert (
        json.loads((results_dir / "experiment_238_results.json").read_text(encoding="utf-8"))
        == payload
    )
