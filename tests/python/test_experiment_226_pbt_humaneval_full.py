"""Spec: REQ-CODE-012, REQ-CODE-013, REQ-CODE-014,
SCENARIO-CODE-011, SCENARIO-CODE-012.
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
    module_path = repo_root / "scripts" / "experiment_226_pbt_humaneval_full.py"
    python_dir = str(repo_root / "python")
    with_removed = False
    if python_dir in sys.path:
        sys.path.remove(python_dir)
        with_removed = True
    spec = importlib.util.spec_from_file_location(
        "experiment_226_pbt_humaneval_full",
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
    prompt_seed = 1000 + dataset_idx
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


def test_build_parser_defaults_and_load_humaneval_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-012: the full benchmark defaults to the official 164-case cohort."""
    module = load_module()

    fake_rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def sub(a, b):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate(3, 2) == 1\n",
            "entry_point": "sub",
        },
    ]

    def fake_load_dataset(name: str, *, split: str):
        assert name == "openai_humaneval"
        assert split == "test"
        return fake_rows

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    parser = module.build_parser()
    args = parser.parse_args([])
    cases = module.load_humaneval_cases(sample_seed=226, sample_size=1)

    assert "T" in module.utc_now()
    assert args.sample_seed == 226
    assert args.sample_size is None
    assert args.max_repairs == 3
    assert args.checkpoint_interval == 10
    assert args.output == module.default_output_path()
    assert args.checkpoint == module.default_checkpoint_path()
    assert len(cases) == 1
    assert cases[0]["case_id"] == "humaneval-0"
    assert cases[0]["sample_position"] == 1
    assert cases[0]["prompt_seeds"]["baseline"] == cases[0]["prompt_seeds"]["verify_only"]
    assert cases[0]["prompt_seeds"]["verify_only"] == cases[0]["prompt_seeds"]["verify_repair"]


def test_build_pbt_repair_prompt_includes_harness_static_and_pbt_feedback() -> None:
    """REQ-CODE-012: repair prompts carry official failures plus PBT counterexamples."""
    module = load_module()
    evaluation = {
        "harness": {
            "passed": False,
            "error_type": "failure",
            "error_message": "AssertionError: expected 3",
            "stdout": "",
        },
        "instrumentation": {
            "constraint_feedback": ["missing return on one branch"],
            "dynamic_violations": ["NameError: name 'oops' is not defined"],
        },
        "pbt": {
            "violations": [
                "sorted_output (prompt_intent) failed for input=([2, 1],): returned [2, 1]"
            ],
            "repair_feedback": "sorted_output (prompt_intent) failed",
        },
    }

    prompt = module.build_pbt_repair_prompt(
        make_case("humaneval-1", dataset_idx=1),
        previous_body="return nums",
        evaluation=evaluation,
        repair_idx=1,
    )

    assert "repair attempt 2" in prompt.lower()
    assert "AssertionError: expected 3" in prompt
    assert "missing return on one branch" in prompt
    assert "NameError: name 'oops' is not defined" in prompt
    assert "sorted_output" in prompt
    assert "Write ONLY the corrected function body" in prompt


def test_evaluate_candidate_combines_harness_instrumentation_and_pbt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-012: verify-only evaluation tracks official-test misses surfaced by PBT."""
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
            "detected": False,
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

    class FakeVerifierResult:
        verified = False
        derived_properties = [SimpleNamespace(name="sorted_output", source="prompt_intent")]
        failures = [SimpleNamespace(property_name="sorted_output", source="prompt_intent")]
        wall_clock_seconds = 0.25

        def to_constraint_results(self):
            return [FakeConstraint("sorted_output failed for input=([2, 1],)")]

        def repair_feedback(self) -> str:
            return "sorted_output failed for input=([2, 1],)"

    class FakeVerifier:
        def __init__(self, max_examples: int) -> None:
            assert max_examples == 32

        def verify(
            self,
            code: str,
            prompt: str,
            entry_point: str,
            official_tests: str,
        ) -> FakeVerifierResult:
            return FakeVerifierResult()

    monkeypatch.setattr(module, "PBTCodeVerifier", FakeVerifier)

    result = module.evaluate_candidate(
        make_case("humaneval-2", dataset_idx=2),
        "def fn_2(x: int) -> int:\n    return x\n",
        pbt_max_examples=32,
    )

    assert result["harness"]["passed"] is True
    assert result["detected"] is True
    assert result["accepted"] is False
    assert result["official_test_miss_caught_by_pbt"] is True
    assert result["pbt"]["n_failures"] == 1
    assert result["pbt"]["violations"] == ["sorted_output failed for input=([2, 1],)"]


def test_run_case_repairs_failed_baseline_and_tracks_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-012: failed baselines get up to 3 PBT-guided repair attempts."""
    module = load_module()
    case = make_case("humaneval-3", dataset_idx=3)
    generated_prompts: list[str] = []

    monkeypatch.setattr(
        module,
        "_generate_text",
        lambda *, model, tokenizer, prompt, prompt_seed, max_new_tokens: (
            generated_prompts.append(prompt)
            or ["return 0", "return 1", "return 2"][len(generated_prompts) - 1]
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
                "harness": {
                    "passed": False,
                    "error_type": "failure",
                    "error_message": "baseline failed",
                    "stdout": "",
                },
                "instrumentation": {
                    "constraint_feedback": ["baseline static"],
                    "dynamic_violations": [],
                    "n_static_violations": 1,
                    "n_dynamic_violations": 0,
                    "probe_inputs": [{"x": 1}],
                },
                "pbt": {
                    "n_failures": 1,
                    "violations": ["baseline pbt"],
                    "repair_feedback": "fb",
                },
                "detected": True,
                "accepted": False,
                "official_test_miss_caught_by_pbt": False,
            },
            {
                "harness": {
                    "passed": False,
                    "error_type": "failure",
                    "error_message": "repair one failed",
                    "stdout": "",
                },
                "instrumentation": {
                    "constraint_feedback": [],
                    "dynamic_violations": ["dynamic one"],
                    "n_static_violations": 0,
                    "n_dynamic_violations": 1,
                    "probe_inputs": [{"x": 2}],
                },
                "pbt": {
                    "n_failures": 1,
                    "violations": ["repair one pbt"],
                    "repair_feedback": "fb1",
                },
                "detected": True,
                "accepted": False,
                "official_test_miss_caught_by_pbt": False,
            },
            {
                "harness": {
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
                    "probe_inputs": [{"x": 3}],
                },
                "pbt": {
                    "n_failures": 0,
                    "violations": [],
                    "repair_feedback": "",
                },
                "detected": False,
                "accepted": True,
                "official_test_miss_caught_by_pbt": False,
            },
        ]
    )
    monkeypatch.setattr(
        module,
        "evaluate_candidate",
        lambda case, code, pbt_max_examples: next(evaluations),
    )
    monkeypatch.setattr(
        module,
        "build_pbt_repair_prompt",
        lambda case, previous_body, evaluation, repair_idx: f"repair prompt {repair_idx + 1}",
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

    assert len(generated_prompts) == 3
    assert result["baseline"]["passed"] is False
    assert result["verify_only"]["detected"] is True
    assert result["verify_repair"]["passed"] is True
    assert result["verify_repair"]["repaired"] is True
    assert result["verify_repair"]["n_repairs"] == 2
    assert len(result["history"]) == 3
    assert result["history"][1]["repair_prompt"] == "repair prompt 1"
    assert result["history"][2]["harness"]["passed"] is True


def test_run_case_skips_repairs_for_passing_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-012: official HumanEval passes do not enter the repair loop."""
    module = load_module()
    case = make_case("humaneval-4", dataset_idx=4)

    monkeypatch.setattr(
        module,
        "_generate_text",
        lambda **kwargs: "return x + 1",
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
            "harness": {
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
                "probe_inputs": [{"x": 1}],
            },
            "pbt": {"n_failures": 1, "violations": ["identity bug"], "repair_feedback": "fb"},
            "detected": True,
            "accepted": False,
            "official_test_miss_caught_by_pbt": True,
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

    assert result["baseline"]["passed"] is True
    assert result["verify_only"]["official_test_miss_caught_by_pbt"] is True
    assert result["verify_repair"]["passed"] is True
    assert result["verify_repair"]["n_repairs"] == 0
    assert len(result["history"]) == 1


def test_load_checkpoint_edges_and_run_benchmark_checkpoints_every_tenth_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-013, SCENARIO-CODE-011: resume stays ordered and checkpoints every 10."""
    module = load_module()
    checkpoint_path = tmp_path / "exp226.json"
    expected_ids = ["case-a"]

    assert module.load_checkpoint(checkpoint_path, expected_ids) == {
        "case_ids": expected_ids,
        "results_by_case": {},
    }

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(
        json.dumps({"case_ids": ["other"], "results_by_case": {"other": {}}}),
        encoding="utf-8",
    )
    assert module.load_checkpoint(mismatch_path, expected_ids) == {
        "case_ids": expected_ids,
        "results_by_case": {},
    }

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"case_ids": expected_ids, "results_by_case": []}),
        encoding="utf-8",
    )
    assert module.load_checkpoint(invalid_path, expected_ids) == {
        "case_ids": expected_ids,
        "results_by_case": {},
    }

    resume_cases = [
        make_case("humaneval-r0", dataset_idx=20),
        make_case("humaneval-r1", dataset_idx=21),
    ]
    resume_path = tmp_path / "resume.json"
    module.save_checkpoint(
        resume_path,
        {
            "case_ids": [case["case_id"] for case in resume_cases],
            "results_by_case": {
                resume_cases[0]["case_id"]: {
                    "case_id": resume_cases[0]["case_id"],
                    "task_id": resume_cases[0]["task_id"],
                    "baseline": {"passed": True},
                    "verify_only": {
                        "detected": False,
                        "official_test_miss_caught_by_pbt": False,
                        "n_pbt_failures": 0,
                    },
                    "verify_repair": {"passed": True, "repaired": False, "n_repairs": 0},
                    "history": [],
                }
            },
        },
    )
    matched = module.load_checkpoint(
        resume_path,
        [case["case_id"] for case in resume_cases],
    )
    assert (
        matched["results_by_case"][resume_cases[0]["case_id"]]["case_id"]
        == resume_cases[0]["case_id"]
    )

    resumed: list[str] = []
    monkeypatch.setattr(
        module,
        "run_case",
        lambda case, model, tokenizer, device_str, max_repairs, pbt_max_examples, max_new_tokens: (
            resumed.append(str(case["case_id"]))
            or {
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "baseline": {"passed": False},
                "verify_only": {
                    "detected": True,
                    "official_test_miss_caught_by_pbt": False,
                    "n_pbt_failures": 1,
                },
                "verify_repair": {"passed": True, "repaired": True, "n_repairs": 1},
                "history": [],
            }
        ),
    )
    resumed_ordered = module.run_benchmark(
        resume_cases,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        checkpoint_path=resume_path,
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
    )
    assert resumed == [resume_cases[1]["case_id"]]
    assert [item["case_id"] for item in resumed_ordered] == [
        case["case_id"] for case in resume_cases
    ]

    cases = [make_case(f"humaneval-{idx}", dataset_idx=idx) for idx in range(12)]
    saved_counts: list[int] = []
    real_save_checkpoint = module.save_checkpoint

    def tracking_save(path: Path, payload: dict[str, object]) -> None:
        saved_counts.append(len(payload["results_by_case"]))
        real_save_checkpoint(path, payload)

    monkeypatch.setattr(module, "save_checkpoint", tracking_save)
    monkeypatch.setattr(
        module,
        "run_case",
        lambda case, model, tokenizer, device_str, max_repairs, pbt_max_examples, max_new_tokens: {
            "case_id": case["case_id"],
            "task_id": case["task_id"],
            "baseline": {"passed": False},
            "verify_only": {
                "detected": True,
                "official_test_miss_caught_by_pbt": False,
                "n_pbt_failures": 1,
            },
            "verify_repair": {"passed": True, "repaired": True, "n_repairs": 1},
            "history": [],
        },
    )

    ordered = module.run_benchmark(
        cases,
        model=object(),
        tokenizer=object(),
        device_str="cuda:0",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        max_new_tokens=32,
    )

    assert [item["case_id"] for item in ordered] == [case["case_id"] for case in cases]
    assert saved_counts == [10, 12]
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["case_ids"] == [case["case_id"] for case in cases]
    assert len(checkpoint_payload["results_by_case"]) == 12


def test_summarize_results_reports_delta_ci_published_comparison_and_report_text() -> None:
    """REQ-CODE-014, SCENARIO-CODE-012: the artifact summary is publishable."""
    module = load_module()
    cases = [
        {
            "case_id": "humaneval-0",
            "baseline": {"passed": False},
            "verify_only": {
                "detected": True,
                "official_test_miss_caught_by_pbt": False,
                "n_pbt_failures": 2,
            },
            "verify_repair": {"passed": True, "repaired": True, "n_repairs": 1},
        },
        {
            "case_id": "humaneval-1",
            "baseline": {"passed": True},
            "verify_only": {
                "detected": True,
                "official_test_miss_caught_by_pbt": True,
                "n_pbt_failures": 1,
            },
            "verify_repair": {"passed": True, "repaired": False, "n_repairs": 0},
        },
        {
            "case_id": "humaneval-2",
            "baseline": {"passed": False},
            "verify_only": {
                "detected": False,
                "official_test_miss_caught_by_pbt": False,
                "n_pbt_failures": 0,
            },
            "verify_repair": {"passed": False, "repaired": False, "n_repairs": 2},
        },
    ]
    published = [
        {
            "label": "Gemma4-E4B-it model card",
            "metric": "HumanEval pass@1",
            "value": 0.40,
            "source_title": "Gemma4-E4B-it on Hugging Face",
            "source_url": "https://example.invalid/gemma4",
        }
    ]

    statistics = module.summarize_results(
        cases,
        n_bootstrap=128,
        seed=226,
        published_baselines=published,
    )

    assert statistics["baseline"]["n_correct"] == 1
    assert statistics["verify_repair"]["n_correct"] == 2
    assert statistics["verify_only"]["n_wrong_answers"] == 2
    assert statistics["verify_only"]["n_wrong_detected"] == 1
    assert statistics["verify_only"]["false_positives"] == 1
    assert statistics["verify_only"]["official_test_misses_caught_by_pbt"] == 1
    assert statistics["verify_only"]["total_pbt_failures"] == 3
    assert statistics["improvement"]["delta"] == pytest.approx(1 / 3)
    comparison = statistics["published_comparison"][0]
    assert comparison["label"] == "Gemma4-E4B-it model card"
    assert comparison["baseline_delta"] == pytest.approx((1 / 3) - 0.40)
    assert comparison["verify_repair_delta"] == pytest.approx((2 / 3) - 0.40)
    assert "164-problem" in statistics["technical_report_summary"]["paragraph"]
    assert "Gemma4-E4B-it model card" in statistics["technical_report_summary"]["paragraph"]


def test_runtime_helper_branches_cover_seeding_gpu_selection_and_live_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-013: runtime helpers stay deterministic and bounded in fallback paths."""
    module = load_module()

    numpy_calls: list[int] = []
    torch_calls: list[tuple[str, int]] = []
    empty_cache_calls: list[str] = []

    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: torch_calls.append(("manual_seed", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: torch_calls.append(("manual_seed_all", seed)),
            device_count=lambda: 2,
            mem_get_info=lambda index: (100 if index == 0 else 250, 500),
            empty_cache=lambda: empty_cache_calls.append("empty"),
        ),
    )
    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: numpy_calls.append(value))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(
        sys.modules,
        "carnot.inference.model_loader",
        SimpleNamespace(
            load_model=lambda model_name, device: (f"model:{device}", "tok"),
            generate=lambda model, tokenizer, prompt, max_new_tokens: f"{prompt}|{max_new_tokens}",
        ),
    )

    module._seed_runtime(7)
    assert numpy_calls == [7]
    assert torch_calls == [("manual_seed", 7), ("manual_seed_all", 7)]
    assert module._best_cuda_device() == "cuda:1"
    model, tokenizer, device = module._load_live_model()
    assert (model, tokenizer, device) == ("model:cuda:1", "tok", "cuda:1")
    assert (
        module._generate_text(
            model=object(),
            tokenizer=object(),
            prompt="prompt",
            prompt_seed=3,
            max_new_tokens=12,
        )
        == "prompt|12"
    )
    module._unload_live_model(object(), object())
    assert empty_cache_calls == ["empty"]
    assert os.environ["CARNOT_FORCE_LIVE"] == "1"
    assert os.environ["CARNOT_FORCE_CPU"] == "0"

    broken_torch = SimpleNamespace(
        manual_seed=lambda seed: (_ for _ in ()).throw(RuntimeError("seed fail")),
        cuda=SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("cuda fail")),
            manual_seed_all=lambda seed: None,
        ),
    )
    broken_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: (_ for _ in ()).throw(RuntimeError("np fail")))
    )
    monkeypatch.setitem(sys.modules, "torch", broken_torch)
    monkeypatch.setitem(sys.modules, "numpy", broken_numpy)
    module._seed_runtime(8)
    module._unload_live_model(object(), object())

    no_cuda_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", no_cuda_torch)
    with pytest.raises(RuntimeError):
        module._best_cuda_device()

    flaky_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            mem_get_info=lambda index: (
                (_ for _ in ()).throw(RuntimeError("gpu probe failed")) if index == 0 else (10, 20)
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", flaky_torch)
    assert module._best_cuda_device() == "cuda:1"

    monkeypatch.setitem(
        sys.modules,
        "carnot.inference.model_loader",
        SimpleNamespace(load_model=lambda model_name, device: (None, None)),
    )
    monkeypatch.setattr(module, "_best_cuda_device", lambda: "cuda:0")
    with pytest.raises(RuntimeError):
        module._load_live_model()


def test_summary_helper_handles_missing_publications_and_empty_inputs() -> None:
    """REQ-CODE-014: report text stays explicit when no published comparison is attached."""
    module = load_module()

    report = module._technical_report_summary(
        n_cases=3,
        baseline={"pass_at_1": 1 / 3, "ci_lower": 0.0, "ci_upper": 1.0},
        verify_repair={"pass_at_1": 2 / 3, "ci_lower": 0.0, "ci_upper": 1.0},
        improvement={"delta": 1 / 3, "ci_lower": 0.0, "ci_upper": 1.0},
        repair_stats={
            "n_repaired": 1,
            "n_problems_needing_repair": 2,
            "repair_success_rate": 0.5,
        },
        published_comparison=[],
    )

    assert "No published Gemma4-E4B-it HumanEval baseline" in report["paragraph"]
    with pytest.raises(ValueError):
        module.summarize_results([], n_bootstrap=8, seed=1, published_baselines=[])


def test_build_results_payload_and_main_write_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-014: the CLI writes a stable live artifact with report-ready summary text."""
    module = load_module()
    output_path = tmp_path / "experiment_226_results.json"
    checkpoint_path = tmp_path / "experiment_226_ckpt.json"
    cases = [make_case("humaneval-0", dataset_idx=0)]
    case_results = [
        {
            "case_id": "humaneval-0",
            "task_id": "HumanEval/0",
            "baseline": {"passed": False},
            "verify_only": {
                "detected": True,
                "official_test_miss_caught_by_pbt": False,
                "n_pbt_failures": 1,
            },
            "verify_repair": {"passed": True, "repaired": True, "n_repairs": 1},
            "history": [],
        }
    ]
    statistics = module.summarize_results(
        case_results,
        n_bootstrap=64,
        seed=226,
        published_baselines=[
            {
                "label": "Gemma baseline",
                "metric": "HumanEval pass@1",
                "value": 0.25,
                "source_title": "Gemma source",
                "source_url": "https://example.invalid/source",
            }
        ],
    )

    payload = module.build_results_payload(
        started_at="2026-04-12T16:00:00Z",
        finished_at="2026-04-12T16:10:00Z",
        runtime_seconds=600.0,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        device_str="cuda:1",
        sample_seed=226,
        checkpoint_interval=10,
        max_repairs=3,
        pbt_max_examples=64,
        n_bootstrap=64,
        cohort=cases,
        case_results=case_results,
        statistics=statistics,
    )
    assert payload["experiment"] == 226
    assert payload["metadata"]["checkpoint_interval"] == 10
    assert payload["statistics"]["technical_report_summary"]["paragraph"]

    monkeypatch.setattr(module, "utc_now", lambda: "2026-04-12T16:00:00Z")
    monkeypatch.setattr(
        module,
        "load_humaneval_cases",
        lambda sample_seed, sample_size=None: cases,
    )
    monkeypatch.setattr(module, "_load_live_model", lambda: (object(), object(), "cuda:1"))
    monkeypatch.setattr(module, "_unload_live_model", lambda model, tokenizer: None)
    monkeypatch.setattr(module, "run_benchmark", lambda *args, **kwargs: case_results)
    monkeypatch.setattr(
        module,
        "PUBLISHED_BASELINES",
        [
            {
                "label": "Gemma baseline",
                "metric": "HumanEval pass@1",
                "value": 0.25,
                "source_title": "Gemma source",
                "source_url": "https://example.invalid/source",
            }
        ],
    )

    exit_code = module.main(
        [
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--bootstrap-samples",
            "64",
        ]
    )

    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["metadata"]["output_path"] == str(output_path)
    assert written["metadata"]["device"] == "cuda:1"
    assert written["statistics"]["verify_repair"]["n_correct"] == 1


def test_main_guard_executes_via_runpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CODE-012: the script entrypoint remains executable as `__main__`."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_226_pbt_humaneval_full.py"
    output_path = tmp_path / "guard_results.json"
    checkpoint_path = tmp_path / "guard_checkpoint.json"

    import carnot.inference.model_loader as model_loader
    import carnot.pipeline.humaneval_live_benchmark as humaneval_module
    import carnot.pipeline.pbt_code_verifier as pbt_module

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda name, *, split: [
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n    pass\n",
                    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                    "entry_point": "add",
                }
            ]
        ),
    )
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
        model_loader,
        "load_model",
        lambda model_name, device="cpu": ("model", "tok"),
    )
    monkeypatch.setattr(
        model_loader,
        "generate",
        lambda model, tokenizer, prompt, max_new_tokens=256: "return a + b",
    )
    monkeypatch.setattr(
        humaneval_module,
        "build_candidate_code",
        lambda prompt, body: f"{prompt}    {body}\n",
    )
    monkeypatch.setattr(
        humaneval_module,
        "execute_humaneval",
        lambda code, problem, timeout=5.0: humaneval_module.HarnessResult(
            passed=True,
            error_type="none",
            error_message="",
            stdout="",
        ),
    )
    monkeypatch.setattr(
        humaneval_module,
        "run_instrumentation",
        lambda code, prompt, entry_point, official_tests=None: {
            "detected": False,
            "constraint_feedback": [],
            "dynamic_violations": [],
            "n_static_violations": 0,
            "n_dynamic_violations": 0,
            "probe_inputs": [{}],
        },
    )

    class FakeVerifierResult:
        verified = True
        derived_properties: list[object] = []
        failures: list[object] = []
        wall_clock_seconds = 0.0
        max_examples = 64

        def to_constraint_results(self):
            return []

        def repair_feedback(self) -> str:
            return ""

    class FakeVerifier:
        def __init__(self, max_examples: int) -> None:
            self.max_examples = max_examples

        def verify(self, code: str, prompt: str, entry_point: str, official_tests: str):
            return FakeVerifierResult()

    monkeypatch.setattr(pbt_module, "PBTCodeVerifier", FakeVerifier)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module_path),
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--bootstrap-samples",
            "8",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")

    assert exit_info.value.code == 0
    assert output_path.exists()
