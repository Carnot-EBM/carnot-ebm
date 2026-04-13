"""Spec: REQ-SAMPLE-008, SCENARIO-SAMPLE-015, SCENARIO-SAMPLE-016, SCENARIO-SAMPLE-017."""

from __future__ import annotations

import importlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest


def load_pipeline_module():
    return importlib.import_module("carnot.inference.repair_reranker")


def load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_243_energy_reranker.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_243_energy_reranker",
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


def exp235_fixture() -> dict[str, object]:
    return {
        "experiment": 235,
        "benchmark": "gsm8k_semantic",
        "run_date": "20260413",
        "cohort": {
            "case_count": 2,
            "case_ids": ["gsm8k-1", "gsm8k-2"],
            "cases": [
                {
                    "case_id": "gsm8k-1",
                    "sample_position": 1,
                    "question": "How many apples are left?",
                    "ground_truth": 11,
                    "task_slice": "live_gsm8k_semantic_failure",
                    "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1},
                },
                {
                    "case_id": "gsm8k-2",
                    "sample_position": 2,
                    "question": "How many pears are left?",
                    "ground_truth": 7,
                    "task_slice": "live_gsm8k_semantic_failure",
                    "prompt_seeds": {"baseline": 2, "verify_only": 2, "verify_repair": 2},
                },
            ],
        },
        "paired_runs": [
            {
                "model_name": "Qwen3.5-0.8B",
                "mode": "verify_repair",
                "cases": [
                    {
                        "case_id": "gsm8k-1",
                        "correct": False,
                        "verified": True,
                        "repaired": False,
                        "n_repairs": 2,
                        "latency_seconds": 1.25,
                        "history": [
                            {
                                "iteration": 0,
                                "response": '{"final_answer": 4}',
                                "verification": {
                                    "verified": False,
                                    "n_violations": 2,
                                    "semantic_verifier_v2": {
                                        "verdict": "violated",
                                        "semantic_error_probability": 0.82,
                                        "monitorability_confidence": 0.94,
                                    },
                                },
                            },
                            {
                                "iteration": 1,
                                "response": '{"final_answer": 11}',
                                "verification": {
                                    "verified": True,
                                    "n_violations": 0,
                                    "semantic_verifier_v2": {
                                        "verdict": "supported",
                                        "semantic_error_probability": 0.04,
                                        "monitorability_confidence": 0.88,
                                    },
                                },
                            },
                            {
                                "iteration": 2,
                                "response": '{"final_answer": 9}',
                                "verification": {
                                    "verified": True,
                                    "n_violations": 0,
                                    "semantic_verifier_v2": {
                                        "verdict": "abstain",
                                        "semantic_error_probability": 0.46,
                                        "monitorability_confidence": 0.37,
                                    },
                                },
                            },
                        ],
                    },
                    {
                        "case_id": "gsm8k-2",
                        "correct": False,
                        "verified": False,
                        "repaired": False,
                        "n_repairs": 0,
                        "latency_seconds": 0.61,
                        "history": [
                            {
                                "iteration": 0,
                                "response": "Answer: 5",
                                "verification": {
                                    "verified": False,
                                    "n_violations": 1,
                                    "semantic_verifier_v2": {
                                        "verdict": "violated",
                                        "semantic_error_probability": 0.64,
                                        "monitorability_confidence": 0.71,
                                    },
                                },
                            }
                        ],
                    },
                ],
            }
        ],
    }


def exp238_fixture() -> dict[str, object]:
    return {
        "experiment": 238,
        "benchmark": "humaneval_dual_model_spec",
        "run_date": "20260413",
        "cohort": {
            "case_count": 1,
            "case_ids": ["humaneval-1"],
            "cases": [
                {
                    "case_id": "humaneval-1",
                    "sample_position": 1,
                    "task_id": "HumanEval/1",
                    "prompt_seeds": {"baseline": 3, "verify_only": 3, "verify_repair": 3},
                }
            ],
        },
        "model_runs": {
            "Gemma4-E4B-it": {
                "model_name": "Gemma4-E4B-it",
                "per_problem_results": [
                    {
                        "case_id": "humaneval-1",
                        "dataset_idx": 1,
                        "task_id": "HumanEval/1",
                        "entry_point": "solve",
                        "baseline": {"official_passed": False},
                        "verify_repair": {
                            "accepted": False,
                            "official_passed": False,
                            "repaired": False,
                            "n_repairs": 2,
                            "final_code": "def solve(x):\n    return x - 1\n",
                        },
                        "history": [
                            {
                                "iteration": 0,
                                "candidate_code": "def solve(x):\n    return x - 1\n",
                                "evaluation": {
                                    "official_tests": {"passed": False},
                                    "pbt": {"verified": False, "n_failures": 2},
                                    "explicit_specs": {"n_violations": 1},
                                    "stage_acceptance": {"spec_aware_verify_only": False},
                                    "latency_seconds": 0.2,
                                },
                            },
                            {
                                "iteration": 1,
                                "candidate_code": "def solve(x):\n    return x + 1\n",
                                "evaluation": {
                                    "official_tests": {"passed": True},
                                    "pbt": {"verified": True, "n_failures": 0},
                                    "explicit_specs": {"n_violations": 0},
                                    "stage_acceptance": {"spec_aware_verify_only": True},
                                    "latency_seconds": 0.11,
                                },
                            },
                            {
                                "iteration": 2,
                                "candidate_code": "def solve(x):\n    return x\n",
                                "evaluation": {
                                    "official_tests": {"passed": True},
                                    "pbt": {"verified": False, "n_failures": 1},
                                    "explicit_specs": {"n_violations": 1},
                                    "stage_acceptance": {"spec_aware_verify_only": False},
                                    "latency_seconds": 0.09,
                                },
                            },
                        ],
                    }
                ],
            }
        },
    }


def write_fixture_repo(repo: Path) -> None:
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_235_results.json").write_text(
        json.dumps(exp235_fixture(), indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "experiment_238_results.json").write_text(
        json.dumps(exp238_fixture(), indent=2) + "\n",
        encoding="utf-8",
    )


class FakeBackend:
    def __init__(self, samples: list[list[int]], name: str = "fake_sampler") -> None:
        self.backend_name = name
        self.samples = np.asarray(samples, dtype=bool)
        self.calls: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        self.calls.append((biases, couplings, n_samples, n_steps, beta))
        return self.samples


def test_build_candidate_set_benchmark_preserves_saved_candidate_order():
    """SCENARIO-SAMPLE-015: Exp 243 replays deterministic semantic and code candidate sets."""
    module = load_pipeline_module()

    cases = module.build_candidate_set_benchmark(exp235_fixture(), exp238_fixture())

    assert [case.case_id for case in cases] == ["gsm8k-1", "gsm8k-2", "humaneval-1"]
    semantic_case = cases[0]
    assert semantic_case.original_selected_candidate_id.endswith(":2")
    assert [candidate.iteration for candidate in semantic_case.candidates] == [0, 1, 2]
    assert semantic_case.candidates[1].actual_success is True
    assert semantic_case.candidates[2].accepted is True

    code_case = cases[2]
    assert code_case.original_selected_candidate_id.endswith(":2")
    assert [candidate.iteration for candidate in code_case.candidates] == [0, 1, 2]
    assert code_case.candidates[1].actual_success is True
    assert code_case.candidates[1].accepted is True


def test_score_candidate_uses_existing_composite_scorer_integration():
    """REQ-SAMPLE-008: replay scoring routes verifier-derived features through
    CompositeEnergyScorer.
    """
    module = load_pipeline_module()
    cases = module.build_candidate_set_benchmark(exp235_fixture(), exp238_fixture())

    semantic_scores = [module.score_candidate(candidate) for candidate in cases[0].candidates]
    code_scores = [module.score_candidate(candidate) for candidate in cases[2].candidates]

    assert semantic_scores[1] < semantic_scores[2] < semantic_scores[0]
    assert code_scores[1] < code_scores[2] < code_scores[0]


def test_rerank_candidate_set_projects_sampler_samples_to_a_top1_choice():
    """SCENARIO-SAMPLE-016: sampler-backed reranking returns one deterministic top-1 candidate."""
    module = load_pipeline_module()
    semantic_case = module.build_candidate_set_benchmark(exp235_fixture(), exp238_fixture())[0]
    backend = FakeBackend([[1, 1, 0], [0, 1, 0]])

    result = module.rerank_candidate_set(semantic_case, backend=backend)

    assert result["selected_candidate_id"] == semantic_case.candidates[1].candidate_id
    assert result["actual_success"] is True
    assert result["accepted"] is True
    assert result["sampler_backend"] == "fake_sampler"
    assert backend.calls[0][0].shape == (3,)
    assert backend.calls[0][1].shape == (3, 3)


def test_summarize_backend_results_reports_quality_precision_yield_and_latency():
    """REQ-SAMPLE-008: result summaries include top-1 quality, precision,
    repair yield, and latency.
    """
    module = load_pipeline_module()
    cases = module.build_candidate_set_benchmark(exp235_fixture(), exp238_fixture())
    results = [
        module.rerank_candidate_set(cases[0], backend=FakeBackend([[0, 1, 0]])),
        module.rerank_candidate_set(cases[1], backend=FakeBackend([[1]])),
        module.rerank_candidate_set(cases[2], backend=FakeBackend([[0, 1, 0]])),
    ]

    summary = module.summarize_backend_results(
        candidate_sets=cases,
        reranked_cases=results,
        sampler_backend="cpu",
        execution_path="cpu",
        run_status="complete",
        blockers=[],
    )

    overall = summary["overall"]
    assert overall["n_cases"] == 3
    assert overall["baseline_top1_quality_rate"] == pytest.approx(1 / 3)
    assert overall["top1_quality_rate"] == pytest.approx(2 / 3)
    assert overall["top1_quality_delta"] == pytest.approx(1 / 3)
    assert overall["verifier_precision"] == pytest.approx(1.0)
    assert overall["repair_yield"] == pytest.approx(2 / 3)
    assert overall["mean_selection_latency_seconds"] >= 0.0
    assert summary["by_benchmark"]["gsm8k_semantic"]["n_cases"] == 2
    assert summary["by_benchmark"]["humaneval_dual_model_spec"]["n_cases"] == 1


def test_run_experiment_and_script_write_exp243_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """SCENARIO-SAMPLE-017: Exp 243 writes the artifact and labels blocked KV260 honestly."""
    pipeline_module = load_pipeline_module()
    script_module = load_script_module()
    repo = make_repo(tmp_path)
    write_fixture_repo(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr(
        pipeline_module,
        "resolve_optional_kv260_backend",
        lambda **_kwargs: {
            "sampler_backend": "kv260",
            "execution_path": "blocked",
            "run_status": "blocked",
            "backend": None,
            "blockers": [{"code": "missing_bitfile_config"}],
            "notes": ["KV260 remained unavailable."],
        },
    )

    payload = pipeline_module.run_experiment(repo_root=repo)
    result_path = repo / "results" / "experiment_243_results.json"
    assert result_path.exists()
    assert payload["metadata"]["output_path"] == "results/experiment_243_results.json"
    assert payload["backends"]["cpu"]["execution_path"] == "cpu"
    assert payload["backends"]["kv260"]["execution_path"] == "blocked"
    assert payload["backends"]["kv260"]["run_status"] == "blocked"
    assert payload["run_status"] == "complete"

    parser = script_module.build_parser()
    args = parser.parse_args([])
    assert args.output == "results/experiment_243_results.json"
    assert script_module.get_repo_root() == repo.resolve()

    argv = sys.argv
    try:
        sys.argv = ["experiment_243_energy_reranker.py"]
        runpy.run_path(str(Path(script_module.__file__)), run_name="__main__")
    finally:
        sys.argv = argv

    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["title"] == "Sampler-backed repair reranking replay benchmark"
    assert written["backends"]["kv260"]["execution_path"] == "blocked"


def test_internal_helpers_cover_empty_and_sparse_paths():
    """REQ-SAMPLE-008: helper branches stay deterministic on empty or malformed replay inputs."""
    module = load_pipeline_module()

    assert module.extract_final_number("no numeric answer here") is None
    assert module.build_candidate_set_benchmark({"cohort": {}}, {"model_runs": {}}) == []

    empty = module.summarize_backend_results(
        candidate_sets=[],
        reranked_cases=[],
        sampler_backend="cpu",
        execution_path="cpu",
        run_status="complete",
        blockers=[],
    )
    assert empty["overall"]["n_cases"] == 0
    assert empty["overall"]["top1_quality_rate"] == 0.0


def test_helper_branches_cover_path_fallbacks_and_sparse_candidate_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """REQ-SAMPLE-008: helper edge branches remain deterministic on sparse replay inputs."""
    module = load_pipeline_module()

    monkeypatch.delenv("CARNOT_REPO_ROOT", raising=False)
    assert module.get_repo_root().name == "carnot"

    repo = make_repo(tmp_path)
    outside = tmp_path / "outside.json"
    assert module._relative_path(outside, repo) == str(outside)
    assert module.extract_final_number("#### 1,234") == 1234
    assert module._sample_positions({"cohort": {"case_ids": ["a", "b"]}}) == {"a": 1, "b": 2}
    assert module._encode_candidate_selection_problem([])[0].size == 0

    fallback_idx, votes, fallback_used, source = module._decode_samples(
        np.asarray([[0, 0], [0, 0]], dtype=bool),
        [3.0, 1.0],
    )
    assert fallback_idx == 1
    assert votes == {}
    assert fallback_used is True
    assert source == "score_fallback"

    semantic_sparse = module._semantic_candidate_sets(
        {
            "cohort": {
                "cases": [{"case_id": "gsm8k-sparse", "sample_position": 1, "ground_truth": 1}]
            },
            "paired_runs": [
                {"mode": "baseline", "cases": []},
                {"mode": "verify_repair", "cases": [{"case_id": "gsm8k-sparse", "history": []}]},
                {"mode": "verify_repair", "cases": [{"case_id": "missing-history"}]},
            ],
        }
    )
    assert semantic_sparse == []

    assert module._code_candidate_sets({"model_runs": []}) == []
    code_sparse = module._code_candidate_sets(
        {
            "cohort": {"case_ids": ["humaneval-sparse"]},
            "model_runs": {
                "bad": {"model_name": "Bad", "per_problem_results": "bad"},
                "empty": {
                    "model_name": "Empty",
                    "per_problem_results": [{"case_id": "humaneval-sparse", "history": []}],
                },
                "missing": {
                    "model_name": "Missing",
                    "per_problem_results": [{"case_id": "humaneval-sparse"}],
                },
            },
        }
    )
    assert code_sparse == []


def test_resolve_optional_kv260_backend_covers_available_and_blocked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """SCENARIO-SAMPLE-017: KV260 backend resolution reports hardware,
    software-model, and blocked paths honestly.
    """
    module = load_pipeline_module()
    repo = make_repo(tmp_path)
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    software_model = module.resolve_optional_kv260_backend(
        repo_root=repo,
        overlay_factory=lambda _bitfile: module.SoftwareFPGAOverlay(),
    )
    assert software_model["execution_path"] == "software_model"
    assert software_model["run_status"] == "complete"

    bitfile = tmp_path / "kv260.bit"
    bitfile.write_text("bitstream", encoding="utf-8")

    class FakeFPGA:
        def __init__(self, *args, **kwargs) -> None:
            self.backend_name = "fpga"

    monkeypatch.setattr(module, "FPGAIsingSampler", FakeFPGA)
    hardware = module.resolve_optional_kv260_backend(repo_root=repo, bitfile_path=bitfile)
    assert hardware["execution_path"] == "hardware"
    assert hardware["backend"].backend_name == "fpga"

    class RaisingFPGA:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("overlay failed")

    monkeypatch.setattr(module, "FPGAIsingSampler", RaisingFPGA)
    blocked_overlay = module.resolve_optional_kv260_backend(repo_root=repo, bitfile_path=bitfile)
    assert blocked_overlay["execution_path"] == "blocked"
    assert blocked_overlay["blockers"][0]["code"] == "overlay_unavailable"

    monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))
    monkeypatch.setattr(module, "FPGAIsingSampler", FakeFPGA)
    from_env = module.resolve_optional_kv260_backend(repo_root=repo)
    assert from_env["execution_path"] == "hardware"
    monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)

    missing_bitfile = module.resolve_optional_kv260_backend(
        repo_root=repo,
        bitfile_path=tmp_path / "missing-kv260.bit",
    )
    assert missing_bitfile["blockers"][0]["code"] == "bitfile_not_found"

    monkeypatch.setattr(module, "FPGAIsingSampler", FakeFPGA)
    (results_dir / "experiment_242_results.json").write_text(
        json.dumps(
            {
                "metadata": {"execution_path": "software_model", "notes": ["soft note"]},
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    from_exp242_software = module.resolve_optional_kv260_backend(repo_root=repo)
    assert from_exp242_software["execution_path"] == "software_model"

    (results_dir / "experiment_242_results.json").write_text(
        json.dumps(
            {
                "metadata": {"execution_path": "hardware"},
                "blockers": [{"code": "needs_live_transport"}],
            }
        ),
        encoding="utf-8",
    )
    from_exp242_hardware = module.resolve_optional_kv260_backend(repo_root=repo)
    assert from_exp242_hardware["execution_path"] == "blocked"
    assert from_exp242_hardware["blockers"][0]["code"] == "needs_live_transport"

    (results_dir / "experiment_242_results.json").write_text(
        json.dumps(
            {
                "metadata": {"execution_path": "blocked", "notes": ["blocked note"]},
                "blockers": "bad-blockers",
            }
        ),
        encoding="utf-8",
    )
    from_exp242_blocked = module.resolve_optional_kv260_backend(repo_root=repo)
    assert from_exp242_blocked["execution_path"] == "blocked"
    assert from_exp242_blocked["notes"] == ["blocked note"]
    assert from_exp242_blocked["blockers"] == []

    (results_dir / "experiment_242_results.json").unlink()
    missing_reference = module.resolve_optional_kv260_backend(repo_root=repo)
    assert missing_reference["blockers"][0]["code"] == "missing_exp242_reference"
