"""Spec: REQ-VERIFY-048, REQ-VERIFY-049, SCENARIO-VERIFY-050, SCENARIO-VERIFY-051."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_235_gsm8k_semantic_v2.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_235_gsm8k_semantic_v2",
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


def write_exp219_reference(repo: Path) -> dict[str, object]:
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": 219,
        "benchmark": "gsm8k_semantic",
        "run_date": "20260412",
        "metadata": {
            "sample_seed": 218,
            "sample_size": 2,
            "source_artifacts": ["results/monitorability_policy_213.json"],
        },
        "cohort": {
            "case_count": 2,
            "case_ids": ["gsm8k-1", "gsm8k-2"],
            "cases": [
                {
                    "case_id": "gsm8k-1",
                    "question": "Q1",
                    "ground_truth": 4,
                    "task_slice": "live_gsm8k_semantic_failure",
                    "prompt_seeds": {
                        "baseline": 11,
                        "verify_only": 11,
                        "verify_repair": 11,
                    },
                },
                {
                    "case_id": "gsm8k-2",
                    "question": "Q2",
                    "ground_truth": 9,
                    "task_slice": "live_gsm8k_semantic_failure",
                    "prompt_seeds": {
                        "baseline": 12,
                        "verify_only": 12,
                        "verify_repair": 12,
                    },
                },
            ],
        },
        "statistics": {
            "Qwen3.5-0.8B": {
                "baseline": {"accuracy": 0.2},
                "verify_only": {
                    "accuracy": 0.18,
                    "n_wrong_detected": 35,
                    "false_positives": 7,
                    "wrong_detection_rate": 0.22293,
                    "false_positive_rate": 0.162791,
                    "confidence_summary": {
                        "mean_error_probability": 0.61,
                        "mean_monitorability_confidence": 0.73,
                    },
                },
                "verify_repair": {"accuracy": 0.215, "repair_yield": 0.0},
                "paired_deltas": {
                    "verify_only_minus_baseline": -0.035,
                    "repair_minus_baseline": 0.0,
                },
            },
            "Gemma4-E4B-it": {
                "baseline": {"accuracy": 0.375},
                "verify_only": {
                    "accuracy": 0.26,
                    "n_wrong_detected": 29,
                    "false_positives": 23,
                    "wrong_detection_rate": 0.232,
                    "false_positive_rate": 0.306667,
                    "confidence_summary": {
                        "mean_error_probability": 0.72,
                        "mean_monitorability_confidence": 0.8,
                    },
                },
                "verify_repair": {"accuracy": 0.38, "repair_yield": 0.072},
                "paired_deltas": {
                    "verify_only_minus_baseline": -0.115,
                    "repair_minus_baseline": 0.005,
                },
            },
        },
    }
    (results_dir / "experiment_219_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "output_policy_233.json").write_text(
        json.dumps({"per_task_slice": {}}) + "\n",
        encoding="utf-8",
    )
    return payload


def test_load_shared_cohort_reuses_exp219_case_order_and_prompt_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-050: Exp 235 reuses the checked-in Exp 219 cohort exactly."""
    module = load_module()
    repo = make_repo(tmp_path)
    reference = write_exp219_reference(repo)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    cohort, cohort_meta = module.load_shared_cohort()

    assert cohort == reference["cohort"]["cases"]
    assert cohort_meta == {
        "source_artifact": "results/experiment_219_results.json",
        "sample_seed": 218,
        "sample_size": 2,
        "case_count": 2,
        "same_as_exp219": True,
    }


def test_exp219_loader_and_path_helpers_cover_error_and_fallback_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-048: Loader helpers fail honestly on malformed reference artifacts."""
    module = load_module()
    repo = make_repo(tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    wrong_benchmark = repo / "results" / "wrong.json"
    wrong_benchmark.parent.mkdir(parents=True, exist_ok=True)
    wrong_benchmark.write_text(json.dumps({"benchmark": "constraint_ir"}), encoding="utf-8")
    with pytest.raises(ValueError, match="gsm8k_semantic"):
        module.load_exp219_reference(wrong_benchmark)

    missing_cohort = repo / "results" / "missing_cohort.json"
    missing_cohort.write_text(
        json.dumps({"benchmark": "gsm8k_semantic", "cohort": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing a cohort block"):
        module.load_shared_cohort(missing_cohort)

    wrong_cases = repo / "results" / "wrong_cases.json"
    wrong_cases.write_text(
        json.dumps({"benchmark": "gsm8k_semantic", "cohort": {"cases": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cohort.cases must be a list"):
        module.load_shared_cohort(wrong_cases)

    metadata_not_dict = repo / "results" / "metadata_not_dict.json"
    metadata_not_dict.write_text(
        json.dumps(
            {
                "benchmark": "gsm8k_semantic",
                "metadata": [],
                "cohort": {"case_count": 0, "cases": []},
            }
        ),
        encoding="utf-8",
    )
    _, cohort_meta = module.load_shared_cohort(metadata_not_dict)
    assert cohort_meta["sample_seed"] == 218
    assert cohort_meta["sample_size"] == 0
    assert module.utc_now().endswith("Z")
    assert module._round_mean([]) == 0.0
    assert module._display_path(tmp_path.parent / "outside.json").endswith("outside.json")


def test_summarize_gsm8k_v2_runs_reports_false_positives_and_confidence_summary() -> None:
    """REQ-VERIFY-048: Exp 235 summaries include calibrated semantic-v2 metrics."""
    module = load_module()
    baseline_runs = [
        {
            "correct": True,
            "latency_seconds": 1.0,
            "prompt_tokens": 10,
            "response_tokens": 3,
            "total_tokens": 13,
        },
        {
            "correct": False,
            "latency_seconds": 1.2,
            "prompt_tokens": 10,
            "response_tokens": 4,
            "total_tokens": 14,
        },
    ]
    verify_only_runs = [
        {
            "correct": True,
            "accepted_correct": False,
            "flagged": True,
            "typed_reasoning_parse_status": "direct_json",
            "semantic_verifier_v2_verdict": "violated",
            "semantic_verifier_v2_detected_wrong_answer": False,
            "semantic_verifier_v2_false_positive": True,
            "semantic_verifier_v2_error_probability": 0.91,
            "semantic_verifier_v2_monitorability_confidence": 0.88,
            "latency_seconds": 0.2,
            "total_tokens": 0,
        },
        {
            "correct": False,
            "accepted_correct": False,
            "flagged": True,
            "typed_reasoning_parse_status": "fallback_text",
            "semantic_verifier_v2_verdict": "violated",
            "semantic_verifier_v2_detected_wrong_answer": True,
            "semantic_verifier_v2_false_positive": False,
            "semantic_verifier_v2_error_probability": 0.93,
            "semantic_verifier_v2_monitorability_confidence": 0.84,
            "latency_seconds": 0.3,
            "total_tokens": 0,
        },
    ]
    verify_repair_runs = [
        {
            "initial_correct": True,
            "correct": True,
            "repaired": False,
            "n_repairs": 0,
            "initial_semantic_verifier_v2_verdict": "violated",
            "final_semantic_verifier_v2_verdict": "supported",
            "initial_semantic_verifier_v2_error_probability": 0.91,
            "final_semantic_verifier_v2_error_probability": 0.22,
            "initial_semantic_verifier_v2_monitorability_confidence": 0.88,
            "final_semantic_verifier_v2_monitorability_confidence": 0.72,
            "unnecessary_repair": False,
            "latency_seconds": 0.0,
            "total_tokens": 0,
        },
        {
            "initial_correct": False,
            "correct": True,
            "repaired": True,
            "n_repairs": 2,
            "initial_semantic_verifier_v2_verdict": "violated",
            "final_semantic_verifier_v2_verdict": "supported",
            "initial_semantic_verifier_v2_error_probability": 0.95,
            "final_semantic_verifier_v2_error_probability": 0.18,
            "initial_semantic_verifier_v2_monitorability_confidence": 0.86,
            "final_semantic_verifier_v2_monitorability_confidence": 0.78,
            "unnecessary_repair": False,
            "latency_seconds": 0.5,
            "total_tokens": 20,
        },
    ]

    summary = module.summarize_gsm8k_v2_runs(
        baseline_runs=baseline_runs,
        verify_only_runs=verify_only_runs,
        verify_repair_runs=verify_repair_runs,
    )

    assert summary["baseline"]["accuracy"] == 0.5
    assert summary["verify_only"]["accuracy"] == 0.0
    assert summary["verify_only"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["false_positives"] == 1
    assert summary["verify_only"]["semantic_verifier_v2_false_positives"] == 1
    assert summary["verify_only"]["parse_coverage"] == 1.0
    assert summary["verify_only"]["confidence_summary"] == {
        "mean_error_probability": 0.92,
        "mean_monitorability_confidence": 0.86,
        "verdict_counts": {"abstain": 0, "supported": 0, "unavailable": 0, "violated": 2},
    }
    assert summary["verify_repair"]["accuracy"] == 1.0
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["repair_yield"] == 1.0
    assert summary["verify_repair"]["confidence_summary"] == {
        "initial_mean_error_probability": 0.93,
        "final_mean_error_probability": 0.2,
        "initial_mean_monitorability_confidence": 0.87,
        "final_mean_monitorability_confidence": 0.75,
        "initial_verdict_counts": {
            "abstain": 0,
            "supported": 0,
            "unavailable": 0,
            "violated": 2,
        },
        "final_verdict_counts": {
            "abstain": 0,
            "supported": 2,
            "unavailable": 0,
            "violated": 0,
        },
    }


def test_build_exp219_comparison_reports_budget_decisions_and_blockers() -> None:
    """SCENARIO-VERIFY-051: Exp 235 comparison states whether false positives improved enough."""
    module = load_module()
    exp219_statistics = {
        "Qwen3.5-0.8B": {
            "baseline": {"accuracy": 0.215},
            "verify_only": {
                "accuracy": 0.18,
                "n_wrong_detected": 35,
                "false_positives": 7,
                "wrong_detection_rate": 0.22293,
                "false_positive_rate": 0.162791,
                "confidence_summary": {
                    "mean_error_probability": 0.61,
                    "mean_monitorability_confidence": 0.73,
                },
            },
            "verify_repair": {"accuracy": 0.215, "repair_yield": 0.0},
            "paired_deltas": {"verify_only_minus_baseline": -0.035, "repair_minus_baseline": 0.0},
        },
        "Gemma4-E4B-it": {
            "baseline": {"accuracy": 0.375},
            "verify_only": {
                "accuracy": 0.26,
                "n_wrong_detected": 29,
                "false_positives": 23,
                "wrong_detection_rate": 0.232,
                "false_positive_rate": 0.306667,
                "confidence_summary": {
                    "mean_error_probability": 0.72,
                    "mean_monitorability_confidence": 0.8,
                },
            },
            "verify_repair": {"accuracy": 0.38, "repair_yield": 0.072},
            "paired_deltas": {"verify_only_minus_baseline": -0.115, "repair_minus_baseline": 0.005},
        },
    }
    current_statistics = {
        "Qwen3.5-0.8B": {
            "baseline": {"accuracy": 0.22},
            "verify_only": {
                "accuracy": 0.22,
                "n_wrong_detected": 33,
                "false_positives": 1,
                "wrong_detection_rate": 0.21,
                "false_positive_rate": 0.02,
                "confidence_summary": {
                    "mean_error_probability": 0.82,
                    "mean_monitorability_confidence": 0.79,
                },
            },
            "verify_repair": {"accuracy": 0.24, "repair_yield": 0.12},
            "paired_deltas": {"verify_only_minus_baseline": 0.0, "repair_minus_baseline": 0.02},
        },
        "Gemma4-E4B-it": {
            "baseline": {"accuracy": 0.38},
            "verify_only": {
                "accuracy": 0.34,
                "n_wrong_detected": 30,
                "false_positives": 8,
                "wrong_detection_rate": 0.24,
                "false_positive_rate": 0.13,
                "confidence_summary": {
                    "mean_error_probability": 0.8,
                    "mean_monitorability_confidence": 0.83,
                },
            },
            "verify_repair": {"accuracy": 0.41, "repair_yield": 0.11},
            "paired_deltas": {"verify_only_minus_baseline": -0.04, "repair_minus_baseline": 0.03},
        },
    }

    comparison = module.build_exp219_comparison(
        current_statistics=current_statistics,
        exp219_statistics=exp219_statistics,
        same_cohort=True,
        blockers=[
            {
                "model_name": "Gemma4-E4B-it",
                "stage": "verify_repair",
                "error": "checkpoint pending",
            }
        ],
    )

    assert comparison["source_artifact"] == "results/experiment_219_results.json"
    assert comparison["same_cohort_as_exp219"] is True
    assert comparison["blockers"] == [
        {"model_name": "Gemma4-E4B-it", "stage": "verify_repair", "error": "checkpoint pending"}
    ]
    assert comparison["models"]["Qwen3.5-0.8B"] == {
        "verify_only_accuracy_delta": 0.04,
        "verify_repair_accuracy_delta": 0.025,
        "wrong_detection_delta": -2,
        "false_positive_delta": -6,
        "wrong_detection_rate_delta": -0.01293,
        "false_positive_rate_delta": -0.142791,
        "repair_yield_delta": 0.12,
        "mean_error_probability_delta": 0.21,
        "mean_monitorability_confidence_delta": 0.06,
        "verify_only_justified": True,
    }
    assert comparison["models"]["Gemma4-E4B-it"]["verify_only_justified"] is False
    assert comparison["overall"] == {
        "verify_only_models_justified": ["Qwen3.5-0.8B"],
        "verify_only_models_not_justified": ["Gemma4-E4B-it"],
    }

    missing_reference = module.build_exp219_comparison(
        current_statistics={"Qwen3.5-0.8B": current_statistics["Qwen3.5-0.8B"]},
        exp219_statistics={},
        same_cohort=False,
        blockers=[],
    )
    assert missing_reference["models"]["Qwen3.5-0.8B"] == {"status": "missing_exp219_reference"}
    assert missing_reference["overall"]["verify_only_models_not_justified"] == ["Qwen3.5-0.8B"]


def test_build_artifact_payload_preserves_v1_schema_and_adds_comparison_block() -> None:
    """REQ-VERIFY-048, REQ-VERIFY-049: Exp 235 keeps v1 shape and adds comparison data."""
    module = load_module()
    cohort = [
        {
            "case_id": "gsm8k-1",
            "question": "Q1",
            "ground_truth": 4,
            "task_slice": "live_gsm8k_semantic_failure",
            "prompt_seeds": {"baseline": 11, "verify_only": 11, "verify_repair": 11},
        }
    ]
    paired_runs = [
        {
            "benchmark": "gsm8k_semantic",
            "mode": "baseline",
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "summary": {"n_cases": 1},
            "cases": [{"case_id": "gsm8k-1"}],
        }
    ]
    statistics = {"Qwen3.5-0.8B": {"baseline": {"n_cases": 1}}}
    comparison = {"same_cohort_as_exp219": True}

    payload = module.build_artifact_payload(
        output_path=Path("results/experiment_235_results.json"),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics=statistics,
        comparison_to_exp219=comparison,
        blockers=[],
        started_at="2026-04-13T10:00:00Z",
        finished_at="2026-04-13T10:05:00Z",
        runtime_seconds=300.0,
        checkpoint_dir=Path("results/checkpoints/experiment_235"),
        max_repairs=3,
        policy_path=Path("results/output_policy_233.json"),
        cohort_meta={
            "source_artifact": "results/experiment_219_results.json",
            "sample_seed": 218,
            "sample_size": 1,
            "case_count": 1,
            "same_as_exp219": True,
        },
        run_status="complete",
    )

    assert payload["experiment"] == 235
    assert payload["benchmark"] == "gsm8k_semantic"
    assert payload["run_date"] == "20260413"
    assert payload["schema"] == {
        "artifact": "carnot.live_dual_model_suite.v1",
        "benchmark_case_schema": "gsm8k_semantic.v1",
    }
    assert payload["metadata"]["sample_seed"] == 218
    assert payload["metadata"]["sample_size"] == 1
    assert payload["metadata"]["source_artifacts"] == [
        "results/experiment_219_results.json",
        "results/output_policy_233.json",
        "python/carnot/pipeline/semantic_verifier_v2.py",
    ]
    assert payload["metadata"]["checkpoint_dir"] == "results/checkpoints/experiment_235"
    assert payload["metadata"]["checkpoint_pattern"] == (
        "results/checkpoints/experiment_235/<benchmark>__<model>__<mode>.json"
    )
    assert payload["metadata"]["semantic_verifier"] == {
        "name": "semantic_verifier_v2",
        "run_date": "20260413",
        "policy_source": "results/output_policy_233.json",
    }
    assert payload["cohort"]["case_ids"] == ["gsm8k-1"]
    assert payload["cohort"]["shared_with_experiment_219"] is True
    assert payload["paired_runs"] == paired_runs
    assert payload["statistics"] == statistics
    assert payload["comparison_to_experiment_219"] == comparison
    assert payload["blockers"] == []
    assert payload["run_status"] == "complete"


def test_main_writes_payload_and_uses_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-048: The Exp 235 CLI writes the artifact in-place with repo defaults."""
    module = load_module()
    repo = make_repo(tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    written_payload = {"experiment": 235, "run_status": "complete"}

    def fake_run_live_benchmark(args) -> dict[str, object]:
        assert args.output == repo / "results" / "experiment_235_results.json"
        assert args.checkpoint_dir == repo / "results" / "checkpoints" / "experiment_235"
        assert args.max_repairs == module.DEFAULT_MAX_REPAIRS
        return written_payload

    monkeypatch.setattr(module, "_run_live_benchmark", fake_run_live_benchmark)

    exit_code = module.main([])

    assert exit_code == 0
    output_path = repo / "results" / "experiment_235_results.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == written_payload
