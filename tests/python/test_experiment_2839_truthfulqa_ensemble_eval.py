"""Tests for the Exp 2839 TruthfulQA dual-condition third attempt.

Spec: REQ-VERIFY-2839-TQA,
      SCENARIO-VERIFY-2839-TQA-BLOCKED,
      SCENARIO-VERIFY-2839-TQA-LIVE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import truthfulqa_ensemble_v7b_exp2839 as mod
from carnot.eval.truthfulqa_ensemble_v7b_exp2839 import (
    ExperimentConfig,
    PreconditionCheck,
    SeedMeasurement,
    run_experiment,
)


def _state(root: Path) -> None:
    path = root / "results" / "fr11_policy_cache_events_1512.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"state": true}\n', encoding="utf-8")


def _all_checks() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("cuda", True, "cmd=.venv/bin/python3; cuda available"),
        PreconditionCheck("hf_truthfulqa_generation", True, "TruthfulQA accessible"),
        PreconditionCheck("qwen36_gguf_cache", True, "real qwen cached"),
        PreconditionCheck("fr11_state_files", True, "state present"),
        PreconditionCheck("bleurt_base_128", True, "BLEURT-base-128 usable"),
    ]


def _measurements() -> list[SeedMeasurement]:
    return [
        SeedMeasurement(
            seed=42,
            condition_a_ensemble_auroc=0.61,
            condition_b_ensemble_auroc=0.59,
            condition_a_per_verifier={"semantic": 0.62, "tier0r": 0.60},
            condition_b_per_verifier={"semantic": 0.58, "tier0r": 0.57},
            bleurt_threshold=0.41,
        ),
        SeedMeasurement(
            seed=137,
            condition_a_ensemble_auroc=0.67,
            condition_b_ensemble_auroc=0.63,
            condition_a_per_verifier={"semantic": 0.68, "tier0r": 0.65},
            condition_b_per_verifier={"semantic": 0.64, "tier0r": 0.61},
            bleurt_threshold=0.45,
        ),
    ]


def test_scenario_verify_2839_tqa_blocked_writes_new_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2839-TQA-BLOCKED: blocked runs do not infer metrics."""

    _state(tmp_path)
    measured = False

    def measurement_runner(
        _config: ExperimentConfig,
        _state_files: list[dict[str, object]],
    ) -> list[SeedMeasurement]:
        nonlocal measured
        measured = True
        return _measurements()

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            random_seeds=(42, 137),
            started_at=10.0,
            clock=lambda: 14.0,
        ),
        precondition_probe=lambda _config, _state_files: [
            PreconditionCheck("cuda", True, "cmd=.venv/bin/python3; cuda ok"),
            PreconditionCheck("hf_truthfulqa_generation", True, "dataset ok"),
            PreconditionCheck("qwen36_gguf_cache", False, "no real gguf"),
            PreconditionCheck("fr11_state_files", True, "state present"),
            PreconditionCheck("bleurt_base_128", False, "BLEURT missing"),
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    assert measured is False
    assert artifact["artifact"] == "experiment_2839_truthfulqa_ensemble_eval"
    assert artifact["schema"] == "carnot.truthfulqa_ensemble_v7b_exp2839"
    assert artifact["honest_verdict"] == "blocked_model_not_cached_qwen36_35b_a3b_gguf"
    assert artifact["blocked_resources"] == ["qwen36_gguf_cache", "bleurt_base_128"]
    assert artifact["corpus"] == "TruthfulQA-generation"
    assert artifact["n_questions"] == 200
    assert artifact["n_seeds"] == 2
    assert artifact["scoring_method"] == "BLEURT-base-128, threshold tuned on 50-Q held-out"
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["bleurt_threshold"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["duration_s"] == pytest.approx(4.0)
    assert "principle" in artifact["field_provenance"]["duration_s"]

    saved_path = tmp_path / "results" / "experiment_2839_truthfulqa_ensemble_eval.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == artifact
    assert not (tmp_path / "results" / "experiment_2831_truthfulqa_ensemble_eval.json").exists()


def test_scenario_verify_2839_tqa_live_success_and_backend_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2839-TQA-LIVE: measured seed rows summarize both conditions."""

    _state(tmp_path)
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
        random_seeds=(42, 137),
        started_at=1.0,
        clock=lambda: 11.0,
        published_bleurt_verifier_comparators=(
            {"label": "published BLEURT verifier comparator", "auroc": 0.60},
        ),
    )

    artifact = run_experiment(
        config,
        precondition_probe=lambda _config, _state_files: _all_checks(),
        measurement_runner=lambda _config, _state_files: _measurements(),
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.64)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.61)
    assert artifact["learning_contribution"] == pytest.approx(0.03)
    assert artifact["bleurt_threshold"] == pytest.approx(0.43)
    assert artifact["per_verifier_condition_a_auroc"] == {
        "semantic": [0.62, 0.68],
        "tier0r": [0.60, 0.65],
    }
    assert artifact["baseline_comparison"]["production_minus_bleurt_comparator_best"] == (
        pytest.approx(0.04)
    )
    assert artifact["artifact"] == "experiment_2839_truthfulqa_ensemble_eval"

    backend_block = run_experiment(
        config,
        precondition_probe=lambda _config, _state_files: _all_checks(),
        write=False,
    )
    assert backend_block["honest_verdict"] == "blocked_live_qwen36_backend_unavailable"
    assert backend_block["blocked_resources"] == ["live_backend"]


def test_req_verify_2839_tqa_probe_preconditions_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-2839-TQA: resource checks include the venv CUDA precondition."""

    _state(tmp_path)

    checks = mod.probe_preconditions(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        mod.discover_fr11_state_files(tmp_path),
        cuda_check=lambda _config: PreconditionCheck(
            "cuda", True, "cmd=.venv/bin/python3; torch cuda ok"
        ),
        hf_truthfulqa_check=lambda: PreconditionCheck(
            "hf_truthfulqa_generation", True, "TruthfulQA rows loaded"
        ),
        qwen_cache_check=lambda _root: PreconditionCheck(
            "qwen36_gguf_cache", True, "real GGUF cached"
        ),
        bleurt_check=lambda _root: PreconditionCheck(
            "bleurt_base_128", True, "BLEURT package importable"
        ),
    )

    assert [check.resource for check in checks] == [
        "cuda",
        "hf_truthfulqa_generation",
        "qwen36_gguf_cache",
        "fr11_state_files",
        "bleurt_base_128",
    ]
    assert checks[0].detail.startswith("cmd=.venv/bin/python3")
    assert checks[3] == PreconditionCheck("fr11_state_files", True, "1 FR-11 state files discovered")


def test_req_verify_2839_tqa_venv_cuda_probe_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2839-TQA: .venv CUDA probe and CLI paths stay deterministic."""

    missing = mod.venv_cuda_check(ExperimentConfig(repo_root=tmp_path))
    assert missing == PreconditionCheck(
        "cuda", False, f"{tmp_path / '.venv' / 'bin' / 'python3'} missing"
    )

    python_path = tmp_path / ".venv" / "bin" / "python3"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    def ok_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[:2] == [str(python_path), "-c"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"available": True, "detail": "cuda_available=True"}),
            "",
        )

    assert mod.venv_cuda_check(
        ExperimentConfig(repo_root=tmp_path),
        command_runner=ok_runner,
    ) == PreconditionCheck("cuda", True, "cmd=.venv/bin/python3; cuda_available=True")

    def error_runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, "stdout", "stderr")

    failed = mod.venv_cuda_check(ExperimentConfig(repo_root=tmp_path), command_runner=error_runner)
    assert failed == PreconditionCheck("cuda", False, "stderr")

    def invalid_runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    invalid = mod.venv_cuda_check(
        ExperimentConfig(repo_root=tmp_path),
        command_runner=invalid_runner,
    )
    assert invalid.available is False
    assert "invalid JSON" in invalid.detail

    def raising_runner(_command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("cannot execute")

    raised = mod.venv_cuda_check(
        ExperimentConfig(repo_root=tmp_path),
        command_runner=raising_runner,
    )
    assert raised.available is False
    assert "cannot execute" in raised.detail

    calls: list[ExperimentConfig] = []

    def fake_run_experiment(config: ExperimentConfig) -> dict[str, object]:
        calls.append(config)
        return {"honest_verdict": "blocked_unit_test"}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                str(tmp_path / "custom-results"),
                "--n-questions",
                "17",
            ]
        )
        == 0
    )
    assert calls[0].repo_root == tmp_path
    assert calls[0].results_dir == tmp_path / "custom-results"
    assert calls[0].n_questions == 17
