import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import carnot.eval.truthfulqa_ensemble_v7b as mod
from carnot.eval.truthfulqa_ensemble_v7b import (
    ExperimentConfig,
    PreconditionCheck,
    SeedMeasurement,
    build_reproducibility_checksum,
    compute_auroc,
    discover_fr11_state_files,
    run_experiment,
    select_truthfulqa_indices,
    summarize_measurements,
)


def _all_preconditions() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("cuda", True, "cuda available"),
        PreconditionCheck("hf_truthfulqa_generation", True, "TruthfulQA accessible"),
        PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
        PreconditionCheck("fr11_state_files", True, "state present"),
        PreconditionCheck("bleurt_base_128", True, "BLEURT cacheable"),
    ]


def _measurements() -> list[SeedMeasurement]:
    return [
        SeedMeasurement(
            seed=42,
            condition_a_ensemble_auroc=0.66,
            condition_b_ensemble_auroc=0.61,
            condition_a_per_verifier={"tier_a": 0.70, "tier_b": 0.60},
            condition_b_per_verifier={"tier_a": 0.62, "tier_b": 0.58},
            bleurt_threshold=0.62,
        ),
        SeedMeasurement(
            seed=137,
            condition_a_ensemble_auroc=0.70,
            condition_b_ensemble_auroc=0.63,
            condition_a_per_verifier={"tier_a": 0.72, "tier_b": 0.64},
            condition_b_per_verifier={"tier_a": 0.64, "tier_b": 0.60},
            bleurt_threshold=0.64,
        ),
    ]


def test_req_verify_2831_auroc_sample_split_and_seed_summary() -> None:
    """REQ-VERIFY-2831: AUROC, held-out split, and seed summaries are computed."""

    assert compute_auroc([0, 0, 1, 1], [0.10, 0.40, 0.35, 0.80]) == pytest.approx(0.75)
    assert compute_auroc([0, 1], [0.50, 0.50]) == pytest.approx(0.50)
    with pytest.raises(ValueError, match="same length"):
        compute_auroc([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least one seed"):
        summarize_measurements([])

    split = select_truthfulqa_indices(total_rows=817, n_questions=200, calibration_size=50, seed=42)
    assert len(split["test_indices"]) == 200
    assert len(split["calibration_indices"]) == 50
    assert set(split["test_indices"]).isdisjoint(split["calibration_indices"])
    assert split == select_truthfulqa_indices(817, 200, 50, 42)
    with pytest.raises(ValueError, match="not enough rows"):
        select_truthfulqa_indices(total_rows=249, n_questions=200, calibration_size=50, seed=42)

    summary = summarize_measurements(_measurements())

    assert summary["condition_a_production_auroc_mean"] == pytest.approx(0.68)
    assert summary["condition_a_production_auroc_std"] == pytest.approx(0.02)
    assert summary["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.62)
    assert summary["condition_b_architecture_only_auroc_std"] == pytest.approx(0.01)
    assert summary["learning_contribution"] == pytest.approx(0.06)
    assert summary["bleurt_threshold"] == pytest.approx(0.63)
    assert summary["per_verifier_condition_a_auroc"] == {
        "tier_a": [0.70, 0.72],
        "tier_b": [0.60, 0.64],
    }
    assert summary["per_verifier_condition_b_auroc"] == {
        "tier_a": [0.62, 0.64],
        "tier_b": [0.58, 0.60],
    }


def test_scenario_verify_2831_blocked_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2831: missing CUDA blocks before measurement and keeps schema."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=results_dir,
        random_seeds=(42, 137),
        started_at=100.0,
        clock=lambda: 104.5,
    )
    (results_dir / "nexus_constraint_memory_v2.json").write_text("state", encoding="utf-8")
    calls = {"measured": False}

    def measurement_runner(_config: ExperimentConfig, _state_files: list[dict[str, object]]):
        calls["measured"] = True
        return _measurements()

    artifact = run_experiment(
        config=config,
        precondition_probe=lambda _config, _state_files: [
            PreconditionCheck("cuda", False, "torch.cuda.is_available() returned False"),
            PreconditionCheck("hf_truthfulqa_generation", True, "TruthfulQA accessible"),
            PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
            PreconditionCheck("fr11_state_files", True, "state present"),
            PreconditionCheck("bleurt_base_128", True, "BLEURT cacheable"),
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    assert calls["measured"] is False
    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["corpus"] == "TruthfulQA-generation"
    assert artifact["n_questions"] == 200
    assert artifact["n_seeds"] == 2
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["bleurt_threshold"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert len(artifact["preconditions_checked"]) == 5
    assert artifact["baseline_comparison"]["production_minus_gpt3_mc1_approx"] is None
    assert (results_dir / "experiment_2831_truthfulqa_ensemble_eval.json").exists()


def test_req_verify_2831_fr11_state_manifest_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-2831: FR-11 state files are named by path, SHA, and size."""

    (tmp_path / "results" / "session_memory_1447" / "run").mkdir(parents=True)
    state_a = tmp_path / "results" / "nexus_constraint_memory_v2.json"
    state_b = tmp_path / "results" / "session_memory_1447" / "run" / "session_state.json"
    state_a.write_text("alpha", encoding="utf-8")
    state_b.write_text("beta", encoding="utf-8")

    files = discover_fr11_state_files(tmp_path)
    checksum_a = build_reproducibility_checksum(
        seeds=(42, 137),
        n_questions=200,
        sample_seed=42,
        calibration_size=50,
        state_files=files,
        model_specs={"name": "Qwen3.6-35B-A3B-GGUF", "quant": "Q4_K_M", "revision_sha": "abc"},
        scoring_method=mod.SCORING_METHOD,
    )

    state_b.write_text("beta2", encoding="utf-8")
    checksum_b = build_reproducibility_checksum(
        seeds=(42, 137),
        n_questions=200,
        sample_seed=42,
        calibration_size=50,
        state_files=discover_fr11_state_files(tmp_path),
        model_specs={"name": "Qwen3.6-35B-A3B-GGUF", "quant": "Q4_K_M", "revision_sha": "abc"},
        scoring_method=mod.SCORING_METHOD,
    )

    assert [item["path"] for item in files] == [
        "results/nexus_constraint_memory_v2.json",
        "results/session_memory_1447/run/session_state.json",
    ]
    assert files[0]["n_bytes"] == 5
    assert len(files[0]["sha256"]) == 64
    assert checksum_a != checksum_b
    assert mod.state_files_restored_sha_match(tmp_path, files) is False


def test_scenario_verify_2831_live_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2831-LIVE: all measured fields are populated on success."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "fr11_policy_cache_events_1512.jsonl").write_text("{}", encoding="utf-8")
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=results_dir,
        random_seeds=(42, 137),
        started_at=10.0,
        clock=lambda: 20.0,
        published_bleurt_verifier_comparators=(
            {"label": "published BLEURT verifier comparator", "auroc": 0.65, "source": "unit fixture"},
        ),
    )

    artifact = run_experiment(
        config=config,
        precondition_probe=lambda _config, _state_files: _all_preconditions(),
        measurement_runner=lambda _config, _state_files: _measurements(),
        write=True,
    )

    saved = json.loads((results_dir / "experiment_2831_truthfulqa_ensemble_eval.json").read_text())
    assert saved == artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.68)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.62)
    assert artifact["learning_contribution"] == pytest.approx(0.06)
    assert artifact["bleurt_threshold"] == pytest.approx(0.63)
    assert artifact["baseline_comparison"]["production_minus_gpt3_mc1_approx"] == pytest.approx(0.40)
    assert artifact["baseline_comparison"]["production_minus_bleurt_comparator_best"] == pytest.approx(0.03)
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["model_specs"]["name"] == "Qwen3.6-35B-A3B-GGUF"
    assert artifact["fr11_state_files"][0]["path"] == "results/fr11_policy_cache_events_1512.jsonl"
    assert artifact["state_files_restored_sha_match"] is True
    assert "learning_contribution" in artifact["field_principles"]
    assert "principle" in artifact["field_provenance"]["duration_s"]


def test_req_verify_2831_default_preconditions_and_backend_block(tmp_path: Path) -> None:
    """REQ-VERIFY-2831: default checks and missing live backend block honestly."""

    artifact = run_experiment(
        config=ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"].startswith("blocked_")
    assert any(
        check["resource"] == "hf_truthfulqa_generation"
        for check in artifact["preconditions_checked"]
    )

    backend_block = run_experiment(
        config=ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_probe=lambda _config, _state_files: _all_preconditions(),
        write=False,
    )
    assert backend_block["honest_verdict"] == "blocked_live_qwen36_backend_unavailable"
    assert backend_block["blocked_resources"] == ["live_backend"]

    unknown_block = run_experiment(
        config=ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_probe=lambda _config, _state_files: [
            PreconditionCheck("mystery", False, "unit test resource")
        ],
        write=False,
    )
    assert unknown_block["honest_verdict"] == "blocked_mystery"
    assert mod._blocked_verdict(_all_preconditions()) == "blocked_unknown_resource"


def test_req_verify_2831_resource_probe_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2831: dataset, CUDA, BLEURT, and model-cache probes report status."""

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *_args, **_kwargs: [{"question": "q"}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        mod.importlib.util,
        "find_spec",
        lambda name: object() if name in {"datasets", "bleurt"} else None,
    )
    assert mod._hf_truthfulqa_check().available is True
    assert mod._bleurt_check(tmp_path).available is True

    fake_datasets.load_dataset = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[attr-defined]
    failed = mod._hf_truthfulqa_check()
    assert failed.available is False
    assert "offline" in failed.detail

    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    def successful_dataset_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:2] == [sys.executable, "-c"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"available": True, "detail": "loaded validation[:1], n=1"}),
            "",
        )

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: object() if name == "datasets" else None)
    monkeypatch.setattr(mod.subprocess, "run", successful_dataset_runner)
    assert mod._hf_truthfulqa_check() == PreconditionCheck(
        "hf_truthfulqa_generation", True, "loaded validation[:1], n=1"
    )

    def failed_dataset_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, "", "dataset stderr")

    monkeypatch.setattr(mod.subprocess, "run", failed_dataset_runner)
    assert mod._hf_truthfulqa_check() == PreconditionCheck(
        "hf_truthfulqa_generation", False, "dataset stderr"
    )

    def invalid_dataset_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    monkeypatch.setattr(mod.subprocess, "run", invalid_dataset_runner)
    invalid_dataset = mod._hf_truthfulqa_check()
    assert invalid_dataset.available is False
    assert "invalid JSON" in invalid_dataset.detail

    def raising_dataset_runner(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("dataset subprocess failed")

    monkeypatch.setattr(mod.subprocess, "run", raising_dataset_runner)
    raised_dataset = mod._hf_truthfulqa_check()
    assert raised_dataset.available is False
    assert "dataset subprocess failed" in raised_dataset.detail

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    assert mod._hf_truthfulqa_check().available is False
    assert mod._bleurt_check(tmp_path).available is False

    fake_hub = types.ModuleType("huggingface_hub")

    class FakeHfApi:
        def model_info(self, model_id: str, files_metadata: bool = False) -> object:
            assert model_id == "Elron/bleurt-base-128"
            assert files_metadata is False
            return types.SimpleNamespace(sha="bleurt-sha", private=False)

    fake_hub.HfApi = FakeHfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(
        mod.importlib.util,
        "find_spec",
        lambda name: object() if name == "huggingface_hub" else None,
    )
    bleurt_hf = mod._bleurt_check(tmp_path)
    assert bleurt_hf.available is True
    assert "Elron/bleurt-base-128" in bleurt_hf.detail
    assert "bleurt-sha" in bleurt_hf.detail

    class FailingHfApi:
        def model_info(self, _model_id: str, files_metadata: bool = False) -> object:
            raise RuntimeError("hf offline")

    fake_hub.HfApi = FailingHfApi  # type: ignore[attr-defined]
    bleurt_hf_failed = mod._bleurt_check(tmp_path)
    assert bleurt_hf_failed.available is False
    assert "hf offline" in bleurt_hf_failed.detail

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: None)
    bleurt_dir = tmp_path / "models" / "bleurt-base-128"
    bleurt_dir.mkdir(parents=True)
    assert mod._bleurt_check(tmp_path).available is True

    models_dir = tmp_path / "models" / "qwen"
    models_dir.mkdir(parents=True)
    (models_dir / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"gguf")
    qwen = mod._qwen_cache_check(tmp_path)
    assert qwen.available is True
    assert mod.model_specs(tmp_path)["cache_complete"] is True

    def successful_cuda_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:2] == [sys.executable, "-c"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"available": True, "detail": "torch=cuda; device_count=2"}),
            "",
        )

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(mod.subprocess, "run", successful_cuda_runner)
    assert mod._cuda_check() == PreconditionCheck("cuda", True, "torch=cuda; device_count=2")

    def failed_cuda_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, "", "cuda stderr")

    monkeypatch.setattr(mod.subprocess, "run", failed_cuda_runner)
    assert mod._cuda_check() == PreconditionCheck("cuda", False, "cuda stderr")

    def invalid_cuda_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    monkeypatch.setattr(mod.subprocess, "run", invalid_cuda_runner)
    invalid_cuda = mod._cuda_check()
    assert invalid_cuda.available is False
    assert "invalid JSON" in invalid_cuda.detail

    def raising_cuda_runner(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("cuda subprocess failed")

    monkeypatch.setattr(mod.subprocess, "run", raising_cuda_runner)
    raised_cuda = mod._cuda_check()
    assert raised_cuda.available is False
    assert "cuda subprocess failed" in raised_cuda.detail

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert mod._cuda_check().available is True
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = mod._cuda_check()
    assert cuda.available is False
    assert "torch import failed" in cuda.detail
