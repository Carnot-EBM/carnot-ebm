import builtins
import json
import sys
import types
from pathlib import Path

import pytest

import carnot.eval.humaneval_ensemble_v7b as mod
from carnot.eval.humaneval_ensemble_v7b import (
    ExperimentConfig,
    PreconditionCheck,
    SeedMeasurement,
    build_reproducibility_checksum,
    compute_auroc,
    discover_fr11_state_files,
    run_experiment,
    summarize_measurements,
)


def _all_preconditions() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("cuda", True, "cuda available"),
        PreconditionCheck("hf_openai_humaneval", True, "HumanEval accessible"),
        PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
        PreconditionCheck("fr11_state_files", True, "state present"),
    ]


def _measurements() -> list[SeedMeasurement]:
    return [
        SeedMeasurement(
            seed=42,
            condition_a_ensemble_auroc=0.91,
            condition_b_ensemble_auroc=0.71,
            condition_a_per_verifier={"tier_a": 0.82, "tier_b": 0.76},
            condition_b_per_verifier={"tier_a": 0.62, "tier_b": 0.70},
            vanilla_pass_at_1=0.35,
            production_pass_at_1=0.55,
            architecture_only_pass_at_1=0.45,
        ),
        SeedMeasurement(
            seed=137,
            condition_a_ensemble_auroc=0.81,
            condition_b_ensemble_auroc=0.61,
            condition_a_per_verifier={"tier_a": 0.92, "tier_b": 0.86},
            condition_b_per_verifier={"tier_a": 0.72, "tier_b": 0.80},
            vanilla_pass_at_1=0.45,
            production_pass_at_1=0.65,
            architecture_only_pass_at_1=0.55,
        ),
    ]


def test_req_verify_2830_auroc_pass_at_1_and_seed_summary() -> None:
    """REQ-VERIFY-2830: AUROC, pass@1, means, stds, and verifier lists are computed."""

    assert compute_auroc([0, 0, 1, 1], [0.10, 0.40, 0.35, 0.80]) == pytest.approx(0.75)
    assert compute_auroc([0, 1], [0.50, 0.50]) == pytest.approx(0.50)
    with pytest.raises(ValueError, match="same length"):
        compute_auroc([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least one seed"):
        summarize_measurements([])

    summary = summarize_measurements(_measurements())

    assert summary["condition_a_production_auroc_mean"] == pytest.approx(0.86)
    assert summary["condition_a_production_auroc_std"] == pytest.approx(0.05)
    assert summary["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.66)
    assert summary["condition_b_architecture_only_auroc_std"] == pytest.approx(0.05)
    assert summary["auroc_learning_contribution"] == pytest.approx(0.20)
    assert summary["pass_at_1_vanilla"] == pytest.approx(0.40)
    assert summary["pass_at_1_after_carnot_correct_production"] == pytest.approx(0.60)
    assert summary["pass_at_1_after_carnot_correct_architecture_only"] == pytest.approx(0.50)
    assert summary["learning_contribution"] == pytest.approx(0.10)
    assert summary["repair_lift_production_vs_vanilla"] == pytest.approx(0.20)
    assert summary["repair_lift_architecture_only_vs_vanilla"] == pytest.approx(0.10)
    assert summary["per_verifier_condition_a_auroc"] == {
        "tier_a": [0.82, 0.92],
        "tier_b": [0.76, 0.86],
    }
    assert summary["per_verifier_condition_b_auroc"] == {
        "tier_a": [0.62, 0.72],
        "tier_b": [0.70, 0.80],
    }


def test_scenario_verify_2830_blocked_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2830: missing CUDA blocks before measurement and keeps schema."""

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
            PreconditionCheck("hf_openai_humaneval", True, "HumanEval accessible"),
            PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
            PreconditionCheck("fr11_state_files", True, "state present"),
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    assert calls["measured"] is False
    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["corpus"] == "HumanEval-full"
    assert artifact["n_problems"] == 164
    assert artifact["n_seeds"] == 2
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["pass_at_1_vanilla"] is None
    assert artifact["pass_at_1_after_carnot_correct_production"] is None
    assert artifact["pass_at_1_after_carnot_correct_architecture_only"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["peer_humaneval_verifier_baselines"] == []
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert len(artifact["preconditions_checked"]) == 4
    assert (results_dir / "experiment_2830_humaneval_full_ensemble_eval.json").exists()


def test_req_verify_2830_fr11_state_manifest_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-2830: FR-11 state files are named by path, SHA, and size."""

    (tmp_path / "results" / "session_memory_1447" / "run").mkdir(parents=True)
    state_a = tmp_path / "results" / "nexus_constraint_memory_v2.json"
    state_b = tmp_path / "results" / "session_memory_1447" / "run" / "session_state.json"
    state_a.write_text("alpha", encoding="utf-8")
    state_b.write_text("beta", encoding="utf-8")

    files = discover_fr11_state_files(tmp_path)
    checksum_a = build_reproducibility_checksum(
        seeds=(42, 137),
        n_problems=164,
        state_files=files,
        model_specs={"name": "Qwen3.6-35B-A3B-GGUF", "quant": "Q4_K_M", "revision_sha": "abc"},
    )

    state_b.write_text("beta2", encoding="utf-8")
    checksum_b = build_reproducibility_checksum(
        seeds=(42, 137),
        n_problems=164,
        state_files=discover_fr11_state_files(tmp_path),
        model_specs={"name": "Qwen3.6-35B-A3B-GGUF", "quant": "Q4_K_M", "revision_sha": "abc"},
    )

    assert [item["path"] for item in files] == [
        "results/nexus_constraint_memory_v2.json",
        "results/session_memory_1447/run/session_state.json",
    ]
    assert files[0]["n_bytes"] == 5
    assert len(files[0]["sha256"]) == 64
    assert checksum_a != checksum_b
    assert mod.state_files_restored_sha_match(tmp_path, files) is False


def test_scenario_verify_2830_live_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2830-LIVE: all measured fields are populated on success."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "fr11_policy_cache_events_1512.jsonl").write_text("{}", encoding="utf-8")
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=results_dir,
        random_seeds=(42, 137),
        started_at=10.0,
        clock=lambda: 20.0,
        peer_baselines=(
            {"label": "peer HumanEval verifier", "pass_at_1": 0.52, "source": "unit fixture"},
        ),
    )

    artifact = run_experiment(
        config=config,
        precondition_probe=lambda _config, _state_files: _all_preconditions(),
        measurement_runner=lambda _config, _state_files: _measurements(),
        write=True,
    )

    saved = json.loads(
        (results_dir / "experiment_2830_humaneval_full_ensemble_eval.json").read_text()
    )
    assert saved == artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.86)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.66)
    assert artifact["pass_at_1_vanilla"] == pytest.approx(0.40)
    assert artifact["pass_at_1_after_carnot_correct_production"] == pytest.approx(0.60)
    assert artifact["pass_at_1_after_carnot_correct_architecture_only"] == pytest.approx(0.50)
    assert artifact["learning_contribution"] == pytest.approx(0.10)
    assert artifact["baseline_comparison"]["production_minus_vanilla"] == pytest.approx(0.20)
    assert artifact["baseline_comparison"]["production_minus_peer_best"] == pytest.approx(0.08)
    assert artifact["peer_humaneval_verifier_baselines"][0]["label"] == "peer HumanEval verifier"
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["model_specs"]["name"] == "Qwen3.6-35B-A3B-GGUF"
    assert artifact["fr11_state_files"][0]["path"] == "results/fr11_policy_cache_events_1512.jsonl"
    assert artifact["state_files_restored_sha_match"] is True


def test_req_verify_2830_default_preconditions_and_backend_block(tmp_path: Path) -> None:
    """REQ-VERIFY-2830: default checks and missing live backend block honestly."""

    artifact = run_experiment(
        config=ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"].startswith("blocked_")
    assert any(check["resource"] == "hf_openai_humaneval" for check in artifact["preconditions_checked"])

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


def test_req_verify_2830_humaneval_probe_and_qwen_cache_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2830: dataset and model-cache probes report concrete status."""

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *_args, **_kwargs: [{"prompt": "problem"}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(
        mod.importlib.util, "find_spec", lambda name: object() if name == "datasets" else None
    )
    assert mod._hf_humaneval_check().available is True

    fake.load_dataset = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[attr-defined]
    failed = mod._hf_humaneval_check()
    assert failed.available is False
    assert "offline" in failed.detail

    models_dir = tmp_path / "models" / "qwen"
    models_dir.mkdir(parents=True)
    (models_dir / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"gguf")
    qwen = mod._qwen_cache_check(tmp_path)
    assert qwen.available is True
    assert mod.model_specs(tmp_path)["cache_complete"] is True

    real_import = builtins.__import__

    def failing_torch_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_torch_import)
    cuda = mod._cuda_check()
    assert cuda.available is False
    assert "torch missing" in cuda.detail
