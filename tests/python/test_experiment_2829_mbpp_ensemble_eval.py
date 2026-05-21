import json
import sys
import types
from pathlib import Path

import pytest

import carnot.eval.mbpp_ensemble_v7b as mod
from carnot.eval.mbpp_ensemble_v7b import (
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
        PreconditionCheck("hf_mbpp", True, "mbpp accessible"),
        PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
        PreconditionCheck("fr11_state_files", True, "state present"),
    ]


def _measurements() -> list[SeedMeasurement]:
    return [
        SeedMeasurement(
            seed=42,
            condition_a_ensemble_auroc=0.90,
            condition_b_ensemble_auroc=0.70,
            condition_a_per_verifier={"tier_a": 0.80, "tier_b": 0.75},
            condition_b_per_verifier={"tier_a": 0.60, "tier_b": 0.70},
            vanilla_pass_at_1=0.40,
        ),
        SeedMeasurement(
            seed=137,
            condition_a_ensemble_auroc=0.80,
            condition_b_ensemble_auroc=0.60,
            condition_a_per_verifier={"tier_a": 0.90, "tier_b": 0.85},
            condition_b_per_verifier={"tier_a": 0.70, "tier_b": 0.80},
            vanilla_pass_at_1=0.60,
        ),
    ]


def test_req_verify_2829_auroc_and_seed_summary() -> None:
    """REQ-VERIFY-2829-6: AUROC, means, stds, and per-verifier lists are computed."""

    assert compute_auroc([0, 0, 1, 1], [0.10, 0.40, 0.35, 0.80]) == pytest.approx(0.75)
    assert compute_auroc([0, 1], [0.50, 0.50]) == pytest.approx(0.50)
    with pytest.raises(ValueError, match="same length"):
        compute_auroc([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least one seed"):
        summarize_measurements([])

    summary = summarize_measurements(_measurements())

    assert summary["condition_a_production_auroc_mean"] == pytest.approx(0.85)
    assert summary["condition_a_production_auroc_std"] == pytest.approx(0.05)
    assert summary["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.65)
    assert summary["condition_b_architecture_only_auroc_std"] == pytest.approx(0.05)
    assert summary["learning_contribution"] == pytest.approx(0.20)
    assert summary["per_verifier_condition_a_auroc"] == {
        "tier_a": [0.80, 0.90],
        "tier_b": [0.75, 0.85],
    }
    assert summary["per_verifier_condition_b_auroc"] == {
        "tier_a": [0.60, 0.70],
        "tier_b": [0.70, 0.80],
    }
    assert summary["vanilla_qwen36_pass_at_1"] == pytest.approx(0.50)


def test_scenario_verify_2829_blocked_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2829: missing CUDA blocks before measurement and keeps schema."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=results_dir,
        random_seeds=(42, 137),
        n_problems=100,
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
            PreconditionCheck("hf_mbpp", True, "mbpp accessible"),
            PreconditionCheck("qwen36_gguf_cache", True, "qwen cached"),
            PreconditionCheck("fr11_state_files", True, "state present"),
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    assert calls["measured"] is False
    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["corpus"] == "MBPP-sanitized-test"
    assert artifact["n_problems"] == 100
    assert artifact["n_seeds"] == 2
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["vanilla_qwen36_pass_at_1"] is None
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert len(artifact["preconditions_checked"]) == 4
    assert (results_dir / "experiment_2829_mbpp_ensemble_eval.json").exists()


def test_req_verify_2829_fr11_state_manifest_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-2829-1/5: FR-11 state files are named by path, SHA, and size."""

    (tmp_path / "results" / "session_memory_1447" / "run").mkdir(parents=True)
    state_a = tmp_path / "results" / "nexus_constraint_memory_v2.json"
    state_b = tmp_path / "results" / "session_memory_1447" / "run" / "session_state.json"
    state_a.write_text("alpha", encoding="utf-8")
    state_b.write_text("beta", encoding="utf-8")

    files = discover_fr11_state_files(tmp_path)
    checksum_a = build_reproducibility_checksum(
        seeds=(42, 137),
        n_problems=100,
        state_files=files,
        model_specs={"name": "Qwen3.6-35B-A3B-GGUF", "quant": "Q4_K_M", "revision_sha": "abc"},
    )

    state_b.write_text("beta2", encoding="utf-8")
    checksum_b = build_reproducibility_checksum(
        seeds=(42, 137),
        n_problems=100,
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


def test_scenario_verify_2829_live_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2829-LIVE: all measured fields are populated on success."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "fr11_policy_cache_events_1512.jsonl").write_text("{}", encoding="utf-8")
    config = ExperimentConfig(
        repo_root=tmp_path,
        results_dir=results_dir,
        random_seeds=(42, 137),
        n_problems=100,
        started_at=10.0,
        clock=lambda: 20.0,
    )

    artifact = run_experiment(
        config=config,
        precondition_probe=lambda _config, _state_files: _all_preconditions(),
        measurement_runner=lambda _config, _state_files: _measurements(),
        write=True,
    )

    saved = json.loads((results_dir / "experiment_2829_mbpp_ensemble_eval.json").read_text())
    assert saved == artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.85)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.65)
    assert artifact["learning_contribution"] == pytest.approx(0.20)
    assert artifact["random_seeds_used"] == [42, 137]
    assert artifact["model_specs"]["name"] == "Qwen3.6-35B-A3B-GGUF"
    assert artifact["fr11_state_files"][0]["path"] == "results/fr11_policy_cache_events_1512.jsonl"
    assert artifact["state_files_restored_sha_match"] is True


def test_req_verify_2829_default_preconditions_and_backend_block(tmp_path: Path) -> None:
    """REQ-VERIFY-2829-1/2: default checks and missing live backend block honestly."""

    artifact = run_experiment(
        config=ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"].startswith("blocked_")
    assert any(check["resource"] == "hf_mbpp" for check in artifact["preconditions_checked"])

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


def test_req_verify_2829_mbpp_probe_and_qwen_cache_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2829-1: dataset and model-cache probes report concrete status."""

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *_args, **_kwargs: [  # type: ignore[attr-defined]
        {"text": "problem"}
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(
        mod.importlib.util, "find_spec", lambda name: object() if name == "datasets" else None
    )
    assert mod._hf_mbpp_check().available is True

    fake.load_dataset = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[attr-defined]
    failed = mod._hf_mbpp_check()
    assert failed.available is False
    assert "offline" in failed.detail

    models_dir = tmp_path / "models" / "qwen"
    models_dir.mkdir(parents=True)
    (models_dir / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"gguf")
    qwen = mod._qwen_cache_check(tmp_path)
    assert qwen.available is True
    assert mod.model_specs(tmp_path)["cache_complete"] is True
