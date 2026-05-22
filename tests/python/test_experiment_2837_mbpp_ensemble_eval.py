"""Tests for Exp 2837 MBPP ensemble-v7b retry artifact.

Spec: REQ-VERIFY-MBPP-2837, SCENARIO-VERIFY-MBPP-2837.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import mbpp_ensemble_v7b_exp2837 as mod
from carnot.eval.mbpp_ensemble_v7b import (
    ExperimentConfig,
    PreconditionCheck,
    SeedMeasurement,
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
            condition_a_ensemble_auroc=0.91,
            condition_b_ensemble_auroc=0.71,
            condition_a_per_verifier={"tier0r": 0.82, "semantic": 0.77},
            condition_b_per_verifier={"tier0r": 0.62, "semantic": 0.67},
            vanilla_pass_at_1=0.41,
        ),
        SeedMeasurement(
            seed=137,
            condition_a_ensemble_auroc=0.81,
            condition_b_ensemble_auroc=0.61,
            condition_a_per_verifier={"tier0r": 0.92, "semantic": 0.87},
            condition_b_per_verifier={"tier0r": 0.72, "semantic": 0.77},
            vanilla_pass_at_1=0.61,
        ),
    ]


def test_req_verify_mbpp_2837_required_field_principles_are_present() -> None:
    """REQ-VERIFY-MBPP-2837: each requested schema field has a principle."""

    assert mod.OUTPUT_FILENAME == "experiment_2837_mbpp_ensemble_eval.json"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in mod.FIELD_PRINCIPLES
        assert mod.FIELD_PRINCIPLES[field]
    assert mod.FIELD_PRINCIPLES["honest_verdict"] == "Terminal prefix."
    assert mod.FIELD_PRINCIPLES["corpus"] == "Identifies corpus."


def test_scenario_verify_mbpp_2837_blocked_qwen_cache_writes_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-MBPP-2837: missing Qwen cache blocks before measurement."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "fr11_policy_cache_events_1512.jsonl").write_text("{}", encoding="utf-8")
    calls = {"measured": False}

    def measurement_runner(
        _config: ExperimentConfig, _state_files: list[dict[str, object]]
    ) -> list[SeedMeasurement]:
        calls["measured"] = True
        return _measurements()

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=results_dir,
            random_seeds=(42, 137),
            n_problems=100,
            started_at=10.0,
            clock=lambda: 13.25,
        ),
        precondition_probe=lambda _config, _state_files: [
            PreconditionCheck("cuda", True, "cuda available"),
            PreconditionCheck("hf_mbpp", True, "mbpp accessible"),
            PreconditionCheck("qwen36_gguf_cache", False, "only .no_exist sentinel found"),
            PreconditionCheck("fr11_state_files", True, "state present"),
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    saved = json.loads((results_dir / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact
    assert calls["measured"] is False
    assert artifact["artifact"] == "experiment_2837_mbpp_ensemble_eval"
    assert artifact["schema"] == "carnot.mbpp_ensemble_v7b.exp2837"
    assert artifact["honest_verdict"] == "blocked_model_not_cached_qwen36_35b_a3b_gguf"
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_a_production_auroc_std"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_std"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["vanilla_qwen36_pass_at_1"] is None
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)


def test_req_verify_mbpp_2837_success_uses_exp2837_filename(tmp_path: Path) -> None:
    """REQ-VERIFY-MBPP-2837: measured seed rows summarize into the requested file."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "fr11_policy_cache_events_1512.jsonl").write_text("{}", encoding="utf-8")

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=results_dir,
            random_seeds=(42, 137),
            n_problems=100,
            started_at=100.0,
            clock=lambda: 112.0,
        ),
        precondition_probe=lambda _config, _state_files: _all_preconditions(),
        measurement_runner=lambda _config, _state_files: _measurements(),
        write=True,
    )

    assert (results_dir / mod.OUTPUT_FILENAME).is_file()
    assert not (results_dir / "experiment_2829_mbpp_ensemble_eval.json").exists()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.86)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.66)
    assert artifact["learning_contribution"] == pytest.approx(0.20)
    assert artifact["per_verifier_condition_a_auroc"] == {
        "tier0r": [0.82, 0.92],
        "semantic": [0.77, 0.87],
    }
    assert artifact["vanilla_qwen36_pass_at_1"] == pytest.approx(0.51)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES


def test_req_verify_mbpp_2837_cli_builds_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-MBPP-2837: CLI entrypoint targets the exp2837 artifact path."""

    captured: dict[str, object] = {}

    def fake_run(config: ExperimentConfig) -> dict[str, object]:
        captured["repo_root"] = config.repo_root
        captured["results_dir"] = config.results_dir
        captured["n_problems"] = config.n_problems
        return {"honest_verdict": "blocked_test"}

    monkeypatch.setattr(mod, "run_experiment", fake_run)

    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                str(tmp_path / "out"),
                "--n-problems",
                "100",
            ]
        )
        == 0
    )
    assert captured == {
        "repo_root": tmp_path,
        "results_dir": tmp_path / "out",
        "n_problems": 100,
    }
