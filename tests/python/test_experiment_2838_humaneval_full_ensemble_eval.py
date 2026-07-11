"""Tests for Exp 2838 HumanEval-full ensemble-v7b retry artifact.

Spec: REQ-VERIFY-HUMANEVAL-2838,
      SCENARIO-VERIFY-HUMANEVAL-2838-BLOCKED,
      SCENARIO-VERIFY-HUMANEVAL-2838-LIVE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import humaneval_full_ensemble_exp2838 as mod
from carnot.eval.humaneval_dual_condition_v3 import (
    ExperimentConfig,
    PreconditionCheck,
    SeedEvaluation,
)


SEEDS = (42, 137)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_exp2836(path: Path) -> None:
    _write_json(
        path,
        {
            "sota_runtime_ready": True,
            "selected_python": "/venv/python",
            "cached_sota_pair_result": {"called": True, "error": None, "result": None},
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": "/cache/qwen3.6-35b-a3b-q4_k_m.gguf",
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "sota_models_cached": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "path": "/cache/qwen3.6-35b-a3b-q4_k_m.gguf",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            ],
            "model_specs": {
                "primary": list(mod.base.PRIMARY_SOTA_MODEL_IDS),
                "legacy_cpu_smoke_only": list(mod.base.LEGACY_CPU_SMOKE_ONLY),
            },
        },
    )


def _write_state(root: Path) -> None:
    _write_json(root / "results" / "nexus_constraint_memory_v2.json", {"rules": []})
    _write_json(
        root / "results" / "session_memory_1447" / "run" / "session_state.json",
        {"case_memory": {"entries": []}},
    )


def _minimal_repo(root: Path) -> None:
    _write_exp2836(root / "results" / mod.base.EXP2836_FILENAME)
    _write_state(root)


def _all_checks() -> list[PreconditionCheck]:
    return [
        PreconditionCheck(
            "cuda",
            True,
            '.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" passed',
        ),
        PreconditionCheck("qwen36_gguf_cache", True, "/cache/qwen3.6-35b-a3b-q4_k_m.gguf"),
        PreconditionCheck("exp2836_artifact", True, "present"),
        PreconditionCheck("exp2836_sota_runtime_ready", True, "ready"),
        PreconditionCheck("exp2836_selected_python", True, "/venv/python"),
        PreconditionCheck("mandated_sota_model_path", True, "/cache/model.gguf"),
        PreconditionCheck("selected_python_cuda", True, "cuda available"),
        PreconditionCheck("humaneval_dataset", True, "loaded 164 rows"),
        PreconditionCheck("sandboxed_unit_test_execution", True, "runsc smoke passed"),
        PreconditionCheck("fr11_state_files", True, "count=2"),
    ]


def _evaluations() -> list[SeedEvaluation]:
    return [
        SeedEvaluation(
            seed=42,
            n_tasks=164,
            n_candidates=328,
            condition_a_ensemble_auroc=0.80,
            condition_b_ensemble_auroc=0.70,
            condition_a_per_verifier_auroc={"tier0r": 0.78, "sota_gguf_scorer": 0.76},
            condition_b_per_verifier_auroc={"tier0r": 0.68, "sota_gguf_scorer": 0.66},
            vanilla_pass_at_1=0.40,
            condition_a_ranked_pass_at_1=0.55,
            condition_b_ranked_pass_at_1=0.48,
            scorer_or_generator_model_path="/cache/model.gguf",
            candidate_label_sha256="a" * 64,
        ),
        SeedEvaluation(
            seed=137,
            n_tasks=164,
            n_candidates=328,
            condition_a_ensemble_auroc=0.90,
            condition_b_ensemble_auroc=0.60,
            condition_a_per_verifier_auroc={"tier0r": 0.88, "sota_gguf_scorer": 0.86},
            condition_b_per_verifier_auroc={"tier0r": 0.58, "sota_gguf_scorer": 0.56},
            vanilla_pass_at_1=0.50,
            condition_a_ranked_pass_at_1=0.70,
            condition_b_ranked_pass_at_1=0.52,
            scorer_or_generator_model_path="/cache/model.gguf",
            candidate_label_sha256="b" * 64,
        ),
    ]


def test_req_verify_humaneval_2838_required_fields_have_principles() -> None:
    """REQ-VERIFY-HUMANEVAL-2838: requested fields carry principle annotations."""

    assert mod.OUTPUT_FILENAME == "experiment_2838_humaneval_full_ensemble_eval.json"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in mod.FIELD_PRINCIPLES
        assert mod.FIELD_PRINCIPLES[field]
    assert mod.FIELD_PRINCIPLES["pass_at_1_vanilla"] == (
        "Generator baseline before Carnot verifier ranking."
    )
    assert mod.FIELD_PRINCIPLES["preconditions_checked"].startswith(
        "Records .venv/bin/python3 CUDA"
    )


def test_scenario_verify_humaneval_2838_blocked_dataset_writes_requested_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-HUMANEVAL-2838-BLOCKED: dataset failure leaves metrics null."""

    _minimal_repo(tmp_path)
    measured = False

    def measurement_runner(
        _config: ExperimentConfig,
        _state_files: list[dict[str, object]],
        _model_specs: dict[str, object],
    ) -> list[SeedEvaluation]:
        nonlocal measured
        measured = True
        return _evaluations()

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            random_seeds=SEEDS,
            started_at=10.0,
            clock=lambda: 14.25,
        ),
        precondition_probe=lambda _config, _state_files, _model_specs: [
            *_all_checks()[:7],
            PreconditionCheck("humaneval_dataset", False, "datasets package missing"),
            *_all_checks()[8:],
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact
    assert measured is False
    assert artifact["artifact"] == "experiment_2838_humaneval_full_ensemble_eval"
    assert artifact["schema"] == "carnot.humaneval_full_ensemble_eval.exp2838"
    assert artifact["honest_verdict"] == "blocked_humaneval_dataset"
    assert artifact["corpus"] == "HumanEval-full"
    assert artifact["n_problems"] == 164
    assert artifact["pass_at_1_vanilla"] is None
    assert artifact["pass_at_1_after_carnot_correct_production"] is None
    assert artifact["pass_at_1_after_carnot_correct_architecture_only"] is None
    assert artifact["peer_humaneval_verifier_baselines"] == {}
    assert artifact["peer_baseline_comparison"] is None
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["candidate_execution_summary"]["n_labeled_candidates"] == 0
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["state_files_restored_sha_match"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    assert artifact["field_provenance"]["pass_at_1_vanilla"] == {
        "principle": "Generator baseline before Carnot verifier ranking.",
        "satisfied_by": "blocked before measurement",
    }


def test_scenario_verify_humaneval_2838_live_success_maps_pass_at_1_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-HUMANEVAL-2838-LIVE: measured rows expose required pass@1 fields."""

    _minimal_repo(tmp_path)
    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            random_seeds=SEEDS,
            started_at=1.0,
            clock=lambda: 9.0,
        ),
        precondition_probe=lambda _config, _state_files, _model_specs: _all_checks(),
        measurement_runner=lambda _config, _state_files, _model_specs: _evaluations(),
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.85)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.65)
    assert artifact["learning_contribution"] == pytest.approx(0.20)
    assert artifact["pass_at_1_vanilla"] == pytest.approx(0.45)
    assert artifact["vanilla_qwen36_pass_at_1"] == pytest.approx(0.45)
    assert artifact["pass_at_1_after_carnot_correct_production"] == pytest.approx(0.625)
    assert artifact["pass_at_1_after_carnot_correct_architecture_only"] == pytest.approx(0.50)
    assert artifact["per_verifier_condition_b_auroc"] == {
        "sota_gguf_scorer": [0.66, 0.56],
        "tier0r": [0.68, 0.58],
    }
    assert artifact["candidate_execution_summary"] == {
        "n_labeled_candidates": 656,
        "n_tasks": 164,
        "n_seeds": 2,
    }
    assert artifact["model_specs"]["scorer_or_generator_model_paths_used"] == ["/cache/model.gguf"]
    assert artifact["field_provenance"]["pass_at_1_vanilla"]["satisfied_by"] == ("measured output")


def test_req_verify_humaneval_2838_default_probe_uses_venv_python3_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-HUMANEVAL-2838: preconditions record the mandated CUDA command."""

    config = ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results")
    qwen_path = tmp_path / "models" / "qwen.gguf"
    qwen_path.parent.mkdir()
    qwen_path.write_text("model", encoding="utf-8")
    model_specs = {
        "selected_python": "/venv/python",
        "selected_model_hf_id": mod.REQUIRED_QWEN36_HF_ID,
        "selected_model_path": str(qwen_path),
    }
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True and text is True and check is False
        assert timeout > 0
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "cuda ok\n", "")

    monkeypatch.setattr(
        mod.base,
        "probe_preconditions",
        lambda _config, _state_files, _model_specs: [
            PreconditionCheck("humaneval_dataset", True, "loaded 164 rows")
        ],
    )

    checks = mod.probe_preconditions(
        config,
        [{"path": "results/state.json", "sha256": "a" * 64, "size_bytes": 2}],
        model_specs,
        command_runner=fake_runner,
    )

    assert calls == [
        [
            str(tmp_path / ".venv" / "bin" / "python3"),
            "-c",
            "import torch; assert torch.cuda.is_available()",
        ]
    ]
    assert checks[0].resource == "cuda"
    assert checks[0].available is True
    assert (
        '.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"' in checks[0].detail
    )
    assert checks[1].resource == "qwen36_gguf_cache"
    assert checks[1].available is True
    assert checks[2:] == [PreconditionCheck("humaneval_dataset", True, "loaded 164 rows")]

    def failing_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(args[0], 2, "", "no cuda")

    failed = mod._venv_python3_cuda_check(config, command_runner=failing_runner)
    assert failed == PreconditionCheck(
        "cuda",
        False,
        '.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" failed; no cuda',
    )

    missing_qwen = mod._qwen36_cache_check({"selected_model_hf_id": "other"})
    assert missing_qwen.resource == "qwen36_gguf_cache"
    assert missing_qwen.available is False
    assert mod.REQUIRED_QWEN36_HF_ID in missing_qwen.detail

    cached_qwen = mod._qwen36_cache_check(
        {
            "sota_models_cached": [
                {
                    "hf_id": mod.REQUIRED_QWEN36_HF_ID,
                    "path": str(qwen_path),
                    "resolved_path": str(qwen_path),
                    "model_path": str(qwen_path),
                }
            ]
        }
    )
    assert cached_qwen.available is True

    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: tmp_path / "empty-home"))
    uncached_qwen = mod._qwen36_cache_check({"selected_model_hf_id": mod.REQUIRED_QWEN36_HF_ID})
    assert uncached_qwen.available is False
    assert "no real" in uncached_qwen.detail


def test_req_verify_humaneval_2838_cli_builds_requested_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-HUMANEVAL-2838: CLI targets the requested artifact path."""

    captured: dict[str, object] = {}

    def fake_run(config: ExperimentConfig) -> dict[str, object]:
        captured["repo_root"] = config.repo_root
        captured["results_dir"] = config.results_dir
        captured["n_tasks"] = config.n_tasks
        return {"honest_verdict": "blocked_test"}

    monkeypatch.setattr(mod, "run_experiment", fake_run)

    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                str(tmp_path / "out"),
                "--n-tasks",
                "164",
            ]
        )
        == 0
    )
    assert captured == {
        "repo_root": tmp_path,
        "results_dir": tmp_path / "out",
        "n_tasks": 164,
    }
