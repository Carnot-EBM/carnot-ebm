"""Tests for Exp 2838 MBPP dual-condition v3.

Spec: REQ-VERIFY-2838,
      SCENARIO-VERIFY-2838-BLOCKED,
      SCENARIO-VERIFY-2838-LIVE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import mbpp_dual_condition_v3 as mod
from carnot.eval.mbpp_dual_condition_v3 import (
    ExperimentConfig,
    PreconditionCheck,
    SeedEvaluation,
    model_specs_from_exp2836,
    run_experiment,
    summarize_evaluations,
)


SEEDS = (42, 137)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_exp2836(
    path: Path, *, ready: bool = True, selected_python: str = "/venv/python"
) -> None:
    _write_json(
        path,
        {
            "sota_runtime_ready": ready,
            "selected_python": selected_python,
            "cached_sota_pair_result": {"called": True, "error": None, "result": None},
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "sota_models_cached": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "path": "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            ],
            "model_specs": {
                "primary": list(mod.PRIMARY_SOTA_MODEL_IDS),
                "legacy_cpu_smoke_only": list(mod.LEGACY_CPU_SMOKE_ONLY),
            },
        },
    )


def _write_state(root: Path) -> None:
    _write_json(root / "results" / "nexus_constraint_memory_v2.json", {"rules": []})
    _write_json(
        root / "results" / "session_memory_1447" / "run" / "session_state.json",
        {"case_memory": {"entries": []}},
    )


def _minimal_repo(root: Path, *, ready: bool = True) -> None:
    _write_exp2836(root / "results" / mod.EXP2836_FILENAME, ready=ready)
    _write_state(root)


def _all_checks() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("exp2836_artifact", True, "present"),
        PreconditionCheck("exp2836_sota_runtime_ready", True, "ready"),
        PreconditionCheck("exp2836_selected_python", True, "/venv/python"),
        PreconditionCheck("mandated_sota_model_path", True, "/cache/model.gguf"),
        PreconditionCheck("selected_python_cuda", True, "cuda available"),
        PreconditionCheck("mbpp_dataset", True, "100 rows"),
        PreconditionCheck("sandboxed_unit_test_execution", True, "runsc smoke passed"),
        PreconditionCheck("fr11_state_files", True, "count=2"),
    ]


def _evaluations() -> list[SeedEvaluation]:
    return [
        SeedEvaluation(
            seed=42,
            n_tasks=100,
            n_candidates=200,
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
            n_tasks=100,
            n_candidates=200,
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


def test_req_verify_2838_model_specs_record_mandated_sota_path(tmp_path: Path) -> None:
    """REQ-VERIFY-2838: Exp 2836 selected Python and SOTA GGUF path are recorded."""

    preflight_path = tmp_path / "results" / mod.EXP2836_FILENAME
    _write_exp2836(preflight_path, selected_python="/tmp/venv/bin/python")
    specs = model_specs_from_exp2836(mod.load_exp2836_preflight(preflight_path))

    assert specs["sota_runtime_ready"] is True
    assert specs["selected_python"] == "/tmp/venv/bin/python"
    assert specs["selected_model_path"] == "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf"
    assert specs["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert specs["headline_required_any_of"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert specs["legacy_cpu_smoke_only"] == list(mod.LEGACY_CPU_SMOKE_ONLY)
    assert mod._extract_model_paths({"x": [{"resolved_gguf": "/nested/model.gguf"}]}) == [
        "/nested/model.gguf"
    ]


def test_scenario_verify_2838_blocked_mbpp_dataset_writes_required_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2838-BLOCKED: missing MBPP rows block without fabricated metrics."""

    _minimal_repo(tmp_path)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            random_seeds=SEEDS,
            started_at=10.0,
            clock=lambda: 14.25,
        ),
        precondition_probe=lambda _config, _state_files, _model_specs: [
            *_all_checks()[:5],
            PreconditionCheck("mbpp_dataset", False, "datasets package missing"),
            *_all_checks()[6:],
        ],
        measurement_runner=lambda _config, _state_files, _model_specs: _evaluations(),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_mbpp_dataset"
    assert artifact["n_tasks"] == 100
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["pass_at_1"] is None
    assert artifact["ranking_lift"] is None
    assert artifact["candidate_execution_summary"]["n_labeled_candidates"] == 0
    assert artifact["state_files_restored_sha_match"] is True
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_verify_2838_live_success_summary(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2838-LIVE: measured seed rows summarize A/B AUROC and ranking."""

    _minimal_repo(tmp_path)
    artifact = run_experiment(
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
    assert artifact["per_verifier_condition_b_auroc"] == {
        "sota_gguf_scorer": [0.66, 0.56],
        "tier0r": [0.68, 0.58],
    }
    assert artifact["pass_at_1"] == {
        "vanilla_mean": pytest.approx(0.45),
        "condition_a_ranked_mean": pytest.approx(0.625),
        "condition_b_ranked_mean": pytest.approx(0.50),
    }
    assert artifact["ranking_lift"]["condition_a_vs_vanilla_mean"] == pytest.approx(0.175)
    assert artifact["ranking_lift"]["condition_b_vs_vanilla_mean"] == pytest.approx(0.05)
    assert artifact["candidate_execution_summary"] == {
        "n_labeled_candidates": 400,
        "n_tasks": 100,
        "n_seeds": 2,
    }
    assert artifact["model_specs"]["scorer_or_generator_model_paths_used"] == ["/cache/model.gguf"]


def test_req_verify_2838_summarize_rejects_empty_and_mismatched_evaluations() -> None:
    """REQ-VERIFY-2838: summaries require measured rows for the configured task count."""

    with pytest.raises(ValueError, match="at least one"):
        summarize_evaluations([], n_tasks=100)

    bad = SeedEvaluation(
        seed=42,
        n_tasks=99,
        n_candidates=1,
        condition_a_ensemble_auroc=0.5,
        condition_b_ensemble_auroc=0.5,
        condition_a_per_verifier_auroc={},
        condition_b_per_verifier_auroc={},
        vanilla_pass_at_1=0.0,
        condition_a_ranked_pass_at_1=0.0,
        condition_b_ranked_pass_at_1=0.0,
        scorer_or_generator_model_path="/cache/model.gguf",
        candidate_label_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="n_tasks"):
        summarize_evaluations([bad], n_tasks=100)


def test_req_verify_2838_probe_preconditions_and_live_backend_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2838: default probes name all gates and backend absence blocks honestly."""

    _minimal_repo(tmp_path)
    specs = model_specs_from_exp2836(
        mod.load_exp2836_preflight(tmp_path / "results" / mod.EXP2836_FILENAME)
    )
    monkeypatch.setattr(mod.Path, "is_file", lambda self: True)

    def fake_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True and text is True and check is False
        assert timeout > 0 and "PYTHONPATH" in env
        script = command[-1]
        if "torch" in script:
            payload = {"available": True, "detail": "cuda ok"}
        elif "datasets" in script:
            payload = {"available": True, "detail": "loaded MBPP sanitized test, n=1"}
        else:
            payload = {"available": True, "detail": "runsc smoke passed"}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    checks = mod.probe_preconditions(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        mod.discover_fr11_state_files(tmp_path),
        specs,
        command_runner=fake_runner,
    )
    assert [check.resource for check in checks] == [
        "exp2836_artifact",
        "exp2836_sota_runtime_ready",
        "exp2836_selected_python",
        "mandated_sota_model_path",
        "selected_python_cuda",
        "mbpp_dataset",
        "sandboxed_unit_test_execution",
        "fr11_state_files",
    ]
    assert all(check.available for check in checks)

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_probe=lambda _config, _state_files, _model_specs: _all_checks(),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_live_mbpp_backend_unavailable"
    assert artifact["blocked_resources"] == ["live_backend"]


def test_req_verify_2838_probe_failure_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2838: probe failures and CLI invocation remain terminal and explicit."""

    config = ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results")
    assert config.preflight_path() == tmp_path / "results" / mod.EXP2836_FILENAME
    custom = ExperimentConfig(repo_root=tmp_path, exp2836_path=tmp_path / "custom" / "exp2836.json")
    assert custom.preflight_path() == tmp_path / "custom" / "exp2836.json"
    assert mod.load_exp2836_preflight(config.preflight_path()) == {}
    assert mod._blocked_verdict([PreconditionCheck("mystery", False, "x")]) == "blocked_mystery"
    assert mod._run_json_probe(
        selected_python="",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
    ) == PreconditionCheck("probe", False, "selected_python missing")

    def failing_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], 7, "not-json", "boom")

    failed = mod._run_json_probe(
        selected_python="/venv/python",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
        command_runner=failing_runner,
    )
    assert failed == PreconditionCheck("probe", False, "boom")

    def invalid_json_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(args[0], 0, "not-json", "")

    invalid = mod._run_json_probe(
        selected_python="/venv/python",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
        command_runner=invalid_json_runner,
    )
    assert invalid.resource == "probe"
    assert invalid.available is False
    assert "invalid JSON" in invalid.detail

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
                "--n-tasks",
                "17",
            ]
        )
        == 0
    )
    assert calls[0].repo_root == tmp_path
    assert calls[0].results_dir == tmp_path / "custom-results"
    assert calls[0].n_tasks == 17
