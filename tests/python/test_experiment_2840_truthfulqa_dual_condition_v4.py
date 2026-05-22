"""Tests for Exp 2840 TruthfulQA dual-condition v4.

Spec: REQ-VERIFY-2840,
      SCENARIO-VERIFY-2840-BLOCKED,
      SCENARIO-VERIFY-2840-LIVE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import truthfulqa_dual_condition_v4 as mod
from carnot.eval.truthfulqa_dual_condition_v4 import (
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
    path: Path,
    *,
    model_path: Path,
    ready: bool = True,
    selected_python: str = "/venv/python",
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
                    "model_path": str(model_path),
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "sota_models_cached": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "path": str(model_path),
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


def _write_retired_2823_evidence(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired_extras:\n"
        "  - experiment_id: 2823\n"
        "    reason: Fabricated TruthfulQA artifact retired and moved to legacy.\n",
        encoding="utf-8",
    )
    _write_json(
        root / "legacy" / "fabricated" / "experiment_2823_truthfulqa_ensemble_eval.json",
        {"honest_verdict": "retired_fabricated", "flagged_adversarial": True},
    )


def _minimal_repo(root: Path, *, ready: bool = True) -> Path:
    model_path = root / "models" / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"gguf")
    _write_exp2836(root / "results" / mod.EXP2836_FILENAME, ready=ready, model_path=model_path)
    _write_state(root)
    _write_retired_2823_evidence(root)
    return model_path


def _local_scorer_detail() -> str:
    return json.dumps(
        {
            "name": "sklearn_tfidf_cosine",
            "version": "1.7.2",
            "reference": "openspec/capabilities/pipeline/spec.md",
            "closed_weight_judge": False,
        },
        sort_keys=True,
    )


def _all_checks() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("exp2836_artifact", True, "present"),
        PreconditionCheck("exp2836_sota_runtime_ready", True, "ready"),
        PreconditionCheck("exp2836_selected_python", True, "/venv/python"),
        PreconditionCheck("mandated_sota_model_path", True, "/cache/model.gguf"),
        PreconditionCheck("selected_python_cuda", True, "cuda available"),
        PreconditionCheck("truthfulqa_generation_split", True, "loaded 817 rows"),
        PreconditionCheck("local_scorer", True, _local_scorer_detail()),
        PreconditionCheck("fr11_state_files", True, "count=2"),
        PreconditionCheck("retired_exp2823", True, "manifest and legacy artifact present"),
    ]


def _evaluations() -> list[SeedEvaluation]:
    return [
        SeedEvaluation(
            seed=42,
            n_questions=200,
            n_candidates=200,
            condition_a_ensemble_auroc=0.70,
            condition_b_ensemble_auroc=0.60,
            condition_a_per_verifier_auroc={"tier0r": 0.72, "local_semantic": 0.68},
            condition_b_per_verifier_auroc={"tier0r": 0.62, "local_semantic": 0.58},
            scorer_name="sklearn_tfidf_cosine",
            scorer_version="1.7.2",
            scorer_threshold=0.42,
            calibration_size=50,
            scorer_or_generator_model_path="/cache/model.gguf",
            candidate_label_sha256="a" * 64,
        ),
        SeedEvaluation(
            seed=137,
            n_questions=200,
            n_candidates=200,
            condition_a_ensemble_auroc=0.80,
            condition_b_ensemble_auroc=0.70,
            condition_a_per_verifier_auroc={"tier0r": 0.82, "local_semantic": 0.78},
            condition_b_per_verifier_auroc={"tier0r": 0.72, "local_semantic": 0.68},
            scorer_name="sklearn_tfidf_cosine",
            scorer_version="1.7.2",
            scorer_threshold=0.46,
            calibration_size=50,
            scorer_or_generator_model_path="/cache/model.gguf",
            candidate_label_sha256="b" * 64,
        ),
    ]


def test_req_verify_2840_model_specs_record_exp2836_sota_path(tmp_path: Path) -> None:
    """REQ-VERIFY-2840: Exp 2836 selected Python and mandated GGUF path are recorded."""

    model_path = _minimal_repo(tmp_path)
    preflight_path = tmp_path / "results" / mod.EXP2836_FILENAME
    _write_exp2836(
        preflight_path,
        selected_python="/tmp/venv/bin/python",
        model_path=model_path,
    )

    specs = model_specs_from_exp2836(mod.load_exp2836_preflight(preflight_path))

    assert specs["sota_runtime_ready"] is True
    assert specs["selected_python"] == "/tmp/venv/bin/python"
    assert specs["selected_model_path"] == str(model_path)
    assert specs["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert specs["headline_required_any_of"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert specs["legacy_cpu_smoke_only"] == list(mod.LEGACY_CPU_SMOKE_ONLY)
    assert mod._extract_model_paths({"x": [{"resolved_gguf": "/nested/model.gguf"}]}) == [
        "/nested/model.gguf"
    ]


def test_scenario_verify_2840_blocked_dataset_writes_required_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2840-BLOCKED: missing TruthfulQA rows block without metrics."""

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
            PreconditionCheck("truthfulqa_generation_split", False, "datasets package missing"),
            *_all_checks()[6:],
        ],
        measurement_runner=measurement_runner,
        write=True,
    )

    assert measured is False
    assert artifact["honest_verdict"] == "blocked_truthfulqa_generation_split"
    assert artifact["run_date"] == "20260522"
    assert artifact["n_questions"] == 200
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["local_scorer"]["name"] == "sklearn_tfidf_cosine"
    assert artifact["retired_exp2823_not_used"] is True
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_condition_a_auroc"] == {}
    assert artifact["per_verifier_condition_b_auroc"] == {}
    assert artifact["calibration"] is None
    assert artifact["candidate_summary"]["n_labeled_candidates"] == 0
    assert artifact["state_files_restored_sha_match"] is True
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_verify_2840_live_success_summary(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2840-LIVE: measured seed rows summarize A/B AUROC."""

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
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.75)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.65)
    assert artifact["learning_contribution"] == pytest.approx(0.10)
    assert artifact["local_scorer"] == {
        "available": True,
        "closed_weight_judge": False,
        "name": "sklearn_tfidf_cosine",
        "reference": "openspec/capabilities/pipeline/spec.md",
        "version": "1.7.2",
    }
    assert artifact["calibration"] == {
        "threshold_mean": pytest.approx(0.44),
        "threshold_std": pytest.approx(0.02),
        "calibration_size": 50,
        "label_source": "local_scorer_against_best_answer",
    }
    assert artifact["per_verifier_condition_b_auroc"] == {
        "local_semantic": [0.58, 0.68],
        "tier0r": [0.62, 0.72],
    }
    assert artifact["candidate_summary"] == {
        "n_labeled_candidates": 400,
        "n_questions": 200,
        "n_seeds": 2,
    }
    assert artifact["model_specs"]["scorer_or_generator_model_paths_used"] == [
        "/cache/model.gguf"
    ]


def test_req_verify_2840_summarize_rejects_empty_and_mismatched_evaluations() -> None:
    """REQ-VERIFY-2840: summaries require measured rows for the configured N=200."""

    with pytest.raises(ValueError, match="at least one"):
        summarize_evaluations([], n_questions=200)

    bad = SeedEvaluation(
        seed=42,
        n_questions=199,
        n_candidates=1,
        condition_a_ensemble_auroc=0.5,
        condition_b_ensemble_auroc=0.5,
        condition_a_per_verifier_auroc={},
        condition_b_per_verifier_auroc={},
        scorer_name="sklearn_tfidf_cosine",
        scorer_version="1.7.2",
        scorer_threshold=0.0,
        calibration_size=50,
        scorer_or_generator_model_path="/cache/model.gguf",
        candidate_label_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="n_questions"):
        summarize_evaluations([bad], n_questions=200)


def test_req_verify_2840_probe_preconditions_and_live_backend_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2840: default probes name all gates and backend absence blocks."""

    _minimal_repo(tmp_path)
    specs = model_specs_from_exp2836(
        mod.load_exp2836_preflight(tmp_path / "results" / mod.EXP2836_FILENAME)
    )

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
        elif "truthful_qa" in script:
            payload = {"available": True, "detail": "loaded truthful_qa generation, n=817"}
        elif "TfidfVectorizer" in script:
            payload = {
                "available": True,
                "detail": _local_scorer_detail(),
            }
        else:
            payload = {"available": False, "detail": "unexpected probe"}
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
        "truthfulqa_generation_split",
        "local_scorer",
        "fr11_state_files",
        "retired_exp2823",
    ]
    assert all(check.available for check in checks)

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_probe=lambda _config, _state_files, _model_specs: _all_checks(),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_live_truthfulqa_backend_unavailable"
    assert artifact["blocked_resources"] == ["live_backend"]
    assert artifact["retired_exp2823_not_used"] is True


def test_req_verify_2840_probe_failure_retirement_check_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2840: probe failures, retirement checks, and CLI remain explicit."""

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

    assert mod._retired_exp2823_check(tmp_path).available is False
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        "items:\n  - experiment_id: 2823\n    reason: active\n",
        encoding="utf-8",
    )
    assert mod._retired_exp2823_check(tmp_path).available is False
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        "items:\n  - experiment_id: 2823\n    reason: fabricated and retired\n",
        encoding="utf-8",
    )
    missing_legacy = mod._retired_exp2823_check(tmp_path)
    assert missing_legacy.available is False
    assert "legacy/fabricated" in missing_legacy.detail
    _write_retired_2823_evidence(tmp_path)
    assert mod._retired_exp2823_check(tmp_path).available is True

    no_scorer = mod._local_scorer_from_checks([])
    assert no_scorer["available"] is False
    assert "not checked" in str(no_scorer["detail"])
    invalid_scorer = mod._local_scorer_from_checks(
        [PreconditionCheck("local_scorer", False, "not-json")]
    )
    assert invalid_scorer["detail"] == "not-json"
    list_scorer = mod._local_scorer_from_checks(
        [PreconditionCheck("local_scorer", False, "[]")]
    )
    assert list_scorer["detail"] == "[]"

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
