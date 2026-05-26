"""Tests for Exp 3123 SOTA cache/preconditions manifest v2.

Spec: REQ-INFER-SOTA-023,
      SCENARIO-INFER-SOTA-023-001,
      SCENARIO-INFER-SOTA-023-002,
      SCENARIO-INFER-SOTA-023-003
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_cache_preconditions_manifest_v2 as mod
from carnot.reporting.sota_cache_preconditions_manifest_v2 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_sota_cache_manifest_v2,
    run_experiment,
)


SELECTED_PYTHON = "/repo/.venv/bin/python"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _command(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def _runner(*, cuda: bool = True, nvidia_smi: bool = True) -> mod.CommandRunner:
    def fake(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None) -> dict[str, Any]:
        del timeout_s, env
        if command[0] == SELECTED_PYTHON and "import torch" in command[-1]:
            return _command(command, stdout=f"2.11.0+cu128 {cuda} {2 if cuda else 0}\n")
        if command[:1] == ["nvidia-smi"] and nvidia_smi:
            return _command(
                command,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 1024, 23552, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 5, 24122, 595.71.05\n"
                ),
            )
        if command[:1] == ["nvidia-smi"]:
            return _command(command, returncode=1, stderr="nvidia-smi missing\n")
        raise AssertionError(f"unexpected command: {command}")

    return fake


def _write_source_artifacts(project_root: Path) -> list[Path]:
    paths = [
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("scripts/experiment_template.py"),
        Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"),
        Path("results/experiment_3120_cross_corpus_matrix_v24.json"),
        Path("results/experiment_3121_capstone_v290.json"),
    ]
    for index, relpath in enumerate(paths):
        target = project_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if relpath.suffix == ".json":
            payload: dict[str, Any] = {"fixture": index}
            if "3110" in relpath.name:
                payload.update(
                    {
                        "present_model_ids": [GEMMA26],
                        "missing_model_ids": [QWEN, GEMMA31],
                        "cached_sota_pair_available": False,
                    }
                )
            target.write_text(json.dumps(payload), encoding="utf-8")
        else:
            target.write_text(f"fixture {index}\n", encoding="utf-8")
    return paths


def _write_cached_model(project_root: Path, hf_id: str, content: str = "tiny gguf fixture") -> Path:
    hub = project_root / "hf" / "hub"
    snapshot = hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    filename = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
    gguf = snapshot / f"{filename}-UD-Q4_K_M.gguf"
    gguf.write_text(content, encoding="utf-8")
    return gguf


def _write_zero_byte_marker(project_root: Path, hf_id: str) -> Path:
    marker = (
        project_root
        / "hf"
        / "hub"
        / f"models--{hf_id.replace('/', '--')}"
        / ".no_exist"
        / "rev"
        / f"{hf_id.split('/', 1)[-1].removesuffix('-GGUF')}-Q4_K_M.gguf"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    return marker


def test_exp3123_single_cached_model_opens_only_single_model_attempts(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-023 / SCENARIO-INFER-SOTA-023-001: one model is not a pair."""
    source_paths = _write_source_artifacts(tmp_path)
    gguf = _write_cached_model(tmp_path, GEMMA26)
    _write_zero_byte_marker(tmp_path, QWEN)
    legacy = tmp_path / "models" / "gemma-4-E4B-it-GGUF" / "gemma-4-E4B-it-Q4_K_M.gguf"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy smoke fixture", encoding="utf-8")

    artifact = build_sota_cache_manifest_v2(
        project_root=tmp_path,
        run_date="20260526",
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf" / "hub")},
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        source_paths=source_paths,
        monotonic=iter([10.0, 10.25]).__next__,
        tests_run=["REQ-INFER-SOTA-023 unit"],
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["sota_cache_manifest_v2_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["mandatory_headline_model_ids"] == list(mod.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["present_model_ids"] == [GEMMA26]
    assert artifact["missing_model_ids"] == [QWEN, GEMMA31]
    assert artifact["selected_headline_model_ids"] == [GEMMA26]
    assert artifact["cached_sota_pair_available"] is False
    assert artifact["any_single_sota_available"] is True
    assert artifact["headline_claim_allowed"] is True
    assert artifact["gpu_preflight"]["cuda_available"] is True
    assert artifact["gpu_preflight"]["gpu_count"] == 2
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["downstream_usage"]["live_llm_headline_tasks"]["required_action"] == (
        "attempt_at_least_one_present_mandated_model_or_write_blocked_diagnostic_artifact"
    )
    assert artifact["downstream_usage"]["pair_or_comparative_headline_tasks"]["headline_claim_allowed"] is False
    assert artifact["downstream_usage"]["solver_only_tasks"]["allowed_without_model_availability"] is True
    assert artifact["smoke_test_model_ids"] == list(mod.SMOKE_TEST_MODEL_IDS)
    assert GEMMA26 in artifact["selected_headline_model_ids"]
    assert "gemma-4-E4B" not in json.dumps(artifact["present_model_ids"])

    by_id = {row["hf_id"]: row for row in artifact["cache_inventory"]}
    assert by_id[GEMMA26]["cache_status"] == "resolved"
    assert by_id[GEMMA26]["path"] == str(gguf)
    assert by_id[QWEN]["cache_status"] == "missing"
    assert by_id[QWEN]["zero_byte_marker_count"] == 1
    assert by_id[GEMMA31]["cache_status"] == "missing"
    assert all(row["present"] and row["sha256"] for row in artifact["source_artifacts"])
    assert artifact["prior_artifact_model_id_observations"] == [
        {
            "path": "results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json",
            "present_model_ids": [GEMMA26],
            "missing_model_ids": [QWEN, GEMMA31],
            "selected_headline_model_ids": [],
            "cached_sota_pair_available": False,
            "headline_claim_allowed": None,
        }
    ]
    assert artifact["duration_s"] == pytest.approx(0.25)
    assert artifact["tests_run"] == ["REQ-INFER-SOTA-023 unit"]


def test_exp3123_no_cached_models_blocks_live_claims_but_not_solver_tasks(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-023-002: missing cache blocks live claims only."""
    source_paths = _write_source_artifacts(tmp_path)

    artifact = build_sota_cache_manifest_v2(
        project_root=tmp_path,
        run_date="20260526",
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(cuda=False, nvidia_smi=False),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        source_paths=source_paths,
        monotonic=iter([1.0, 1.1]).__next__,
    )

    assert artifact["honest_verdict"].startswith("blocked_model_cache")
    assert artifact["sota_cache_manifest_v2_ready"] is True
    assert artifact["present_model_ids"] == []
    assert artifact["missing_model_ids"] == list(mod.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["selected_headline_model_ids"] == []
    assert artifact["any_single_sota_available"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["gpu_preflight"]["cuda_available"] is False
    assert artifact["gpu_preflight"]["nvidia_smi_available"] is False
    assert artifact["downstream_usage"]["live_llm_headline_tasks"]["headline_claim_allowed"] is False
    assert artifact["downstream_usage"]["live_llm_headline_tasks"]["when_no_present_mandated_model"] == (
        "write_blocked_model_cache_or_diagnostic_artifact_before_verifier_repair_self_learning_or_energy_sidecar"
    )
    assert artifact["downstream_usage"]["solver_only_tasks"]["allowed_without_model_availability"] is True
    assert artifact["downstream_usage"]["legacy_small_models"]["headline_claim_allowed"] is False


def test_exp3123_cached_pair_selects_pair_for_comparative_headlines(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-023: pair availability is separated from single-model readiness."""
    source_paths = _write_source_artifacts(tmp_path)
    qwen_path = _write_cached_model(tmp_path, QWEN, "qwen fixture")
    gemma31_path = _write_cached_model(tmp_path, GEMMA31, "gemma31 fixture")

    def cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
        assert gpu_indices == (0, 1)
        assert preferred_quant == "Q4_K_M"
        return [
            {"name": "Qwen", "hf_id": QWEN, "gpu": 0, "model_path": str(qwen_path)},
            {"name": "Gemma31", "hf_id": GEMMA31, "gpu": 1, "model_path": str(gemma31_path)},
        ]

    artifact = build_sota_cache_manifest_v2(
        project_root=tmp_path,
        run_date="20260526",
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf" / "hub")},
        command_runner=_runner(),
        cached_pair_fn=cached_pair,
        source_paths=source_paths,
        monotonic=iter([2.0, 2.5]).__next__,
    )

    assert artifact["cached_sota_pair_available"] is True
    assert artifact["any_single_sota_available"] is True
    assert artifact["present_model_ids"] == [QWEN, GEMMA31]
    assert artifact["missing_model_ids"] == [GEMMA26]
    assert artifact["selected_headline_model_ids"] == [QWEN, GEMMA31]
    assert artifact["downstream_usage"]["pair_or_comparative_headline_tasks"]["headline_claim_allowed"] is True
    assert artifact["downstream_usage"]["live_llm_headline_tasks"]["allowed_model_ids"] == [QWEN, GEMMA31]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["reproducibility_checksum"]) == 64


def test_exp3123_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-023: run_experiment persists the manifest contract."""
    source_paths = _write_source_artifacts(tmp_path)
    output = tmp_path / "results" / "experiment_3123_sota_cache_preconditions_manifest_v2.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260526",
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        source_paths=source_paths,
        monotonic=iter([3.0, 3.2]).__next__,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact"] == "experiment_3123_sota_cache_preconditions_manifest_v2"
    assert artifact["schema"] == "carnot.sota_cache_preconditions_manifest.v2"
