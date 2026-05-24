"""Tests for Exp 3001 SOTA GGUF cache carry-forward checksum refresh.

Spec: REQ-INFER-SOTA-020,
      SCENARIO-INFER-SOTA-020-001,
      SCENARIO-INFER-SOTA-020-002
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "llm-ebm-inference" / "spec.md"
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


def _runner(*, torch_cuda: bool = True, llama_gpu: bool = True) -> exp.CommandRunner:
    def fake(
        command: list[str],
        *,
        timeout_s: int = 10,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del timeout_s, env
        if command[:1] == ["git"]:
            return _command(command, stdout="abc123def456\n")
        if command[0] == SELECTED_PYTHON and "import torch" in command[-1]:
            return _command(command, stdout=f"2.11.0+cu128 {torch_cuda} 2\n")
        if command[:1] == ["nvidia-smi"]:
            return _command(
                command,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 1024, 23552, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 5, 24122, 595.71.05\n"
                ),
            )
        if command[0] == SELECTED_PYTHON and "llama_supports_gpu_offload" in command[-1]:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "llama_cpp_import_ok": True,
                        "llama_cpp_origin": "/repo/.venv/lib/python3.14/site-packages/llama_cpp/__init__.py",
                        "llama_cpp_version": "0.3.23",
                        "llama_cpp_supports_gpu_offload": llama_gpu,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake


def _write_cached_model(tmp_path: Path, hf_id: str = QWEN) -> tuple[Path, dict[str, str]]:
    hub = tmp_path / "hf" / "hub"
    repo = hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "rev1"
    repo.mkdir(parents=True)
    filename = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
    gguf = repo / f"{filename}-Q4_K_M.gguf"
    gguf.write_text("tiny gguf fixture\n", encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def test_req_infer_sota_020_spec_anchor_exists() -> None:
    """REQ-INFER-SOTA-020: Exp 3001 is anchored in OpenSpec before implementation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-INFER-SOTA-020" in spec
    assert "SCENARIO-INFER-SOTA-020-001" in spec
    assert "SCENARIO-INFER-SOTA-020-002" in spec
    assert exp.ARTIFACT_FILENAME in spec


def test_scenario_020_001_fresh_headline_transcript_opens_gate(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-020-001: a fresh headline transcript opens .282 gate."""
    gguf, env = _write_cached_model(tmp_path)
    call_order: list[str] = []

    def command_runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
        call_order.append("precondition")
        return _runner()(command, **kwargs)

    def prompt_runner(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        call_order.append("prompt")
        return {
            "attempted": True,
            "load_status": "loaded",
            "generation_status": "generated",
            "usable": True,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "prompt": exp.DEFAULT_PROMPT,
            "response_text": "exp3001 cache refresh live",
            "tokens_generated": 4,
            "duration_seconds": 1.5,
            "inference_substrate": "llama_cpp_gpu",
        }

    artifact = exp.build_refresh_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=command_runner,
        prompt_runner_fn=prompt_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([10.0, 12.75]).__next__,
        tests_run=("focused-exp3001",),
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["artifact"] == exp.ARTIFACT_NAME
    assert artifact["sota_headline_ready"] is True
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["preconditions_checked"] is True
    assert artifact["model_specs"]["experiment_id"] == 3001
    assert artifact["model_specs"]["headline_models"] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["sota_models_available"] == [
        {"hf_id": QWEN, "path": str(gguf), "status": "cache_resolved"}
    ]
    assert artifact["cache_paths"]["headline_models"][QWEN] == str(gguf)
    assert artifact["model_checksums"][QWEN]["sha256"]
    assert artifact["precondition_evidence"]["checksum_feasibility"]["feasible"] is True
    assert artifact["sota_models_attempted"][0]["load_status"] == "loaded"
    assert artifact["sota_models_attempted"][0]["generation_status"] == "generated"
    assert artifact["sota_models_attempted"][0]["transcript_sha256"]
    assert artifact["live_transcript_paths"]
    transcript = Path(artifact["live_transcript_paths"][0])
    assert transcript.is_file()
    assert exp.ARTIFACT_NAME in str(transcript)
    assert json.loads(transcript.read_text(encoding="utf-8"))["response_text"] == (
        "exp3001 cache refresh live"
    )
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["duration_seconds"] == pytest.approx(2.75)
    assert artifact["tests_run"] == ["focused-exp3001"]
    assert call_order.index("precondition") < call_order.index("prompt")


def test_scenario_020_002_missing_cache_blocks_without_legacy_promotion(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-020-002: missing headline cache is terminally blocked."""
    artifact = exp.build_refresh_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([3.0, 3.4]).__next__,
    )

    assert artifact["sota_headline_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_model_cache")
    assert artifact["sota_models_available"] == []
    assert artifact["live_transcript_paths"] == []
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["legacy_smoke_context"]["smoke_only"] is False
    assert artifact["legacy_smoke_context"]["used_for_headline_readiness"] is False
    assert [row["hf_id"] for row in artifact["sota_models_attempted"]] == [QWEN, GEMMA31, GEMMA26]
    assert {row["cache_status"] for row in artifact["sota_models_attempted"]} == {"missing"}
    assert artifact["inference_substrate"] == "blocked_no_headline_cache"
    assert artifact["precondition_evidence"]["checksum_feasibility"]["feasible"] is False


def test_req_020_runtime_precondition_failure_skips_large_load(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-020: failed runtime preconditions are recorded before any load."""
    _gguf, env = _write_cached_model(tmp_path, GEMMA31)

    artifact = exp.build_refresh_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(llama_gpu=False),
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": GEMMA31}],
        monotonic=iter([7.0, 8.0]).__next__,
    )

    gemma_attempt = next(row for row in artifact["sota_models_attempted"] if row["hf_id"] == GEMMA31)
    assert artifact["sota_headline_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_runtime_preconditions")
    assert gemma_attempt["cache_status"] == "resolved"
    assert gemma_attempt["load_status"] == "not_attempted_runtime_precondition_failed"
    assert gemma_attempt["generation_status"] == "not_attempted"
    assert artifact["precondition_evidence"]["llama_cpp"]["llama_cpp_supports_gpu_offload"] is False


def test_req_020_helpers_prompt_parser_writer_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-020: helpers, writer, prompt parser, and CLI preserve the contract."""
    assert exp._checksum_feasibility(
        {
            QWEN: {"status": "available", "sha256": "a"},
            GEMMA31: {"status": "available", "bounded_sha256": "b"},
            GEMMA26: {"status": "missing"},
        }
    ) == {
        "model_count": 3,
        "available_model_count": 2,
        "available_models": [QWEN, GEMMA31],
        "full_sha256_model_count": 1,
        "bounded_sha256_model_count": 1,
        "feasible": True,
        "method": "sha256_full_for_small_files_or_bounded_head_tail_for_large_files",
    }

    gguf, env = _write_cached_model(tmp_path)
    parsed = exp._run_bounded_headline_prompt(
        {"hf_id": QWEN, "path": str(gguf), "gpu": 2},
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(
            command,
            stdout=json.dumps(
                {
                    "attempted": True,
                    "load_status": "loaded",
                    "generation_status": "generated",
                    "usable": True,
                    "gpu_backed": True,
                    "hf_id": QWEN,
                    "model_path": str(gguf),
                    "prompt": exp.DEFAULT_PROMPT,
                    "response_text": "ok",
                    "tokens_generated": 1,
                    "duration_seconds": 0.5,
                    "inference_substrate": "llama_cpp_gpu",
                    "main_gpu": 0,
                    "requested_gpu": 2,
                },
                sort_keys=True,
            )
            + "\n",
        ),
        env={},
        timeout_s=2,
    )
    assert parsed["usable"] is True
    assert parsed["requested_gpu"] == 2

    failed = exp._run_bounded_headline_prompt(
        {"hf_id": QWEN, "path": str(gguf), "gpu": 0},
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, returncode=1, stderr="load failed"),
        env={},
        timeout_s=2,
    )
    assert failed["generation_status"] == "failed"
    assert failed["blocker"] == "load failed"

    output = tmp_path / "results" / exp.ARTIFACT_FILENAME
    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        prompt_runner_fn=lambda model, **_: {
            "attempted": True,
            "load_status": "loaded",
            "generation_status": "generated",
            "usable": True,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "prompt": exp.DEFAULT_PROMPT,
            "response_text": "ok",
            "tokens_generated": 1,
            "duration_seconds": 0.5,
            "inference_substrate": "llama_cpp_gpu",
        },
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([1.0, 1.2]).__next__,
        tests_run=("coverage",),
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact"] == exp.ARTIFACT_NAME

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(exp, "run_experiment", fake_run_experiment)
    assert exp.main(["--output", str(tmp_path / "out.json"), "--selected-python", SELECTED_PYTHON, "--test-run", "unit"]) == 0
    assert exp.main(["--output", str(tmp_path / "out2.json"), "--selected-python", SELECTED_PYTHON, "--prompt-timeout-s", "7"]) == 0
    assert calls == [
        {
            "output_path": tmp_path / "out.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": ["unit"],
        },
        {
            "output_path": tmp_path / "out2.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": [],
            "prompt_timeout_s": 7,
        },
    ]
