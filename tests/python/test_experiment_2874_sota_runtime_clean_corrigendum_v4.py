"""Tests for Exp 2874 SOTA runtime clean corrigendum v4.

Spec: REQ-INFER-SOTA-015,
      SCENARIO-INFER-SOTA-015-001,
      SCENARIO-INFER-SOTA-015-002,
      SCENARIO-INFER-SOTA-015-003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_runtime_clean_corrigendum_v4 as mod
from carnot.reporting.sota_runtime_clean_corrigendum_v4 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_corrigendum_artifact,
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


def _runner(*, llama_gpu: bool = True, torch_cuda: bool = True) -> mod.CommandRunner:
    def fake(
        command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None
    ) -> dict[str, Any]:
        del timeout_s, env
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
                    }
                )
                + "\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake


def _write_cached_model(tmp_path: Path, hf_id: str = GEMMA26) -> tuple[Path, dict[str, str]]:
    hub = tmp_path / "hf" / "hub"
    repo = hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "rev1"
    repo.mkdir(parents=True)
    filename = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
    gguf = repo / f"{filename}-UD-Q4_K_M.gguf"
    gguf.write_text("tiny gguf fixture", encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def _clean_suite(model: dict[str, Any], prompts: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    return {
        "attempted": True,
        "returncode": 0,
        "prompt_suite": [
            {
                "prompt_id": prompts[0]["prompt_id"],
                "prompt_text": prompts[0]["prompt_text"],
                "max_tokens": prompts[0]["max_tokens"],
                "response_text": "4",
                "tokens_generated": 2,
                "duration_s": 61.25,
                "tokens_per_second": 0.032653,
                "usable": True,
                "nonempty": True,
                "gpu_backed": True,
                "gpu_memory_before": [{"index": 0, "memory_used_mib": 1024}],
                "gpu_memory_after": [{"index": 0, "memory_used_mib": 9000}],
                "seed": mod.RANDOM_SEED,
            }
        ],
        "gpu_memory_evidence": {
            "before_load": [{"index": 0, "memory_used_mib": 1024}],
            "after_load": [{"index": 0, "memory_used_mib": 8500}],
            "after_prompt_suite": [{"index": 0, "memory_used_mib": 9000}],
            "after_close": [{"index": 0, "memory_used_mib": 1024}],
        },
        "stdout_summary": "",
        "stderr_summary": "",
        "model_path": model["path"],
    }


def test_exp2874_single_mandated_gpu_response_opens_v4_gate(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-015 / SCENARIO-INFER-SOTA-015-001: one clean GPU response opens v4."""
    gguf, env = _write_cached_model(tmp_path)
    suite_calls: list[str] = []

    def prompt_suite_runner(model: dict[str, Any], prompts: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        suite_calls.append(model["hf_id"])
        return _clean_suite(model, prompts, **kwargs)

    artifact = build_corrigendum_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_suite_runner_fn=prompt_suite_runner,
        monotonic=iter([10.0, 72.5]).__next__,
        tests_run=["pytest tests/python/test_experiment_2874_sota_runtime_clean_corrigendum_v4.py"],
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["sota_runtime_clean"] is True
    assert artifact["sota_runtime_ready_v4"] is True
    assert artifact["selected_model_hf_id"] == GEMMA26
    assert artifact["selected_model_path"] == str(gguf)
    assert "size_bytes=" in artifact["selected_model_checksum_or_fingerprint"]
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["llama_cpp_gpu_offload_verified"] is True
    assert artifact["usable_response_count"] == 1
    assert artifact["nonempty_response_count"] == 1
    assert artifact["total_tokens_generated"] == 2
    assert artifact["tokens_per_second"] == pytest.approx(2 / 61.25, abs=1e-6)
    assert artifact["legacy_small_models_used_only_for_smoke"] is True
    assert artifact["tests_run"] == [
        "pytest tests/python/test_experiment_2874_sota_runtime_clean_corrigendum_v4.py"
    ]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(62.5)
    assert artifact["prompt_suite"][0]["prompt_text"] == mod.DEFAULT_PROMPT_SUITE[0]["prompt_text"]
    assert artifact["gpu_memory_evidence"]["after_prompt_suite"][0]["memory_used_mib"] == 9000
    assert suite_calls == [GEMMA26]
    assert {row["resource"] for row in artifact["preconditions_checked"]} >= {
        "venv_torch_cuda",
        "nvidia_smi_inventory",
        "llama_cpp_gpu_offload",
        "cached_sota_pair",
        "local_cache_resolution",
    }


def test_exp2874_blocks_when_llama_cpp_gpu_offload_missing(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-015 / SCENARIO-INFER-SOTA-015-002: CPU-only llama.cpp blocks v4."""
    _gguf, env = _write_cached_model(tmp_path)

    artifact = build_corrigendum_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(llama_gpu=False),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_suite_runner_fn=lambda model, prompts, **_: pytest.fail(
            f"unexpected prompt suite: {model}, {prompts}"
        ),
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert artifact["sota_runtime_clean"] is False
    assert artifact["sota_runtime_ready_v4"] is False
    assert artifact["honest_verdict"].startswith("blocked_llama_cpp_gpu_offload")
    assert artifact["llama_cpp_gpu_offload_verified"] is False
    assert artifact["prompt_suite"] == []
    assert artifact["legacy_small_models_used_only_for_smoke"] is True


def test_exp2874_pair_readiness_separate_from_single_model_runtime(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-015-003: one model may be clean while pair readiness is false."""
    _gguf, env = _write_cached_model(tmp_path)

    artifact = build_corrigendum_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": GEMMA26}],
        prompt_suite_runner_fn=_clean_suite,
        monotonic=iter([20.0, 82.0]).__next__,
    )

    assert artifact["sota_runtime_clean"] is True
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["models_missing_from_cache"] == [QWEN, GEMMA31]


def test_exp2874_prompt_suite_helper_parses_success_and_failure(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-015: prompt helper preserves exact prompt and runtime evidence."""
    model = {"hf_id": GEMMA26, "path": str(tmp_path / "model.gguf"), "gpu": 0}
    prompts = [{"prompt_id": "p1", "prompt_text": "Say ok.", "max_tokens": 8}]

    success = mod._run_prompt_suite(
        model,
        prompts=prompts,
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(
            command,
            stdout=json.dumps(
                {
                    "attempted": True,
                    "prompt_suite": [
                        {
                            "prompt_id": "p1",
                            "prompt_text": "Say ok.",
                            "max_tokens": 8,
                            "response_text": "ok",
                            "tokens_generated": 1,
                            "duration_s": 2.0,
                            "tokens_per_second": 0.5,
                            "usable": True,
                            "nonempty": True,
                            "gpu_backed": True,
                            "gpu_memory_before": [],
                            "gpu_memory_after": [],
                            "seed": mod.RANDOM_SEED,
                        }
                    ],
                    "gpu_memory_evidence": {"before_load": [], "after_close": []},
                }
            )
            + "\n",
        ),
        env={},
    )
    assert success["prompt_suite"][0]["response_text"] == "ok"
    assert success["command"][0] == SELECTED_PYTHON

    failed = mod._run_prompt_suite(
        model,
        prompts=prompts,
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, returncode=1, stderr="load failed"),
        env={},
    )
    assert failed["attempted"] is True
    assert failed["prompt_suite"] == []
    assert failed["blocker"] == "load failed"


def test_exp2874_helper_edges_and_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-INFER-SOTA-015: helper edges and CLI keep the v4 artifact contract stable."""
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    assert mod._repo_root() == tmp_path.resolve()
    assert mod._selected_python(tmp_path) == sys.executable
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._model_fingerprint(tmp_path / "missing.gguf").startswith("missing:")
    blob = tmp_path / ("a" * 64)
    blob.write_text("model bytes", encoding="utf-8")
    assert mod._model_fingerprint(blob).startswith("sha256:")
    assert mod._honest_verdict(
        clean=False, torch_cuda=False, llama_gpu=True, cached_count=1, attempted=False
    ).startswith("blocked_cuda")
    assert mod._honest_verdict(
        clean=False, torch_cuda=True, llama_gpu=True, cached_count=0, attempted=False
    ).startswith("blocked_model_cache")
    assert mod._honest_verdict(
        clean=False, torch_cuda=True, llama_gpu=True, cached_count=1, attempted=True
    ).startswith("blocked_prompt_suite")
    assert mod._honest_verdict(
        clean=False, torch_cuda=True, llama_gpu=True, cached_count=1, attempted=False
    ).startswith("blocked_preconditions")

    output = tmp_path / "results" / "experiment_2874.json"
    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260522",
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_suite_runner_fn=lambda model, prompts, **_: pytest.fail(
            f"unexpected prompt suite: {model}, {prompts}"
        ),
        monotonic=iter([5.0, 5.25]).__next__,
        tests_run=["unit", "coverage"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact"] == "experiment_2874_sota_runtime_clean_corrigendum_v4"

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--run-date",
                "20260522",
                "--output",
                str(tmp_path / "out.json"),
                "--selected-python",
                SELECTED_PYTHON,
                "--test-run",
                "unit",
                "--test-run",
                "coverage",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "run_date": "20260522",
            "output_path": tmp_path / "out.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": ["unit", "coverage"],
        }
    ]
