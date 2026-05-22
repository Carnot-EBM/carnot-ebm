"""Tests for Exp 2862 SOTA runtime cache/offload resolver v3.

Spec: REQ-INFER-SOTA-013,
      SCENARIO-INFER-SOTA-013-001,
      SCENARIO-INFER-SOTA-013-002,
      SCENARIO-INFER-SOTA-013-003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_runtime_cache_offload_resolver_v3 as mod
from carnot.reporting.sota_runtime_cache_offload_resolver_v3 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_runtime_resolver_artifact,
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


def _write_cached_gemma(tmp_path: Path, hf_id: str = GEMMA26) -> tuple[Path, dict[str, str]]:
    hub = tmp_path / "hf" / "hub"
    repo = hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "rev1"
    repo.mkdir(parents=True)
    filename = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
    gguf = repo / f"{filename}-UD-Q4_K_M.gguf"
    gguf.write_text("tiny gguf fixture", encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def test_exp2862_single_mandated_gpu_response_opens_v3_gate(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-013 / SCENARIO-INFER-SOTA-013-001: one GPU response opens v3."""
    gguf, env = _write_cached_gemma(tmp_path)
    smoke_calls: list[str] = []

    def prompt_runner(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        smoke_calls.append(model["hf_id"])
        return {
            "attempted": True,
            "usable": True,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "response_text": "4",
            "tokens_generated": 7,
            "tokens_per_second": 3.5,
            "duration_s": 2.0,
            "gpu_memory": {"before": [], "during": [{"index": 0, "memory_used_mib": 18000}]},
        }

    artifact = build_runtime_resolver_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_runner_fn=prompt_runner,
        monotonic=iter([10.0, 13.0]).__next__,
        tests_run=["unit coverage command"],
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["sota_runtime_ready_v3"] is True
    assert artifact["selected_model_hf_id"] == GEMMA26
    assert artifact["selected_model_path"] == str(gguf)
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["llama_cpp_gpu_offload_verified"] is True
    assert artifact["usable_response_count"] == 1
    assert artifact["total_tokens_generated"] == 7
    assert artifact["tokens_per_second"] == pytest.approx(3.5)
    assert artifact["legacy_small_models_used_only_for_smoke"] is True
    assert artifact["tests_run"] == ["unit coverage command"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert {row["resource"] for row in artifact["preconditions_checked"]} >= {
        "venv_torch_cuda",
        "nvidia_smi_inventory",
        "llama_cpp_gpu_offload",
        "cached_sota_pair",
        "mandated_sota_gguf_cache",
    }
    assert artifact["cache_inventory"][2]["cache_status"] == "resolved"
    assert smoke_calls == [GEMMA26]
    assert artifact["duration_s"] == pytest.approx(3.0)


def test_exp2862_cpu_only_llama_cpp_blocks_and_documents_reinstall(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-013-002: CPU-only llama.cpp cannot open v3 readiness."""
    _gguf, env = _write_cached_gemma(tmp_path)

    artifact = build_runtime_resolver_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(llama_gpu=False),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert artifact["sota_runtime_ready_v3"] is False
    assert artifact["honest_verdict"].startswith("blocked_llama_cpp_gpu_offload")
    assert artifact["llama_cpp_gpu_offload_verified"] is False
    assert artifact["usable_response_count"] == 0
    assert 'CMAKE_ARGS="-DGGML_CUDA=on"' in artifact["llama_cpp_cuda_reinstall_command"]


def test_exp2862_missing_pair_and_cache_are_recorded_separately(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-013-003: missing mandated cache keeps pair false without legacy."""
    artifact = build_runtime_resolver_artifact(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": GEMMA26}],
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        monotonic=iter([2.0, 2.1]).__next__,
    )

    assert artifact["sota_runtime_ready_v3"] is False
    assert artifact["honest_verdict"].startswith("blocked_model_cache")
    assert artifact["selected_model_hf_id"] == ""
    assert artifact["selected_model_path"] == ""
    assert artifact["models_missing_from_cache"] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["legacy_small_models_used_only_for_smoke"] is True


def test_exp2862_cache_inventory_handles_project_models_and_zero_byte_markers(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-013: cache inspection reports exact paths and missing status."""
    hub = tmp_path / "hf" / "hub"
    qwen_missing = (
        hub
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / ".no_exist"
        / "rev"
        / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    qwen_missing.parent.mkdir(parents=True)
    qwen_missing.write_text("", encoding="utf-8")
    project_dir = tmp_path / "models" / "gemma-4-31B-it-GGUF" / "nested"
    project_dir.mkdir(parents=True)
    (project_dir / "mmproj-F16.gguf").write_text("ignore", encoding="utf-8")
    local = project_dir / "gemma-4-31B-it-Q4_K_M.gguf"
    local.write_text("local model", encoding="utf-8")

    rows = mod._inspect_mandated_cache(
        tmp_path,
        {"HUGGINGFACE_HUB_CACHE": str(hub)},
    )

    by_id = {row["hf_id"]: row for row in rows}
    assert by_id[QWEN]["cache_status"] == "missing"
    assert by_id[QWEN]["zero_byte_marker_count"] == 1
    assert by_id[GEMMA31]["cache_status"] == "resolved"
    assert by_id[GEMMA31]["path"] == str(local)
    assert by_id[GEMMA31]["project_candidate_count"] == 1
    assert by_id[GEMMA26]["cache_status"] == "missing"


def test_exp2862_command_and_prompt_helpers_parse_success_and_failure(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-013: subprocess helpers preserve honest success/failure evidence."""
    assert mod._repo_root().name == "carnot"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._selected_python(tmp_path / "missing") == sys.executable

    ok = mod._run_command(["printf", "ok"], timeout_s=5)
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    failed = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert failed["returncode"] is None
    assert "FileNotFoundError" in failed["stderr_summary"]

    model = {"hf_id": GEMMA26, "path": str(tmp_path / "model.gguf"), "gpu": 0}
    success = mod._run_bounded_prompt(
        model,
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(
            command,
            stdout=json.dumps(
                {
                    "attempted": True,
                    "usable": True,
                    "gpu_backed": True,
                    "hf_id": GEMMA26,
                    "model_path": model["path"],
                    "response_text": "ok",
                    "tokens_generated": 3,
                    "tokens_per_second": 1.5,
                    "duration_s": 2.0,
                    "gpu_memory": {"before": [], "during": []},
                }
            )
            + "\n",
        ),
        env={},
    )
    assert success["usable"] is True
    assert success["tokens_generated"] == 3

    failed_prompt = mod._run_bounded_prompt(
        model,
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, returncode=1, stderr="load failed"),
        env={},
    )
    assert failed_prompt["usable"] is False
    assert failed_prompt["blocker"] == "load failed"

    bad_llama = mod._llama_cpp_probe(
        SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, stdout="", stderr="bad llama"),
        env={},
    )
    assert bad_llama["llama_cpp_import_ok"] is False
    assert bad_llama["error"] == "bad llama"
    assert mod._cache_roots(tmp_path, {"HF_HOME": str(tmp_path / "hfhome")})[
        "huggingface_hub_cache"
    ].endswith("hfhome/hub")
    assert ".cache/huggingface/hub" in mod._cache_roots(tmp_path, {})["huggingface_hub_cache"]

    broken = tmp_path / "broken.gguf"
    broken.symlink_to(tmp_path / "absent.gguf")
    broken_record = mod._candidate_record(broken, GEMMA26, "project_models")
    assert broken_record["exists"] is False
    assert broken_record["size_bytes"] == 0
    assert mod._quantization_suffix("gemma-4-26B-A4B-it-random.gguf") == "unknown"
    fallback = mod._select_candidate(
        [{"path": "gemma-4-26B-A4B-it-random.gguf", "usable_candidate": True, "size_bytes": 1}]
    )
    assert fallback is not None
    assert fallback["path"] == "gemma-4-26B-A4B-it-random.gguf"
    assert (
        mod._exercise_cached_sota_pair(
            lambda **_: (_ for _ in ()).throw(RuntimeError("pair exploded"))
        )["error"]
        == "RuntimeError: pair exploded"
    )
    assert mod._honest_verdict(
        ready=False,
        torch_cuda=False,
        llama_gpu=True,
        cached_count=1,
        prompt_attempted=False,
    ).startswith("blocked_cuda")
    assert mod._honest_verdict(
        ready=False,
        torch_cuda=True,
        llama_gpu=True,
        cached_count=1,
        prompt_attempted=True,
    ).startswith("blocked_prompt_smoke")
    assert mod._honest_verdict(
        ready=False,
        torch_cuda=True,
        llama_gpu=True,
        cached_count=1,
        prompt_attempted=False,
    ).startswith("blocked_preconditions")


def test_exp2862_run_experiment_writes_json_and_main_delegates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-013: writer and CLI keep the v3 artifact contract stable."""
    output = tmp_path / "results" / "experiment_2862.json"
    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260522",
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        monotonic=iter([5.0, 5.2]).__next__,
        tests_run=[
            "pytest tests/python/test_experiment_2862_sota_runtime_cache_offload_resolver_v3.py"
        ],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact"] == "experiment_2862_sota_runtime_cache_offload_resolver_v3"

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
