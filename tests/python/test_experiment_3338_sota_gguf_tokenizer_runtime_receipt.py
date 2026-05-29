"""Tests for Exp 3338 SOTA GGUF tokenizer/runtime receipt.

Spec refs: REQ-INFER-SOTA-3338,
SCENARIO-INFER-SOTA-3338-001, SCENARIO-INFER-SOTA-3338-002.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_gguf_tokenizer_runtime_receipt_3338 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "files_updated",
    "model_specs",
    "cache_status",
    "tokenizer_status",
    "loader_status",
    "gpu_status",
    "smoke_generation_status",
    "runtime_receipt_clean",
    "blocked_reasons",
}


def _write_model(cache_root: Path, hf_id: str, filename: str, content: bytes = b"gguf") -> Path:
    owner, name = hf_id.split("/", 1)
    path = cache_root / f"models--{owner}--{name}" / "snapshots" / "rev1" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _command(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {"command": command, "returncode": returncode, "stdout": stdout, "stderr": stderr}


def _worker_payload(
    model_id: str,
    *,
    ok: bool = True,
    output: str = "READY",
    error: str = "",
    tokens: int = 1,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "model_id": model_id,
        "load_status": "loaded" if ok else "failed",
        "tokenize_status": "tokenized" if ok else "not_attempted",
        "generation_status": "generated" if ok else "failed",
        "prompt_token_count": 6 if ok else 0,
        "output_text": output if ok else "",
        "tokens_generated": tokens if ok else 0,
        "error": error,
        "duration_s": 1.25,
        "usage": {"prompt_tokens": 6, "completion_tokens": tokens, "total_tokens": 6 + tokens},
    }


def _runner(
    worker_payloads: dict[str, dict[str, Any]] | None = None,
    *,
    loader_import_ok: bool = True,
    torch_cuda: bool = True,
) -> tuple[mod.CommandRunner, list[list[str]]]:
    calls: list[list[str]] = []
    worker_payloads = worker_payloads or {}

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        calls.append(command)
        joined = "\n".join(command)
        if command and command[0] == "nvidia-smi":
            return _command(
                command,
                stdout="0, NVIDIA RTX 3090, 24576, 1024, 23552, 595.71.05\n",
            )
        if "exp3338_tokenizer_dependency_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "dependencies": {
                            "sentencepiece": {"available": True, "version": "0.2.1"},
                            "tiktoken": {"available": True, "version": "0.12.0"},
                            "tokenizers": {"available": True, "version": "0.21.0"},
                        }
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if "exp3338_llama_cpp_loader_probe" in joined:
            return _command(
                command,
                returncode=0 if loader_import_ok else 1,
                stdout=json.dumps(
                    {
                        "llama_cpp_import_ok": loader_import_ok,
                        "llama_cpp_version": "0.3.23" if loader_import_ok else None,
                        "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py"
                        if loader_import_ok
                        else None,
                        "llama_cpp_supports_gpu_offload": loader_import_ok,
                        "loader_error": "" if loader_import_ok else "ModuleNotFoundError",
                    },
                    sort_keys=True,
                )
                + "\n",
                stderr="" if loader_import_ok else "No module named llama_cpp\n",
            )
        if "exp3338_torch_cuda_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "torch_import_ok": True,
                        "torch_version": "2.11.0+cu128",
                        "cuda_available": torch_cuda,
                        "device_count": 2 if torch_cuda else 0,
                        "cuda_version": "12.8" if torch_cuda else None,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if "--exp3338-runtime-worker" in command:
            model_id = command[command.index("--model-id") + 1]
            payload = worker_payloads.get(model_id, _worker_payload(model_id))
            return _command(
                command,
                returncode=0 if payload["ok"] else 1,
                stdout=json.dumps(payload, sort_keys=True) + "\n",
                stderr="llama_model_load: ok\n" if payload["ok"] else payload["error"] + "\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return run, calls


def test_req_infer_sota_3338_spec_anchor_exists() -> None:
    """REQ-INFER-SOTA-3338: OpenSpec declares the runtime receipt before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-INFER-SOTA-3338" in spec
    assert "SCENARIO-INFER-SOTA-3338-001" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "runtime_receipt_clean" in spec
    assert Path(mod.__file__).exists()


def test_scenario_infer_sota_3338_one_mandated_success_opens_receipt(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-3338-001: one clean mandated smoke opens the gate."""

    cache_root = tmp_path / "hf-cache"
    qwen_path = _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", b"qwen")
    gemma_path = _write_model(cache_root, GEMMA31, "gemma-4-31B-it-UD-Q4_K_M.gguf", b"gemma31")
    runner, calls = _runner(
        {
            QWEN: _worker_payload(QWEN, ok=True, output="READY", tokens=1),
            GEMMA31: _worker_payload(
                GEMMA31,
                ok=False,
                error="ValueError: tokenizer rejected fixture",
            ),
        }
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        cached_pair_resolver=lambda gpu_indices=(0, 1): [
            {"hf_id": QWEN, "name": "Qwen", "gpu": gpu_indices[0], "model_path": str(qwen_path)},
            {
                "hf_id": GEMMA31,
                "name": "Gemma31",
                "gpu": gpu_indices[1],
                "model_path": str(gemma_path),
            },
        ],
        monotonic=iter([10.0, 15.5]).__next__,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3338"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["runtime_receipt_clean"] is True
    assert artifact["blocked_reasons"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(5.5)

    assert [row["hf_id"] for row in artifact["model_specs"]] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["model_specs"][0]["model_path"] == str(qwen_path)
    assert artifact["model_specs"][0]["gpu"] == 0
    assert artifact["model_specs"][1]["gpu"] == 1
    assert artifact["cache_status"]["cached_sota_pair_returned_two_loadable_specs"] is True
    assert artifact["cache_status"]["mandated_models"][QWEN]["file_evidence"]["sha256"] == mod.hash_bytes(
        b"qwen"
    )
    assert artifact["cache_status"]["mandated_models"][GEMMA31]["file_evidence"]["sha256"] == mod.hash_bytes(
        b"gemma31"
    )
    assert artifact["cache_status"]["mandated_models"][GEMMA26]["cached"] is False

    per_model = artifact["smoke_generation_status"]["per_model"]
    assert per_model[QWEN]["generation_status"] == "generated"
    assert per_model[QWEN]["tokens_generated"] == 1
    assert per_model[GEMMA31]["generation_status"] == "failed"
    assert "ValueError: tokenizer rejected fixture" in per_model[GEMMA31]["exception"]
    assert artifact["smoke_generation_status"]["legacy_cpu_loader_controls"] == []
    worker_calls = [call for call in calls if "--exp3338-runtime-worker" in call]
    assert [call[call.index("--model-id") + 1] for call in worker_calls] == [QWEN, GEMMA31]


def test_scenario_infer_sota_3338_all_runtime_failures_block_precisely(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-3338-002: failed mandated smokes produce a blocker."""

    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", b"qwen")
    _write_model(cache_root, GEMMA26, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", b"gemma26")
    qwen_error = "ValueError: Couldn't instantiate the backend tokenizer"
    gemma_error = "RuntimeError: llama.cpp load failed"
    runner, calls = _runner(
        {
            QWEN: _worker_payload(QWEN, ok=False, error=qwen_error),
            GEMMA26: _worker_payload(GEMMA26, ok=False, error=gemma_error),
        }
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        cached_pair_resolver=lambda gpu_indices=(0, 1): None,
        monotonic=iter([1.0, 3.0]).__next__,
    )

    assert artifact["runtime_receipt_clean"] is False
    assert artifact["honest_verdict"].startswith("blocked_runtime_receipt:")
    assert artifact["blocked_reasons"] == [
        f"{QWEN}: {qwen_error}",
        f"{GEMMA26}: {gemma_error}",
    ]
    assert artifact["cache_status"]["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["smoke_generation_status"]["clean_mandated_model_ids"] == []
    assert artifact["smoke_generation_status"]["per_model"][QWEN]["exception"] == qwen_error
    assert artifact["smoke_generation_status"]["per_model"][GEMMA26]["exception"] == gemma_error
    assert artifact["smoke_generation_status"]["legacy_cpu_loader_controls"] == []
    assert len([call for call in calls if "--exp3338-runtime-worker" in call]) == 2


def test_req_infer_sota_3338_missing_cache_and_writer(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-3338: no mandated cache writes a parseable blocked receipt."""

    runner, calls = _runner()
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        cached_pair_resolver=lambda gpu_indices=(0, 1): [],
        monotonic=iter([2.0, 2.25]).__next__,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["runtime_receipt_clean"] is False
    assert artifact["honest_verdict"].startswith("blocked_runtime_receipt:")
    assert artifact["blocked_reasons"] == ["no mandated SOTA GGUF files available locally"]
    assert artifact["cache_status"]["cached_model_ids"] == []
    assert artifact["cache_status"]["missing_model_ids"] == [QWEN, GEMMA31, GEMMA26]
    assert all(row["model_path"] is None for row in artifact["model_specs"])
    assert [row["hf_id"] for row in artifact["model_specs"]] == [QWEN, GEMMA31, GEMMA26]
    assert not [call for call in calls if "--exp3338-runtime-worker" in call]
