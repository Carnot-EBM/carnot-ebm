"""Tests for Exp 3193 llama.cpp CUDA/offload health probe v1.

Spec refs: REQ-VERIFY-3193, SCENARIO-VERIFY-3193.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import llama_cpp_cuda_offload_health_probe_v1 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"


def _model_path(cache_root: Path, hf_id: str, filename: str) -> Path:
    owner, name = hf_id.split("/", 1)
    return cache_root / f"models--{owner}--{name}" / "snapshots" / "rev1" / filename


def _write_model(cache_root: Path, hf_id: str, filename: str, content: bytes = b"gguf") -> Path:
    path = _model_path(cache_root, hf_id, filename)
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
    *,
    response: str = "READY",
    memory_delta: int = 12288,
    backend_log: str = "llama_model_load: offloaded 42/42 layers to GPU\n",
) -> dict[str, Any]:
    return {
        "ok": True,
        "model_id": QWEN,
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_effective": 42,
        "prompt": mod.DEFAULT_PROMPT,
        "response_text": response,
        "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
        "wall_clock_s": 2.25,
        "backend_log_tail": backend_log,
        "gpu_memory": {
            "before": [{"index": 0, "memory_used_mib": 100}],
            "after_load": [{"index": 0, "memory_used_mib": 100 + memory_delta}],
            "after_generate": [{"index": 0, "memory_used_mib": 100 + memory_delta}],
        },
    }


def _runner(
    *,
    torch_cuda: bool = True,
    nvidia_ok: bool = True,
    llama_import_ok: bool = True,
    llama_supports_gpu: bool = True,
    worker_payload: dict[str, Any] | None = None,
    worker_returncode: int = 0,
    worker_stderr: str = "llama_model_load: offloaded 42/42 layers to GPU\n",
):
    def run(
        command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None
    ) -> dict[str, Any]:
        del timeout_s, env
        joined = "\n".join(command)
        if command and command[0] == "nvidia-smi":
            if not nvidia_ok:
                return _command(command, returncode=127, stderr="nvidia-smi unavailable\n")
            return _command(
                command,
                stdout="0, NVIDIA GeForce RTX 3090, 24576, 100, 24476, 595.71.05\n",
            )
        if "exp3193_torch_cuda_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "torch_present": True,
                        "torch_import_ok": True,
                        "torch_version": "2.11.0+cu128",
                        "cuda_available": torch_cuda,
                        "device_count": 1 if torch_cuda else 0,
                        "cuda_version": "12.8" if torch_cuda else None,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if "exp3193_llama_cpp_backend_probe" in joined:
            return _command(
                command,
                returncode=0 if llama_import_ok else 1,
                stdout=json.dumps(
                    {
                        "llama_cpp_import_ok": llama_import_ok,
                        "loader_name": "llama_cpp.Llama",
                        "llama_cpp_version": "0.3.23" if llama_import_ok else None,
                        "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py",
                        "llama_cpp_supports_gpu_offload": llama_supports_gpu
                        if llama_import_ok
                        else False,
                        "backend_error": "" if llama_import_ok else "ModuleNotFoundError",
                    },
                    sort_keys=True,
                )
                + "\n",
                stderr="" if llama_import_ok else "no module named llama_cpp\n",
            )
        if "--exp3193-offload-worker" in command:
            payload = worker_payload or _worker_payload()
            return _command(
                command,
                returncode=worker_returncode,
                stdout=json.dumps(payload, sort_keys=True) + "\n",
                stderr=worker_stderr,
            )
        raise AssertionError(f"unexpected command: {command}")

    return run


def test_req_verify_3193_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3193: OpenSpec declares the offload probe before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3193" in spec
    assert "SCENARIO-VERIFY-3193" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3193_full_cuda_offload_unlocks_clean_rerun(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3193: offload-backed mandated receipt opens the clean gate."""

    cache_root = tmp_path / "hf-cache"
    qwen_path = _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", b"qwen")
    _write_model(cache_root, GEMMA26, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", b"gemma")

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(),
        monotonic=iter([10.0, 13.0]).__next__,
        tests_run=["SCENARIO-VERIFY-3193 focused"],
    )

    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3193"
    assert artifact["selected_model"] == QWEN
    assert artifact["selected_model_path"] == str(qwen_path)
    assert artifact["selected_model_file_hash"] == mod.sha256_file(qwen_path)
    assert artifact["n_gpu_layers_requested"] == -1
    assert artifact["n_gpu_layers_effective"] == 42
    assert artifact["substrate_classification"] == "full_local_sota_receipt"
    assert artifact["clean_rerun_allowed"] is True
    assert artifact["headline_claim_allowed"] is True
    assert artifact["receipt_count"] == 1
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["blocker_reasons"] == []

    assert [row["hf_id"] for row in artifact["cache_inventory"]] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["cache_inventory"][0]["available_quant_files"][0]["path"] == str(qwen_path)
    assert artifact["cache_inventory"][1]["cache_status"] == "missing"
    assert artifact["preconditions_checked"]["nvidia_smi"]["available"] is True
    assert artifact["preconditions_checked"]["torch_cuda"]["cuda_available"] is True
    assert artifact["llama_cpp_backend_metadata"]["llama_cpp_supports_gpu_offload"] is True
    assert artifact["gpu_observations"]["offload_evidenced"] is True
    assert artifact["gpu_observations"]["max_memory_delta_mib"] == 12288

    receipt = artifact["receipts"][0]
    assert receipt["selected_model"] == QWEN
    assert receipt["model_path"] == str(qwen_path)
    assert receipt["prompt_hash"] == mod.hash_text(mod.DEFAULT_PROMPT)
    assert receipt["response_hash"] == mod.hash_text("READY")
    assert receipt["token_counts"]["completion_tokens"] == 1
    assert receipt["stderr_backend_tail"].endswith("GPU")


def test_req_verify_3193_backend_absent_blocks_without_cpu_claim(tmp_path: Path) -> None:
    """REQ-VERIFY-3193: CPU-only llama.cpp cannot set clean/headline gates."""

    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", b"gemma")

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(llama_supports_gpu=False),
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert artifact["selected_model"] == GEMMA26
    assert artifact["selected_model_file_hash"] is not None
    assert artifact["receipt_count"] == 0
    assert artifact["receipts"] == []
    assert artifact["substrate_classification"] == "cuda_backend_absent"
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["n_gpu_layers_effective"] is None
    assert artifact["honest_verdict"].startswith("blocked_cuda_backend_absent:")
    assert artifact["blocker_reasons"] == ["llama_cpp backend does not report GPU offload support"]


def test_req_verify_3193_cpu_fallback_receipt_stays_non_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-3193: a receipt without offload evidence is loud CPU fallback."""

    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", b"qwen")
    payload = _worker_payload(memory_delta=0, backend_log="llama_model_load: CPU only\n")
    payload["n_gpu_layers_effective"] = 0

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(worker_payload=payload, worker_stderr="llama_model_load: CPU only\n"),
        monotonic=iter([2.0, 6.0]).__next__,
    )

    assert artifact["receipt_count"] == 1
    assert artifact["substrate_classification"] == "cpu_fallback_receipt_only"
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["n_gpu_layers_effective"] == 0
    assert artifact["gpu_observations"]["offload_evidenced"] is False
    assert artifact["honest_verdict"].startswith("blocked_cpu_fallback_receipt_only:")
    assert artifact["blocker_reasons"] == ["receipt completed without CUDA/offload evidence"]


def test_req_verify_3193_missing_cache_loader_cuda_and_worker_failures(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3193: every blocked substrate records a precise reason."""

    missing_cache = mod.build_artifact(
        tmp_path,
        cache_root=tmp_path / "empty",
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(),
        monotonic=iter([1.0, 1.1]).__next__,
    )
    assert missing_cache["substrate_classification"] == "model_cache_missing"
    assert missing_cache["honest_verdict"].startswith("blocked_model_cache_missing:")

    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", b"qwen")
    loader_missing = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(llama_import_ok=False),
        monotonic=iter([2.0, 2.2]).__next__,
    )
    assert loader_missing["substrate_classification"] == "loader_missing"
    assert loader_missing["honest_verdict"].startswith("blocked_loader_missing:")

    cuda_missing = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(torch_cuda=False, nvidia_ok=False, llama_supports_gpu=False),
        monotonic=iter([3.0, 3.3]).__next__,
    )
    assert cuda_missing["substrate_classification"] == "cuda_unavailable"
    assert cuda_missing["honest_verdict"].startswith("blocked_cuda_unavailable:")
    assert cuda_missing["blocker_reasons"] == [
        "nvidia-smi did not report a visible NVIDIA GPU",
        "selected Python torch.cuda.is_available() is false",
        "llama_cpp backend did not report GPU offload support",
    ]

    worker_failed = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(
            worker_payload={"ok": False, "error": "load failed", "backend_log_tail": "bad"},
            worker_returncode=1,
            worker_stderr="load failed\n",
        ),
        monotonic=iter([4.0, 4.5]).__next__,
    )
    assert worker_failed["substrate_classification"] == "gpu_offload_unhealthy"
    assert worker_failed["blocker_reasons"] == ["load failed"]
    assert worker_failed["honest_verdict"].startswith("blocked_gpu_offload_unhealthy:")


def test_req_verify_3193_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3193: writer persists JSON and rejects ambiguous gate artifacts."""

    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, QWEN, "Qwen3.6-35B-A3B-random.gguf", b"qwen")
    ignored = _write_model(cache_root, QWEN, "mmproj-F16.gguf", b"ignore")
    output = mod.write_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python=SELECTED_PYTHON,
        env={"EXTRA_FIXTURE_ENV": "1"},
        command_runner=_runner(),
        monotonic=iter([5.0, 7.0]).__next__,
        tests_run=["focused", "coverage"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(2.0)
    assert saved["tests_run"] == ["focused", "coverage"]
    assert ignored.name not in {
        Path(row["path"]).name
        for family in saved["cache_inventory"]
        for row in family["available_quant_files"]
    }
    assert mod.quantization_from_name("foo-UD-Q4_K_M.gguf") == "UD-Q4_K_M"
    assert mod.quantization_from_name("foo.gguf") == "unknown"
    assert mod.default_hf_cache_root({}).as_posix().endswith(".cache/huggingface/hub")
    assert mod.default_hf_cache_root({"HF_HOME": str(tmp_path / "hfhome")}) == (
        tmp_path / "hfhome" / "hub"
    )
    assert mod.default_hf_cache_root({"HUGGINGFACE_HUB_CACHE": str(tmp_path / "hub")}) == (
        tmp_path / "hub"
    )
    assert mod.selected_python_for(tmp_path / "missing") != ""
    assert mod.first_json_line("not json\n{\"ok\": true}\n") == {"ok": True}
    assert mod.first_json_line("not json\n") == {}
    assert mod.safe_int("x") is None
    assert mod.safe_float("x") is None
    assert mod.duration(3.0, 1.0) == 0.0
    assert mod.truncate_tail("abcdef", limit=3) == "def"
    assert mod.max_gpu_memory_delta(
        {"before": [{"index": 0, "memory_used_mib": 5}], "after_load": []}
    ) == 0
    assert mod.parse_nvidia_smi_rows("bad\nname, nope, nope, nope, nope, nope\n") == []
    assert mod.memory_by_index([{"index": 0, "memory_used_mib": 5}, "bad"]) == {0: 5}

    missing_hash = tmp_path / "missing.gguf"
    assert mod.sha256_file(missing_hash) is None
    assert mod.run_command([str(tmp_path / "missing-command")], timeout_s=1)["returncode"] is None
    ok = mod.run_command(["printf", "ok"], timeout_s=1, env={"LC_ALL": "C"})
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    assert mod.parse_offloaded_layers("offloaded 7/42 layers to GPU") == 7
    assert mod.parse_offloaded_layers("offloading 5 repeating layers to GPU") == 5
    assert mod.parse_offloaded_layers("cpu only") is None
    assert mod.run_receipt_worker(
        selected_python=SELECTED_PYTHON,
        selected_model=None,
        n_gpu_layers_requested=-1,
        command_runner=_runner(),
        timeout_s=1,
    ) == mod.empty_worker()
    worker = mod.run_receipt_worker(
        selected_python=SELECTED_PYTHON,
        selected_model={"hf_id": QWEN, "path": str(cache_root / "model.gguf")},
        n_gpu_layers_requested=-1,
        command_runner=_runner(
            worker_payload={
                "ok": True,
                "prompt": mod.DEFAULT_PROMPT,
                "response_text": "READY",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            worker_stderr="offloaded 3/42 layers to GPU\n",
        ),
        timeout_s=1,
    )
    assert worker["payload"]["backend_log_tail"] == "offloaded 3/42 layers to GPU"
    assert mod.receipt_from_worker(
        selected_model=None,
        selected_model_hash=None,
        command_hash="x",
        worker_payload={},
        worker_returncode=0,
        stderr_tail="",
    ) == []
    assert mod.receipt_from_worker(
        selected_model={"hf_id": QWEN, "path": "model.gguf"},
        selected_model_hash="hash",
        command_hash="x",
        worker_payload={"ok": True, "response_text": ""},
        worker_returncode=0,
        stderr_tail="",
    ) == []
    assert mod.token_counts_for("two words", "one", {}) == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "source": "whitespace_estimate",
    }
    assert (
        mod.infer_effective_gpu_layers(
            payload={},
            stderr_tail="offloaded 8/42 layers to GPU",
            offload_evidenced=False,
            requested=-1,
        )
        == 8
    )
    assert (
        mod.infer_effective_gpu_layers(
            payload={},
            stderr_tail="",
            offload_evidenced=True,
            requested=-1,
        )
        == -1
    )
    assert (
        mod.classify_substrate(
            selected_model={"hf_id": QWEN},
            llama_meta={"llama_cpp_import_ok": True, "llama_cpp_supports_gpu_offload": True},
            nvidia={"available": True},
            torch_cuda={"cuda_available": True},
            worker={"attempted": False, "returncode": None},
            receipts=[],
            offload_evidenced=False,
        )
        == "gpu_offload_unhealthy"
    )
    assert mod.blocker_reasons(
        substrate_classification="gpu_offload_unhealthy",
        selected_model={"hf_id": QWEN},
        llama_meta={},
        nvidia={},
        torch_cuda={},
        worker={"payload": {}, "stderr_tail": ""},
        receipts=[],
    ) == ["worker did not produce a usable receipt"]
    assert mod.blocker_reasons(
        substrate_classification="gpu_offload_unhealthy",
        selected_model={"hf_id": QWEN},
        llama_meta={},
        nvidia={},
        torch_cuda={},
        worker={"payload": {}, "stderr_tail": ""},
        receipts=[{"transcript_hash": "x"}],
    ) == ["GPU offload receipt unhealthy"]

    broken = dict(saved)
    del broken["receipts"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["substrate_classification"] = "wrong"
    with pytest.raises(ValueError, match="substrate_classification"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["clean_rerun_allowed"] = True
    broken["substrate_classification"] = "cpu_fallback_receipt_only"
    with pytest.raises(ValueError, match="clean rerun"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["headline_claim_allowed"] = True
    broken["clean_rerun_allowed"] = False
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["honest_verdict"] = "complete: wrong"
    broken["substrate_classification"] = "model_cache_missing"
    broken["clean_rerun_allowed"] = False
    broken["headline_claim_allowed"] = False
    with pytest.raises(ValueError, match="blocked_"):
        mod.validate_artifact(broken)
