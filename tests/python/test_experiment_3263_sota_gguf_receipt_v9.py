"""Tests for Exp 3263 SOTA GGUF receipt v9.

Spec refs: REQ-REPORT-3263, SCENARIO-REPORT-3263.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_gguf_receipt_3263 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "sota_gguf_receipt_v9_ready",
    "sota_gguf_receipt_ready",
    "model_specs",
    "per_model_receipts",
    "gpu_mem_used_mib",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_exp3262(root: Path, *, ready: bool = True) -> Path:
    path = root / mod.EXP3262_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact": "experiment_3262_llama_cpp_cuda_receipt_smoke_v4",
                "llama_cpp_cuda_receipt_ready": ready,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_model(
    cache_root: Path,
    hf_id: str,
    *,
    filename: str | None = None,
    content: bytes = b"gguf fixture",
) -> Path:
    owner, name = hf_id.split("/", 1)
    stem = name.removesuffix("-GGUF")
    path = (
        cache_root
        / f"models--{owner}--{name}"
        / "snapshots"
        / "rev1"
        / (filename or f"{stem}-Q4_K_M.gguf")
    )
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
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def _worker_payload(
    hf_id: str,
    *,
    output: str = "READY from CUDA",
    tokens: int = 4,
    baseline: int = 100,
    used: int = 2300,
    layers: int = 61,
    duration: float = 12.5,
) -> dict[str, Any]:
    return {
        "ok": bool(output),
        "model_id": hf_id,
        "load_status": "loaded",
        "generation_status": "generated" if output else "empty_response",
        "output_text": output,
        "tokens_generated": tokens,
        "n_gpu_layers_requested": -1,
        "gpu_layers_offloaded": layers,
        "gpu_mem_baseline_mib": baseline,
        "gpu_mem_used_mib": used,
        "gpu_mem_delta_mib": max(0, used - baseline),
        "duration_s": duration,
        "usage": {"prompt_tokens": 9, "completion_tokens": tokens, "total_tokens": 9 + tokens},
    }


def _runner(
    payloads: dict[str, dict[str, Any]] | None = None,
    returncodes: dict[str, int] | None = None,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    payloads = payloads or {}
    returncodes = returncodes or {}

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        hf_id = command[command.index("--model-id") + 1]
        payload = payloads.get(hf_id, _worker_payload(hf_id))
        returncode = returncodes.get(hf_id, 0)
        return _command(
            command,
            returncode=returncode,
            stdout=json.dumps(payload, sort_keys=True) + "\n",
            stderr="llama_model_load: offloaded 61/61 layers to GPU\n"
            if returncode == 0
            else "worker failed\n",
        )

    return run, calls


def test_req_report_3263_spec_anchor_exists() -> None:
    """REQ-REPORT-3263: OpenSpec declares the SOTA GGUF receipt before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3263" in spec
    assert "SCENARIO-REPORT-3263" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "sota_gguf_receipt_v9_ready" in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3263_gated_skip_when_exp3262_not_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3263: Exp 3262 controls the SOTA receipt gate."""

    _write_exp3262(tmp_path, ready=False)
    runner, calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([1.0, 1.25]).__next__,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3263"
    assert artifact["sota_gguf_receipt_v9_ready"] is True
    assert artifact["sota_gguf_receipt_ready"] is False
    assert artifact["blocked_reason"] == "gated_exp3262_llama_cpp_cuda_receipt_not_ready"
    assert artifact["model_specs"]["mandated_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["model_specs"]["headline_model_id"] is None
    assert artifact["per_model_receipts"] == []
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert calls == []


def test_req_report_3263_blocks_when_no_mandated_gguf_cached(tmp_path: Path) -> None:
    """REQ-REPORT-3263: no cached mandated SOTA GGUF writes an honest block."""

    _write_exp3262(tmp_path, ready=True)
    runner, calls = _runner()

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([2.0, 2.5]).__next__,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["sota_gguf_receipt_v9_ready"] is True
    assert artifact["sota_gguf_receipt_ready"] is False
    assert artifact["blocked_reason"] == "blocked_sota_gguf_not_cached"
    assert artifact["missing_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["cached_model_ids"] == []
    assert artifact["per_model_receipts"] == []
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked_sota_gguf_not_cached" in artifact["honest_verdict"]
    assert calls == []


def test_scenario_report_3263_cached_models_generate_per_model_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3263: cached mandated GGUFs are each loaded with CUDA offload."""

    _write_exp3262(tmp_path, ready=True)
    cache_root = tmp_path / "hf-cache"
    qwen_path = _write_model(cache_root, QWEN)
    gemma_path = _write_model(cache_root, GEMMA26)
    runner, calls = _runner(
        {
            QWEN: _worker_payload(QWEN, used=2300, layers=61, duration=12.5),
            GEMMA26: _worker_payload(GEMMA26, used=1900, layers=48, duration=10.0),
        }
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        env={"EXTRA_ENV_FOR_TEST": "1"},
        command_runner=runner,
        monotonic=iter([10.0, 24.0]).__next__,
        random_seed=3263,
        max_tokens=8,
        n_gpu_layers=-1,
    )

    assert artifact["sota_gguf_receipt_ready"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["cached_model_ids"] == [QWEN, GEMMA26]
    assert artifact["missing_model_ids"] == [GEMMA31]
    assert artifact["model_specs"]["headline_model_id"] == QWEN
    assert artifact["model_specs"]["headline_model_path"] == str(qwen_path)
    assert artifact["model_specs"]["mandated_models"][QWEN]["cached"] is True
    assert artifact["model_specs"]["mandated_models"][GEMMA26]["cached"] is True
    assert artifact["model_specs"]["mandated_models"][GEMMA31]["cached"] is False
    assert artifact["gpu_mem_used_mib"] == 2300
    assert artifact["random_seed"] == 3263
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(14.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "sota_gguf_receipt_ready=true" in artifact["honest_verdict"]
    assert len(artifact["per_model_receipts"]) == 2

    first = artifact["per_model_receipts"][0]
    assert first["model_id"] == QWEN
    assert first["model_path"] == str(qwen_path)
    assert first["model_load_evidence"]["load_status"] == "loaded"
    assert first["generation_evidence"]["tokens_generated"] == 4
    assert first["generation_evidence"]["output_preview"] == "READY from CUDA"
    assert first["gpu_evidence"]["gpu_layers_offloaded"] == 61
    assert first["receipt_passed"] is True

    assert len(calls) == 2
    assert "--exp3263-sota-gguf-worker" in calls[0]["command"]
    assert str(qwen_path) in calls[0]["command"]
    assert str(gemma_path) in calls[1]["command"]
    assert calls[0]["kwargs"]["env"]["PYTHONHASHSEED"] == "3263"
    assert calls[0]["kwargs"]["env"]["EXTRA_ENV_FOR_TEST"] == "1"


def test_req_report_3263_worker_failures_keep_receipt_gate_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3263: failed or incomplete generations cannot open the gate."""

    _write_exp3262(tmp_path, ready=True)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, QWEN)
    runner, _calls = _runner(
        {QWEN: _worker_payload(QWEN, output="", tokens=0, baseline=100, used=100, layers=0)},
        {QWEN: 1},
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([4.0, 8.0]).__next__,
    )

    assert artifact["sota_gguf_receipt_ready"] is False
    assert artifact["blocked_reason"] == "sota_gguf_receipt_incomplete"
    assert artifact["model_specs"]["headline_model_id"] is None
    assert artifact["per_model_receipts"][0]["receipt_passed"] is False
    assert artifact["per_model_receipts"][0]["worker_attempt"]["returncode"] == 1
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_helpers_cover_cache_selection_and_command_execution(tmp_path: Path) -> None:
    """REQ-REPORT-3263: helper behavior is deterministic and JSON-safe."""

    assert mod._selected_python(tmp_path) == sys.executable
    candidate = tmp_path / ".venv" / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# placeholder\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(candidate)

    assert mod._safe_model_slug(QWEN) == "unsloth_Qwen3_6-35B-A3B-GGUF"
    assert mod._model_id_from_path(
        Path("/cache/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/rev/model.gguf")
    ) == QWEN
    assert mod._model_id_from_path(Path("/plain/model.gguf")) == "local/model"

    cache_root = tmp_path / "hf-cache"
    mmproj = _write_model(cache_root, QWEN, filename="mmproj-F16.gguf")
    zero = _write_model(cache_root, QWEN, filename="Qwen3.6-35B-A3B-Q4_K_M-zero.gguf")
    zero.write_bytes(b"")
    no_exist = (
        cache_root
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / ".no_exist"
        / "rev"
        / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    no_exist.parent.mkdir(parents=True)
    no_exist.write_bytes(b"marker")
    good = _write_model(cache_root, QWEN, filename="Qwen3.6-35B-A3B-Q4_K_M.gguf")
    bigger = _write_model(cache_root, QWEN, filename="Qwen3.6-35B-A3B-Q5_K_M.gguf", content=b"x" * 40)

    resolved = mod._resolve_cached_mandated_ggufs(tmp_path, [cache_root])
    assert [row["model_id"] for row in resolved] == [QWEN]
    assert resolved[0]["path"] == str(good)
    assert resolved[0]["candidate_count"] == 5
    assert str(mmproj) in resolved[0]["candidate_paths"]
    assert str(bigger) in resolved[0]["candidate_paths"]
    assert mod._select_candidate(
        [{"usable_candidate": True, "path": "/tmp/custom.gguf", "size_bytes": 1}]
    ) == {"usable_candidate": True, "path": "/tmp/custom.gguf", "size_bytes": 1}

    evidence = mod._file_evidence(good)
    assert evidence["status"] == "available"
    assert evidence["sha256"]
    assert mod._file_evidence(None)["status"] == "missing"
    assert mod._file_evidence(tmp_path / "missing.gguf")["status"] == "missing"
    bounded = mod._file_evidence(good, full_sha_max_bytes=1)
    assert bounded["checksum_algorithm"] == "sha256_head_tail_1mib_plus_size_mtime"
    assert bounded["bounded_sha256"]

    parsed_worker, _calls = _runner(
        {QWEN: {**_worker_payload(QWEN), "gpu_layers_offloaded": 0}}
    )
    worker_out = mod._run_model_worker(
        selected_python=SELECTED_PYTHON,
        model={"model_id": QWEN, "path": str(good)},
        n_gpu_layers=-1,
        max_tokens=4,
        random_seed=3263,
        env={},
        command_runner=parsed_worker,
    )
    assert worker_out["payload"]["gpu_layers_offloaded"] == 61

    payload = _worker_payload(QWEN)
    receipt = mod._receipt_from_worker(
        model={"model_id": QWEN, "path": str(good), "filename": good.name, "size_bytes": 12},
        worker={
            "attempted": True,
            "returncode": 0,
            "command_hash": "hash",
            "stderr_summary": "llama_model_load: offloading 42 repeating layers to GPU",
            "payload": {**payload, "gpu_layers_offloaded": 0},
        },
    )
    assert receipt["gpu_evidence"]["gpu_layers_offloaded"] == 42
    assert receipt["receipt_passed"] is True
    no_delta_payload = dict(payload)
    no_delta_payload.pop("gpu_mem_delta_mib")
    no_delta_receipt = mod._receipt_from_worker(
        model={"model_id": QWEN, "path": str(good), "filename": good.name, "size_bytes": 12},
        worker={
            "attempted": True,
            "returncode": 0,
            "command_hash": "hash",
            "stderr_summary": "llama_model_load: offloaded 61/61 layers to GPU",
            "payload": no_delta_payload,
        },
    )
    assert no_delta_receipt["gpu_evidence"]["gpu_mem_delta_mib"] == 2200

    result = mod._run_command([sys.executable, "-c", "print('ok')"], timeout_s=10)
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ok"


def test_main_prints_artifact(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-REPORT-3263: CLI entrypoint emits the written artifact payload."""

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda **_kwargs: {"artifact": mod.ARTIFACT, "sota_gguf_receipt_ready": False},
    )

    assert mod.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["artifact"] == mod.ARTIFACT
