"""Tests for Exp 3268 SOTA receipt methodology supplement v1.

Spec refs: REQ-REPORT-3268, SCENARIO-REPORT-3268.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from carnot.reporting import sota_receipt_methodology_supplement_3268 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "sota_receipt_methodology_supplement_v1_ready",
    "clean_sota_receipt_eligible",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "gpu_mem_used_mib",
    "tokens_generated",
    "receipt_duration_floor_met",
    "methodology_findings",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_prior_3263(root: Path, *, ready: bool = True, duration: float = 19.25) -> Path:
    path = root / mod.EXP3263_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact": "experiment_3263_sota_gguf_receipt_v9",
                "sota_gguf_receipt_ready": ready,
                "duration_s": duration,
                "flagged_adversarial": True,
                "corrigendum_pending": [{"kind": "duration_floor"}],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_model(cache_root: Path, hf_id: str, *, content: bytes = b"gguf fixture") -> Path:
    owner, name = hf_id.split("/", 1)
    stem = name.removesuffix("-GGUF")
    path = (
        cache_root
        / f"models--{owner}--{name}"
        / "snapshots"
        / "rev1"
        / f"{stem}-Q4_K_M.gguf"
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


def _cuda_probe_stdout(*, cuda_available: bool = True, llama_cpp_cuda: bool = True) -> str:
    return (
        json.dumps(
            {
                "python": SELECTED_PYTHON,
                "torch_import_ok": True,
                "cuda_available": cuda_available,
                "cuda_device_count": 2 if cuda_available else 0,
                "cuda_device_name": "NVIDIA GeForce RTX 3090" if cuda_available else "",
                "llama_cpp_import_ok": True,
                "llama_cpp_supports_gpu_offload": llama_cpp_cuda,
                "llama_cpp_system_info": "CUDA : ARCHS = 860 | USE_GRAPHS = 1"
                if llama_cpp_cuda
                else "CPU : AVX2 = 1",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _worker_payload(
    hf_id: str,
    *,
    output: str = "READY from live CUDA receipt",
    tokens: int = 2048,
    baseline: int = 4,
    used: int = 9000,
    layers: int = 31,
    duration: float = 62.5,
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
        "usage": {"prompt_tokens": 32, "completion_tokens": tokens, "total_tokens": tokens + 32},
        "live_generation_calls": 4,
    }


def _runner(
    *,
    nvidia_ok: bool = True,
    cuda_available: bool = True,
    llama_cpp_cuda: bool = True,
    worker_payloads: dict[str, dict[str, Any]] | None = None,
    worker_returncodes: dict[str, int] | None = None,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    worker_payloads = worker_payloads or {}
    worker_returncodes = worker_returncodes or {}

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if command[:1] == ["nvidia-smi"]:
            if not nvidia_ok:
                return _command(command, returncode=1, stderr="nvidia-smi failed\n")
            return _command(
                command,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                ),
            )
        if "exp3268_cuda_probe" in joined:
            return _command(
                command,
                returncode=0 if cuda_available else 1,
                stdout=_cuda_probe_stdout(
                    cuda_available=cuda_available,
                    llama_cpp_cuda=llama_cpp_cuda,
                ),
                stderr="" if cuda_available else "cuda unavailable\n",
            )
        if "--exp3268-sota-methodology-worker" in command:
            hf_id = command[command.index("--model-id") + 1]
            payload = worker_payloads.get(hf_id, _worker_payload(hf_id))
            returncode = worker_returncodes.get(hf_id, 0)
            return _command(
                command,
                returncode=returncode,
                stdout=json.dumps(payload, sort_keys=True) + "\n",
                stderr="llama_model_load: offloaded 31/31 layers to GPU\n"
                if returncode == 0
                else "worker failed\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return run, calls


def test_req_report_3268_spec_anchor_exists() -> None:
    """REQ-REPORT-3268: OpenSpec declares the supplement before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3268" in spec
    assert "SCENARIO-REPORT-3268" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "clean_sota_receipt_eligible" in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3268_no_cached_gguf_writes_noneligible_boundary(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3268: missing mandated GGUFs produce a precise blocker."""

    _write_prior_3263(tmp_path)
    runner, calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3268"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["sota_receipt_methodology_supplement_v1_ready"] is True
    assert artifact["clean_sota_receipt_eligible"] is False
    assert artifact["models_used"] == []
    assert [row["model_id"] for row in artifact["missing_model_specs"]] == list(
        mod.MANDATED_MODEL_IDS
    )
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["tokens_generated"] == 0
    assert artifact["receipt_duration_floor_met"] is False
    assert "no_mandated_sota_gguf_cached" in artifact["methodology_findings"]
    assert artifact["prior_receipt_boundary"]["clean_reuse_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(calls) == 2
    assert not any("--exp3268-sota-methodology-worker" in call["command"] for call in calls)


def test_scenario_report_3268_clean_live_receipt_is_eligible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3268: live local offloaded non-empty receipts can open the gate."""

    _write_prior_3263(tmp_path)
    cache_root = tmp_path / "hf-cache"
    gemma_path = _write_model(cache_root, GEMMA26)
    runner, calls = _runner(
        worker_payloads={GEMMA26: _worker_payload(GEMMA26, used=9000, duration=62.5)}
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        env={"EXTRA_ENV_FOR_TEST": "1"},
        command_runner=runner,
        monotonic=iter([10.0, 75.0]).__next__,
    )

    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["clean_sota_receipt_eligible"] is True
    assert artifact["receipt_duration_floor_met"] is True
    assert artifact["methodology_findings"] == ["methodology_clean_live_receipt_available"]
    assert artifact["models_used"] == [
        {
            "model_id": GEMMA26,
            "model_path": str(gemma_path),
            "filename": gemma_path.name,
            "cached": True,
            "attempted_live_receipt": True,
            "clean_row": True,
        }
    ]
    assert [row["model_id"] for row in artifact["missing_model_specs"]] == [QWEN, GEMMA31]
    assert artifact["model_specs"]["mandated_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["model_specs"]["mandated_models"][GEMMA26]["cached"] is True
    assert artifact["model_specs"]["mandated_models"][QWEN]["cached"] is False
    assert artifact["gpu_mem_used_mib"] == 9000
    assert artifact["tokens_generated"] == 2048
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == 65.0
    assert artifact["honest_verdict"].startswith("complete:")

    worker_calls = [call for call in calls if "--exp3268-sota-methodology-worker" in call["command"]]
    assert len(worker_calls) == 1
    assert str(gemma_path) in worker_calls[0]["command"]
    assert "--target-duration-s" in worker_calls[0]["command"]
    assert worker_calls[0]["kwargs"]["env"]["PYTHONHASHSEED"] == "3268"
    assert worker_calls[0]["kwargs"]["env"]["EXTRA_ENV_FOR_TEST"] == "1"


def test_req_report_3268_precondition_failures_block_live_loading(tmp_path: Path) -> None:
    """REQ-REPORT-3268: precondition failures are recorded before any model load."""

    _write_prior_3263(tmp_path)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, calls = _runner(nvidia_ok=False, cuda_available=False)

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([2.0, 3.0]).__next__,
    )

    assert artifact["clean_sota_receipt_eligible"] is False
    assert "nvidia_smi_unavailable" in artifact["methodology_findings"]
    assert "selected_python_cuda_unavailable" in artifact["methodology_findings"]
    assert "live_receipt_preconditions_failed" in artifact["methodology_findings"]
    assert artifact["models_used"][0]["attempted_live_receipt"] is False
    assert not any("--exp3268-sota-methodology-worker" in call["command"] for call in calls)


def test_req_report_3268_duration_floor_blocks_otherwise_valid_receipt(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3268: offloaded output is not clean if the duration floor is short."""

    _write_prior_3263(tmp_path)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, _calls = _runner(
        worker_payloads={GEMMA26: _worker_payload(GEMMA26, used=8500, duration=18.0)}
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([4.0, 23.0]).__next__,
    )

    assert artifact["clean_sota_receipt_eligible"] is False
    assert artifact["receipt_duration_floor_met"] is False
    assert "duration_floor_not_met: duration_s=19.0 < 60.0" in artifact["methodology_findings"]
    assert artifact["models_used"][0]["clean_row"] is True
    assert artifact["gpu_mem_used_mib"] == 8500
    assert artifact["tokens_generated"] == 2048


def test_req_report_3268_worker_failures_keep_supplement_noneligible(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3268: failed or empty live rows cannot become clean evidence."""

    _write_prior_3263(tmp_path)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root, GEMMA26)
    runner, _calls = _runner(
        worker_payloads={
            GEMMA26: _worker_payload(
                GEMMA26,
                output="",
                tokens=0,
                baseline=100,
                used=100,
                layers=0,
                duration=61.0,
            )
        },
        worker_returncodes={GEMMA26: 1},
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([5.0, 70.0]).__next__,
    )

    assert artifact["receipt_duration_floor_met"] is True
    assert artifact["clean_sota_receipt_eligible"] is False
    assert "no_methodology_clean_live_receipts" in artifact["methodology_findings"]
    assert artifact["per_model_receipts"][0]["methodology_clean"] is False
    assert artifact["models_used"][0]["clean_row"] is False
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["tokens_generated"] == 0


def test_helpers_cover_parsing_and_entrypoint(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """REQ-REPORT-3268: helpers stay deterministic and JSON-safe."""

    assert mod._selected_python(tmp_path) == sys.executable
    candidate = tmp_path / ".venv" / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# placeholder\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(candidate)

    assert mod._parse_nvidia_smi_csv("bad\n0, GPU, 24576, 4, 0, 595.71.05\n") == [
        {
            "index": 0,
            "name": "GPU",
            "memory_total_mib": 24576,
            "memory_used_mib": 4,
            "utilization_gpu_pct": 0,
            "driver_version": "595.71.05",
        }
    ]
    assert mod._prior_receipt_boundary({"duration_s": "bad"})["prior_duration_s"] == 0.0

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda **_kwargs: {"artifact": mod.ARTIFACT, "clean_sota_receipt_eligible": False},
    )
    assert mod.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["artifact"] == mod.ARTIFACT
