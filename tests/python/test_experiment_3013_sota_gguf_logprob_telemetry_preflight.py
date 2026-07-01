"""Tests for Exp 3013 SOTA GGUF logprob telemetry preflight.

Spec: REQ-INFER-SOTA-021,
      SCENARIO-INFER-SOTA-021-001,
      SCENARIO-INFER-SOTA-021-002,
      SCENARIO-INFER-SOTA-021-003,
      SCENARIO-INFER-SOTA-021-004,
      SCENARIO-INFER-SOTA-021-005
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from scripts import experiment_3013_sota_gguf_logprob_telemetry_preflight_v1 as exp


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
    fixture_sizes = {QWEN: 30, GEMMA31: 20, GEMMA26: 10}
    gguf.write_text("x" * fixture_sizes.get(hf_id, 12), encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def _write_all_cached_models(tmp_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for hf_id in (QWEN, GEMMA31, GEMMA26):
        _gguf, env = _write_cached_model(tmp_path, hf_id)
    return env


def _free_port(port: int = 45555) -> dict[str, Any]:
    return {
        "available": True,
        "host": "127.0.0.1",
        "port": port,
        "endpoint_url": f"http://127.0.0.1:{port}",
        "error": None,
    }


def _server_unavailable(env: dict[str, str]) -> dict[str, Any]:
    path = env.get("CARNOT_LLAMA_SERVER") or "/missing/llama-server"
    return {
        "available": False,
        "selected_path": None,
        "candidates": [
            {
                "source": "env:CARNOT_LLAMA_SERVER",
                "path": path,
                "exists": False,
                "is_file": False,
                "executable": False,
            }
        ],
        "missing_diagnostic": f"llama-server binary not found or not executable: {path}",
    }


def _server_available(path: Path) -> exp.JsonDict:
    return {
        "available": True,
        "selected_path": str(path),
        "candidates": [
            {
                "source": "test",
                "path": str(path),
                "exists": True,
                "is_file": True,
                "executable": True,
            }
        ],
        "missing_diagnostic": None,
    }


def _sample_with_topk(endpoint: str, timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "ready": True,
        "route": f"{endpoint}/completion",
        "status": 200,
        "completion_text": "exp3013 endpoint live",
        "logprob_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "evidence": {
            "token_logprob_count": 2,
            "top_logprob_row_count": 2,
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [{" exp": -0.1, " run": -1.2}, {" live": -0.2, " cached": -1.4}],
            "raw_response_keys": ["content", "completion_probabilities"],
        },
        "error": None,
    }


def _raw_with_topk(text: str = " exp3013 live") -> dict[str, Any]:
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": [" exp", "3013", " live"],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                    "top_logprobs": [
                        {" exp": -0.1, " run": -1.5},
                        {"3013": -0.2, "3001": -1.7},
                        {" live": -0.3, " cached": -2.2},
                    ],
                },
            }
        ],
        "usage": {"completion_tokens": 3},
    }


def test_req_infer_sota_021_spec_anchor_exists() -> None:
    """REQ-INFER-SOTA-021: Exp 3013 is anchored in OpenSpec before implementation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-INFER-SOTA-021" in spec
    assert "SCENARIO-INFER-SOTA-021-001" in spec
    assert "SCENARIO-INFER-SOTA-021-002" in spec
    assert "SCENARIO-INFER-SOTA-021-003" in spec
    assert "SCENARIO-INFER-SOTA-021-004" in spec
    assert "SCENARIO-INFER-SOTA-021-005" in spec
    assert exp.ARTIFACT_FILENAME in spec


def test_scenario_021_004_cached_models_without_endpoint_are_partial_not_live(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-004: cached GGUFs do not imply live endpoint readiness."""
    env = _write_all_cached_models(tmp_path)

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [
                {
                    "endpoint": endpoints[0],
                    "completion_probe": {
                        "ready": False,
                        "status": None,
                        "detail": "connection refused",
                    },
                    "telemetry_probe": {
                        "ready": False,
                        "status": None,
                        "detail": "skipped: completion probe failed",
                    },
                }
            ],
        },
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected direct load: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([10.0, 11.0]).__next__,
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert (
        artifact["honest_verdict"]
        == "blocked_llamacpp_logprob_endpoint_bringup_no_binary_or_runtime"
    )
    assert artifact["sota_models_ready"] is True
    assert artifact["completion_endpoint_ready"] is False
    assert artifact["logprob_endpoint_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    assert artifact["tool_first_verifier_ready"] is True
    assert artifact["live_completion_invoked"] is False
    assert artifact["inference_substrate"] != exp.LIVE_LLM_SUBSTRATE
    assert artifact["flagged_adversarial"] is False
    assert artifact["server_command"] is None
    assert artifact["endpoint_url"] == "http://127.0.0.1:45555"
    assert artifact["sample_completion"] is None
    assert artifact["sample_logprob_evidence"]["ready"] is False
    assert artifact["blocker_root_cause"]["kind"] == "llama_server_binary_unavailable"
    assert artifact["preconditions_checked"]["free_port"]["available"] is True
    assert artifact["preconditions_checked"]["llama_cpp_server"]["available"] is False
    assert artifact["preconditions_checked"]["resolved_local_gguf_paths"][QWEN].endswith(".gguf")
    assert "endpoint_completion_unavailable" in artifact["skip_reasons"]
    assert "live_completion_not_invoked" in artifact["skip_reasons"]
    assert [row["hf_id"] for row in artifact["usable_sota_models"]] == [QWEN, GEMMA31, GEMMA26]
    assert all(row["model_path"].endswith(".gguf") for row in artifact["usable_sota_models"])
    assert artifact["model_specs"]["headline_models"] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["model_specs"]["resolved_models"]["flagship_moe"]["resolved_path"]
    assert artifact["model_specs"]["resolved_models"]["flagship_dense"]["resolved_path"]
    assert artifact["model_specs"]["resolved_models"]["middle_moe"]["resolved_path"]
    assert artifact["model_specs"]["resolved_models"]["flagship_moe"]["resolved_path"].endswith(
        ".gguf"
    )


def test_scenario_021_004_endpoint_completion_with_toplogprobs_is_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-004: live endpoint top-logprobs open the runtime path."""
    env = _write_all_cached_models(tmp_path)

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0],
            "completion_ready": True,
            "top_logprob_ready": True,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs",
            "duration_s": 65.0,
            "probes": [
                {
                    "endpoint": endpoints[0],
                    "completion_probe": {
                        "ready": True,
                        "status": 200,
                        "detail": "completion returned non-empty content",
                        "route": f"{endpoints[0]}/completion",
                    },
                    "telemetry_probe": {
                        "ready": True,
                        "status": 200,
                        "detail": "top-logprob telemetry present",
                        "signal": "top_logprobs",
                        "route": f"{endpoints[0]}/completion",
                    },
                }
            ],
        },
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected direct load: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        endpoint_sample_fn=_sample_with_topk,
        monotonic=iter([1.0, 66.0]).__next__,
    )

    assert artifact["honest_verdict"] == "success_llamacpp_logprob_endpoint_ready"
    assert artifact["sota_headline_ready"] is True
    assert artifact["sota_logprob_ready"] is True
    assert artifact["completion_endpoint_ready"] is True
    assert artifact["logprob_endpoint_ready"] is True
    assert artifact["top_logprob_or_confidence_ready"] is True
    assert artifact["live_completion_invoked"] is True
    assert artifact["inference_substrate"] == exp.LIVE_LLM_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert artifact["endpoint_summary"]["duration_s"] == pytest.approx(65.0)
    assert artifact["flagged_adversarial"] is False
    assert artifact["skip_reasons"] == []
    assert artifact["sample_completion"]["text"] == "exp3013 endpoint live"
    assert artifact["sample_logprob_evidence"]["token_logprob_count"] == 2


def test_scenario_021_005_bringup_starts_smallest_cached_sota_and_cleans_up(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-005: bring-up records command, PID, sample, and cleanup."""
    env = _write_all_cached_models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    start_calls: list[dict[str, Any]] = []
    cleanup_calls: list[Any] = []
    probe_calls: list[list[str]] = []

    class FakeProcess:
        pid = 4242

        def poll(self) -> None:
            return None

    def endpoint_probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
        probe_calls.append(list(endpoints))
        ready = endpoints == ["http://127.0.0.1:45678"]
        return {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0] if ready else None,
            "completion_ready": ready,
            "top_logprob_ready": ready,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs" if ready else None,
            "duration_s": timeout_s,
            "probes": [],
        }

    def start_server(command: list[str], server_env: dict[str, str], log_path: Path) -> FakeProcess:
        start_calls.append({"command": command, "env": server_env, "log_path": log_path})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("server ready\n", encoding="utf-8")
        return FakeProcess()

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=endpoint_probe,
        endpoint_sample_fn=_sample_with_topk,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected direct load: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=lambda runtime_env: _server_available(server),
        free_port_fn=lambda host: _free_port(45678),
        server_start_fn=start_server,
        server_cleanup_fn=lambda process: cleanup_calls.append(process) or {"terminated": True},
        monotonic=iter([1.0, 66.0]).__next__,
    )

    assert artifact["honest_verdict"] == "success_llamacpp_logprob_endpoint_ready"
    assert artifact["server_pid"] == 4242
    assert artifact["server_command"] == start_calls[0]["command"]
    assert artifact["server_command"][0] == str(server)
    assert artifact["server_command"][1:3] == ["-m", artifact["model_specs"]["resolved_models"]["middle_moe"]["resolved_path"]]
    assert artifact["endpoint_url"] == "http://127.0.0.1:45678"
    assert artifact["sample_completion"]["text"] == "exp3013 endpoint live"
    assert artifact["sample_logprob_evidence"]["top_logprob_row_count"] == 2
    assert artifact["server_logs"]["tail"] == "server ready\n"
    assert artifact["cleanup_behavior"] == {"terminated": True}
    assert cleanup_calls and cleanup_calls[0].pid == 4242
    assert probe_calls[0] == ["http://127.0.0.1:8080"]
    assert probe_calls[1] == ["http://127.0.0.1:45678"]
    assert artifact["blocker_root_cause"] is None
    assert artifact["live_completion_invoked"] is True


def test_scenario_021_001_live_transcript_with_topk_opens_both_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-001: transcript plus top-k telemetry opens both gates."""
    gguf, env = _write_cached_model(tmp_path)
    call_order: list[str] = []

    def command_runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
        call_order.append("precondition")
        return _runner()(command, **kwargs)

    def prompt_runner(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        call_order.append("prompt")
        telemetry = exp._extract_loader_telemetry(_raw_with_topk())
        return {
            "attempted": True,
            "load_status": "loaded",
            "generation_status": "generated",
            "usable": True,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "prompt": exp.DEFAULT_PROMPT,
            "response_text": "exp3013 live",
            "tokens_generated": 3,
            "duration_s": 1.5,
            "inference_substrate": "llama_cpp_gpu",
            "raw_response": _raw_with_topk(),
            **telemetry,
        }

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=command_runner,
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        prompt_runner_fn=prompt_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([10.0, 12.75]).__next__,
        tests_run=("focused-exp3013",),
        direct_load_enabled=True,
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["artifact"] == exp.ARTIFACT_NAME
    assert artifact["sota_headline_ready"] is True
    assert artifact["sota_logprob_ready"] is True
    assert (
        artifact["honest_verdict"]
        == "blocked_llamacpp_logprob_endpoint_bringup_no_binary_or_runtime"
    )
    assert artifact["preconditions_checked"]["recorded_before_model_load"] is True
    assert artifact["model_specs"]["experiment_id"] == 3013
    assert artifact["model_specs"]["headline_models"] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["completion_endpoint_ready"] is False
    assert artifact["live_completion_invoked"] is False
    assert artifact["headline_models_available"] == [
        {
            "hf_id": QWEN,
            "path": str(gguf),
            "status": "cache_resolved",
            "generated": True,
        }
    ]
    assert artifact["cache_paths"]["headline_models"][QWEN] == str(gguf)
    assert artifact["model_checksums"][QWEN]["sha256"]
    assert artifact["precondition_evidence"]["checksum_feasibility"]["feasible"] is True
    attempt = artifact["headline_models_attempted"][0]
    assert attempt["load_status"] == "loaded"
    assert attempt["generation_status"] == "generated"
    assert attempt["token_logprobs_exposed"] is True
    assert attempt["topk_logprobs_exposed"] is True
    assert attempt["logits_exposed"] is False
    assert attempt["transcript_sha256"]
    assert artifact["telemetry_capabilities"]["overall"]["token_logprobs_exposed"] is True
    assert artifact["telemetry_capabilities"]["overall"]["topk_logprobs_exposed"] is True
    assert artifact["telemetry_capabilities"]["blockers"] == []
    transcript = Path(artifact["live_transcript_paths"][0])
    assert transcript.is_file()
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert payload["response_text"] == "exp3013 live"
    assert payload["telemetry"]["topk_logprobs_exposed"] is True
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["inference_substrate"] == "llama_cpp_gpu"
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert artifact["tests_run"] == ["focused-exp3013"]
    assert call_order.index("precondition") < call_order.index("prompt")


def test_scenario_021_002_transcript_without_telemetry_keeps_logprob_gate_false(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-002: text alone cannot become logprob readiness."""
    _gguf, env = _write_cached_model(tmp_path, GEMMA31)

    def prompt_runner(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        telemetry = exp._extract_loader_telemetry({"choices": [{"text": "text", "logprobs": None}]})
        return {
            "attempted": True,
            "load_status": "loaded",
            "generation_status": "generated",
            "usable": True,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "prompt": exp.DEFAULT_PROMPT,
            "response_text": "headline transcript only",
            "tokens_generated": 3,
            "duration_s": 0.5,
            "inference_substrate": "llama_cpp_gpu",
            "raw_response": {"choices": [{"text": "headline transcript only", "logprobs": None}]},
            **telemetry,
        }

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        prompt_runner_fn=prompt_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": GEMMA31}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 2.0]).__next__,
        direct_load_enabled=True,
    )

    assert artifact["sota_headline_ready"] is True
    assert artifact["sota_logprob_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_llamacpp_logprob_endpoint_bringup_")
    assert artifact["telemetry_capabilities"]["overall"]["any_live_generation"] is True
    assert artifact["telemetry_capabilities"]["overall"]["token_logprobs_exposed"] is False
    assert artifact["telemetry_capabilities"]["overall"]["topk_logprobs_exposed"] is False
    assert "token_logprobs_unavailable" in artifact["telemetry_capabilities"]["blockers"]
    assert "topk_logprobs_unavailable" in artifact["telemetry_capabilities"]["blockers"]
    gemma_attempt = next(row for row in artifact["headline_models_attempted"] if row["hf_id"] == GEMMA31)
    assert gemma_attempt["transcript_path"]
    assert gemma_attempt["telemetry_blockers"] == [
        "token_logprobs_unavailable",
        "topk_logprobs_unavailable",
        "logits_unavailable",
    ]


def test_scenario_021_003_missing_headline_blocks_without_legacy_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-021-003: no headline run writes blocked SOTA verdict."""
    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([3.0, 3.4]).__next__,
    )

    assert artifact["sota_headline_ready"] is False
    assert artifact["sota_logprob_ready"] is False
    assert artifact["honest_verdict"] == "blocked_llamacpp_logprob_endpoint_bringup_no_usable_sota_model"
    assert artifact["headline_models_available"] == []
    assert artifact["live_transcript_paths"] == []
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["legacy_smoke_context"]["smoke_only"] is False
    assert artifact["legacy_smoke_context"]["used_for_headline_readiness"] is False
    assert [row["hf_id"] for row in artifact["headline_models_attempted"]] == [
        QWEN,
        GEMMA31,
        GEMMA26,
    ]
    assert {row["cache_status"] for row in artifact["headline_models_attempted"]} == {"missing"}
    assert artifact["inference_substrate"] == "blocked_no_headline_cache"
    assert artifact["telemetry_capabilities"]["overall"]["any_live_generation"] is False
    assert artifact["live_completion_invoked"] is False


def test_req_021_runtime_precondition_failure_skips_large_load_when_direct_enabled(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-021: failed runtime preconditions are recorded before load."""
    _gguf, env = _write_cached_model(tmp_path, GEMMA26)

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(llama_gpu=False),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": GEMMA26}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([7.0, 8.0]).__next__,
        direct_load_enabled=True,
    )

    gemma_attempt = next(row for row in artifact["headline_models_attempted"] if row["hf_id"] == GEMMA26)
    assert artifact["honest_verdict"].startswith("blocked_llamacpp_logprob_endpoint_bringup_")
    assert gemma_attempt["cache_status"] == "resolved"
    assert gemma_attempt["load_status"] == "not_attempted_runtime_precondition_failed"
    assert gemma_attempt["generation_status"] == "not_attempted"
    assert artifact["precondition_evidence"]["llama_cpp"]["llama_cpp_supports_gpu_offload"] is False
    assert artifact["telemetry_capabilities"]["blockers"] == ["no_live_headline_generation"]
    assert artifact["live_completion_invoked"] is False


def test_req_021_helpers_prompt_parser_writer_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-021: helpers, prompt parser, writer, and CLI preserve contract."""
    assert exp._model_specs()["experiment_id"] == 3013
    assert exp._telemetry_blockers(
        {
            "token_logprobs_exposed": True,
            "topk_logprobs_exposed": False,
            "logits_exposed": False,
        }
    ) == ["topk_logprobs_unavailable", "logits_unavailable"]
    assert exp._extract_loader_telemetry({"choices": []})["telemetry_blockers"] == [
        "token_logprobs_unavailable",
        "topk_logprobs_unavailable",
        "logits_unavailable",
    ]
    assert exp._finite_float("not-a-number") is None
    assert exp._extract_loader_telemetry({"choices": ["not-a-dict"]})[
        "telemetry_blockers"
    ] == [
        "token_logprobs_unavailable",
        "topk_logprobs_unavailable",
        "logits_unavailable",
    ]
    odd = exp._extract_loader_telemetry(
        {
            "choices": [
                {
                    "text": "x",
                    "logprobs": {
                        "tokens": ["x"],
                        "token_logprobs": [None, True, "-0.7"],
                        "top_logprobs": [None, {"x": "-0.7", "y": False, "z": -2.0}],
                    },
                    "logits": [0.1, 0.2],
                }
            ]
        }
    )
    assert odd["token_logprobs_exposed"] is True
    assert odd["topk_logprobs_exposed"] is True
    assert odd["logits_exposed"] is True
    assert odd["telemetry_observation"]["token_logprob_count"] == 1

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
                    "duration_s": 0.5,
                    "inference_substrate": "llama_cpp_gpu",
                    "requested_gpu": 2,
                    "main_gpu": 0,
                    "loader_logits_available": True,
                    "loader_logits_shape": [3, 8],
                    "raw_response": _raw_with_topk("ok"),
                },
                sort_keys=True,
            )
            + "\n",
        ),
        env={},
        timeout_s=2,
    )
    assert parsed["usable"] is True
    assert parsed["topk_logprobs_exposed"] is True
    assert parsed["logits_exposed"] is True
    assert parsed["telemetry_observation"]["logits_shape"] == [3, 8]
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
    assert failed["telemetry_blockers"] == [
        "token_logprobs_unavailable",
        "topk_logprobs_unavailable",
        "logits_unavailable",
    ]

    output = tmp_path / "results" / exp.ARTIFACT_FILENAME
    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
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
            "duration_s": 0.5,
            "inference_substrate": "llama_cpp_gpu",
            "raw_response": _raw_with_topk("ok"),
            **exp._extract_loader_telemetry(_raw_with_topk("ok")),
        },
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 1.2]).__next__,
        tests_run=("coverage",),
        direct_load_enabled=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact"] == exp.ARTIFACT_NAME
    assert exp._inference_substrate(
        ready=False,
        cached_count=1,
        attempted_live=True,
        generated_substrates=[],
    ) == "llama_cpp_failed"

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(exp, "run_experiment", fake_run_experiment)
    assert exp.main(
        ["--output", str(tmp_path / "out.json"), "--selected-python", SELECTED_PYTHON, "--test-run", "unit"]
    ) == 0
    assert exp.main(
        ["--output", str(tmp_path / "out2.json"), "--selected-python", SELECTED_PYTHON, "--prompt-timeout-s", "7"]
    ) == 0
    assert exp.main(
        [
            "--output",
            str(tmp_path / "out3.json"),
            "--selected-python",
            SELECTED_PYTHON,
            "--endpoint",
            "http://127.0.0.1:45555",
            "--endpoint-timeout-s",
            "3.5",
            "--direct-load",
        ]
    ) == 0
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
        {
            "output_path": tmp_path / "out3.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": [],
            "endpoints": ["http://127.0.0.1:45555"],
            "endpoint_timeout_s": 3.5,
            "direct_load_enabled": True,
        },
    ]


def test_scenario_021_005_endpoint_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFER-SOTA-021-005: endpoint helper edges stay deterministic."""
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    monkeypatch.setattr(exp.shutil, "which", lambda name: str(server) if name == "llama-server" else None)

    endpoints = exp._default_endpoint_list({"CARNOT_5085_ENDPOINTS": " http://a:1/, http://b:2 "})
    assert endpoints == ["http://a:1", "http://b:2"]
    assert exp._normalize_endpoints(["http://a:1/", "http://a:1", "http://b:2/"]) == [
        "http://a:1",
        "http://b:2",
    ]
    monkeypatch.setattr(
        exp,
        "_probe_llama_cpp_endpoints",
        lambda endpoints, timeout_s: {"completion_ready": True},
    )
    assert exp._probe_endpoint_summary(["http://a:1"], 1.0)["completion_ready"] is True
    assert exp._find_free_port()["available"] is True
    assert exp._find_free_port("256.256.256.256")["available"] is False

    availability = exp._llama_server_availability(
        {"CARNOT_LLAMA_SERVER": str(server), "LLAMA_SERVER": str(server)}
    )
    assert availability["available"] is True
    assert availability["selected_path"] == str(server)
    assert exp._build_server_command(
        server_path=str(server),
        model_path="/models/small.gguf",
        host="127.0.0.1",
        port=45555,
        extra_args="--parallel 1",
    )[-2:] == ["--parallel", "1"]

    class FakeResponse:
        status = 201

        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    responses = iter([FakeResponse(b'{"ok": true}'), FakeResponse(b"not-json")])
    monkeypatch.setattr(exp, "urlopen", lambda request, timeout: next(responses))
    assert exp._http_post_json("http://endpoint/completion", {"x": 1}, 2.0) == (
        201,
        {"ok": True},
    )
    assert exp._http_post_json("http://endpoint/completion", {"x": 1}, 2.0) == (
        201,
        {"raw": "not-json"},
    )

    http_error = HTTPError("http://endpoint", 500, "bad", {}, None)
    assert "HTTPError 500" in exp._http_error_detail(http_error)
    class BadBody:
        def read(self) -> bytes:
            raise OSError("unreadable")

        def close(self) -> None:
            return None

    unreadable_http_error = HTTPError("http://endpoint", 502, "bad", {}, BadBody())
    assert exp._http_error_detail(unreadable_http_error) == "HTTPError 502"
    assert "URLError" in exp._http_error_detail(URLError("down"))
    assert exp._http_error_detail(RuntimeError("bad")) == "RuntimeError: bad"
    assert exp._response_text([]) == ""
    assert exp._response_text({"choices": [{"message": {"content": " ok "}}]}) == "ok"
    assert exp._response_text({"choices": [{"logprobs": None}]}) == ""
    assert exp._top_logprob_row(
        [{"token": "A", "logprob": "-0.5"}, {"token": "B", "logprob": True}]
    ) == {"A": -0.5}
    assert exp._parse_endpoint_sample_payload([]) == {
        "text": "",
        "token_logprobs": [],
        "top_logprobs": [],
    }

    native = exp._parse_endpoint_sample_payload(
        {
            "content": "native",
            "completion_probabilities": [
                "bad",
                {"logprob": "-0.1", "top_logprobs": [{"token": "native", "logprob": -0.1}]}
            ],
        }
    )
    choices = exp._parse_endpoint_sample_payload(
        {
            "choices": [
                {
                    "text": "choice",
                    "logprobs": {
                        "content": [
                            "bad",
                            {
                                "token": "choice",
                                "logprob": -0.2,
                                "top_logprobs": [{"token": "choice", "logprob": -0.2}],
                            },
                        ],
                        "token_logprobs": ["-0.3"],
                        "top_logprobs": [{"choice": -0.2, "other": -1.4}],
                    },
                }
            ]
        }
    )
    assert native["text"] == "native"
    assert native["token_logprobs"] == [-0.1]
    assert choices["text"] == "choice"
    assert choices["token_logprobs"] == [-0.2, -0.3]
    assert choices["top_logprobs"] == [{"choice": -0.2}, {"choice": -0.2, "other": -1.4}]

    def ok_post(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
        assert url.endswith("/completion")
        assert payload["n_probs"] == exp.LOGPROBS_REQUESTED
        assert timeout_s == pytest.approx(2.0)
        return (
            200,
            {
                "content": "sample",
                "completion_probabilities": [
                    {"logprob": -0.4, "top_logprobs": [{"token": "sample", "logprob": -0.4}]}
                ],
            },
        )

    monkeypatch.setattr(exp, "_http_post_json", ok_post)
    sample = exp._sample_endpoint_telemetry("http://127.0.0.1:45555", 2.0)
    assert sample["ready"] is True
    assert sample["evidence"]["token_logprob_count"] == 1

    monkeypatch.setattr(exp, "_http_post_json", lambda *_args, **_kwargs: (200, {"content": ""}))
    empty = exp._sample_endpoint_telemetry("http://127.0.0.1:45555", 2.0)
    assert empty["ready"] is False
    assert "empty_or_unrecognized_completion" in empty["error"]

    monkeypatch.setattr(exp, "_http_post_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")))
    failed = exp._sample_endpoint_telemetry("http://127.0.0.1:45555", 2.0)
    assert failed["ready"] is False
    assert "URLError" in failed["error"]
    assert exp._tail_file(None) == ""
    assert exp._tail_file(tmp_path / "missing.log") == ""

    log_path = tmp_path / "started.log"
    proc = exp._start_llama_server_process(
        [exp.sys.executable, "-c", "print('hello')"],
        {},
        log_path,
    )
    proc.wait(timeout=5)
    already_done = exp._cleanup_llama_server_process(proc)
    assert already_done["already_exited"] is True
    assert "hello" in log_path.read_text(encoding="utf-8")

    class FakeRunningProcess:
        pid = 999

        def __init__(self) -> None:
            self.terminated = False
            self._carnot_log_handle = (tmp_path / "cleanup.log").open("w", encoding="utf-8")

        def poll(self) -> int | None:
            return 0 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> None:
            assert timeout == 10

    cleanup = exp._cleanup_llama_server_process(FakeRunningProcess())
    assert cleanup["terminated"] is True

    class TimeoutProcess:
        def __init__(self) -> None:
            self.killed = False
            self.wait_count = 0
            self._carnot_log_handle = (tmp_path / "timeout.log").open("w", encoding="utf-8")

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            self.wait_count += 1
            if self.wait_count == 1:
                raise exp.subprocess.TimeoutExpired("cmd", timeout)

        def kill(self) -> None:
            self.killed = True

    timeout_cleanup = exp._cleanup_llama_server_process(TimeoutProcess())
    assert timeout_cleanup["terminated"] is True

    annotated = exp._annotate_model_specs_for_bringup(
        {"resolved_models": {"bad": "not-a-dict"}},
        None,
        {},
    )
    assert annotated == {"resolved_models": {"bad": "not-a-dict"}}
    assert exp._honest_verdict(headline_ready=False, logprob_ready=False).startswith("blocked_")
    assert exp._honest_verdict(headline_ready=True, logprob_ready=True).startswith("complete_")
    assert exp._honest_verdict(headline_ready=True, logprob_ready=False).endswith("partial_ready")
    assert exp._runtime_honest_verdict(
        sota_models_ready=True,
        completion_endpoint_ready=True,
        top_logprob_or_confidence_ready=False,
        tool_first_verifier_ready=True,
    ).endswith("no_logprob_telemetry")
    assert exp._runtime_honest_verdict(
        sota_models_ready=True,
        completion_endpoint_ready=False,
        top_logprob_or_confidence_ready=False,
        tool_first_verifier_ready=True,
        blocker_root_cause={"kind": "no_usable_sota_model"},
    ).endswith("no_usable_sota_model")
    assert "tool_first_verifier_unavailable" in exp._runtime_skip_reasons(
        sota_models_ready=True,
        completion_endpoint_ready=True,
        logprob_endpoint_ready=True,
        top_logprob_or_confidence_ready=True,
        tool_first_verifier_ready=False,
        live_completion_invoked=True,
    )


def test_scenario_021_005_bringup_blocker_edges(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-021-005: bring-up blocker branches preserve evidence."""
    usable = [{"role": "middle_moe", "hf_id": GEMMA26, "model_path": "/models/gemma26.gguf"}]
    checksums = {GEMMA26: {"size_bytes": 10}}
    unavailable_port = {
        "available": False,
        "host": "127.0.0.1",
        "port": None,
        "endpoint_url": None,
        "error": "bind failed",
    }
    server = _server_available(tmp_path / "llama-server")

    no_port = exp._attempt_llama_server_bringup(
        project_root=tmp_path,
        env={},
        usable_sota_models=usable,
        model_checksums=checksums,
        endpoint_probe_fn=lambda endpoints, timeout_s: pytest.fail("probe should not run"),
        endpoint_sample_fn=lambda endpoint, timeout_s: pytest.fail("sample should not run"),
        endpoint_timeout_s=2.0,
        server_finder_fn=lambda env: server,
        free_port_fn=lambda host: unavailable_port,
        server_start_fn=lambda command, env, log_path: pytest.fail("server should not start"),
        server_cleanup_fn=lambda process: pytest.fail("cleanup should not run"),
        sleep_fn=lambda seconds: pytest.fail("sleep should not run"),
        start_timeout_s=0.0,
    )
    assert no_port["blocker_root_cause"]["kind"] == "free_port_unavailable"

    start_failed = exp._attempt_llama_server_bringup(
        project_root=tmp_path,
        env={},
        usable_sota_models=usable,
        model_checksums=checksums,
        endpoint_probe_fn=lambda endpoints, timeout_s: {"completion_ready": False},
        endpoint_sample_fn=lambda endpoint, timeout_s: pytest.fail("sample should not run"),
        endpoint_timeout_s=2.0,
        server_finder_fn=lambda env: server,
        free_port_fn=lambda host: _free_port(45679),
        server_start_fn=lambda command, env, log_path: (_ for _ in ()).throw(RuntimeError("boom")),
        server_cleanup_fn=lambda process: pytest.fail("cleanup should not run"),
        sleep_fn=lambda seconds: None,
        start_timeout_s=0.0,
    )
    assert start_failed["blocker_root_cause"]["kind"] == "server_start_failed"

    class FakeProcess:
        pid = 101

        def poll(self) -> None:
            return None

    cleanup_calls: list[Any] = []
    not_ready = exp._attempt_llama_server_bringup(
        project_root=tmp_path,
        env={},
        usable_sota_models=usable,
        model_checksums=checksums,
        endpoint_probe_fn=lambda endpoints, timeout_s: {"completion_ready": False},
        endpoint_sample_fn=lambda endpoint, timeout_s: pytest.fail("sample should not run"),
        endpoint_timeout_s=2.0,
        server_finder_fn=lambda env: server,
        free_port_fn=lambda host: _free_port(45680),
        server_start_fn=lambda command, env, log_path: FakeProcess(),
        server_cleanup_fn=lambda process: cleanup_calls.append(process) or {"terminated": True},
        sleep_fn=lambda seconds: None,
        start_timeout_s=0.0,
    )
    assert not_ready["server_pid"] == 101
    assert not_ready["blocker_root_cause"]["kind"] == "server_started_but_endpoint_not_ready"
    assert cleanup_calls and cleanup_calls[0].pid == 101

    class ExitedProcess:
        pid = 202

        def poll(self) -> int:
            return 1

    exited = exp._attempt_llama_server_bringup(
        project_root=tmp_path,
        env={},
        usable_sota_models=usable,
        model_checksums=checksums,
        endpoint_probe_fn=lambda endpoints, timeout_s: {"completion_ready": False},
        endpoint_sample_fn=lambda endpoint, timeout_s: pytest.fail("sample should not run"),
        endpoint_timeout_s=2.0,
        server_finder_fn=lambda env: server,
        free_port_fn=lambda host: _free_port(45681),
        server_start_fn=lambda command, env, log_path: ExitedProcess(),
        server_cleanup_fn=lambda process: {"terminated": False, "returncode": process.poll()},
        sleep_fn=lambda seconds: pytest.fail("sleep should not run"),
        start_timeout_s=1.0,
    )
    assert exited["server_pid"] == 202
    assert exited["blocker_root_cause"]["kind"] == "server_started_but_endpoint_not_ready"

    monotonic_values = iter([0.0, 0.0, 0.5, 2.0])
    sleep_calls: list[float] = []
    original_monotonic = exp.time.monotonic
    exp.time.monotonic = lambda: next(monotonic_values)
    try:
        waited = exp._attempt_llama_server_bringup(
            project_root=tmp_path,
            env={},
            usable_sota_models=usable,
            model_checksums=checksums,
            endpoint_probe_fn=lambda endpoints, timeout_s: {"completion_ready": False},
            endpoint_sample_fn=lambda endpoint, timeout_s: pytest.fail("sample should not run"),
            endpoint_timeout_s=2.0,
            server_finder_fn=lambda env: server,
            free_port_fn=lambda host: _free_port(45682),
            server_start_fn=lambda command, env, log_path: FakeProcess(),
            server_cleanup_fn=lambda process: {"terminated": True},
            sleep_fn=lambda seconds: sleep_calls.append(seconds),
            start_timeout_s=1.0,
        )
    finally:
        exp.time.monotonic = original_monotonic
    assert waited["blocker_root_cause"]["kind"] == "server_started_but_endpoint_not_ready"
    assert sleep_calls == [0.5]


def test_scenario_021_005_final_artifact_blocker_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFER-SOTA-021-005: final artifact blocker branches are explicit."""
    no_cache = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf"),
            "CARNOT_LLAMA_SERVER_START_TIMEOUT_S": "not-a-number",
        },
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0],
            "completion_ready": True,
            "top_logprob_ready": True,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs",
            "duration_s": timeout_s,
            "probes": [],
        },
        endpoint_sample_fn=_sample_with_topk,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 66.0]).__next__,
    )
    assert no_cache["blocker_root_cause"]["kind"] == "no_usable_sota_model"
    assert no_cache["honest_verdict"].endswith("no_usable_sota_model")

    env_for_sample_fail = _write_all_cached_models(tmp_path / "sample-fail")
    sample_failed = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env_for_sample_fail,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0],
            "completion_ready": True,
            "top_logprob_ready": True,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs",
            "duration_s": timeout_s,
            "probes": [],
        },
        endpoint_sample_fn=lambda endpoint, timeout_s: exp._bringup_blocked_sample("sample failed"),
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert sample_failed["blocker_root_cause"]["kind"] == "endpoint_sample_failed"

    env_for_completion_fail = _write_all_cached_models(tmp_path / "completion-fail")
    monkeypatch.setattr(
        exp,
        "_attempt_llama_server_bringup",
        lambda **kwargs: {
            "selected_model": None,
            "server_command": None,
            "server_pid": None,
            "endpoint_url": "http://127.0.0.1:45555",
            "sample": exp._bringup_blocked_sample("no completion"),
            "server_logs": {"path": None, "tail": "", "exists": False},
            "cleanup_behavior": {"started_by_preflight": False, "terminated": False},
            "blocker_root_cause": None,
            "endpoint_summary": {"completion_ready": False},
        },
    )
    completion_failed = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env_for_completion_fail,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": None,
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert completion_failed["blocker_root_cause"]["kind"] == "completion_endpoint_unavailable"

    text_no_probs = {
        "ready": True,
        "route": "http://127.0.0.1:8080/completion",
        "status": 200,
        "completion_text": "text",
        "logprob_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "evidence": {
            "token_logprob_count": 0,
            "top_logprob_row_count": 0,
            "token_logprobs": [],
            "top_logprobs": [],
        },
        "error": None,
    }
    env = _write_all_cached_models(tmp_path / "telemetry")
    no_telemetry = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0],
            "completion_ready": True,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "duration_s": timeout_s,
            "probes": [],
        },
        endpoint_sample_fn=lambda endpoint, timeout_s: text_no_probs,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 66.0]).__next__,
    )
    assert no_telemetry["blocker_root_cause"]["kind"] == "logprob_telemetry_unavailable"
    assert no_telemetry["honest_verdict"].endswith("no_logprob_telemetry")

    flagged = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        endpoint_probe_fn=lambda endpoints, timeout_s: {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0],
            "completion_ready": True,
            "top_logprob_ready": True,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs",
            "duration_s": timeout_s,
            "probes": [],
        },
        endpoint_sample_fn=_sample_with_topk,
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [{"hf_id": QWEN}],
        llama_server_finder_fn=_server_unavailable,
        free_port_fn=lambda host: _free_port(),
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert flagged["flagged_adversarial"] is True
    assert flagged["corrigendum_pending"][0]["kind"] == "DURATION_TOO_SHORT"
