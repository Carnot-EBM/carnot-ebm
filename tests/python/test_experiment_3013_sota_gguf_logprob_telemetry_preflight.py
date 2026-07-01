"""Tests for Exp 3013 SOTA GGUF logprob telemetry preflight.

Spec: REQ-INFER-SOTA-021,
      SCENARIO-INFER-SOTA-021-001,
      SCENARIO-INFER-SOTA-021-002,
      SCENARIO-INFER-SOTA-021-003,
      SCENARIO-INFER-SOTA-021-004
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    gguf.write_text("tiny gguf fixture\n", encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def _write_all_cached_models(tmp_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for hf_id in (QWEN, GEMMA31, GEMMA26):
        _gguf, env = _write_cached_model(tmp_path, hf_id)
    return env


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
        monotonic=iter([10.0, 11.0]).__next__,
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "complete_gguf_logprob_preflight_partial_ready"
    assert artifact["sota_models_ready"] is True
    assert artifact["completion_endpoint_ready"] is False
    assert artifact["logprob_endpoint_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    assert artifact["tool_first_verifier_ready"] is True
    assert artifact["live_completion_invoked"] is False
    assert artifact["inference_substrate"] != exp.LIVE_LLM_SUBSTRATE
    assert artifact["flagged_adversarial"] is False
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
        monotonic=iter([1.0, 66.0]).__next__,
    )

    assert artifact["honest_verdict"] == "complete_gguf_logprob_preflight_ready"
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
        monotonic=iter([10.0, 12.75]).__next__,
        tests_run=("focused-exp3013",),
        direct_load_enabled=True,
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["artifact"] == exp.ARTIFACT_NAME
    assert artifact["sota_headline_ready"] is True
    assert artifact["sota_logprob_ready"] is True
    assert artifact["honest_verdict"] == "complete_gguf_logprob_preflight_partial_ready"
    assert artifact["preconditions_checked"] is True
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
        monotonic=iter([1.0, 2.0]).__next__,
        direct_load_enabled=True,
    )

    assert artifact["sota_headline_ready"] is True
    assert artifact["sota_logprob_ready"] is False
    assert artifact["honest_verdict"] == "complete_gguf_logprob_preflight_partial_ready"
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
        monotonic=iter([3.0, 3.4]).__next__,
    )

    assert artifact["sota_headline_ready"] is False
    assert artifact["sota_logprob_ready"] is False
    assert artifact["honest_verdict"] == "blocked_gguf_logprob_preflight_no_ready_paths"
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
        monotonic=iter([7.0, 8.0]).__next__,
        direct_load_enabled=True,
    )

    gemma_attempt = next(row for row in artifact["headline_models_attempted"] if row["hf_id"] == GEMMA26)
    assert artifact["honest_verdict"] == "complete_gguf_logprob_preflight_partial_ready"
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
