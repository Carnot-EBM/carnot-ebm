"""Tests for Exp 2989 SOTA GGUF cache provenance preflight.

Spec: REQ-INFER-SOTA-019,
      SCENARIO-INFER-SOTA-019-001,
      SCENARIO-INFER-SOTA-019-002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_2989_sota_gguf_cache_provenance_preflight_v1 as exp


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
    def fake(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None) -> dict[str, Any]:
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


def test_req_infer_sota_019_spec_anchor_exists() -> None:
    """REQ-INFER-SOTA-019: Exp 2989 is anchored in OpenSpec before implementation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-INFER-SOTA-019" in spec
    assert "SCENARIO-INFER-SOTA-019-001" in spec
    assert "SCENARIO-INFER-SOTA-019-002" in spec
    assert exp.ARTIFACT_FILENAME in spec


def test_scenario_019_001_available_headline_writes_transcript(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-019-001: a headline GGUF transcript opens the gate."""
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
            "response_text": "cache preflight OK",
            "tokens_generated": 3,
            "duration_seconds": 1.25,
            "inference_substrate": "llama_cpp_gpu",
        }

    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=command_runner,
        prompt_runner_fn=prompt_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([10.0, 12.5]).__next__,
        tests_run=("focused-exp2989",),
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["sota_headline_ready"] is True
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["preconditions_checked"] is True
    assert artifact["sota_models_available"] == [
        {"hf_id": QWEN, "path": str(gguf), "status": "cache_resolved"}
    ]
    assert artifact["cache_paths"]["headline_models"][QWEN] == str(gguf)
    assert artifact["model_checksums"][QWEN]["sha256"]
    assert artifact["sota_models_attempted"][0]["load_status"] == "loaded"
    assert artifact["sota_models_attempted"][0]["generation_status"] == "generated"
    assert artifact["sota_models_attempted"][0]["transcript_sha256"]
    assert artifact["sota_models_attempted"][1]["cache_status"] == "missing"
    assert artifact["live_transcript_paths"]
    transcript = Path(artifact["live_transcript_paths"][0])
    assert transcript.is_file()
    assert json.loads(transcript.read_text(encoding="utf-8"))["response_text"] == "cache preflight OK"
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["duration_seconds"] == pytest.approx(2.5)
    assert call_order.index("precondition") < call_order.index("prompt")


def test_scenario_019_002_missing_cache_blocks_without_legacy_promotion(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-019-002: missing headline cache is terminally blocked."""
    artifact = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_runner(),
        prompt_runner_fn=lambda model, **_: pytest.fail(f"unexpected prompt: {model}"),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([3.0, 3.4]).__next__,
        run_legacy_smoke=True,
    )

    assert artifact["sota_headline_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_model_cache")
    assert artifact["sota_models_available"] == []
    assert artifact["live_transcript_paths"] == []
    assert artifact["legacy_smoke_only_used"] is True
    assert artifact["legacy_smoke_context"]["smoke_only"] is True
    assert artifact["legacy_smoke_context"]["used_for_headline_readiness"] is False
    assert [row["hf_id"] for row in artifact["sota_models_attempted"]] == [QWEN, GEMMA31, GEMMA26]
    assert {row["cache_status"] for row in artifact["sota_models_attempted"]} == {"missing"}
    assert artifact["inference_substrate"] == "blocked_no_headline_cache"


def test_req_019_runtime_precondition_failure_skips_large_load(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-019: failed runtime preconditions are recorded before any load."""
    _gguf, env = _write_cached_model(tmp_path, GEMMA31)

    artifact = exp.build_preflight_artifact(
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


def test_req_019_helpers_and_cli_write_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-019: helpers, writer, prompt parser, and CLI keep the contract stable."""
    selected = tmp_path / ".venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert exp._selected_python(tmp_path) == str(selected)
    assert exp._selected_python(tmp_path / "missing") == sys.executable
    assert exp._summarize("x" * 5, limit=10) == "x" * 5
    assert exp._summarize("x" * 12, limit=10).endswith("<truncated>")
    assert exp._stdout({"stdout": "ok"}) == "ok"
    assert exp._stderr({"stderr": "bad"}) == "bad"
    assert exp._quantization_suffix("/tmp/model-Q4_K_M.gguf") == "Q4_K_M"
    assert exp._quantization_suffix("/tmp/model-random.gguf") == "unknown"
    assert exp._model_filename_token(QWEN) == "qwen3.6-35b-a3b"
    assert exp._safe_model_slug("unsloth/Qwen3.6-35B-A3B-GGUF") == "unsloth_Qwen3_6-35B-A3B-GGUF"
    assert exp._cache_roots(tmp_path, {"HF_HOME": str(tmp_path / "hfhome")})[
        "huggingface_hub_cache"
    ].endswith("hfhome/hub")
    assert ".cache/huggingface/hub" in exp._cache_roots(tmp_path, {})[
        "huggingface_hub_cache"
    ]

    missing_cmd = exp._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert missing_cmd["returncode"] is None
    assert "FileNotFoundError" in missing_cmd["stderr_summary"]
    ok_cmd = exp._run_command(["printf", "ok"], timeout_s=1)
    assert ok_cmd["stdout"] == "ok"

    gguf, env = _write_cached_model(tmp_path)
    records = exp._candidate_records(
        tmp_path,
        exp._cache_roots(tmp_path, env),
        QWEN,
    )
    assert exp._select_candidate(records)["path"] == str(gguf)
    project_dir = tmp_path / "models" / "Qwen3.6-35B-A3B-GGUF"
    project_dir.mkdir(parents=True)
    project_model = project_dir / "Qwen3.6-35B-A3B-noquant.gguf"
    project_model.write_text("project model", encoding="utf-8")
    project_records = exp._candidate_records(tmp_path, exp._cache_roots(tmp_path, {}), QWEN)
    assert any(record["source"] == "project_models" for record in project_records)
    assert exp._select_candidate(
        [{"path": "Qwen3.6-35B-A3B-noquant.gguf", "usable_candidate": True, "size_bytes": 1}]
    )["path"] == "Qwen3.6-35B-A3B-noquant.gguf"
    assert exp._file_evidence(gguf)["sha256"]
    assert exp._file_evidence(None)["status"] == "missing"
    assert exp._file_evidence(tmp_path / "missing.gguf")["status"] == "missing"
    tiny_bounded = tmp_path / "tiny-bounded.gguf"
    tiny_bounded.write_bytes(b"abc")
    assert exp._file_evidence(tiny_bounded, full_sha_max_bytes=1)["bounded_sha256"]
    assert exp._loadable_pair([{"hf_id": QWEN, "model_path": str(gguf)}, {"hf_id": GEMMA31, "model_path": "x"}])
    assert not exp._loadable_pair([{"hf_id": QWEN}])
    assert exp._exercise_cached_sota_pair(lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))[
        "error"
    ] == "RuntimeError: boom"
    bad_llama = exp._llama_cpp_probe(
        SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, stdout="", stderr="bad llama"),
        env={},
    )
    assert bad_llama["llama_cpp_import_ok"] is False
    assert bad_llama["error"] == "bad llama"

    parsed = exp._run_bounded_headline_prompt(
        {"hf_id": QWEN, "path": str(gguf), "gpu": 0},
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
                },
                sort_keys=True,
            )
            + "\n",
        ),
        env={},
        timeout_s=2,
    )
    assert parsed["usable"] is True
    failed = exp._run_bounded_headline_prompt(
        {"hf_id": QWEN, "path": str(gguf), "gpu": 0},
        selected_python=SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, returncode=1, stderr="load failed"),
        env={},
        timeout_s=2,
    )
    assert failed["generation_status"] == "failed"

    blocked_generation = exp.build_preflight_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_runner(),
        prompt_runner_fn=lambda model, **_: {
            "attempted": True,
            "load_status": "loaded",
            "generation_status": "empty_response",
            "usable": False,
            "gpu_backed": True,
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "response_text": "",
            "tokens_generated": 0,
            "duration_seconds": 0.25,
            "inference_substrate": "llama_cpp_gpu",
        },
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        monotonic=iter([2.0, 2.5]).__next__,
    )
    assert blocked_generation["honest_verdict"].startswith("blocked_generation")
    assert blocked_generation["inference_substrate"] == "live_llm_inference_failed"
    assert exp._honest_verdict(
        ready=False,
        cached_count=1,
        torch_cuda=True,
        llama_gpu=True,
        attempted_live=False,
    ).startswith("blocked_preconditions")

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
    assert exp.main(
        [
            "--output",
            str(tmp_path / "out2.json"),
            "--selected-python",
            SELECTED_PYTHON,
            "--run-legacy-smoke",
            "--prompt-timeout-s",
            "7",
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
            "run_legacy_smoke": True,
            "prompt_timeout_s": 7,
        },
    ]
