"""Tests for Exp 2836 SOTA runtime/cache manifest.

Spec: REQ-INFER-SOTA-012,
      SCENARIO-INFER-SOTA-012-001,
      SCENARIO-INFER-SOTA-012-002,
      SCENARIO-INFER-SOTA-012-003
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_runtime_preflight as mod
from carnot.reporting.sota_runtime_preflight import (
    REQUIRED_ARTIFACT_FIELDS,
    build_runtime_cache_manifest,
    run_experiment,
)


SELECTED_PYTHON = "/repo/.venv/bin/python"
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


def _fake_runner(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None) -> dict[str, Any]:
    del timeout_s, env
    if command[0] == SELECTED_PYTHON and "import torch" in command[-1]:
        return _command(command, stdout="2.11.0+cu128 True\n")
    if command[0] == "python3" and "import torch" in command[-1]:
        return _command(
            command,
            returncode=1,
            stderr="ModuleNotFoundError: No module named 'torch'\n",
        )
    if command[:2] == ["df", "-k"]:
        return _command(command, stdout="Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/root 100 40 60 40% /\n")
    if command[:1] == ["nvidia-smi"]:
        return _command(
            command,
            stdout="0, NVIDIA GeForce RTX 3090, 24576, 5, 24122\n1, NVIDIA GeForce RTX 3090, 24576, 5, 24122\n",
        )
    if command[0] == SELECTED_PYTHON and "llama_supports_gpu_offload" in command[-1]:
        return _command(
            command,
            stdout=json.dumps(
                {
                    "llama_cpp_import_ok": True,
                    "llama_cpp_origin": "/repo/.venv/lib/python3.14/site-packages/llama_cpp/__init__.py",
                    "llama_cpp_version": "0.3.23",
                    "llama_cpp_supports_gpu_offload": False,
                    "llama_cpp_supports_mmap": True,
                }
            )
            + "\n",
        )
    raise AssertionError(f"unexpected command: {command}")


def _write_cached_gemma(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    hub = tmp_path / "hf" / "hub"
    snapshot = hub / "models--unsloth--gemma-4-26B-A4B-it-GGUF" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    gguf = snapshot / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    gguf.write_text("tiny gguf test fixture", encoding="utf-8")
    return gguf, {"HUGGINGFACE_HUB_CACHE": str(hub)}


def test_exp2836_ready_manifest_records_venv_cuda_and_one_smoked_sota(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-012 / SCENARIO-INFER-SOTA-012-001: one smoke-loaded SOTA opens gate."""
    gguf, env = _write_cached_gemma(tmp_path)
    expected_sha = hashlib.sha256(gguf.read_bytes()).hexdigest()
    smoke_calls: list[str] = []

    def smoke_loader(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        smoke_calls.append(model["hf_id"])
        return {
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "load_attempted": True,
            "load_success": True,
            "headline_usable": True,
            "load_mode": "llama_cpp_full_context_load",
            "elapsed_s": 0.25,
            "blocker": None,
        }

    artifact = build_runtime_cache_manifest(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=_fake_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        smoke_loader_fn=smoke_loader,
        monotonic=iter([10.0, 12.5]).__next__,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["sota_runtime_ready"] is True
    assert artifact["selected_python"] == SELECTED_PYTHON
    assert artifact["venv_torch_cuda_available"] is True
    assert artifact["system_python_torch_cuda_available"] is False
    assert artifact["cached_sota_pair_result"]["called"] is True
    assert artifact["cached_sota_pair_result"]["result"] is None
    assert artifact["model_specs"]["primary"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert artifact["model_specs"]["legacy_cpu_smoke_only"] == list(mod.LEGACY_CPU_SMOKE_ONLY)
    assert [row["hf_id"] for row in artifact["sota_models_cached"]] == [GEMMA26]
    assert artifact["sota_models_cached"][0]["sha256"] == expected_sha
    assert artifact["sota_models_cached"][0]["model_family"] == "gemma"
    assert artifact["sota_models_cached"][0]["size_bytes"] == gguf.stat().st_size
    assert artifact["smoke_load_results"][0]["load_success"] is True
    assert artifact["loader_probe"]["llama_cpp_supports_gpu_offload"] is False
    assert "venv_torch_cuda" in {row["resource"] for row in artifact["preconditions_checked"]}
    assert smoke_calls == [GEMMA26]
    assert artifact["duration_s"] == pytest.approx(2.5)


def test_exp2836_missing_cache_blocks_without_blind_download(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-012-002: no cache and no credentials writes blocked_model_cache."""

    def smoke_loader(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        raise AssertionError(f"smoke loader must not run for missing cache: {model}")

    artifact = build_runtime_cache_manifest(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_fake_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        smoke_loader_fn=smoke_loader,
        monotonic=iter([1.0, 1.1]).__next__,
    )

    assert artifact["honest_verdict"].startswith("blocked_model_cache")
    assert artifact["sota_runtime_ready"] is False
    assert artifact["sota_models_cached"] == []
    assert artifact["models_missing_from_cache"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert artifact["blocked_model_cache"]["attempted"] is False
    assert artifact["blocked_model_cache"]["status"] == "skipped_no_local_credentials"
    assert artifact["smoke_load_results"] == []


def test_exp2836_venv_cuda_gate_is_separate_from_system_python(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-012-003: readiness is gated on selected Python, not python3."""
    _gguf, env = _write_cached_gemma(tmp_path)

    def no_cuda_runner(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None) -> dict[str, Any]:
        if command[0] == SELECTED_PYTHON and "import torch" in command[-1]:
            return _command(command, stdout="2.11.0+cu128 False\n")
        return _fake_runner(command, timeout_s=timeout_s, env=env)

    artifact = build_runtime_cache_manifest(
        project_root=tmp_path,
        run_date="20260522",
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=no_cuda_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        smoke_loader_fn=lambda model, **_: {
            "hf_id": model["hf_id"],
            "load_attempted": True,
            "load_success": True,
            "headline_usable": True,
            "blocker": None,
        },
        monotonic=iter([3.0, 4.0]).__next__,
    )

    assert artifact["venv_torch_cuda_available"] is False
    assert artifact["system_python_torch_cuda_available"] is False
    assert artifact["sota_runtime_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_cuda")


def test_exp2836_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-012: run_experiment persists the terminal manifest."""
    output = tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260522",
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf")},
        command_runner=_fake_runner,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        smoke_loader_fn=lambda model, **_: pytest.fail(f"unexpected smoke: {model}"),
        monotonic=iter([5.0, 5.2]).__next__,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["run_date"] == "20260522"
    assert artifact["artifact"] == "experiment_2836_sota_runtime_preflight"


def test_exp2836_default_paths_commands_and_parse_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-012: helper probes handle local command and parsing edge cases."""
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    assert mod._repo_root() == tmp_path
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._selected_python(tmp_path / "missing") == os.sys.executable
    assert mod._summarize(None) == ""
    assert mod._summarize("abcdef", limit=3) == "abc...<truncated>"

    ok = mod._run_command(["printf", "ok"], timeout_s=5)
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    failed = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert failed["returncode"] is None
    assert "FileNotFoundError" in failed["stderr_summary"]

    bad_gpu = mod._gpu_memory_probe(
        command_runner=lambda command, **_: _command(
            command,
            stdout="bad\n0, RTX, not-int, 5, 10\n",
        )
    )
    assert bad_gpu["available"] is False

    bad_loader = mod._loader_probe(
        SELECTED_PYTHON,
        command_runner=lambda command, **_: _command(command, stdout="", stderr="boom"),
        env={},
    )
    assert bad_loader["llama_cpp_import_ok"] is False
    assert bad_loader["error"] == "boom"

    assert mod._cache_roots(tmp_path, {"HF_HOME": str(tmp_path / "hfhome")})[
        "huggingface_hub_cache"
    ].endswith("hfhome/hub")
    default_roots = mod._cache_roots(tmp_path, {})
    assert default_roots["huggingface_hub_cache"].endswith(".cache/huggingface/hub")

    assert mod._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert mod._quantization_suffix(None) is None
    assert mod._quantization_suffix("model-no-token.gguf") == "unknown"


def test_exp2836_cache_candidate_helpers_cover_local_models_and_broken_paths(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-012: cache search stays deterministic across HF and models/ layouts."""
    roots = mod._cache_roots(tmp_path, {"HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf" / "hub")})
    local_dir = tmp_path / "models" / "gemma-4-26B-A4B-it"
    local_dir.mkdir(parents=True)
    ignored = local_dir / "mmproj-F16.gguf"
    ignored.write_text("projector", encoding="utf-8")
    local = local_dir / "gemma-4-26B-A4B-it-random.gguf"
    local.write_text("local", encoding="utf-8")
    preferred = local_dir / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    preferred.write_text("preferred", encoding="utf-8")
    broken = local_dir / "broken.gguf"
    broken.symlink_to(local_dir / "absent.gguf")

    candidates = mod._candidate_paths(tmp_path, GEMMA26, roots)

    assert preferred in candidates
    assert ignored not in candidates
    assert mod._candidate_size(broken) == 0
    assert mod._select_candidate([broken]) is None
    assert mod._select_candidate([local]) == local
    assert mod._select_candidate(candidates) == preferred


def test_exp2836_json_pair_credentials_and_verdict_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-012: small pure helpers preserve terminal and JSON contracts."""
    token_home = tmp_path / "hfhome"
    token_home.mkdir()
    (token_home / "token").write_text("token", encoding="utf-8")
    default_token = tmp_path / ".cache" / "huggingface" / "token"
    default_token.parent.mkdir(parents=True)
    default_token.write_text("token", encoding="utf-8")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    assert mod._json_safe(Path("x")) == "x"
    assert mod._json_safe({"p": Path("x")}) == {"p": "x"}
    assert mod._json_safe([Path("x")]) == ["x"]
    assert mod._json_safe(object()).startswith("<object object")
    assert mod._exercise_cached_sota_pair(lambda **_: (_ for _ in ()).throw(RuntimeError("bad")))[
        "error"
    ] == "RuntimeError: bad"
    assert mod._hf_credentials_configured({"HF_TOKEN": "x"}) is True
    assert mod._hf_credentials_configured({"HF_HOME": str(token_home)}) is True
    assert mod._hf_credentials_configured({}) is True
    assert mod._blocked_model_cache_attempt(
        missing_models=["m"],
        cached_models_present=False,
        env={"HF_TOKEN": "x"},
    )["status"] == "metadata_only_probe_allowed_but_not_needed_for_automated_preflight"
    assert mod._honest_verdict(
        ready=False,
        venv_cuda=True,
        cached_count=1,
        smoke_success=False,
    ).startswith("blocked_loader_smoke")
    assert mod._honest_verdict(
        ready=False,
        venv_cuda=True,
        cached_count=1,
        smoke_success=True,
    ).startswith("blocked_unknown")


def test_exp2836_default_smoke_loader_parses_success_and_failure(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-012: default smoke loader records subprocess success and blockers."""
    model = {"hf_id": GEMMA26, "path": str(tmp_path / "model.gguf")}
    import_blocked = mod._smoke_load_model(
        model,
        selected_python=SELECTED_PYTHON,
        loader_probe={"llama_cpp_import_ok": False, "error": "ImportError: no llama"},
        gpu_probe={},
        command_runner=lambda *_args, **_kwargs: pytest.fail("command runner not expected"),
        env={},
    )
    assert import_blocked["load_attempted"] is False
    assert import_blocked["blocker"] == "ImportError: no llama"

    success = mod._smoke_load_model(
        model,
        selected_python=SELECTED_PYTHON,
        loader_probe={"llama_cpp_import_ok": True},
        gpu_probe={},
        command_runner=lambda command, **_: _command(
            command,
            stdout=json.dumps(
                {
                    "load_success": True,
                    "headline_usable": True,
                    "elapsed_s": 0.1,
                    "load_mode": "llama_cpp_full_context_load",
                }
            )
            + "\n",
        ),
        env={},
    )
    assert success["load_success"] is True
    assert success["headline_usable"] is True
    assert success["blocker"] is None

    failed = mod._smoke_load_model(
        model,
        selected_python=SELECTED_PYTHON,
        loader_probe={"llama_cpp_import_ok": True},
        gpu_probe={},
        command_runner=lambda command, **_: _command(
            command,
            returncode=1,
            stdout="not json",
            stderr="load failed",
        ),
        env={},
    )
    assert failed["load_success"] is False
    assert failed["blocker"] == "load failed"


def test_exp2836_cli_entrypoint_delegates_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-012: CLI argument parsing preserves the output contract."""
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
            ]
        )
        == 0
    )
    assert calls == [
        {
            "run_date": "20260522",
            "output_path": tmp_path / "out.json",
            "selected_python": SELECTED_PYTHON,
        }
    ]
