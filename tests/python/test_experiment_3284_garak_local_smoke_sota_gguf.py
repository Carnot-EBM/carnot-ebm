"""Tests for Exp 3284 Garak local smoke on SOTA GGUFs.

Spec refs: REQ-REPORT-3284, SCENARIO-REPORT-3284.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest

from carnot.reporting import garak_local_smoke_sota_gguf_3284 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "garak_local_smoke_v1_ready",
    "garak_smoke_ready",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "local_target_adapter_started",
    "garak_probe_count",
    "attack_success_rate",
    "detector_or_defense_response_summary",
    "gpu_mem_used_mib",
    "tokens_generated",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_prior_3282(root: Path, *, ready: bool = True) -> None:
    _write_json(
        root,
        mod.EXP3282_REL_PATH,
        {
            "experiment_id": "exp3282",
            "garak_runner_ready": ready,
            "garak_available": ready,
            "garak_import_command": "uv run --no-project --with garak python -c import",
            "garak_cli_command": "uv run --no-project --with garak garak --version",
            "local_target_adapter_plan": {
                "adapter_kind": "llama_cpp_openai_compatible_rest",
                "openai_compatible_base_url": "http://127.0.0.1:8080/v1",
            },
            "honest_verdict": "complete: garak runner ready",
        },
    )


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
    command: Sequence[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "command": list(command),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def _runner(
    *,
    nvidia_ok: bool = True,
    cuda_ok: bool = True,
) -> tuple[mod.CommandRunner, list[list[str]]]:
    calls: list[list[str]] = []

    def run(command: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        rendered = list(command)
        calls.append(rendered)
        joined = " ".join(rendered)
        if rendered[:1] == ["nvidia-smi"] and "memory.total" in joined:
            if not nvidia_ok:
                return _command(rendered, returncode=1, stderr="nvidia-smi failed")
            return _command(
                rendered,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 4, 0, 595.71.05\n"
                ),
            )
        if "exp3284_cuda_probe" in joined:
            payload = {
                "python": SELECTED_PYTHON,
                "torch_import_ok": True,
                "cuda_available": cuda_ok,
                "cuda_device_count": 2 if cuda_ok else 0,
                "cuda_device_name": "NVIDIA GeForce RTX 3090" if cuda_ok else "",
                "llama_cpp_import_ok": True,
                "llama_cpp_supports_gpu_offload": cuda_ok,
                "llama_cpp_system_info": "CUDA : ARCHS = 860" if cuda_ok else "CPU : AVX2 = 1",
                "probe_error": "" if cuda_ok else "cuda unavailable",
            }
            return _command(
                rendered,
                returncode=0 if cuda_ok else 1,
                stdout="exp3284_cuda_probe\n" + json.dumps(payload, sort_keys=True) + "\n",
                stderr="" if cuda_ok else "cuda unavailable",
            )
        raise AssertionError(f"unexpected command: {rendered}")

    return run, calls


def _fake_smoke_success(**kwargs: Any) -> mod.SmokeRunResult:
    return mod.SmokeRunResult(
        adapter_started=True,
        adapter_command=["python", "adapter.py"],
        adapter_error="",
        garak_command=["python", "garak_smoke.py"],
        probe_count=int(kwargs["probe_count"]),
        attack_success_rate=0.1,
        detector_or_defense_response_summary={
            "attack_success_count": 2,
            "refusal_count": 6,
            "empty_response_count": 0,
            "error_count": 0,
            "detector": "garak.promptinject_rogue_string_substring",
        },
        gpu_mem_used_mib=9000,
        tokens_generated=640,
        duration_s=12.5,
        raw_report_path="",
    )


def _fake_smoke_should_not_run(**kwargs: Any) -> mod.SmokeRunResult:
    raise AssertionError(f"smoke runner should not run: {kwargs}")


def _fake_smoke_adapter_block(**kwargs: Any) -> mod.SmokeRunResult:
    return mod.SmokeRunResult(
        adapter_started=False,
        adapter_command=["python", "adapter.py"],
        adapter_error="bind failed",
        garak_command=[],
        probe_count=0,
        attack_success_rate=0.0,
        detector_or_defense_response_summary={},
        gpu_mem_used_mib=0,
        tokens_generated=0,
        duration_s=1.0,
    )


def test_req_report_3284_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3284: OpenSpec declares the smoke contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3284" in spec
    assert "SCENARIO-REPORT-3284" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "cached_sota_pair(gpu_indices=(0, 1))" in spec
    assert "Legacy tiny models" in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3284_missing_models_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3284: missing mandated GGUFs produce exact blocker evidence."""

    _write_prior_3282(tmp_path)
    runner, calls = _runner()

    artifact = mod.build_artifact(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        smoke_runner=_fake_smoke_should_not_run,
        monotonic=iter([10.0, 10.5]).__next__,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3284"
    assert artifact["garak_local_smoke_v1_ready"] is True
    assert artifact["garak_smoke_ready"] is False
    assert artifact["models_used"] == []
    assert [row["model_id"] for row in artifact["missing_model_specs"]] == list(
        mod.MANDATED_MODEL_IDS
    )
    assert artifact["local_target_adapter_started"] is False
    assert artifact["garak_probe_count"] == 0
    assert artifact["attack_success_rate"] == 0.0
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["tokens_generated"] == 0
    assert artifact["duration_s"] == pytest.approx(0.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "missing_mandated_sota_gguf" in artifact["honest_verdict"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["model_specs"]["cached_sota_pair_used"] is False
    assert artifact["model_specs"]["mandated_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["detector_or_defense_response_summary"]["status"] == "blocked"
    assert {row["name"] for row in artifact["preconditions_checked"]} >= {
        "prior_exp3282_garak_runner_ready",
        "nvidia_smi",
        "selected_python_cuda",
        "local_gguf_cache",
    }
    assert any("exp3284_cuda_probe" in " ".join(call) for call in calls)


def test_scenario_report_3284_one_cached_mandated_model_can_smoke(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3284: one cached mandated GGUF can produce smoke evidence."""

    _write_prior_3282(tmp_path)
    cache_root = tmp_path / "hf-cache"
    model_path = _write_model(cache_root, GEMMA26)
    runner, calls = _runner()

    artifact = mod.build_artifact(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        smoke_runner=_fake_smoke_success,
        monotonic=iter([100.0, 125.0]).__next__,
        probe_count=20,
    )

    assert artifact["garak_smoke_ready"] is True
    assert artifact["local_target_adapter_started"] is True
    assert artifact["garak_probe_count"] == 20
    assert artifact["attack_success_rate"] == pytest.approx(0.1)
    assert artifact["gpu_mem_used_mib"] == 9000
    assert artifact["tokens_generated"] == 640
    assert artifact["detector_or_defense_response_summary"]["refusal_count"] == 6
    assert artifact["models_used"] == [
        {
            "model_id": GEMMA26,
            "model_path": str(model_path),
            "filename": model_path.name,
            "fallback_legacy": False,
            "local_target_adapter_started": True,
            "garak_probe_count": 20,
            "tokens_generated": 640,
        }
    ]
    assert {row["model_id"] for row in artifact["missing_model_specs"]} == {QWEN, GEMMA31}
    assert artifact["model_specs"]["mandated_models"][GEMMA26]["cached"] is True
    assert artifact["model_specs"]["mandated_models"][QWEN]["cached"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "garak_smoke_ready=true" in artifact["honest_verdict"]
    assert any(call[0] == "nvidia-smi" for call in calls)


def test_req_report_3284_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3284: validation and parsing helpers reject ambiguous evidence."""

    assert mod._safe_int("12") == 12
    assert mod._safe_int("bad") is None
    assert mod._safe_float("1.25") == pytest.approx(1.25)
    assert mod._safe_float(None) == 0.0
    assert mod._duration(10.0, 8.0) == 0.0
    assert mod._summarize("a\n" * 20, limit=12).endswith("...")
    assert mod._json_from_last_line({"stdout": "bad\n{\"ok\": true}\n"}) == {"ok": True}
    assert mod._json_from_last_line({"stdout": "bad\n"}) == {}
    assert mod._parse_nvidia_smi_csv("bad\nx, GPU, bad, 4, 0, 555\n0, GPU, 100, 4, 0, 555\n") == [
        {
            "index": 0,
            "name": "GPU",
            "memory_total_mib": 100,
            "memory_used_mib": 4,
            "utilization_gpu_pct": 0,
            "driver_version": "555",
        }
    ]

    artifact = {
        "garak_local_smoke_v1_ready": True,
        "garak_smoke_ready": False,
        "model_specs": {"mandated_model_ids": list(mod.MANDATED_MODEL_IDS)},
        "models_used": [],
        "missing_model_specs": [],
        "preconditions_checked": [],
        "local_target_adapter_started": False,
        "garak_probe_count": 0,
        "attack_success_rate": 0.0,
        "detector_or_defense_response_summary": {},
        "gpu_mem_used_mib": 0,
        "tokens_generated": 0,
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": 0.0,
        "honest_verdict": "complete: blocked",
    }
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: artifact[key] for key in REQUIRED_FIELDS - {"duration_s"}})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="garak_probe_count"):
        mod.validate_artifact(artifact | {"garak_probe_count": 51})
    with pytest.raises(ValueError, match="attack_success_rate"):
        mod.validate_artifact(artifact | {"attack_success_rate": 1.5})
    with pytest.raises(ValueError, match="garak_smoke_ready"):
        mod.validate_artifact(artifact | {"garak_smoke_ready": True, "models_used": []})

    path = tmp_path / "bad.json"
    assert mod.read_json_object(path) == {}
    path.write_text("{", encoding="utf-8")
    assert mod.read_json_object(path) == {}
    assert mod._models_used(None, mod._blocked_smoke_result("blocked")) == []

    assert mod._selected_python(tmp_path) == sys.executable
    selected = tmp_path / ".venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.touch()
    assert mod._selected_python(tmp_path) == selected.as_posix()


def test_req_report_3284_default_cache_pair_and_adapter_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3284: cached_sota_pair default path and adapter-block rows are explicit."""

    _write_prior_3282(tmp_path)
    cache_root = tmp_path / "default-cache"
    model_path = _write_model(cache_root, GEMMA26)
    runner, _calls = _runner()
    monkeypatch.setattr(mod, "_default_cache_roots", lambda project_root, env: [cache_root])
    monkeypatch.setattr(
        mod,
        "cached_sota_pair",
        lambda gpu_indices=(0, 1): [{"hf_id": GEMMA26, "model_path": str(model_path)}],
    )
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda model_id: None)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        smoke_runner=_fake_smoke_adapter_block,
        monotonic=iter([1.0, 2.0]).__next__,
    )

    assert artifact["model_specs"]["cached_sota_pair_used"] is True
    assert artifact["models_used"][0]["model_id"] == GEMMA26
    assert artifact["local_target_adapter_started"] is False
    assert artifact["garak_smoke_ready"] is False
    assert artifact["detector_or_defense_response_summary"]["status"] == "adapter_blocked"
    assert artifact["detector_or_defense_response_summary"]["adapter_error"] == "bind failed"


def test_req_report_3284_default_cache_single_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3284: default cache fallback can resolve one mandated GGUF."""

    cache_root = tmp_path / "default-cache"
    model_path = _write_model(cache_root, GEMMA26)
    monkeypatch.setattr(mod, "_default_cache_roots", lambda project_root, env: [cache_root])
    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(
        mod,
        "resolve_cached_gguf",
        lambda model_id: str(model_path) if model_id == GEMMA26 else None,
    )

    available, missing, cache_check, model_specs = mod.resolve_model_cache(
        project_root=tmp_path,
        cache_roots=None,
        env={},
    )

    assert [row["model_id"] for row in available] == [GEMMA26]
    assert {row["model_id"] for row in missing} == {QWEN, GEMMA31}
    assert cache_check["passed"] is True
    assert model_specs["cached_sota_pair_used"] is False
