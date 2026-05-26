"""Tests for Exp 3151 live-inference authenticity preflight.

Spec refs: REQ-VERIFY-3151, SCENARIO-VERIFY-3151.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import live_inference_authenticity_preflight_v1 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "live_inference_authenticity_preflight_ready",
    "model_specs",
    "locally_usable_model_ids",
    "selected_model_ids",
    "preflight_passed",
    "live_call_count",
    "model_load_evidence",
    "transcript_hashes",
    "minimum_duration_requirement_s",
    "headline_claim_allowed",
    "blocked_reason",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_docs(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text(
        "All headline results must have live GPU provenance.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text(
        "def cached_sota_pair():\n    return None\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3151\nSCENARIO-VERIFY-3151\n"
        "results/experiment_3151_live_inference_authenticity_preflight_v1.json\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3139_REL_PATH,
        {
            "artifact": "experiment_3139_live_sota_verifier_rerun_v7",
            "live_call_count": 6,
            "headline_claim_allowed": True,
            "inference_substrate": {
                "gpu_preflight": {
                    "no_model_loaded": True,
                    "no_inference_run": True,
                }
            },
            "honest_verdict": "complete: prior artifact exposed authenticity mismatch",
        },
    )


def _write_manifest(root: Path, *, model_path: Path | None) -> None:
    inventory = [
        {
            "hf_id": QWEN,
            "name": "Qwen3.6-35B-A3B",
            "role": "moe",
            "cache_status": "missing",
            "path": None,
            "resolved_path": None,
        },
        {
            "hf_id": GEMMA31,
            "name": "Gemma4-31B-it",
            "role": "dense",
            "cache_status": "missing",
            "path": None,
            "resolved_path": None,
        },
        {
            "hf_id": GEMMA26,
            "name": "Gemma4-26B-A4B-it",
            "role": "moe",
            "cache_status": "resolved" if model_path is not None else "missing",
            "path": str(model_path) if model_path is not None else None,
            "resolved_path": str(model_path) if model_path is not None else None,
        },
    ]
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
            "mandatory_headline_model_ids": [QWEN, GEMMA31, GEMMA26],
            "present_model_ids": [GEMMA26] if model_path is not None else [],
            "selected_headline_model_ids": [GEMMA26] if model_path is not None else [],
            "headline_claim_allowed": model_path is not None,
            "cache_inventory": inventory,
            "gpu_preflight": {
                "cuda_available": True,
                "gpu_count": 2,
                "no_model_loaded": True,
                "no_inference_run": True,
            },
        },
    )


def _write_sources(root: Path, *, model_present: bool = True) -> Path | None:
    _write_docs(root)
    model_path: Path | None = None
    if model_present:
        model_path = root / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"GGUF unit fixture")
    _write_manifest(root, model_path=model_path)
    return model_path


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


def _runner(
    *,
    cuda: bool = True,
    smoke: dict[str, Any] | None = None,
    smoke_returncode: int = 0,
    smoke_stderr: str = "",
    seen_smoke: list[list[str]] | None = None,
) -> mod.CommandRunner:
    def fake(
        command: list[str],
        *,
        timeout_s: int = 10,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del timeout_s, env
        if command[0] == SELECTED_PYTHON and "import torch" in command[-1]:
            return _command(command, stdout=f"2.11.0+cu128 {cuda} {2 if cuda else 0}\n")
        if command[:1] == ["nvidia-smi"]:
            return _command(
                command,
                stdout=(
                    "0, NVIDIA GeForce RTX 3090, 24576, 4, 24123, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 4, 24123, 595.71.05\n"
                ),
            )
        if "--exp3151-smoke-worker" in command:
            if seen_smoke is not None:
                seen_smoke.append(command)
            stdout = json.dumps(smoke or {"ok": True, "output_text": "VALID"}) + "\n"
            return _command(command, returncode=smoke_returncode, stdout=stdout, stderr=smoke_stderr)
        raise AssertionError(f"unexpected command: {command}")

    return fake


def _smoke_payload(*, load_s: float = 62.0, output: str = "VALID") -> dict[str, Any]:
    return {
        "ok": True,
        "runtime": "llama_cpp",
        "load_wall_time_s": load_s,
        "generation_wall_time_s": 1.25,
        "total_worker_wall_time_s": load_s + 1.25,
        "output_text": output,
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 1,
            "total_tokens": 12,
        },
    }


def test_req_verify_3151_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3151: OpenSpec declares the preflight before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3151" in spec
    assert "SCENARIO-VERIFY-3151" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "model_load_evidence" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3151_records_successful_smoke_evidence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3151: one safe mandated smoke call records authenticity evidence."""

    model_path = _write_sources(tmp_path, model_present=True)
    seen_smoke: list[list[str]] = []

    artifact = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(smoke=_smoke_payload(), seen_smoke=seen_smoke),
        started_s=10.0,
        now_s=75.5,
        tests_run=["REQ-VERIFY-3151 focused"],
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["live_inference_authenticity_preflight_ready"] is True
    assert artifact["preflight_passed"] is True
    assert artifact["live_call_count"] == 1
    assert artifact["headline_claim_allowed"] is False
    assert artifact["blocked_reason"] == ""
    assert artifact["locally_usable_model_ids"] == [GEMMA26]
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["random_seed"] == 20260526
    assert artifact["minimum_duration_requirement_s"] == pytest.approx(60.0)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["tests_run"] == ["REQ-VERIFY-3151 focused"]
    assert artifact["duration_s"] == pytest.approx(65.5)
    assert artifact["honest_verdict"].startswith("complete:")

    load = artifact["model_load_evidence"]
    assert load["load_attempted"] is True
    assert load["selected_model_id"] == GEMMA26
    assert load["selected_model_path"] == str(model_path)
    assert load["path_exists"] is True
    assert load["load_wall_time_s"] == pytest.approx(62.0)
    assert load["load_command"] == seen_smoke[0]
    assert len(load["load_command_sha256"]) == 64

    assert artifact["token_counts"] == {
        "prompt_tokens": 11,
        "completion_tokens": 1,
        "total_tokens": 12,
        "source": "llama_cpp_usage",
    }
    assert len(artifact["transcript_hashes"]) == 1
    transcript = artifact["transcript_hashes"][0]
    assert transcript["model_id"] == GEMMA26
    assert transcript["prompt_hash"]
    assert transcript["response_hash"]
    assert transcript["transcript_sha256"]
    assert transcript["output_token_count"] == 1

    assert artifact["model_specs"][2]["hf_id"] == GEMMA26
    assert artifact["model_specs"][2]["path_exists"] is True
    assert artifact["model_specs"][2]["selected_for_smoke"] is True
    assert artifact["inference_substrate"]["executes_models"] is True
    assert artifact["inference_substrate"]["live_model_calls"] == 1
    assert artifact["inference_substrate"]["downloads_models"] is False
    assert artifact["inference_substrate"]["gpu_probe"]["cuda_available"] is True
    assert artifact["preflight_contract_for_exp3152"]["must_not_score_verifier_panel"] is True
    assert "model_load_evidence" in artifact["preflight_contract_for_exp3152"]["required_fields"]


def test_req_verify_3151_blocks_without_usable_mandated_model(tmp_path: Path) -> None:
    """REQ-VERIFY-3151: missing mandated GGUF writes a complete blocked artifact."""

    seen_smoke: list[list[str]] = []
    _write_sources(tmp_path, model_present=False)

    artifact = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(smoke=_smoke_payload(), seen_smoke=seen_smoke),
        started_s=1.0,
        now_s=1.5,
    )

    assert artifact["live_inference_authenticity_preflight_ready"] is True
    assert artifact["preflight_passed"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["headline_claim_allowed"] is False
    assert artifact["selected_model_ids"] == []
    assert artifact["locally_usable_model_ids"] == []
    assert artifact["transcript_hashes"] == []
    assert artifact["model_load_evidence"]["load_attempted"] is False
    assert "no mandated local SOTA GGUF" in artifact["blocked_reason"]
    assert artifact["honest_verdict"].startswith("blocked_no_mandated_sota_gguf:")
    assert seen_smoke == []
    assert all(row["path_exists"] is False for row in artifact["model_specs"])


def test_req_verify_3151_runtime_failure_and_duration_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3151: runtime and duration blockers fail closed with evidence."""

    _write_sources(tmp_path, model_present=True)
    gpu_blocked = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(cuda=False, smoke=_smoke_payload()),
        started_s=1.0,
        now_s=70.0,
    )
    failed = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(
            smoke={"ok": False, "error": "llama load failed"},
            smoke_returncode=1,
            smoke_stderr="boom\n",
        ),
        started_s=1.0,
        now_s=70.0,
    )
    short = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(smoke=_smoke_payload(load_s=3.0)),
        started_s=1.0,
        now_s=9.0,
    )
    ok_false = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(smoke={"ok": False, "error": "worker refused"}),
        started_s=1.0,
        now_s=70.0,
    )

    assert gpu_blocked["preflight_passed"] is False
    assert gpu_blocked["live_call_count"] == 0
    assert gpu_blocked["model_load_evidence"]["load_attempted"] is False
    assert gpu_blocked["honest_verdict"].startswith("blocked_gpu_substrate:")
    assert failed["preflight_passed"] is False
    assert failed["live_call_count"] == 0
    assert failed["model_load_evidence"]["load_attempted"] is True
    assert "llama load failed" in failed["model_load_evidence"]["runtime_error"]
    assert failed["honest_verdict"].startswith("blocked_smoke_runtime:")

    assert short["preflight_passed"] is False
    assert short["live_call_count"] == 1
    assert short["transcript_hashes"]
    assert "shorter than minimum" in short["blocked_reason"]
    assert short["honest_verdict"].startswith("blocked_duration_too_short:")
    assert ok_false["model_load_evidence"]["runtime_error"] == "worker refused"


def test_req_verify_3151_write_artifact_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3151: writer persists JSON and validation rejects overclaims."""

    _write_sources(tmp_path, model_present=True)
    output = mod.write_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=_runner(smoke=_smoke_payload()),
        started_s=3.0,
        now_s=66.0,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["preflight_passed"] is True
    assert saved["source_checksums"][mod.EXP3123_REL_PATH.as_posix()]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad-json}\n", encoding="utf-8")
    large = tmp_path / "large.gguf"
    large.write_bytes(b"a" * (1024 * 1024 + 2))
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.path_evidence(tmp_path, "missing.gguf") == {
        "path": str(tmp_path / "missing.gguf"),
        "exists": False,
        "size_bytes": None,
        "bounded_sha256": None,
    }
    assert len(mod.bounded_file_hash(large)) == 64
    assert mod.parse_smoke_stdout("{\"ok\": true}\nnot-json\n") == ({"ok": True}, None)
    assert mod.token_counts_for("one two", "three four", None) == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "source": "whitespace_estimate",
    }
    assert mod.int_or_none("not-int") is None
    assert mod.duration(5.0, 4.0) == 0.0

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="headline_claim_allowed"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(saved | {"live_call_count": -1})
    with pytest.raises(ValueError, match="passed preflight requires live call"):
        mod.validate_artifact(saved | {"live_call_count": 0, "preflight_passed": True})
    with pytest.raises(ValueError, match="transcript_hashes"):
        mod.validate_artifact(saved | {"transcript_hashes": [], "preflight_passed": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "ready: no"})
    with pytest.raises(ValueError, match="blocked_"):
        mod.validate_artifact(saved | {"preflight_passed": False, "honest_verdict": "complete: wrong"})
