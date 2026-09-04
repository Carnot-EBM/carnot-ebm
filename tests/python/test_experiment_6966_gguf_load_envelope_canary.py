"""Tests for the task-owned three-family GGUF load envelope.

Spec refs: REQ-INFRA-6966, SCENARIO-INFRA-6966-CONFIG,
SCENARIO-INFRA-6966-OWNERSHIP, SCENARIO-INFRA-6966-TEARDOWN, and
SCENARIO-INFRA-6966-BARE-GATES.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6966_gguf_load_envelope_canary as exp


REPO = Path(__file__).resolve().parents[2]


def _resolved_specs(tmp_path: Path) -> list[dict]:
    rows = []
    for index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]))
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1],
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
            }
        )
    return rows


def _gpu_probe(*, foreign: bool = False) -> dict:
    process_rows = (
        [
            {
                "pid": 99991,
                "gpu_uuid": "GPU-a",
                "used_memory_mb": 1000,
                "process_name": "foreign-server",
            }
        ]
        if foreign
        else []
    )
    return {
        "query_ok": True,
        "devices": [
            {
                "index": 0,
                "uuid": "GPU-a",
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_mb": 24576,
                "memory_used_mb": 4,
                "memory_free_mb": 24572,
                "utilization_gpu_pct": 0,
                "temperature_c": 40,
            },
            {
                "index": 1,
                "uuid": "GPU-b",
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_mb": 24576,
                "memory_used_mb": 4,
                "memory_free_mb": 24572,
                "utilization_gpu_pct": 0,
                "temperature_c": 41,
            },
        ],
        "processes": process_rows,
        "raw_receipts": [],
    }


def _passing_preflight(tmp_path: Path) -> dict:
    return exp.collect_preconditions(
        model_specs=_resolved_specs(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: _gpu_probe(),
        llama_probe=lambda: {
            "importable": True,
            "gpu_offload": True,
            "version": "test",
        },
    )


def _generation_row(model_id: str, *, terminal: str = "complete") -> dict:
    return {
        "model_id": model_id,
        "config_id": exp.PROMOTED_CONFIG_ID,
        "terminal_state": terminal,
        "live_cuda": terminal == "complete",
        "output": "CANARY",
        "output_hash": exp.sha256_text("CANARY"),
        "process_exit_code": 0 if terminal == "complete" else 1,
        "owned_process_absent": True,
        "teardown_complete": True,
        "vram_release_passed": True,
        "offloaded_layers": 49 if terminal == "complete" else 0,
        "live_duration_s": 1.0 if terminal == "complete" else 0.0,
        "tokens_per_second": 8.0 if terminal == "complete" else 0.0,
    }


def test_req_infra_6966_spec_anchors_the_full_contract() -> None:
    """REQ-INFRA-6966 exists before its implementation."""

    text = (REPO / "openspec/capabilities/research-harnesses/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6966") :]
    for anchor in (
        "SCENARIO-INFRA-6966-CONFIG",
        "SCENARIO-INFRA-6966-OWNERSHIP",
        "SCENARIO-INFRA-6966-TEARDOWN",
        "SCENARIO-INFRA-6966-BARE-GATES",
        "cached_sota_pair(gpu_indices=(0, 1))",
        "n_ctx >= 16384",
        "512 MiB",
        "gguf_load_canary_complete_score",
        "gguf_runtime_ready_score",
    ):
        assert anchor in section


def test_scenario_infra_6966_config_resolves_exact_three_families(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6966-CONFIG requires the pair call and exact extension."""

    calls: list[dict] = []
    paths = _resolved_specs(tmp_path)

    def pair(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [
            {
                "name": paths[0]["name"],
                "hf_id": paths[0]["hf_id"],
                "gpu": 0,
                "model_path": paths[0]["model_path"],
            },
            {
                "name": paths[2]["name"],
                "hf_id": paths[2]["hf_id"],
                "gpu": 1,
                "model_path": paths[2]["model_path"],
            },
        ]

    resolved = exp.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda model_id, _quant: (
            paths[1]["model_path"] if model_id == exp.REQUIRED_MODEL_IDS[1] else None
        ),
    )
    assert calls == [{"gpu_indices": (0, 1)}]
    assert [row["hf_id"] for row in resolved] == list(exp.REQUIRED_MODEL_IDS)
    assert all(row["gpu_indices"] == [0, 1] for row in resolved)
    assert all(row["headline_eligible"] is True for row in resolved)
    assert exp.model_spec_errors(resolved) == []

    missing = deepcopy(resolved)
    missing[1]["model_path"] = ""
    assert "model_path_missing:unsloth/gemma-4-31B-it-GGUF" in exp.model_spec_errors(missing)


def test_scenario_infra_6966_config_ladder_changes_one_factor() -> None:
    """SCENARIO-INFRA-6966-CONFIG freezes one change per ladder step."""

    rows = exp.configuration_ladder()
    assert rows[0] == exp.EXP6962_LOAD_CONFIG
    assert rows[-1]["config_id"] == exp.PROMOTED_CONFIG_ID
    assert rows[-1]["n_ctx"] >= 16_384
    assert rows[-1]["visible_devices"] == [0, 1]
    assert exp.configuration_ladder_errors(rows) == []

    bad = deepcopy(rows)
    bad[-1]["n_ctx"] = 8192
    bad[-1]["n_batch"] = 256
    assert {"ladder_multiple_factors_changed", "promoted_context_too_small"} <= set(
        exp.configuration_ladder_errors(bad)
    )
    assert {
        "ladder_config_order_mismatch",
        "promoted_context_too_small",
        "promoted_not_dual_cuda",
    } <= set(exp.configuration_ladder_errors([]))


def test_scenario_infra_6966_config_rejects_every_model_spec_drift(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6966-CONFIG fails closed on substitutions and weak paths."""

    rows = _resolved_specs(tmp_path)
    rows.reverse()
    rows[0]["model_path"] = str(tmp_path / "mmproj-model.bin")
    rows[0]["gpu_indices"] = [0]
    rows[0]["headline_eligible"] = False
    errors = exp.model_spec_errors(rows)
    assert "model_ids_mismatch" in errors
    assert any(error.startswith("model_path_not_primary_gguf:") for error in errors)
    assert any(error.startswith("dual_gpu_indices_missing:") for error in errors)
    assert any(error.startswith("headline_eligibility_missing:") for error in errors)


def test_scenario_infra_6966_ownership_preflight_rejects_foreign_process(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6966-OWNERSHIP blocks before a worker can start."""

    preflight = exp.collect_preconditions(
        model_specs=_resolved_specs(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: _gpu_probe(foreign=True),
        llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
    )
    assert preflight["all_passed"] is False
    summary = exp.gate_summary(preflight["checks"])
    assert summary["failed_check"] == "foreign_gpu_compute_processes"
    assert summary["expected_value"] == []
    assert summary["observed_value"][0]["pid"] == 99991

    broken = exp.collect_preconditions(
        model_specs=_resolved_specs(tmp_path),
        checkpoint_path=tmp_path / "missing" / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": False, "devices": [], "processes": []},
        llama_probe=lambda: {"importable": False, "gpu_offload": False, "version": None},
        writable_probe=lambda _path: False,
    )
    failed = {row["check"] for row in broken["checks"] if row["passed"] is False}
    assert {"nvidia_device_count", "llama_cpp_cuda_bindings", "checkpoint_writable"} <= failed


def test_req_infra_6966_host_command_and_writable_probe_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 preserves host-probe success and operating-system failure."""

    success = exp._run_command([sys.executable, "-c", "print('probe')"])
    assert success["passed"] is True
    assert success["stdout"].strip() == "probe"
    failed = exp._run_command([str(tmp_path / "missing-command")])
    assert failed["passed"] is False
    assert failed["exit_code"] is None
    monkeypatch.setattr(
        exp.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError("no"))
    )
    assert exp.checkpoint_is_writable(tmp_path / "checkpoint.json") is False


def test_scenario_infra_6966_ownership_captures_process_identity(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6966-OWNERSHIP binds a live child to its parent."""

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    try:
        row = exp.capture_process_ownership(proc.pid, os.getpid(), [sys.executable, "-c"])
        assert row["owned"] is True
        assert row["pid"] == proc.pid
        assert row["parent_pid"] == os.getpid()
        assert row["pid_start_ticks"] > 0
        assert row["command_hash"].startswith("sha256:")
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    absent = exp.capture_process_ownership(999_999_999, os.getpid(), ["missing"])
    assert absent["owned"] is False


def test_scenario_infra_6966_only_signals_the_exact_owned_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6966-OWNERSHIP never signals a stale or foreign PID."""

    monkeypatch.setattr(exp, "_proc_stat", lambda _pid: (111, 222))
    assert exp._owned_process_absent(7, 999) is True
    assert exp._owned_process_absent(7, 222) is False
    assert exp._terminate_owned_process(7, 111, 999) is False
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(exp.os, "killpg", lambda pid, sig: signalled.append((pid, sig)))
    assert exp._terminate_owned_process(7, 111, 222) is True
    assert signalled == [(7, exp.signal.SIGTERM)]
    monkeypatch.setattr(exp, "_proc_stat", lambda _pid: None)
    assert exp._owned_process_absent(7, 222) is True


def test_scenario_infra_6966_failure_forensics_parse_backend_receipts() -> None:
    """REQ-INFRA-6966 keeps the exact allocation and layer evidence."""

    stderr = (
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18432.00 MiB on device 0: "
        "cudaMalloc failed: out of memory\n"
        "llama_model_load: offloaded 48/49 layers to GPU\n"
    )
    assert exp.parse_allocation_request(stderr) == {
        "requested_mib": 18432.0,
        "device": 0,
        "error": "cudaMalloc failed: out of memory",
    }
    assert exp.parse_offloaded_layers(stderr) == {"offloaded": 48, "total": 49}
    assert exp.parse_allocation_request("quiet failure") is None
    assert exp.parse_offloaded_layers("quiet failure") == {"offloaded": 0, "total": None}


def test_scenario_infra_6966_teardown_requires_both_devices_within_tolerance() -> None:
    """SCENARIO-INFRA-6966-TEARDOWN checks both devices before reuse."""

    baseline = [
        {"index": 0, "memory_used_mb": 4},
        {"index": 1, "memory_used_mb": 8},
    ]
    passed = exp.build_vram_release_row(
        model_id=exp.REQUIRED_MODEL_IDS[0],
        baseline_rows=baseline,
        after_rows=[
            {"index": 0, "memory_used_mb": 516},
            {"index": 1, "memory_used_mb": 1},
        ],
    )
    assert passed["passed"] is True
    assert passed["max_increase_mb"] == 512

    failed = exp.build_vram_release_row(
        model_id=exp.REQUIRED_MODEL_IDS[0],
        baseline_rows=baseline,
        after_rows=[{"index": 0, "memory_used_mb": 517}],
    )
    assert failed["passed"] is False
    assert failed["missing_device_indices"] == [1]
    assert exp._memory_rows({"devices": baseline}) == [
        {"index": 0, "uuid": None, "memory_used_mb": 4, "memory_free_mb": None},
        {"index": 1, "uuid": None, "memory_used_mb": 8, "memory_free_mb": None},
    ]


def test_scenario_infra_6966_bare_gates_require_exact_live_rows() -> None:
    """SCENARIO-INFRA-6966-BARE-GATES rejects legacy and weak rows."""

    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    assert exp.reduce_scores(rows) == (1, 1)
    weak = deepcopy(rows)
    weak[1]["output"] = ""
    assert exp.reduce_scores(weak) == (1, 0)
    legacy = deepcopy(rows)
    legacy[2]["model_id"] = "Qwen/Qwen3.5-0.8B"
    assert exp.reduce_scores(legacy) == (0, 0)


def test_scenario_infra_6966_bare_gates_validate_blocked_and_positive_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6966-BARE-GATES recomputes fields and principles."""

    specs = _resolved_specs(tmp_path)
    preflight = exp.collect_preconditions(
        model_specs=specs,
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: _gpu_probe(foreign=True),
        llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
    )
    blocked = exp.build_artifact(
        run_date="20260904",
        duration_s=1.0,
        live_duration_s=0.0,
        model_specs=specs,
        preconditions=preflight,
    )
    assert blocked["honest_verdict"] == "blocked_gguf_load_envelope_canary"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gguf_load_canary_complete_score"] == 0
    assert blocked["gguf_runtime_ready_score"] == 0
    assert type(blocked["gguf_runtime_ready_score"]) is int
    assert set(blocked["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(blocked) == []

    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    positive = exp.build_artifact(
        run_date="20260904",
        duration_s=3.0,
        live_duration_s=3.0,
        model_specs=specs,
        preconditions=_passing_preflight(tmp_path),
        live_generation_rows=rows,
        reproduction_rows=[{"model_id": exp.REQUIRED_MODEL_IDS[0], "terminal_state": "failed"}],
        teardown_rows=[{"model_id": row["model_id"], "passed": True} for row in rows],
        vram_release_rows=[{"model_id": row["model_id"], "passed": True} for row in rows],
        checkpoint_rows=[{"model_id": row["model_id"], "written": True} for row in rows],
    )
    assert positive["verdict_class"] == "positive"
    assert positive["honest_verdict"].startswith("complete:")
    assert exp.validate_artifact(positive) == []

    wrapped = deepcopy(positive)
    wrapped["gguf_runtime_ready_score"] = {"value": 1}
    wrapped["reproducibility_checksum"] = exp.artifact_checksum(wrapped)
    assert "gate_score_not_bare_int:gguf_runtime_ready_score" in exp.validate_artifact(wrapped)


def test_req_infra_6966_validator_rejects_mutated_gate_and_verdict_fields(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6966 independently rejects each critical top-level mutation."""

    specs = _resolved_specs(tmp_path)
    passed = _passing_preflight(tmp_path)
    rows = [_generation_row(model_id) for model_id in exp.REQUIRED_MODEL_IDS]
    positive = exp.build_artifact(
        run_date="20260904",
        duration_s=3.0,
        live_duration_s=3.0,
        model_specs=specs,
        preconditions=passed,
        live_generation_rows=rows,
    )
    assert exp.validate_artifact({}) == [
        f"missing_field:{field}" for field in exp.REQUIRED_ARTIFACT_FIELDS
    ]
    mutated = deepcopy(positive)
    mutated.update(
        {
            "field_principles": {},
            "inference_substrate": "cpu",
            "models_used": [],
            "MODEL_SPECS": [],
            "load_config_rows": [],
            "gguf_load_canary_complete_score": 0,
            "gguf_runtime_ready_score": 0,
            "random_seed": 0,
            "verifier_is_oracle": True,
            "duration_s": -1,
            "verdict_class": "null",
            "honest_verdict": "wrong",
        }
    )
    errors = set(exp.validate_artifact(mutated))
    assert {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "models_used_mismatch",
        "model_specs_mismatch",
        "gate_score_mismatch:gguf_load_canary_complete_score",
        "gate_score_mismatch:gguf_runtime_ready_score",
        "random_seed_mismatch",
        "verifier_is_oracle_mismatch",
        "duration_invalid",
        "positive_verdict_mismatch",
        "reproducibility_checksum_mismatch",
    } <= errors

    blocked = exp.build_artifact(
        run_date="20260904",
        duration_s=0,
        live_duration_s=0,
        model_specs=specs,
        preconditions={**passed, "all_passed": False},
    )
    blocked["verdict_class"] = "null"
    blocked["honest_verdict"] = "wrong"
    blocked["gate_check_summary"] = None
    assert {
        "blocked_verdict_mismatch",
        "blocked_gate_summary_incomplete",
    } <= set(exp.validate_artifact(blocked))

    null = exp.build_artifact(
        run_date="20260904",
        duration_s=0,
        live_duration_s=0,
        model_specs=specs,
        preconditions=passed,
        live_generation_rows=[
            _generation_row(model_id, terminal="failed") for model_id in exp.REQUIRED_MODEL_IDS
        ],
    )
    assert null["verdict_class"] == "null"
    null["honest_verdict"] = "wrong"
    assert "null_verdict_mismatch" in exp.validate_artifact(null)

    partial = exp.build_artifact(
        run_date="20260904",
        duration_s=0,
        live_duration_s=0,
        model_specs=specs,
        preconditions=passed,
    )
    assert partial["verdict_class"] == "partial"
    partial["honest_verdict"] = "wrong"
    assert "partial_verdict_mismatch" in exp.validate_artifact(partial)


def test_req_infra_6966_atomic_checkpoint_rejects_manifest_drift(tmp_path: Path) -> None:
    """REQ-INFRA-6966 checkpoints every accepted terminal model row."""

    path = tmp_path / "checkpoint.json"
    first = exp.checkpoint_model_row(
        path, "sha256:manifest", _generation_row(exp.REQUIRED_MODEL_IDS[0])
    )
    assert first["written"] is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["manifest_hash"] == "sha256:manifest"
    assert len(stored["rows"]) == 1

    same = exp.checkpoint_model_row(
        path, "sha256:manifest", _generation_row(exp.REQUIRED_MODEL_IDS[0])
    )
    assert same["written"] is False
    with pytest.raises(ValueError, match="checkpoint_manifest_mismatch"):
        exp.checkpoint_model_row(path, "sha256:other", _generation_row(exp.REQUIRED_MODEL_IDS[1]))
    changed = _generation_row(exp.REQUIRED_MODEL_IDS[0])
    changed["output"] = "DIFFERENT"
    with pytest.raises(ValueError, match="checkpoint_model_row_mismatch"):
        exp.checkpoint_model_row(path, "sha256:manifest", changed)


def test_req_infra_6966_atomic_writer_removes_failed_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 leaves no partial JSON when atomic replacement fails."""

    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError, match="fail"):
        exp.write_json_atomic(tmp_path / "result.json", {"receipt": True})
    assert list(tmp_path.glob("*.tmp")) == []


def test_req_infra_6966_run_blocks_without_launching_worker(tmp_path: Path) -> None:
    """REQ-INFRA-6966 writes the blocked artifact before any live attempt."""

    output = tmp_path / "artifact.json"
    launched: list[dict] = []
    artifact = exp.run(
        run_date="20260904",
        result_path=output,
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_resolved_specs(tmp_path),
        preflight_fn=lambda _specs, _path: exp.collect_preconditions(
            model_specs=_resolved_specs(tmp_path),
            checkpoint_path=tmp_path / "checkpoint.json",
            gpu_probe=lambda: _gpu_probe(foreign=True),
            llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
        ),
        attempt_runner=lambda **kwargs: launched.append(kwargs) or {},
    )
    assert launched == []
    assert output.is_file()
    assert artifact == json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(artifact) == []


def _attempt_row(model: dict, config: dict, generation: bool, *, release: bool = True) -> dict:
    """Build a deterministic child receipt for controller-path tests."""

    if generation:
        row = _generation_row(model["hf_id"])
    else:
        row = {
            "model_id": model["hf_id"],
            "config_id": config["config_id"],
            "terminal_state": "failed",
            "live_duration_s": 0.0,
        }
    row.update(
        {
            "config_id": config["config_id"],
            "vram_release_passed": release,
            "vram_release": {"model_id": model["hf_id"], "passed": release},
            "process_ownership": {"model_id": model["hf_id"], "owned": True},
            "gpu_samples": [{"pid": 77}],
            "max_gpu_utilization_pct": 90,
            "garbage_collection_ran": True,
        }
    )
    return row


def test_req_infra_6966_run_executes_ladder_then_three_promoted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 controller serializes the ladder and all three families."""

    specs = _resolved_specs(tmp_path)
    monkeypatch.setattr(
        exp,
        "embedded_tokenizer_probe",
        lambda model: {"model_id": model["hf_id"], "passed": True},
    )
    calls: list[tuple[str, str, bool]] = []

    def attempt(**kwargs: object) -> dict:
        model = dict(kwargs["model"])
        config = dict(kwargs["config"])
        generation = bool(kwargs["generation"])
        calls.append((model["hf_id"], config["config_id"], generation))
        return _attempt_row(model, config, generation)

    artifact = exp.run(
        result_path=tmp_path / "positive.json",
        checkpoint_path=tmp_path / "positive-checkpoint.json",
        model_specs=specs,
        preflight_fn=lambda _specs, _path: _passing_preflight(tmp_path),
        attempt_runner=attempt,
    )
    assert len(calls) == 6
    assert calls[0][1] == exp.EXP6962_LOAD_CONFIG["config_id"]
    assert calls[-1][0] == exp.REQUIRED_MODEL_IDS[-1]
    assert artifact["gguf_load_canary_complete_score"] == 1
    assert artifact["gguf_runtime_ready_score"] == 1
    assert len(artifact["checkpoint_rows"]) == 3
    assert exp.validate_artifact(artifact) == []


def test_req_infra_6966_run_stops_after_release_or_tokenizer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 never admits the next family after a failed safety gate."""

    specs = _resolved_specs(tmp_path)
    passed = _passing_preflight(tmp_path)
    monkeypatch.setattr(
        exp,
        "embedded_tokenizer_probe",
        lambda model: {"model_id": model["hf_id"], "passed": model["hf_id"] != specs[1]["hf_id"]},
    )
    blocked = exp.run(
        result_path=tmp_path / "tokenizer-blocked.json",
        checkpoint_path=tmp_path / "tokenizer-checkpoint.json",
        model_specs=specs,
        preflight_fn=lambda _specs, _path: passed,
        attempt_runner=lambda **_kwargs: pytest.fail("worker launched after tokenizer failure"),
    )
    assert blocked["honest_verdict"] == "blocked_gguf_load_envelope_canary"

    monkeypatch.setattr(
        exp,
        "embedded_tokenizer_probe",
        lambda model: {"model_id": model["hf_id"], "passed": True},
    )
    calls: list[dict] = []

    def release_failure(**kwargs: object) -> dict:
        calls.append(dict(kwargs))
        return _attempt_row(
            dict(kwargs["model"]),
            dict(kwargs["config"]),
            bool(kwargs["generation"]),
            release=len(calls) == 1,
        )

    partial = exp.run(
        result_path=tmp_path / "release-partial.json",
        checkpoint_path=tmp_path / "release-checkpoint.json",
        model_specs=specs,
        preflight_fn=lambda _specs, _path: passed,
        attempt_runner=release_failure,
    )
    assert len(calls) == 2
    assert partial["verdict_class"] == "partial"


def test_req_infra_6966_run_uses_default_preflight_and_rejects_invalid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 covers the public preflight path and validation fail-closed path."""

    specs = _resolved_specs(tmp_path)
    passed = _passing_preflight(tmp_path)
    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda **_kwargs: {
            **passed,
            "all_passed": False,
            "checks": [exp._check("forced", True, False, False)],
        },
    )
    blocked = exp.run(
        result_path=tmp_path / "default-preflight.json",
        checkpoint_path=tmp_path / "default-preflight-checkpoint.json",
        model_specs=specs,
    )
    assert blocked["verdict_class"] == "blocked"

    monkeypatch.setattr(
        exp,
        "embedded_tokenizer_probe",
        lambda model: {"model_id": model["hf_id"], "passed": True},
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced-invalid"])
    with pytest.raises(RuntimeError, match="artifact_validation_failed"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            checkpoint_path=tmp_path / "invalid-checkpoint.json",
            model_specs=specs,
            preflight_fn=lambda _specs, _path: passed,
            attempt_runner=lambda **kwargs: _attempt_row(
                dict(kwargs["model"]), dict(kwargs["config"]), bool(kwargs["generation"])
            ),
        )


def test_req_infra_6966_worker_closes_model_and_collects_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 worker teardown closes the model before exit."""

    class FakeLlama:
        closed = False

        def __init__(self, **kwargs: object) -> None:
            assert kwargs["n_ctx"] == 16_384
            assert kwargs["tensor_split"] == [0.5, 0.5]

        def tokenize(self, value: bytes, **_kwargs: object) -> list[int]:
            return list(range(max(1, len(value) // 4)))

        def create_completion(self, *_args: object, **kwargs: object) -> dict:
            assert kwargs["max_tokens"] == 128
            return {
                "choices": [{"text": "CANARY", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

        def close(self) -> None:
            self.closed = True

    collected: list[bool] = []
    monkeypatch.setattr(exp.gc, "collect", lambda: collected.append(True) or 0)
    row = exp.worker_execute(
        {
            "model_id": exp.REQUIRED_MODEL_IDS[0],
            "model_path": str(tmp_path / "model.gguf"),
            "config": exp.configuration_ladder()[-1],
            "generation": True,
        },
        llama_factory=FakeLlama,
        clock=iter([1_000_000_000, 2_000_000_000, 4_000_000_000]).__next__,
    )
    assert row["terminal_state"] == "complete"
    assert row["output"] == "CANARY"
    assert row["output_hash"] == exp.sha256_text("CANARY")
    assert row["live_duration_s"] == 2.0
    assert collected == [True]


def test_req_infra_6966_worker_uses_binding_and_token_count_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 worker supports its real import and missing usage count."""

    class FakeLlama:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create_completion(self, *_args: object, **_kwargs: object) -> dict:
            return {"choices": [{"text": "TOKENS"}], "usage": {}}

        def tokenize(self, _value: bytes, **_kwargs: object) -> list[int]:
            return [1, 2, 3]

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    generated = exp.worker_execute(
        {
            "model_id": exp.REQUIRED_MODEL_IDS[0],
            "model_path": str(tmp_path / "model.gguf"),
            "config": exp.configuration_ladder()[-1],
            "generation": True,
        },
        clock=iter([0, 1_000_000_000, 2_000_000_000]).__next__,
    )
    assert generated["completion_tokens"] == 3
    no_generation = exp.worker_execute(
        {
            "model_id": exp.REQUIRED_MODEL_IDS[0],
            "model_path": str(tmp_path / "model.gguf"),
            "config": exp.configuration_ladder()[0],
            "generation": False,
        },
        llama_factory=FakeLlama,
        clock=iter([0, 1]).__next__,
    )
    assert no_generation["output"] == ""


def test_req_infra_6966_worker_preserves_exception_and_still_collects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6966 failed loads keep exact exception text and teardown."""

    class BrokenLlama:
        def __init__(self, **_kwargs: object) -> None:
            raise ValueError("Failed to load model from file")

    collected: list[bool] = []
    monkeypatch.setattr(exp.gc, "collect", lambda: collected.append(True) or 0)
    row = exp.worker_execute(
        {
            "model_id": exp.REQUIRED_MODEL_IDS[0],
            "model_path": str(tmp_path / "model.gguf"),
            "config": exp.configuration_ladder()[0],
            "generation": False,
        },
        llama_factory=BrokenLlama,
    )
    assert row["terminal_state"] == "failed"
    assert row["exception_type"] == "ValueError"
    assert "Failed to load" in row["exception_message"]
    assert row["teardown_complete"] is True
    assert collected == [True]
