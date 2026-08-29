"""Focused tests for the task-owned ARC transport preflight.

Spec refs: REQ-ARC-WMTE-6752 and SCENARIO-ARC-WMTE-6752-*.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from carnot import experiment_6752_arc_code_carrying_tool_preflight as exp


def _model(tmp_path: Path, index: int) -> dict:
    spec = exp.MODEL_SPECS[index]
    path = tmp_path / spec["filename"]
    path.write_bytes(f"fixture-{spec['model_id']}".encode())
    return exp.model_receipt(spec, path, file_hasher=exp.sha256_file)


def _passing_row(tmp_path: Path, index: int) -> dict:
    model = _model(tmp_path, index)
    predicate = exp.FIND_OBJECTS_PREDICATE_CODE
    response = {
        "ok": True,
        "t": 0,
        "which": "before",
        "predicate_applied": True,
        "n_components_scanned": 3,
        "n_objects_matched": 1,
        "objects": [
            {
                "color": 2,
                "pixel_count": 3,
                "bbox": [1, 1, 2, 2],
                "centroid": [1.33, 1.33],
                "height": 2,
                "width": 2,
            }
        ],
        "truncated": False,
        "response_bytes": 320,
    }
    bounded = "<tool_response>\n" + json.dumps(response) + "\n</tool_response>"
    raw = exp.expected_xml_call()
    row = {
        **model,
        "call_shape": "find_objects_t_which_predicate_code_max_objects",
        "owned_pid": 8_000 + index,
        "server_pid": 9_000 + index,
        "assigned_device": {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 3090",
        },
        "context_requested": exp.CONTEXT_REQUESTED,
        "context_observed_by_model": exp.CONTEXT_REQUESTED,
        "gpu_layers": {"requested": 999, "offloaded": 49, "total": 49},
        "peak_vram_mb": 18_000,
        "live_model_invoked": True,
        "live_path_reached": True,
        "production_route": "induce_with_tool_loop/selfparse/dispatch_tool",
        "raw_emission_sha256": exp.sha256_text(raw),
        "parsed_tool": "find_objects",
        "parsed_arguments": {
            "t": 0,
            "which": "before",
            "predicate_code": predicate,
            "max_objects": exp.REQUESTED_MAX_OBJECTS,
        },
        "dispatch_result": response,
        "bounded_response_bytes": len(bounded.encode()),
        "bounded_response_sha256": exp.sha256_text(bounded),
        "latency_s": 1.25,
        "failure_class": None,
        "transcript_sha256": exp.sha256_json([raw, bounded]),
        "process_exit_code": 0,
        "solve_claim": False,
    }
    row["row_sha256"] = exp.row_checksum(row)
    return row


def _passing_preflight(models: list[dict]) -> dict:
    checks = [
        {"check": "llama_cpp_cuda_offload", "expected": True, "observed": True, "passed": True},
        {
            "check": "arc_registry_no_solve_target",
            "expected": True,
            "observed": True,
            "passed": True,
        },
    ]
    for model in models:
        checks.extend(
            [
                {
                    "check": f"cache:{model['model_id']}",
                    "expected": True,
                    "observed": model["resolved"],
                    "passed": model["resolved"],
                },
                {
                    "check": f"free_vram:{model['model_id']}",
                    "expected": {"at_least_mb": 100},
                    "observed": 24_000,
                    "passed": True,
                },
            ]
        )
    return {"all_passed": True, "checks": checks, "gpu_inventory": []}


def test_req_arc_wmte_6752_fixed_models_fields_and_fixture() -> None:
    """REQ-ARC-WMTE-6752 pins the scored generator, canary, fields, and no-game fixture."""
    assert [(row["model_id"], row["role"]) for row in exp.MODEL_SPECS] == [
        ("unsloth/Qwen3.8-27B-GGUF", "immutable_scored_arc_generator"),
        ("unsloth/Qwen3.6-35B-A3B-GGUF", "flagship_moe_transport_canary"),
    ]
    assert exp.CONTEXT_REQUESTED == 32_768
    assert exp.INFERENCE_SUBSTRATE == "local llama.cpp CUDA GGUF production tool route"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(exp.FIELD_PRINCIPLES)
    assert {key for key in exp.FIELD_PRINCIPLES if key.startswith("gate:")} == {
        "gate:exact_models",
        "gate:cuda_offload",
        "gate:cached_paths",
        "gate:free_vram",
        "gate:registry_no_solve_target",
        "gate:owned_32k",
        "gate:multi_parameter_parse",
        "gate:dispatch",
        "gate:bounded_response",
        "gate:no_solve",
    }
    manifest = exp.fixture_manifest()
    encoded = exp.canonical_json(manifest).lower()
    assert manifest["source_class"] == "synthetic_bounded_transition_fixture"
    assert all(token not in encoded for token in ("game_id", "bfs", "solve_trace", "adapter"))
    assert exp.fixture_checksum() == exp.sha256_json(manifest)


def test_scenario_arc_wmte_6752_owned_context_environment(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6752-OWNED-CONTEXT fixes 32K before proposer construction."""
    model = _model(tmp_path, 0)
    env = exp.worker_environment(
        {"KEEP": "yes", "CARNOT_ARC_INDUCE_N_CTX": "1"}, model, device_index=1
    )
    assert env["KEEP"] == "yes"
    assert env["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert env["CARNOT_ARC_GENERATOR_CUDA_GPU"] == "1"
    assert env["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] == "1"
    assert env["CARNOT_ARC_GGUF_PATH"] == model["model_path"]
    assert env["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "1"
    assert env["CARNOT_ARC_GENERATOR_SEED"] == str(exp.RANDOM_SEED)


def test_scenario_arc_wmte_6752_prompt_demands_live_typed_code_call() -> None:
    """SCENARIO-ARC-WMTE-6752-TYPED-CODE-DISPATCH freezes the exact call shape."""
    instruction = exp.build_probe_instruction()
    assert "FIRST AND ONLY tool call" in instruction
    assert "<function=find_objects>" in instruction
    assert "<parameter=t>" in instruction
    assert "<parameter=which>" in instruction
    assert "<parameter=predicate_code>" in instruction
    assert exp.FIND_OBJECTS_PREDICATE_CODE in instruction
    assert f"<parameter=max_objects>\n{exp.REQUESTED_MAX_OBJECTS}" in instruction
    assert exp.expected_xml_call() in instruction


def test_scenario_arc_wmte_6752_row_validation_and_readiness(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6752-READY-AND-NO-SOLVE requires both exact live CUDA rows."""
    rows = [_passing_row(tmp_path, 0), _passing_row(tmp_path, 1)]
    assert all(exp.model_row_errors(row) == [] for row in rows)
    assert exp.reduce_ready(rows) is True

    attacks = {
        "context_observed_by_model": 16_384,
        "peak_vram_mb": 0,
        "parsed_tool": "list_transitions",
        "live_path_reached": False,
        "solve_claim": True,
    }
    for field, value in attacks.items():
        mutated = deepcopy(rows)
        mutated[0][field] = value
        mutated[0]["row_sha256"] = exp.row_checksum(mutated[0])
        assert exp.model_row_errors(mutated[0])
        assert exp.reduce_ready(mutated) is False


def test_req_arc_wmte_6752_blocked_preflight_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6752 records owned blocks and never invokes a substitute worker."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    calls: list[str] = []

    def blocked(_: list[dict]) -> dict:
        check = {
            "check": "free_vram:unsloth/Qwen3.8-27B-GGUF",
            "expected": {"at_least_mb": 20_000},
            "observed": 10_000,
            "passed": False,
        }
        return {"all_passed": False, "checks": [check], "gpu_inventory": []}

    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        resolver=lambda: models,
        preflight_fn=blocked,
        worker_runner=lambda model: calls.append(model["model_id"]),
        clock=iter((1_000, 2_000, 3_000)).__next__,
    )
    assert calls == []
    assert len(artifact["rows"]) == len(exp.MODEL_SPECS)
    assert all(row["failure_class"].startswith("preflight_blocked:") for row in artifact["rows"])
    assert artifact["honest_verdict"].startswith("complete_blocked_arc_transport")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == blocked(models)["checks"]
    assert artifact["arc_context_tool_preflight_ready"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["solve_claim"] is False
    assert exp.validate_artifact(artifact) == []
    assert json.loads((tmp_path / "blocked.json").read_text()) == artifact


def test_req_arc_wmte_6752_partial_rows_are_valid_terminal_evidence(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6752 writes honest live failure rows instead of rejecting the artifact."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    rows = [_passing_row(tmp_path, 0), _passing_row(tmp_path, 1)]
    rows[1]["parsed_tool"] = None
    rows[1]["failure_class"] = "no_live_find_objects_call"
    rows[1]["row_sha256"] = exp.row_checksum(rows[1])

    artifact = exp.run(
        result_path=tmp_path / "partial.json",
        resolver=lambda: models,
        preflight_fn=_passing_preflight,
        worker_runner=lambda model: deepcopy(
            rows[0 if model["model_id"] == rows[0]["model_id"] else 1]
        ),
        clock=iter((1_000, 2_000)).__next__,
    )
    assert artifact["verdict_class"] == "partial"
    assert artifact["arc_context_tool_preflight_ready"] is False
    assert artifact["live_model_invoked"] is True
    assert artifact["live_path_reached"] is True
    assert exp.validate_artifact(artifact) == []


def test_req_arc_wmte_6752_complete_artifact_counts_rows(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6752 reduces only retained rows and preserves the no-solve boundary."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    rows = [_passing_row(tmp_path, 0), _passing_row(tmp_path, 1)]
    artifact = exp.run(
        result_path=tmp_path / "complete.json",
        resolver=lambda: models,
        preflight_fn=_passing_preflight,
        worker_runner=lambda model: deepcopy(
            rows[0 if model["model_id"] == rows[0]["model_id"] else 1]
        ),
        clock=iter((1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000)).__next__,
    )
    assert artifact["arc_context_tool_preflight_ready"] is True
    assert artifact["multi_parameter_parse_successes"] == 2
    assert artifact["multi_parameter_dispatch_successes"] == 2
    assert artifact["bounded_response_successes"] == 2
    assert artifact["context_observed_by_model"] == {
        row["model_id"]: exp.CONTEXT_REQUESTED for row in rows
    }
    assert len(artifact["gpu_admission_by_model"]) == 2
    assert artifact["live_model_invoked"] is True
    assert artifact["live_path_reached"] is True
    assert artifact["solve_claim"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete")
    assert set(artifact).issubset(artifact["field_principles"])
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []


def test_scenario_arc_wmte_6752_fresh_subprocess_receives_owned_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-6752-OWNED-CONTEXT uses a new interpreter per model."""
    model = _model(tmp_path, 0)
    row = _passing_row(tmp_path, 0)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output_index = command.index("--worker-output") + 1
        Path(command[output_index]).write_text(json.dumps(row))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", fake_run)
    observed = exp.run_model_subprocess(model, device_index=1, timeout_s=10)
    assert "--worker" in captured["command"]
    assert captured["env"]["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert captured["env"]["CARNOT_ARC_GENERATOR_CUDA_GPU"] == "1"
    assert observed["model_id"] == model["model_id"]


def test_req_arc_wmte_6752_validator_rejects_hash_and_claim_attacks(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6752 fails closed on evidence drift or a solve claim."""
    rows = [_passing_row(tmp_path, 0), _passing_row(tmp_path, 1)]
    artifact = exp.build_artifact(
        rows=rows,
        preflight=_passing_preflight(rows),
        started_ns=1_000,
        finished_ns=2_000,
    )
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["raw_emission_sha256"] = "sha256:changed"
    assert exp.validate_artifact(changed)
    solve = deepcopy(artifact)
    solve["solve_claim"] = True
    solve["reproducibility_checksum"] = exp.artifact_checksum(solve)
    assert "solve_claim_must_be_false" in exp.validate_artifact(solve)


def test_req_arc_wmte_6752_fixture_and_cache_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6752 converts the fixture and resolves only exact cached names."""
    transitions = exp.fixture_transitions()
    assert len(transitions) == 1
    assert transitions[0].grid.shape == (8, 8)
    assert int(transitions[0].next_grid[1, 1]) == 4

    paths = {}
    for spec in exp.MODEL_SPECS:
        path = tmp_path / spec["filename"]
        path.write_bytes(spec["model_id"].encode())
        paths[spec["repo_substr"]] = str(path)
    from carnot.agentic import arc_executable_world_model as awm

    monkeypatch.setattr(awm, "_resolve_gguf", lambda name: paths.get(name))
    monkeypatch.setattr(exp, "sha256_file", lambda path: exp.sha256_text(Path(path).name))
    resolved = exp.resolve_model_specs()
    assert all(row["resolved"] for row in resolved)
    assert [row["model_id"] for row in resolved] == [row["model_id"] for row in exp.MODEL_SPECS]

    wrong = exp.model_receipt(exp.MODEL_SPECS[0], tmp_path / "absent.gguf")
    assert wrong["resolved"] is False
    assert wrong["model_sha256"] is None


def test_req_arc_wmte_6752_host_probes_and_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6752 preserves host failures and computes every preflight gate."""
    stdout = (
        "0, GPU-zero, NVIDIA GeForce RTX 3090, 24576, 4, 24120\n"
        "bad,row\n"
        "x, GPU-bad, name, total, used, free\n"
    )
    completed = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: completed)
    assert exp._run_text_command(["probe"])["ok"] is True
    assert exp.nvidia_smi_inventory()["devices"][0]["uuid"] == "GPU-zero"

    def raise_oserror(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(exp.subprocess, "run", raise_oserror)
    assert exp._run_text_command(["missing"])["ok"] is False

    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    for model in models:
        model["required_vram_mb"] = 100
    registry = tmp_path / "registry.yaml"
    registry.write_text("games: {}\n")
    monkeypatch.setattr(exp, "REGISTRY_PATH", registry)
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {
            "ok": True,
            "devices": [{"index": 0, "memory_free_mb": 24_000}],
            "stdout": "",
            "stderr": "",
        },
    )
    passed = exp.live_preflight(models)
    assert passed["all_passed"] is True

    real_import = builtins.__import__

    def fail_llama(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("fixture")
        return real_import(name, *args, **kwargs)

    missing_registry = tmp_path / "missing.yaml"
    monkeypatch.setattr(exp, "REGISTRY_PATH", missing_registry)
    models[0]["resolved"] = False
    models[0]["device_index"] = 99
    monkeypatch.setattr(builtins, "__import__", fail_llama)
    blocked = exp.live_preflight(models[:1])
    assert blocked["all_passed"] is False
    assert any(check["observed"] is None for check in blocked["checks"])


def test_req_arc_wmte_6752_gpu_receipt_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-6752 binds PID memory, layer logs, and physical device identity."""
    receipt = {
        "stdout": "9000, 1024\nbad\nx, nope\n9000, 2048\n",
        "ok": True,
    }
    monkeypatch.setattr(exp, "_run_text_command", lambda command: receipt)
    assert exp._pid_vram_mb(9000) == 2048
    assert exp._pid_vram_mb(1) == 0
    assert exp._gpu_layers_from_log("offloaded 49/50 layers to GPU", 999) == {
        "requested": 999,
        "offloaded": 49,
        "total": 50,
    }
    assert exp._gpu_layers_from_log("offloaded 12 repeating layers", 999)["total"] == 12
    assert exp._gpu_layers_from_log("no layer receipt", 999)["offloaded"] == 0
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {"devices": [{"index": 1, "uuid": "GPU-one", "name": "RTX 3090"}]},
    )
    assert exp._assigned_device(1)["uuid"] == "GPU-one"
    assert exp._assigned_device(2)["uuid"] is None


class _LiveFakeProposer:
    """Small proposer double for live-worker control-flow coverage."""

    def __init__(self, **kwargs):
        self.n_gpu_layers = kwargs["n_gpu_layers"]
        self._proc = SimpleNamespace(pid=9000)
        self.last_tool_loop_stats = {"turns": 1}
        self._stderr_log_path = None
        self.ensure = True
        self.healthy = True
        self.stopped = False

    def _ensure_server(self):
        return self.ensure

    def _healthy(self):
        return self.healthy

    def observed_n_ctx(self):
        return exp.CONTEXT_REQUESTED

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize(
    ("event_kind", "failure"),
    [
        ("success", None),
        ("missing", "no_live_find_objects_call"),
        ("dispatch", "dispatch_failure"),
        ("oversized", "response_bound_failure"),
    ],
)
def test_req_arc_wmte_6752_live_worker_reduces_transport_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_kind: str,
    failure: str | None,
) -> None:
    """REQ-ARC-WMTE-6752 retains success and each production transport failure class."""
    from carnot.agentic import arc_executable_world_model as awm
    from carnot.agentic import arc_induction_tool_loop as loop

    log = tmp_path / "server.log"
    log.write_text("offloaded 49/49 layers to GPU")
    made = []

    def factory(**kwargs):
        proposer = _LiveFakeProposer(**kwargs)
        proposer._stderr_log_path = log
        made.append(proposer)
        return proposer

    def fake_loop(proposer, game, transitions, cell, **kwargs):
        assert game == "transport_fixture"
        assert transitions[0].grid.shape == (8, 8)
        time.sleep(0.12)
        sink = kwargs["tool_event_sink"]
        if event_kind != "missing":
            result = {"ok": event_kind != "dispatch", "objects": []}
            bounded = "x" * (
                exp.MAX_FIND_OBJECT_RESPONSE_BYTES + 65 if event_kind == "oversized" else 20
            )
            sink.append(
                {
                    "parsed_tool": "find_objects",
                    "raw_emission": exp.expected_xml_call(),
                    "parsed_arguments": {
                        "t": 0,
                        "which": "before",
                        "predicate_code": exp.FIND_OBJECTS_PREDICATE_CODE,
                        "max_objects": exp.REQUESTED_MAX_OBJECTS,
                    },
                    "dispatch_result": result,
                    "bounded_response": bounded,
                }
            )
        return False, "probe complete"

    monkeypatch.setattr(awm, "LocalGGUFProposer", factory)
    monkeypatch.setattr(awm, "_free_port", lambda: 12345)
    monkeypatch.setattr(loop, "induce_with_tool_loop", fake_loop)
    monkeypatch.setattr(exp, "_pid_vram_mb", lambda pid: 1024)
    monkeypatch.setattr(
        exp,
        "_assigned_device",
        lambda index: {"physical_index": index, "uuid": "GPU", "name": "RTX"},
    )
    row = exp.run_live_worker(_model(tmp_path, 0), device_index=0)
    assert row["failure_class"] == failure
    assert row["live_path_reached"] is True
    assert row["gpu_layers"]["offloaded"] == 49
    assert made[0].stopped is True


def test_req_arc_wmte_6752_live_worker_handles_admission_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6752 turns server admission and worker exceptions into rows."""
    from carnot.agentic import arc_executable_world_model as awm

    made = []

    def factory(**kwargs):
        proposer = _LiveFakeProposer(**kwargs)
        proposer.ensure = False
        proposer.healthy = False
        proposer._stderr_log_path = tmp_path / "missing.log"
        made.append(proposer)
        return proposer

    monkeypatch.setattr(awm, "LocalGGUFProposer", factory)
    monkeypatch.setattr(awm, "_free_port", lambda: 12345)
    monkeypatch.setattr(exp, "_assigned_device", lambda index: {})
    row = exp.run_live_worker(_model(tmp_path, 0), device_index=0)
    assert row["failure_class"] == "cuda_admission_or_worker_failure"
    assert row["context_observed_by_model"] is None

    def boom(**kwargs):
        proposer = _LiveFakeProposer(**kwargs)

        def raise_ensure():
            raise RuntimeError("boom")

        proposer._ensure_server = raise_ensure
        return proposer

    monkeypatch.setattr(awm, "LocalGGUFProposer", boom)
    row = exp.run_live_worker(_model(tmp_path, 0), device_index=0)
    assert row["failure_class"] == "cuda_admission_or_worker_failure"


@pytest.mark.parametrize("failure_kind", ["timeout", "oserror", "missing", "bad_json"])
def test_req_arc_wmte_6752_subprocess_failures_are_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """REQ-ARC-WMTE-6752 keeps the denominator for every subprocess failure shape."""
    model = _model(tmp_path, 0)

    def fake_run(command, **kwargs):
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure_kind == "oserror":
            raise OSError("boom")
        output = Path(command[command.index("--worker-output") + 1])
        if failure_kind == "bad_json":
            output.write_text("{")
        return SimpleNamespace(returncode=2, stdout="stdout", stderr="stderr")

    monkeypatch.setattr(exp.subprocess, "run", fake_run)
    row = exp.run_model_subprocess(model, timeout_s=1)
    assert row["failure_class"]
    assert row["row_sha256"] == exp.row_checksum(row)


def test_req_arc_wmte_6752_row_and_artifact_attack_coverage(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6752 names every malformed row and artifact field independently."""
    row = _passing_row(tmp_path, 0)
    broken = deepcopy(row)
    broken.update(
        {
            "model_id": "other",
            "context_requested": 1,
            "context_observed_by_model": None,
            "gpu_layers": None,
            "peak_vram_mb": None,
            "live_model_invoked": False,
            "live_path_reached": False,
            "production_route": "helper",
            "parsed_tool": "other",
            "parsed_arguments": None,
            "dispatch_result": None,
            "bounded_response_bytes": 0,
            "failure_class": "failed",
            "process_exit_code": 2,
            "solve_claim": True,
        }
    )
    assert set(exp.model_row_errors(broken)) == {
        "row_sha256",
        "model_id",
        "context_requested",
        "context_observed_by_model",
        "gpu_layers",
        "peak_vram_mb",
        "live_model_invoked",
        "production_route",
        "parsed_tool",
        "parsed_arguments",
        "dispatch_result",
        "bounded_response",
        "failure_class",
        "process_exit_code",
        "solve_claim",
    }

    malformed_evidence = deepcopy(row)
    malformed_evidence.update(
        {
            "model_id": "other",
            "context_requested": 1,
            "call_shape": "other",
            "raw_emission_sha256": None,
            "bounded_response_sha256": "sha256:short",
            "transcript_sha256": 7,
            "latency_s": -1,
            "solve_claim": True,
        }
    )
    assert set(exp.row_evidence_errors(malformed_evidence)) == {
        "row_sha256",
        "model_id",
        "context_requested",
        "call_shape",
        "raw_emission_sha256",
        "bounded_response_sha256",
        "transcript_sha256",
        "latency_s",
        "solve_claim",
    }

    rows = [_passing_row(tmp_path, 0), _passing_row(tmp_path, 1)]
    artifact = exp.build_artifact(
        rows=rows[:-1],
        preflight=_passing_preflight(rows),
        started_ns=2_000,
        finished_ns=1_000,
    )
    assert artifact["verdict_class"] == "partial"
    assert artifact["duration_s"] == 0
    attacks = {
        "missing": lambda value: value.pop("honest_verdict"),
        "field_principles": lambda value: value.update(field_principles={}),
        "gate_principles": lambda value: value["field_principles"].pop("gate:no_solve"),
        "substrate": lambda value: value.update(inference_substrate="cpu"),
        "context": lambda value: value.update(context_requested=1),
        "verdict": lambda value: value.update(verdict_class="other"),
        "ready": lambda value: value.update(arc_context_tool_preflight_ready=True),
        "parse": lambda value: value.update(multi_parameter_parse_successes=99),
        "dispatch": lambda value: value.update(multi_parameter_dispatch_successes=99),
        "bounded": lambda value: value.update(bounded_response_successes=99),
        "fixture": lambda value: value.update(fixture_checksum="changed"),
        "duration": lambda value: value.update(duration_s=-1),
        "models": lambda value: value.update(models_used=[]),
        "contexts": lambda value: value.update(context_observed_by_model={}),
        "gpu_receipts": lambda value: value.update(gpu_admission_by_model={}),
        "live_decode": lambda value: value.update(live_model_invoked=False),
        "live_path": lambda value: value.update(live_path_reached=False),
        "gate_summary": lambda value: value.update(gate_check_summary=[]),
        "honest": lambda value: value.update(honest_verdict="complete_ready"),
    }
    expected = {
        "missing": "missing_field:honest_verdict",
        "field_principles": "field_principles_incomplete",
        "gate_principles": "gate_principles_incomplete",
        "substrate": "inference_substrate",
        "context": "context_requested",
        "verdict": "verdict_class",
        "ready": "ready_reduction",
        "parse": "parse_count",
        "dispatch": "dispatch_count",
        "bounded": "bounded_count",
        "fixture": "fixture_checksum",
        "duration": "duration_s",
        "models": "models_used",
        "contexts": "context_observed_by_model",
        "gpu_receipts": "gpu_admission_by_model",
        "live_decode": "live_model_invoked",
        "live_path": "live_path_reached",
        "gate_summary": "gate_check_summary",
        "honest": "honest_verdict",
    }
    for name, attack in attacks.items():
        mutated = deepcopy(artifact)
        attack(mutated)
        assert expected[name] in exp.validate_artifact(mutated)


def test_req_arc_wmte_6752_invalid_run_worker_entry_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-6752 validates before write and covers both explicit CLI modes."""
    models = [_model(tmp_path, 0), _model(tmp_path, 1)]
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp6752 artifact"):
        exp.run(
            result_path=tmp_path / "bad.json",
            resolver=lambda: models,
            preflight_fn=lambda value: {"all_passed": False, "checks": []},
            clock=iter((1, 2)).__next__,
        )

    row = _passing_row(tmp_path, 0)
    monkeypatch.setattr(exp, "run_live_worker", lambda model, device_index: row)
    output = tmp_path / "worker.json"
    assert exp._worker_entry(json.dumps(models[0]), output, 0) == 0
    assert json.loads(output.read_text())["model_id"] == row["model_id"]

    monkeypatch.setattr(exp, "_worker_entry", lambda *args: 7)
    assert (
        exp.main(
            [
                "--worker",
                "--worker-output",
                str(output),
                "--model-json",
                json.dumps(models[0]),
            ]
        )
        == 7
    )
    with pytest.raises(SystemExit):
        exp.main(["--worker"])

    monkeypatch.setattr(
        exp,
        "run",
        lambda: {
            "arc_context_tool_preflight_ready": True,
            "honest_verdict": "complete",
        },
    )
    assert exp.main([]) == 0
    assert '"ready": true' in capsys.readouterr().out
