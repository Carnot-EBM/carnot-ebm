"""Tests for Exp 5119 SOTA endpoint root-cause diagnostics.

Spec refs: REQ-INFER-SOTA-028,
SCENARIO-INFER-SOTA-028-SUCCESS,
SCENARIO-INFER-SOTA-028-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5119_sota_endpoint_rootcause as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "llm-ebm-inference" / "spec.md"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_models(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    sizes = {QWEN: 30, GEMMA31: 20, GEMMA26: 10}
    for hf_id, size in sizes.items():
        path = tmp_path / "models" / hf_id.split("/", 1)[1] / f"{hf_id.split('/', 1)[1]}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * size, encoding="utf-8")
        paths[hf_id] = path.as_posix()
    return paths


def _resolver(paths: dict[str, str]) -> mod.ModelResolver:
    return lambda hf_id, preferred_quant: paths.get(hf_id)


def _cached_pair(paths: dict[str, str]) -> mod.CachedPairFn:
    def fake(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": QWEN,
                "gpu": gpu_indices[0],
                "model_path": paths[QWEN],
                "preferred_quant": preferred_quant,
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": GEMMA26,
                "gpu": gpu_indices[1],
                "model_path": paths[GEMMA26],
                "preferred_quant": preferred_quant,
            },
        ]

    return fake


def _preconditions(*, cuda: bool = True) -> dict[str, Any]:
    return {
        "cuda_status": {
            "cuda_available": cuda,
            "gpu_count": 2 if cuda else 0,
            "gpus": [
                {"index": 0, "name": "RTX 3090", "free_vram_mb": 22000},
                {"index": 1, "name": "RTX 3090", "free_vram_mb": 21900},
            ]
            if cuda
            else [],
        },
        "llama_cpp_python": {"available": True, "detail": "llama_cpp import ok"},
        "disk_ram": {
            "disk_free_gib": 100.0,
            "ram_available_gib": 64.0,
        },
    }


def _server_unavailable(_env: dict[str, str]) -> dict[str, Any]:
    return {
        "available": False,
        "selected_path": None,
        "candidates": [
            {
                "source": "test",
                "path": "/missing/llama-server",
                "exists": False,
                "is_file": False,
                "executable": False,
            }
        ],
        "missing_diagnostic": "llama-server binary not found or not executable",
    }


def _server_available(path: Path) -> dict[str, Any]:
    return {
        "available": True,
        "selected_path": path.as_posix(),
        "candidates": [
            {
                "source": "test",
                "path": path.as_posix(),
                "exists": True,
                "is_file": True,
                "executable": True,
            }
        ],
        "missing_diagnostic": None,
    }


def _free_port(port: int = 45119) -> dict[str, Any]:
    return {
        "available": True,
        "host": "127.0.0.1",
        "port": port,
        "endpoint_url": f"http://127.0.0.1:{port}",
        "error": None,
    }


def _blocked_probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
    return {
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
                "completion_probe": {"ready": False, "detail": "connection refused"},
                "telemetry_probe": {"ready": False, "detail": "completion probe failed"},
            }
        ],
    }


def _ready_probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "candidate_endpoints": list(endpoints),
        "selected_endpoint": endpoints[0],
        "completion_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "duration_s": 0.25,
        "probes": [],
    }


def _ready_sample(endpoint: str, timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "ready": True,
        "route": endpoint.rstrip("/") + "/completion",
        "status": 200,
        "completion_text": "exp5119 endpoint live",
        "logprob_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "evidence": {
            "token_logprob_count": 2,
            "top_logprob_row_count": 2,
            "token_logprobs": [-0.11, -0.22],
            "top_logprobs": [{" exp": -0.11, " run": -1.3}, {" live": -0.22}],
            "raw_response_keys": ["content", "completion_probabilities"],
        },
        "error": None,
    }


def _clean_adversarial(_path: Path) -> dict[str, Any]:
    return {"flags": [], "summary": {"critical_count": 0}}


def test_req_infer_sota_028_spec_declares_exp5119_contract() -> None:
    """REQ-INFER-SOTA-028: OpenSpec anchors fields, blockers, paths, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-INFER-SOTA-028",
        "SCENARIO-INFER-SOTA-028-SUCCESS",
        "SCENARIO-INFER-SOTA-028-BLOCKED",
        "results/experiment_5119_sota_endpoint_rootcause_v469.json",
        mod.EXPERIMENT_ID,
        mod.MILESTONE,
        QWEN,
        GEMMA31,
        GEMMA26,
        "AutoTokenizer",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for blocker in mod.ROOT_CAUSE_BLOCKERS:
        assert blocker in spec
    assert "AutoTokenizer" not in Path(mod.__file__).read_text(encoding="utf-8")


def test_scenario_infer_sota_028_blocked_records_root_cause_without_cache_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-028-BLOCKED: no live logprobs means no cache readiness."""

    paths = _write_models(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_blocked_probe,
        endpoint_sample=lambda endpoint, timeout_s: pytest.fail("sample must not run"),
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=_clean_adversarial,
        now=iter([10.0, 12.5]).__next__,
        duration_floor_s=0.0,
        write=True,
        tests_run=[{"command": "pytest tests/python/test_experiment_5119_sota_endpoint_rootcause.py", "status": "passed"}],
    )

    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == "blocked_sota_endpoint_rootcause_no_live_logprobs"
    assert artifact["inference_substrate"].startswith("precondition_check_only")
    assert artifact["cache_ready"] is False
    assert artifact["completion_proof"]["ready"] is False
    assert artifact["logprob_proof"]["ready"] is False
    assert artifact["server_command"] is None
    assert artifact["endpoint_lifetime_s"] == 0.0
    assert artifact["cached_sota_pair_attempted"] is True
    assert set(artifact["gguf_paths"]) == {QWEN, GEMMA31, GEMMA26}
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["cuda_status"]["cuda_available"] is True
    assert artifact["root_cause_tree"]["missing_binary"]["present"] is True
    assert artifact["root_cause_tree"]["unsupported_logprob_api"]["present"] is True
    assert artifact["root_cause_tree"]["wrong_model_path"]["present"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["tests_run"][0]["status"] == "passed"
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_infer_sota_028_success_records_transcript_and_cache_readiness(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-028-SUCCESS: live top-logprobs prove local cache readiness."""

    paths = _write_models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    start_calls: list[dict[str, Any]] = []
    cleanup_calls: list[Any] = []

    class FakeProcess:
        pid = 5119

        def poll(self) -> None:
            return None

    def start(command: list[str], env: dict[str, str], log_path: Path) -> FakeProcess:
        start_calls.append({"command": command, "env": env, "log_path": log_path})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("llama server ready\n", encoding="utf-8")
        return FakeProcess()

    def probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
        if endpoints == ["http://127.0.0.1:45119"]:
            return _ready_probe(endpoints, timeout_s)
        return _blocked_probe(endpoints, timeout_s)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=probe,
        endpoint_sample=_ready_sample,
        server_finder=lambda env: _server_available(server),
        free_port=lambda host: _free_port(45119),
        server_start=start,
        server_cleanup=lambda process: cleanup_calls.append(process) or {
            "started_by_preflight": True,
            "terminated": True,
            "returncode": 0,
        },
        adversarial_verify=_clean_adversarial,
        now=iter([100.0, 170.0]).__next__,
        duration_floor_s=60.0,
        write=True,
        tests_run=[{"command": "pytest tests/python -q", "status": "passed"}],
    )

    assert artifact["honest_verdict"] == "success_sota_endpoint_rootcause_live_logprobs"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["completion_proof"]["ready"] is True
    assert artifact["completion_proof"]["text"] == "exp5119 endpoint live"
    assert artifact["logprob_proof"]["ready"] is True
    assert artifact["logprob_proof"]["token_logprob_count"] == 2
    assert artifact["cache_ready"] is True
    assert artifact["endpoint_url"] == "http://127.0.0.1:45119"
    assert artifact["endpoint_lifetime_s"] == pytest.approx(70.0)
    assert artifact["server_pid"] == 5119
    assert artifact["server_command"] == start_calls[0]["command"]
    assert artifact["server_command"][0] == server.as_posix()
    assert artifact["server_command"][1:3] == ["-m", paths[GEMMA26]]
    assert artifact["startup_log"]["tail"] == "llama server ready\n"
    assert artifact["shutdown_behavior"]["terminated"] is True
    assert cleanup_calls and cleanup_calls[0].pid == 5119
    assert artifact["request_response_transcript"]["completion_request"]["endpoint"] == (
        "http://127.0.0.1:45119/completion"
    )
    assert artifact["root_cause_tree"]["summary"] == "no_blocker_live_logprobs_observed"
    assert artifact["flagged_adversarial"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_infer_sota_028_root_cause_tree_distinguishes_blockers() -> None:
    """REQ-INFER-SOTA-028: root-cause classes are machine-readable and independent."""

    tree = mod.build_root_cause_tree(
        server={"available": False, "missing_diagnostic": "missing"},
        model_specs=[
            {"hf_id": QWEN, "resolved_path": "/missing.gguf", "cache_status": "missing"},
        ],
        cuda_status={"cuda_available": False, "error": "driver missing"},
        sample={"ready": False, "error": "timed out: CUDA out of memory; schema mismatch"},
        completion_ready=False,
        logprob_ready=False,
        duration_s=1.0,
    )

    assert set(mod.ROOT_CAUSE_BLOCKERS) <= set(tree)
    assert tree["missing_binary"]["present"] is True
    assert tree["wrong_model_path"]["present"] is True
    assert tree["unsupported_logprob_api"]["present"] is True
    assert tree["cuda_failure"]["present"] is True
    assert tree["oom"]["present"] is True
    assert tree["timeout"]["present"] is True
    assert tree["cache_schema_mismatch"]["present"] is True
    assert tree["summary"] == "blocked_missing_binary"


def test_req_infer_sota_028_helper_edges_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-028: helper edge cases stay deterministic and inspectable."""

    assert mod._finite_float(None) is None
    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    assert mod._select_bringup_model([]) is None
    assert mod._critical_flags({"flags": "not-a-list"}) == []

    evidence = mod._sample_evidence(
        {
            "evidence": {
                "token_logprobs": [None, True, "bad", "-0.5", "nan"],
                "top_logprobs": [{"bad": "value", " ok": -1.25}, "ignored"],
                "raw_response_keys": ("content", "completion_probabilities"),
            }
        }
    )
    assert evidence["token_logprob_count"] == 1
    assert evidence["top_logprob_row_count"] == 1
    assert evidence["token_logprobs"] == [-0.5]
    assert evidence["top_logprobs"] == [{" ok": -1.25}]
    assert evidence["raw_response_keys"] == ["content", "completion_probabilities"]

    cache = tmp_path / ".pytest_cache" / "v" / "cache" / "lastfailed"
    cache.parent.mkdir(parents=True)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    cache.write_text("not json", encoding="utf-8")
    invalid_rows = mod._default_tests_run()
    assert invalid_rows[-1]["pytest_cache_lastfailed_count"] is None

    cache.write_text('{"a": true, "b": false}', encoding="utf-8")
    valid_rows = mod._default_tests_run()
    assert valid_rows[-1]["pytest_cache_lastfailed_count"] == 2


def test_scenario_infer_sota_028_immediate_endpoint_short_duration_is_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFER-SOTA-028-SUCCESS: too-short live proof is blocked."""

    paths = _write_models(tmp_path)
    duration_floor_calls: list[dict[str, Any]] = []

    def duration_floor_probe(
        endpoint: str,
        *,
        run_started_s: float,
        target_duration_s: float,
        timeout_s: float,
        max_probes: int,
    ) -> dict[str, Any]:
        duration_floor_calls.append(
            {
                "endpoint": endpoint,
                "run_started_s": run_started_s,
                "target_duration_s": target_duration_s,
                "timeout_s": timeout_s,
                "max_probes": max_probes,
            }
        )
        return {"checked": True, "endpoint": endpoint}

    monkeypatch.setattr(mod.exp3013, "_run_duration_floor_endpoint_probe", duration_floor_probe)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_ready_probe,
        endpoint_sample=_ready_sample,
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=_clean_adversarial,
        env={"CARNOT_5119_ENDPOINTS": "http://ready.test,http://ready.test/"},
        duration_floor_s=999999.0,
        tests_run=None,
        write=True,
    )

    assert duration_floor_calls
    assert duration_floor_calls[0]["endpoint"] == "http://ready.test"
    assert artifact["duration_floor_evidence"] == {"checked": True, "endpoint": "http://ready.test"}
    assert artifact["preconditions_checked"]["environment_variables"]["CARNOT_5119_ENDPOINTS"] == (
        "http://ready.test,http://ready.test/"
    )
    assert artifact["endpoint_url"] == "http://ready.test"
    assert artifact["cache_ready"] is False
    assert artifact["cache_rows_written"] == 0
    assert artifact["flagged_adversarial"] is True
    assert artifact["honest_verdict"] == "blocked_sota_endpoint_rootcause_adversarial_flag"
    assert artifact["root_cause_tree"]["summary"] == "blocked_adversarial_verify"
    assert artifact["root_cause_tree"]["adversarial_verify"]["present"] is True


def test_scenario_infer_sota_028_server_exit_records_error_without_cache_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-028-BLOCKED: early server exit is explicit root-cause evidence."""

    paths = _write_models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    cleanup_calls: list[Any] = []

    class DeadProcess:
        pid = 51190

        def poll(self) -> int:
            return 1

    def start(command: list[str], env: dict[str, str], log_path: Path) -> DeadProcess:
        del command, env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("server exited\n", encoding="utf-8")
        return DeadProcess()

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_blocked_probe,
        endpoint_sample=lambda endpoint, timeout_s: pytest.fail("sample must not run"),
        server_finder=lambda env: _server_available(server),
        free_port=lambda host: _free_port(45190),
        server_start=start,
        server_cleanup=lambda process: cleanup_calls.append(process) or {
            "started_by_preflight": True,
            "terminated": False,
            "returncode": 1,
        },
        adversarial_verify=_clean_adversarial,
        now=iter([20.0, 21.0]).__next__,
        duration_floor_s=0.0,
        server_start_timeout_s=1.0,
        tests_run=[],
        write=False,
    )

    assert artifact["cache_ready"] is False
    assert artifact["server_pid"] == 51190
    assert artifact["server_errors"] == ["server process exited before endpoint became ready"]
    assert artifact["startup_log"]["tail"] == "server exited\n"
    assert cleanup_calls and cleanup_calls[0].pid == 51190


def test_scenario_infer_sota_028_monotonic_server_lifetime_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFER-SOTA-028-SUCCESS: live server lifetime uses wall monotonic time."""

    paths = _write_models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    sleeps: list[float] = []
    post_start_probe_count = 0
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    class FakeProcess:
        pid = 51200

        def poll(self) -> None:
            return None

    def start(command: list[str], env: dict[str, str], log_path: Path) -> FakeProcess:
        del command, env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ready\n", encoding="utf-8")
        return FakeProcess()

    def probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
        nonlocal post_start_probe_count
        if endpoints == ["http://127.0.0.1:45200"]:
            post_start_probe_count += 1
            if post_start_probe_count == 1:
                return _blocked_probe(endpoints, timeout_s)
            return _ready_probe(endpoints, timeout_s)
        return _blocked_probe(endpoints, timeout_s)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=probe,
        endpoint_sample=_ready_sample,
        server_finder=lambda env: _server_available(server),
        free_port=lambda host: _free_port(45200),
        server_start=start,
        server_cleanup=lambda process: {
            "started_by_preflight": True,
            "terminated": True,
            "returncode": 0,
            "pid": process.pid,
        },
        adversarial_verify=_clean_adversarial,
        duration_floor_s=0.0,
        tests_run=[],
        write=True,
    )

    assert artifact["cache_ready"] is True
    assert artifact["endpoint_lifetime_s"] >= 0.0
    assert artifact["server_pid"] == 51200
    assert sleeps


def test_req_infer_sota_028_committed_artifact_is_schema_valid() -> None:
    """REQ-INFER-SOTA-028: the checked-in deliverable satisfies the root-cause schema."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == [QWEN, GEMMA31, GEMMA26]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["cached_sota_pair_attempted"] is True
    assert artifact["cache_ready"] is (
        bool(artifact["logprob_proof"]["ready"]) and not bool(artifact["flagged_adversarial"])
    )
