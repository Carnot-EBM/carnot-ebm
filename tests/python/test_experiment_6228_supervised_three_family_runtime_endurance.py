"""Tests for Exp6228 supervised three-family runtime endurance.

Spec refs: REQ-INFRA-6228,
SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED,
SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS,
SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS,
SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY,
SCENARIO-INFRA-6228-ARTIFACT-SCORES-ARE-CONJUNCTIVE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6228_supervised_three_family_runtime_endurance as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"
FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_llama_server_supervisor.py "
    "tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py,"
    "python/carnot/inference/llama_server_supervisor.py "
    "-m pytest tests/python/test_llama_server_supervisor.py "
    "tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py,"
    "python/carnot/inference/llama_server_supervisor.py --fail-under=100 --show-missing"
)
FULL_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_llama_server_supervisor.py "
    "tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py"
)
ARTIFACT_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6228_supervised_three_family_runtime_endurance --date 20260809"
)
TEST_COMMANDS = [
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    FULL_COMMAND,
    SPEC_COMMAND,
    ARTIFACT_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _upstream(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_6227_llama_server_signal_sender_diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete_unlocalized",
                "finite_wait_retry_cleanup_contract": {"retry_budget": 1},
                "runtime_diagnostic_ready_score": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _ggufs(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in mod.MODEL_SPECS:
        model_dir = (
            tmp_path
            / "hub"
            / f"models--{spec['hf_id'].replace('/', '--')}"
            / "snapshots"
            / f"rev-{spec['family']}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{spec['family']}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF" + spec["hf_id"].encode("utf-8"))
        paths[spec["hf_id"]] = path
    return paths


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


def _metadata(path: Path) -> dict[str, Any]:
    template = "embedded template for " + path.name
    return {
        "chat_template_present": True,
        "chat_template_sha256": mod.sha256_text(template),
        "metadata_summary_sha256": mod.sha256_json({"tokenizer.chat_template": template}),
        "metadata_keys": ["tokenizer.chat_template"],
        "tokenizer_detail": "embedded GGUF metadata parsed without AutoTokenizer",
    }


def _gpu_snapshot(*, busy: bool = False, label: str = "fixture") -> dict[str, Any]:
    apps = (
        [
            {
                "gpu_index": 0,
                "pid": 7777,
                "process_name": "external-owner",
                "used_memory_mb": 22000,
                "command": "python train.py",
                "owned_by_task": False,
            }
        ]
        if busy
        else []
    )
    return {
        "ok": True,
        "label": label,
        "gpu_count": 2,
        "devices": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization_pct": 0 if not busy else 88,
                "memory_total_mb": 24576,
                "memory_used_mb": 4 if not busy else 22004,
                "memory_free_mb": 24120 if not busy else 2572,
            },
            {
                "index": 1,
                "name": "RTX 3090",
                "utilization_pct": 0,
                "memory_total_mb": 24576,
                "memory_used_mb": 4,
                "memory_free_mb": 24120,
            },
        ],
        "compute_apps": apps,
    }


class FakeRuntime:
    """REQ-INFRA-6228: deterministic adapter for artifact tests."""

    def __init__(
        self,
        *,
        result_path: Path,
        busy: bool = False,
        missing_server: bool = False,
        incomplete_family: str | None = None,
    ) -> None:
        self.result_path = result_path
        self.busy = busy
        self.missing_server = missing_server
        self.incomplete_family = incomplete_family
        self.calls: list[str] = []

    def gpu_snapshot(self, label: str) -> dict[str, Any]:
        return _gpu_snapshot(busy=self.busy, label=label)

    def llama_cpp_receipt(self) -> dict[str, Any]:
        return {
            "native_llama_server_path": "/tmp/llama-server",
            "native_llama_server_exists": not self.missing_server,
            "native_llama_server_version": "llama.cpp fixture CUDA",
            "native_llama_server_cuda_build": True,
            "llama_cpp_python_gpu_offload": True,
            "llama_cpp_python_system_info": "CUDA fixture",
            "no_autotokenizer_used": True,
        }

    def qualify_family(
        self,
        spec: dict[str, Any],
        gguf: dict[str, Any],
        command: list[str],
        contract: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        precondition = json.loads(self.result_path.read_text(encoding="utf-8"))
        assert precondition["status"] == "preconditions_recorded"
        assert precondition["preconditions_checked"]["written_before_model_load"] is True
        self.calls.append(spec["hf_id"])
        pid = 622800 + len(self.calls)
        token = f"{spec['family']}:A".encode("utf-8")
        samples = [
            {
                "sample_index": index,
                "raw_token_bytes_b64": mod.base64_bytes(token),
                "raw_token_bytes_sha256": mod.sha256_bytes(token),
                "latency_s": 0.2 + index,
                "path": str(output_dir / f"{spec['family']}.{index}.token"),
            }
            for index in range(3)
        ]
        deterministic = spec["family"] != self.incomplete_family
        cuda_ready = spec["family"] != self.incomplete_family
        return {
            "family": spec["family"],
            "hf_id": spec["hf_id"],
            "gpu_intervals": [
                _gpu_snapshot(label="before_family"),
                {
                    **_gpu_snapshot(label="during_initial"),
                    "compute_apps": [
                        {
                            "pid": pid,
                            "process_name": "llama-server",
                            "used_memory_mb": 9000,
                            "owned_by_task": True,
                        }
                    ],
                },
                _gpu_snapshot(label="after_final_cleanup"),
            ],
            "servers": [
                {
                    "phase": "initial",
                    "command": command,
                    "command_hash": mod.command_hash(command),
                    "pid": pid,
                    "start_time_ticks": 12345,
                    "process_group_id": pid,
                    "parent_identity": {"pid": 1, "start_time_ticks": 99},
                    "owned_process": True,
                    "started_utc": "2026-08-09T00:00:00Z",
                    "ended_utc": "2026-08-09T00:00:10Z",
                    "lifetime_s": 10.0,
                    "exit_code": -15,
                    "stderr_tail": (
                        "offloaded 64/64 layers to GPU\nCUDA0 buffer size = 9000 MiB"
                        if cuda_ready
                        else "--n-gpu-layers all"
                    ),
                },
                {
                    "phase": "recovery",
                    "command": command,
                    "command_hash": mod.command_hash(command),
                    "pid": pid + 100,
                    "start_time_ticks": 22345,
                    "process_group_id": pid + 100,
                    "parent_identity": {"pid": 1, "start_time_ticks": 99},
                    "owned_process": True,
                    "started_utc": "2026-08-09T00:00:11Z",
                    "ended_utc": "2026-08-09T00:00:20Z",
                    "lifetime_s": 9.0,
                    "exit_code": 0,
                    "stderr_tail": (
                        "offloaded 64/64 layers to GPU\nCUDA0 buffer size = 9000 MiB"
                        if cuda_ready
                        else "--n-gpu-layers all"
                    ),
                },
            ],
            "cuda": {
                "family": spec["family"],
                "hf_id": spec["hf_id"],
                "cuda_layers_offloaded": 64 if cuda_ready else 0,
                "total_layers": 64 if cuda_ready else 0,
                "cuda_tensor_or_buffer_confirmed": cuda_ready,
                "log_cuda_evidence_present": cuda_ready,
                "gpu_interval_owned_vram_confirmed": cuda_ready,
                "cuda_placement_confirmed": cuda_ready,
                "evidence": "offloaded 64/64 layers to GPU" if cuda_ready else "",
            },
            "tokens": {
                "samples": samples,
                "sample_count": len(samples),
                "unique_raw_token_hashes": [samples[0]["raw_token_bytes_sha256"]],
                "deterministic_repeated_output": deterministic,
                "min_latency_s": 0.2,
                "max_latency_s": 2.2,
            },
            "endurance": {
                "passed": deterministic,
                "window_s": contract["endurance_interval_s"],
                "started_utc": "2026-08-09T00:00:00Z",
                "ended_utc": "2026-08-09T00:00:20Z",
                "health_sample_count": 3,
                "health_samples": [
                    {"status": 200, "elapsed_s": 0.1, "classification": "no_failure"},
                    {"status": 200, "elapsed_s": 6.0, "classification": "no_failure"},
                    {"status": 200, "elapsed_s": 12.0, "classification": "no_failure"},
                ],
            },
            "recovery": {
                "controlled_failure_bounded": True,
                "controlled_failure_signal": "SIGTERM",
                "cleanup_action": "terminated",
                "recovery_success": deterministic,
                "retry_attempts_used": 1,
                "recovery_latency_s": 9.0,
                "unrelated_process_kill_count_delta": 0,
            },
            "leak_check": {
                "leak_free": deterministic,
                "process_alive_after_cleanup": False,
                "owned_vram_mb_after_cleanup": 0,
            },
        }


def test_req_infra_6228_spec_declares_supervised_endurance_contract() -> None:
    """REQ-INFRA-6228: OpenSpec names fields, models, and scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6228") :]

    for marker in (
        "REQ-INFRA-6228",
        "SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED",
        "SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS",
        "SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS",
        "SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY",
        "SCENARIO-INFRA-6228-ARTIFACT-SCORES-ARE-CONJUNCTIVE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for spec in mod.MODEL_SPECS:
        assert spec["hf_id"] in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6228_complete_artifact_qualifies_all_families(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    runtime = FakeRuntime(result_path=result_path)
    artifact = mod.run(
        result_path=result_path,
        upstream_diagnostic_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=62.28,
        write=True,
    )

    assert runtime.calls == [spec["hf_id"] for spec in mod.MODEL_SPECS]
    assert artifact["status"] == "complete_ready"
    assert artifact["qwen_runtime_ready_score"] == 1
    assert artifact["gemma_4_31b_runtime_ready_score"] == 1
    assert artifact["gemma_4_26b_runtime_ready_score"] == 1
    assert artifact["two_family_runtime_ready_score"] == 1
    assert artifact["three_family_runtime_ready_score"] == 1
    assert artifact["preconditions_checked"]["written_before_model_load"] is True
    assert artifact["unrelated_process_kill_count"] == 0
    assert type(artifact["unrelated_process_kill_count"]) is int
    assert artifact["gguf_mutation_count"] == 0
    assert type(artifact["gguf_mutation_count"]) is int
    assert artifact["protected_files_unchanged"]["scripts_research_conductor_py_untouched"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["embedded_chat_template_receipts"]["no_autotokenizer_used"] is True
    assert mod.validate_artifact(artifact) == []
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact

    for spec in mod.MODEL_SPECS:
        family = spec["family"]
        command = artifact[
            "server_command_pid_starttime_process_group_and_lifetime_by_family"
        ][family][0]["command"]
        assert command[0] == "/tmp/llama-server"
        assert str(_ggufs(tmp_path)[spec["hf_id"]]) in command
        assert str(_ggufs(tmp_path)[spec["hf_id"]].parent) not in command
        assert artifact["parsed_cuda_layer_or_tensor_placement_by_family"][family][
            "cuda_placement_confirmed"
        ] is True
        assert artifact["repeated_raw_token_hashes_and_latencies_by_family"][family][
            "deterministic_repeated_output"
        ] is True


def test_scenario_infra_6228_blocks_unsafe_preconditions(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS."""

    result_path = tmp_path / "blocked.json"
    runtime = FakeRuntime(result_path=result_path, busy=True)
    artifact = mod.run(
        result_path=result_path,
        upstream_diagnostic_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.228,
        write=True,
    )

    assert artifact["status"] == "blocked"
    assert runtime.calls == []
    assert "external_gpu_owner_present" in artifact["preconditions_checked"]["blocked_reasons"]
    assert artifact["preconditions_checked"]["safe_gpu_admission"] is False
    assert artifact["unrelated_process_kill_count"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_infra_6228_conjunctive_scores_and_validation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6228-ARTIFACT-SCORES-ARE-CONJUNCTIVE."""

    result_path = tmp_path / "partial.json"
    artifact = mod.run(
        result_path=result_path,
        upstream_diagnostic_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=FakeRuntime(result_path=result_path, incomplete_family="qwen3_35b_a3b_moe"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=62.28,
        write=True,
    )

    assert artifact["status"] == "complete_partial"
    assert artifact["qwen_runtime_ready_score"] == 0
    assert artifact["gemma_4_31b_runtime_ready_score"] == 1
    assert artifact["gemma_4_26b_runtime_ready_score"] == 1
    assert artifact["two_family_runtime_ready_score"] == 1
    assert artifact["three_family_runtime_ready_score"] == 0
    assert artifact["field_provenance"]["status"][0] == "REQ-INFRA-6228"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    mutations = [
        (lambda item: item.pop("status"), "missing:status"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (
            lambda item: item.update({"unrelated_process_kill_count": False}),
            "unrelated_process_kill_count",
        ),
        (lambda item: item.update({"gguf_mutation_count": "0"}), "gguf_mutation_count"),
        (lambda item: item.update({"qwen_runtime_ready_score": 1}), "qwen_runtime_ready_score"),
        (
            lambda item: item.update({"three_family_runtime_ready_score": 1}),
            "three_family_runtime_ready_score",
        ),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance:status"),
        (lambda item: item["field_principles"].pop("status"), "field_principles:status"),
        (
            lambda item: item["protected_files_unchanged"].update({"unchanged": False}),
            "protected_files_unchanged",
        ),
        (lambda item: item.update({"honest_verdict": "maybe"}), "honest_verdict"),
        (
            lambda item: item.update({"reproducibility_checksum": "sha256:bad"}),
            "reproducibility_checksum",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        assert expected in mod.validate_artifact(candidate)


def test_req_infra_6228_model_receipts_and_precondition_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-INFRA-6228: exact files, ports, and blockers are explicit."""

    paths = _ggufs(tmp_path)
    records = mod.resolve_model_records(
        model_resolver=_resolver(paths),
        metadata_reader=_metadata,
    )
    assert not records["blocked_reasons"]
    assert [row["hf_id"] for row in records["records"]] == [spec["hf_id"] for spec in mod.MODEL_SPECS]
    assert all(row["path_is_file"] for row in records["records"])
    assert all(row["no_autotokenizer_used"] for row in records["records"])
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert mod.observed_quantization(Path("/flat/model.gguf")) == "unknown"

    port_plan = mod.reserve_ports((spec["family"] for spec in mod.MODEL_SPECS), port_factory=iter([1, 2, 3]).__next__)
    duplicate_plan = mod.reserve_ports(("a", "b"), port_factory=iter([4, 4, 5]).__next__)
    commands = mod.build_family_commands(
        server_path="/tmp/llama-server",
        model_records=records["records"],
        port_plan=port_plan,
    )
    assert sorted(port_plan.values()) == [1, 2, 3]
    assert duplicate_plan == {"a": 4, "b": 5}
    for row in records["records"]:
        command = commands[row["family"]]
        assert str(paths[row["hf_id"]]) in command
        assert str(paths[row["hf_id"]].parent) not in command
        assert "--n-gpu-layers" in command
        assert "--jinja" in command

    directory = tmp_path / "directory-not-file"
    directory.mkdir()
    directory_records = mod.resolve_model_records(
        model_resolver=lambda _hf_id, _preferred_quant: str(directory),
        metadata_reader=lambda _path: {"chat_template_present": False},
    )
    assert "qwen3_35b_a3b_moe_resolved_path_not_file" in directory_records["blocked_reasons"]
    assert "qwen3_35b_a3b_moe_embedded_chat_template_missing" in directory_records["blocked_reasons"]
    assert mod.safe_gpu_admission({"ok": False, "devices": [], "compute_apps": []})[
        "blocked_reasons"
    ] == ["dual_gpu_snapshot_unavailable"]
    assert mod.status_from_scores(
        [],
        {
            "three_family_runtime_ready_score": 0,
            "two_family_runtime_ready_score": 0,
            "qwen_runtime_ready_score": 0,
            "gemma_4_31b_runtime_ready_score": 0,
            "gemma_4_26b_runtime_ready_score": 0,
        },
        {},
    ) == "blocked"

    missing = mod.run(
        result_path=tmp_path / "missing.json",
        upstream_diagnostic_path=tmp_path / "missing-upstream.json",
        model_resolver=lambda _hf_id, _preferred_quant: None,
        metadata_reader=_metadata,
        runtime=FakeRuntime(result_path=tmp_path / "missing.json", missing_server=True),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.228,
        write=False,
    )
    assert missing["status"] == "blocked"
    assert "upstream_diagnostic_missing" in missing["preconditions_checked"]["blocked_reasons"]
    assert "native_llama_server_missing" in missing["preconditions_checked"]["blocked_reasons"]
    assert any(
        reason.endswith("_gguf_not_cached")
        for reason in missing["preconditions_checked"]["blocked_reasons"]
    )
    assert mod.validate_artifact(missing) == []

    unwritable_path = tmp_path / "unwritable.json"
    monkeypatch.setattr(mod, "_parent_writable", lambda _path: False)
    unwritable = mod.run(
        result_path=unwritable_path,
        upstream_diagnostic_path=_upstream(tmp_path),
        model_resolver=_resolver(paths),
        metadata_reader=_metadata,
        runtime=FakeRuntime(result_path=unwritable_path),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.228,
        write=False,
    )
    assert "output_parent_not_writable" in unwritable["preconditions_checked"]["blocked_reasons"]
