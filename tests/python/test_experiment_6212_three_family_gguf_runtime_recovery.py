"""Tests for Exp6212 three-family GGUF runtime recovery.

Spec refs: REQ-INFRA-6212,
SCENARIO-INFRA-6212-BLOCKS-WHEN-GPU-OWNED,
SCENARIO-INFRA-6212-CLASSIFIES-EXP6200-LOAD-FAILURE,
SCENARIO-INFRA-6212-READINESS-REQUIRES-TOKEN-AND-CUDA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6212_three_family_gguf_runtime_recovery as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/code-verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6212_three_family_gguf_runtime_recovery.py "
    "-m pytest tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6212_three_family_gguf_runtime_recovery.py "
    "--fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py"
)
ARTIFACT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6212_three_family_gguf_runtime_recovery "
    "--date 20260808"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ARTIFACT_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _upstream(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_6200_three_family_raw_code_transport_canary.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete_partial",
                "honest_verdict": "complete_partial: fixture",
                "gpu_offload_and_interval_receipts": {
                    "generation_lifecycle": [
                        {
                            "family": spec["family"],
                            "backend_exception": (
                                "ValueError: Failed to load model from file: "
                                f"/cache/{spec['family']}.gguf"
                            ),
                        }
                        for spec in mod.MODEL_SPECS
                    ]
                },
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
        model_dir = tmp_path / "hub" / spec["family"] / "snapshots" / f"rev-{spec['family']}"
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{spec['family']}-UD-Q4_K_M.gguf"
        path.write_bytes(b"GGUF" + spec["hf_id"].encode("utf-8"))
        paths[spec["hf_id"]] = path
    return paths


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        return str(paths.get(hf_id)) if hf_id in paths else None

    return resolve


def _metadata(path: Path) -> dict[str, Any]:
    template = f"template for {path.name}"
    return {
        "chat_template_present": True,
        "chat_template_sha256": mod.sha256_text(template),
        "metadata_summary_sha256": mod.sha256_json({"tokenizer.chat_template": template}),
        "metadata_keys": ["tokenizer.chat_template"],
        "tokenizer_detail": "embedded GGUF tokenizer OK",
    }


def _gpu_snapshot(*, busy: bool = False) -> dict[str, Any]:
    compute_apps = (
        [
            {
                "gpu_index": 1,
                "pid": 4242,
                "process_name": "external-train",
                "used_memory_mb": 20_000,
                "command": "python train.py",
                "owned_by_task": False,
            }
        ]
        if busy
        else []
    )
    return {
        "ok": True,
        "gpu_count": 2,
        "devices": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization_pct": 0,
                "memory_total_mb": 24_576,
                "memory_used_mb": 4,
                "memory_free_mb": 24_120,
            },
            {
                "index": 1,
                "name": "RTX 3090",
                "utilization_pct": 0 if not busy else 90,
                "memory_total_mb": 24_576,
                "memory_used_mb": 4 if not busy else 20_004,
                "memory_free_mb": 24_120 if not busy else 4_572,
            },
        ],
        "compute_apps": compute_apps,
    }


class FakeRuntime:
    """REQ-INFRA-6212: fake process runner with deterministic receipts."""

    def __init__(self, *, busy: bool = False) -> None:
        self.busy = busy
        self.failure_calls: list[str] = []
        self.server_calls: list[str] = []
        self.gpu_snapshots = 0

    def gpu_snapshot(self) -> dict[str, Any]:
        self.gpu_snapshots += 1
        return _gpu_snapshot(busy=self.busy)

    def loader_receipt(self) -> dict[str, Any]:
        return {
            "llama_cpp_python_version": "0.0.fixture",
            "llama_cpp_python_gpu_offload": True,
            "llama_cpp_python_system_info": "CUDA fixture",
            "native_llama_server_path": "/tmp/llama-server",
            "native_llama_server_version": "llama.cpp b9999 fixture",
            "native_llama_server_cuda_build": True,
        }

    def reproduce_failure(self, spec: dict[str, Any], gguf: dict[str, Any]) -> dict[str, Any]:
        self.failure_calls.append(spec["hf_id"])
        stderr = (
            "llama_model_load: error loading model architecture: unknown key type\n"
            "ValueError: Failed to load model from file"
        )
        return {
            "family": spec["family"],
            "hf_id": spec["hf_id"],
            "command": ["python", "-c", "Llama(model_path=...)"],
            "pid": 621200,
            "started_utc": "2026-08-08T00:00:00Z",
            "ended_utc": "2026-08-08T00:00:01Z",
            "lifetime_s": 1.0,
            "exit_code": 1,
            "stderr": stderr,
            "stdout": "",
            "classification": mod.classify_failure(stderr, 1),
            "gguf_sha256": gguf["sha256"],
            "read_only": True,
        }

    def run_server_canary(
        self,
        spec: dict[str, Any],
        gguf: dict[str, Any],
        command: list[str],
    ) -> dict[str, Any]:
        self.server_calls.append(spec["hf_id"])
        token = f"{spec['family']}:A".encode("utf-8")
        return {
            "family": spec["family"],
            "hf_id": spec["hf_id"],
            "command": command,
            "pid": 621200 + len(self.server_calls),
            "started_utc": "2026-08-08T00:00:02Z",
            "ended_utc": "2026-08-08T00:00:03Z",
            "lifetime_s": 1.0,
            "stderr": "llama_model_load: offloaded 64/64 layers to GPU\nCUDA0 buffer size = 10 MiB",
            "exit_code": 0,
            "owned_process": True,
            "health_http_status": 200,
            "first_token_text": token.decode("utf-8"),
            "first_token_bytes_b64": mod.base64_bytes(token),
            "first_token_bytes_sha256": mod.sha256_bytes(token),
            "first_token_latency_s": 0.125,
            "raw_token_path": f"/tmp/{spec['family']}.token",
            "cuda_layer_offload": {
                "family": spec["family"],
                "hf_id": spec["hf_id"],
                "cuda_layers_offloaded": 64,
                "total_layers": 64,
                "cuda_layer_offload_confirmed": True,
                "evidence": "offloaded 64/64 layers to GPU",
            },
        }


def test_req_infra_6212_spec_declares_runtime_recovery_contract() -> None:
    """REQ-INFRA-6212: OpenSpec names models, fields, and scenarios."""

    section = "REQ-INFRA-6212" + SPEC.read_text(encoding="utf-8").split("### REQ-INFRA-6212", 1)[1]

    for marker in (
        "REQ-INFRA-6212",
        "SCENARIO-INFRA-6212-BLOCKS-WHEN-GPU-OWNED",
        "SCENARIO-INFRA-6212-CLASSIFIES-EXP6200-LOAD-FAILURE",
        "SCENARIO-INFRA-6212-READINESS-REQUIRES-TOKEN-AND-CUDA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "unrelated_process_kill_count",
        "gguf_mutation_count",
    ):
        assert marker in section
    for spec in mod.MODEL_SPECS:
        assert spec["hf_id"] in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6212_blocks_when_external_gpu_owner_exists(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6212-BLOCKS-WHEN-GPU-OWNED: no server starts."""

    runtime = FakeRuntime(busy=True)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        upstream_exp6200_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.212,
        write=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert runtime.failure_calls == []
    assert runtime.server_calls == []
    assert artifact["preconditions_checked"]["safe_admission_available"] is False
    assert artifact["gpu_owner_pid_memory_and_utilization_before_after"]["blocked_owner_pids"] == [4242]
    assert artifact["three_family_runtime_ready_score"] == 0
    assert artifact["gemma_4_31b_runtime_ready_score"] == 0
    assert artifact["unrelated_process_kill_count"] == 0
    assert artifact["gguf_mutation_count"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_infra_6212_complete_ready_needs_owned_cuda_and_token(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6212-READINESS-REQUIRES-TOKEN-AND-CUDA: all receipts pass."""

    runtime = FakeRuntime()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        upstream_exp6200_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.212,
        write=True,
    )

    assert runtime.failure_calls == [spec["hf_id"] for spec in mod.MODEL_SPECS]
    assert runtime.server_calls == [spec["hf_id"] for spec in mod.MODEL_SPECS]
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["three_family_runtime_ready_score"] == 1
    assert artifact["gemma_4_31b_runtime_ready_score"] == 1
    assert artifact["root_cause_classification"]["classification"] == (
        "loader_compatibility_or_bad_loader_flags"
    )
    assert artifact["root_cause_classification"]["task_owned_fix"] == (
        "use_native_llama_server_cuda_canary_with_explicit_model_file"
    )
    assert artifact["protected_files_unchanged"]["scripts_research_conductor_py_untouched"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    for spec in mod.MODEL_SPECS:
        family = spec["family"]
        command = artifact["per_family_server_command_pid_lifetime_stderr_and_exit"][family][
            "command"
        ]
        assert command[0] == "/tmp/llama-server"
        assert str(_ggufs(tmp_path)[spec["hf_id"]].parent) not in command
        assert str(_ggufs(tmp_path)[spec["hf_id"]]) in command
        assert artifact["per_family_cuda_layer_offload"][family][
            "cuda_layer_offload_confirmed"
        ] is True
        token = artifact["per_family_first_token_bytes_hash_and_latency"][family]
        assert token["first_token_bytes_sha256"].startswith("sha256:")
        assert token["first_token_latency_s"] == pytest.approx(0.125)


def test_scenario_infra_6212_classification_and_validation_branches(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6212-CLASSIFIES-EXP6200-LOAD-FAILURE: branches are explicit."""

    assert mod.classify_failure("invalid magic; not a gguf", 1) == "file_integrity"
    assert mod.classify_failure("unknown architecture in gguf metadata", 1) == (
        "loader_compatibility_or_bad_loader_flags"
    )
    assert mod.classify_failure("cudaMalloc failed: out of memory", 1) == "vram_admission"
    assert mod.classify_failure("CUDA error: invalid device ordinal", 1) == "cuda_placement"
    assert mod.classify_failure("SIGTERM from parent", -15) == "external_termination"
    assert mod.classify_failure("context size has been exceeded", 1) == "bad_flags"
    assert mod.classify_failure("", 0) == "no_failure"

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        upstream_exp6200_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=FakeRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.212,
        write=False,
    )
    assert artifact["field_provenance"]["status"][0] == "REQ-INFRA-6212"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    mutations = [
        (lambda item: item.pop("status"), "missing:status"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (lambda item: item.update({"unrelated_process_kill_count": "0"}), "unrelated_process_kill_count"),
        (lambda item: item.update({"gguf_mutation_count": 1}), "gguf_mutation_count"),
        (lambda item: item.update({"three_family_runtime_ready_score": 1}), "three_family_runtime_ready_score"),
        (lambda item: item.update({"gemma_4_31b_runtime_ready_score": 0}), "gemma_4_31b_runtime_ready_score"),
        (lambda item: item.update({"honest_verdict": "maybe"}), "honest_verdict"),
        (lambda item: item.update({"reproducibility_checksum": "sha256:bad"}), "reproducibility_checksum"),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        if expected == "three_family_runtime_ready_score":
            candidate["per_family_first_token_bytes_hash_and_latency"][
                "qwen3_35b_a3b_moe"
            ] = {}
        mutate(candidate)
        assert expected in mod.validate_artifact(candidate)


def test_req_infra_6212_command_builder_and_model_receipts(tmp_path: Path) -> None:
    """REQ-INFRA-6212: command and GGUF receipts use exact files."""

    paths = _ggufs(tmp_path)
    records = mod.resolve_model_records(
        model_resolver=_resolver(paths),
        metadata_reader=_metadata,
    )
    assert not records["blocked_reasons"]
    assert [row["hf_id"] for row in records["records"]] == [spec["hf_id"] for spec in mod.MODEL_SPECS]
    assert all(row["path_is_file"] for row in records["records"])
    assert all(row["embedded_chat_template_present"] for row in records["records"])

    command = mod.build_server_command(
        server_path="/tmp/llama-server",
        model_path=paths[mod.MODEL_SPECS[0]["hf_id"]],
        port=6212,
        n_ctx=384,
    )
    assert command[:2] == ["/tmp/llama-server", "--model"]
    assert str(paths[mod.MODEL_SPECS[0]["hf_id"]]) in command
    assert str(paths[mod.MODEL_SPECS[0]["hf_id"]].parent) not in command
    assert "--n-gpu-layers" in command
    assert "--ctx-size" in command
    assert "--no-webui" in command


def test_req_infra_6212_alternate_gate_branches(tmp_path: Path, monkeypatch) -> None:
    """REQ-INFRA-6212: blocked and partial branches are explicit."""

    assert mod.utc_now().endswith("Z")
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert mod.classify_failure("ValueError: Failed to load model from file", 1) == (
        "loader_compatibility_or_bad_loader_flags"
    )
    assert mod.classify_failure("unmatched failure", 2) == "process_lifecycle"
    assert mod.safe_admission({"ok": False, "devices": [], "compute_apps": []})[
        "blocked_reasons"
    ] == ["dual_gpu_snapshot_unavailable"]

    missing_records = mod.resolve_model_records(
        model_resolver=lambda _hf_id, _preferred_quant: None,
        metadata_reader=_metadata,
    )
    assert len(missing_records["blocked_reasons"]) == len(mod.MODEL_SPECS)
    assert missing_records["records"][0]["exists"] is False

    model_dir = tmp_path / "directory-not-file"
    model_dir.mkdir()
    dir_records = mod.resolve_model_records(
        model_resolver=lambda _hf_id, _preferred_quant: str(model_dir),
        metadata_reader=lambda _path: {"chat_template_present": False},
    )
    assert "qwen3_35b_a3b_moe_resolved_path_not_file" in dir_records["blocked_reasons"]
    assert "qwen3_35b_a3b_moe_embedded_chat_template_missing" in dir_records["blocked_reasons"]

    mixed = mod.root_cause_receipt(
        {
            "a": {"classification": "file_integrity"},
            "b": {"classification": "vram_admission"},
        },
        {},
    )
    assert mixed["classification"] == "mixed"
    assert mod.honest_verdict({"status": "complete_partial", "preconditions_checked": {}}).startswith(
        "complete_partial:"
    )

    missing_upstream = mod.run(
        result_path=tmp_path / "missing_upstream.json",
        upstream_exp6200_path=tmp_path / "missing.json",
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=FakeRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.212,
        write=False,
    )
    assert missing_upstream["status"] == "blocked"
    assert "upstream_exp6200_missing" in missing_upstream["preconditions_checked"]["blocked_reasons"]

    monkeypatch.setattr(mod, "_parent_writable", lambda _path: False)
    unwritable = mod.run(
        result_path=tmp_path / "unwritable.json",
        upstream_exp6200_path=_upstream(tmp_path),
        model_resolver=_resolver(_ggufs(tmp_path)),
        metadata_reader=_metadata,
        runtime=FakeRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.212,
        write=False,
    )
    assert unwritable["status"] == "blocked"
    assert "output_parent_not_writable" in unwritable["preconditions_checked"]["blocked_reasons"]
