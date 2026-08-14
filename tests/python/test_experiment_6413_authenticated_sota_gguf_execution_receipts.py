"""Tests for Exp6413 authenticated local GGUF execution receipts.

Spec refs: REQ-INFRA-6413, SCENARIO-INFRA-6413-1,
SCENARIO-INFRA-6413-2, SCENARIO-INFRA-6413-3,
SCENARIO-INFRA-6413-4, SCENARIO-INFRA-6413-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nfixture weights\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str, text: str) -> dict[str, Any]:
    tokens = [part for part in text.encode("utf-8").split() if part]
    return {
        "source": mod.TOKENIZER_SOURCE,
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "prompt_tokens": len(tokens),
        "token_count": len(tokens),
        "tokenizer_detail": f"fixture embedded tokenizer for {Path(path).name}",
        "tokenizer_sha256": mod.sha256_json(
            {
                "path": Path(path).name,
                "method": mod.TOKENIZER_METHOD,
                "token_count": len(tokens),
            }
        ),
        "autotokenizer_used": False,
    }


def _good_receipt(model: dict[str, Any], raw_dir: Path, index: int = 0) -> dict[str, Any]:
    model_id = str(model["hf_id"])
    raw_path = raw_dir / f"{mod.model_slug(model_id)}.raw.bin"
    raw_bytes = f"fixture raw output {index} for {model_id}\n".encode("utf-8")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)
    stderr_path = raw_dir / f"{mod.model_slug(model_id)}.stderr.txt"
    stderr_path.write_text("", encoding="utf-8")
    command = [sys.executable, "-c", "fixture child", model_id]
    config = {
        "max_tokens": mod.MAX_TOKENS,
        "n_ctx": mod.N_CTX,
        "n_gpu_layers": -1,
        "seed": mod.RANDOM_SEED + index,
    }
    prompt = mod.CANARY_PROMPTS[model_id]
    prompt_tokens = int(model["prompt_tokens_for_tokenizer_precheck"])
    tokenizer_hash = str(model["tokenizer_sha256"])
    pid = 7000 + index
    ppid = 6000
    gpu = int(model["gpu"])
    uuid = f"GPU-fixture-{gpu}"
    return {
        "schema": mod.RECEIPT_SCHEMA,
        "model_hf_id": model_id,
        "model_family": model["model_family"],
        "pid": pid,
        "parent_pid": ppid,
        "executable": sys.executable,
        "command": command,
        "command_hash": mod.sha256_json(command),
        "config": config,
        "config_hash": mod.sha256_json(config),
        "model": {
            "hub_id": model_id,
            "revision": model["revision"],
            "quantization": model["quantization"],
            "path": model["model_path"],
            "sha256": model["model_file_sha256"],
            "child_stat_size_bytes": Path(model["model_path"]).stat().st_size,
            "child_open_sample_sha256": mod.sha256_file_prefix(model["model_path"]),
            "access_confirmed_by_child": True,
        },
        "tokenizer": {
            "source": mod.TOKENIZER_SOURCE,
            "method": mod.TOKENIZER_METHOD,
            "prompt_tokens": prompt_tokens,
            "tokenizer_sha256": tokenizer_hash,
            "autotokenizer_used": False,
        },
        "device": {
            "gpu_index": gpu,
            "uuid": uuid,
            "cuda_visible_devices": str(gpu),
        },
        "clocks": {
            "parent_launch_monotonic_ns": 90,
            "process_start_monotonic_ns": 100,
            "load_start_monotonic_ns": 110,
            "load_end_monotonic_ns": 200,
            "first_token_monotonic_ns": 230,
            "completion_monotonic_ns": 260,
            "process_end_monotonic_ns": 300,
            "parent_end_monotonic_ns": 310,
        },
        "gpu_samples": [
            {
                "phase": "before_load",
                "pid": pid,
                "pid_bound": False,
                "device_uuid": uuid,
                "gpu_index": gpu,
                "pid_memory_mb": 0,
                "device_memory_used_mb": 10,
                "utilization_pct": 0,
            },
            {
                "phase": "after_load",
                "pid": pid,
                "pid_bound": True,
                "device_uuid": uuid,
                "gpu_index": gpu,
                "pid_memory_mb": 1024,
                "device_memory_used_mb": 1200,
                "utilization_pct": 3,
            },
            {
                "phase": "during_generation",
                "pid": pid,
                "pid_bound": True,
                "device_uuid": uuid,
                "gpu_index": gpu,
                "pid_memory_mb": 1030,
                "device_memory_used_mb": 1210,
                "utilization_pct": 7,
            },
            {
                "phase": "after_cleanup",
                "pid": pid,
                "pid_bound": False,
                "device_uuid": uuid,
                "gpu_index": gpu,
                "pid_memory_mb": 0,
                "device_memory_used_mb": 12,
                "utilization_pct": 0,
            },
        ],
        "prompt": {
            "text_sha256": mod.sha256_text(prompt),
            "byte_length": len(prompt.encode("utf-8")),
        },
        "raw_output": {
            "path": str(raw_path),
            "sha256": mod.sha256_bytes(raw_bytes),
            "byte_length": len(raw_bytes),
            "stored_before_parse": True,
        },
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 3,
            "total_tokens": prompt_tokens + 3,
        },
        "exit_status": {"returncode": 0, "timed_out": False, "signal": None},
        "stderr": {
            "path": str(stderr_path),
            "sha256": mod.sha256_text(""),
            "byte_length": 0,
        },
        "cleanup": {
            "closed": True,
            "process_exited": True,
            "released_cuda_context": True,
        },
        "llama_cpp": {
            "supports_gpu_offload": True,
            "authenticated_gpu_offload": True,
            "n_gpu_layers": -1,
        },
        "legacy_model_smoke_only": False,
        "inherited_upstream_receipt": False,
    }


class FakeRuntime:
    """SCENARIO-INFRA-6413-2: fake runtime blocks or returns fixture receipts."""

    def __init__(self, tmp_path: Path, *, blocked: bool = False) -> None:
        self.tmp_path = tmp_path
        self.blocked = blocked
        self.run_calls: list[str] = []

    def preflight_receipts(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        blockers = ["fixture_precondition_blocked"] if self.blocked else []
        return {
            "both_rtx_3090_gpus_present": not self.blocked,
            "both_gpus_visible": not self.blocked,
            "free_vram_ready": not self.blocked,
            "protected_training_process_present": False,
            "protected_training_pids": [],
            "disk_ready": not self.blocked,
            "storage_free_gb": 100.0,
            "sequential_schedule": [
                {"model_hf_id": row["hf_id"], "gpu": row["gpu"], "order": i}
                for i, row in enumerate(models)
            ],
            "blocked_reasons": blockers,
        }

    def run_model(
        self,
        model: dict[str, Any],
        prompt: str,
        output_dir: Path,
        index: int,
    ) -> dict[str, Any]:
        self.run_calls.append(str(model["hf_id"]))
        assert prompt == mod.CANARY_PROMPTS[str(model["hf_id"])]
        return _good_receipt(model, output_dir / "raw", index=index)


def _artifact(tmp_path: Path, *, blocked: bool = False) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260814",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        runtime=FakeRuntime(tmp_path, blocked=blocked),
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=11.0,
        write=False,
    )


def test_req_infra_6413_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6413: OpenSpec owns the authenticated receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6413") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6413-1",
        "SCENARIO-INFRA-6413-2",
        "SCENARIO-INFRA-6413-3",
        "SCENARIO-INFRA-6413-4",
        "SCENARIO-INFRA-6413-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_infra_6413_model_specs_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6413-1: helper rows produce all three mandated models."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["all_resolved"] is True
    assert resolved["autotokenizer_usage_count"] == 0
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])
    assert all(row["model_file_sha256"].startswith("sha256:") for row in resolved["MODEL_SPECS"])

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, text: {  # noqa: ARG005
            "source": mod.TOKENIZER_SOURCE,
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": "missing",
            "autotokenizer_used": False,
        },
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]
    assert any(reason.startswith("embedded_tokenizer_unavailable:") for reason in missing["blocked_reasons"])

    autotokenizer = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=lambda path, text: {  # noqa: ARG005
            "source": mod.TOKENIZER_SOURCE,
            "method": mod.TOKENIZER_METHOD,
            "loadable": True,
            "prompt_tokens": 1,
            "tokenizer_detail": "bad fallback",
            "autotokenizer_used": True,
        },
    )
    assert any(reason.startswith("autotokenizer_used:") for reason in autotokenizer["blocked_reasons"])


def test_scenario_infra_6413_receipt_validation_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6413-3 and SCENARIO-INFRA-6413-4: receipts fail closed."""

    model_paths = _model_paths(tmp_path / "models")
    models = mod.build_model_specs(
        cached_pair_func=_cached_pair(model_paths, []),
        tokenizer_func=_tokenizer,
    )["MODEL_SPECS"]
    receipts = {
        str(model["hf_id"]): _good_receipt(model, tmp_path / "raw", index=i)
        for i, model in enumerate(models)
    }

    for model in models:
        verdict = mod.validate_receipt(receipts[str(model["hf_id"])], model)
        assert verdict == {"accepted": True, "reasons": []}

    assert mod.authentic_family_count(receipts, models) == 3
    matrix = mod.mutation_attack_matrix(receipts, models)
    assert {row["attack_id"] for row in matrix["rows"]} == set(mod.ATTACK_IDS)
    assert matrix["all_fail_closed"] is True
    assert matrix["false_accept_count"] == 0

    bad = deepcopy(receipts[mod.MANDATED_MODEL_IDS[0]])
    bad["gpu_samples"][1]["pid"] = 1
    assert "gpu_sample_pid_mismatch" in mod.validate_receipt(bad, models[0])["reasons"]


def test_req_infra_6413_defensive_edges_and_failure_reasons(tmp_path: Path) -> None:
    """REQ-INFRA-6413: defensive receipt branches reject malformed evidence."""

    assert mod.sha256_file(tmp_path / "missing.bin") is None
    assert mod.sha256_file_prefix(tmp_path / "missing.bin") is None
    assert mod.revision_from_path("/cache/snapshots/rev/model.gguf") == "rev"
    assert mod.revision_from_path("/cache/no-snapshot/model.gguf") is None
    assert mod.quantization_from_path("model-without-quant.gguf") == "unknown"
    assert mod._int_or_none(None) is None
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._as_mapping([]) == {}
    assert mod._signal_name(None) is None
    assert mod._signal_name(0) is None
    assert mod._signal_name(-9) == "SIGKILL"
    assert mod._signal_name(-999_999) == "signal_999999"
    assert mod._parse_child_receipt("x\nCARNOT_CHILD_RECEIPT:{bad}") is None
    assert mod._parse_child_receipt('CARNOT_CHILD_RECEIPT:{"ok": true}') == {"ok": True}
    bytes_path = tmp_path / "bytes" / "out.bin"
    assert mod.write_bytes_atomic(bytes_path, b"abc") == bytes_path
    assert bytes_path.read_bytes() == b"abc"
    schema_path = tmp_path / "schema.json"
    assert mod.write_payload_or_hash(schema_path, {"x": 1}, write=True) == mod.sha256_file(schema_path)

    model_paths = _model_paths(tmp_path / "models")
    models = mod.build_model_specs(
        cached_pair_func=_cached_pair(model_paths, []),
        tokenizer_func=_tokenizer,
    )["MODEL_SPECS"]
    model = models[0]
    base = _good_receipt(model, tmp_path / "raw", index=0)

    missing = deepcopy(base)
    del missing["pid"]
    assert mod.validate_receipt(missing, model)["reasons"] == ["missing_receipt_field:pid"]

    def reasons(mutator) -> list[str]:
        row = deepcopy(base)
        mutator(row)
        return mod.validate_receipt(row, model)["reasons"]

    cases = [
        (lambda row: row.update(schema="old"), "receipt_schema_mismatch"),
        (lambda row: row.update(model_hf_id="wrong"), "model_hf_id_mismatch"),
        (lambda row: row.update(model_family="wrong"), "model_family_mismatch"),
        (lambda row: row.update(legacy_model_smoke_only=True), "legacy_model_smoke_only"),
        (lambda row: row.update(inherited_upstream_receipt=True), "inherited_upstream_receipt"),
        (lambda row: row.update(pid=0), "pid_invalid_or_forged"),
        (lambda row: row.update(parent_pid=row["pid"]), "parent_pid_invalid"),
        (lambda row: row.update(executable=""), "executable_missing"),
        (lambda row: row.update(command_hash="sha256:bad"), "command_hash_mismatch"),
        (lambda row: row.update(config_hash="sha256:bad"), "config_hash_mismatch"),
        (lambda row: row["model"].update(hub_id="wrong"), "model_hub_id_mismatch"),
        (lambda row: row["model"].update(path="wrong"), "model_path_mismatch"),
        (lambda row: row["model"].update(sha256="sha256:bad"), "model_file_hash_mismatch"),
        (
            lambda row: row["model"].update(child_open_sample_sha256="sha256:bad"),
            "model_child_open_sample_mismatch",
        ),
        (
            lambda row: row["model"].update(access_confirmed_by_child=False),
            "model_file_access_not_confirmed",
        ),
        (
            lambda row: row["model"].update(child_stat_size_bytes=-1),
            "model_child_stat_size_mismatch",
        ),
        (lambda row: row["tokenizer"].update(source="AutoTokenizer"), "tokenizer_source_not_embedded_gguf"),
        (lambda row: row["tokenizer"].update(method="AutoTokenizer"), "tokenizer_method_mismatch"),
        (lambda row: row["tokenizer"].update(autotokenizer_used=True), "autotokenizer_used"),
        (lambda row: row["tokenizer"].update(prompt_tokens=0), "tokenizer_prompt_tokens_nonpositive"),
        (lambda row: row["tokenizer"].update(tokenizer_sha256="sha256:bad"), "tokenizer_hash_mismatch"),
        (lambda row: row["clocks"].update(load_start_monotonic_ns=999), "clock_order_invalid"),
        (lambda row: row["clocks"].update(first_token_monotonic_ns=None), "missing_clock:first_token_monotonic_ns"),
        (lambda row: row["raw_output"].update(sha256="sha256:bad"), "raw_output_hash_mismatch"),
        (lambda row: row["raw_output"].update(byte_length=-1), "raw_output_byte_length_mismatch"),
        (lambda row: row["raw_output"].update(stored_before_parse=False), "raw_output_not_stored_before_parse"),
        (lambda row: row["tokens"].update(prompt_tokens=0), "prompt_tokens_nonpositive"),
        (lambda row: row["tokens"].update(completion_tokens=0), "completion_tokens_nonpositive"),
        (lambda row: row["exit_status"].update(returncode=1), "exit_status_nonzero"),
        (lambda row: row["exit_status"].update(timed_out=True), "process_timed_out"),
        (lambda row: row["stderr"].update(sha256="sha256:bad"), "stderr_hash_mismatch"),
        (lambda row: row["cleanup"].update(closed=False), "cleanup_incomplete"),
        (lambda row: row["llama_cpp"].update(supports_gpu_offload=False), "llama_cpp_gpu_offload_not_supported"),
        (lambda row: row["llama_cpp"].update(authenticated_gpu_offload=False), "gpu_offload_not_authenticated"),
        (lambda row: row["llama_cpp"].update(n_gpu_layers=0), "cpu_only_receipt"),
        (lambda row: row.update(gpu_samples="bad"), "gpu_samples_missing"),
        (lambda row: row["device"].update(gpu_index=99), "gpu_index_mismatch"),
        (lambda row: row["device"].update(uuid=""), "device_uuid_missing"),
        (lambda row: row["gpu_samples"].pop(1), "missing_gpu_phase:after_load"),
        (
            lambda row: [sample.update(device_memory_used_mb=100) for sample in row["gpu_samples"]],
            "gpu_memory_constant_or_missing",
        ),
        (lambda row: row["gpu_samples"][1].update(pid_bound=False), "pid_bound_gpu_sample_missing:after_load"),
        (lambda row: row["gpu_samples"][1].update(device_uuid="wrong"), "gpu_sample_device_uuid_mismatch"),
        (lambda row: row["gpu_samples"][1].update(gpu_index=99), "gpu_sample_gpu_index_mismatch"),
        (lambda row: row["gpu_samples"][1].update(pid_memory_mb=0), "pid_bound_gpu_memory_nonpositive"),
    ]
    for mutate, expected in cases:
        assert expected in reasons(mutate)

    zero = deepcopy(base)
    raw_path = Path(zero["raw_output"]["path"])
    raw_path.write_bytes(b"")
    zero["raw_output"]["sha256"] = mod.sha256_bytes(b"")
    zero["raw_output"]["byte_length"] = 0
    assert "raw_output_zero_length" in mod.validate_receipt(zero, model)["reasons"]

    inherited = deepcopy(base)
    inherited["inherited_upstream_receipt"] = True
    constant = deepcopy(base)
    for sample in constant["gpu_samples"]:
        sample["device_memory_used_mb"] = 1
    counts = {
        str(model["hf_id"]): inherited,
        mod.MANDATED_MODEL_IDS[1]: constant,
        mod.MANDATED_MODEL_IDS[2]: base,
    }
    assert mod.constant_or_inherited_receipt_count(counts, models) == 2
    assert mod.legacy_headline_cell_count({"legacy/model": base}) == 1

    preconditions = mod.preconditions_from(
        date="20260814",
        model_resolution={"blocked_reasons": [], "MODEL_SPECS": models},
        runtime_preflight={"blocked_reasons": []},
        llama_support={"llama_supports_gpu_offload": False},
        schema_receipt={"sha256": None},
        source_before={"module": None},
        protected_before={"ops": None},
    )
    assert preconditions["all_preconditions_passed"] is False
    assert {
        "llama_cpp_gpu_offload_unavailable",
        "receipt_schema_hash_missing",
        "source_hash_missing",
        "protected_hash_missing",
    } <= set(preconditions["blocked_reasons"])


def test_scenario_infra_6413_artifact_score_schema_and_blocked_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """SCENARIO-INFRA-6413-2 and SCENARIO-INFRA-6413-5: artifact gates readiness."""

    artifact = _artifact(tmp_path)

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["authenticated_receipt_contract_ready_score"] == 1.0
    assert artifact["authentic_family_count"] == 3
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["constant_or_inherited_receipt_count"] == 0
    assert artifact["legacy_headline_cell_count"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blocked_runtime = FakeRuntime(tmp_path / "blocked", blocked=True)
    paths = _model_paths(tmp_path / "blocked" / "models")
    blocked = mod.run(
        date="20260814",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked" / "data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        runtime=blocked_runtime,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["authenticated_receipt_contract_ready_score"] == 0.0
    assert blocked_runtime.run_calls == []

    bad = deepcopy(artifact)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "wrong"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["MODEL_SPECS"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "MODEL_SPECS mandated ids mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["autotokenizer_usage_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "autotokenizer_usage_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["legacy_headline_cell_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "legacy_headline_cell_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "invalid prefix"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["mutation_attack_matrix"]["rows"][0]["fail_closed"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["authenticated_receipt_contract_ready_score"] == 0.0

    result_path = tmp_path / "written.json"
    mod.write_artifact(artifact, result_path)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda artifact: ["forced schema error"])  # noqa: ARG005
        forced = mod.run(
            date="20260814",
            result_path=tmp_path / "forced_schema.json",
            data_dir=tmp_path / "forced" / "data",
            cached_pair_func=_cached_pair(_model_paths(tmp_path / "forced" / "models"), []),
            tokenizer_func=_tokenizer,
            runtime=FakeRuntime(tmp_path / "forced"),
            test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
            duration_s=1.0,
            write=True,
        )
    assert forced["status"] == "failed_schema"
    assert (tmp_path / "forced_schema.json").is_file()

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    validate_path = tmp_path / mod.RESULT_RELATIVE_PATH
    mod.write_artifact({"status": "bad"}, validate_path)
    assert mod.main(["--validate"]) == 1
    mod.write_artifact(artifact, validate_path)
    assert mod.main(["--validate"]) == 0

    monkeypatch.setattr(
        mod,
        "run",
        lambda **kwargs: artifact,  # noqa: ARG005
    )
    assert mod.main(["--date", "20260814"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out
