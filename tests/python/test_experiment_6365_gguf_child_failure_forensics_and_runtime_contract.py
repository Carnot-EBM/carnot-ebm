"""Tests for Exp6365 GGUF child failure forensics.

Spec refs: REQ-INFRA-6365, SCENARIO-INFRA-6365-1,
SCENARIO-INFRA-6365-2, SCENARIO-INFRA-6365-3,
SCENARIO-INFRA-6365-4, SCENARIO-INFRA-6365-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from carnot import experiment_6365_gguf_child_failure_forensics_and_runtime_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\n").encode("utf-8"))
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
                "name": mod.MODEL_SPECS_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str, prompt: str) -> dict[str, Any]:
    tokens = [part for part in prompt.encode("utf-8").split() if part]
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "prompt_tokens": len(tokens),
        "tokenizer_detail": f"fixture embedded tokenizer for {Path(path).name}",
    }


def _runtime_row(model_id: str, tmp_path: Path, *, success: bool = True) -> dict[str, Any]:
    raw_path = tmp_path / f"{mod.model_slug(model_id)}.raw.txt"
    raw_path.write_bytes(b"ok" if success else b"")
    gpu_samples = {
        phase: [
            {
                "model_hf_id": model_id,
                "phase": phase,
                "gpu_index": 0,
                "timestamp_utc": "2026-08-13T00:00:00Z",
                "memory_used_mb": memory,
                "memory_free_mb": 24576 - memory,
                "utilization_pct": 1,
                "process_identity": {"pid": 123, "cmdline": "fixture child"},
            }
        ]
        for phase, memory in {
            "before_load": 4,
            "after_load": 1024,
            "during_generation": 1200,
            "after_unload": 4,
            "after_cleanup": 4,
        }.items()
    }
    return {
        "model_hf_id": model_id,
        "stdout_path": str(raw_path),
        "stdout_sha256": mod.sha256_file(raw_path),
        "stdout_byte_count": raw_path.stat().st_size,
        "stderr_path": str(tmp_path / f"{mod.model_slug(model_id)}.stderr.txt"),
        "stderr_sha256": mod.sha256_bytes(b""),
        "stderr_excerpt": "",
        "raw_output_path": str(raw_path),
        "raw_output_sha256": mod.sha256_file(raw_path),
        "raw_output_bytes": raw_path.stat().st_size,
        "returncode": 0 if success else 1,
        "signal": None,
        "timed_out": False,
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "usage_receipt_valid": success,
        "live_autoregressive_generation_invoked": success,
        "authenticated_gpu_offload": success,
        "gpu_samples_by_phase": gpu_samples,
        "phase_timings": {
            phase: {"started_ns": index, "ended_ns": index + 1, "duration_s": 0.1}
            for index, phase in enumerate(mod.REQUIRED_TIMING_PHASES)
        },
        "prompt_context": {
            "prompt_tokens": 3,
            "requested_output_tokens": 1,
            "n_ctx": 32,
            "capacity_margin": 28,
            "fits": True,
        },
        "source_hash_ok": True,
        "stdout_nonempty": success,
    }


class FakeRuntime:
    """REQ-INFRA-6365: fake model runtime for artifact tests."""

    def __init__(self, tmp_path: Path, *, missing_gpu_sample: bool = False) -> None:
        self.tmp_path = tmp_path
        self.missing_gpu_sample = missing_gpu_sample

    def preflight_receipts(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "both_rtx_3090_gpus_present": True,
            "disk_ready": True,
            "ram_ready": True,
            "protected_training_process_present": False,
            "sequential_vram_ready": True,
            "vram_probe_model_hf_id": mod.MANDATED_MODEL_IDS[0],
            "vram_probe_proved_rise_before_cuda_ready": True,
            "blocked_reasons": [],
        }

    def run_model(self, model: dict[str, Any], prompt: str, output_dir: Path) -> dict[str, Any]:
        row = _runtime_row(str(model["hf_id"]), self.tmp_path)
        if self.missing_gpu_sample:
            row["gpu_samples_by_phase"].pop("during_generation")
        return row


def _artifact(tmp_path: Path, *, missing_gpu_sample: bool = False) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        output_dir=tmp_path / "run",
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        runtime=FakeRuntime(tmp_path, missing_gpu_sample=missing_gpu_sample),
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )


def test_req_infra_6365_defensive_helper_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6365: helper edge cases are deterministic and fail closed."""

    missing_path = tmp_path / "missing.gguf"
    assert mod.sha256_file(missing_path) is None
    assert mod.revision_from_path(Path("/cache/snapshots/rev/model.gguf")) == "rev"
    assert mod.quantization_from_path(Path("model-without-quant.gguf")) == "unknown"
    assert mod.parse_json_lines('CARNOT_PHASE:{"phase":"load"}\nCARNOT_PHASE:{bad}', "PHASE") == [
        {"phase": "load"}
    ]
    assert "truncated" in mod.sanitize_argv(["x" * 400])[0]
    assert mod.read_process_identity(999_999_999)["exists"] is False
    assert mod._signal_name(-9) == "SIGKILL"
    assert mod._signal_name(-999_999) == "signal_999999"
    timings = mod.phase_timings_from_events(
        [{"phase": "load", "started_ns": 1, "ended_ns": 3, "duration_s": 0.2}],
        1,
        4,
    )
    assert timings["load"]["duration_s"] == 0.2
    assert mod.missing_gpu_phases({"gpu_samples_by_phase": []}) == list(mod.REQUIRED_GPU_PHASES)
    assert mod.classify_terminal("blocked", "") == "terminal_blocked"
    assert mod.classify_terminal("complete_positive", "") == "terminal_positive"
    assert mod.classify_terminal("mystery", "") == "terminal_unknown"
    assert mod.is_protected_training_cmdline("python train_llama_cuda.py") is True
    assert mod.is_protected_training_cmdline("python inference.py") is False

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, prompt: {  # noqa: ARG005
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": "missing",
        },
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]
    assert any(reason.startswith("embedded_tokenizer_unavailable:") for reason in missing["blocked_reasons"])

    preconditions = mod.preconditions_from(
        date="20260813",
        model_resolution={"blocked_reasons": [], "MODEL_SPECS": []},
        runtime_preflight={"blocked_reasons": [], "vram_probe_proved_rise_before_cuda_ready": None},
        llama_support={"llama_supports_gpu_offload": False},
        vram_receipts={mod.MANDATED_MODEL_IDS[0]: {"proved_rise_and_release": False}},
    )
    assert "llama_cpp_gpu_offload_not_supported" in preconditions["blocked_reasons"]
    assert "vram_probe_did_not_prove_rise_and_release" in preconditions["blocked_reasons"]
    assert mod.status_for({"preconditions_checked": {"all_preconditions_passed": False}}) == (
        "blocked_precondition"
    )
    assert mod.verdict_for(
        {"status": "blocked_precondition", "preconditions_checked": {"blocked_reasons": ["x"]}}
    ).startswith("blocked:")


def test_req_infra_6365_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-INFRA-6365: OpenSpec owns fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6365") : text.index("REQ-INFRA-6351")]

    for marker in (
        "SCENARIO-INFRA-6365-1",
        "SCENARIO-INFRA-6365-2",
        "SCENARIO-INFRA-6365-3",
        "SCENARIO-INFRA-6365-4",
        "SCENARIO-INFRA-6365-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_infra_6365_exp6352_reconstruction_records_drift() -> None:
    """SCENARIO-INFRA-6365-1: Exp6352 is reconstructed without diagnosis."""

    receipt = mod.reconstruct_exp6352(REPO)

    assert receipt["upstream"]["present"] is True
    assert receipt["terminal_class"] == "terminal_null"
    assert receipt["source_artifact_sampling_drift"]["n_ctx_mismatch"] is True
    assert receipt["source_artifact_sampling_drift"]["source_sampling_n_ctx"] == 2048
    assert receipt["source_artifact_sampling_drift"]["artifact_process_n_ctx_values"] == [512]
    assert receipt["generation_failure"]["all_generation_children_returned_code_1"] is True
    assert receipt["generation_failure"]["total_raw_byte_count"] == 0
    assert receipt["generation_failure"]["stderr_preserved_in_artifact"] is False
    assert receipt["generation_failure"]["root_cause_inferred"] is False


def test_scenario_infra_6365_model_specs_and_embedded_context_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6365-2: embedded tokenizer counts context before load."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    models = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=lambda path, prompt: _tokenizer(path, prompt),
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in models["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert models["all_resolved"] is True
    assert models["autotokenizer_usage_count"] == 0
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in models["MODEL_SPECS"])

    context = mod.context_capacity_receipt(
        model_id=mod.MANDATED_MODEL_IDS[0],
        prompt_tokens=30,
        requested_output_tokens=4,
        n_ctx=32,
    )
    assert context["fits"] is False
    assert context["capacity_margin"] == -2
    with pytest.raises(ValueError, match="context_overflow"):
        mod.ensure_context_capacity(context)


def test_scenario_infra_6365_observable_child_runner_failures(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6365-3: child failures preserve sidecars and fail closed."""

    nonzero = mod.run_observable_child(
        call_id="nonzero",
        model_hf_id="fixture/nonzero-GGUF",
        argv=[
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)",
        ],
        prompt="prompt",
        prompt_token_count=1,
        requested_output_tokens=1,
        n_ctx=8,
        output_dir=tmp_path,
        timeout_s=5,
        source_hash="sha256:source",
        dispatcher="fixture",
        env_allowlist={"CUDA_VISIBLE_DEVICES": "0", "SECRET_TOKEN": "hidden"},
    )
    assert nonzero["returncode"] == 7
    assert nonzero["contract_ok"] is False
    assert nonzero["stdout_byte_count"] > 0
    assert nonzero["stderr_byte_count"] > 0
    assert "SECRET_TOKEN" not in json.dumps(nonzero["argv_sanitized"])
    assert "SECRET_TOKEN" not in json.dumps(nonzero["environment_allowlist"])

    timeout = mod.run_observable_child(
        call_id="timeout",
        model_hf_id="fixture/timeout-GGUF",
        argv=[sys.executable, "-c", "import time; time.sleep(2)"],
        prompt="prompt",
        prompt_token_count=1,
        requested_output_tokens=1,
        n_ctx=8,
        output_dir=tmp_path,
        timeout_s=0.1,
        source_hash="sha256:source",
        dispatcher="fixture",
        env_allowlist={},
    )
    assert timeout["timed_out"] is True
    assert timeout["contract_ok"] is False
    assert timeout["signal"] in {"SIGKILL", None}

    empty = mod.run_observable_child(
        call_id="empty",
        model_hf_id="fixture/empty-GGUF",
        argv=[sys.executable, "-c", "import sys; print('CARNOT_USAGE:{}', file=sys.stderr)"],
        prompt="prompt",
        prompt_token_count=1,
        requested_output_tokens=1,
        n_ctx=8,
        output_dir=tmp_path,
        timeout_s=5,
        source_hash="sha256:source",
        dispatcher="fixture",
        env_allowlist={},
    )
    assert empty["returncode"] == 0
    assert empty["stdout_byte_count"] == 0
    assert empty["usage_receipt_valid"] is False
    assert empty["contract_ok"] is False

    matrix = mod.failure_injection_matrix(tmp_path)
    assert {row["injection"] for row in matrix["rows"]} == set(mod.FAILURE_INJECTION_NAMES)
    assert matrix["all_fail_closed"] is True
    assert all(row["diagnostics_preserved"] for row in matrix["rows"])


def test_scenario_infra_6365_gpu_samples_and_vram_are_required(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6365-4: missing GPU samples and VRAM proof fail closed."""

    good = _runtime_row(mod.MANDATED_MODEL_IDS[0], tmp_path)
    assert mod.live_model_contract_ok(good) is True
    assert mod.vram_rise_and_release_receipt(good)["proved_rise_and_release"] is True

    gpu1_row = deepcopy(good)
    gpu1_row["gpu_samples_by_phase"]["after_load"] = [
        {**good["gpu_samples_by_phase"]["after_load"][0], "gpu_index": 0, "memory_used_mb": 4},
        {**good["gpu_samples_by_phase"]["after_load"][0], "gpu_index": 1, "memory_used_mb": 1200},
    ]
    assert mod.vram_rise_and_release_receipt(gpu1_row)["peak_memory_used_mb"] == 1200

    missing_sample = deepcopy(good)
    missing_sample["gpu_samples_by_phase"].pop("after_load")
    assert mod.missing_gpu_phases(missing_sample) == ["after_load"]
    assert mod.live_model_contract_ok(missing_sample) is False

    no_release = deepcopy(good)
    no_release["gpu_samples_by_phase"]["after_cleanup"][0]["memory_used_mb"] = 900
    receipt = mod.vram_rise_and_release_receipt(no_release)
    assert receipt["proved_rise_and_release"] is False
    assert mod.live_model_contract_ok(no_release) is False


def test_scenario_infra_6365_artifact_schema_score_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-INFRA-6365-5: artifact is annotated, checksummed, and non-claiming."""

    artifact = _artifact(tmp_path)
    errors = mod.validate_artifact(artifact)

    assert errors == []
    assert artifact["status"] == "complete"
    assert artifact["gguf_runtime_observability_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_proposal_quality_or_utility_claim"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert "gguf_runtime_observability_ready_score" in artifact["field_principles"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blocked = _artifact(tmp_path / "blocked", missing_gpu_sample=True)
    assert blocked["gguf_runtime_observability_ready_score"] == 0.0
    assert blocked["status"] == "complete_null"
    assert blocked["honest_verdict"].startswith("complete_null:")

    bad = deepcopy(artifact)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "wrong"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["autotokenizer_usage_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "autotokenizer_usage_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["no_proposal_quality_or_utility_claim"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "proposal quality or utility claim is forbidden" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    del bad["field_principles"]["gguf_runtime_observability_ready_score"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing score gate principle" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    overflow = mod.context_overflow_row(
        {"hf_id": mod.MANDATED_MODEL_IDS[0]},
        {
            "model_hf_id": mod.MANDATED_MODEL_IDS[0],
            "prompt_tokens": 9,
            "requested_output_tokens": 1,
            "n_ctx": 4,
            "capacity_margin": -6,
            "fits": False,
        },
        tmp_path / "overflow",
    )
    assert overflow["usage_receipt_valid"] is False
    assert Path(str(overflow["stderr_path"])).is_file()

    path = mod.write_artifact(artifact, tmp_path / "artifact.json")
    assert json.loads(path.read_text(encoding="utf-8")) == artifact

    paths = _model_paths(tmp_path / "write" / "models")
    calls: list[dict[str, Any]] = []
    written_path = tmp_path / "write" / mod.RESULT_RELATIVE_PATH.name
    written = mod.run(
        date="20260813",
        result_path=written_path,
        output_dir=tmp_path / "write" / "run",
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        runtime=FakeRuntime(tmp_path / "write"),
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=True,
    )
    assert written_path.is_file()
    assert written["status"] == "complete"

    def overflow_tokenizer(path: str, prompt: str) -> dict[str, Any]:  # noqa: ARG001
        return {
            "method": mod.TOKENIZER_METHOD,
            "loadable": True,
            "prompt_tokens": 999,
            "tokenizer_detail": "overflow",
        }

    overflow_run = mod.run(
        date="20260813",
        result_path=tmp_path / "overflow-run.json",
        output_dir=tmp_path / "overflow-run",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=overflow_tokenizer,
        runtime=FakeRuntime(tmp_path / "overflow-run"),
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )
    assert overflow_run["status"] == "complete_null"

    validate_root = tmp_path / "validate-root"
    validate_path = validate_root / mod.RESULT_RELATIVE_PATH
    mod.write_artifact(artifact, validate_path)
    monkeypatch.setattr(mod, "REPO_ROOT", validate_root)
    assert mod.main(["--validate"]) == 0

    monkeypatch.setattr(mod, "validate_artifact", lambda payload: ["forced"])  # noqa: ARG005
    failed = mod.run(
        date="20260813",
        result_path=tmp_path / "failed.json",
        output_dir=tmp_path / "failed-run",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        runtime=FakeRuntime(tmp_path / "failed-run"),
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )
    assert failed["status"] == "failed_schema"

    monkeypatch.setattr(
        mod,
        "run",
        lambda **kwargs: {  # noqa: ARG005
            "status": "blocked_test",
            "honest_verdict": "blocked: test",
            "reproducibility_checksum": "sha256:test",
        },
    )
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out
