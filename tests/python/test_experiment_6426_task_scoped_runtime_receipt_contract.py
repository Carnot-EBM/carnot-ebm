"""Tests for Exp6426 task-scoped runtime receipt contract.

Spec refs: REQ-INFRA-6426, SCENARIO-INFRA-6426-1,
SCENARIO-INFRA-6426-2, SCENARIO-INFRA-6426-3,
SCENARIO-INFRA-6426-4, SCENARIO-INFRA-6426-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from carnot import experiment_6426_task_scoped_runtime_receipt_contract as mod
from carnot import task_runtime_receipts as receipts


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _sha(text: str) -> str:
    return receipts.sha256_text(text)


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.write_bytes(b"fixture gguf bytes for exp6426\n")
    return path


def _cached_pair(path: Path, calls: list[dict[str, Any]]):
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
        return [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": mod.MANDATED_POWERED_MODEL_ID,
                "gpu": gpu_indices[0],
                "model_path": str(path),
            },
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": gpu_indices[1],
                "model_path": str(path.with_name("qwen.fixture.gguf")),
            },
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
        "autotokenizer_used": False,
    }


def _runner_selection(control_id: str, substrate: str = "cuda_gguf") -> dict[str, Any]:
    selection = {
        "runner_id": f"fixture-{control_id}",
        "binary_path": sys.executable,
        "binary_sha256": _sha(sys.executable),
        "substrate": substrate,
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


def _row(
    *,
    control_id: str,
    phase: str,
    start: int,
    end: int,
    child_pid: int | None,
    raw: bytes,
    concurrency_group: str | None = None,
    runner_substrate: str = "cuda_gguf",
    device_ids: list[str] | None = None,
    gpu_samples: list[dict[str, Any]] | None = None,
    synthesized: int = 0,
    cpu_fallback: bool = False,
    exit_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    child_pids = [] if child_pid is None else [child_pid]
    return receipts.build_phase_row(
        task_id=mod.TASK_ID,
        control_id=control_id,
        phase=phase,
        monotonic_start_ns=start,
        monotonic_end_ns=end,
        wall_clock_start="2026-08-14T00:00:00Z",
        wall_clock_end="2026-08-14T00:00:01Z",
        parent_pid=6000,
        child_pids=child_pids,
        command=[sys.executable, "-c", control_id],
        config={"control_id": control_id, "phase": phase},
        model_identity={
            "hf_id": mod.MANDATED_POWERED_MODEL_ID,
            "model_sha256": "sha256:" + "1" * 64,
            "model_identity_bound": True,
        },
        runner_selection=_runner_selection(control_id, runner_substrate),
        device_ids=device_ids or ["GPU-fixture-0"],
        concurrency_group=concurrency_group or f"exp6426-{control_id}",
        raw_output_bytes=raw,
        exit_status=exit_status or {"returncode": 0, "timed_out": False, "signal": None},
        attribution_confidence=1.0,
        gpu_samples=gpu_samples or [],
        synthesized_runtime_fields=synthesized,
        cpu_fallback=cpu_fallback,
        blocked_reason="",
        extra={"first_token_or_completion_evidence": {"sha256": receipts.sha256_bytes(raw)}},
    )


def _control_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t = 1_000_000_000
    for control_index, control_id in enumerate(mod.CONTROL_IDS):
        child_pid = 7000 + control_index if control_id != "blocked" else None
        raw = f"raw output for {control_id}".encode()
        for phase in receipts.REQUIRED_PHASES:
            start = t
            end = t + 10_000_000
            samples: list[dict[str, Any]] = []
            if control_id == "powered" and phase == "generation":
                samples = [
                    {
                        "pid": child_pid,
                        "device_uuid": "GPU-fixture-0",
                        "gpu_index": 0,
                        "pid_memory_mb": 2048,
                        "device_memory_used_mb": 4096,
                        "monotonic_ns": start + 1_000_000,
                        "sample_age_s": 0.0,
                    }
                ]
            rows.append(
                _row(
                    control_id=control_id,
                    phase=phase,
                    start=start,
                    end=end,
                    child_pid=child_pid,
                    raw=raw,
                    runner_substrate="cpu" if control_id == "cpu" else "cuda_gguf",
                    device_ids=["CPU"]
                    if control_id in {"cpu", "blocked", "interrupted"}
                    else ["GPU-fixture-0"],
                    gpu_samples=samples,
                    exit_status={"returncode": -15, "timed_out": False, "signal": "SIGTERM"}
                    if control_id == "interrupted"
                    else None,
                )
            )
            t = end + 1_000_000
    return rows


class FixtureRuntime:
    """SCENARIO-INFRA-6426-2: deterministic powered rows stand in for CUDA."""

    def __init__(self, rows: list[dict[str, Any]], blocked: bool = False) -> None:
        self.rows = rows
        self.blocked = blocked
        self.calls = 0

    def preflight_receipts(self, model_specs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "both_rtx_3090_devices_visible": not self.blocked,
            "free_vram_ready": not self.blocked,
            "model_cache_ready": all(row.get("exists") for row in model_specs),
            "llama_cpp_cuda_supported": not self.blocked,
            "runner_binary_ready": not self.blocked,
            "tokenizer_metadata_ready": all(row.get("tokenizer_loadable") for row in model_specs),
            "disk_ready": True,
            "cpu_ready": True,
            "ram_ready": True,
            "monotonic_clock_ready": True,
            "blocked_reasons": ["fixture_powered_preflight_block"] if self.blocked else [],
        }

    def powered_control_rows(
        self,
        *,
        task_id: str,
        model: dict[str, Any],
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        assert task_id == mod.TASK_ID
        assert model["hf_id"] == mod.MANDATED_POWERED_MODEL_ID
        assert output_dir.exists()
        self.calls += 1
        return [deepcopy(row) for row in self.rows if row["control_id"] == "powered"]


def test_req_infra_6426_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6426: OpenSpec owns the runtime receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6426") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6426-1",
        "SCENARIO-INFRA-6426-2",
        "SCENARIO-INFRA-6426-3",
        "SCENARIO-INFRA-6426-4",
        "SCENARIO-INFRA-6426-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MANDATED_POWERED_MODEL_ID,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_infra_6426_helper_writes_atomic_partial_receipts(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6426-1: helper preserves partial rows after interruption."""

    path = tmp_path / "partial.json"
    writer = receipts.TaskScopedReceiptWriter(path, task_id=mod.TASK_ID)
    row = _row(
        control_id="interrupted",
        phase="generation",
        start=100,
        end=200,
        child_pid=7331,
        raw=b"partial",
        exit_status={"returncode": -15, "timed_out": False, "signal": "SIGTERM"},
    )
    writer.record_phase(row)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["schema_version"] == receipts.SCHEMA_VERSION
    assert payload["task_id"] == mod.TASK_ID
    assert payload["rows"] == [row]
    assert not list(tmp_path.glob("*.tmp"))

    writer.finalize({"status": "complete_partial_fixture"})
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["status"] == "complete_partial_fixture"
    assert final["rows"] == [row]


def test_scenario_infra_6426_duration_recompute_rejects_bad_intervals() -> None:
    """SCENARIO-INFRA-6426-4: duration uses monotonic phase intervals only."""

    rows = _control_rows()
    report = receipts.validate_contract_rows(rows, expected_controls=mod.CONTROL_IDS)
    assert report["accepted"] is True
    assert report["recomputed_duration_s"] == 0.2
    assert report["synthesized_runtime_field_count"] == 0
    assert report["cpu_fallback_count"] == 0

    bad = deepcopy(rows)
    bad[0]["monotonic_end_ns"] = bad[0]["monotonic_start_ns"] - 1
    assert (
        "negative_interval"
        in receipts.validate_contract_rows(bad, expected_controls=mod.CONTROL_IDS)["reasons"]
    )

    bad = deepcopy(rows)
    bad[1]["monotonic_start_ns"] = bad[0]["monotonic_start_ns"] + 1
    assert (
        "overlap_unexplained"
        in receipts.validate_contract_rows(bad, expected_controls=mod.CONTROL_IDS)["reasons"]
    )

    bad = deepcopy(rows)
    bad[0]["synthesized_runtime_fields"] = 1
    assert (
        "synthesized_runtime_field"
        in receipts.validate_contract_rows(bad, expected_controls=mod.CONTROL_IDS)["reasons"]
    )

    bad = deepcopy(rows)
    del bad[0]["monotonic_start_ns"]
    assert (
        "missing_monotonic_interval"
        in receipts.validate_contract_rows(bad, expected_controls=mod.CONTROL_IDS)["reasons"]
    )


def test_scenario_infra_6426_attack_matrix_fails_closed() -> None:
    """SCENARIO-INFRA-6426-5: critical attribution attacks fail closed."""

    rows = _control_rows()
    matrix = receipts.mutation_attack_matrix(rows, expected_controls=mod.CONTROL_IDS)

    assert {row["attack_id"] for row in matrix["rows"]} == set(receipts.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0

    cpu_fallback = receipts.mutate_rows_for_attack("cpu_fallback", rows)
    report = receipts.validate_contract_rows(cpu_fallback, expected_controls=mod.CONTROL_IDS)
    assert "cpu_fallback" in report["reasons"]


def test_req_infra_6426_defensive_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6426: defensive branches reject malformed receipt evidence."""

    assert receipts.sha256_file(tmp_path / "missing.bin") is None
    assert receipts._int_value(None) is None
    assert receipts._int_value(True) is None
    assert receipts._int_value("bad") is None
    assert receipts._as_mapping([]) == {}
    assert mod._signal_name(None) is None
    assert mod._signal_name(0) is None
    assert mod._signal_name(-9) == "SIGKILL"
    assert mod._signal_name(-999_999) == "signal_999999"
    assert mod._revision_from_path("/cache/snapshots/rev/model.gguf") == "rev"
    assert mod._revision_from_path("/cache/no-snapshot/model.gguf") is None
    assert mod._quantization_from_path("model-without-quant.gguf") == "unknown"
    assert mod._file_prefix_sha256(tmp_path / "missing.gguf") is None

    rows = _control_rows()
    bad = deepcopy(rows)
    bad[0]["wall_clock_start"] = ""
    bad[0]["parent_pid"] = 1
    bad[0]["command_hash"] = "bad"
    bad[0]["config_hash"] = "bad"
    bad[0]["raw_output_hash"] = "bad"
    bad[0]["runner_selection"]["selected"] = False
    bad[0]["attribution_confidence"] = 0.0
    bad[0]["monotonic_start_ns"] = None
    powered = next(
        row for row in bad if row["control_id"] == "powered" and row["phase"] == "generation"
    )
    powered["gpu_samples"] = []
    report = receipts.validate_contract_rows(bad, expected_controls=mod.CONTROL_IDS)
    assert {
        "missing_monotonic_interval",
        "wall_clock_interval_missing",
        "parent_pid_invalid",
        "command_hash_missing",
        "config_hash_missing",
        "raw_output_hash_missing",
        "runner_not_selected",
        "low_attribution_confidence",
        "pid_linked_gpu_sample_missing",
    } <= set(report["reasons"])

    assert (
        receipts.raw_hash_duplicate_count(receipts.mutate_rows_for_attack("raw_output_reuse", rows))
        == 1
    )
    assert receipts.control_phase_counter(rows) == {control_id: 5 for control_id in mod.CONTROL_IDS}
    with pytest.raises(ValueError, match="unknown attack_id"):
        receipts.mutate_rows_for_attack("unknown", rows)

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, text: {  # noqa: ARG005
            "source": mod.TOKENIZER_SOURCE,
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": "missing",
            "autotokenizer_used": True,
        },
    )
    assert {
        f"missing_cached_sota_pair_row:{mod.MANDATED_POWERED_MODEL_ID}",
        f"missing_gguf_file:{mod.MANDATED_POWERED_MODEL_ID}",
        f"embedded_tokenizer_unavailable:{mod.MANDATED_POWERED_MODEL_ID}",
        f"autotokenizer_used:{mod.MANDATED_POWERED_MODEL_ID}",
    } <= set(missing["blocked_reasons"])

    preconditions = mod.preconditions_from(
        date="20260814",
        model_resolution={"blocked_reasons": [], "MODEL_SPECS": []},
        runtime_preflight={"blocked_reasons": []},
        source_before={"source": None},
        protected_before={"protected": None},
    )
    assert {"source_hash_missing", "protected_hash_missing"} <= set(
        preconditions["blocked_reasons"]
    )
    assert (
        mod.status({"blocked_reason": "", "runtime_receipt_contract_ready_score": 0.0})
        == "complete_null"
    )
    assert mod.honest_verdict({"status": "complete_null"}).startswith("complete_null:")


def test_scenario_infra_6426_model_specs_and_artifact_ready(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6426-2 and SCENARIO-INFRA-6426-3: artifact gates readiness."""

    model_path = _model_file(tmp_path)
    calls: list[dict[str, Any]] = []
    rows = _control_rows()
    runtime = FixtureRuntime(rows)
    artifact = mod.run(
        date="20260814",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        cached_pair_func=_cached_pair(model_path, calls),
        tokenizer_func=_tokenizer,
        runtime=runtime,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    assert calls == [
        {
            "gpu_indices": (0, 1),
            "preferred_quant": "Q4_K_M",
            "model_indices": (1, 0),
        }
    ]
    assert runtime.calls == 1
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["runtime_receipt_contract_ready_score"] == 1.0
    assert artifact["MODEL_SPECS"][0]["hf_id"] == mod.MANDATED_POWERED_MODEL_ID
    assert artifact["models_used"] == [mod.MANDATED_POWERED_MODEL_ID]
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["synthesized_runtime_field_count"] == 0
    assert artifact["cpu_fallback_count"] == 0
    assert artifact["attribution_failure_count"] == 0
    assert artifact["reported_vs_recomputed_duration_delta"] <= 0.1
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blocked_runtime = FixtureRuntime(rows, blocked=True)
    blocked = mod.run(
        date="20260814",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        cached_pair_func=_cached_pair(model_path, []),
        tokenizer_func=_tokenizer,
        runtime=blocked_runtime,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    assert blocked_runtime.calls == 0
    assert blocked["status"] == "blocked_precondition"
    assert blocked["runtime_receipt_contract_ready_score"] == 0.0
    assert "fixture_powered_preflight_block" in blocked["blocked_reason"]

    bad = deepcopy(artifact)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["MODEL_SPECS"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "MODEL_SPECS missing mandated Gemma26 GGUF" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["autotokenizer_usage_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "autotokenizer_usage_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "wrong"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "invalid"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    path = tmp_path / "artifact.json"
    mod.write_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_req_infra_6426_run_write_and_schema_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-INFRA-6426: run writes artifacts and reports schema failure."""

    model_path = _model_file(tmp_path)
    written_path = tmp_path / "written.json"
    runtime = FixtureRuntime(_control_rows())
    written = mod.run(
        date="20260814",
        result_path=written_path,
        data_dir=tmp_path / "written-data",
        cached_pair_func=_cached_pair(model_path, []),
        tokenizer_func=_tokenizer,
        runtime=runtime,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )
    assert json.loads(written_path.read_text(encoding="utf-8")) == written

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda artifact: ["forced schema error"])  # noqa: ARG005
        failed = mod.run(
            date="20260814",
            result_path=tmp_path / "failed.json",
            data_dir=tmp_path / "failed-data",
            cached_pair_func=_cached_pair(model_path, []),
            tokenizer_func=_tokenizer,
            runtime=FixtureRuntime(_control_rows()),
            test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
            write=False,
        )
    assert failed["status"] == "failed_schema"
    assert failed["honest_verdict"].startswith("complete_failed_schema:")
