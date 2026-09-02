"""Tests for the live relation acquisition authenticity canary.

Spec refs: REQ-INFERENCE-6899 and SCENARIO-INFERENCE-6899-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_6899_live_relation_acquisition_canary as mod


ROOT = Path(__file__).resolve().parents[2]
NOW_NS = 9_000_000_000_000


def _upstream() -> dict[str, Any]:
    rows = mod.base.build_frozen_source_records()
    return {
        "relation_fixture_ready_score": 1,
        "relation_schema_version": "anchored_relation_v1",
        "calibration_group_manifest": {
            "public_fixture_manifest_hash": mod.EXPECTED_FIXTURE_HASHES["calibration"]
        },
        "sealed_held_group_manifest": {
            "public_fixture_manifest_hash": mod.EXPECTED_FIXTURE_HASHES["held"]
        },
        "rows": [
            {
                "row_type": "fixture",
                "fixture_id": row["fixture_id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "split": row["split"],
                "source_text_hash": row["source_text_hash"],
            }
            for row in rows
        ],
        "enoki_asset_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
    }


def _models() -> list[dict[str, Any]]:
    rows = []
    for index, hf_id in enumerate(mod.MODEL_SPECS):
        binding = mod.EXPECTED_MODEL_BINDINGS[hf_id]
        rows.append(
            {
                "hf_id": hf_id,
                "model_path": f"/cache/snapshots/{binding['snapshot_identity']}/{binding['filename']}",
                "sha256": binding["sha256"],
                "snapshot_identity": binding["snapshot_identity"],
                "model_size_bytes": binding["size_bytes"],
                "gpu": 0,
            }
        )
    return rows


def _tokenizers() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": True,
            "used_hf_autotokenizer": False,
            "probe_matches_frozen_receipt": True,
            "canonical_tokenizer_payload_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
            "model_sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "receipt_monotonic_ns": NOW_NS - 1_000_000,
        }
        for hf_id in mod.MODEL_SPECS
    ]


def _gpu_inventory() -> list[dict[str, Any]]:
    return [
        {
            "index": 0,
            "gpu_uuid": "GPU-test",
            "name": "NVIDIA GeForce RTX 3090",
            "free_vram_mb": mod.MIN_FREE_VRAM_MB + 120,
            "total_vram_mb": 24_576,
        }
    ]


def _lease_probes() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "owned": True,
            "released": True,
            "gpu_uuid": "GPU-test",
        }
        for hf_id in mod.MODEL_SPECS
    ]


def _precondition_args() -> dict[str, Any]:
    return {
        "upstream": _upstream(),
        "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
        "models": _models(),
        "tokenizer_receipts": _tokenizers(),
        "enoki_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
        "gpu_inventory": _gpu_inventory(),
        "lease_probe_rows": _lease_probes(),
        "cuda_offload_supported": True,
        "held_sidecar_access_count": 0,
        "now_monotonic_ns": NOW_NS,
    }


def _preconditions() -> dict[str, Any]:
    report = mod.evaluate_preconditions(**_precondition_args())
    assert report["gate_check_summary"]["passed"] is True
    return report


def _live_rows(*, accepted: bool = True) -> list[dict[str, Any]]:
    rows = []
    sources = mod.select_canary_sources(_upstream())
    for model_index, hf_id in enumerate(mod.MODEL_SPECS):
        for source_index, source in enumerate(sources):
            for seed in mod.SEEDS:
                raw_output = mod.base._rule_output(source)
                parse_rows = mod.base.parse_relation_output(raw_output, source)
                if not accepted and source_index >= 2:
                    raw_output = "not a protocol line"
                    parse_rows = mod.base.parse_relation_output(raw_output, source)
                request_bytes = mod.build_request_bytes(source, seed)
                output_bytes = raw_output.encode("utf-8")
                rows.append(
                    mod.build_live_cell(
                        hf_id=hf_id,
                        source=source,
                        seed=seed,
                        raw_request_bytes=request_bytes,
                        raw_http_response_bytes=b'{"choices":[{"message":{"content":"x"}}]}',
                        raw_output_bytes=output_bytes,
                        native_prompt_tokens=100,
                        generated_tokens=20,
                        stop_reason="stop",
                        wall_time_s=1.0,
                        timed_out=False,
                        truncated=False,
                        parser_attempted=True,
                        parse_rows=parse_rows,
                        runtime_receipt={
                            "authentic": True,
                            "model_sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
                            "tokenizer_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
                            "server_pid": 200 + model_index,
                            "server_start_time_ticks": 1_000 + model_index,
                            "command_hash": f"sha256:command-{model_index}",
                            "process_identity_match": True,
                            "receipt_age_s": 0.1,
                            "gpu_uuid": f"GPU-{model_index}",
                            "offload_layers": 40,
                            "owned_cuda_residency": True,
                            "vram_before_mb": 100,
                            "vram_after_load_mb": 20_000,
                            "stderr_tail": "offloaded 40 layers",
                            "teardown_outcome": {"leak_free": True},
                        },
                    )
                )
    return rows


def _llama_receipts() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "pid": 200 + index,
            "start_time_ticks": 1_000 + index,
            "command_hash": f"sha256:command-{index}",
            "model_sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "tokenizer_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
            "gpu_uuid": f"GPU-{index}",
            "offload_layers": 40,
            "owned_cuda_residency": True,
            "stderr_tail": "offloaded 40 layers",
        }
        for index, hf_id in enumerate(mod.MODEL_SPECS)
    ]


def _lifecycle_rows() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "pid": 200 + index,
            "start_time_ticks": 1_000 + index,
            "process_identity_match": True,
            "process_exit_confirmed": True,
            "process_reaped": True,
            "port_release_confirmed": True,
            "lease_released": True,
            "unrelated_process_signal_count": 0,
            "leak_free": True,
            "teardown_error": "",
        }
        for index, hf_id in enumerate(mod.MODEL_SPECS)
    ]


def _lease_rows() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "gpu_uuid": f"GPU-{index}",
            "owned": True,
            "released": True,
        }
        for index, hf_id in enumerate(mod.MODEL_SPECS)
    ]


def _controls() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = mod.select_canary_sources(_upstream())
    enoki = [
        {
            "control": "enoki",
            "fixture_id": source["fixture_id"],
            "family": source["family"],
            "raw_output": "[]",
            "parser_attempted": True,
        }
        for source in sources
    ]
    rule = [
        {
            "control": "rule",
            "fixture_id": source["fixture_id"],
            "family": source["family"],
            "raw_output": mod.base._rule_output(source),
            "parser_attempted": True,
        }
        for source in sources
    ]
    return enoki, rule


def _acquisition() -> dict[str, Any]:
    enoki, rule = _controls()
    return {
        "rows": _live_rows(),
        "llama_cpp_receipts": _llama_receipts(),
        "gpu_lease_rows": _lease_rows(),
        "server_lifecycle_rows": _lifecycle_rows(),
        "enoki_control_rows": enoki,
        "rule_control_rows": rule,
        "live_duration_s": 61.0,
    }


def test_req_inference_6899_spec_owns_contract() -> None:
    """REQ-INFERENCE-6899 is present before implementation behavior."""

    text = (ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text()
    assert "REQ-INFERENCE-6899" in text
    for suffix in (
        "PRECONDITIONS",
        "CELL-AUTHENTICITY",
        "FAILURES",
        "PROCESS",
        "TEARDOWN",
        "CONTROLS",
    ):
        assert f"SCENARIO-INFERENCE-6899-{suffix}" in text


def test_scenario_6899_preconditions_reject_each_authenticity_drift() -> None:
    """SCENARIO-INFERENCE-6899-PRECONDITIONS rejects frozen-input drift."""

    evaluate = mod.evaluate_preconditions(**_precondition_args())
    assert evaluate["gate_check_summary"]["passed"] is True
    mutations = {
        "cache_drift": lambda value: value.update(upstream_sha256="sha256:drift"),
        "fixture_drift": lambda value: value["upstream"]["calibration_group_manifest"].update(
            public_fixture_manifest_hash="sha256:drift"
        ),
        "wrong_model_file": lambda value: value["models"][0].update(model_path="/cache/wrong.gguf"),
        "tokenizer_substitution": lambda value: value["tokenizer_receipts"][0].update(
            source="transformers_autotokenizer", used_hf_autotokenizer=True
        ),
        "stale_receipt": lambda value: value["tokenizer_receipts"][0].update(
            receipt_monotonic_ns=NOW_NS - mod.RECEIPT_MAX_AGE_NS - 1
        ),
        "low_vram": lambda value: value["gpu_inventory"][0].update(
            free_vram_mb=mod.MIN_FREE_VRAM_MB - 1
        ),
        "lease_absent": lambda value: value["lease_probe_rows"][0].update(owned=False),
        "held_access": lambda value: value.update(held_sidecar_access_count=1),
    }
    for expected, mutate in mutations.items():
        values = deepcopy(_precondition_args())
        mutate(values)
        report = mod.evaluate_preconditions(**values)
        assert report["gate_check_summary"]["passed"] is False, expected
        failure = report["gate_check_summary"]["failed_checks"][0]
        assert {"check", "expected", "observed", "passed"} <= set(failure)


def test_scenario_6899_resolves_pair_then_dense(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6899-PRECONDITIONS resolves every exact GGUF."""

    paths = [tmp_path / binding["filename"] for binding in mod.EXPECTED_MODEL_BINDINGS.values()]
    for path in paths:
        path.write_bytes(b"gguf")
    calls: list[str] = []

    def pair_provider(**_: Any) -> list[dict[str, Any]]:
        calls.append("pair")
        return [
            {"hf_id": mod.MODEL_SPECS[0], "model_path": str(paths[0]), "gpu": 0},
            {"hf_id": mod.MODEL_SPECS[2], "model_path": str(paths[2]), "gpu": 0},
        ]

    def dense_resolver(hf_id: str, _: str) -> str:
        calls.append(hf_id)
        return str(paths[1])

    rows = mod.resolve_three_models(pair_provider=pair_provider, dense_resolver=dense_resolver)
    assert calls == ["pair", mod.MODEL_SPECS[1]]
    assert [row["hf_id"] for row in rows] == list(mod.MODEL_SPECS)
    absent = mod.resolve_three_models(pair_provider=lambda **_: [], dense_resolver=lambda *_: "")
    assert all("model_path" not in row for row in absent)


def test_scenario_6899_native_tokenizer_probe_is_process_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFERENCE-6899-PRECONDITIONS frees tokenizer CUDA state."""

    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "probe_token_ids": mod.base.EXPECTED_NATIVE_PROBE_IDS[mod.MODEL_SPECS[0]],
                    "vocabulary_size": 248_320,
                }
            ),
            stderr="native warning",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    receipt = mod._native_tokenizer_receipt(_models()[0])
    assert receipt["loadable"] is True
    assert receipt["probe_matches_frozen_receipt"] is True
    assert receipt["used_hf_autotokenizer"] is False
    assert receipt["model_sha256"] == mod.EXPECTED_MODEL_BINDINGS[mod.MODEL_SPECS[0]]["sha256"]
    assert "from llama_cpp import Llama" in captured["command"][2]
    assert captured["kwargs"]["timeout"] == mod.TOKENIZER_TIMEOUT_S


def test_scenario_6899_prompt_is_source_only_plain_protocol() -> None:
    """SCENARIO-INFERENCE-6899-CELL-AUTHENTICITY freezes one plain protocol."""

    sources = mod.select_canary_sources(_upstream())
    assert len(sources) == 5
    assert {row["family"] for row in sources} == set(mod.base.FAMILIES)
    assert mod.sha256_json(sources) == mod.EXPECTED_CANARY_SOURCE_HASH
    prompt = mod.build_prompt(sources[0])
    assert sources[0]["source_text"] in prompt
    assert mod.PROTOCOL_LINE in prompt
    assert "0:9=Evidence:" in prompt
    assert "first field must be the literal REL" in prompt
    assert "actual U+0009 TAB" in prompt
    assert mod.base.audit_arm_input(prompt) == []
    request = json.loads(mod.build_request_bytes(sources[0], mod.SEEDS[0]))
    assert request["messages"] == [{"role": "user", "content": prompt}]
    assert "grammar" not in request
    assert "response_format" not in request


def test_scenario_6899_cell_rejects_transport_and_parser_failures() -> None:
    """SCENARIO-INFERENCE-6899-FAILURES covers each terminal transport failure."""

    base_row = _live_rows()[0]
    assert mod.live_cell_errors(base_row) == []
    mutations = {
        "zero_offload": lambda row: row["runtime_receipt"].update(offload_layers=0),
        "empty_bytes": lambda row: row.update(
            raw_output_b64="", raw_output_sha256=mod.sha256_bytes(b""), output_byte_count=0
        ),
        "zero_generated_tokens": lambda row: row.update(generated_tokens=0),
        "parser_bypass": lambda row: row.update(parser_attempted=False),
        "timeout": lambda row: row.update(timed_out=True, stop_reason="timeout"),
        "truncation": lambda row: row.update(truncated=True, stop_reason="length"),
        "pid_reuse": lambda row: row["runtime_receipt"].update(process_identity_match=False),
        "stale_receipt": lambda row: row["runtime_receipt"].update(
            receipt_age_s=mod.RECEIPT_MAX_AGE_S + 1
        ),
    }
    for expected, mutate in mutations.items():
        row = deepcopy(base_row)
        mutate(row)
        assert expected in mod.live_cell_errors(row)


def test_scenario_6899_cell_rejects_every_corrupt_receipt_field() -> None:
    """SCENARIO-INFERENCE-6899-FAILURES rejects corrupt replay evidence."""

    base_row = _live_rows()[0]
    mutations = {
        "request_bytes": lambda row: row.update(raw_request_b64="not base64!"),
        "request_hash": lambda row: row.update(raw_request_sha256="sha256:wrong"),
        "http_response_bytes": lambda row: row.update(raw_http_response_b64=""),
        "http_response_hash": lambda row: row.update(raw_http_response_sha256="sha256:wrong"),
        "output_hash": lambda row: row.update(raw_output_sha256="sha256:wrong"),
        "zero_prompt_tokens": lambda row: row.update(native_prompt_tokens=0),
        "missing_stop_reason": lambda row: row.update(stop_reason=""),
        "runtime_authenticity": lambda row: row["runtime_receipt"].update(authentic=False),
        "wrong_model_file": lambda row: row["runtime_receipt"].update(model_sha256="sha256:wrong"),
        "tokenizer_substitution": lambda row: row["runtime_receipt"].update(
            tokenizer_sha256="sha256:wrong"
        ),
        "cuda_authenticity": lambda row: row["runtime_receipt"].update(owned_cuda_residency=False),
        "server_pid": lambda row: row["runtime_receipt"].update(server_pid=0),
        "server_start_time_ticks": lambda row: row["runtime_receipt"].update(
            server_start_time_ticks=0
        ),
        "teardown_failure": lambda row: row.update(teardown_outcome={"leak_free": False}),
    }
    for expected, mutate in mutations.items():
        row = deepcopy(base_row)
        mutate(row)
        assert expected in mod.live_cell_errors(row)


def test_scenario_6899_source_selection_rejects_family_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFERENCE-6899-PRECONDITIONS binds the exact five sources."""

    valid = mod.select_canary_sources(_upstream())
    wrong_families = deepcopy(valid)
    wrong_families[0]["family"] = "not_a_fixture_family"
    monkeypatch.setattr(mod.base, "select_source_records", lambda _: valid + wrong_families)
    with pytest.raises(mod.RelationCanaryError, match="canary_family_matrix_drift"):
        mod.select_canary_sources(_upstream())
    drifted = deepcopy(valid)
    drifted[0]["source_text"] += " drift"
    monkeypatch.setattr(mod.base, "select_source_records", lambda _: valid + drifted)
    with pytest.raises(mod.RelationCanaryError, match="canary_source_hash_drift"):
        mod.select_canary_sources(_upstream())


def test_scenario_6899_process_identity_detects_pid_reuse() -> None:
    """SCENARIO-INFERENCE-6899-PROCESS compares immutable Linux identity."""

    recorded = {
        "pid": 200,
        "start_time_ticks": 1_000,
        "uid": 1_000,
        "command_hash": "sha256:command",
    }
    assert mod.process_identity_errors(recorded, dict(recorded)) == []
    for field in ("pid", "start_time_ticks", "uid", "command_hash"):
        current = dict(recorded)
        current[field] = "changed"
        assert field in mod.process_identity_errors(recorded, current)


def test_scenario_6899_canned_rows_are_source_bound() -> None:
    """SCENARIO-INFERENCE-6899-FAILURES rejects repeated output across prompts."""

    rows = _live_rows()
    assert mod.detect_canned_cells(rows) == []
    first = rows[0]
    other = next(
        row
        for row in rows
        if row["hf_id"] == first["hf_id"] and row["fixture_id"] != first["fixture_id"]
    )
    other.update(
        raw_output_b64=first["raw_output_b64"],
        raw_output_sha256=first["raw_output_sha256"],
        output_byte_count=first["output_byte_count"],
    )
    canned = mod.detect_canned_cells(rows)
    assert {first["cell_identity"], other["cell_identity"]} <= {
        row["cell_identity"] for row in canned
    }
    cross_model = next(
        row
        for row in rows
        if row["hf_id"] != first["hf_id"] and row["fixture_id"] != first["fixture_id"]
    )
    cross_model.update(
        raw_output_b64=first["raw_output_b64"],
        raw_output_sha256=first["raw_output_sha256"],
        output_byte_count=first["output_byte_count"],
    )
    cross_only = mod.detect_canned_cells([first, cross_model])
    assert {first["cell_identity"], cross_model["cell_identity"]} == {
        row["cell_identity"] for row in cross_only
    }


def test_req_6899_readiness_rejects_each_live_gate() -> None:
    """REQ-INFERENCE-6899 derives readiness only from authentic GGUF rows."""

    inputs = {
        "duration_s": 61.0,
        "rows": _live_rows(),
        "llama_cpp_receipts": _llama_receipts(),
        "gpu_lease_rows": _lease_rows(),
        "server_lifecycle_rows": _lifecycle_rows(),
        "tokenizer_receipts": _tokenizers(),
        "held_sidecar_access_count": 0,
    }
    score, summary = mod.readiness(**inputs)
    assert score == 1
    assert summary["passed"] is True
    mutations = {
        "duration": lambda value: value.update(duration_s=59.9),
        "parse_coverage": lambda value: value.update(rows=_live_rows(accepted=False)),
        "teardown": lambda value: value["server_lifecycle_rows"][0].update(leak_free=False),
        "lease": lambda value: value["gpu_lease_rows"][0].update(released=False),
        "model_receipt": lambda value: value["llama_cpp_receipts"][0].update(
            model_sha256="sha256:wrong"
        ),
        "missing_cell": lambda value: value["rows"].pop(),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(inputs)
        mutate(changed)
        score, summary = mod.readiness(**changed)
        assert score == 0, expected
        assert summary["passed"] is False


def test_scenario_6899_controls_never_replace_live_rows() -> None:
    """SCENARIO-INFERENCE-6899-CONTROLS excludes both controls from readiness."""

    enoki, rule = _controls()
    score, _ = mod.readiness(
        duration_s=61.0,
        rows=[],
        llama_cpp_receipts=[],
        gpu_lease_rows=[],
        server_lifecycle_rows=[],
        tokenizer_receipts=_tokenizers(),
        held_sidecar_access_count=0,
    )
    assert len(enoki) == len(rule) == 5
    assert score == 0


def test_req_6899_artifact_has_replayable_required_fields() -> None:
    """REQ-INFERENCE-6899 emits every required field and one principle each."""

    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        acquisition=_acquisition(),
    )
    assert artifact["relation_canary_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["rows"]) == len(mod.MODEL_SPECS) * 5 * len(mod.SEEDS)
    assert artifact["empty_cell_count"] == 0
    assert artifact["canned_cell_count"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert mod.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["rows"][0]["output_byte_count"] = 0
    assert "row_authenticity" in mod.validate_artifact(changed)


def test_req_6899_validator_rejects_every_summary_drift() -> None:
    """REQ-INFERENCE-6899 independently replays every terminal summary field."""

    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        acquisition=_acquisition(),
    )
    mutations = {
        "required_fields": lambda row: row.pop("models_used"),
        "field_principles": lambda row: row.update(field_principles={}),
        "empty_cell_count": lambda row: row.update(empty_cell_count=1),
        "canned_cell_count": lambda row: row.update(canned_cell_count=1),
        "parse_coverage_by_model": lambda row: row.update(parse_coverage_by_model={}),
        "honest_verdict": lambda row: row.update(honest_verdict="not_terminal"),
        "verdict_class": lambda row: row.update(verdict_class="unknown"),
        "verifier_is_oracle": lambda row: row.update(verifier_is_oracle=True),
        "held_sidecar_access_count": lambda row: row.update(held_sidecar_access_count=1),
        "relation_canary_ready_score": lambda row: row.update(relation_canary_ready_score=0),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(artifact)
        mutate(changed)
        assert any(
            error == expected or error.startswith(f"{expected}:")
            for error in mod.validate_artifact(changed)
        )


def test_scenario_6899_precondition_records_invalid_source_manifest() -> None:
    """SCENARIO-INFERENCE-6899-PRECONDITIONS reports source manifest drift."""

    values = _precondition_args()
    values["upstream"]["rows"] = []
    report = mod.evaluate_preconditions(**values)
    assert report["gate_check_summary"]["passed"] is False
    source_check = next(
        row
        for row in report["gate_check_summary"]["checks"]
        if row["check"] == "canary_source_hash"
    )
    assert source_check["observed"] == "invalid"


def test_scenario_6899_teardown_error_is_explicit() -> None:
    """SCENARIO-INFERENCE-6899-TEARDOWN preserves the cleanup exception."""

    lifecycle = _lifecycle_rows()
    lifecycle[0]["teardown_error"] = "forced failure"
    assert any(error.endswith(":teardown_error") for error in mod._lifecycle_errors(lifecycle))


def test_req_6899_blocked_run_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-INFERENCE-6899 writes a full blocked artifact before acquisition."""

    result = tmp_path / "result.json"

    def blocked(_: Path) -> dict[str, Any]:
        values = _preconditions()
        values["gate_check_summary"] = mod.gate_summary(
            [mod.gate_check("exact_model_files", True, False)]
        )
        values.update(
            {
                "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
                "models": _models(),
                "tokenizer_receipts": _tokenizers(),
            }
        )
        return values

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result,
        precondition_collector=blocked,
    )
    assert artifact["honest_verdict"] == "complete_blocked_live_relation_acquisition_canary"
    assert artifact["relation_canary_ready_score"] == 0
    assert result.is_file()
    assert mod.validate_artifact(artifact) == []


def test_req_6899_injected_run_writes_only_requested_path(tmp_path: Path) -> None:
    """REQ-INFERENCE-6899 keeps unit tests outside the tracked result path."""

    result = tmp_path / "result.json"

    def ready(_: Path) -> dict[str, Any]:
        return {
            **_preconditions(),
            "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
        }

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result,
        precondition_collector=ready,
        acquisition_runner=lambda **_: _acquisition(),
    )
    assert artifact["relation_canary_ready_score"] == 1
    assert result.is_file()
    assert not (ROOT / "experiment_6899_live_relation_acquisition_canary.py").exists()


def test_req_6899_injected_run_refuses_invalid_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFERENCE-6899 fails closed before writing invalid live evidence."""

    def ready(_: Path) -> dict[str, Any]:
        return {
            **_preconditions(),
            "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
        }

    monkeypatch.setattr(mod, "validate_artifact", lambda _: ["forced_invalid"])
    with pytest.raises(mod.RelationCanaryError, match="artifact_invalid:forced_invalid"):
        mod.run(
            date="20260902",
            root=tmp_path,
            result_path=tmp_path / "invalid.json",
            precondition_collector=ready,
            acquisition_runner=lambda **_: _acquisition(),
        )


def test_req_6899_source_forbids_retired_and_constrained_mechanisms() -> None:
    """REQ-INFERENCE-6899 keeps GGUF generation native and unconstrained."""

    source = (
        ROOT / "python/carnot/experiment_6899_live_relation_acquisition_canary.py"
    ).read_text()
    assert "cached_sota_pair(" in source
    assert "resolve_cached_gguf(" in source
    assert "AutoTokenizer.from_pretrained" not in source
    for forbidden in ("grammar=", "response_format", "json_schema", "finite_answer_id"):
        assert forbidden not in source
    wrapper = ROOT / "scripts/experiments/experiment_6899_live_relation_acquisition_canary.py"
    assert wrapper.is_file()
