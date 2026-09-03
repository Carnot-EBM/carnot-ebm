"""Tests for the alias-safe immutable relation-corpus reducer.

Spec refs: REQ-REPORT-6912 and SCENARIO-REPORT-6912-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6900_authentic_anchored_relation_corpus as producer
from carnot import experiment_6912_alias_safe_relation_corpus_reducer as mod


def _source_record() -> dict[str, Any]:
    text = "Node n0 has color red."
    return {
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": "calibration",
        "source_order": 0,
        "source_text": text,
        "source_text_hash": producer.sha256_bytes(text.encode()),
        "relation_schema_version": "anchored_relation_v1",
        "allowed_predicates": ["has_color"],
    }


def _runtime(hf_id: str, index: int) -> dict[str, Any]:
    return {
        "authentic_attempt": True,
        "model_sha256": producer.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
        "tokenizer_sha256": producer.EXPECTED_TOKENIZER_HASHES[hf_id],
        "server_pid": 300 + index,
        "server_start_time_ticks": 2_000 + index,
        "command_hash": f"sha256:command-{index}",
        "process_identity_match": True,
        "receipt_age_s": 0.1,
        "receipt_monotonic_ns": 10_000_000_000 + index * 2_000_000_000,
        "gpu_uuid": f"GPU-{index}",
        "offload_layers": 40,
        "owned_cuda_residency": True,
        "vram_before_mb": 24_000,
        "vram_after_load_mb": 18_000,
    }


def _gguf_cell(source: dict[str, Any], hf_id: str, index: int) -> dict[str, Any]:
    raw_output = producer.base._rule_output(source).encode()
    raw_http = json.dumps({"choices": [{"finish_reason": "stop"}]}).encode()
    return producer.build_gguf_cell(
        hf_id=hf_id,
        source=source,
        seed=6899,
        raw_request_bytes=json.dumps({"seed": 6899, "model": hf_id}).encode(),
        raw_http_response_bytes=raw_http,
        raw_output_bytes=raw_output,
        native_prompt_tokens=12,
        generated_tokens=8,
        stop_reason="stop",
        wall_time_s=0.25,
        timed_out=False,
        truncated=False,
        parser_attempted=True,
        parse_rows=producer.parse_relation_output(raw_output.decode(), source),
        runtime_receipt=_runtime(hf_id, index),
    )


def _control_cell(source: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == producer.ENOKI_ARM:
        result = {
            "sentence": source["source_text"],
            "triples": [
                {
                    "confidence": 0.9,
                    "subject": "Node n0",
                    "relation": "has color",
                    "object": "red",
                }
            ],
        }
        raw_output = producer.canonical_json(result).encode()
        parse_rows = producer.base.parse_enoki_result(result, source)
        receipt = {
            "authentic_attempt": True,
            "encoder_revision": producer.ENOKI_REVISION,
            "encoder_asset_hash": producer.ENOKI_ASSET_HASH,
            "compatibility_adapter_sha256": producer.ENOKI_TRANSFORMERS5_ADAPTER_SHA256,
            "lease_released": True,
            "duration_s": 0.5,
        }
        stop_reason = "encoder_complete"
    else:
        raw_output = producer.base._rule_output(source).encode()
        parse_rows = producer.parse_relation_output(raw_output.decode(), source)
        receipt = {
            "authentic_attempt": True,
            "rule_version": producer.RULE_VERSION,
            "rule_source_sha256": producer.RULE_SOURCE_SHA256,
        }
        stop_reason = "rule_complete"
    return producer.build_control_cell(
        arm=arm,
        source=source,
        raw_request_bytes=source["source_text"].encode(),
        raw_output_bytes=raw_output,
        stop_reason=stop_reason,
        wall_time_s=0.1,
        parse_rows=parse_rows,
        runtime_receipt=receipt,
    )


def _cells() -> list[dict[str, Any]]:
    source = _source_record()
    cells = [_gguf_cell(source, hf_id, index) for index, hf_id in enumerate(mod.REQUIRED_MODELS)]
    cells.extend(
        [_control_cell(source, producer.ENOKI_ARM), _control_cell(source, producer.RULE_ARM)]
    )
    return cells


def _prior_attempt(source: dict[str, Any]) -> dict[str, Any]:
    return producer.build_control_cell(
        arm=producer.ENOKI_ARM,
        source=source,
        raw_request_bytes=source["source_text"].encode(),
        raw_output_bytes=b"",
        stop_reason="encoder_failure",
        wall_time_s=0.0,
        parse_rows=producer.parse_relation_output("", source),
        runtime_receipt={
            "authentic_attempt": False,
            "encoder_revision": producer.ENOKI_REVISION,
            "encoder_asset_hash": producer.ENOKI_ASSET_HASH,
            "compatibility_adapter_sha256": producer.ENOKI_TRANSFORMERS5_ADAPTER_SHA256,
            "lease_released": False,
            "duration_s": 0.4,
        },
    )


def _request_ref(cell: dict[str, Any], attempt: str | None = None) -> dict[str, Any]:
    row = {
        "cell_identity": cell["cell_identity"],
        "raw_request_b64": cell["raw_request_b64"],
        "raw_request_sha256": cell["raw_request_sha256"],
        "request_byte_count": cell["request_byte_count"],
    }
    if attempt:
        row["attempt_identity"] = attempt
    return row


def _output_ref(cell: dict[str, Any], attempt: str | None = None) -> dict[str, Any]:
    row = {
        "cell_identity": cell["cell_identity"],
        "raw_http_response_b64": cell["raw_http_response_b64"],
        "raw_http_response_sha256": cell["raw_http_response_sha256"],
        "http_response_byte_count": cell["http_response_byte_count"],
        "raw_output_b64": cell["raw_output_b64"],
        "raw_output_sha256": cell["raw_output_sha256"],
        "output_byte_count": cell["output_byte_count"],
    }
    if attempt:
        row["attempt_identity"] = attempt
    return row


def _source_artifact() -> dict[str, Any]:
    cells = _cells()
    source = _source_record()
    prior = _prior_attempt(source)
    flattened = producer.flatten_terminal_rows(cells) + producer._prior_terminal_rows([prior])
    return {
        "duration_s": 10.0,
        "live_duration_s": 10.0,
        "flagged_adversarial": True,
        "honest_verdict": mod.EXPECTED_SOURCE_VERDICT,
        "verdict_class": "positive",
        "corrigendum_pending": [deepcopy(mod.EXPECTED_TAUTOLOGY_FINDING)],
        "source_artifact_hashes": {"exp6899": {"sha256": mod.EXPECTED_SOURCE_HASHES["exp6899"]}},
        "prompt_manifest": {"source_count": 1, "seeds": [6899]},
        "cell_manifest": cells,
        "prior_attempt_cells": [prior],
        "raw_request_manifest": [*map(_request_ref, cells), _request_ref(prior, "prior::0")],
        "raw_output_manifest": [*map(_output_ref, cells), _output_ref(prior, "prior::0")],
        "rows": flattened,
        "parse_failure_rows": [
            row
            for row in flattened
            if row["status"] in {"malformed", "unsupported", "invalid_span", "duplicate"}
        ],
        "empty_rows": [row for row in flattened if row["status"] == "empty"],
        "timeout_rows": [row for row in flattened if row["status"] == "timeout"],
        "truncation_rows": [row for row in flattened if row["status"] == "truncated"],
        "abstention_rows": [row for row in flattened if row["status"] == "abstention"],
        "per_arm_family_counts": {
            arm: {"graph_coloring": 1}
            for arm in (*[f"gguf:{model}" for model in mod.REQUIRED_MODELS], *mod.CONTROL_ARMS)
        },
        "server_lifecycle_rows": [
            {
                "hf_id": hf_id,
                "pid": 300 + index,
                "start_time_ticks": 2_000 + index,
                "process_identity_match": True,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "port_release_confirmed": True,
                "lease_released": True,
                "leak_free": True,
            }
            for index, hf_id in enumerate(mod.REQUIRED_MODELS)
        ],
    }


def _replay(source: dict[str, Any]) -> dict[str, Any]:
    return mod.replay_cells(
        source,
        [_source_record()],
        required_models=mod.REQUIRED_MODELS,
        required_seeds=(6899,),
        expected_cell_count=5,
    )


def test_req_report_6912_spec_declares_requested_failure_boundaries() -> None:
    """REQ-REPORT-6912 owns every requested failure before implementation."""

    text = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6912") :]
    for suffix in (
        "SOURCE-HASH",
        "CELL-IDENTITY",
        "RAW-HASH",
        "IDENTITY-DRIFT",
        "OUTCOME-DRIFT",
        "DURATION-ALIAS",
        "TIMING",
        "AGGREGATES",
        "FLAG",
        "ADVERSARIAL",
    ):
        assert f"SCENARIO-REPORT-6912-{suffix}" in section


def test_source_hash_mutation_reports_exact_expected_and_observed() -> None:
    """SCENARIO-REPORT-6912-SOURCE-HASH rejects changed source bytes."""

    report = mod.check_exact_hashes({"exp6899": "sha256:changed"}, {"exp6899": "sha256:expected"})
    assert report[0] == {
        "check": "source_hash:exp6899",
        "expected": "sha256:expected",
        "observed": "sha256:changed",
        "passed": False,
    }


def test_missing_cell_and_duplicate_identity_block() -> None:
    """SCENARIO-REPORT-6912-CELL-IDENTITY preserves missing and duplicate IDs."""

    source = _source_artifact()
    missing = deepcopy(source)
    missing["cell_manifest"].pop()
    assert _replay(missing)["gate_check_summary"]["failed_check"] == "exact_cell_identity_set"

    duplicate = deepcopy(source)
    duplicate["cell_manifest"].append(deepcopy(duplicate["cell_manifest"][0]))
    replay = _replay(duplicate)
    assert replay["duplicate_cell_count"] == 1
    assert replay["gate_check_summary"]["passed"] is False


def test_raw_hash_and_file_reference_mismatch_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6912-RAW-HASH hashes embedded and file-backed bytes."""

    source = _source_artifact()
    source["raw_output_manifest"][0]["raw_output_b64"] = base64.b64encode(b"changed").decode()
    replay = mod.replay_raw_references(source, tmp_path)
    assert replay["passed"] is False
    assert replay["raw_hash_rows"][0]["output"]["passed"] is False

    raw_file = tmp_path / "response.bin"
    raw_file.write_bytes(b"real")
    source = _source_artifact()
    source["raw_output_manifest"][0].pop("raw_output_b64")
    source["raw_output_manifest"][0]["raw_output_path"] = raw_file.name
    source["raw_output_manifest"][0]["raw_output_sha256"] = mod.sha256_bytes(b"wrong")
    replay = mod.replay_raw_references(source, tmp_path)
    assert replay["referenced_raw_file_count"] == 1
    assert replay["passed"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (("hf_id", "wrong/model", "model_identity"), ("seed", 6900, "seed_identity")),
)
def test_model_and_seed_drift_block(field: str, value: Any, error: str) -> None:
    """SCENARIO-REPORT-6912-IDENTITY-DRIFT rejects model and seed substitutions."""

    source = _source_artifact()
    source["cell_manifest"][0][field] = value
    replay = _replay(source)
    assert error in replay["rows"][0]["replay_result"]["errors"]
    assert replay["passed"] is False


def test_source_group_parser_and_terminal_state_drift_block() -> None:
    """SCENARIO-REPORT-6912-OUTCOME-DRIFT replays group, parser, and terminal state."""

    group_drift = _source_artifact()
    group_drift["cell_manifest"][0]["group_id"] = "changed"
    assert "source_group_identity" in _replay(group_drift)["rows"][0]["replay_result"]["errors"]

    parser_drift = _source_artifact()
    parser_drift["cell_manifest"][0]["parse_rows"][0]["status"] = "malformed"
    assert "parser_outcome" in _replay(parser_drift)["rows"][0]["replay_result"]["errors"]

    terminal_drift = _source_artifact()
    terminal_drift["cell_manifest"][0]["terminal"] = False
    assert "terminal_state" in _replay(terminal_drift)["rows"][0]["replay_result"]["errors"]


def test_timeout_and_truncation_are_replayed_without_filtering() -> None:
    """SCENARIO-REPORT-6912-OUTCOME-DRIFT keeps adverse transport cells."""

    source = _source_artifact()
    timeout = source["cell_manifest"][0]
    timeout.update({"timed_out": True, "stop_reason": "timeout"})
    timeout["raw_http_response_b64"] = base64.b64encode(b"").decode()
    timeout["raw_http_response_sha256"] = mod.sha256_bytes(b"")
    timeout["http_response_byte_count"] = 0
    truncated = source["cell_manifest"][1]
    truncated.update({"truncated": True, "stop_reason": "length"})
    length_http = json.dumps({"choices": [{"finish_reason": "length"}]}).encode()
    truncated["raw_http_response_b64"] = base64.b64encode(length_http).decode()
    truncated["raw_http_response_sha256"] = mod.sha256_bytes(length_http)
    truncated["http_response_byte_count"] = len(length_http)
    replay = _replay(source)
    assert {row["timed_out"] for row in replay["timeout_rows"]} == {True, False}
    assert {row["truncated"] for row in replay["truncation_rows"]} == {True, False}
    assert replay["rows"][0]["replay_result"]["errors"] == []


def test_duration_alias_and_non_monotonic_interval_block() -> None:
    """SCENARIO-REPORT-6912-DURATION-ALIAS and TIMING fail closed."""

    assert mod.duration_independence_check(2.0, 2.0)["passed"] is False
    assert mod.duration_independence_check(2.0, 3.0)["passed"] is True

    source = _source_artifact()
    source["cell_manifest"][0]["wall_time_s"] = -0.1
    timing = mod.derive_source_live_duration(
        source["cell_manifest"], source["server_lifecycle_rows"]
    )
    assert timing["passed"] is False
    assert "negative_elapsed" in timing["timing_rows"][0]["errors"]


def test_aggregate_row_disagreement_blocks() -> None:
    """SCENARIO-REPORT-6912-AGGREGATES keeps reported and recomputed values."""

    source = _source_artifact()
    source["per_arm_family_counts"][producer.RULE_ARM]["graph_coloring"] = 0
    replay = _replay(source)
    row = next(
        row
        for row in replay["reported_vs_recomputed_metrics"]
        if row["metric"] == "per_arm_family_counts"
    )
    assert row["reported"] != row["recomputed"]
    assert row["passed"] is False


def test_flag_erasure_and_source_alias_absence_block() -> None:
    """SCENARIO-REPORT-6912-FLAG preserves quarantine and source alias evidence."""

    source = _source_artifact()
    assert mod.check_source_flag_preservation(source)["passed"] is True
    source["corrigendum_pending"] = []
    assert mod.check_source_flag_preservation(source)["passed"] is False
    source = _source_artifact()
    source["live_duration_s"] = 9.0
    assert mod.check_source_flag_preservation(source)["duration_alias_detected"] is False


def test_ready_artifact_has_required_rows_principles_and_clean_verifier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6912-ADVERSARIAL opens only a clean new receipt."""

    source = _source_artifact()
    artifact = mod.reduce_corpus(
        source=source,
        source_records=[_source_record()],
        root=tmp_path,
        date="20260903",
        duration_s=2.25,
        source_artifact_hashes={"exp6900": {"sha256": "sha256:test"}},
        preconditions_checked={"passed": True, "checks": []},
        required_models=mod.REQUIRED_MODELS,
        required_seeds=(6899,),
        expected_cell_count=5,
        verify_fn=lambda _path: {"loaded": True, "flags": [], "gate_version": "test"},
    )
    assert artifact["clean_relation_corpus_ready_score"] == 1
    assert artifact["duration_s"] == 2.25
    assert artifact["source_live_duration_s"] != artifact["duration_s"]
    assert len([row for row in artifact["rows"] if row["row_type"] == "source_cell_replay"]) == 5
    assert any(row["row_type"] == "reducer_check" for row in artifact["rows"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith("complete_")

    flagged = mod.reduce_corpus(
        source=source,
        source_records=[_source_record()],
        root=tmp_path,
        date="20260903",
        duration_s=2.25,
        source_artifact_hashes={},
        preconditions_checked={"passed": True, "checks": []},
        required_models=mod.REQUIRED_MODELS,
        required_seeds=(6899,),
        expected_cell_count=5,
        verify_fn=lambda _path: {
            "loaded": True,
            "flags": [{"kind": "TEST", "severity": "critical", "detail": "fixture"}],
            "gate_version": "test",
        },
    )
    assert flagged["clean_relation_corpus_ready_score"] == 0
    assert flagged["gate_check_summary"]["failed_check"] == "fresh_adversarial_critical_count"


def test_exact_replay_count_pair_is_not_a_tautology(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6912-ADVERSARIAL permits the structural 1,400/1,400 pair."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "source_cell_count": 1400,
                "replayed_cell_count": 1400,
                "duration_s": 0.25,
                "source_live_duration_s": 623.0,
            }
        ),
        encoding="utf-8",
    )
    from scripts.adversarial_verify import verify_artifact

    kinds = {row["kind"] for row in verify_artifact(candidate)["flags"]}
    assert "TAUTOLOGY" not in kinds


def test_run_writes_ready_or_blocked_artifact_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6912 writes terminal output while source bytes stay immutable."""

    for key in ("exp6899", "producer_module", "producer_entrypoint"):
        path = tmp_path / mod.SOURCE_PATHS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key, encoding="utf-8")
    source = _source_artifact()
    source["source_artifact_hashes"]["exp6899"]["sha256"] = mod.sha256_file(
        tmp_path / mod.SOURCE_PATHS["exp6899"]
    )
    source_path = tmp_path / mod.SOURCE_PATHS["exp6900"]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    expected = {key: mod.sha256_file(tmp_path / path) for key, path in mod.SOURCE_PATHS.items()}
    before = source_path.read_bytes()
    output = tmp_path / "ready.json"
    clock_values = iter((10.0, 12.0))
    artifact = mod.run(
        date="20260903",
        root=tmp_path,
        output_path=output,
        expected_hashes=expected,
        source_records=[_source_record()],
        required_models=mod.REQUIRED_MODELS,
        required_seeds=(6899,),
        expected_cell_count=5,
        verify_fn=lambda _path: {"loaded": True, "flags": [], "gate_version": "test"},
        clock=lambda: next(clock_values),
    )
    assert output.is_file()
    assert artifact["clean_relation_corpus_ready_score"] == 1
    assert source_path.read_bytes() == before

    monkeypatch.setitem(expected, "exp6900", "sha256:wrong")
    blocked = mod.run(
        date="20260903",
        root=tmp_path,
        output_path=tmp_path / "blocked.json",
        expected_hashes=expected,
        source_records=[_source_record()],
        required_models=mod.REQUIRED_MODELS,
        required_seeds=(6899,),
        expected_cell_count=5,
        verify_fn=lambda _path: {"loaded": True, "flags": [], "gate_version": "test"},
        clock=iter((20.0, 21.0)).__next__,
    )
    assert blocked["honest_verdict"] == "complete_blocked_alias_safe_relation_corpus_reducer"
    assert blocked["gate_check_summary"]["failed_check"] == "source_hash:exp6900"


def test_main_parses_date_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-6912 exposes the required command-line entrypoint."""

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(mod, "run", lambda **kwargs: output.write_text(kwargs["date"]))
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0
    assert output.read_text() == "20260903"
