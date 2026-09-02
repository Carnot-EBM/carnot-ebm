"""Tests for the authentic anchored relation acquisition corpus.

Spec refs: REQ-INFERENCE-6900 and SCENARIO-INFERENCE-6900-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6900_authentic_anchored_relation_corpus as mod


ROOT = Path(__file__).resolve().parents[2]
NOW_NS = 9_000_000_000_000


def _upstream() -> dict[str, Any]:
    sources = mod.reconstruct_source_records()
    return {
        "relation_fixture_ready_score": 1,
        "relation_schema_version": "anchored_relation_v1",
        "calibration_group_manifest": {
            "public_fixture_manifest_hash": mod.canary.EXPECTED_FIXTURE_HASHES["calibration"]
        },
        "sealed_held_group_manifest": {
            "public_fixture_manifest_hash": mod.canary.EXPECTED_FIXTURE_HASHES["held"]
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
            for row in sources
        ],
        "enoki_asset_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
    }


def _canary() -> dict[str, Any]:
    return json.loads(
        (ROOT / "results/experiment_6899_live_relation_acquisition_canary.json").read_text()
    )


def _models() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "model_path": (
                f"/cache/snapshots/{binding['snapshot_identity']}/{binding['filename']}"
            ),
            "sha256": binding["sha256"],
            "snapshot_identity": binding["snapshot_identity"],
            "model_size_bytes": binding["size_bytes"],
            "gpu": 0,
        }
        for hf_id, binding in mod.EXPECTED_MODEL_BINDINGS.items()
    ]


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
        {"hf_id": hf_id, "owned": True, "released": True, "gpu_uuid": "GPU-test"}
        for hf_id in mod.MODEL_SPECS
    ]


def _precondition_args() -> dict[str, Any]:
    return {
        "canary_artifact": _canary(),
        "canary_sha256": mod.EXPECTED_EXP6899_SHA256,
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
        "protocol_hashes": mod.current_protocol_hashes(),
    }


def _preconditions() -> dict[str, Any]:
    report = mod.evaluate_preconditions(**_precondition_args())
    assert report["gate_check_summary"]["passed"] is True
    return report


def _runtime(hf_id: str, model_index: int) -> dict[str, Any]:
    return {
        "authentic_attempt": True,
        "model_sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
        "tokenizer_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
        "server_pid": 300 + model_index,
        "server_start_time_ticks": 2_000 + model_index,
        "command_hash": f"sha256:command-{model_index}",
        "process_identity_match": True,
        "receipt_age_s": 0.1,
        "gpu_uuid": f"GPU-{model_index}",
        "offload_layers": 40,
        "owned_cuda_residency": True,
        "vram_before_mb": 24_500,
        "vram_after_load_mb": 18_000,
        "stderr_tail": "offloaded 40 layers",
    }


def _gguf_cell(
    source: dict[str, Any],
    hf_id: str,
    seed: int,
    model_index: int,
    *,
    raw_output: bytes | None = None,
    timed_out: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    output = (
        mod.base._rule_output(source).encode("utf-8") if raw_output is None else raw_output
    )
    parse_rows = mod.parse_relation_output(output.decode("utf-8", errors="replace"), source)
    return mod.build_gguf_cell(
        hf_id=hf_id,
        source=source,
        seed=seed,
        raw_request_bytes=mod.build_request_bytes(source, seed),
        raw_http_response_bytes=b'{"choices":[]}',
        raw_output_bytes=output,
        native_prompt_tokens=100,
        generated_tokens=0 if not output else 20,
        stop_reason="timeout" if timed_out else ("length" if truncated else "stop"),
        wall_time_s=1.0,
        timed_out=timed_out,
        truncated=truncated,
        parser_attempted=True,
        parse_rows=parse_rows,
        runtime_receipt=_runtime(hf_id, model_index),
    )


def _control_cell(source: dict[str, Any], arm: str) -> dict[str, Any]:
    raw_output = mod.base._rule_output(source).encode("utf-8")
    receipt = (
        {
            "authentic_attempt": True,
            "encoder_revision": mod.ENOKI_REVISION,
            "encoder_asset_hash": mod.ENOKI_ASSET_HASH,
            "compatibility_adapter_sha256": mod.ENOKI_TRANSFORMERS5_ADAPTER_SHA256,
            "lease_released": True,
        }
        if arm == mod.ENOKI_ARM
        else {
            "authentic_attempt": True,
            "rule_version": mod.RULE_VERSION,
            "rule_source_sha256": mod.RULE_SOURCE_SHA256,
        }
    )
    return mod.build_control_cell(
        arm=arm,
        source=source,
        raw_request_bytes=source["source_text"].encode("utf-8"),
        raw_output_bytes=raw_output,
        stop_reason="encoder_complete" if arm == mod.ENOKI_ARM else "rule_complete",
        wall_time_s=0.1,
        parse_rows=mod.parse_relation_output(raw_output.decode(), source),
        runtime_receipt=receipt,
    )


def _all_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for source in mod.reconstruct_source_records():
        for model_index, hf_id in enumerate(mod.MODEL_SPECS):
            for seed in mod.SEEDS:
                cells.append(_gguf_cell(source, hf_id, seed, model_index))
        cells.append(_control_cell(source, mod.ENOKI_ARM))
        cells.append(_control_cell(source, mod.RULE_ARM))
    return cells


def _llama_receipts() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "pid": 300 + index,
            "start_time_ticks": 2_000 + index,
            "command_hash": f"sha256:command-{index}",
            "model_sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "tokenizer_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
            "gpu_uuid": f"GPU-{index}",
            "offload_layers": 40,
            "owned_cuda_residency": True,
            "stderr_tail": "offloaded 40 layers",
            "phase_error": "",
        }
        for index, hf_id in enumerate(mod.MODEL_SPECS)
    ]


def _lifecycle_rows() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "pid": 300 + index,
            "start_time_ticks": 2_000 + index,
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


def _acquisition(*, cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "cells": _all_cells() if cells is None else cells,
        "llama_cpp_receipts": _llama_receipts(),
        "gpu_lease_rows": _lease_rows(),
        "server_lifecycle_rows": _lifecycle_rows(),
        "enoki_asset_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
        "deterministic_rule_receipts": [
            {
                "rule_version": mod.RULE_VERSION,
                "rule_source_sha256": mod.RULE_SOURCE_SHA256,
                "deterministic": True,
            }
        ],
        "live_duration_s": 61.0,
        "checkpoint_manifest": {"complete": True},
    }


def test_req_inference_6900_spec_owns_contract() -> None:
    """REQ-INFERENCE-6900 declares each required failure boundary."""

    text = (ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text()
    section = text[text.index("### REQ-INFERENCE-6900") :]
    for suffix in (
        "PRECONDITIONS",
        "PROTOCOL-DRIFT",
        "BALANCE",
        "RAW-ROWS",
    ):
        assert f"SCENARIO-INFERENCE-6900-{suffix}" in section


def test_scenario_6900_balance_freezes_one_hundred_public_sources() -> None:
    """SCENARIO-INFERENCE-6900-BALANCE uses 20 source views per family."""

    sources = mod.select_source_records(_upstream())
    assert len(sources) == 100
    assert len({row["fixture_id"] for row in sources}) == 100
    assert mod.sha256_json(sources) == mod.EXPECTED_SOURCE_HASH
    counts = mod.source_family_counts(sources)
    assert counts == {family: 20 for family in mod.FAMILIES}
    matrix = mod.expected_cell_identities(sources)
    assert len(matrix) == 100 * (len(mod.MODEL_SPECS) * len(mod.SEEDS) + 2)


def test_scenario_6900_source_and_resolver_guards_reject_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFERENCE-6900-PRECONDITIONS covers source and cache guards."""

    upstream = _upstream()
    upstream["rows"].pop()
    with pytest.raises(mod.RelationCorpusError, match="upstream_public_fixture_missing"):
        mod.select_source_records(upstream)
    upstream = _upstream()
    upstream["rows"][0]["source_text_hash"] = "sha256:drift"
    with pytest.raises(mod.RelationCorpusError, match="upstream_public_fixture_drift"):
        mod.select_source_records(upstream)
    monkeypatch.setattr(mod, "EXPECTED_SOURCE_HASH", "sha256:drift")
    with pytest.raises(mod.RelationCorpusError, match="frozen_source_hash_drift"):
        mod.select_source_records(_upstream())
    monkeypatch.undo()

    paths = [tmp_path / binding["filename"] for binding in mod.EXPECTED_MODEL_BINDINGS.values()]
    pair = [
        {"hf_id": mod.MODEL_SPECS[0], "model_path": str(paths[0])},
        {"hf_id": mod.MODEL_SPECS[2], "model_path": str(paths[2])},
    ]
    models = mod.resolve_three_models(
        pair_provider=lambda **_: pair,
        dense_resolver=lambda *_: str(paths[1]),
    )
    assert [row["hf_id"] for row in models] == list(mod.MODEL_SPECS)
    assert all(Path(row["model_path"]).is_absolute() for row in models)
    absent = mod.resolve_three_models(pair_provider=lambda **_: [], dense_resolver=lambda *_: "")
    assert all("model_path" not in row for row in absent)


def test_scenario_6900_protocol_is_the_exact_canary_protocol() -> None:
    """SCENARIO-INFERENCE-6900-PROTOCOL-DRIFT reuses canary code and settings."""

    source = mod.reconstruct_source_records()[0]
    assert mod.build_prompt(source) == mod.canary.build_prompt(source)
    assert mod.build_request_bytes(source, mod.SEEDS[0]) == mod.canary.build_request_bytes(
        source, mod.SEEDS[0]
    )
    assert mod.parse_relation_output is mod.base.parse_relation_output
    assert mod.current_protocol_hashes() == mod.EXPECTED_PROTOCOL_HASHES


def test_scenario_6900_preconditions_reject_gate_and_substrate_drift() -> None:
    """SCENARIO-INFERENCE-6900-PRECONDITIONS rejects each frozen-input drift."""

    assert mod.evaluate_preconditions(**_precondition_args())["passed"] is True
    mutations = {
        "canary_gate": lambda value: value["canary_artifact"].update(
            relation_canary_ready_score=0
        ),
        "canary_hash": lambda value: value.update(canary_sha256="sha256:drift"),
        "prompt_parser_drift": lambda value: value["protocol_hashes"].update(
            parser="sha256:drift"
        ),
        "model_drift": lambda value: value["models"][0].update(sha256="sha256:drift"),
        "tokenizer_drift": lambda value: value["tokenizer_receipts"][0].update(
            canonical_tokenizer_payload_sha256="sha256:drift"
        ),
        "held_access": lambda value: value.update(held_sidecar_access_count=1),
    }
    for name, mutate in mutations.items():
        values = deepcopy(_precondition_args())
        mutate(values)
        report = mod.evaluate_preconditions(**values)
        assert report["passed"] is False, name
        assert report["gate_check_summary"]["failed_checks"]
        assert all(
            {"check", "expected", "observed", "passed"} <= set(row)
            for row in report["gate_check_summary"]["failed_checks"]
        )
    invalid_source = _precondition_args()
    invalid_source["upstream"]["rows"] = []
    report = mod.evaluate_preconditions(**invalid_source)
    balance = next(
        row for row in report["gate_check_summary"]["checks"] if row["check"] == "balanced_source_hash"
    )
    assert balance["observed"] == "invalid"


def test_scenario_6900_raw_rows_preserve_all_parser_outcomes() -> None:
    """SCENARIO-INFERENCE-6900-RAW-ROWS retains invalid lines and failures."""

    source = mod.reconstruct_source_records()[0]
    valid = mod.base._rule_output(source)
    raw = "\n".join(
        [
            valid,
            valid,
            "BAD",
            "REL\t0\t4\tnot_allowed\t5\t9\tpositive",
            "ABSTAIN",
        ]
    ).encode()
    cell = _gguf_cell(source, mod.MODEL_SPECS[0], mod.SEEDS[0], 0, raw_output=raw)
    statuses = [row["status"] for row in mod.flatten_terminal_rows([cell])]
    assert statuses == ["accepted", "duplicate", "malformed", "unsupported", "abstention"]
    empty = _gguf_cell(source, mod.MODEL_SPECS[0], mod.SEEDS[0], 0, raw_output=b"")
    empty_rows = mod.flatten_terminal_rows([empty])
    assert empty_rows[0]["status"] == "empty"
    assert empty["raw_output_b64"] == ""
    assert empty["raw_output_sha256"] == mod.sha256_bytes(b"")


def test_scenario_6900_timeout_and_truncation_do_not_mask_raw_bytes() -> None:
    """SCENARIO-INFERENCE-6900-RAW-ROWS keeps timed and truncated cells terminal."""

    source = mod.reconstruct_source_records()[0]
    timeout = _gguf_cell(
        source,
        mod.MODEL_SPECS[0],
        mod.SEEDS[0],
        0,
        raw_output=b"REL\tpartial",
        timed_out=True,
    )
    truncated = _gguf_cell(
        source,
        mod.MODEL_SPECS[0],
        mod.SEEDS[1],
        0,
        raw_output=b"REL\tpartial",
        truncated=True,
    )
    statuses = [row["status"] for row in mod.flatten_terminal_rows([timeout, truncated])]
    assert "timeout" in statuses
    assert "truncated" in statuses
    assert timeout["raw_output_b64"] != ""
    assert truncated["raw_output_b64"] != ""


def test_req_6900_completion_is_independent_of_parser_quality() -> None:
    """REQ-INFERENCE-6900 scores authentic acquisition, not semantic quality."""

    sources = mod.reconstruct_source_records()
    cells = _all_cells()
    for row in cells:
        row.update(
            raw_output_b64="",
            raw_output_sha256=mod.sha256_bytes(b""),
            output_byte_count=0,
            generated_tokens=0,
            parse_rows=mod.parse_relation_output("", sources[0]),
            parser_input_sha256=mod.sha256_bytes(b""),
        )
    score, summary = mod.completion_score(
        duration_s=61.0,
        sources=sources,
        acquisition=_acquisition(cells=cells),
        tokenizer_receipts=_tokenizers(),
        held_sidecar_access_count=0,
    )
    assert score == 1
    assert summary["passed"] is True


def test_req_6900_completion_rejects_duplicates_stale_pid_and_server_crash() -> None:
    """REQ-INFERENCE-6900 fails closed on identity and lifecycle defects."""

    values = {
        "duration_s": 61.0,
        "sources": mod.reconstruct_source_records(),
        "acquisition": _acquisition(),
        "tokenizer_receipts": _tokenizers(),
        "held_sidecar_access_count": 0,
    }
    score, _ = mod.completion_score(**values)
    assert score == 1
    mutations = {
        "duplicate": lambda value: value["acquisition"]["cells"].append(
            deepcopy(value["acquisition"]["cells"][0])
        ),
        "stale_pid": lambda value: value["acquisition"]["cells"][0][
            "runtime_receipt"
        ].update(receipt_age_s=mod.RECEIPT_MAX_AGE_S + 1),
        "server_crash": lambda value: value["acquisition"]["server_lifecycle_rows"][
            0
        ].update(process_exit_confirmed=False, leak_free=False),
        "sidecar": lambda value: value.update(held_sidecar_access_count=1),
    }
    for name, mutate in mutations.items():
        changed = deepcopy(values)
        mutate(changed)
        score, summary = mod.completion_score(**changed)
        assert score == 0, name
        assert summary["passed"] is False


def test_scenario_6900_server_failure_emits_only_absent_cells() -> None:
    """SCENARIO-INFERENCE-6900-RAW-ROWS never uses a control as crash fallback."""

    sources = mod.reconstruct_source_records()[:2]
    expected = {
        mod.gguf_cell_identity(mod.MODEL_SPECS[0], seed, source["fixture_id"])
        for source in sources
        for seed in mod.SEEDS
    }
    completed = _gguf_cell(sources[0], mod.MODEL_SPECS[0], mod.SEEDS[0], 0)
    failures = mod.build_absent_failure_cells(
        hf_id=mod.MODEL_SPECS[0],
        sources=sources,
        pending_identities=expected,
        completed_cells=[completed],
        stop_reason="server_crash",
        runtime_receipt={"authentic_attempt": False, "error": "worker exited"},
    )
    assert {completed["cell_identity"]} | {row["cell_identity"] for row in failures} == expected
    assert all(row["raw_output_b64"] == "" for row in failures)
    assert all(row["stop_reason"] == "server_crash" for row in failures)
    assert "server_crash" in {
        row["status"] for row in mod.flatten_terminal_rows(failures[:1])
    }


def test_req_6900_cell_integrity_names_every_provenance_gap() -> None:
    """REQ-INFERENCE-6900 replays raw bytes and each arm's own provenance."""

    source = mod.reconstruct_source_records()[0]
    cell = _gguf_cell(source, mod.MODEL_SPECS[0], mod.SEEDS[0], 0)
    authentic_only = deepcopy(cell)
    authentic_only["runtime_receipt"].pop("authentic_attempt")
    authentic_only["runtime_receipt"]["authentic"] = True
    rebuilt = mod.build_gguf_cell(
        hf_id=mod.MODEL_SPECS[0],
        source=source,
        seed=mod.SEEDS[0],
        raw_request_bytes=mod.build_request_bytes(source, mod.SEEDS[0]),
        raw_http_response_bytes=b"{}",
        raw_output_bytes=b"",
        native_prompt_tokens=1,
        generated_tokens=0,
        stop_reason="stop",
        wall_time_s=0.1,
        timed_out=False,
        truncated=False,
        parser_attempted=True,
        parse_rows=mod.parse_relation_output("", source),
        runtime_receipt=authentic_only["runtime_receipt"],
    )
    assert rebuilt["runtime_receipt"]["authentic_attempt"] is True
    assert mod._unb64("not base64!") is None

    mutations = {
        "request_bytes": lambda row: row.update(raw_request_b64="not base64!"),
        "request_hash": lambda row: row.update(raw_request_sha256="sha256:wrong"),
        "http_response_bytes": lambda row: row.update(http_response_byte_count=999),
        "http_response_hash": lambda row: row.update(raw_http_response_sha256="sha256:wrong"),
        "output_bytes": lambda row: row.update(output_byte_count=999),
        "output_hash": lambda row: row.update(raw_output_sha256="sha256:wrong"),
        "not_terminal": lambda row: row.update(terminal=False),
        "parser_bypass": lambda row: row.update(parser_attempted=False),
        "parser_input_hash": lambda row: row.update(parser_input_sha256="sha256:wrong"),
        "authentic_attempt": lambda row: row["runtime_receipt"].update(
            authentic_attempt=False
        ),
        "model_sha256": lambda row: row["runtime_receipt"].update(model_sha256="wrong"),
        "tokenizer_sha256": lambda row: row["runtime_receipt"].update(
            tokenizer_sha256="wrong"
        ),
        "server_pid": lambda row: row["runtime_receipt"].update(server_pid=0),
        "server_start_time_ticks": lambda row: row["runtime_receipt"].update(
            server_start_time_ticks=0
        ),
        "pid_reuse": lambda row: row["runtime_receipt"].update(
            process_identity_match=False
        ),
        "stale_pid_receipt": lambda row: row["runtime_receipt"].update(
            receipt_age_s=mod.RECEIPT_MAX_AGE_S + 1
        ),
        "offload": lambda row: row["runtime_receipt"].update(offload_layers=0),
        "cuda": lambda row: row["runtime_receipt"].update(owned_cuda_residency=False),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(cell)
        mutate(changed)
        assert expected in mod._cell_integrity_errors(changed)

    enoki = _control_cell(source, mod.ENOKI_ARM)
    for expected, mutation in (
        ("enoki_revision", {"encoder_revision": "wrong"}),
        ("enoki_asset_hash", {"encoder_asset_hash": "wrong"}),
        ("enoki_adapter", {"compatibility_adapter_sha256": "wrong"}),
        ("enoki_lease", {"lease_released": False}),
    ):
        changed = deepcopy(enoki)
        changed["runtime_receipt"].update(mutation)
        assert expected in mod._cell_integrity_errors(changed)
    rule = _control_cell(source, mod.RULE_ARM)
    for expected, mutation in (
        ("rule_version", {"rule_version": "wrong"}),
        ("rule_source_sha256", {"rule_source_sha256": "wrong"}),
    ):
        changed = deepcopy(rule)
        changed["runtime_receipt"].update(mutation)
        assert expected in mod._cell_integrity_errors(changed)
    unknown = deepcopy(rule)
    unknown["arm"] = "unknown"
    assert "unknown_arm" in mod._cell_integrity_errors(unknown)
    assert mod.ENOKI_TRANSFORMERS5_ADAPTER_SHA256.startswith("sha256:")


def test_scenario_6900_partial_resume_preserves_rows_and_rejects_drift(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6900 resumes only absent identities from a bound checkpoint."""

    sources = mod.reconstruct_source_records()[:1]
    expected = sorted(mod.expected_cell_identities(sources))
    completed = [_gguf_cell(sources[0], mod.MODEL_SPECS[0], mod.SEEDS[0], 0)]
    checkpoint = mod.build_checkpoint("sha256:inputs", expected, completed)
    path = tmp_path / "checkpoint.json"
    mod.write_json_atomic(path, checkpoint)
    loaded = mod.load_checkpoint(path, input_checksum="sha256:inputs")
    assert loaded["cells"] == completed
    assert mod.pending_cell_identities(expected, loaded) == [
        identity for identity in expected if identity != completed[0]["cell_identity"]
    ]
    changed = deepcopy(checkpoint)
    changed["input_checksum"] = "sha256:drift"
    mod.write_json_atomic(path, changed)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_input_hash_drift"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")
    duplicate = mod.build_checkpoint("sha256:inputs", expected, completed + completed)
    mod.write_json_atomic(path, duplicate)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_duplicate_cell"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")
    invalid_hash = deepcopy(checkpoint)
    invalid_hash["checkpoint_sha256"] = "sha256:drift"
    mod.write_json_atomic(path, invalid_hash)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_hash_invalid"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")
    unexpected = mod.build_checkpoint(
        "sha256:inputs",
        expected,
        [{**completed[0], "cell_identity": "unexpected"}],
    )
    mod.write_json_atomic(path, unexpected)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_unexpected_cell"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")


def test_req_6900_artifact_has_required_rows_manifests_and_principles() -> None:
    """REQ-INFERENCE-6900 emits the complete replayable artifact schema."""

    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        canary_sha256=mod.EXPECTED_EXP6899_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        sources=mod.reconstruct_source_records(),
        acquisition=_acquisition(),
    )
    assert artifact["relation_canary_ready_score"] == 1
    assert artifact["relation_corpus_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["raw_request_manifest"]) == len(artifact["raw_output_manifest"])
    assert artifact["per_arm_family_counts"] == {
        arm: {family: 20 for family in mod.FAMILIES} for arm in mod.PROPOSAL_ARMS
    }
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert "relation_canary_ready_score" in artifact["field_principles"]
    assert mod.validate_artifact(artifact) == []


def test_req_6900_retry_preserves_prior_failed_attempt_bytes() -> None:
    """REQ-INFERENCE-6900 keeps failed attempts when a repaired control reruns."""

    source = mod.reconstruct_source_records()[0]
    prior = _control_cell(source, mod.ENOKI_ARM)
    prior.update(
        raw_output_b64="",
        raw_output_sha256=mod.sha256_bytes(b""),
        output_byte_count=0,
        raw_http_response_b64="",
        raw_http_response_sha256=mod.sha256_bytes(b""),
        http_response_byte_count=0,
        stop_reason="encoder_failure",
        parse_rows=mod.parse_relation_output("", source),
        parser_input_sha256=mod.sha256_bytes(b""),
    )
    prior["runtime_receipt"].update(
        authentic_attempt=False,
        error="AttributeError: removed Transformers method",
    )
    acquisition = _acquisition()
    acquisition["prior_attempt_cells"] = [prior]
    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        canary_sha256=mod.EXPECTED_EXP6899_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        sources=mod.reconstruct_source_records(),
        acquisition=acquisition,
    )
    assert artifact["prior_attempt_cells"] == [prior]
    assert any(row["status"] == "encoder_failure" for row in artifact["rows"])
    assert len(artifact["raw_output_manifest"]) == len(artifact["cell_manifest"]) + 1
    assert mod.validate_artifact(artifact) == []
    corrupt = deepcopy(artifact)
    corrupt["prior_attempt_cells"][0]["raw_output_b64"] = "not base64!"
    assert "prior_attempt_integrity" in mod.validate_artifact(corrupt)


def test_req_6900_validator_names_each_summary_drift() -> None:
    """REQ-INFERENCE-6900 rejects corrupt derived fields and terminal metadata."""

    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        canary_sha256=mod.EXPECTED_EXP6899_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        sources=mod.reconstruct_source_records(),
        acquisition=_acquisition(),
    )
    mutations = {
        "required_fields": lambda row: row.pop("model_specs"),
        "field_principles": lambda row: row.update(field_principles={}),
        "rows": lambda row: row["rows"].pop(),
        "row_duplication": lambda row: row["rows"].append(deepcopy(row["rows"][0])),
        "raw_request_manifest": lambda row: row["raw_request_manifest"].pop(),
        "raw_output_manifest": lambda row: row["raw_output_manifest"].pop(),
        "per_arm_family_counts": lambda row: row.update(per_arm_family_counts={}),
        "parse_failure_rows": lambda row: row["parse_failure_rows"].append({}),
        "abstention_rows": lambda row: row["abstention_rows"].append({}),
        "empty_rows": lambda row: row["empty_rows"].append({}),
        "truncation_rows": lambda row: row["truncation_rows"].append({}),
        "timeout_rows": lambda row: row["timeout_rows"].append({}),
        "honest_verdict": lambda row: row.update(honest_verdict="partial"),
        "verdict_class": lambda row: row.update(verdict_class="wrong"),
        "verifier_is_oracle": lambda row: row.update(verifier_is_oracle=True),
        "held_sidecar_access_count": lambda row: row.update(held_sidecar_access_count=1),
        "relation_corpus_complete_score": lambda row: row.update(
            relation_corpus_complete_score=0
        ),
        "reproducibility_checksum": lambda row: row.update(
            reproducibility_checksum="sha256:wrong"
        ),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(artifact)
        mutate(changed)
        assert any(error.startswith(expected) for error in mod.validate_artifact(changed))


def test_req_6900_blocked_run_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-INFERENCE-6900 writes all fields before blocked acquisition exits."""

    result = tmp_path / "result.json"

    def blocked(_: Path) -> dict[str, Any]:
        values = _preconditions()
        values["gate_check_summary"] = mod.gate_summary(
            [mod.gate_check("relation_canary_ready_score", 1, 0)]
        )
        values.update(
            {
                "canary_sha256": mod.EXPECTED_EXP6899_SHA256,
                "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
                "models": _models(),
                "tokenizer_receipts": _tokenizers(),
                "enoki_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
            }
        )
        return values

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result,
        checkpoint_path=tmp_path / "checkpoint.json",
        precondition_collector=blocked,
    )
    assert artifact["honest_verdict"] == "complete_blocked_authentic_anchored_relation_corpus"
    assert artifact["relation_corpus_complete_score"] == 0
    assert result.is_file()
    assert mod.validate_artifact(artifact) == []


def test_req_6900_injected_success_run_and_invalid_artifact_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFERENCE-6900 writes injected runs only after artifact replay passes."""

    result = tmp_path / "result.json"

    def ready(_: Path) -> dict[str, Any]:
        return {
            **_preconditions(),
            "canary_sha256": mod.EXPECTED_EXP6899_SHA256,
            "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
            "upstream": _upstream(),
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
            "enoki_receipts": deepcopy(mod.base.EXPECTED_ENOKI_RECEIPTS),
            "eligible_gpus": _gpu_inventory(),
        }

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result,
        checkpoint_path=tmp_path / "checkpoint.json",
        precondition_collector=ready,
        acquisition_runner=lambda **_: _acquisition(),
    )
    assert artifact["relation_corpus_complete_score"] == 1
    assert result.is_file()

    monkeypatch.setattr(mod, "validate_artifact", lambda _: ["forced_invalid"])
    with pytest.raises(mod.RelationCorpusError, match="artifact_invalid:forced_invalid"):
        mod.run(
            date="20260902",
            root=tmp_path,
            result_path=tmp_path / "invalid.json",
            checkpoint_path=tmp_path / "checkpoint-invalid.json",
            precondition_collector=ready,
            acquisition_runner=lambda **_: _acquisition(),
        )


def test_req_6900_source_forbids_held_or_constrained_acquisition() -> None:
    """REQ-INFERENCE-6900 keeps formal authority and constrained decode absent."""

    source = (
        ROOT / "python/carnot/experiment_6900_authentic_anchored_relation_corpus.py"
    ).read_text()
    assert "cached_sota_pair(" in source
    assert "resolve_cached_gguf(" in source
    assert "AutoTokenizer.from_pretrained" not in source
    for forbidden in ("grammar=", "response_format", "json_schema", "finite_answer_id"):
        assert forbidden not in source
    wrapper = ROOT / "scripts/experiments/experiment_6900_authentic_anchored_relation_corpus.py"
    assert wrapper.is_file()
