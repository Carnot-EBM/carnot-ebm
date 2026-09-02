"""Tests for the authentic five-arm relation proposal corpus.

Spec refs: REQ-INFERENCE-6887 and SCENARIO-INFERENCE-6887-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6887_three_family_relation_proposal_corpus as mod


ROOT = Path(__file__).resolve().parents[2]


def _upstream() -> dict[str, Any]:
    fixtures = mod.build_frozen_source_records()
    return {
        "relation_fixture_ready_score": 1,
        "relation_schema_version": "anchored_relation_v1",
        "rows": [
            {
                "row_type": "fixture",
                "fixture_id": row["fixture_id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "split": row["split"],
                "source_text_hash": row["source_text_hash"],
                "prompt_view_hash": mod.sha256_json(row),
            }
            for row in fixtures
        ],
        "enoki_asset_receipts": deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
    }


def _models() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "model_path": f"/cache/{index}.gguf",
            "sha256": mod.EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "snapshot_identity": mod.EXPECTED_MODEL_BINDINGS[hf_id]["snapshot_identity"],
            "model_size_bytes": 100 + index,
            "gpu": 0,
        }
        for index, hf_id in enumerate(mod.MODEL_SPECS)
    ]


def _tokenizers() -> list[dict[str, Any]]:
    return [
        {
            "hf_id": hf_id,
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": True,
            "canonical_tokenizer_payload_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
            "probe_token_count": 4,
        }
        for hf_id in mod.MODEL_SPECS
    ]


def _preconditions() -> dict[str, Any]:
    return mod.evaluate_preconditions(
        upstream=_upstream(),
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        enoki_receipts=deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
        cuda_supported=True,
        gpu_offload_supported=True,
        gpu_lease_available=True,
        held_sidecar_access_count=0,
    )


def _runtime(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "authentic": True,
        "server_pid": 7000 if arm.startswith("gguf:") else None,
        "gpu_uuid": "GPU-test" if arm.startswith("gguf:") else None,
        "offload_layers": -1 if arm.startswith("gguf:") else None,
        "owned_cuda_residency": arm.startswith("gguf:"),
    }


def _complete_cells() -> list[dict[str, Any]]:
    rows = []
    for source in mod.build_frozen_source_records():
        for arm in mod.PROPOSAL_ARMS:
            rows.append(
                mod.build_terminal_cell(
                    source=source,
                    arm=arm,
                    raw_output="",
                    prompt=mod.build_prompt(source) if arm.startswith("gguf:") else "",
                    prompt_token_count=12 if arm.startswith("gguf:") else 0,
                    output_token_count=0,
                    latency_s=0.01,
                    stop_reason="stop",
                    timed_out=False,
                    truncated=False,
                    runtime_receipt=_runtime(arm),
                )
            )
    return rows


def _clean_lifecycle() -> list[dict[str, Any]]:
    return [
        {
            "arm": f"gguf:{hf_id}",
            "process_exit_confirmed": True,
            "process_reaped": True,
            "port_release_confirmed": True,
            "lease_released": True,
            "unrelated_process_signal_count": 0,
            "leak_free": True,
        }
        for hf_id in mod.MODEL_SPECS
    ]


def _acquisition() -> dict[str, Any]:
    return {
        "cells": _complete_cells(),
        "llama_cpp_receipts": [
            {
                "hf_id": row["hf_id"],
                "pid": 7000 + index,
                "gpu_uuid": "GPU-test",
                "offload_layers": -1,
                "owned_cuda_residency": True,
            }
            for index, row in enumerate(_models())
        ],
        "gpu_lease_rows": [
            {"hf_id": hf_id, "owned": True, "released": True} for hf_id in mod.MODEL_SPECS
        ],
        "enoki_asset_receipts": deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
        "deterministic_rule_receipts": [{"version": mod.RULE_VERSION, "deterministic": True}],
        "server_lifecycle_rows": _clean_lifecycle(),
        "checkpoint_manifest": {"complete": True},
    }


def test_req_inference_6887_spec_owns_contract() -> None:
    """REQ-INFERENCE-6887 declares every required scenario before code."""

    text = (ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text()
    section = text[text.index("### REQ-INFERENCE-6887") :]
    for token in (
        "SCENARIO-INFERENCE-6887-PRECONDITIONS",
        "SCENARIO-INFERENCE-6887-PROMPT-SEAL",
        "SCENARIO-INFERENCE-6887-PROTOCOL",
        "SCENARIO-INFERENCE-6887-TIMEOUT-AND-TRUNCATION",
        "SCENARIO-INFERENCE-6887-RESUME",
        "SCENARIO-INFERENCE-6887-TEARDOWN",
        "SCENARIO-INFERENCE-6887-COMPLETION",
    ):
        assert token in section
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= {
        word.strip("`,.") for word in section.replace("\n", " ").split()
    }


def test_scenario_6887_preconditions_require_exact_resources() -> None:
    """SCENARIO-INFERENCE-6887-PRECONDITIONS blocks each unsafe substitution."""

    assert _preconditions()["gate_check_summary"]["passed"] is True
    mutations = {
        "missing_cache": lambda values: values["models"].pop(),
        "revision_drift": lambda values: values["models"][0].update(snapshot_identity="drift"),
        "tokenizer_substitution": lambda values: values["tokenizer_receipts"][0].update(
            canonical_tokenizer_payload_sha256="sha256:wrong"
        ),
        "cpu_headline_fallback": lambda values: values.update(cuda_supported=False),
        "gpu_offload_absent": lambda values: values.update(gpu_offload_supported=False),
        "lease_absent": lambda values: values.update(gpu_lease_available=False),
        "held_access": lambda values: values.update(held_sidecar_access_count=1),
        "upstream_drift": lambda values: values.update(upstream_sha256="sha256:drift"),
        "enoki_drift": lambda values: values["enoki_receipts"][0].update(revision="drift"),
    }
    for name, mutate in mutations.items():
        values = {
            "upstream": _upstream(),
            "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
            "enoki_receipts": deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
            "cuda_supported": True,
            "gpu_offload_supported": True,
            "gpu_lease_available": True,
            "held_sidecar_access_count": 0,
        }
        mutate(values)
        report = mod.evaluate_preconditions(**values)
        assert report["gate_check_summary"]["passed"] is False, name
        failed = report["gate_check_summary"]["failed_checks"]
        assert all({"check", "expected", "observed", "passed"} <= set(row) for row in failed)


def test_scenario_6887_resolver_calls_pair_then_dense(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6887-PRECONDITIONS uses exact GGUF paths only."""

    paths = []
    for index in range(3):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"model-{index}".encode())
        paths.append(path)
    calls: list[str] = []

    def pair_provider(**_: Any) -> list[dict[str, Any]]:
        calls.append("pair")
        return [
            {"hf_id": mod.MODEL_SPECS[0], "model_path": str(paths[0]), "gpu": 0},
            {"hf_id": mod.MODEL_SPECS[2], "model_path": str(paths[2]), "gpu": 1},
        ]

    def dense_resolver(hf_id: str, _: str) -> str:
        calls.append(hf_id)
        return str(paths[1])

    rows = mod.resolve_three_models(pair_provider=pair_provider, dense_resolver=dense_resolver)
    assert calls == ["pair", mod.MODEL_SPECS[1]]
    assert [row["hf_id"] for row in rows] == list(mod.MODEL_SPECS)
    assert all(row["model_path"].endswith(".gguf") for row in rows)


def test_scenario_6887_prompt_uses_frozen_public_views_only() -> None:
    """SCENARIO-INFERENCE-6887-PROMPT-SEAL keeps all formal authority hidden."""

    sources = mod.select_source_records(_upstream())
    assert len(sources) == 90
    assert set(mod.FAMILIES) == {row["family"] for row in sources}
    assert all(sum(row["family"] == family for row in sources) == 18 for family in mod.FAMILIES)
    prompt = mod.build_prompt(sources[0])
    assert sources[0]["source_text"] in prompt
    assert mod.PROTOCOL_LINE in prompt
    assert mod.audit_arm_input(prompt) == []
    for forbidden in mod.FORBIDDEN_PROMPT_TOKENS:
        assert forbidden not in prompt
    leaked = prompt + "\nasp_program"
    assert mod.audit_arm_input(leaked) == ["forbidden_prompt_token:asp_program"]


def test_scenario_6887_protocol_preserves_surface_rows() -> None:
    """SCENARIO-INFERENCE-6887-PROTOCOL keeps valid, bad, duplicate, and empty rows."""

    source = mod.build_frozen_source_records()[0]
    text = source["source_text"]
    subject_start = len(text[: text.index("n0")].encode())
    object_start = len(text[: text.index("red")].encode())
    valid = (
        f"REL\t{subject_start}\t{subject_start + 2}\thas_color\t"
        f"{object_start}\t{object_start + 3}\tpositive"
    )
    raw = "\n".join(
        [
            valid,
            valid,
            valid.replace("has_color", "free_relation"),
            "REL\tbad",
            f"REL\t1\t2\thas_color\t{object_start}\t{object_start + 3}\tpositive",
            "ABSTAIN",
        ]
    )
    rows = mod.parse_relation_output(raw, source)
    assert [row["status"] for row in rows] == [
        "accepted",
        "duplicate",
        "unsupported",
        "malformed",
        "invalid_span",
        "abstention",
    ]
    assert rows[0]["subject"]["text"] == "n0"
    assert rows[0]["object"]["text"] == "red"
    assert rows[0]["normalized_tuple"] == ["n0", "has_color", "red", "positive"]
    empty = mod.parse_relation_output("", source)
    assert empty == [{"line_index": 0, "raw_line": "", "status": "empty", "reason": "empty_output"}]


def test_scenario_6887_timeout_and_truncation_are_terminal() -> None:
    """SCENARIO-INFERENCE-6887-TIMEOUT-AND-TRUNCATION preserves raw terminal cells."""

    source = mod.build_frozen_source_records()[0]
    cell = mod.build_terminal_cell(
        source=source,
        arm=f"gguf:{mod.MODEL_SPECS[0]}",
        raw_output="REL\tpartial",
        prompt=mod.build_prompt(source),
        prompt_token_count=40,
        output_token_count=mod.OUTPUT_TOKEN_BUDGET,
        latency_s=10.0,
        stop_reason="length",
        timed_out=True,
        truncated=True,
        runtime_receipt=_runtime(f"gguf:{mod.MODEL_SPECS[0]}"),
    )
    assert cell["timed_out"] is True
    assert cell["truncated"] is True
    assert cell["raw_output"] == "REL\tpartial"
    assert cell["raw_output_sha256"] == mod.sha256_text("REL\tpartial")
    assert cell["terminal"] is True
    assert cell["parse_rows"][0]["status"] == "malformed"


def test_scenario_6887_resume_only_returns_absent_cells(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6887-RESUME preserves completed cells and rejects drift."""

    source = mod.build_frozen_source_records()[0]
    expected = [
        mod.cell_identity(arm, source["fixture_id"]) for arm in mod.PROPOSAL_ARMS[:2]
    ]
    cell = _complete_cells()[0]
    checkpoint = mod.build_checkpoint("sha256:inputs", expected, [cell])
    path = tmp_path / "checkpoint.json"
    mod.write_json_atomic(path, checkpoint)
    loaded = mod.load_checkpoint(path, input_checksum="sha256:inputs")
    assert mod.pending_cell_identities(expected, loaded) == [expected[1]]
    changed = deepcopy(checkpoint)
    changed["input_checksum"] = "sha256:drift"
    mod.write_json_atomic(path, changed)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_input_hash_drift"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")
    changed = deepcopy(checkpoint)
    changed["checkpoint_sha256"] = "sha256:drift"
    mod.write_json_atomic(path, changed)
    with pytest.raises(mod.RelationCorpusError, match="checkpoint_hash_invalid"):
        mod.load_checkpoint(path, input_checksum="sha256:inputs")


def test_scenario_6887_completion_ignores_proposal_quality() -> None:
    """SCENARIO-INFERENCE-6887-COMPLETION gates acquisition, not extraction quality."""

    cells = _complete_cells()
    assert all(cell["parse_rows"][0]["status"] == "empty" for cell in cells)
    assert mod.completion_score(
        sources=mod.build_frozen_source_records(),
        cells=cells,
        tokenizer_receipts=_tokenizers(),
        llama_cpp_receipts=_acquisition()["llama_cpp_receipts"],
        gpu_lease_rows=_acquisition()["gpu_lease_rows"],
        enoki_asset_receipts=deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
        server_lifecycle_rows=_clean_lifecycle(),
        held_sidecar_access_count=0,
    ) == 1
    assert mod.completion_score(
        sources=mod.build_frozen_source_records(),
        cells=cells[:-1],
        tokenizer_receipts=_tokenizers(),
        llama_cpp_receipts=_acquisition()["llama_cpp_receipts"],
        gpu_lease_rows=_acquisition()["gpu_lease_rows"],
        enoki_asset_receipts=deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
        server_lifecycle_rows=_clean_lifecycle(),
        held_sidecar_access_count=0,
    ) == 0


def test_scenario_6887_teardown_rejects_any_lifecycle_gap() -> None:
    """SCENARIO-INFERENCE-6887-TEARDOWN requires reap, port, lease, and narrow signals."""

    clean = _clean_lifecycle()[0]
    assert mod.server_lifecycle_errors(clean) == []
    for field in (
        "process_exit_confirmed",
        "process_reaped",
        "port_release_confirmed",
        "lease_released",
        "leak_free",
    ):
        changed = deepcopy(clean)
        changed[field] = False
        assert field in mod.server_lifecycle_errors(changed)
    changed = deepcopy(clean)
    changed["unrelated_process_signal_count"] = 1
    assert "unrelated_process_signal_count" in mod.server_lifecycle_errors(changed)


def test_req_6887_artifact_replays_rows_and_required_fields() -> None:
    """REQ-INFERENCE-6887 emits a stable complete artifact from raw cells."""

    artifact = mod.build_artifact(
        date="20260902",
        duration_s=61.0,
        upstream=_upstream(),
        upstream_sha256=mod.EXPECTED_EXP6886_SHA256,
        sources=mod.build_frozen_source_records(),
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        preconditions=_preconditions(),
        acquisition=_acquisition(),
    )
    assert artifact["relation_corpus_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["raw_output_manifest"]) == 450
    assert artifact["held_sidecar_access_count"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert mod.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["rows"][0]["raw_output"] = "changed"
    assert "raw_output_hash" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["relation_corpus_complete_score"] = 0
    assert "relation_corpus_complete_score" in mod.validate_artifact(changed)


def test_req_6887_blocked_run_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-INFERENCE-6887 writes a full blocked artifact when cache gates fail."""

    result_path = tmp_path / "results" / "experiment_6887.json"

    def blocked(_: Path) -> dict[str, Any]:
        values = _preconditions()
        values["gate_check_summary"] = mod.gate_summary(
            [mod.gate_check("exact_model_caches", True, False)]
        )
        values.update(
            {
                "upstream": _upstream(),
                "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
                "models": _models()[:-1],
                "tokenizer_receipts": _tokenizers(),
                "enoki_receipts": deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
            }
        )
        return values

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        precondition_collector=blocked,
    )
    assert artifact["honest_verdict"] == "complete_blocked_three_family_relation_corpus"
    assert artifact["relation_corpus_complete_score"] == 0
    assert result_path.is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(json.loads(result_path.read_text()))


def test_req_6887_injected_run_writes_only_requested_paths(tmp_path: Path) -> None:
    """REQ-INFERENCE-6887 keeps tests away from the tracked research record."""

    result_path = tmp_path / "results" / "experiment_6887.json"
    checkpoint_path = tmp_path / "checkpoint.json"

    def ready(_: Path) -> dict[str, Any]:
        return {
            **_preconditions(),
            "upstream": _upstream(),
            "upstream_sha256": mod.EXPECTED_EXP6886_SHA256,
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
            "enoki_receipts": deepcopy(mod.EXPECTED_ENOKI_RECEIPTS),
        }

    artifact = mod.run(
        date="20260902",
        root=tmp_path,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        precondition_collector=ready,
        acquisition_runner=lambda **_: _acquisition(),
    )
    assert artifact["relation_corpus_complete_score"] == 1
    assert result_path.is_file()
    assert not (ROOT / "experiment_6887_three_family_relation_proposal_corpus.py").exists()


def test_req_6887_source_forbids_gguf_transformers_and_schema_decode() -> None:
    """REQ-INFERENCE-6887 keeps GGUF tokenization native and decoding unconstrained."""

    source = (ROOT / "python/carnot/experiment_6887_three_family_relation_proposal_corpus.py").read_text()
    assert "cached_sota_pair(" in source
    assert "resolve_cached_gguf(" in source
    assert "AutoTokenizer.from_pretrained(hf_id" not in source
    assert "grammar=" not in source
    assert "response_format" not in source
    assert "json_schema" not in source
    wrapper = ROOT / "scripts/experiments/experiment_6887_three_family_relation_proposal_corpus.py"
    assert wrapper.is_file()
