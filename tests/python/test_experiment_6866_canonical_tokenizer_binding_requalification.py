"""REQ-INFERENCE-6866 canonical tokenizer binding requalification tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6866_canonical_tokenizer_binding_requalification as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _semantic_evidence(
    *,
    pieces: tuple[str, ...] = ("a", "b", "c"),
    bos_token_id: int | None = 2,
    add_bos: bool = True,
    chat_template: str = "<start>{content}<end>",
) -> dict[str, object]:
    return {
        "tokenizer_model": "test-bpe",
        "tokenizer_pretokenizer": "test-pre",
        "vocabulary_size": len(pieces),
        "token_pieces_sha256": exp.sha256_sequence(pieces),
        "token_scores_sha256": exp.sha256_sequence((0.0, -1.0, -2.0)),
        "token_types_sha256": exp.sha256_sequence((1, 1, 3)),
        "merges_sha256": exp.sha256_sequence(("a b", "b c")),
        "special_token_ids": {
            "bos_token_id": bos_token_id,
            "eos_token_id": 3,
            "unknown_token_id": 0,
            "padding_token_id": 1,
            "mask_token_id": None,
            "separator_token_id": None,
        },
        "add_bos_metadata_default": add_bos,
        "chat_template": chat_template,
    }


def _probe_outputs(token_id: int = 7) -> list[dict[str, object]]:
    return [
        {
            "probe_id": "empty_text",
            "text_utf8_sha256": exp.sha256_bytes(b""),
            "settings": {
                "setting_id": "plain",
                "add_bos": False,
                "special": False,
                "utf8_error_mode": "replace",
                "unicode_normalization": "none",
            },
            "token_ids": [token_id],
        }
    ]


def _reduced(
    evidence: dict[str, object] | None = None,
    *,
    probe_outputs: list[dict[str, object]] | None = None,
    source_receipt: dict[str, object] | None = None,
    source_kind: str = "archived_exp6850",
) -> dict[str, object]:
    return exp.reduce_tokenizer_receipt(
        evidence or _semantic_evidence(),
        probe_outputs or _probe_outputs(),
        source_receipt=source_receipt
        or {
            "hf_id": "example/model",
            "detail": "wrapper prose",
            "probe_token_ids": [7],
            "tokenizer_sha256": "sha256:legacy",
        },
        source_kind=source_kind,
    )


def _write_fixture_sources(root: Path) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    model_bytes: dict[str, bytes] = {}
    frozen_specs: list[dict[str, object]] = []
    frozen_hashes: dict[str, dict[str, object]] = {}
    tokenizer_receipts: list[dict[str, object]] = []
    for index, hf_id in enumerate(exp.MODEL_SPECS):
        path = root / "cache" / hf_id.split("/", 1)[1] / "snapshots" / f"snap-{index}"
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"model-{index}-Q4_K_M.gguf"
        content = f"model-{index}".encode()
        model_path.write_bytes(content)
        model_bytes[hf_id] = content
        digest = exp.sha256_bytes(content)
        row = {
            "hf_id": hf_id,
            "model_path": str(model_path),
            "model_sha256": digest,
            "model_size_bytes": len(content),
            "quantization": "Q4_K_M",
        }
        frozen_specs.append(row)
        frozen_hashes[hf_id] = {
            "path": str(model_path),
            "sha256": digest,
            "size_bytes": len(content),
        }
        tokenizer_receipts.append(
            {
                "hf_id": hf_id,
                "source": "embedded_gguf_llama_cpp_vocab_only",
                "metadata_present": True,
                "loadable": True,
                "metadata_key_count": 7,
                "probe_token_ids": [7],
                "detail": "old wrapper prose",
                "tokenizer_sha256": f"sha256:legacy-{index}",
            }
        )

    payloads = {
        "exp6850": {
            "model_specs": frozen_specs,
            "model_artifact_hashes": frozen_hashes,
            "tokenizer_receipts": tokenizer_receipts,
        },
        "exp6863": {
            "tokenizer_receipts": [
                {
                    "hf_id": hf_id,
                    "metadata_present": True,
                    "loadable": True,
                    "special_tokens": {"tokenizer.ggml.bos_token_id": "2"},
                    "tokenize_settings": {"add_bos": False, "special": False},
                }
                for hf_id in exp.MODEL_SPECS
            ]
        },
        "exp6865": {
            "v601_evidence_contract_ready_score": 1,
            "canonical_tokenizer_payload_schema": {
                "schema_id": exp.CANONICAL_PAYLOAD_SCHEMA_VERSION,
                "version": "v1",
            },
        },
        "exp6862": {
            "frozen_sequence_template_manifest": {
                "raw_prompt_template": exp.EXP6862_RAW_PROMPT_TEMPLATE,
                "candidate_sequence_template": exp.EXP6862_CANDIDATE_SEQUENCE_TEMPLATE,
            }
        },
    }
    for source_id, payload in payloads.items():
        path = root / exp.SOURCE_PATHS[source_id]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return frozen_specs, model_bytes


class _FakeNativeTokenizer:
    def __init__(self, _path: str) -> None:
        self.evidence = _semantic_evidence()
        self.closed = False

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        return [int(add_bos), int(special), len(text), sum(text) % 251]

    def close(self) -> None:
        self.closed = True


def test_req_inference_6866_spec_declares_the_complete_contract() -> None:
    """REQ-INFERENCE-6866 exists before implementation code."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-INFERENCE-6866:", 1)[1]
    for scenario in (
        "SCENARIO-INFERENCE-6866-CANONICAL-ORDER",
        "SCENARIO-INFERENCE-6866-SEMANTIC-DRIFT",
        "SCENARIO-INFERENCE-6866-TEXT-NORMALIZATION",
        "SCENARIO-INFERENCE-6866-OLD-RECEIPT-MIGRATION",
        "SCENARIO-INFERENCE-6866-PROBE-MATRIX",
        "SCENARIO-INFERENCE-6866-BLOCKED-ARTIFACT",
    ):
        assert scenario in section
    assert "canonical_tokenizer_binding_ready_score" in section
    assert "token_likelihood_call_count" in section
    assert "generated_answer_count" in section


def test_scenario_6866_schema_order_and_wrapper_changes_do_not_change_payload() -> None:
    """SCENARIO-INFERENCE-6866-CANONICAL-ORDER ignores wrapper-only changes."""

    first_evidence = _semantic_evidence()
    second_evidence = dict(reversed(list(first_evidence.items())))
    first = _reduced(
        first_evidence,
        source_receipt={"hf_id": "model", "detail": "old", "timestamp": "then"},
    )
    second = _reduced(
        second_evidence,
        source_receipt={
            "timestamp": "now",
            "nested_receipt_hash": "sha256:new",
            "detail": "new",
            "hf_id": "model",
            "new_wrapper": True,
        },
        source_kind="live",
    )

    assert first["canonical_payload"] == second["canonical_payload"]
    assert first["canonical_payload_sha256"] == second["canonical_payload_sha256"]
    comparison = exp.compare_reduced_receipts(first, second)
    assert comparison["comparison_status"] == "wrapper_schema_only"
    assert comparison["mismatch_reasons"] == []

    same_wrapper = _reduced(first_evidence)
    assert exp.compare_reduced_receipts(first, same_wrapper)["comparison_status"] == (
        "semantic_payload_equal"
    )


@pytest.mark.parametrize(
    ("mutation", "reason_prefix"),
    [
        ("vocabulary", "vocabulary_drift:token_pieces_sha256"),
        ("special", "special_token_drift:bos_token_id"),
        ("add_bos", "add_bos_drift:metadata_default"),
        ("chat_template", "chat_template_drift:template_sha256"),
    ],
)
def test_scenario_6866_semantic_drift_is_typed_and_blocking(
    mutation: str, reason_prefix: str
) -> None:
    """SCENARIO-INFERENCE-6866-SEMANTIC-DRIFT covers every named drift class."""

    baseline_evidence = _semantic_evidence()
    changed_evidence = deepcopy(baseline_evidence)
    if mutation == "vocabulary":
        changed_evidence["token_pieces_sha256"] = exp.sha256_sequence(("a", "b", "d"))
    elif mutation == "special":
        changed_evidence["special_token_ids"]["bos_token_id"] = 9  # type: ignore[index]
    elif mutation == "add_bos":
        changed_evidence["add_bos_metadata_default"] = False
    else:
        changed_evidence["chat_template"] = "<changed>{content}</changed>"

    comparison = exp.compare_reduced_receipts(
        _reduced(baseline_evidence), _reduced(changed_evidence, source_kind="live")
    )

    assert comparison["comparison_status"] == "semantic_payload_drift"
    assert reason_prefix in comparison["mismatch_reasons"]
    assert comparison["blocks_model"] is True


def test_scenario_6866_probe_and_setting_drift_are_typed() -> None:
    """SCENARIO-INFERENCE-6866-PROBE-MATRIX blocks token and setting drift."""

    token_drift = exp.compare_reduced_receipts(
        _reduced(), _reduced(probe_outputs=_probe_outputs(8), source_kind="live")
    )
    assert token_drift["mismatch_reasons"] == ["token_output_drift:empty_text:plain"]

    changed = _probe_outputs()
    changed[0]["settings"]["special"] = True  # type: ignore[index]
    setting_drift = exp.compare_reduced_receipts(
        _reduced(), _reduced(probe_outputs=changed, source_kind="live")
    )
    assert setting_drift["mismatch_reasons"] == ["tokenization_setting_drift:empty_text:plain"]

    changed_text = _probe_outputs()
    changed_text[0]["text_utf8_sha256"] = exp.sha256_bytes(b"changed")
    text_drift = exp.compare_reduced_receipts(
        _reduced(), _reduced(probe_outputs=changed_text, source_kind="live")
    )
    assert text_drift["mismatch_reasons"] == ["probe_text_drift:empty_text:plain"]

    malformed = deepcopy(_reduced())
    malformed["canonical_payload"]["frozen_probe_outputs"].append(None)  # type: ignore[index,union-attr]
    assert exp.compare_reduced_receipts(malformed, malformed)["blocks_model"] is False

    archived = _reduced()
    live = deepcopy(archived)
    live["canonical_payload"]["frozen_probe_outputs"][0]["settings"]["special"] = True  # type: ignore[index,union-attr]
    setting_rows, setting_witnesses = exp._probe_comparison_rows("model", archived, live)
    assert setting_rows[0]["mismatch_reason"].startswith("tokenization_setting_drift:")
    assert setting_witnesses

    live = deepcopy(archived)
    live["canonical_payload"]["frozen_probe_outputs"][0]["text_utf8_sha256"] = "sha256:new"  # type: ignore[index,union-attr]
    text_rows, text_witnesses = exp._probe_comparison_rows("model", archived, live)
    assert text_rows[0]["mismatch_reason"].startswith("probe_text_drift:")
    assert text_witnesses


def test_scenario_6866_unicode_normalization_and_invalid_utf8_are_explicit() -> None:
    """SCENARIO-INFERENCE-6866-TEXT-NORMALIZATION fixes bytes before tokenization."""

    nfd = "cafe\u0301"
    assert exp.prepare_probe_bytes(nfd, normalization="NFC") == "caf\u00e9".encode()
    assert exp.prepare_probe_bytes("caf\u00e9", normalization="NFD") == nfd.encode()
    assert exp.prepare_probe_bytes(b"bad\xffutf8", normalization="none") == (
        "bad\ufffdutf8".encode()
    )
    with pytest.raises(ValueError, match="unsupported Unicode normalization"):
        exp.prepare_probe_bytes("text", normalization="NFKC")
    with pytest.raises(ValueError, match="utf8_error_mode must be replace"):
        exp.prepare_probe_bytes(b"text", normalization="none", utf8_error_mode="strict")


def test_scenario_6866_old_receipt_migration_records_provenance_only() -> None:
    """SCENARIO-INFERENCE-6866-OLD-RECEIPT-MIGRATION maps every legacy field."""

    legacy = {
        "hf_id": "example/model",
        "detail": "native tokenizer returned tokens",
        "loadable": True,
        "metadata_key_count": 7,
        "metadata_present": True,
        "probe_token_ids": [2, 7],
        "source": "embedded_gguf_llama_cpp_vocab_only",
        "tokenizer_sha256": "sha256:old-wrapper-hash",
    }
    reduced = _reduced(source_receipt=legacy)
    field_map = {row["source_field"]: row for row in reduced["source_field_map"]}

    assert set(field_map) == set(legacy)
    assert field_map["detail"]["disposition"] == "excluded_wrapper_only"
    assert field_map["probe_token_ids"]["disposition"] == "recomputed"
    assert field_map["tokenizer_sha256"]["disposition"] == "excluded_nested_hash"
    assert "old-wrapper-hash" not in exp.canonical_json(reduced["canonical_payload"])


def test_scenario_6866_frozen_matrix_covers_every_required_probe_and_setting() -> None:
    """SCENARIO-INFERENCE-6866-PROBE-MATRIX freezes all adversarial probe classes."""

    manifest = exp.semantic_probe_manifest()
    classes = {row["probe_class"] for row in manifest["probes"]}
    assert set(exp.REQUIRED_PROBE_CLASSES) <= classes
    assert manifest["raw_prompt_template"] == exp.EXP6862_RAW_PROMPT_TEMPLATE
    assert manifest["candidate_sequence_template"] == exp.EXP6862_CANDIDATE_SEQUENCE_TEMPLATE

    rows = exp.run_probe_matrix(_FakeNativeTokenizer("unused"), manifest)
    expected = len(manifest["probes"]) * len(exp.PROBE_SETTINGS)
    assert len(rows) == expected
    assert all(
        set(row) == {"probe_id", "text_utf8_sha256", "settings", "token_ids"} for row in rows
    )
    assert {row["settings"]["setting_id"] for row in rows} == {
        setting["setting_id"] for setting in exp.PROBE_SETTINGS
    }


def test_model_binding_checks_hash_size_quantization_hub_and_snapshot() -> None:
    """REQ-INFERENCE-6866 binds every exact model property."""

    current = {
        "hf_id": exp.MODEL_SPECS[0],
        "sha256": "sha256:model",
        "size_bytes": 10,
        "quantization": "Q4_K_M",
        "snapshot_identity": "snapshot-a",
    }
    assert exp.model_binding_mismatch_reasons(current, dict(current)) == []

    changed = {
        "hf_id": "wrong/model",
        "sha256": "sha256:changed",
        "size_bytes": 11,
        "quantization": "Q5_K_M",
        "snapshot_identity": "snapshot-b",
    }
    assert exp.model_binding_mismatch_reasons(current, changed) == [
        "hub_id_drift",
        "model_hash_drift",
        "model_size_drift",
        "quantization_drift",
        "snapshot_identity_drift",
    ]
    assert exp.snapshot_identity("/cache/model/snapshots/abc/model.gguf") == "abc"
    assert exp.snapshot_identity("/cache/no-snapshot/model.gguf") == ""
    assert exp._token_id("7") == 7
    assert exp._token_id(-1) is None
    assert exp._token_id(None) is None


def test_blocked_precondition_writes_complete_required_schema(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6866-BLOCKED-ARTIFACT always returns terminal evidence."""

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=lambda _exp6850: [],
        tokenizer_factory=_FakeNativeTokenizer,
    )

    assert artifact["canonical_tokenizer_binding_ready_score"] == 0
    assert artifact["honest_verdict"] == (
        "complete_blocked_canonical_tokenizer_binding_requalification"
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "required_source_readability"
    assert artifact["gate_check_summary"]["expected"] == "all required artifacts readable"
    assert artifact["gate_check_summary"]["observed"]
    assert exp.validate_artifact(artifact) == []

    malformed_root = tmp_path / "malformed"
    for source_id, relative in exp.SOURCE_PATHS.items():
        path = malformed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]" if source_id == "exp6850" else "{}", encoding="utf-8")
    malformed = exp.build_artifact(
        malformed_root,
        "20260902",
        model_resolver=lambda _exp6850: [],
        tokenizer_factory=_FakeNativeTokenizer,
    )
    assert "JSON object required" in malformed["gate_check_summary"]["observed"][0]["error"]


def test_synthetic_three_model_requalification_is_positive_and_append_only(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6866 requalifies all exact byte-identical bindings."""

    frozen_specs, _model_bytes = _write_fixture_sources(tmp_path)

    def resolver(_exp6850: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "sha256": row["model_sha256"],
                "size_bytes": row["model_size_bytes"],
                "quantization": row["quantization"],
                "snapshot_identity": exp.snapshot_identity(str(row["model_path"])),
                "cached_sota_pair_called": True,
                "embedded_tokenizer_metadata": True,
            }
            for row in frozen_specs
        ]

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=resolver,
        tokenizer_factory=_FakeNativeTokenizer,
    )

    assert artifact["canonical_tokenizer_binding_ready_score"] == 1
    assert artifact["models_used"] == list(exp.MODEL_SPECS)
    assert len(artifact["archived_receipt_rows"]) == 3
    assert len(artifact["live_receipt_rows"]) == 3
    assert len(artifact["canonical_payload_hash_rows"]) == 3
    assert all(
        row["comparison_status"] == "wrapper_schema_only"
        for row in artifact["canonical_payload_hash_rows"]
    )
    assert len(artifact["rows"]) == (
        3 * len(exp.semantic_probe_manifest()["probes"]) * len(exp.PROBE_SETTINGS)
    )
    assert artifact["mismatch_witnesses"] == []
    correction = artifact["append_only_correction_receipt"]
    assert correction["old_artifacts_rewritten"] is False
    assert correction["correction_kind"] == "wrapper_schema_only"
    assert correction["target_artifacts"] == [
        exp.SOURCE_PATHS["exp6850"].as_posix(),
        exp.SOURCE_PATHS["exp6863"].as_posix(),
    ]
    assert artifact["token_likelihood_call_count"] == 0
    assert artifact["generated_answer_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []


def test_alternate_live_path_hashes_the_frozen_file_and_still_matches(tmp_path: Path) -> None:
    """REQ-INFERENCE-6866 recomputes both files when their path strings differ."""

    frozen_specs, model_bytes = _write_fixture_sources(tmp_path)
    live_rows: list[dict[str, object]] = []
    for index, frozen in enumerate(frozen_specs):
        hf_id = str(frozen["hf_id"])
        live_path = tmp_path / "live" / "snapshots" / f"snap-{index}" / f"live-{index}.gguf"
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_bytes(model_bytes[hf_id])
        live_rows.append(
            {
                "hf_id": hf_id,
                "model_path": str(live_path),
                "sha256": frozen["model_sha256"],
                "size_bytes": frozen["model_size_bytes"],
                "quantization": frozen["quantization"],
                "snapshot_identity": f"snap-{index}",
                "cached_sota_pair_called": True,
            }
        )

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=lambda _exp6850: live_rows,
        tokenizer_factory=_FakeNativeTokenizer,
    )
    assert artifact["canonical_tokenizer_binding_ready_score"] == 1


def test_missing_frozen_file_and_missing_pair_call_fail_before_tokenization(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6866 preserves exact model precondition failures."""

    frozen_specs, _ = _write_fixture_sources(tmp_path)
    exp6850_path = tmp_path / exp.SOURCE_PATHS["exp6850"]
    exp6850 = json.loads(exp6850_path.read_text(encoding="utf-8"))
    missing_path = str(tmp_path / "missing.gguf")
    exp6850["model_specs"][0]["model_path"] = missing_path
    exp6850["model_artifact_hashes"][exp.MODEL_SPECS[0]]["path"] = missing_path
    exp6850_path.write_text(json.dumps(exp6850), encoding="utf-8")

    live_rows = [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "sha256": row["model_sha256"],
            "size_bytes": row["model_size_bytes"],
            "quantization": row["quantization"],
            "snapshot_identity": exp.snapshot_identity(str(row["model_path"])),
            "cached_sota_pair_called": index != 1,
        }
        for index, row in enumerate(frozen_specs)
    ]
    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=lambda _exp6850: live_rows,
        tokenizer_factory=_FakeNativeTokenizer,
    )
    assert artifact["canonical_tokenizer_binding_ready_score"] == 0
    failed = artifact["gate_check_summary"]["failed_checks"]
    assert any("model_hash_drift" in row["observed"] for row in failed)
    assert any("cached_sota_pair_not_called" in row["observed"] for row in failed)


def test_semantic_probe_drift_blocks_the_model_and_preserves_witness(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFERENCE-6866-SEMANTIC-DRIFT preserves each mismatch."""

    frozen_specs, _ = _write_fixture_sources(tmp_path)

    class DriftingTokenizer(_FakeNativeTokenizer):
        load_count = 0

        def __init__(self, path: str) -> None:
            super().__init__(path)
            type(self).load_count += 1
            self.side = "archived" if type(self).load_count % 2 else "live"

        def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
            tokens = super().tokenize(text, add_bos=add_bos, special=special)
            return tokens if self.side == "archived" else [*tokens, 99]

    def resolver(_exp6850: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "sha256": row["model_sha256"],
                "size_bytes": row["model_size_bytes"],
                "quantization": row["quantization"],
                "snapshot_identity": exp.snapshot_identity(str(row["model_path"])),
                "cached_sota_pair_called": True,
                "embedded_tokenizer_metadata": True,
            }
            for row in frozen_specs
        ]

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=resolver,
        tokenizer_factory=DriftingTokenizer,
    )

    assert artifact["canonical_tokenizer_binding_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["mismatch_witnesses"]
    assert all(
        row["reason"].startswith("token_output_drift:") for row in artifact["mismatch_witnesses"]
    )
    assert artifact["gate_check_summary"]["failed_check"] == "canonical_semantic_and_probe_match"
    assert exp.validate_artifact(artifact) == []


def test_native_tokenizer_load_or_metadata_failure_stays_complete_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFERENCE-6866-BLOCKED-ARTIFACT covers tokenizer failures."""

    frozen_specs, _ = _write_fixture_sources(tmp_path)

    def resolver(_exp6850: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "sha256": row["model_sha256"],
                "size_bytes": row["model_size_bytes"],
                "quantization": row["quantization"],
                "snapshot_identity": exp.snapshot_identity(str(row["model_path"])),
                "cached_sota_pair_called": True,
            }
            for row in frozen_specs
        ]

    class MissingMetadata(_FakeNativeTokenizer):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.evidence["tokenizer_model"] = ""

    missing = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=resolver,
        tokenizer_factory=MissingMetadata,
    )
    assert missing["canonical_tokenizer_binding_ready_score"] == 0
    assert missing["gate_check_summary"]["failed_check"].endswith(".embedded_tokenizer_metadata")
    assert missing["append_only_correction_receipt"]["correction_kind"] == ("blocked_precondition")

    def failed_factory(_path: str) -> _FakeNativeTokenizer:
        raise RuntimeError("native tokenizer failed")

    failed = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=resolver,
        tokenizer_factory=failed_factory,
    )
    assert failed["canonical_tokenizer_binding_ready_score"] == 0
    assert failed["gate_check_summary"]["failed_check"].endswith(".native_tokenizer_load")
    assert failed["gate_check_summary"]["observed"] == "RuntimeError:native tokenizer failed"
    assert exp.validate_artifact(failed) == []


def test_metadata_only_drift_becomes_a_non_probe_mismatch_witness(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6866-SEMANTIC-DRIFT records metadata-only changes."""

    frozen_specs, _ = _write_fixture_sources(tmp_path)

    def resolver(_exp6850: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "sha256": row["model_sha256"],
                "size_bytes": row["model_size_bytes"],
                "quantization": row["quantization"],
                "snapshot_identity": exp.snapshot_identity(str(row["model_path"])),
                "cached_sota_pair_called": True,
            }
            for row in frozen_specs
        ]

    class MetadataDrift(_FakeNativeTokenizer):
        load_count = 0

        def __init__(self, path: str) -> None:
            super().__init__(path)
            type(self).load_count += 1
            if type(self).load_count % 2 == 0:
                self.evidence["special_token_ids"]["bos_token_id"] = 99  # type: ignore[index]

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=resolver,
        tokenizer_factory=MetadataDrift,
    )
    assert {row["reason"] for row in artifact["mismatch_witnesses"]} == {
        "special_token_drift:bos_token_id"
    }
    assert all(row["probe_id"] is None for row in artifact["mismatch_witnesses"])


def test_validator_rejects_counters_terminal_shape_and_checksum(tmp_path: Path) -> None:
    """REQ-INFERENCE-6866 validates all consumer-facing gate fields."""

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=lambda _exp6850: [],
        tokenizer_factory=_FakeNativeTokenizer,
    )
    broken = deepcopy(artifact)
    del broken["rows"]
    broken["token_likelihood_call_count"] = 1
    broken["generated_answer_count"] = 1
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "unknown"
    broken["honest_verdict"] = "blocked"
    broken["canonical_tokenizer_binding_ready_score"] = 2
    broken["field_principles"] = {}
    broken["reproducibility_checksum"] = "sha256:bad"

    assert exp.validate_artifact(broken) == [
        "missing_required_fields:rows",
        "field_principles_must_cover_every_field",
        "token_likelihood_call_count_must_be_zero",
        "generated_answer_count_must_be_zero",
        "verifier_is_oracle_must_be_false",
        "invalid_verdict_class",
        "honest_verdict_not_terminal",
        "invalid_ready_score",
        "reproducibility_checksum_mismatch",
    ]


def test_atomic_writer_and_cli_use_only_the_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFERENCE-6866 never writes a test artifact into tracked results."""

    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        model_resolver=lambda _exp6850: [],
        tokenizer_factory=_FakeNativeTokenizer,
    )
    output = tmp_path / "exp6866.json"
    exp.write_json_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 6866

    monkeypatch.setattr(exp, "build_artifact", lambda _root, _date: artifact)
    assert exp.main(["--date", "20260902", "--output", str(output)]) == 0

    broken = deepcopy(artifact)
    broken["honest_verdict"] = "invalid"
    monkeypatch.setattr(exp, "build_artifact", lambda _root, _date: broken)
    with pytest.raises(ValueError, match="honest_verdict_not_terminal"):
        exp.main(["--date", "20260902", "--output", str(output)])

    monkeypatch.setattr(exp.os, "replace", lambda _source, _target: None)
    unused_output = tmp_path / "not-replaced.json"
    exp.write_json_atomic(unused_output, artifact)
    assert not unused_output.exists()
    assert not list(tmp_path.glob(f".{unused_output.name}.*"))
