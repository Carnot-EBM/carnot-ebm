"""Tests for Exp5922 GGUF schema decoder bridge.

Spec refs: REQ-VERIFY-5922, SCENARIO-VERIFY-5922.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_5921_schema_derived_constraintir_support as exp5921
from carnot import experiment_5922_gguf_schema_decoder_bridge as exp5922


class FakeEmbeddedTokenizer:
    """Small deterministic tokenizer with the same methods the bridge needs."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        *,
        eos_token_id: int = 999,
        unsupported: set[bytes] | None = None,
        split: dict[bytes, list[int]] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.eos_token_id = eos_token_id
        self.unsupported = unsupported or set()
        self.split = split or {}
        self.reverse = {value: key for key, value in self.vocab.items()}

    def tokenize(self, data: bytes) -> list[int]:
        if data in self.unsupported:
            return []
        if data in self.split:
            return list(self.split[data])
        if data in self.reverse:
            return [self.reverse[data]]
        output: list[int] = []
        index = 0
        ordered = sorted(self.reverse, key=len, reverse=True)
        while index < len(data):
            for piece in ordered:
                if data.startswith(piece, index):
                    output.append(self.reverse[piece])
                    index += len(piece)
                    break
            else:
                raise ValueError(f"no fake token for byte {data[index:index + 1]!r}")
        return output

    def detokenize(self, token_ids: list[int]) -> bytes:
        return b"".join(self.vocab[token_id] for token_id in token_ids)


class KwargEmbeddedTokenizer(FakeEmbeddedTokenizer):
    def tokenize(self, data: bytes, *, add_bos: bool = False, special: bool = False) -> list[int]:
        assert add_bos is False
        assert special is False
        return super().tokenize(data)


def _byte_vocab() -> dict[int, bytes]:
    vocab = {idx + 1: bytes([idx]) for idx in range(32, 127)}
    vocab[200] = "é".encode()
    vocab[201] = "β".encode()
    vocab[202] = b"\xc3"
    return vocab


def _fake_vocab() -> exp5922.TokenVocabulary:
    tokenizer = FakeEmbeddedTokenizer(
        {
            1: b"{",
            2: b"}",
            3: b" ",
            4: b'"schema',
            5: b'_version"',
            6: b">",
            7: b"=",
            8: b"@",
            9: b'"domains"',
            10: b":",
            11: b"[",
            12: b"]",
            13: b",",
        },
        unsupported={b"UNSUPPORTED_TERMINAL"},
        split={b'"schema_version"': [4, 5], b">=": [6, 7]},
    )
    return exp5922.vocabulary_from_embedded_tokenizer(
        model_id="fake/model-GGUF",
        model_path="/tmp/fake.gguf",
        tokenizer=tokenizer,
        token_bytes_by_id=tokenizer.vocab,
        eos_token_id=tokenizer.eos_token_id,
    )


def _char_vocab() -> exp5922.TokenVocabulary:
    tokenizer = FakeEmbeddedTokenizer(_byte_vocab(), eos_token_id=5000)
    return exp5922.vocabulary_from_embedded_tokenizer(
        model_id="fake/char-GGUF",
        model_path="/tmp/char.gguf",
        tokenizer=tokenizer,
        token_bytes_by_id=tokenizer.vocab,
        eos_token_id=tokenizer.eos_token_id,
    )


def _valid_text() -> str:
    support = exp5921.compile_schema_support()
    cases = {case["case_id"]: case for case in exp5921.build_adversary_cases()}
    payload = cases["held_family_menu_canonical"]["candidate"]
    assert exp5921.validate_with_support(payload, support)["full_support_valid"] is True
    return exp5921.canonical_json(payload)


def test_terminal_mapping_records_single_multi_and_unsupported_receipts() -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    support = exp5921.compile_schema_support()
    vocab = _fake_vocab()
    terminals = exp5922.grammar_terminal_strings(support) + ["UNSUPPORTED_TERMINAL"]
    mapping = exp5922.map_terminals_to_token_ids(vocab, terminals)

    mismatch_tokenizer = FakeEmbeddedTokenizer({1: b"x"}, split={b"A": [1]})
    mismatch_vocab = exp5922.vocabulary_from_embedded_tokenizer(
        model_id="fake/mismatch",
        model_path="/tmp/mismatch.gguf",
        tokenizer=mismatch_tokenizer,
        token_bytes_by_id=mismatch_tokenizer.vocab,
        eos_token_id=None,
    )
    kw_tokenizer = KwargEmbeddedTokenizer({1: b"{"})
    kw_vocab = exp5922.vocabulary_from_embedded_tokenizer(
        model_id="fake/kwargs",
        model_path="/tmp/kwargs.gguf",
        tokenizer=kw_tokenizer,
        token_bytes_by_id=kw_tokenizer.vocab,
        eos_token_id=None,
    )
    mismatch = exp5922.map_terminals_to_token_ids(mismatch_vocab, ["A"])

    assert mapping["single_token_terminals"]["{"]["token_ids"] == [1]
    assert kw_vocab.encode_text("{") == [1]
    assert mapping["multi_token_terminals"]['"schema_version"']["token_ids"] == [4, 5]
    assert mapping["multi_token_terminals"][">="]["token_ids"] == [6, 7]
    assert mapping["unsupported_terminals"]["UNSUPPORTED_TERMINAL"]["reason"] == "tokenize_empty"
    assert mismatch["unsupported_terminals"]["A"]["reason"] == "roundtrip_mismatch"
    assert mapping["used_hf_autotokenizer"] is False
    assert mapping["embedded_tokenizer_only"] is True


def test_bridge_masks_logits_eos_and_dead_ends_fail_closed() -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    bridge = exp5922.SchemaDecoderBridge(exp5921.compile_schema_support(), _fake_vocab())
    allowed_start = bridge.allowed_token_ids(b"")
    assert 1 in allowed_start
    assert 3 in allowed_start
    assert 8 not in allowed_start
    assert bridge.vocabulary.eos_token_id not in allowed_start

    complete_allowed = bridge.allowed_token_ids(_valid_text().encode())
    assert bridge.vocabulary.eos_token_id in complete_allowed

    assert bridge.allowed_token_ids(b"@") == []
    processor = exp5922.LlamaCppSchemaLogitsProcessor(bridge)
    scores = np.zeros(bridge.vocabulary.max_token_id + 1, dtype=float)
    start_masked = processor([], scores.copy())
    assert start_masked[1] == 0.0
    assert np.isneginf(start_masked[8])
    masked = processor([8], scores.copy())
    assert np.isneginf(masked).all()
    assert processor.last_receipt["dead_end"] is True
    empty_piece_vocab = exp5922.TokenVocabulary(
        model_id="fake/empty",
        model_path="/tmp/empty.gguf",
        token_bytes_by_id={1: b""},
        eos_token_id=None,
        tokenizer_receipt={},
    )
    assert exp5922.SchemaDecoderBridge(exp5921.compile_schema_support(), empty_piece_vocab).token_preserves_continuation(b"", 1) is False
    assert bridge.vocabulary.encode_text("{") == [1]


def test_known_valid_and_adversarial_token_replay_use_reference_support() -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    support = exp5921.compile_schema_support()
    bridge = exp5922.SchemaDecoderBridge(support, _char_vocab())
    valid = bridge.replay_text(_valid_text())
    assert valid["accepted"] is True
    assert valid["complete_valid"] is True
    assert valid["rejected_token"] is None

    invalid = bridge.replay_text(
        '{"domains":[{"id":"x","type":"float","values":["a"]}],"entities":[],'
        '"facts":[],"predicates":[],"query":{"vars":{},"where":{"node":"and","terms":[]}},'
        '"rules":[],"schema_version":"carnot.constraint_ir.v1"}'
    )
    assert invalid["accepted"] is False
    assert invalid["rejected_token"] is not None
    assert "no_schema_valid_continuation" in invalid["reason"]


def test_utf8_whitespace_numeric_identifier_matrix() -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    bridge = exp5922.SchemaDecoderBridge(exp5921.compile_schema_support(), _char_vocab())
    matrix = exp5922.utf8_whitespace_numeric_identifier_matrix(bridge)

    assert matrix["whitespace"]["space_prefix_admitted"] is True
    assert matrix["numeric_literals"]["integer_value_prefix_admitted"] is True
    assert matrix["identifiers"]["identifier_string_prefix_admitted"] is True
    assert matrix["utf8"]["complete_utf8_string_prefix_admitted"] is True
    assert matrix["utf8"]["partial_utf8_waits_for_continuation"] is True
    assert matrix["invalid"]["invalid_leading_byte_rejected"] is True


def test_prefix_scanner_reports_defensive_edge_reasons() -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    support = exp5921.compile_schema_support()
    samples = {
        b'"\\x"': "json_decode_error",
        b'{"domains":["x",]': "array_closed_before_value",
        b'{"domains":[{"id":"a\\n': "open_string",
        b'{"domains":[{"zzz":': "unsupported_object_key",
        b'{"domains":[{"zzz': "unsupported_object_key_prefix",
        b'{"domains":[{"id":"x","type":"float"': "unsupported_terminal_string",
        b'{"domains":[{"id":"x","type":"flo': "unsupported_terminal_string_prefix",
        b'{"domains":[{"id":"\x01': "control_character_in_string",
        b'{"facts":[truX': "unsupported_literal",
        b'{"domains":[{"id":"x","type":"int","values":[-x': "invalid_integer",
        b'{"domains":[{"id":"x","type":"int","values":[1.': "non_integer_number",
    }
    for data, reason in samples.items():
        assert reason in exp5922.classify_schema_prefix_bytes(data, support).reason

    assert exp5922.classify_schema_prefix_bytes(b"true", support).reason == "complete_json_not_schema_valid"
    assert exp5922._consume_integer("-", 0, []).reason == "partial_integer"
    assert "id" in exp5922._allowed_keys({"container_key": "unknown"}, support)


def test_artifact_validation_and_reproducibility_with_injected_receipts(tmp_path: Path) -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    paths = {}
    for spec in exp5922.MODEL_SPECS:
        path = tmp_path / f"{spec['name']}.gguf"
        path.write_bytes(b"GGUF-fixture-" + spec["hf_id"].encode())
        paths[spec["hf_id"]] = str(path)

    def resolver() -> list[dict[str, object]]:
        return [
            {**spec, "gpu": index % 2, "model_path": paths[spec["hf_id"]]}
            for index, spec in enumerate(exp5922.MODEL_SPECS)
        ]

    def loader(spec: dict[str, object]) -> exp5922.TokenVocabulary:
        vocab = _char_vocab()
        return exp5922.TokenVocabulary(
            model_id=str(spec["hf_id"]),
            model_path=str(spec["model_path"]),
            token_bytes_by_id=vocab.token_bytes_by_id,
            eos_token_id=vocab.eos_token_id,
            tokenizer_receipt={"source": "embedded_gguf_llama_cpp_vocab_only_fixture"},
        )

    def public_api() -> dict[str, object]:
        return {
            "ok": True,
            "binding": "llama_cpp.Llama.__call__",
            "logits_processor_parameter": True,
        }

    artifact = exp5922.build_artifact(
        root=tmp_path,
        output_path=tmp_path / exp5922.RESULT_RELATIVE_PATH,
        duration_s=0.0,
        model_resolver=resolver,
        tokenizer_loader=loader,
        cuda_smoke_runner=lambda specs, bridges: exp5922.fake_one_step_cuda_smoke(specs),
        public_api_checker=public_api,
    )

    assert artifact["status"] == "complete_ready"
    assert artifact["gguf_schema_decoder_bridge_ready_score"] == 1.0
    assert artifact["full_answer_enumeration_used"] is False
    assert artifact["inference_substrate"] == "public_llama_cpp_cuda_tokenizer_bridge_smoke"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert set(exp5922.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(artifact["model_file_hashes"]) == {spec["hf_id"] for spec in exp5922.MODEL_SPECS}
    assert all(row["embedded_tokenizer_only"] for row in artifact["embedded_tokenizer_receipts"].values())
    assert artifact["tokenizer_reference_support_parity"]["all_models_parity"] is True
    assert artifact["one_step_cuda_smoke"]["all_smokes_ok"] is True
    exp5922.validate_artifact(artifact)

    output = tmp_path / exp5922.RESULT_RELATIVE_PATH
    written = exp5922.write_artifact(
        root=tmp_path,
        output_path=output,
        duration_s=0.0,
        model_resolver=resolver,
        tokenizer_loader=loader,
        cuda_smoke_runner=lambda specs, bridges: exp5922.fake_one_step_cuda_smoke(specs),
        public_api_checker=public_api,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == written
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    no_duration = exp5922.write_artifact(
        root=tmp_path,
        output_path=tmp_path / "no_duration.json",
        model_resolver=resolver,
        tokenizer_loader=loader,
        cuda_smoke_runner=lambda specs, bridges: exp5922.fake_one_step_cuda_smoke(specs),
        public_api_checker=public_api,
    )
    assert no_duration["duration_s"] >= 0.0

    refreshed = exp5922.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "coverage": 0},
    )
    assert refreshed["test_exit_codes"] == {"focused": 0, "coverage": 0}

    for key, value, message in [
        ("full_answer_enumeration_used", True, "full_answer_enumeration_used"),
        ("gguf_schema_decoder_bridge_ready_score", 0.5, "ready_score"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("honest_verdict", "blocked: wrong", "complete_ready"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5922.validate_artifact(broken)

    parity_broken = json.loads(json.dumps(artifact))
    parity_broken["tokenizer_reference_support_parity"]["all_models_parity"] = False
    with pytest.raises(ValueError, match="tokenizer parity"):
        exp5922.validate_artifact(parity_broken)

    smoke_broken = json.loads(json.dumps(artifact))
    smoke_broken["one_step_cuda_smoke"]["all_smokes_ok"] = False
    with pytest.raises(ValueError, match="one-step CUDA smoke"):
        exp5922.validate_artifact(smoke_broken)

    bad_prefix = json.loads(json.dumps(artifact))
    bad_prefix["gguf_schema_decoder_bridge_ready_score"] = 0.0
    bad_prefix["honest_verdict"] = "bad"
    with pytest.raises(ValueError, match="terminal prefix"):
        exp5922.validate_artifact(bad_prefix)

    missing = dict(artifact)
    del missing["model_specs"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp5922.validate_artifact(missing)

    assert exp5922._gate_replay_receipt(exp5922.REPO_ROOT, exp5921.compile_schema_support())["ok"] is True


def test_blocked_artifact_preserves_precondition_failures(tmp_path: Path) -> None:
    # REQ-VERIFY-5922, SCENARIO-VERIFY-5922
    artifact = exp5922.build_artifact(
        root=tmp_path,
        output_path=tmp_path / exp5922.RESULT_RELATIVE_PATH,
        duration_s=0.0,
        model_resolver=lambda: [{**exp5922.MODEL_SPECS[0], "model_path": None}],
        tokenizer_loader=lambda spec: _char_vocab(),
        cuda_smoke_runner=lambda specs, bridges: {"all_smokes_ok": False, "smokes": []},
        public_api_checker=lambda: {"ok": False, "logits_processor_parameter": False},
    )

    assert artifact["status"] == "blocked"
    assert artifact["gguf_schema_decoder_bridge_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["preconditions_checked"]["all_preconditions_ok"] is False
    exp5922.validate_artifact(artifact)
