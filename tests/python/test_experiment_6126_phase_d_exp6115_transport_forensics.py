"""Tests for Exp6126 immutable Exp6115 transport forensics.

Spec refs: REQ-VERIFY-6126, SCENARIO-VERIFY-6126-CONSERVATION,
SCENARIO-VERIFY-6126-ATTRIBUTION, SCENARIO-VERIFY-6126-TEMPLATE,
SCENARIO-VERIFY-6126-CONTRACT.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from carnot import experiment_6126_phase_d_exp6115_transport_forensics as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-m pytest tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6126_phase_d_exp6115_transport_forensics.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6126_phase_d_exp6115_transport_forensics.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = "git status --short -- scripts/research_conductor.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _write_fake_gguf(path: Path) -> None:
    """SCENARIO-VERIFY-6126-TEMPLATE: write a tiny GGUF metadata surface."""

    metadata: dict[str, Any] = {
        "general.architecture": "gemma4",
        "general.name": "fake-gemma-4-26b",
        "test.uint8": (mod.GGUF_TYPE_UINT8, 7),
        "test.int8": (mod.GGUF_TYPE_INT8, -7),
        "test.uint16": (mod.GGUF_TYPE_UINT16, 512),
        "test.int16": (mod.GGUF_TYPE_INT16, -512),
        "test.int32": (mod.GGUF_TYPE_INT32, -4096),
        "test.float32": (mod.GGUF_TYPE_FLOAT32, 1.25),
        "test.bool": (mod.GGUF_TYPE_BOOL, True),
        "test.uint64": (mod.GGUF_TYPE_UINT64, 2**33),
        "test.int64": (mod.GGUF_TYPE_INT64, -(2**33)),
        "test.float64": (mod.GGUF_TYPE_FLOAT64, 2.5),
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.chat_template": (
            "{{ bos_token }}{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}{% endfor %}"
        ),
        "tokenizer.ggml.tokens": ["<pad>", "</s>", "<s>", "Final", " answer"],
    }
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 0, len(metadata)))
        for key, value in metadata.items():
            encoded_key = key.encode("utf-8")
            handle.write(struct.pack("<Q", len(encoded_key)))
            handle.write(encoded_key)
            if isinstance(value, tuple):
                value_type, raw_value = value
                handle.write(struct.pack("<I", value_type))
                if value_type == mod.GGUF_TYPE_UINT8:
                    handle.write(struct.pack("<B", raw_value))
                elif value_type == mod.GGUF_TYPE_INT8:
                    handle.write(struct.pack("<b", raw_value))
                elif value_type == mod.GGUF_TYPE_UINT16:
                    handle.write(struct.pack("<H", raw_value))
                elif value_type == mod.GGUF_TYPE_INT16:
                    handle.write(struct.pack("<h", raw_value))
                elif value_type == mod.GGUF_TYPE_INT32:
                    handle.write(struct.pack("<i", raw_value))
                elif value_type == mod.GGUF_TYPE_FLOAT32:
                    handle.write(struct.pack("<f", raw_value))
                elif value_type == mod.GGUF_TYPE_BOOL:
                    handle.write(struct.pack("<?", raw_value))
                elif value_type == mod.GGUF_TYPE_UINT64:
                    handle.write(struct.pack("<Q", raw_value))
                elif value_type == mod.GGUF_TYPE_INT64:
                    handle.write(struct.pack("<q", raw_value))
                elif value_type == mod.GGUF_TYPE_FLOAT64:
                    handle.write(struct.pack("<d", raw_value))
                else:  # pragma: no cover - fixture guard.
                    raise ValueError(value_type)
            elif isinstance(value, str):
                encoded_value = value.encode("utf-8")
                handle.write(struct.pack("<I", mod.GGUF_TYPE_STRING))
                handle.write(struct.pack("<Q", len(encoded_value)))
                handle.write(encoded_value)
            elif isinstance(value, int):
                handle.write(struct.pack("<I", mod.GGUF_TYPE_UINT32))
                handle.write(struct.pack("<I", value))
            elif isinstance(value, list):
                handle.write(struct.pack("<I", mod.GGUF_TYPE_ARRAY))
                handle.write(struct.pack("<IQ", mod.GGUF_TYPE_STRING, len(value)))
                for item in value:
                    encoded_item = str(item).encode("utf-8")
                    handle.write(struct.pack("<Q", len(encoded_item)))
                    handle.write(encoded_item)
            else:  # pragma: no cover - fixture guard.
                raise TypeError(value)


def _artifact(tmp_path: Path) -> dict[str, Any]:
    fake_gguf = tmp_path / "fake.gguf"
    _write_fake_gguf(fake_gguf)
    return mod.run(
        result_path=tmp_path / "experiment_6126.json",
        gguf_metadata_path=fake_gguf,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )


def test_req_verify_6126_spec_declares_forensics_contract() -> None:
    """REQ-VERIFY-6126: OpenSpec names fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6126") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6126",
        "SCENARIO-VERIFY-6126-CONSERVATION",
        "SCENARIO-VERIFY-6126-ATTRIBUTION",
        "SCENARIO-VERIFY-6126-TEMPLATE",
        "SCENARIO-VERIFY-6126-CONTRACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6126_conserves_rows_and_recomputes_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6126-CONSERVATION: all 720 row identities are scored."""

    artifact = _artifact(tmp_path)
    mod.validate_artifact(artifact)

    counts = artifact["expected_observed_and_missing_row_counts"]
    assert counts["expected_row_count"] == 720
    assert counts["observed_row_count"] == 720
    assert counts["unique_candidate_row_id_count"] == 720
    assert counts["duplicate_candidate_row_id_count"] == 0
    assert counts["missing_candidate_row_ids"] == []
    assert counts["identity_complete"] is True

    metrics = artifact[
        "nonempty_empty_whitespace_channel_leak_terminal_field_parse_method_and_accuracy_metrics"
    ]
    assert metrics["candidate_count"] == 720
    assert metrics["nonempty_count"] == 93
    assert metrics["exact_empty_count"] == 627
    assert metrics["whitespace_only_count"] == 0
    assert metrics["channel_token_leak_count"] == 6
    assert metrics["terminal_field_reach_count"] == 36
    assert metrics["parseable_count"] == 44
    assert metrics["method_valid_count"] == 8
    assert metrics["exact_correct_count"] == 15
    assert metrics["all_wrong_question_count"] == 79
    assert metrics["newline_in_raw_generation_count"] == 0
    assert metrics["finish_reason_stop_count"] == 709
    assert metrics["finish_reason_length_count"] == 11
    assert metrics["answer_accuracy"] == 0.020833
    assert metrics["parseability"] == 0.061111
    assert metrics["method_validity"] == 0.011111
    assert metrics["all_wrong_rate"] == 0.877778


def test_scenario_verify_6126_attributes_only_observed_failure_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6126-ATTRIBUTION: causal labels stay receipt-bound."""

    artifact = _artifact(tmp_path)
    attribution = artifact["row_level_failure_attribution_and_unknown_count"]

    assert attribution["attribution_policy"]["uses_hidden_labels_for_cause"] is False
    assert attribution["unknown_transport_receipt_cause_count"] == 41
    assert attribution["counts_by_observed_signal"]["exact_empty_completion"] == 627
    assert attribution["counts_by_observed_signal"]["truncated_length_finish_reason"] == 11
    assert attribution["counts_by_observed_signal"]["channel_token_leakage"] == 6
    assert attribution["counts_by_observed_signal"]["terminal_answer_field_reached"] == 36
    assert attribution["counts_by_observed_signal"]["parser_failure"] == 676
    assert attribution["counts_by_observed_signal"]["method_failure"] == 712
    assert len(attribution["rows"]) == 720

    first = attribution["rows"][0]
    assert first["candidate_row_id"].endswith("sample-00")
    assert "exact_empty_completion" in first["observed_signals"]
    assert first["receipt_fields_used"] == [
        "raw_generation",
        "finish_reason",
        "generated_token_count",
        "max_new_tokens",
        "parser.parseable",
        "parser.failure_reason",
        "method_valid",
        "method_validity_reason",
        "exact_correct",
    ]
    assert first["causal_overreach_guard"] == "observed_receipts_only"


def test_scenario_verify_6126_inspects_gguf_metadata_and_freezes_v2_contract(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6126-CONTRACT: the justified contract is label-blind."""

    artifact = _artifact(tmp_path)
    provenance = artifact["gguf_model_tokenizer_and_chat_template_provenance"]
    contract = artifact["frozen_v2_messages_reasoning_terminal_field_budget_and_stop_contract"]

    assert provenance["metadata_reader"] == "header_only_no_model_load_no_generation"
    assert provenance["chat_template_present"] is True
    assert provenance["tokenizer_metadata"]["tokenizer.ggml.model"] == "llama"
    raw_metadata = mod.read_gguf_metadata(tmp_path / "fake.gguf")
    assert raw_metadata["tokenizer_metadata"]["tokenizer.ggml.tokens"]["length"] == 5
    scalars = raw_metadata["metadata_scalar_values"]
    assert scalars["test.uint8"] == 7
    assert scalars["test.int8"] == -7
    assert scalars["test.uint16"] == 512
    assert scalars["test.int16"] == -512
    assert scalars["test.int32"] == -4096
    assert scalars["test.float32"] == 1.25
    assert scalars["test.bool"] is True
    assert scalars["test.uint64"] == 2**33
    assert scalars["test.int64"] == -(2**33)
    assert scalars["test.float64"] == 2.5
    assert provenance["runtime_chat_template_api"]["llama_cpp_importable"] is True
    assert provenance["runtime_chat_template_api"]["llama_chat_apply_template_available"] is True

    assert contract["contract_id"] == "exp6126_v2_model_native_messages_no_newline_stop"
    assert contract["label_blind"] is True
    assert contract["serialization"]["api"] == "llama_cpp.Llama.create_chat_completion"
    assert contract["reasoning_region"]["type"] == "natural_assistant_content_before_terminal_field"
    assert contract["terminal_answer_field"]["pattern"] == "Final answer: <A|B|C|D>"
    assert contract["budget"]["max_new_tokens"] == 1024
    assert contract["stop"]["explicit_stop_strings"] == []
    assert contract["stop"]["newline_stop_forbidden"] is True

    assert artifact["model_native_chat_change_justified_score"] == 1
    assert artifact["retirement_triggered"] is False
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")


def test_req_verify_6126_schema_provenance_and_transport_semantics_are_stable(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6126: schema, protected files, and semantic separation hold."""

    artifact = _artifact(tmp_path)
    fake_gguf = tmp_path / "fake.gguf"
    output_path = tmp_path / "written.json"
    written = mod.run(
        result_path=output_path,
        gguf_metadata_path=fake_gguf,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert output_path.exists()
    assert written["status"] == "complete_ready"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["hidden_label_retry_count"] == 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    separation = artifact["transport_semantics_separation_receipt"]
    assert separation["parse_success_used_to_infer_accuracy"] is False
    assert separation["parse_success_used_to_infer_method_validity"] is False
    assert separation["accuracy_source_field"] == "exact_correct"
    assert separation["method_validity_source_field"] == "method_valid"
    assert separation["semantic_metrics_are_label_replay_not_transport"] is True

    provenance = artifact["field_provenance"]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in provenance
        assert provenance[field]["principle"]

    exp6115 = mod._read_json(REPO / mod.EXP6115_ARTIFACT_RELATIVE_PATH)
    assert mod._extract_exp6115_model_path(exp6115).endswith(".gguf")
    assert mod._status_and_verdict(blockers=["missing"], identity_complete=True, score=1)[
        0
    ] == "blocked"
    assert mod._status_and_verdict(blockers=[], identity_complete=True, score=0)[0] == "retired"


def test_scenario_verify_6126_blocks_when_conservation_or_protected_receipts_fail(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6126-CONSERVATION: incomplete receipts block."""

    fake_gguf = tmp_path / "fake.gguf"
    _write_fake_gguf(fake_gguf)
    source_lines = (REPO / mod.EXP6115_ROWS_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()
    short_rows = tmp_path / "short.rows.jsonl"
    short_rows.write_text(source_lines[0] + "\n", encoding="utf-8")
    preconditions = mod.collect_preconditions(
        result_path=tmp_path / "short.json",
        exp6115_rows_path=short_rows,
        gguf_metadata_path=fake_gguf,
    )
    preconditions["row_identity_requirement"]["complete"] = True
    short_artifact = mod.run(
        result_path=tmp_path / "short.json",
        exp6115_rows_path=short_rows,
        gguf_metadata_path=fake_gguf,
        preconditions_checked=preconditions,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert short_artifact["status"] == "blocked"
    assert "expected_observed_row_identity_mismatch" in short_artifact["preconditions_checked"][
        "blocked_reasons"
    ]

    bad_preconditions = mod.collect_preconditions(
        result_path=tmp_path / "bad.json",
        gguf_metadata_path=fake_gguf,
    )
    bad_preconditions["row_identity_requirement"]["complete"] = False
    bad_preconditions["protected_file_hashes_before"] = {
        "scripts/research_conductor.py": "sha256:not-the-real-hash"
    }
    bad_artifact = mod.run(
        result_path=tmp_path / "bad.json",
        gguf_metadata_path=fake_gguf,
        preconditions_checked=bad_preconditions,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert bad_artifact["status"] == "blocked"
    assert "row_identity_precondition_incomplete" in bad_artifact["preconditions_checked"][
        "blocked_reasons"
    ]
    assert "protected_files_changed" in bad_artifact["preconditions_checked"]["blocked_reasons"]
