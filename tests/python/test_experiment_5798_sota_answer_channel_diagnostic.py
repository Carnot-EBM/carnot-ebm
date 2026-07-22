"""Tests for Exp5798 offline SOTA answer-channel diagnostics.

Spec refs: REQ-VERIFY-5798, SCENARIO-VERIFY-5798,
SCENARIO-VERIFY-5798-CONTROLS, REQ-REPORT-5798,
SCENARIO-REPORT-5798, SCENARIO-REPORT-5798-BLOCKED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5798_sota_answer_channel_diagnostic as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5798_sota_answer_channel_diagnostic.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5798_sota_answer_channel_diagnostic.py "
    "-m pytest tests/python/test_experiment_5798_sota_answer_channel_diagnostic.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5798_sota_answer_channel_diagnostic.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


def _model_meta(hf_id: str) -> dict[str, Any]:
    family = mod.model_family(hf_id)
    return {
        "model_path": f"/cache/{family}.gguf",
        "model_hash": mod.sha256_text(f"model::{hf_id}"),
        "chat_template_hash": mod.sha256_text(f"template::{hf_id}"),
        "chat_template_checked": True,
        "gguf_filename": f"{family}-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
    }


def _row(
    *,
    model_hf_id: str,
    fixture_row_id: str,
    raw_response_text: str,
    finish_reason: str,
    output_tokens: int,
    parse_ok: bool,
    selected_label: str = "",
    exact_label: str = "A",
) -> dict[str, Any]:
    parsed_labels = {fixture_row_id: selected_label} if parse_ok else {}
    exact_error = bool(parse_ok and selected_label != exact_label)
    row = {
        "schema": "carnot.experiment_5786.sota_constraint_stream.v1.row",
        "stream_sequence_index": 0,
        "model_hf_id": model_hf_id,
        "model_family": mod.model_family(model_hf_id),
        "model_hash": mod.sha256_text(f"model::{model_hf_id}"),
        "fixture_row_id": fixture_row_id,
        "fixture_unit_id": fixture_row_id.rsplit("-", 1)[0],
        "fixture_row_hash": mod.sha256_text(f"fixture::{fixture_row_id}"),
        "fixture_chronology_index": 0,
        "split": "train",
        "family": "finite_domain_scheduling",
        "surface_kind": "canonical",
        "proof_preserving": True,
        "solver_effort_bin": "low",
        "satisfiability": "sat",
        "exact_label": exact_label,
        "exact_answer": "FEASIBLE",
        "exact_certificate_hash": mod.sha256_text(f"cert::{fixture_row_id}"),
        "prompt_hash": mod.sha256_text(f"prompt::{fixture_row_id}::{model_hf_id}"),
        "raw_response_text": raw_response_text,
        "raw_response_sha256": mod.sha256_text(raw_response_text),
        "finish_reason": finish_reason,
        "output_tokens": output_tokens,
        "timing": {"generation_s": 0.01},
        "generation_error": "",
        "parser_receipt": {
            "parse_ok": parse_ok,
            "parser_failure_reason": "" if parse_ok else "truncation",
            "parsed_labels": parsed_labels,
            "boundary": "exp5785_row_id_to_candidate_label",
        },
        "selected_label": selected_label if parse_ok else "",
        "selected_candidate": "FEASIBLE" if parse_ok and selected_label == "A" else "",
        "taxonomy": {
            "parse_ok": parse_ok,
            "parser_failure": not parse_ok,
            "parser_failure_reason": "" if parse_ok else "truncation",
            "parsed_labels": parsed_labels,
            "selected_label": selected_label if parse_ok else "",
            "selected_candidate": "FEASIBLE" if parse_ok and selected_label == "A" else "",
            "selected_candidate_hash": mod.sha256_text("FEASIBLE")
            if parse_ok and selected_label == "A"
            else "",
            "exact_answer_error": exact_error,
            "contradiction": False,
            "satisfiable_drift": exact_error,
            "protected_fact_distortion": False,
            "abstention": False,
            "truncation": finish_reason == "length" or not parse_ok,
            "valid_correct_response": bool(parse_ok and not exact_error),
            "failure_mode": "valid_correct_response"
            if parse_ok and not exact_error
            else ("satisfiable_drift" if exact_error else "truncation"),
        },
        "row_hash": "",
    }
    row["row_hash"] = mod.stream_row_hash(row)
    return row


def _fixture_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5785.hardness_surface_fixture.v1",
        "fixture_ready_score": 1.0,
        "row_file_sha256": mod.sha256_text("fixture-rows"),
        "parser_control_pass_rate": 1.0,
        "exact_label_coverage": 1.0,
    }


def _stream_artifact(rows: list[dict[str, Any]], rows_path: Path) -> dict[str, Any]:
    models = {hf_id: _model_meta(hf_id) for hf_id in mod.MANDATED_MODEL_IDS}
    runtime_receipts = {
        hf_id: {
            "model_hf_id": hf_id,
            "model_family": mod.model_family(hf_id),
            "llama_cpp_version": "0.3.33",
            "llama_cpp_build_info": {
                "cuda_backend": True,
                "supports_gpu_offload": True,
                "system_info": "CUDA : ARCHS = 860 | REPACK = 1 | ",
                "module": "llama_cpp",
            },
            "chat_template": {
                "available": True,
                "used": True,
                "chat_template_hash": models[hf_id]["chat_template_hash"],
            },
            "cuda_offload_authenticated": True,
            "n_gpu_layers_requested": -1,
            "n_gpu_layers_offloaded": 40,
            "gpu_memory_before_mb": 100,
            "gpu_memory_peak_mb": 4000,
            "gpu_memory_after_mb": 120,
            "rows_attempted": sum(row["model_hf_id"] == hf_id for row in rows),
        }
        for hf_id in mod.MANDATED_MODEL_IDS
    }
    receipts = {
        mod.stream_cell_key(row): {
            "row_hash": row["row_hash"],
            "raw_response_sha256": row["raw_response_sha256"],
            "prompt_hash": row["prompt_hash"],
            "fixture_row_hash": row["fixture_row_hash"],
        }
        for row in rows
    }
    return {
        "schema": "carnot.experiment_5786.sota_constraint_stream.v1",
        "generation_config": {
            "max_tokens": 48,
            "stop": ["<|eot_id|>", "<stop>", "\n\n"],
            "chat_template_required": True,
        },
        "preconditions_checked": {
            "preconditions_ready": True,
            "disk": {"available_mb": 100000, "ok": True, "required_mb": 4096},
            "memory": {"available_mb": 64000, "ok": True, "required_mb": 32768},
            "llama_cpp": {
                "ok": True,
                "version": "0.3.33",
                "cuda_backend": True,
                "supports_gpu_offload": True,
                "system_info": "CUDA : ARCHS = 860 | REPACK = 1 | ",
            },
            "models": models,
        },
        "model_runtime_receipts": runtime_receipts,
        "raw_response_receipts": receipts,
        "raw_response_coverage": 1.0,
        "row_file": str(rows_path),
        "row_file_sha256": mod.sha256_file(rows_path),
        "producer_gate_fields": [
            "stream_ready_score",
            "raw_response_coverage",
            "exact_label_coverage",
            "parser_failure_rate",
        ],
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metadata_reader(model_path: str) -> dict[str, Any]:
    if "qwen" in model_path:
        template = "{% if enable_thinking is false %}<think></think>{% endif %}"
    else:
        template = "{% if enable_thinking %}<|channel>analysis<|message|>{% endif %}"
    return {
        "metadata_source": "unit_test_vocab_only_no_generation",
        "tokenizer.chat_template": template,
    }


def _build_fixture_bundle(
    tmp_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    rows = [
        _row(
            model_hf_id=mod.QWEN_ID,
            fixture_row_id="exp5785-train-finite-domain-scheduling-000-canonical",
            raw_response_text="The user wants to evaluate a constraint problem.\nFacts:\n- slots: 0,1,2",
            finish_reason="length",
            output_tokens=48,
            parse_ok=False,
        ),
        _row(
            model_hf_id=mod.QWEN_ID,
            fixture_row_id="exp5785-train-finite-domain-scheduling-001-canonical",
            raw_response_text="The user wants me to evaluate the fixture and stopped early.",
            finish_reason="stop",
            output_tokens=41,
            parse_ok=False,
        ),
        _row(
            model_hf_id=mod.GEMMA31_ID,
            fixture_row_id="exp5785-train-finite-domain-scheduling-000-canonical",
            raw_response_text="exp5785-train-finite-domain-scheduling-000-canonical: A",
            finish_reason="stop",
            output_tokens=21,
            parse_ok=True,
            selected_label="A",
        ),
        _row(
            model_hf_id=mod.GEMMA26_ID,
            fixture_row_id="exp5785-train-finite-domain-scheduling-000-canonical",
            raw_response_text="exp5785-train-finite-domain-scheduling-000-canonical: B",
            finish_reason="stop",
            output_tokens=21,
            parse_ok=True,
            selected_label="B",
        ),
    ]
    rows_path = tmp_path / "exp5786.rows.jsonl"
    _write_rows(rows_path, rows)
    stream_artifact = _stream_artifact(rows, rows_path)
    artifact = mod.build_diagnostic_artifact(
        fixture_artifact=_fixture_artifact(),
        stream_artifact=stream_artifact,
        rows=rows,
        input_paths={"exp5786_rows": rows_path},
        metadata_reader=_metadata_reader,
        test_commands=TEST_COMMANDS,
        test_exit_codes={TEST_COMMAND: 0},
    )
    return artifact, rows, stream_artifact, rows_path


def _build_fixture_artifact(tmp_path: Path) -> dict[str, Any]:
    artifact, _, _, _ = _build_fixture_bundle(tmp_path)
    return artifact


def test_req_report_5798_specs_declare_required_contract() -> None:
    """REQ-REPORT-5798: OpenSpec declares the offline diagnostic fields."""

    verify_text = VERIFY_SPEC.read_text(encoding="utf-8")
    report_text = REPORT_SPEC.read_text(encoding="utf-8")
    verify_section = verify_text[
        verify_text.index("### REQ-VERIFY-5798") : verify_text.index("### REQ-VERIFY-5734")
    ]
    report_section = report_text[report_text.index("### REQ-REPORT-5798") :]
    combined = " ".join((verify_section + report_section).split())

    for marker in (
        "REQ-VERIFY-5798",
        "SCENARIO-VERIFY-5798",
        "SCENARIO-VERIFY-5798-CONTROLS",
        "REQ-REPORT-5798",
        "SCENARIO-REPORT-5798",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "without running model inference",
        "Grammar or JSON validity SHALL NOT be credited as semantic correctness",
    ):
        assert marker in combined
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in report_section


def test_scenario_verify_5798_reasoning_final_token_and_stop_split(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5798: row attribution keeps channels and stops separate."""

    artifact = _build_fixture_artifact(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete"
    assert artifact["row_count"] == 4
    assert artifact["raw_response_coverage"] == 1.0
    assert artifact["llm_calls_made"] == 0
    assert artifact["channel_diagnostic_ready_score"] == 1.0
    assert artifact["qwen_answer_sentinel_count"] == 0
    assert artifact["qwen_empty_final_count"] == 2
    assert artifact["qwen_exact_cap_count"] == 1
    assert artifact["stop_reason_counts"][mod.QWEN_ID] == {"length": 1, "stop": 1}
    assert artifact["token_length_distributions"][mod.QWEN_ID]["exact_cap_fraction"] == 0.5
    qwen_attr = artifact["per_model_failure_attribution"][mod.QWEN_ID]
    assert qwen_attr["larger_bounded_budget_parse_status"] == "not_established_from_existing_rows"
    assert "parser_boundary_missing" in qwen_attr["failure_class_counts"]
    assert artifact["reasoning_content_receipts"][mod.QWEN_ID]["reasoning_nonempty_count"] == 2
    assert artifact["final_content_receipts"][mod.GEMMA31_ID]["final_nonempty_count"] == 1
    assert artifact["per_model_failure_attribution"][mod.GEMMA26_ID]["exact_mismatch_count"] == 1


def test_scenario_verify_5798_candidate_modes_and_adversarial_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5798-CONTROLS: canary modes and controls are bounded."""

    artifact = _build_fixture_artifact(tmp_path)
    modes_by_family: dict[str, list[dict[str, Any]]] = {}
    for mode in artifact["candidate_mode_matrix"]:
        modes_by_family.setdefault(mode["model_family"], []).append(mode)
        assert mode["max_tokens"] > 48
        assert mode["timeout_s"] > 0
        assert mode["finalizer"]
        assert mode["parser"]
        assert mode["fail_closed_conditions"]
        assert mode["bounded"] is True
        assert mode["executable"] is True

    assert artifact["candidate_mode_count"] == len(artifact["candidate_mode_matrix"])
    assert set(modes_by_family) == {mod.model_family(hf_id) for hf_id in mod.MANDATED_MODEL_IDS}
    assert all(len(rows) >= 2 for rows in modes_by_family.values())
    assert any(mode["mode_type"] == "embedded_template_final_sentinel" for mode in modes_by_family["qwen3-6-35b-a3b"])
    assert any(mode["mode_type"] == "reasoning_disabled_final_sentinel" for mode in modes_by_family["qwen3-6-35b-a3b"])
    assert all(mode["mode_type"] != "grammar_json" for mode in artifact["candidate_mode_matrix"])

    expected_controls = {
        "empty_final_content",
        "reasoning_only_output",
        "invalid_candidate_id",
        "duplicate_candidate_id",
        "schema_control_plane_injection",
        "stop_collision",
        "unclosed_thinking",
        "max_token_exhaustion",
        "exact_answer_mismatch",
    }
    observed_controls = {row["control_id"] for row in artifact["adversarial_control_matrix"]}
    assert expected_controls <= observed_controls
    assert all(row["grammar_can_establish_semantic_correctness"] is False for row in artifact["adversarial_control_matrix"])
    assert artifact["mode_acceptance_rules"]["zero_parser_failures_required"] is True
    assert artifact["mode_retirement_rules"]["retire_on_any_unbounded_generation"] is True


def test_scenario_report_5798_hash_replay_and_blockers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5798-BLOCKED: hash and schema failures block readiness."""

    artifact, rows, stream_artifact, rows_path = _build_fixture_bundle(tmp_path)

    assert mod.verify_input_rows(rows=rows, stream_artifact=stream_artifact, rows_path=rows_path)

    tampered_rows = deepcopy(rows)
    tampered_rows[0]["raw_response_sha256"] = mod.sha256_text("tampered")
    with pytest.raises(mod.ManifestReplayError, match="raw_response_sha256"):
        mod.verify_input_rows(rows=tampered_rows, stream_artifact=stream_artifact, rows_path=rows_path)

    tampered_rows = deepcopy(rows)
    tampered_rows[0]["row_hash"] = mod.sha256_text("tampered-row")
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_input_rows(rows=tampered_rows, stream_artifact=stream_artifact, rows_path=rows_path)

    duplicate_rows = rows + [deepcopy(rows[0])]
    with pytest.raises(mod.ManifestReplayError, match="duplicate stream cell"):
        mod.verify_input_rows(rows=duplicate_rows, stream_artifact=stream_artifact, rows_path=rows_path)

    bad = deepcopy(artifact)
    bad["llm_calls_made"] = 1
    bad["channel_diagnostic_ready_score"] = mod.channel_diagnostic_ready_score(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="llm_calls_made"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["adversarial_control_matrix"] = bad["adversarial_control_matrix"][:-1]
    bad["channel_diagnostic_ready_score"] = mod.channel_diagnostic_ready_score(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="adversarial_control_matrix"):
        mod.validate_artifact(bad)


def test_scenario_report_5798_deterministic_replay_and_real_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5798: real Exp5786 rows replay deterministically offline."""

    first = mod.run(
        result_path=tmp_path / "experiment_5798_sota_answer_channel_diagnostic.json",
        metadata_reader=None,
        test_commands=TEST_COMMANDS,
        test_exit_codes={TEST_COMMAND: 0},
        write=True,
    )
    second = mod.run(
        result_path=tmp_path / "experiment_5798_sota_answer_channel_diagnostic_2.json",
        metadata_reader=None,
        test_commands=TEST_COMMANDS,
        test_exit_codes={TEST_COMMAND: 0},
        write=True,
    )

    assert first["row_count"] == 1080
    assert first["raw_response_coverage"] == 1.0
    assert first["qwen_answer_sentinel_count"] == 0
    assert first["qwen_empty_final_count"] == 360
    assert first["qwen_exact_cap_count"] == 358
    assert first["per_model_failure_attribution"][mod.QWEN_ID]["all_rows_exact_cap"] is False
    assert first["per_model_failure_attribution"][mod.QWEN_ID]["parser_failure_count"] == 360
    assert first["final_content_receipts"][mod.GEMMA31_ID]["final_empty_count"] == 0
    assert first["final_content_receipts"][mod.GEMMA26_ID]["final_empty_count"] == 0
    assert first["local_upstream_distinction"]["upstream_issue_prose_is_local_receipt"] is False
    assert first["upstream_issue_receipts"]["20345"]["evidence_role"] == "motivation_only"
    assert first["honest_verdict"].startswith("complete:")
    assert first["channel_diagnostic_ready_score"] == 1.0
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert json.loads((tmp_path / "experiment_5798_sota_answer_channel_diagnostic.json").read_text()) == first


def test_req_verify_5798_edge_branches_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5798: branch-specific replay, metadata, and schema gates close."""

    artifact, rows, stream_artifact, rows_path = _build_fixture_bundle(tmp_path)
    assert mod.model_family("local/custom-GGUF") == "custom"

    wrong_file_hash = deepcopy(stream_artifact)
    wrong_file_hash["row_file_sha256"] = mod.sha256_text("wrong-file")
    with pytest.raises(mod.ManifestReplayError, match="row_file_sha256"):
        mod.verify_input_rows(rows=rows, stream_artifact=wrong_file_hash, rows_path=rows_path)

    missing_receipt = deepcopy(stream_artifact)
    first_key = mod.stream_cell_key(rows[0])
    del missing_receipt["raw_response_receipts"][first_key]
    with pytest.raises(mod.ManifestReplayError, match="missing receipt"):
        mod.verify_input_rows(rows=rows, stream_artifact=missing_receipt)

    field_mismatch = deepcopy(stream_artifact)
    field_mismatch["raw_response_receipts"][first_key]["prompt_hash"] = mod.sha256_text("wrong")
    with pytest.raises(mod.ManifestReplayError, match="prompt_hash"):
        mod.verify_input_rows(rows=rows, stream_artifact=field_mismatch)

    row_count_mismatch = deepcopy(stream_artifact)
    extra_key = "extra::receipt"
    row_count_mismatch["raw_response_receipts"][extra_key] = deepcopy(
        next(iter(row_count_mismatch["raw_response_receipts"].values()))
    )
    with pytest.raises(mod.ManifestReplayError, match="row count"):
        mod.verify_input_rows(rows=rows, stream_artifact=row_count_mismatch)

    receipt_set_mismatch = deepcopy(stream_artifact)
    first_receipt = receipt_set_mismatch["raw_response_receipts"].pop(first_key)
    receipt_set_mismatch["raw_response_receipts"]["other::cell"] = first_receipt
    with pytest.raises(mod.ManifestReplayError, match="missing receipt"):
        mod.verify_input_rows(rows=rows, stream_artifact=receipt_set_mismatch)

    unclosed = deepcopy(rows[0])
    unclosed["raw_response_text"] = "<think>\nreasoning only"
    unclosed["raw_response_sha256"] = mod.sha256_text(unclosed["raw_response_text"])
    unclosed["row_hash"] = mod.stream_row_hash(unclosed)
    assert "unclosed_thinking" in mod.split_reasoning_final(unclosed, max_tokens=48)["failure_classes"]

    split = [mod.split_reasoning_final(row, max_tokens=48) for row in rows]
    gemma_failure = deepcopy(rows[2])
    gemma_failure["parser_receipt"]["parse_ok"] = False
    gemma_failure["parser_receipt"]["parser_failure_reason"] = "missing_boundary"
    gemma_failure["selected_label"] = ""
    gemma_failure["selected_candidate"] = ""
    gemma_failure["raw_response_text"] = "reasoning without strict final"
    gemma_failure["raw_response_sha256"] = mod.sha256_text(gemma_failure["raw_response_text"])
    gemma_failure["taxonomy"]["parse_ok"] = False
    gemma_failure["taxonomy"]["parser_failure"] = True
    gemma_failure["taxonomy"]["exact_answer_error"] = False
    gemma_failure["row_hash"] = mod.stream_row_hash(gemma_failure)
    branch_rows = [rows[0], gemma_failure, rows[3]]
    branch_split = [mod.split_reasoning_final(row, max_tokens=48) for row in branch_rows]
    failure, *_ = mod._per_model_summaries(branch_rows, branch_split, max_tokens=48)
    assert (
        failure[mod.GEMMA31_ID]["larger_bounded_budget_parse_status"]
        == "mixed_current_rows_canary_required"
    )
    assert split[0]["token_exhausted"] is True

    fake_binary = tmp_path / "llama-cli"
    fake_binary.write_bytes(b"llama")
    monkeypatch.setattr(mod.shutil, "which", lambda name: str(fake_binary) if name == "llama-cli" else None)
    runtime = mod.llama_cpp_runtime_receipts(stream_artifact)
    assert runtime["package"]["standalone_binary_present"] is True

    monkeypatch.setattr(
        mod,
        "_read_llama_cpp_metadata",
        lambda path: {
            "metadata_source": "monkeypatch",
            "tokenizer.chat_template": "{% if enable_thinking %}<think>{% endif %}",
        },
    )
    metadata = mod.embedded_template_metadata(stream_artifact)
    assert metadata[mod.QWEN_ID]["template_metadata_source"] == "monkeypatch"

    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        fixture_artifact_path=tmp_path / "missing.json",
        metadata_reader=None,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert mod.honest_verdict(blocked).startswith("blocked:")
    assert json.loads((tmp_path / "blocked.json").read_text()) == blocked

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required fields"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"candidate_mode_count": 999}), "candidate_mode_matrix"),
        (
            lambda item: item["candidate_mode_matrix"][0].pop("max_tokens"),
            "candidate_mode_matrix",
        ),
        (
            lambda item: item["candidate_mode_matrix"][0].update({"bounded": False}),
            "candidate_mode_matrix",
        ),
        (
            lambda item: item["candidate_mode_matrix"][0].update({"max_tokens": 48}),
            "candidate_mode_matrix",
        ),
        (
            lambda item: item["candidate_mode_matrix"][0].update({"finalizer": ""}),
            "candidate_mode_matrix",
        ),
        (
            lambda item: item.update({"channel_diagnostic_ready_score": 0.0}),
            "channel_diagnostic_ready_score",
        ),
        (lambda item: item.update({"honest_verdict": "blocked: wrong"}), "honest_verdict"),
        (
            lambda item: item.update(
                {
                    "status": "blocked",
                    "channel_diagnostic_ready_score": 0.0,
                    "honest_verdict": "complete: wrong",
                }
            ),
            "honest_verdict",
        ),
        (lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}), "reproducibility_checksum"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" in bad and match != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)
