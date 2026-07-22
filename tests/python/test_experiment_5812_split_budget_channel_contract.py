"""Tests for Exp5812 split-budget channel contract.

Spec refs: REQ-VERIFY-5812, SCENARIO-VERIFY-5812-CONTRACT,
SCENARIO-VERIFY-5812-CONTROLS, SCENARIO-VERIFY-5812-GRAMMAR-BOUNDARY,
SCENARIO-VERIFY-5812-REPLAY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5812_split_budget_channel_contract as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5812_split_budget_channel_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5812_split_budget_channel_contract.py "
    "-m pytest tests/python/test_experiment_5812_split_budget_channel_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5812_split_budget_channel_contract.py "
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
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _row() -> dict:
    return next(
        row
        for row in fixture.generate_fixture_rows()
        if row["surface_kind"] == "canonical" and row["split"] == "train"
    )


def _other_row() -> dict:
    rows = fixture.generate_fixture_rows()
    first = _row()["row_id"]
    return next(row for row in rows if row["row_id"] != first)


def test_req_verify_5812_spec_declares_split_budget_contract() -> None:
    """REQ-VERIFY-5812: OpenSpec anchors fields, principles, and readiness gates."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5812") : text.index("### REQ-VERIFY-5734")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5812",
        "SCENARIO-VERIFY-5812-CONTRACT",
        "SCENARIO-VERIFY-5812-CONTROLS",
        "SCENARIO-VERIFY-5812-GRAMMAR-BOUNDARY",
        "SCENARIO-VERIFY-5812-REPLAY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`split_budget_contract_ready_score=1.0`",
        "Hugging Face `AutoTokenizer`",
        "without modifying GGUF chat templates",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5812_contract_ready_artifact_and_write(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5812-CONTRACT: split stages are independent and leak-free."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )
    written = mod.build_and_write_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.validate_artifact(written) is True
    assert written["split_budget_contract_ready_score"] == 1.0
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == written
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["split_budget_contract_ready_score"] == 1.0
    assert artifact["llm_calls_made"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["headline_model_loaded"] is False
    assert artifact["preconditions_checked"]["autotokenizer_used_on_gguf"] is False
    assert artifact["preconditions_checked"]["gguf_templates_modified"] is False
    assert artifact["preconditions_checked"]["disk"]["available_mb"] >= 0
    assert artifact["preconditions_checked"]["memory"]["available_mb"] >= 0
    assert set(artifact["contract_version_and_code_hashes"]["hashed_inputs"]) >= {
        "exp5811_audit",
        "sealed_fixture_artifact",
        "sealed_fixture_rows",
        "sealed_fixture_parser",
        "current_channel_producer",
        "embedded_template_metadata_fixture",
        "split_budget_tests",
    }
    assert artifact["reasoning_stage_contract"]["call_ordinal"] == 1
    assert artifact["finalization_stage_contract"]["call_ordinal"] == 2
    assert artifact["reasoning_stage_contract"]["max_tokens"] != artifact["finalization_stage_contract"]["max_tokens"]
    assert artifact["reasoning_stage_contract"]["budget_accounting"]["measured_separately"] is True
    assert artifact["finalization_stage_contract"]["budget_accounting"]["measured_separately"] is True
    assert artifact["finalization_stage_contract"]["hidden_label_leakage_detected"] is False
    modes = artifact["preregistered_mode_matrix"]
    assert any(mode["mode_type"] == "shared_budget_control" for mode in modes)
    assert sum(mode["mode_type"] == "split_budget" for mode in modes) >= 2
    assert all(mode["retirement_rules_preregistered"] is True for mode in modes)
    positive = artifact["replay_receipts"]["positive_control"]
    assert positive["replay_ok"] is True
    assert positive["call_count"] == 2
    assert positive["budget_accounting"]["shared_budget_used"] is False
    assert positive["finalization"]["parse_ok"] is True
    assert positive["finalization"]["valid_exact_output"] is True
    assert positive["prompt_leakage"]["hidden_label_leakage_detected"] is False


def test_scenario_verify_5812_adversarial_controls_fail_closed() -> None:
    """SCENARIO-VERIFY-5812-CONTROLS: malformed controls never become successes."""

    controls = mod.adversarial_control_results(_row(), _other_row())

    assert set(controls) == set(mod.EXPECTED_ADVERSARIAL_CONTROLS)
    assert all(receipt["passed"] is True for receipt in controls.values())
    assert controls["empty_reasoning"]["failure_mode"] == "empty_reasoning"
    assert controls["empty_final"]["parser_failure_reason"] == "missing_answer"
    assert controls["overlong_reasoning"]["failure_mode"] == "reasoning_truncation"
    assert controls["stop_collision"]["failure_mode"] == "stop_collision"
    assert controls["unclosed_thinking"]["failure_mode"] == "unclosed_thinking"
    assert controls["duplicate_candidate_id"]["parser_failure_reason"] == "duplicate_id"
    assert controls["invalid_candidate_id"]["parser_failure_reason"] == "invalid_candidate_id"
    assert controls["ghost_candidate_id"]["parser_failure_reason"] == "ghost_candidate_id"
    assert controls["schema_control_plane_injection"]["parser_failure_reason"] == "adversarial_payload"
    assert controls["candidate_label_leakage"]["hidden_label_leakage_detected"] is True
    assert controls["timeout"]["failure_mode"] == "timeout"
    assert controls["replay_mismatch"]["replay_ok"] is False
    assert controls["exact_wrong_answer"]["parse_ok"] is True
    assert controls["exact_wrong_answer"]["exact_answer_error"] is True
    assert controls["exact_wrong_answer"]["valid_exact_output"] is False


def test_scenario_verify_5812_grammar_boundary_is_finite_id_syntax_only() -> None:
    """SCENARIO-VERIFY-5812-GRAMMAR-BOUNDARY: grammar support is not truth."""

    row = _row()
    env = mod.build_candidate_environment(row)
    supported = mod.environment_indexed_grammar_receipt(env, runtime_supports=True)
    unsupported = mod.environment_indexed_grammar_receipt(env, runtime_supports=False)
    wrong_id = next(
        candidate_id
        for candidate_id, candidate in env["candidate_by_id"].items()
        if candidate["label"] != row["exact_label"]
    )
    parsed_wrong = mod.classify_finalization_stage(
        row,
        env,
        f"{row['row_id']}: {wrong_id}",
        finish_reason="stop",
        output_tokens=2,
        config=mod.SPLIT_BUDGET_MODES[0]["finalization"],
        timeout=False,
    )

    assert supported["runtime_supports_environment_indexed_grammar"] is True
    assert supported["enforced_candidate_ids"] == env["candidate_ids"]
    assert supported["claim_boundary"] == mod.GRAMMAR_CLAIM_BOUNDARY
    assert supported["semantic_correctness_claimed"] is False
    assert unsupported["runtime_supports_environment_indexed_grammar"] is False
    assert unsupported["enforced_candidate_ids"] == []
    assert unsupported["parser_remains_fail_closed"] is True
    assert parsed_wrong["grammar_membership_ok"] is True
    assert parsed_wrong["parse_ok"] is True
    assert parsed_wrong["exact_answer_error"] is True
    assert parsed_wrong["valid_exact_output"] is False


def test_scenario_verify_5812_replay_and_artifact_validation_fail_closed() -> None:
    """SCENARIO-VERIFY-5812-REPLAY: drift in receipts or artifact gates is rejected."""

    row = _row()
    mode = mod.SPLIT_BUDGET_MODES[0]
    receipt = mod.positive_control_receipt(row, mode, runtime_supports_grammar=True)
    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert mod.replay_split_receipt(receipt) is True
    for mutate, match in (
        (
            lambda item: item["reasoning"].update({"raw_text": "tampered"}),
            "reasoning_transcript_hash",
        ),
        (
            lambda item: item["finalization"].update({"raw_text": "tampered"}),
            "final_raw_sha256",
        ),
        (
            lambda item: item["finalization"].update({"candidate_environment_hash": "sha256:bad"}),
            "candidate_environment_hash",
        ),
        (
            lambda item: item["finalization"].update({"reasoning_transcript_hash": "sha256:bad"}),
            "finalizer_reasoning_hash",
        ),
        (
            lambda item: item["finalization"].update({"parser_receipt_hash": "sha256:bad"}),
            "parser_receipt_hash",
        ),
        (
            lambda item: item.update({"receipt_hash": "sha256:bad"}),
            "receipt_hash",
        ),
    ):
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(mod.SplitBudgetReplayError, match=match):
            mod.replay_split_receipt(bad)

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required artifact fields"),
        (lambda item: item.update({"inference_substrate": "live_llm_inference"}), "inference_substrate"),
        (lambda item: item.update({"llm_calls_made": 1}), "llm_calls_made"),
        (
            lambda item: item.update({"split_budget_contract_ready_score": 0.0}),
            "split_budget_contract_ready_score",
        ),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance"),
        (lambda item: item.update({"honest_verdict": "ready"}), "honest_verdict"),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad_artifact = deepcopy(artifact)
        mutate(bad_artifact)
        if "reproducibility_checksum" in bad_artifact and match != "reproducibility_checksum":
            bad_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(bad_artifact)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad_artifact)


def test_scenario_verify_5812_parser_and_precondition_edge_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-5812-CONTROLS: parser and precondition edge cases are explicit."""

    row = _row()
    env = mod.build_candidate_environment(row)
    exact_id = env["label_to_candidate_id"][row["exact_label"]]
    config = mod.SPLIT_BUDGET_MODES[0]["finalization"]
    valid = f"{row['row_id']}: {exact_id}"

    assert mod.prompt_leakage_scan(row, f"exact answer {row['exact_answer']}")[
        "hidden_label_leakage_detected"
    ] is True
    assert mod.classify_finalization_stage(
        row,
        env,
        f"{valid} <stop>",
        finish_reason="stop",
        output_tokens=2,
        config=config,
    )["failure_mode"] == "stop_collision"
    assert mod.classify_finalization_stage(
        row,
        env,
        "not a final line",
        finish_reason="stop",
        output_tokens=2,
        config=config,
    )["parser_failure_reason"] == "truncation"
    assert mod.classify_finalization_stage(
        row,
        env,
        f"{row['row_id']}: ",
        finish_reason="stop",
        output_tokens=2,
        config=config,
    )["parser_failure_reason"] == "truncation"
    assert mod.classify_finalization_stage(
        row,
        env,
        f"wrong-row: {exact_id}",
        finish_reason="stop",
        output_tokens=2,
        config=config,
    )["parser_failure_reason"] == "invalid_id"
    assert mod.classify_finalization_stage(
        row,
        env,
        valid,
        finish_reason="stop",
        output_tokens=2,
        config=config,
        timeout=True,
    )["failure_mode"] == "timeout"
    assert mod.classify_finalization_stage(
        row,
        env,
        valid,
        finish_reason="length",
        output_tokens=config["max_tokens"],
        config=config,
    )["failure_mode"] == "final_truncation"

    monkeypatch.setattr(mod, "_input_hashes", lambda root: {"missing_dep": "missing"})
    monkeypatch.setattr(mod, "_structured_gate_replay", lambda root: {"all_passed": False})
    monkeypatch.setattr(
        mod,
        "_memory_probe",
        lambda: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    monkeypatch.setattr(
        mod,
        "_disk_probe",
        lambda root: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    preconditions = mod.collect_preconditions(REPO)

    assert preconditions["preconditions_ready"] is False
    assert preconditions["blocked_reasons"] == [
        "structured_gate_replay_failed",
        "missing_hashed_inputs:missing_dep",
        "insufficient_free_ram",
        "insufficient_free_disk",
    ]
