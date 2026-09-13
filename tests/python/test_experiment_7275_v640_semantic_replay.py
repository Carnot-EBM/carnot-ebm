"""Tests for the V640 independent semantic replay and comparator contract.

Spec refs: REQ-VERIFY-7275 and SCENARIO-VERIFY-7275-*.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil

import pytest

from carnot import experiment_7275_v640_semantic_replay as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, object]]:
    """Provide complete validation evidence without starting recursive pytest."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.001,
            "log_path": f"results/raw/experiment_7275/validation/{name}.log",
            "log_sha256": exp.sha256_bytes(b"fixture validation"),
        }
        for name in exp.REQUIRED_VALIDATION_NAMES
    ]


def test_req_verify_7275_contract_and_no_llm_schema() -> None:
    """REQ-VERIFY-7275 fixes ordinary fields and the no-LLM execution class."""

    assert "REQ-VERIFY-7275" in (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    artifact = exp.base_artifact(exp.RUN_DATE)
    assert artifact["schema"] == "carnot.exp7275.v640_semantic_replay.v1"
    assert artifact["experiment_id"] == "exp7275-semantic-replay"
    assert artifact["milestone"] == "2026.09.640"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    with pytest.raises(argparse.ArgumentTypeError):
        exp._date_argument("20260912")
    assert exp._date_argument(exp.RUN_DATE) == exp.RUN_DATE


def test_scenario_verify_7275_replays_every_historical_row_exactly() -> None:
    """SCENARIO-VERIFY-7275-REPLAY preserves all raw and semantic outcomes."""

    replay = exp.load_and_replay(ROOT)
    assert replay["reconstruction_errors"] == []
    assert len(replay["call_replay_rows"]) == 320
    assert len(replay["semantic_rows"]) == 192
    assert replay["semantic_rows"] == replay["stored_semantic_rows"]
    assert all(row["historical_reconstruction_match"] for row in replay["call_replay_rows"])
    assert all(row["request_bytes_match"] for row in replay["call_replay_rows"])
    assert all(row["response_bytes_match"] for row in replay["call_replay_rows"])
    reductions = replay["historical_arm_reduction"]
    assert reductions["mention_pointer"] == {
        "rows": 64,
        "correct": 63,
        "invalid": 1,
        "abstained": 17,
    }
    assert reductions["direct_judge"] == {
        "rows": 64,
        "correct": 0,
        "invalid": 64,
        "abstained": 0,
    }
    assert reductions["explicit_schema_offset_control"] == {
        "rows": 64,
        "correct": 16,
        "invalid": 64,
        "abstained": 64,
    }


def test_scenario_verify_7275_locates_both_replayed_contract_defects() -> None:
    """SCENARIO-VERIFY-7275-DIVERGENCE names the first producer/consumer split."""

    replay = exp.load_and_replay(ROOT)
    rows = replay["first_divergence_rows"]
    assert len(rows) == 320
    direct = [row for row in rows if row["arm"] == "direct_judge"]
    extraction = [row for row in rows if row["arm"] != "direct_judge"]
    assert len(direct) == 64 and len(extraction) == 256
    assert {row["cause"] for row in rows} == {"contract_mismatch"}
    assert {row["field"] for row in direct} == {"decision"}
    assert all("_direct_grammar" in row["producer_function"] for row in direct)
    assert all("_direct_shape_valid" in row["consumer_function"] for row in direct)
    assert {row["field"] for row in extraction} == {"row_sha256"}
    assert all("build_completion_row" in row["producer_function"] for row in extraction)
    assert all("independent_replay" in row["consumer_function"] for row in extraction)
    diagnostics = replay["corrected_direct_diagnostics"]
    assert len(diagnostics) == 64
    assert all(row["mapping_authorized_by_request"] for row in diagnostics)
    assert all(row["historical_parse_valid"] is False for row in diagnostics)
    assert {row["diagnostic_decision"] for row in diagnostics} == {
        "supported",
        "contradicted",
        "unknown",
    }


def test_scenario_verify_7275_native_fixtures_fail_closed() -> None:
    """SCENARIO-VERIFY-7275-FIXTURES accepts labels and rejects corrupt controls."""

    receipt = exp.run_fixture_matrix()
    rows = {row["fixture_id"]: row for row in receipt["rows"]}
    assert set(rows) == {
        "direct_supported",
        "direct_contradicted",
        "direct_unknown",
        "malformed_output",
        "explicit_unknown",
        "unicode_request",
        "duplicate_mentions",
        "reversed_relation",
        "changed_request_bytes",
        "token_truncation",
    }
    assert all(row["passed"] for row in rows.values())
    assert rows["direct_supported"]["decision"] == "supported"
    assert rows["direct_contradicted"]["decision"] == "contradicted"
    assert rows["direct_unknown"]["decision"] == "unknown"
    assert rows["unicode_request"]["request_bytes_match"] is True
    assert rows["reversed_relation"]["classification"] == "real_model_error"
    assert rows["changed_request_bytes"]["classification"] == "corruption"
    assert rows["malformed_output"]["classification"] == "parser_rejection"
    assert rows["duplicate_mentions"]["accepted"] is False
    assert rows["token_truncation"]["accepted"] is False
    assert receipt["corrupted_controls_total"] == receipt["corrupted_controls_rejected"]
    assert receipt["unknown_cause_count"] == 0


def test_scenario_verify_7275_seals_independent_comparator_contract() -> None:
    """SCENARIO-VERIFY-7275-COMPARATOR fixes full labels and tie-to-unknown."""

    replay = exp.load_and_replay(ROOT)
    contract = exp.build_comparator_contract(replay["historical_identity"])
    assert contract["decision_values"] == ["supported", "contradicted", "unknown"]
    assert contract["draw_count"] == 2
    assert contract["draw_token_budgets"] == [512, 512]
    assert contract["tie_rule"] == "unknown"
    assert contract["public_only_prompts"] is True
    assert contract["embedded_chat_template"]["required"] is True
    assert contract["embedded_chat_template"]["sha256"].startswith("sha256:")
    assert contract["grammar_forwarding"] == "exact_requested_bytes"
    assert contract["mention_method"]["changed"] is False
    assert exp.reduce_two_draws("supported", "supported") == "supported"
    assert exp.reduce_two_draws("supported", "contradicted") == "unknown"
    with pytest.raises(ValueError, match="decision"):
        exp.reduce_two_draws("a", "supported")


def test_scenario_verify_7275_e2e_mutation_blocks_and_artifact_validates(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7275-E2E rejects input mutation before terminal evidence."""

    checks = exp.authenticate_inputs(ROOT)
    assert checks and all(row["passed"] for row in checks)
    changed_hashes = deepcopy(exp.PINNED_INPUT_HASHES)
    changed_hashes[exp.SOURCE_ARTIFACT_PATH] = exp.sha256_bytes(b"mutated")
    failed = exp.authenticate_inputs(ROOT, expected_hashes=changed_hashes)
    assert any(not row["passed"] for row in failed)
    blocked = exp.finalize_blocked_artifact(exp.base_artifact(exp.RUN_DATE), failed, 0.1)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "authenticated_input"
    assert exp.validate_artifact(blocked) == []

    replay = exp.load_and_replay(ROOT)
    sidecars = exp.write_sidecars(tmp_path, replay)
    artifact = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        checks,
        replay,
        sidecars,
        _passing_receipts(),
        duration_s=0.2,
        started_at_utc="2026-09-13T00:00:00Z",
        completed_at_utc="2026-09-13T00:00:01Z",
    )
    assert artifact["semantic_replay_ready_score"] == 1
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["rows"] == replay["semantic_rows"]
    assert artifact["verifier_is_oracle"] is True
    assert exp.validate_artifact(artifact) == []
    broken = deepcopy(artifact)
    broken["semantic_replay_ready_score"] = 0
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert "semantic_replay_ready_score" in exp.validate_artifact(broken)
    broken_checksum = deepcopy(artifact)
    broken_checksum["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum" in exp.validate_artifact(broken_checksum)


def test_scenario_verify_7275_thin_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7275-E2E keeps the executable wrapper free of logic."""

    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert raised.value.code == 0


def test_req_verify_7275_validator_rejects_non_mapping() -> None:
    """REQ-VERIFY-7275 keeps malformed terminal candidates out of results."""

    assert exp.validate_artifact(None) == ["artifact_mapping"]
    assert exp.validate_artifact({}) == ["missing_required_field:schema"]
    pending = exp.base_artifact(exp.RUN_DATE)
    pending["reproducibility_checksum"] = exp.artifact_checksum(pending)
    assert "status" in exp.validate_artifact(pending)


def test_req_verify_7275_defensive_decoders_and_manifest_walk() -> None:
    """REQ-VERIFY-7275 rejects malformed bytes, schemas, and retired IDs."""

    assert exp._manifest_lists_experiment({"experiment_id": 7275}, 7275) is True
    assert exp._manifest_lists_experiment({"experiment_ids": [7275]}, 7275) is True
    assert exp._manifest_lists_experiment([{"nested": 1}], 7275) is False
    with pytest.raises(ValueError, match="base64_type"):
        exp._decode_b64(None)
    with pytest.raises(ValueError, match="base64_invalid"):
        exp._decode_b64("***")
    with pytest.raises(ValueError, match="response_contract"):
        exp._response_content(b"not json")
    duplicate_choices = {
        "choices": [
            {"message": {"content": "one"}},
            {"message": {"content": "two"}},
        ]
    }
    with pytest.raises(ValueError, match="response_contract"):
        exp._response_content(json.dumps(duplicate_choices).encode())
    request = base64.b64decode(
        json.loads((ROOT / exp.SOURCE_RAW_DIR / "call_04.json").read_text())["completion"][
            "raw_request_bytes_b64"
        ]
    )
    invalid_direct = exp._reduce_direct_fixture(request, request, '{"decision":"a"}', "stop")
    assert invalid_direct["accepted"] is False
    mentions = [{"mention_id": "m000"}, {"mention_id": "m001"}]
    expected = {"subject_pointer": "m000", "object_pointer": "m001"}
    assert exp._relation_fixture(mentions, "bad", expected)["accepted"] is False
    assert exp._relation_fixture(mentions, '{"relations":[]}', expected)["accepted"] is False
    invalid_pointer = '{"relations":[{"subject_pointer":"bad","object_pointer":"m001"}]}'
    assert exp._relation_fixture(mentions, invalid_pointer, expected)["accepted"] is False


def test_scenario_verify_7275_replay_reports_authenticated_mutation_details(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7275-E2E retains every mismatch in a private mutated copy."""

    root = tmp_path / "repo"
    for relative in exp.PINNED_INPUT_HASHES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for index in range(320):
        relative = exp.SOURCE_RAW_DIR / f"call_{index:02d}.json"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)

    source_path = root / exp.SOURCE_ARTIFACT_PATH
    source = json.loads(source_path.read_text())
    source["schedule"][0]["seed"] = 0
    source["rows"] = []
    source_path.write_text(json.dumps(source))
    call_path = root / exp.SOURCE_RAW_DIR / "call_00.json"
    call = json.loads(call_path.read_text())
    call["schedule"]["changed"] = True
    call["completion"]["raw_request_bytes_b64"] = "***"
    call["completion"]["raw_completion"] = "changed"
    call_path.write_text(json.dumps(call))

    replay = exp.load_and_replay(root)
    assert {
        "source_schedule_mismatch",
        "call_0:schedule",
        "call_0:historical_reconstruction",
        "call_0:request_reconstruction",
        "call_0:response_reconstruction",
        "semantic_rows_mismatch",
    } <= set(replay["reconstruction_errors"])


def test_req_verify_7275_validator_names_all_terminal_contract_failures(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7275 reports independent terminal-shape failures together."""

    replay = exp.load_and_replay(ROOT)
    artifact = exp.finalize_measured_artifact(
        exp.base_artifact(exp.RUN_DATE),
        exp.authenticate_inputs(ROOT),
        replay,
        exp.write_sidecars(tmp_path, replay),
        _passing_receipts(),
        duration_s=0.2,
        started_at_utc="2026-09-13T00:00:00Z",
        completed_at_utc="2026-09-13T00:00:01Z",
    )
    broken = deepcopy(artifact)
    broken.update(
        {
            "schema": "wrong",
            "experiment_id": "wrong",
            "field_principles": {},
            "MODEL_SPECS": [{"model": "forbidden"}],
            "inference_substrate": "wrong",
            "verifier_is_oracle": False,
            "duration_s": -1,
            "rows": [],
            "honest_verdict": "complete_positive_wrong",
            "verdict_class": "positive",
            "validation_receipts": [],
            "acceptance_gate_results": [],
            "comparator_contract_path": {},
        }
    )
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert {
        "schema",
        "identity",
        "field_principles",
        "no_llm_contract",
        "execution_contract",
        "verifier_is_oracle",
        "duration_s",
        "denominators",
        "verdict",
        "validation_receipts",
        "acceptance_gate_results",
        "comparator_contract_path",
    } <= set(exp.validate_artifact(broken))
    invalid_block = exp.finalize_blocked_artifact(
        exp.base_artifact(exp.RUN_DATE),
        [exp._gate_row("x", 1, 0, False, upstream="u", field="f")],
        0.1,
    )
    invalid_block["verdict_class"] = "null"
    invalid_block["reproducibility_checksum"] = exp.artifact_checksum(invalid_block)
    assert "blocked_terminal_state" in exp.validate_artifact(invalid_block)
