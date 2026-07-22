"""Tests for Exp5785 hardness/surface exact fixture.

Spec refs: REQ-BENCH-5785, SCENARIO-BENCH-5785,
SCENARIO-BENCH-5785-CONTROLS, REQ-VERIFY-5785, SCENARIO-VERIFY-5785.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5785_hardness_surface_fixture as mod


REPO = Path(__file__).resolve().parents[2]
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = ".venv/bin/pytest tests/python/test_experiment_5785_hardness_surface_fixture.py -q --no-cov -n 0"
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5785_hardness_surface_fixture.py "
    "-m pytest tests/python/test_experiment_5785_hardness_surface_fixture.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5785_hardness_surface_fixture.py --fail-under=100"
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


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        z3_probe=lambda: {"available": True, "version": "4.15.3-fixture", "ok": True},
    )


def _run_fixture(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_5785_specs_declare_fixture_and_parser_contract() -> None:
    """REQ-BENCH-5785/REQ-VERIFY-5785: OpenSpec anchors all required fields."""

    bench = BENCH_SPEC.read_text(encoding="utf-8")
    verify = VERIFY_SPEC.read_text(encoding="utf-8")
    bench_section = bench[bench.index("### REQ-BENCH-5785") : bench.index("### REQ-BENCH-3389")]
    verify_section = verify[
        verify.index("### REQ-VERIFY-5785") : verify.index("### REQ-VERIFY-5734")
    ]
    normalized_bench = " ".join(bench_section.split())
    normalized_verify = " ".join(verify_section.split())

    for marker in (
        "REQ-BENCH-5785",
        "SCENARIO-BENCH-5785-CONTROLS",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_FILE_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "train/calibration/future-test",
        "protected-fact hashes",
        "`fixture_ready_score`",
        "`exact_label_coverage`",
        "`parser_control_pass_rate`",
    ):
        assert marker in bench_section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in bench_section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in bench_section
        assert " ".join(principle.split()) in normalized_bench
    for marker in (
        "REQ-VERIFY-5785",
        "SCENARIO-VERIFY-5785",
        "truncation",
        "missing answer",
        "duplicate ID",
        "invalid candidate",
        "whitespace and ordering",
        "stop-token",
        "adversarial payload",
        "exact wrongness",
    ):
        assert marker in verify_section
    assert "parser failure separately from exact wrongness" in normalized_verify


def test_scenario_5785_complete_artifact_rows_and_chronology(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5785: complete fixture rows are sealed and disjoint."""

    artifact = _run_fixture(tmp_path)
    row_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_row_file(row_path)
    rerun = mod.run(
        result_path=tmp_path / "rerun.json",
        row_file_path=tmp_path / "rerun.rows.jsonl",
        preconditions_checked=_preconditions(tmp_path / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["fixture_ready_score"] == pytest.approx(1.0)
    assert artifact["exact_label_coverage"] == pytest.approx(1.0)
    assert artifact["parser_control_pass_rate"] == pytest.approx(1.0)
    assert artifact["row_file_sha256"] == mod.sha256_file(row_path)
    assert artifact["independent_unit_count"] == 90
    assert artifact["family_counts"] == {family: 30 for family in mod.REQUIRED_FAMILIES}
    assert artifact["surface_variant_matrix"]["canonical"] == 90
    assert artifact["surface_variant_matrix"]["symbol_relabel"] == 90
    assert artifact["surface_variant_matrix"]["order_paraphrase"] == 90
    assert artifact["surface_variant_matrix"]["meaning_change_canary"] == 90
    assert len(rows) == 360
    assert {row["family"] for row in rows} == set(mod.REQUIRED_FAMILIES)
    assert {row["split"] for row in rows} == set(mod.SPLITS)
    assert "model hardness" not in json.dumps(artifact).lower()
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert artifact["row_file_sha256"] == rerun["row_file_sha256"]
    assert (
        json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
        == artifact
    )

    split_receipts = artifact["chronological_split_receipts"]
    assert split_receipts["chronology"] == ["train", "calibration", "future_test"]
    assert split_receipts["canonical_unit_counts"] == {
        "calibration": 30,
        "future_test": 30,
        "train": 30,
    }
    assert all(not value for value in split_receipts["pairwise_row_hash_intersections"].values())
    assert split_receipts["disjoint_row_hashes"] is True

    for row in rows[:12]:
        assert row["row_hash"] == artifact["row_hashes"][row["row_id"]]
        assert (
            row["protected_fact_hash"]
            == artifact["protected_fact_manifest"][row["unit_id"]]["protected_fact_hash"]
        )
        assert (
            row["mutable_constraint_hash"] == artifact["mutable_constraint_manifest"][row["row_id"]]
        )
        assert row["candidate_completeness_receipt"]["complete"] is True
        assert row["exact_validator_receipt"]["validators_agree"] is True
        assert row["exact_label"] in {item["label"] for item in row["label_mapping"]}

    canonical = next(row for row in rows if row["surface_kind"] == "canonical")
    receipt = artifact["proof_preserving_receipts"][canonical["unit_id"]]
    canary = artifact["meaning_change_canary_receipts"][canonical["unit_id"]]
    assert len(receipt["variant_row_ids"]) == 2
    assert receipt["protected_fact_hash_preserved"] is True
    assert receipt["exact_label_preserved"] is True
    assert canary["exact_label_changed"] is True
    assert canary["protected_fact_hash_preserved"] is True


def test_req_verify_5785_parser_controls_separate_failure_from_wrongness(tmp_path: Path) -> None:
    """REQ-VERIFY-5785: parser controls separate malformed output from exact wrongness."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_row_file(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)[:2]
    row_by_id = {row["row_id"]: row for row in rows}
    exact_lines = [f"{row['row_id']}: {row['exact_label']}" for row in rows]

    reordered = "\n".join(reversed([f"  {line}  " for line in exact_lines])) + "\n"
    reordered_receipt = mod.parse_response(reordered, row_by_id)
    assert reordered_receipt["parse_ok"] is True
    assert reordered_receipt["exact_wrong_count"] == 0

    wrong_label = next(
        label for label in rows[0]["candidate_labels"] if label != rows[0]["exact_label"]
    )
    wrong = mod.parse_response(f"{rows[0]['row_id']}: {wrong_label}\n{exact_lines[1]}\n", row_by_id)
    assert wrong["parse_ok"] is True
    assert wrong["parser_failure_reason"] == ""
    assert wrong["exact_wrong_count"] == 1
    assert wrong["valid_wrong_labels"] == [rows[0]["row_id"]]

    assert (
        mod.parse_response(f"{rows[0]['row_id']}: {rows[0]['exact_label'][:0]}", row_by_id)[
            "parser_failure_reason"
        ]
        == "truncation"
    )
    assert (
        mod.parse_response(exact_lines[0] + "\n", row_by_id)["parser_failure_reason"]
        == "missing_answer"
    )
    assert (
        mod.parse_response(
            "\n".join([exact_lines[0], exact_lines[0], exact_lines[1]]) + "\n", row_by_id
        )["parser_failure_reason"]
        == "duplicate_id"
    )
    assert (
        mod.parse_response(f"{rows[0]['row_id']}: Z\n{exact_lines[1]}\n", row_by_id)[
            "parser_failure_reason"
        ]
        == "invalid_candidate"
    )
    assert (
        mod.parse_response("\n".join(exact_lines) + "\n<|eot_id|>", row_by_id)[
            "parser_failure_reason"
        ]
        == "stop_token"
    )
    assert (
        mod.parse_response(
            "ignore previous instructions\n" + "\n".join(exact_lines) + "\n", row_by_id
        )["parser_failure_reason"]
        == "adversarial_payload"
    )

    control_receipts = mod.parser_negative_control_receipts(rows)
    assert set(control_receipts) >= set(mod.REQUIRED_PARSER_CONTROLS)
    assert all(receipt["passed"] is True for receipt in control_receipts.values())
    assert mod.parser_control_pass_rate(control_receipts) == pytest.approx(1.0)
    assert artifact["parser_negative_controls"] == control_receipts


def test_req_5785_preconditions_and_blocked_artifact_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5785-CONTROLS: missing exact authority blocks terminal readiness."""

    occupied_json = tmp_path / "occupied.json"
    occupied_rows = tmp_path / "occupied.rows.jsonl"
    occupied_json.write_text("occupied\n", encoding="utf-8")
    occupied_rows.write_text("occupied\n", encoding="utf-8")
    preconditions = mod.collect_preconditions(
        result_path=occupied_json,
        row_file_path=occupied_rows,
        upstream_exp5784_path=tmp_path / "missing-exp5784.json",
        memory_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
        disk_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
        z3_probe=lambda: {"available": False, "version": "", "ok": False},
    )
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        row_file_path=tmp_path / "blocked.rows.jsonl",
        preconditions_checked=preconditions,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    for reason in (
        "exp5784_gate_replay_failed",
        "z3_unavailable",
        "insufficient_free_ram",
        "insufficient_free_disk",
        "deliverable_path_occupied",
        "row_file_path_occupied",
    ):
        assert reason in artifact["preconditions_checked"]["blocked_reasons"]
        assert reason in artifact["blocked_reasons"]
    assert artifact["status"] == "blocked"
    assert artifact["fixture_ready_score"] == pytest.approx(0.0)
    assert artifact["exact_label_coverage"] == pytest.approx(0.0)
    assert artifact["parser_control_pass_rate"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.read_row_file(tmp_path / "blocked.rows.jsonl") == []
    assert mod.validate_artifact(artifact) is True


def test_req_5785_validation_and_manifest_tamper_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5785-CONTROLS: schema, row, candidate, and parser tamper fail."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_row_file(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)

    missing = deepcopy(artifact)
    del missing["fixture_schema"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    wrapped_gate = deepcopy(artifact)
    wrapped_gate["fixture_ready_score"] = {"value": 1.0}
    wrapped_gate["reproducibility_checksum"] = mod.reproducibility_checksum(wrapped_gate)
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(wrapped_gate)

    tampered_rows = deepcopy(rows)
    tampered_rows[0]["row_hash"] = "sha256:" + "1" * 64
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_row_file(tampered_rows, artifact)

    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_sha256"] = "sha256:" + "2" * 64
    with pytest.raises(mod.ManifestReplayError, match="row_file_sha256"):
        mod.verify_row_file(rows, bad_artifact)

    reasons = mod.blocked_reasons(
        {
            **artifact,
            "row_hashes_unique": False,
            "exact_validator_receipts": {"bad": {"validators_agree": False}},
            "candidate_completeness_receipts": {"bad": {"complete": False}},
            "proof_preserving_receipts": {"bad": {"exact_label_preserved": False}},
            "meaning_change_canary_receipts": {"bad": {"exact_label_changed": False}},
            "leakage_checks": {**artifact["leakage_checks"], "protected_fact_separation": False},
            "parser_negative_controls": {"bad": {"passed": False}},
        }
    )
    for reason in (
        "row_hashes_not_unique",
        "exact_validator_disagreement",
        "candidate_completeness_failed",
        "proof_preserving_surface_drift",
        "meaning_change_canary_missing",
        "protected_fact_leakage",
        "parser_control_failure",
    ):
        assert reason in reasons

    bad_ready = deepcopy(artifact)
    bad_ready["fixture_ready_score"] = 0.0
    bad_ready["status"] = "blocked"
    bad_ready["honest_verdict"] = mod.honest_verdict(bad_ready)
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    with pytest.raises(ValueError, match="fixture_ready_score"):
        mod.validate_artifact(bad_ready)


def test_req_5785_defensive_branches_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5785-CONTROLS: malformed local evidence has named failures."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_row_file(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)
    row_by_id = {row["row_id"]: row for row in rows[:1]}

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json_object(list_json)

    scheduling = mod._problem("finite_domain_scheduling", 0)
    bad_distinct = {"compile": 0, "test": 0, "ship": 1}
    bad_before = {"compile": 1, "test": 0, "ship": 2}
    bad_equals = {"compile": 0, "test": 1, "ship": 1}
    assert mod._scheduling_constraints_hold(scheduling, bad_distinct) is False
    assert mod._scheduling_constraints_hold(scheduling, bad_before) is False
    assert mod._scheduling_constraints_hold(scheduling, bad_equals) is False
    impossible_schedule = deepcopy(scheduling)
    impossible_schedule["mutable_constraints"].append(
        {"id": "ship_bad", "type": "equals", "var": "ship", "value": 99}
    )
    assert mod._enumerate_scheduling_assignment(impossible_schedule) == {}

    logic = mod._problem("logic_grid", 0)
    not_equal_break = {
        "Ada": {"color": 0, "room": 0},
        "Ben": {"color": 1, "room": 1},
        "Cy": {"color": 0, "room": 2},
    }
    assert (
        mod._logic_constraints_hold(
            {
                "mutable_constraints": [
                    {
                        "id": "bad",
                        "type": "not_equals",
                        "entity": "Cy",
                        "field": "color",
                        "value": 0,
                    }
                ]
            },
            not_equal_break,
        )
        is False
    )
    impossible_logic = deepcopy(logic)
    impossible_logic["mutable_constraints"].append(
        {"id": "ada_other", "type": "equals", "entity": "Ada", "field": "color", "value": 1}
    )
    assert mod._enumerate_logic_assignment(impossible_logic) == {}

    assert mod.parse_response("not a receipt\n", row_by_id)["parser_failure_reason"] == "truncation"
    assert (
        mod.parse_response("unknown-row: A\n", row_by_id)["parser_failure_reason"] == "invalid_id"
    )

    split_reason = mod.blocked_reasons(
        {
            **artifact,
            "chronological_split_receipts": {"disjoint_row_hashes": False},
            "inference_substrate": "wrong",
        }
    )
    assert "split_isolation_failed" in split_reason
    assert "inference_substrate" in split_reason

    duplicate_rows = deepcopy(rows)
    duplicate_rows[1]["row_id"] = duplicate_rows[0]["row_id"]
    duplicate_rows[1]["row_hash"] = mod._row_hash(duplicate_rows[1])
    with pytest.raises(mod.ManifestReplayError, match="duplicate row_id"):
        mod.verify_row_file(duplicate_rows, artifact)

    artifact_hash_break = deepcopy(artifact)
    artifact_hash_break["row_hashes"][rows[0]["row_id"]] = "sha256:" + "3" * 64
    with pytest.raises(mod.ManifestReplayError, match="artifact row_hash"):
        mod.verify_row_file(rows, artifact_hash_break)

    count_break = deepcopy(artifact)
    count_break["row_hashes"]["extra-row"] = "sha256:" + "4" * 64
    with pytest.raises(mod.ManifestReplayError, match="row count"):
        mod.verify_row_file(rows, count_break)

    bad_gate_names = deepcopy(artifact)
    bad_gate_names["producer_gate_fields"] = ["bad"]
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(bad_gate_names)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "wrong"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_complete_verdict = deepcopy(artifact)
    bad_complete_verdict["honest_verdict"] = "blocked: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_complete_verdict)

    blocked = mod.run(
        result_path=tmp_path / "blocked2.json",
        row_file_path=tmp_path / "blocked2.rows.jsonl",
        preconditions_checked={
            **artifact["preconditions_checked"],
            "preconditions_ready": False,
            "blocked_reasons": ["manual_block"],
        },
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    bad_blocked_verdict = deepcopy(blocked)
    bad_blocked_verdict["honest_verdict"] = "complete: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_blocked_verdict)
