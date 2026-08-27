"""Task-owned readiness tests for the Exp6675 triggered-tail receipt.

Spec refs: REQ-REPORT-6675, SCENARIO-REPORT-6675-OWNED-READY,
SCENARIO-REPORT-6675-GLOBAL-DIAGNOSTIC,
SCENARIO-REPORT-6675-FAIL-CLOSED, and
SCENARIO-REPORT-6675-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6675_triggered_tail_scope_receipt as exp


REPO = Path(__file__).resolve().parents[2]


def _node_set() -> list[str]:
    return [f"{exp.OWNED_NODE_PREFIX}::test_owned_{index:02d}" for index in range(33)]


def _owned_rows() -> list[dict]:
    return [
        exp.make_owned_test_row(
            definition,
            node_set=_node_set(),
            exit_code=0,
            coverage_percent=100.0 if definition["check_id"] == "scoped_coverage" else None,
            duration_s=0.01,
            summary="33 passed" if definition["check_id"] == "focused_tests" else "passed",
            output_sha256=exp.sha256_bytes(b"passed"),
        )
        for definition in exp.OWNED_CHECK_DEFINITIONS
    ]


@pytest.fixture(scope="module")
def replay() -> dict:
    return exp.replay_exp6661_fixture(REPO)


@pytest.fixture(scope="module")
def ready_artifact(replay: dict) -> dict:
    baseline = exp.capture_frozen_snapshot(REPO, replay=replay)
    return exp.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.25,
        owned_test_rows=_owned_rows(),
        global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
        frozen_before=baseline,
        protected_before=exp.protected_hashes(REPO),
        replay=replay,
    )


def test_req_report_6675_spec_and_owned_commands_are_frozen() -> None:
    """REQ-REPORT-6675 freezes the task-owned scope before reduction."""

    spec = exp.REPORT_SPEC_PATH.read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6675",
        "SCENARIO-REPORT-6675-OWNED-READY",
        "SCENARIO-REPORT-6675-GLOBAL-DIAGNOSTIC",
        "SCENARIO-REPORT-6675-FAIL-CLOSED",
        "SCENARIO-REPORT-6675-ATOMIC-PROVENANCE",
    ):
        assert anchor in spec
    assert tuple(row["check_id"] for row in exp.OWNED_CHECK_DEFINITIONS) == (
        "focused_tests",
        "scoped_coverage",
        "ruff_check",
        "format_check",
        "spec_coverage",
    )
    assert [row["command"] for row in exp.OWNED_CHECK_DEFINITIONS] == list(
        exp.EXP6661_OWNED_COMMANDS
    )
    assert exp.INFERENCE_SUBSTRATE == "cpu_fixture_receipt_and_exact_checks_no_llm"


def test_req_report_6675_replays_frozen_fixture_without_rebuilding_corpus(
    replay: dict,
) -> None:
    """REQ-REPORT-6675 reuses Exp6661 builders and matches every frozen row."""

    source = exp.load_json(REPO / exp.EXP6661_ARTIFACT_PATH)
    assert len(replay["manifest"]) == 18
    assert len(replay["arm_contracts"]) == 3
    assert len(replay["fixture_rows"]) == 18
    assert len(replay["exact_checker_rows"]) == 36
    assert len(replay["leakage_attack_rows"]) == 540
    assert replay["aggregate"]["ready"] is True
    assert replay["manifest"] == source["frozen_task_manifest"]
    assert replay["arm_contracts"] == source["arm_contracts"]
    assert replay["fixture_rows"] == source["fixture_rows"]
    assert replay["exact_checker_rows"] == source["exact_checker_rows"]
    assert replay["leakage_attack_rows"] == source["leakage_attack_rows"]
    assert replay["grammar"]["answer_semantics_absent"] is True


def test_scenario_report_6675_global_diagnostic_attributes_every_node(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6675-GLOBAL-DIAGNOSTIC keeps unrelated debt visible."""

    cache = tmp_path / "lastfailed"
    owned = f"{exp.OWNED_NODE_PREFIX}::test_owned"
    unrelated = "tests/python/test_unrelated.py::test_red"
    cache.write_text(json.dumps({owned: True, unrelated: True}), encoding="utf-8")
    diagnostic = exp.load_global_suite_diagnostic(REPO, cache_path=cache)

    assert diagnostic["command"] == exp.FULL_SUITE_COMMAND
    assert diagnostic["failure_count"] == 2
    assert diagnostic["exp6661_owned_failure_count"] == 1
    assert diagnostic["owned_failure_nodes"] == [owned]
    assert diagnostic["unrelated_failure_nodes"] == [unrelated]
    assert diagnostic["gating"] is False
    assert diagnostic["exit_code_receipt_scope"].startswith("Exp6661 recorded run")
    assert diagnostic["cache_sha256"] == exp.sha256_file(cache)
    assert diagnostic["receipt_sha256"] == exp.receipt_hash(
        diagnostic, excluded=("receipt_sha256",)
    )


def test_scenario_report_6675_owned_reducer_fails_closed() -> None:
    """SCENARIO-REPORT-6675-FAIL-CLOSED rejects missing or changed owned rows."""

    clean = _owned_rows()
    failures, summary = exp.reduce_owned_test_rows(clean)
    assert failures == []
    assert summary["ready"] is True
    assert summary["node_count"] == 33
    assert summary["coverage_percent"] == 100.0

    cases = []
    cases.append((clean[:-1], "missing_receipt"))
    failed = deepcopy(clean)
    failed[0] = exp.make_owned_test_row(
        exp.OWNED_CHECK_DEFINITIONS[0],
        node_set=_node_set(),
        exit_code=1,
        coverage_percent=None,
        duration_s=0.01,
        summary="failed",
        output_sha256=exp.sha256_bytes(b"failed"),
    )
    cases.append((failed, "observed_value_mismatch"))
    changed = deepcopy(clean)
    changed[0]["command"] = "wrong"
    changed[0]["receipt_sha256"] = exp.receipt_hash(changed[0], excluded=("receipt_sha256",))
    cases.append((changed, "definition_mismatch"))
    duplicated = clean + [clean[0]]
    cases.append((duplicated, "duplicate_receipt"))
    reordered = deepcopy(clean)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    cases.append((reordered, "receipt_order_mismatch"))

    for rows, reason in cases:
        failures, summary = exp.reduce_owned_test_rows(rows)
        assert summary["ready"] is False
        assert reason in {row["reason"] for row in failures}


def test_scenario_report_6675_ready_artifact_is_owned_and_null(
    ready_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6675-OWNED-READY excludes the red global diagnostic."""

    assert exp.validate_artifact(ready_artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(ready_artifact)
    assert ready_artifact["status"] == "complete_ready"
    assert ready_artifact["honest_verdict"].startswith("complete:")
    assert ready_artifact["verdict_class"] == "null"
    assert ready_artifact["triggered_tail_fixture_ready"] is True
    global_count = ready_artifact["global_suite_diagnostic"]["failure_count"]
    assert global_count >= 3923
    assert ready_artifact["global_suite_diagnostic"]["exp6661_owned_failure_count"] == 0
    assert ready_artifact["global_suite_diagnostic"]["gating"] is False
    assert ready_artifact["aggregate_row_recomputation"]["ready"] is True
    assert ready_artifact["aggregate_row_recomputation"]["counts"] == {
        "arm_contracts": 3,
        "attack_rows": 540,
        "checker_controls": 36,
        "fixture_rows": 18,
        "global_failures": global_count,
        "global_owned_failures": 0,
        "owned_test_nodes": 33,
        "owned_test_rows": 5,
        "per_unit_rows": 627,
        "tasks": 18,
    }
    assert len(ready_artifact["per_unit_rows"]) == 33 + 18 + 36 + 540
    assert {row["row_kind"] for row in ready_artifact["per_unit_rows"]} == {
        "test",
        "task",
        "checker_control",
        "leakage_attack",
    }
    assert ready_artifact["reproducibility_checksum"] == exp.artifact_checksum(ready_artifact)
    assert ready_artifact["field_provenance"]["global_suite_diagnostic"]["sha256"] == (
        ready_artifact["global_suite_diagnostic"]["cache_sha256"]
    )


def test_req_report_6675_attack_matrix_covers_required_boundaries(replay: dict) -> None:
    """REQ-REPORT-6675 retains expected, observed, and receipts for each attack."""

    required = {
        "answer_permutation",
        "label_renaming",
        "grammar_only_generation",
        "trigger_collision",
        "premature_trigger",
        "missing_trigger",
        "malformed_tail",
        "unknown_fields",
        "semantically_wrong_syntactically_valid_tail",
    }
    attacks = replay["leakage_attack_rows"]
    assert required <= {row["attack_type"] for row in attacks}
    assert all(set(row) >= {"expected", "observed", "passed", "row_sha256"} for row in attacks)
    assert all(row["passed"] is True and row["leakage_detected"] is False for row in attacks)


def test_scenario_report_6675_frozen_hashes_and_preconditions_are_measured(
    replay: dict,
) -> None:
    """REQ-REPORT-6675 records stable inputs, resources, and the no-LLM path."""

    baseline = exp.capture_frozen_snapshot(REPO, replay=replay)
    receipts = exp.build_frozen_input_receipts(REPO, baseline, replay)
    preconditions = exp.collect_preconditions(REPO, baseline)
    protected = exp.protected_files_receipt(REPO, exp.protected_hashes(REPO))

    assert receipts["all_hashes_match"] is True
    assert receipts["contract_matches"] == {
        "arm_contracts": True,
        "checker_hashes": True,
        "fixture_rows": True,
        "grammar": True,
        "manifest": True,
        "parser_hashes": True,
    }
    assert all(row["unchanged"] for row in receipts["file_receipts"].values())
    assert preconditions["resources"]["cpu_count"] >= 1
    assert preconditions["resources"]["ram_bytes"] > 0
    assert preconditions["resources"]["disk_free_bytes"] > 0
    assert preconditions["no_llm"] == {
        "declared": exp.INFERENCE_SUBSTRATE,
        "model_load_attempt_count": 0,
        "generation_attempt_count": 0,
        "exact_fixture_replay_only": True,
    }
    assert protected["unchanged"] is True


def test_scenario_report_6675_blocked_results_name_owned_failures(
    replay: dict,
) -> None:
    """SCENARIO-REPORT-6675-FAIL-CLOSED localizes owned defects exactly."""

    baseline = exp.capture_frozen_snapshot(REPO, replay=replay)
    rows = _owned_rows()
    rows[0] = exp.make_owned_test_row(
        exp.OWNED_CHECK_DEFINITIONS[0],
        node_set=_node_set(),
        exit_code=1,
        coverage_percent=None,
        duration_s=0.01,
        summary="one owned failure",
        output_sha256=exp.sha256_bytes(b"red"),
    )
    blocked = exp.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        owned_test_rows=rows,
        global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
        frozen_before=baseline,
        protected_before=exp.protected_hashes(REPO),
        replay=replay,
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["triggered_tail_fixture_ready"] is False
    assert blocked["status"].startswith("blocked_")
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["check"] == "owned_tests"

    global_owned = deepcopy(exp.load_global_suite_diagnostic(REPO))
    global_owned["owned_failure_nodes"] = [f"{exp.OWNED_NODE_PREFIX}::test_red"]
    global_owned["exp6661_owned_failure_count"] = 1
    global_owned["receipt_sha256"] = exp.receipt_hash(global_owned, excluded=("receipt_sha256",))
    blocked = exp.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        owned_test_rows=_owned_rows(),
        global_suite_diagnostic=global_owned,
        frozen_before=baseline,
        protected_before=exp.protected_hashes(REPO),
        replay=replay,
    )
    assert blocked["triggered_tail_fixture_ready"] is False
    assert any(row["check"] == "global_owned_failures" for row in blocked["gate_check_summary"])


def test_scenario_report_6675_validator_rejects_boundary_mutations(
    ready_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6675-ATOMIC-PROVENANCE detects durable mutations."""

    def errors_for(mutator: object) -> list[str]:
        changed = deepcopy(ready_artifact)
        mutator(changed)  # type: ignore[operator]
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert "missing_required_fields" in errors_for(lambda row: row.pop("status"))
    assert "inference_substrate_mismatch" in errors_for(
        lambda row: row.update(inference_substrate="live_llm_inference")
    )
    assert "verifier_is_oracle_mismatch" in errors_for(
        lambda row: row.update(verifier_is_oracle=False)
    )
    assert "owned_test_receipts_invalid" in errors_for(
        lambda row: row["owned_test_rows"][0].update(exit_code=1)
    )
    assert "global_diagnostic_gating" in errors_for(
        lambda row: row["global_suite_diagnostic"].update(gating=True)
    )
    assert "global_diagnostic_count_mismatch" in errors_for(
        lambda row: row["global_suite_diagnostic"].update(failure_count=0)
    )
    assert "global_owned_count_mismatch" in errors_for(
        lambda row: row["global_suite_diagnostic"].update(exp6661_owned_failure_count=7)
    )
    assert "frozen_hash_mismatch" in errors_for(
        lambda row: row["frozen_input_receipts"].update(all_hashes_match=False)
    )
    assert "protected_files_changed" in errors_for(
        lambda row: row["protected_files_unchanged"].update(unchanged=False)
    )
    assert "field_provenance_invalid" in errors_for(
        lambda row: row["field_provenance"].pop("status")
    )
    assert "aggregate_row_recomputation_mismatch" in errors_for(
        lambda row: row["aggregate_row_recomputation"]["counts"].update(tasks=0)
    )
    assert "aggregate_row_recomputation_failed" in errors_for(
        lambda row: row.update(arm_contracts=None)
    )
    assert "honest_verdict_mismatch" in errors_for(lambda row: row.update(honest_verdict="wrong"))
    assert "random_seed_mismatch" in errors_for(lambda row: row.update(random_seed=0))
    assert "duration_invalid" in errors_for(lambda row: row.update(duration_s=-1.0))
    checksum_changed = deepcopy(ready_artifact)
    checksum_changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum_changed)


def test_req_report_6675_owned_command_runner_keeps_nodes_coverage_and_duration(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6675 records the exact focused node set for every command."""

    calls: list[list[str]] = []

    def runner(command: list[str], cwd: Path) -> dict:
        calls.append(command)
        assert cwd == tmp_path
        if "--collect-only" in command:
            output = "\n".join(_node_set()) + "\n\n33 tests collected in 0.01s"
        elif command[:2] == [".venv/bin/coverage", "report"]:
            output = "Name Stmts Miss Cover\nTOTAL 524 0 100%"
        else:
            output = "33 passed in 0.01s"
        return {
            "command": " ".join(command),
            "exit_code": 0,
            "output": output,
            "summary": output.splitlines()[-1],
            "output_sha256": exp.sha256_bytes(output.encode()),
            "duration_s": 0.01,
        }

    rows = exp.run_owned_verification(tmp_path, command_runner=runner)
    assert len(rows) == 5
    assert len(calls) == 6
    assert all(row["node_set"] == _node_set() for row in rows)
    assert rows[1]["coverage_percent"] == 100.0
    assert all(row["passed"] is True for row in rows)
    assert exp.reduce_owned_test_rows(rows)[1]["ready"] is True


def test_req_report_6675_atomic_run_and_cli_validation(
    tmp_path: Path,
    replay: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6675-ATOMIC-PROVENANCE writes and validates one JSON."""

    output = tmp_path / "exp6675.json"
    artifact = exp.run(
        date="20260827",
        root=REPO,
        output_path=output,
        owned_test_rows=_owned_rows(),
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is True

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["errors"] == ["artifact_missing"]
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(unreadable)]) == 1
    assert "artifact_unreadable:JSONDecodeError" in capsys.readouterr().out

    monkeypatch.setattr(exp, "run", lambda **_kwargs: artifact)
    assert exp.main(["--date", "20260827", "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["triggered_tail_fixture_ready"] is True


def test_req_report_6675_helpers_fail_closed_and_measure_process(tmp_path: Path) -> None:
    """REQ-REPORT-6675 keeps missing input and real command outcomes explicit."""

    assert exp.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        exp.load_json(bad_json)
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="expected JSON object"):
        exp.load_json(list_json)
    diagnostic = exp.load_global_suite_diagnostic(REPO, cache_path=tmp_path / "missing-cache")
    assert diagnostic["failure_count"] == 0
    assert diagnostic["cache_read_error"].startswith("FileNotFoundError:")
    row = exp.default_command_runner([exp.sys.executable, "-c", "print('measured')"], tmp_path)
    assert row["exit_code"] == 0
    assert row["summary"] == "measured"
    assert row["duration_s"] >= 0.0
    assert row["output_sha256"].startswith("sha256:")


def test_scenario_report_6675_run_refuses_an_invalid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6675-ATOMIC-PROVENANCE refuses an invalid final write."""

    output = tmp_path / "refused.json"
    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced_invalid"])
    with pytest.raises(ValueError, match="forced_invalid"):
        exp.run(
            date="20260827",
            root=REPO,
            output_path=output,
            owned_test_rows=_owned_rows(),
        )
    assert not output.exists()
