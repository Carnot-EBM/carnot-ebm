"""Task-owned tests for the Exp6683 exact Ising scope receipt.

Spec refs: REQ-REPORT-6683, SCENARIO-REPORT-6683-OWNED-READY,
SCENARIO-REPORT-6683-GLOBAL-DIAGNOSTIC,
SCENARIO-REPORT-6683-FAIL-CLOSED, and
SCENARIO-REPORT-6683-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6683_ising_reference_scope_receipt as exp


REPO = Path(__file__).resolve().parents[2]


def _node_set() -> list[str]:
    existing = [f"{exp.OWNED_NODE_PREFIXES[0]}::test_existing_{index:02d}" for index in range(54)]
    receipt = [
        f"{exp.OWNED_NODE_PREFIXES[1]}::test_receipt_{index:02d}"
        for index in range(exp.EXPECTED_OWNED_NODE_COUNT - 54)
    ]
    return existing + receipt


def _owned_rows() -> list[dict]:
    return [
        exp.make_owned_test_row(
            definition,
            node_set=_node_set(),
            exit_code=0,
            coverage_percent=(100.0 if definition["check_id"] == "scoped_coverage" else None),
            duration_s=0.01,
            summary="passed",
            output_sha256=exp.sha256_bytes(b"passed"),
            spec_anchors=exp.REQUIRED_SPEC_ANCHORS,
        )
        for definition in exp.OWNED_CHECK_DEFINITIONS
    ]


@pytest.fixture(scope="module")
def replay() -> dict:
    return exp.replay_reference()


@pytest.fixture(scope="module")
def ready_artifact(replay: dict) -> dict:
    return exp.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.25,
        owned_test_rows=_owned_rows(),
        global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
        frozen_before=exp.capture_frozen_hashes(REPO),
        protected_before=exp.protected_hashes(REPO),
        replay=replay,
    )


def test_req_report_6683_spec_and_owned_scope_are_frozen() -> None:
    """REQ-REPORT-6683 fixes the owned verification boundary before reduction."""

    spec = exp.REPORT_SPEC_PATH.read_text(encoding="utf-8")
    for anchor in exp.REQUIRED_SPEC_ANCHORS:
        assert anchor in spec
    assert tuple(row["check_id"] for row in exp.OWNED_CHECK_DEFINITIONS) == (
        "focused_tests",
        "scoped_coverage",
        "ruff_check",
        "format_check",
        "spec_coverage",
    )
    assert exp.EXPECTED_OWNED_NODE_COUNT == 66
    assert exp.INFERENCE_SUBSTRATE == "cpu_bounded_treewidth_exact_inference_no_llm"


def test_req_report_6683_replays_every_exact_fixture_and_field(replay: dict) -> None:
    """REQ-REPORT-6683 retains exact state, marginal, correlation, and structure rows."""

    assert len(replay["frozen_fixture_manifest"]) == 15
    assert len(replay["decomposition_rows"]) == 12
    assert len(replay["rejection_rows"]) >= 3
    assert len(replay["exact_probability_rows"]) >= 150
    assert all(row["passed"] is True for row in replay["decomposition_rows"])
    assert all(row["passed"] is True for row in replay["exact_probability_rows"])
    assert all(row["passed"] is True for row in replay["marginal_rows"])
    assert all(row["passed"] is True for row in replay["correlation_rows"])
    assert all(row["passed"] is True for row in replay["rejection_rows"])
    assert all(
        set(row)
        >= {
            "state",
            "energy",
            "unnormalized_weight",
            "partition_function",
            "normalized_probability",
            "node_marginals_plus",
            "pair_correlations",
            "row_sha256",
        }
        for row in replay["exact_probability_rows"]
    )
    assert all(
        set(row) >= {"width", "bags", "separators", "running_intersection", "row_sha256"}
        for row in replay["decomposition_rows"]
    )


def test_req_report_6683_attack_matrix_covers_all_named_boundaries(replay: dict) -> None:
    """REQ-REPORT-6683 attacks topology, numerics, order, normalization, and sampling."""

    attacks = exp.build_attack_rows(replay)
    assert exp.REQUIRED_ATTACKS <= {row["attack_id"] for row in attacks}
    assert {"topology", "temperature", "precision", "normalization", "order"} <= {
        row["category"] for row in attacks
    }
    assert all(row["passed"] is True for row in attacks)
    sample = next(row for row in attacks if row["attack_id"] == "degenerate_exact_sample")
    assert sample["observed"]["replay_equal"] is True
    assert sample["observed"]["unique_state_count"] == 4
    supported = next(item for item in exp.reference.frozen_fixtures() if item.expected_supported)
    assert exp._observed_rejection(supported) == "unexpectedly accepted"

    original = exp.reference.validate_tree_decomposition
    exp.reference.validate_tree_decomposition = lambda *_args: {"valid": True}
    try:
        changed = exp.build_attack_rows(replay)
    finally:
        exp.reference.validate_tree_decomposition = original
    invalid = next(row for row in changed if row["attack_id"] == "invalid_decomposition")
    assert invalid["observed"] == "unexpectedly accepted"
    assert invalid["passed"] is False


def test_scenario_report_6683_global_diagnostic_is_visible_and_non_gating(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6683-GLOBAL-DIAGNOSTIC attributes cached nodes."""

    cache = tmp_path / "lastfailed"
    owned = f"{exp.OWNED_NODE_PREFIXES[0]}::test_owned"
    unrelated = "tests/python/test_unrelated.py::test_red"
    cache.write_text(json.dumps({owned: True, unrelated: True}), encoding="utf-8")
    row = exp.load_global_suite_diagnostic(REPO, cache_path=cache)
    assert row["failure_state"] == "failed"
    assert row["failure_count"] == 2
    assert row["owned_node_count"] == 1
    assert row["owned_failure_nodes"] == [owned]
    assert row["unrelated_failure_nodes"] == [unrelated]
    assert row["gating"] is False
    assert row["readiness_influence"] is False
    assert row["receipt_sha256"] == exp.receipt_hash(row, excluded=("receipt_sha256",))

    malformed = tmp_path / "malformed"
    malformed.write_text("{", encoding="utf-8")
    malformed_row = exp.load_global_suite_diagnostic(REPO, cache_path=malformed)
    assert malformed_row["cache_read_error"].startswith("JSONDecodeError:")
    missing_row = exp.load_global_suite_diagnostic(REPO, cache_path=tmp_path / "missing")
    assert missing_row["cache_read_error"].startswith("FileNotFoundError:")


def test_scenario_report_6683_owned_reducer_fails_closed() -> None:
    """SCENARIO-REPORT-6683-FAIL-CLOSED rejects changed owned receipts."""

    clean = _owned_rows()
    failures, summary = exp.reduce_owned_test_rows(clean)
    assert failures == []
    assert summary["ready"] is True
    assert summary["node_count"] == exp.EXPECTED_OWNED_NODE_COUNT
    assert summary["coverage_percent"] == 100.0

    cases = [(clean[:-1], "missing_receipt")]
    failed = deepcopy(clean)
    failed[0]["exit_code"] = 1
    failed[0]["passed"] = False
    failed[0]["receipt_sha256"] = exp.receipt_hash(failed[0], excluded=("receipt_sha256",))
    cases.append((failed, "observed_value_mismatch"))
    changed = deepcopy(clean)
    changed[0]["command"] = "wrong"
    changed[0]["receipt_sha256"] = exp.receipt_hash(changed[0], excluded=("receipt_sha256",))
    cases.append((changed, "definition_mismatch"))
    cases.append((clean + [clean[0]], "duplicate_receipt"))
    reordered = deepcopy(clean)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    cases.append((reordered, "receipt_order_mismatch"))
    changed_nodes = deepcopy(clean)
    changed_nodes[-1]["node_set"] = changed_nodes[-1]["node_set"][:-1]
    changed_nodes[-1]["receipt_sha256"] = exp.receipt_hash(
        changed_nodes[-1], excluded=("receipt_sha256",)
    )
    cases.append((changed_nodes, "node_set_mismatch"))
    for rows, reason in cases:
        failures, summary = exp.reduce_owned_test_rows(rows)
        assert summary["ready"] is False
        assert reason in {row["reason"] for row in failures}


def test_scenario_report_6683_ready_artifact_is_exact_owned_and_null(
    ready_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6683-OWNED-READY excludes the global diagnostic."""

    assert exp.validate_artifact(ready_artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(ready_artifact)
    assert ready_artifact["status"] == "complete_ready"
    assert ready_artifact["honest_verdict"].startswith("complete:")
    assert ready_artifact["verdict_class"] is None
    assert ready_artifact["ising_reference_ready"] is True
    assert ready_artifact["global_suite_diagnostic"]["gating"] is False
    assert ready_artifact["aggregate_row_recomputation"]["ready"] is True
    assert ready_artifact["aggregate_row_recomputation"]["global_suite_in_reducer"] is False
    assert ready_artifact["numeric_contract"]["coefficient_type"] == "numpy.float64"
    assert ready_artifact["numeric_contract"]["precision_bits"] == 64
    assert ready_artifact["verifier_is_oracle"] is True
    assert ready_artifact["reproducibility_checksum"] == exp.artifact_checksum(ready_artifact)
    assert set(ready_artifact["field_provenance"]) >= set(exp.REQUIRED_ARTIFACT_FIELDS)
    kinds = {row["unit_type"] for row in ready_artifact["per_unit_rows"]}
    assert kinds == {
        "fixture",
        "state",
        "marginal",
        "correlation",
        "rejection",
        "test",
        "attack",
    }


def test_scenario_report_6683_blocked_result_names_owned_failure(replay: dict) -> None:
    """SCENARIO-REPORT-6683-FAIL-CLOSED localizes the failed owned command."""

    rows = _owned_rows()
    rows[0]["exit_code"] = 1
    rows[0]["passed"] = False
    rows[0]["receipt_sha256"] = exp.receipt_hash(rows[0], excluded=("receipt_sha256",))
    blocked = exp.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        owned_test_rows=rows,
        global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
        frozen_before=exp.capture_frozen_hashes(REPO),
        protected_before=exp.protected_hashes(REPO),
        replay=replay,
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["ising_reference_ready"] is False
    assert blocked["status"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["check"] == "owned_tests"

    for key, value, expected in (
        ("status", "complete_ready", "blocked_terminal_state_mismatch"),
        ("verdict_class", None, "blocked_verdict_class_mismatch"),
        ("gate_check_summary", [], "blocked_gate_summary_mismatch"),
    ):
        changed = deepcopy(blocked)
        changed[key] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)


def test_scenario_report_6683_validator_rejects_boundary_mutations(
    ready_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6683-ATOMIC-PROVENANCE detects receipt drift."""

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
    assert "exact_rows_failed" in errors_for(
        lambda row: row["exact_probability_rows"][0].update(passed=False)
    )
    assert "attack_rows_failed" in errors_for(
        lambda row: row["attack_rows"][0].update(passed=False)
    )
    assert "protected_files_changed" in errors_for(
        lambda row: row["protected_files_unchanged"].update(unchanged=False)
    )
    assert "field_provenance_invalid" in errors_for(
        lambda row: row["field_provenance"].pop("status")
    )
    assert "aggregate_row_recomputation_mismatch" in errors_for(
        lambda row: row["aggregate_row_recomputation"]["counts"].update(fixtures=0)
    )
    assert "numeric_contract_mismatch" in errors_for(
        lambda row: row["numeric_contract"].update(precision_bits=32)
    )
    assert "ready_terminal_state_mismatch" in errors_for(
        lambda row: row.update(status="complete_wrong")
    )
    assert "honest_verdict_mismatch" in errors_for(lambda row: row.update(honest_verdict="wrong"))
    assert "ready_gate_summary_mismatch" in errors_for(
        lambda row: row.update(gate_check_summary=[{"check": "wrong"}])
    )
    assert "duration_invalid" in errors_for(lambda row: row.update(duration_s=-1.0))
    checksum_changed = deepcopy(ready_artifact)
    checksum_changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum_changed)


def test_req_report_6683_preconditions_and_frozen_hashes_are_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6683 records resources, libraries, inputs, and no-LLM use."""

    frozen = exp.capture_frozen_hashes(REPO)
    preconditions = exp.collect_preconditions(REPO, frozen)
    protection = exp.protected_files_receipt(REPO, exp.protected_hashes(REPO))
    assert preconditions["resources"]["cpu_count"] >= 1
    assert preconditions["resources"]["ram_bytes"] > 0
    assert preconditions["resources"]["disk_free_bytes"] > 0
    assert preconditions["libraries"]["numpy"] != "missing"
    assert preconditions["no_llm"]["declared"] == exp.INFERENCE_SUBSTRATE
    assert preconditions["no_llm"]["model_load_attempt_count"] == 0
    assert all(value != "missing" for value in frozen.values())
    assert protection["unchanged"] is True
    monkeypatch.setattr(exp.re, "search", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp.platform, "processor", lambda: "fallback-cpu")
    assert exp._cpu_name() == "fallback-cpu"


def test_req_report_6683_owned_command_runner_keeps_nodes_and_coverage(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6683 binds each command to the same collected test nodes."""

    calls: list[list[str]] = []

    def runner(command: list[str], cwd: Path) -> dict:
        calls.append(command)
        assert cwd == tmp_path
        if "--collect-only" in command:
            output = "\n".join(_node_set()) + "\n66 tests collected in 0.01s"
        elif command[:2] == [".venv/bin/coverage", "report"]:
            output = "Name Stmts Miss Cover\nTOTAL 1000 0 100%"
        else:
            output = "66 passed in 0.01s"
        return {
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


def test_scenario_report_6683_atomic_run_and_cli_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6683-ATOMIC-PROVENANCE writes and validates one JSON."""

    output = tmp_path / "exp6683.json"
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
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["ising_reference_ready"] is True


def test_req_report_6683_helpers_and_invalid_run_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6683 keeps malformed inputs and invalid writes explicit."""

    assert exp.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="expected JSON object"):
        exp.load_json(bad)
    command = exp.default_command_runner([exp.sys.executable, "-c", "print('measured')"], tmp_path)
    assert command["exit_code"] == 0
    assert command["summary"] == "measured"
    assert command["duration_s"] >= 0.0

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
