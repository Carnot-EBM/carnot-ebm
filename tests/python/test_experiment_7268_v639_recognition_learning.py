"""Verify prospective learning with autonomous change recognition.

Spec refs: REQ-CL-7268 and SCENARIO-CL-7268-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7268_v639_recognition_learning as exp


@pytest.fixture(scope="module")
def sealed_views() -> exp.prototype.StreamViews:
    """Load the exact Exp7267 prospective seal once for focused tests."""

    return exp.load_sealed_views(exp.REPO_ROOT, exp.ExperimentPaths.defaults())


@pytest.fixture(scope="module")
def one_stream_run(
    tmp_path_factory: pytest.TempPathFactory,
    sealed_views: exp.prototype.StreamViews,
) -> tuple[exp.ExperimentPaths, exp.LearningPanel]:
    """Replay one complete stream across all arms under test-owned paths."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7268-panel"))
    panel = exp.run_learning_panel(
        sealed_views,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
    )
    return paths, panel


@pytest.fixture(scope="module")
def one_stream_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build one complete test-owned artifact for validator checks."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7268-artifact"))
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
        bootstrap_draws=200,
    )
    return paths, artifact


def test_req_cl_7268_contract_is_frozen() -> None:
    """REQ-CL-7268 freezes no-model work, units, bootstrap, and value gates."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7268" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 8
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert not any(exp.INVOCATION_COUNTS.values())
    assert exp.STREAM_COUNT == 24
    assert exp.EVENTS_PER_STREAM == 1_024
    assert len(exp.ARMS) == 8
    assert exp.EXPECTED_EVENT_ROWS == 196_608
    assert exp.QUERY_CEILING == 128
    assert exp.BOOTSTRAP_RESAMPLES == 10_000
    assert exp.MEASUREMENT_LIMIT_S == 1_800
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7268_preconditions_authenticate_exact_seal(tmp_path: Path) -> None:
    """SCENARIO-CL-7268-PRECONDITIONS accepts exact bytes and blocks a changed seal."""

    paths = exp.ExperimentPaths.under(tmp_path / "valid")
    checks, hashes, upstream = exp.collect_preconditions(exp.REPO_ROOT, paths)

    assert exp.prototype.gate_summary(checks)["passed"] is True
    assert hashes[str(exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT)] is not None
    assert upstream["recognition_fixture_ready_score"] == 1

    changed = deepcopy(upstream)
    changed["recognition_fixture_ready_score"] = 0
    bad_upstream = tmp_path / "bad-upstream.json"
    bad_upstream.write_text(json.dumps(changed), encoding="utf-8")
    bad_checks, bad_hashes, _ = exp.collect_preconditions(
        exp.REPO_ROOT,
        exp.ExperimentPaths.under(tmp_path / "blocked"),
        upstream_path=bad_upstream,
    )
    artifact = exp.build_blocked_artifact(bad_checks, bad_hashes, ("prospective-01",))

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["recognition_run_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_checks"]
    assert exp.validate_artifact(artifact, check_files=False) == []


def test_scenario_cl_7268_prequential_rows_and_receipts_are_complete(
    one_stream_run: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """SCENARIO-CL-7268-PREQUENTIAL keeps predictions sealed and feedback charged."""

    paths, panel = one_stream_run
    assert len(panel.event_rows) == exp.EVENTS_PER_STREAM * len(exp.ARMS)
    assert len(panel.rows) == len(exp.ARMS)
    assert exp.prequential_row_errors(panel.event_rows) == []
    assert all(row["prediction_frozen_before_release"] is True for row in panel.event_rows)
    assert all(row["query_count"] <= exp.QUERY_CEILING for row in panel.rows)
    assert {row["kind"] for row in panel.receipts} == set(exp.RECEIPT_KINDS)
    assert panel.maximum_memory_bytes <= exp.MEMORY_CAP_BYTES

    raw_receipt = exp.prototype._atomic_write(
        paths.prequential_rows,
        exp.prototype.jsonl_bytes(panel.event_rows),
    )
    reduced = exp.independent_reduce(paths.prequential_rows)
    assert reduced == panel.rows
    assert raw_receipt["bytes"] == paths.prequential_rows.stat().st_size


def test_scenario_cl_7268_metrics_keep_both_strata_and_full_memory() -> None:
    """SCENARIO-CL-7268-METRICS retains overlap and the full-memory comparator."""

    rows = exp.independent_reduce(exp.UPSTREAM_RAW_ROWS)
    comparisons = exp.build_comparison_rows(rows, draws=200)

    assert {row["stratum"] for row in comparisons} == {
        "overall",
        "separated_recurrence",
        "overlapping_recurrence",
    }
    assert any(row["control_arm"] == "full_version_space_memory" for row in comparisons)
    assert any(
        row["treatment_arm"] == "full_version_space_memory" and row["control_arm"] == "reset"
        for row in comparisons
    )
    assert all(row["independent_unit"] == "stream" for row in comparisons)


def test_scenario_cl_7268_bootstrap_and_value_gates_are_fail_closed() -> None:
    """SCENARIO-CL-7268-BOOTSTRAP scores the frozen CI and causal gates."""

    rows = exp.independent_reduce(exp.UPSTREAM_RAW_ROWS)
    comparisons = exp.build_comparison_rows(rows, draws=200)
    causal = exp.independent_causal_summary(exp.UPSTREAM_RAW_ROWS)
    gates = exp.score_acceptance_gates(comparisons, causal)

    assert causal["prospective_selection_change_count"] > 0
    assert causal["later_changed_prediction_count"] > 0
    assert causal["pre_release_difference_count"] == 0
    assert causal["cap_violation_count"] == 0
    assert gates["prospective_causal_change"]["passed"] is True
    assert gates["future_error_vs_reset"]["expected"] == "ci95_upper<0"
    assert gates["recurrence_error_vs_shuffle"]["passed"] is False
    assert exp.classify_result(gates) == (0, "null")


def test_scenario_cl_7268_cost_rows_cover_full_event_and_durable_commit(
    one_stream_run: tuple[exp.ExperimentPaths, exp.LearningPanel],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7268-COST reports event percentiles and durable I/O."""

    _, panel = one_stream_run
    e2e_rows, durable_ns = exp.run_e2e_controls(tmp_path / "e2e")
    costs = exp.cost_summary(panel.event_rows, durable_ns)
    by_operation = {row["operation"]: row for row in costs}

    assert set(by_operation) == set(exp.COST_OPERATIONS)
    assert by_operation["full_event"]["p95_ns"] >= by_operation["full_event"]["p50_ns"]
    assert by_operation["memory_bytes"]["p95"] > 0
    assert by_operation["durable_commit"]["count"] == 1
    assert durable_ns > 0
    assert all(row["passed"] is True for row in e2e_rows)


def test_scenario_cl_7268_terminal_keeps_completion_separate_from_value(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7268-TERMINAL publishes a measured null without calling it partial."""

    paths, artifact = one_stream_artifact

    assert artifact["status"] == "complete"
    assert artifact["recognition_run_complete_score"] == 1
    assert artifact["recognition_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["prequential_rows_path"] == str(paths.prequential_rows)
    assert (
        exp.validate_artifact(
            artifact,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    candidate = deepcopy(artifact)
    candidate["recognition_run_complete_score"] = 0
    assert "completion_score" in exp.validate_artifact(
        candidate,
        expected_stream_ids=("prospective-01",),
        check_files=False,
    )


def test_req_cl_7268_defensive_row_and_stream_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    one_stream_run: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """REQ-CL-7268 rejects changed seals, bad rows, invalid costs, and empty evidence."""

    _, panel = one_stream_run
    monkeypatch.setattr(exp.prototype, "stream_conformance_errors", lambda *_: ["changed"])
    with pytest.raises(ValueError, match="sealed_stream_conformance:changed"):
        exp.load_sealed_views(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "seal"))
    monkeypatch.undo()

    bad_source = deepcopy(panel.event_rows[0])
    bad_source["full_denominator_error"] = 1 - int(bad_source["full_denominator_error"])
    releases = {bad_source["event_id"]: {"delay": 0}}
    with pytest.raises(ValueError, match="sealed_error_mismatch"):
        exp._enrich_stream_rows([bad_source], releases, 1)

    bad_row = deepcopy(panel.event_rows[0])
    bad_row.update(
        {
            "lookup_cost_ns": -1,
            "query_selected": True,
            "query_release_index": -1,
            "memory_total_bytes": exp.MEMORY_CAP_BYTES + 1,
        }
    )
    errors = exp.prequential_row_errors([bad_row] * (exp.QUERY_CEILING + 1))
    assert {"negative_cost", "release_before_query", "query_cap", "memory_cap"} <= set(errors)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(empty)
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_causal_summary(empty)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_raw_row"):
        exp.independent_reduce(invalid)
    with pytest.raises(ValueError, match="invalid_raw_row"):
        exp.independent_causal_summary(invalid)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps(panel.event_rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_row_validation_failed"):
        exp.independent_reduce(malformed)


def test_req_cl_7268_loop_heartbeat_timeout_and_conformance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed_views: exp.prototype.StreamViews,
    one_stream_run: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """REQ-CL-7268 keeps a live heartbeat and preserves timeout checkpoints."""

    _, panel = one_stream_run
    fake = SimpleNamespace(
        event_rows=panel.event_rows,
        maximum_memory_bytes=panel.maximum_memory_bytes,
    )
    monkeypatch.setattr(exp.prototype, "run_recognition_panel", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(
        exp,
        "_enrich_stream_rows",
        lambda *_args, **_kwargs: (panel.event_rows, panel.receipts),
    )
    clock = [0.0]

    def ticking() -> float:
        clock[0] += 61.0
        return clock[0]

    monkeypatch.setattr(exp.time, "monotonic", ticking)
    monkeypatch.setattr(exp, "MEASUREMENT_LIMIT_S", 100)
    paths = exp.ExperimentPaths.under(tmp_path / "timeout")
    with pytest.raises(TimeoutError, match="measurement_limit_exceeded"):
        exp.run_learning_panel(
            sealed_views,
            paths,
            stream_ids=("prospective-01",),
        )
    assert paths.provisional.exists()

    monkeypatch.setattr(exp, "prequential_row_errors", lambda _rows: ["changed"])
    with pytest.raises(ValueError, match="prequential_conformance:changed"):
        exp.run_learning_panel(
            sealed_views,
            exp.ExperimentPaths.under(tmp_path / "bad-panel"),
            stream_ids=("prospective-01",),
        )


def test_req_cl_7268_build_fails_closed_on_internal_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    one_stream_run: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """REQ-CL-7268 stops on E2E, reducer, and cold-validator mismatches."""

    _, panel = one_stream_run
    monkeypatch.setattr(exp, "run_learning_panel", lambda *_args, **_kwargs: panel)
    bad_upstream = deepcopy(
        json.loads((exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8"))
    )
    bad_upstream["recognition_fixture_ready_score"] = 0
    bad_path = tmp_path / "bad-upstream.json"
    bad_path.write_text(json.dumps(bad_upstream), encoding="utf-8")
    blocked = exp.build_and_seal(
        exp.REPO_ROOT,
        exp.ExperimentPaths.under(tmp_path / "blocked"),
        stream_ids=("prospective-01",),
        upstream_path=bad_path,
        progress=True,
    )
    assert blocked["status"] == "blocked"

    monkeypatch.setattr(exp, "run_e2e_controls", lambda _root: ([{"passed": False}], 1))
    with pytest.raises(ValueError, match="e2e_control_failed"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "e2e-fail"),
            stream_ids=("prospective-01",),
            bootstrap_draws=2,
        )

    e2e = ([{"stage": str(index), "passed": True} for index in range(7)], 1)
    monkeypatch.setattr(exp, "run_e2e_controls", lambda _root: e2e)
    monkeypatch.setattr(exp, "independent_reduce", lambda _path: [])
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "reduce-fail"),
            stream_ids=("prospective-01",),
            bootstrap_draws=2,
        )

    monkeypatch.setattr(exp, "independent_reduce", lambda _path: panel.rows)
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["changed"])
    with pytest.raises(ValueError, match="artifact_validation_failed:changed"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "validate-fail"),
            stream_ids=("prospective-01",),
            bootstrap_draws=2,
        )


def test_req_cl_7268_validator_writer_and_command_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """REQ-CL-7268 retains command logs and rejects invalid terminal writes."""

    _, artifact = one_stream_artifact
    updated = exp.attach_validation_receipts(
        artifact,
        [{"command": "check", "exit_code": 0, "classification": "passed"}],
    )
    assert updated["validation_receipts"][0]["command"] == "check"
    assert updated["reproducibility_checksum"] == exp.reproducibility_checksum(updated)

    receipt = exp._command_receipt(
        [str(exp.REPO_ROOT / ".venv/bin/python"), "-c", "print('command-ok')"]
    )
    assert receipt["exit_code"] == 0
    assert receipt["classification"] == "passed"
    commands = exp._validation_commands(tmp_path / "candidate.json")
    assert len(commands) == 9
    assert exp._parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE

    output = tmp_path / "written.json"
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: [])
    exp.write_artifact(output, {"status": "complete"})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="artifact_validation_failed:bad"):
        exp.write_artifact(output, {})

    nonterminal = deepcopy(artifact)
    nonterminal["status"] = "in_progress"
    nonterminal["reproducibility_checksum"] = exp.reproducibility_checksum(nonterminal)
    monkeypatch.undo()
    assert "status" in exp.validate_artifact(
        nonterminal,
        expected_stream_ids=("prospective-01",),
    )


def test_req_cl_7268_main_success_block_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """REQ-CL-7268 keeps CLI date, blocked, failed-check, and success behavior explicit."""

    _, artifact = one_stream_artifact
    with pytest.raises(SystemExit, match="run_date_must_be_20260913"):
        exp.main(["--date", "20260912", "--output-root", str(tmp_path / "date")])

    writes: list[str] = []
    monkeypatch.setattr(
        exp, "write_artifact", lambda path, *_args, **_kwargs: writes.append(str(path))
    )
    blocked = deepcopy(artifact)
    blocked["status"] = "blocked"
    monkeypatch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: blocked)
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "blocked")]) == 0
    assert writes

    monkeypatch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: deepcopy(artifact))
    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command: {
            "command": " ".join(command),
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": "sha256:" + "0" * 64,
        },
    )
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "success")]) == 0

    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command: {
            "command": " ".join(command),
            "exit_code": 1,
            "classification": "failed",
            "log_sha256": "sha256:" + "1" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="focused_validation_failed"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "failed")])
