"""Verify active recognition before bounded memory reactivation.

Spec refs: REQ-CL-7267 and SCENARIO-CL-7267-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7267_v639_recognition_prototype as exp


def _masks(parameter: int) -> dict[str, int]:
    """Build one archived finite hypothesis without evaluator-only fields."""

    return {family: 1 << parameter for family in exp.FAMILIES}


def _release(event_id: str, value: int, label: str, index: int) -> dict[str, object]:
    """Build one released lower-bound observation for transaction checks."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": value,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


@pytest.fixture(scope="module")
def prospective_views() -> exp.StreamViews:
    """Generate the fixed prospective bytes once for focused tests."""

    return exp.build_stream_views("prospective")


@pytest.fixture(scope="module")
def one_stream_panel(prospective_views: exp.StreamViews) -> exp.RecognitionPanel:
    """Replay one complete stream across all eight arms once."""

    return exp.run_recognition_panel(
        prospective_views,
        stream_ids=("prospective-01",),
        progress=True,
    )


@pytest.fixture(scope="module")
def one_stream_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build one terminal fixture under test-owned paths."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7267-artifact"))
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
    )
    return paths, artifact


def test_req_cl_7267_spec_and_frozen_contract() -> None:
    """REQ-CL-7267 fixes zero-model work, stream counts, arms, and memory."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7267" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 8
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INVOCATION_COUNTS == {
        "attempted_model_loads": 0,
        "completed_model_loads": 0,
        "attempted_generation_calls": 0,
        "completed_generation_calls": 0,
        "usable_answers": 0,
    }
    assert len(exp.ARMS) == 8
    assert exp.DEVELOPMENT_STREAM_COUNT == 8
    assert exp.STREAM_COUNT == 24
    assert exp.EVENTS_PER_STREAM == 1_024
    assert exp.WARMUP_COUNT == 128
    assert exp.QUERY_CEILING == 128
    assert exp.ARCHIVE_CAP == 4
    assert exp.MEMORY_CAPS["total_bytes"] == 69_632
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7267_basis_preserves_old_discriminator() -> None:
    """SCENARIO-CL-7267-BASIS keeps a released discriminator outside recency."""

    diagnosis = exp.run_stable_basis_diagnostic()

    assert diagnosis["rolling_signature_distinct_count"] == 1
    assert diagnosis["stable_signature_distinct_count"] == 2
    assert diagnosis["counterexample_survives_stable_basis"] is True
    assert diagnosis["basis_size"] <= exp.STABLE_BASIS_CAPACITY
    assert set(diagnosis["stable_basis_event_ids"]).isdisjoint(
        diagnosis["fresh_validation_event_ids"]
    )


def test_scenario_cl_7267_query_maximizes_public_disagreement() -> None:
    """SCENARIO-CL-7267-QUERY selects maximum disagreement without authority."""

    archives = [
        exp.archive_row("a", 0, _masks(0)),
        exp.archive_row("b", 1, _masks(8)),
        exp.archive_row("c", 2, _masks(16)),
    ]
    block = [
        {"event_id": "same", "family_id": "lower_bound", "numeric_value": 31},
        {"event_id": "split", "family_id": "lower_bound", "numeric_value": 7},
        {"event_id": "split-more", "family_id": "lower_bound", "numeric_value": 12},
    ]
    selected, receipt = exp.select_disagreement_query(archives, block, [0, 2, 1])

    assert selected["event_id"] == "split-more"
    assert receipt["maximum_disagreement"] == max(
        row["disagreement_pair_count"] for row in receipt["candidate_rows"]
    )
    assert receipt["controller_input_fields"] == ["event_id", "family_id", "numeric_value"]
    leaked = deepcopy(block)
    leaked[0]["hidden_parameter"] = 3
    with pytest.raises(ValueError, match="private_authority_in_query"):
        exp.select_disagreement_query(archives, leaked, [0, 1, 2])
    with pytest.raises(ValueError, match="invalid_public_query"):
        exp.select_disagreement_query(
            archives,
            [{"event_id": "bad", "family_id": "unknown", "numeric_value": 0}],
            [0],
        )


def test_scenario_cl_7267_diagnosis_keeps_failure_stages_separate() -> None:
    """SCENARIO-CL-7267-DIAGNOSIS does not turn aggregate rows into a rebase claim."""

    upstream = json.loads((exp.REPO_ROOT / exp.DEFAULT_AUDIT_ARTIFACT).read_text(encoding="utf-8"))
    rows = exp.reduce_saved_failure_rows(upstream)
    by_stage = {row["stage"]: row for row in rows}

    assert by_stage["nomination"]["observed"] > 0
    assert by_stage["post_shuffle_choice"]["observed"] == 0
    assert by_stage["later_prediction_change"]["observed"] == 0
    assert by_stage["one_eligible_or_none"]["observed"] == by_stage["nomination"]["observed"]
    assert by_stage["archive_size"]["censored"] is True
    assert by_stage["signature_merging"]["censored"] is True
    assert by_stage["safety_rejection"]["censored"] is True


def test_scenario_cl_7267_streams_are_fresh_sealed_and_separated(
    prospective_views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7267-STREAMS seals equal separated and overlap strata."""

    development = exp.build_stream_views("development")
    assert exp.stream_conformance_errors(development, "development") == []
    assert exp.stream_conformance_errors(prospective_views, "prospective") == []
    assert prospective_views.manifest["strata"] == {
        "separated_recurrence": 12,
        "overlapping_recurrence": 12,
    }
    assert not exp.public_leakage_errors(prospective_views.public)
    assert len({row["event_id"] for row in prospective_views.public}) == 24 * 1_024
    changed = deepcopy(prospective_views)
    changed.public[0]["regime_id"] = "private"
    assert exp.stream_conformance_errors(changed, "prospective") == ["public_authority_leakage"]
    with pytest.raises(ValueError, match="invalid_stream_kind"):
        exp.build_stream_views("old")


def test_scenario_cl_7267_development_intervention_changes_a_selection() -> None:
    """SCENARIO-CL-7267-QUERY requires an effective development intervention."""

    result = exp.run_development_selection(exp.build_stream_views("development"))

    assert result["development_stream_count"] == 8
    assert result["active_query_selection_change_count"] > 0
    assert result["parameters_frozen_before_prospective"] is True
    assert result["prospective_authority_used"] is False


def test_scenario_cl_7267_panel_has_all_arms_and_exact_limits(
    one_stream_panel: exp.RecognitionPanel,
) -> None:
    """SCENARIO-CL-7267-PANEL keeps one full stream uncensored and bounded."""

    assert len(one_stream_panel.event_rows) == exp.EVENTS_PER_STREAM * len(exp.ARMS)
    assert len(one_stream_panel.rows) == len(exp.ARMS)
    assert {row["arm"] for row in one_stream_panel.rows} == set(exp.ARMS)
    assert exp.event_row_errors(one_stream_panel.event_rows) == []
    assert all(row["event_count"] == exp.EVENTS_PER_STREAM for row in one_stream_panel.rows)
    assert all(row["query_count"] <= exp.QUERY_CEILING for row in one_stream_panel.rows)
    assert all(
        row["maximum_memory_bytes"] <= exp.MEMORY_CAPS["total_bytes"]
        for row in one_stream_panel.rows
    )
    assert one_stream_panel.completed_stream_count == 1
    assert one_stream_panel.censored_stream_count == 0


def test_scenario_cl_7267_transaction_is_delayed_atomic_and_reversible(tmp_path: Path) -> None:
    """SCENARIO-CL-7267-TRANSACTION covers commit, reload, rejection, and rollback."""

    controller = exp.RecognitionController.from_masks(_masks(0))
    controller.add_archive(_masks(8))
    event = {"event_id": "query", "family_id": "lower_bound", "numeric_value": 7}
    prediction_before = controller.predict(event)
    selected, query = controller.select_request([event], [0])
    controller.record_query(selected, query, request_index=1, release_index=2)
    state_path = tmp_path / "controller.json"
    controller.save(state_path)
    before = controller.state_bytes()

    future = _release("query", 7, "accept", 1)
    future["release_index"] = 2
    with pytest.raises(exp.RecognitionCommitRejected, match="future_release"):
        controller.commit_batch(
            [future],
            current_cycle=1,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    assert controller.state_bytes() == before == state_path.read_bytes()
    with pytest.raises(exp.RecognitionCommitRejected, match="stale_parent"):
        controller.commit_batch(
            [_release("stale", 0, "accept", 2)],
            current_cycle=2,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )

    receipt = controller.commit_batch(
        [{**future, "release_index": 2}],
        current_cycle=2,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    assert receipt["operations"][0]["same_event_correction"] is False
    assert prediction_before == ("accept", 0.0)
    assert exp.RecognitionController.load(state_path).state_hash() == controller.state_hash()
    child = controller.state_bytes()
    with pytest.raises(exp.RecognitionCommitRejected, match="duplicate_release"):
        controller.commit_batch(
            [{**future, "release_index": 2}],
            current_cycle=2,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    assert controller.state_bytes() == child == state_path.read_bytes()
    rollback = controller.rollback(receipt, state_path=state_path)
    assert rollback["byte_identical"] is True
    assert controller.state_bytes() == before == state_path.read_bytes()


def test_scenario_cl_7267_mutations_and_e2e_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7267-TRANSACTION detects all four named mutations."""

    controls = exp.run_mutation_controls(tmp_path / "mutations")
    e2e = exp.run_e2e_controls(tmp_path / "e2e")

    assert {row["mutation"] for row in controls} == {
        "future_label_access",
        "ineffective_shuffled_identity",
        "duplicate_release",
        "stale_parent",
    }
    assert all(row["passed"] is True for row in controls)
    assert [row["stage"] for row in e2e] == [
        "released_event",
        "query",
        "delayed_feedback",
        "atomic_commit",
        "later_prediction",
        "cold_restore",
        "rollback",
    ]
    assert all(row["passed"] is True for row in e2e)


def test_raw_reducer_rejects_leakage_and_incomplete_units(
    one_stream_panel: exp.RecognitionPanel,
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7267-PANEL makes raw chronological rows authoritative."""

    path = tmp_path / "rows.jsonl"
    path.write_bytes(exp.jsonl_bytes(one_stream_panel.event_rows))
    assert exp.independent_reduce(path) == one_stream_panel.rows
    changed = deepcopy(one_stream_panel.event_rows)
    changed[0]["held_out_label_visible_to_controller"] = True
    assert "future_label_leakage" in exp.event_row_errors(changed)
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(tmp_path / "missing.jsonl")


def test_blocked_external_input_is_terminal_and_row_free(tmp_path: Path) -> None:
    """SCENARIO-CL-7267-PRECONDITIONS maps absence to blocked, not partial."""

    paths = exp.ExperimentPaths.under(tmp_path / "out")
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        upstream_overrides={"exp7254": tmp_path / "missing.json"},
        progress=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["recognition_fixture_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_checks"]
    assert exp.validate_artifact(artifact, expected_stream_ids=("prospective-01",)) == []


def test_build_validate_receipts_and_atomic_terminal(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7267-TERMINAL separates fixture readiness from learning value."""

    paths, artifact = one_stream_artifact
    assert artifact["status"] == "complete"
    assert artifact["recognition_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["held_out_learning_value_scored"] is False
    assert artifact["sample_size_budget"]["completed_streams"] == 1
    assert (
        exp.validate_artifact(
            artifact,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    receipt = {
        "command": "focused-check",
        "exit_code": 0,
        "classification": "passed",
        "log_sha256": "sha256:" + "1" * 64,
    }
    attached = exp.attach_validation_receipts(artifact, [receipt])
    assert attached["validation_receipts"] == [receipt]
    exp.write_artifact(
        paths.artifact,
        attached,
        expected_stream_ids=("prospective-01",),
    )
    assert (
        json.loads(paths.artifact.read_text())["reproducibility_checksum"]
        == (attached["reproducibility_checksum"])
    )

    broken = deepcopy(attached)
    broken["recognition_fixture_ready_score"] = 0
    assert "complete_contract" in exp.validate_artifact(
        broken,
        expected_stream_ids=("prospective-01",),
    )
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{"command": "missing fields"}])


def test_thin_wrapper_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-7267 keeps the runnable experiment entrypoint thin."""

    called: list[object] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0
    assert called == [None]


def test_exact_label_dependency_remains_the_shipped_finite_solver() -> None:
    """REQ-CL-7267 reuses the shipped exact hypothesis semantics."""

    event = {"event_id": "x", "family_id": "lower_bound", "numeric_value": 7}
    assert exp.predict_masks(_masks(8), event) == "reject"
    assert exp.predict_masks(_masks(0), event) == "accept"
    assert exp7226.exact_label("lower_bound", 7, 8) == "reject"


def test_controller_state_validation_and_durable_failures(tmp_path: Path) -> None:
    """SCENARIO-CL-7267-TRANSACTION rejects malformed and stale durable state."""

    assert exp.ExperimentPaths.defaults().artifact == exp.REPO_ROOT / exp.DEFAULT_ARTIFACT
    for kwargs in ({"query_mode": "bad"}, {"association_mode": "bad"}):
        with pytest.raises(ValueError):
            exp.RecognitionController(**kwargs)
    disabled = exp.RecognitionController(archive_cap=0)
    assert disabled.add_archive(_masks(0)) is None

    controller = exp.RecognitionController.from_masks(_masks(0))
    controller.add_archive(_masks(8))
    base = controller.state_dict()
    assert base == controller.state_dict()
    cases: list[tuple[dict[str, object], str]] = []
    wrong_schema = deepcopy(base)
    wrong_schema["schema"] = "wrong"
    cases.append((wrong_schema, "invalid_recognition_state"))
    too_many = deepcopy(base)
    too_many["archive_cap"] = 0
    cases.append((too_many, "archive_capacity"))
    bad_identity = deepcopy(base)
    bad_identity["archives"][0]["state_hash"] = "sha256:bad"
    cases.append((bad_identity, "archive_identity"))
    oversized = deepcopy(base)
    oversized["ledger"] = [str(index) for index in range(exp.LEDGER_CAPACITY + 1)]
    cases.append((oversized, "bounded_collection_capacity"))
    too_many_bytes = deepcopy(base)
    too_many_bytes["stable_basis"] = [
        {
            "event_id": "x" * 3_000 + str(index),
            "family_id": "lower_bound",
            "numeric_value": 0,
        }
        for index in range(exp.STABLE_BASIS_CAPACITY)
    ]
    cases.append((too_many_bytes, "recognition_memory_cap"))
    for state, message in cases:
        with pytest.raises(ValueError, match=message):
            exp.RecognitionController.from_state(state)

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_recognition_state"):
        exp.RecognitionController.load(non_object)

    pending = exp.RecognitionController()
    event = {"event_id": "q", "family_id": "lower_bound", "numeric_value": 0}
    selected, receipt = pending.select_request([event], [0])
    for index in range(exp.PENDING_CAPACITY):
        pending.record_query(
            {**selected, "event_id": f"q-{index}"},
            receipt,
            request_index=index,
            release_index=index,
        )
    with pytest.raises(ValueError, match="pending_capacity"):
        pending.record_query(selected, receipt, request_index=5, release_index=5)

    corrupt = exp.RecognitionController()
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(exp.RecognitionCommitRejected, match="corrupt_durable_state"):
        corrupt.commit_batch(
            [_release("corrupt", 0, "accept", 1)],
            current_cycle=1,
            expected_parent_hash=corrupt.state_hash(),
            state_path=corrupt_path,
        )
    stale = exp.RecognitionController()
    stale_path = tmp_path / "stale.json"
    stale.save(stale_path)
    stale.add_archive(_masks(8))
    with pytest.raises(exp.RecognitionCommitRejected, match="stale_durable_parent"):
        stale.commit_batch(
            [_release("stale-durable", 0, "accept", 1)],
            current_cycle=1,
            expected_parent_hash=stale.state_hash(),
            state_path=stale_path,
        )

    rollback_controller = exp.RecognitionController()
    commit = rollback_controller.commit_batch(
        [_release("rollback-errors", 0, "accept", 1)],
        current_cycle=1,
        expected_parent_hash=rollback_controller.state_hash(),
    )
    invalid = deepcopy(commit)
    invalid["parent_bytes_b64"] = "bad"
    with pytest.raises(exp.RecognitionCommitRejected, match="invalid_rollback_receipt"):
        rollback_controller.rollback(invalid)
    wrong_parent = deepcopy(commit)
    wrong_parent["parent_hash"] = "sha256:" + "0" * 64
    with pytest.raises(exp.RecognitionCommitRejected, match="rollback_parent_hash"):
        rollback_controller.rollback(wrong_parent)
    rollback_controller.rollback(commit)
    with pytest.raises(exp.RecognitionCommitRejected, match="stale_rollback"):
        rollback_controller.rollback(commit)


def test_stream_and_raw_defensive_validation(
    prospective_views: exp.StreamViews,
    one_stream_panel: exp.RecognitionPanel,
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7267-STREAMS names count, identity, strata, and chronology failures."""

    assert exp._nested_keys([{"nested": [{"value": 1}]}]) == {"nested", "value"}
    mutations = []
    count = deepcopy(prospective_views)
    count.public.pop()
    mutations.append((count, "event_count"))
    identity = deepcopy(prospective_views)
    identity.authority[0]["event_id"] = "wrong"
    mutations.append((identity, "event_identity"))
    strata = deepcopy(prospective_views)
    strata.manifest["strata"] = {}
    mutations.append((strata, "strata"))
    chronology = deepcopy(prospective_views)
    chronology.public[0]["chronology_index"] = 2
    mutations.append((chronology, "chronology"))
    for changed, expected in mutations:
        assert expected in exp.stream_conformance_errors(changed, "prospective")

    bad_rows = deepcopy(one_stream_panel.event_rows)
    bad_rows[0]["held_out_label_visible_to_controller"] = True
    path = tmp_path / "bad-rows.jsonl"
    path.write_bytes(exp.jsonl_bytes(bad_rows))
    with pytest.raises(ValueError, match="raw_row_validation_failed"):
        exp.independent_reduce(path)
    with pytest.raises(ValueError, match="incomplete_stream"):
        exp.run_recognition_panel(
            prospective_views,
            stream_ids=("prospective-missing",),
        )


def test_long_loop_heartbeat_path(
    prospective_views: exp.StreamViews,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-7267-PANEL emits a monotonic heartbeat inside a long loop."""

    calls = 0

    def monotonic() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 61.0

    monkeypatch.setattr(exp.time, "monotonic", monotonic)
    exp.run_recognition_panel(
        prospective_views,
        stream_ids=("prospective-02",),
        progress=True,
    )
    assert "benchmark heartbeat" in capsys.readouterr().out


def test_build_failure_paths_and_terminal_null(
    prospective_views: exp.StreamViews,
    one_stream_panel: exp.RecognitionPanel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7267-TERMINAL preserves explicit failure and null paths."""

    original_conformance = exp.stream_conformance_errors
    monkeypatch.setattr(exp, "stream_conformance_errors", lambda views, kind: ["forced"])
    with pytest.raises(ValueError, match="stream_conformance_failed"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "streams"))
    monkeypatch.setattr(exp, "stream_conformance_errors", original_conformance)

    monkeypatch.setattr(exp, "run_recognition_panel", lambda *args, **kwargs: one_stream_panel)
    monkeypatch.setattr(exp, "event_row_errors", lambda rows: ["forced"])
    with pytest.raises(ValueError, match="event_row_errors"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "raw-errors"),
            stream_ids=("prospective-01",),
        )

    monkeypatch.setattr(exp, "event_row_errors", lambda rows: [])
    monkeypatch.setattr(exp, "independent_reduce", lambda path: [])
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "reduce-errors"),
            stream_ids=("prospective-01",),
        )

    monkeypatch.setattr(exp, "independent_reduce", lambda path: one_stream_panel.rows)
    original_development = exp.run_development_selection

    def ineffective(views: exp.StreamViews) -> dict[str, object]:
        result = original_development(views)
        result["active_query_selection_change_count"] = 0
        return result

    monkeypatch.setattr(exp, "run_development_selection", ineffective)
    null = exp.build_and_seal(
        exp.REPO_ROOT,
        exp.ExperimentPaths.under(tmp_path / "null"),
        stream_ids=("prospective-01",),
    )
    assert null["status"] == "complete"
    assert null["verdict_class"] == "null"
    assert null["recognition_fixture_ready_score"] == 0
    assert null["honest_verdict"].startswith("complete_null")

    monkeypatch.setattr(exp, "run_development_selection", original_development)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "artifact-errors"),
            stream_ids=("prospective-01",),
        )


def test_validator_and_writer_reject_nonterminal_or_invalid_artifacts(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7267-TERMINAL refuses partial and invalid publication bytes."""

    _, artifact = one_stream_artifact
    partial = deepcopy(artifact)
    partial["status"] = "in_progress"
    partial["reproducibility_checksum"] = exp.reproducibility_checksum(partial)
    assert "status" in exp.validate_artifact(partial)
    broken = deepcopy(artifact)
    broken["schema"] = "wrong"
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(
            tmp_path / "must-not-exist.json",
            broken,
            expected_stream_ids=("prospective-01",),
        )


def test_validation_subprocess_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7267 records subprocess logs and publishes only after validation."""

    receipt = exp._command_receipt([sys.executable, "-c", "print('receipt-ok')"])
    assert receipt["exit_code"] == 0
    assert receipt["classification"] == "passed"
    commands = exp._validation_commands(tmp_path / "candidate.json")
    assert any("coverage" in command[0] and "run" in command for command in commands)
    assert exp._parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    with pytest.raises(SystemExit, match="run_date_must_be"):
        exp.main(["--date", "wrong", "--output-root", str(tmp_path / "wrong-date")])

    paths = exp.ExperimentPaths.under(tmp_path / "cli")
    monkeypatch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: paths))
    monkeypatch.setattr(exp, "write_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_progress", lambda *args: None)
    monkeypatch.setattr(
        exp,
        "build_and_seal",
        lambda *args, **kwargs: {"status": "blocked", "validation_receipts": []},
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0

    complete = {"status": "complete", "validation_receipts": []}
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(complete))
    monkeypatch.setattr(exp, "_validation_commands", lambda candidate: [["check"]])
    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command: {
            "command": "check",
            "exit_code": 0,
            "classification": "passed",
            "log_sha256": "sha256:" + "1" * 64,
        },
    )
    monkeypatch.setattr(
        exp,
        "attach_validation_receipts",
        lambda artifact, receipts: {**artifact, "validation_receipts": receipts},
    )
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "cli")]) == 0

    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command: {
            "command": "check",
            "exit_code": 1,
            "classification": "failed",
            "log_sha256": "sha256:" + "2" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="focused_validation_failed"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "cli-failed")])
