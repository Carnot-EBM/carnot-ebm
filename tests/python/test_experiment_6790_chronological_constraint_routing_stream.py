"""Focused tests for the frozen chronological constraint-routing stream.

Spec refs: REQ-CL-6790 and SCENARIO-CL-6790-*.
"""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6790_chronological_constraint_routing_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/continuous-learning/spec.md"
SOURCE_PATH = REPO_ROOT / exp.SOURCE_ARTIFACT_RELATIVE_PATH


@pytest.fixture(scope="module")
def source() -> exp.JsonDict:
    """Load the frozen exact source once because every focused test is read-only."""

    return exp.load_json_object(SOURCE_PATH)


@pytest.fixture(scope="module")
def events(source: exp.JsonDict) -> list[exp.JsonDict]:
    """Build the event manifest once because its 240 events never mutate."""

    return exp.build_events(source)


@pytest.fixture(scope="module")
def orders(events: list[exp.JsonDict]) -> list[exp.JsonDict]:
    """Freeze all five order replicates once for row and replay checks."""

    return exp.freeze_orders(events)


@pytest.fixture(scope="module")
def rows(events: list[exp.JsonDict], orders: list[exp.JsonDict]) -> list[exp.JsonDict]:
    """Materialize each event-order cell once for row-derived assertions."""

    return exp.build_rows(events, orders)


def test_req_cl_6790_spec_owns_the_stream_contract() -> None:
    """REQ-CL-6790 anchors the route, isolation, replay, and artifact boundary."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-6790", 1)[1]
    for marker in (
        "SCENARIO-CL-6790-CHRONOLOGY",
        "SCENARIO-CL-6790-ROUTES",
        "SCENARIO-CL-6790-HELD-FUTURE",
        "SCENARIO-CL-6790-POISON",
        "SCENARIO-CL-6790-HEADROOM",
        "SCENARIO-CL-6790-REPLAY",
        "SCENARIO-CL-6790-BLOCKED",
        "at least 240 bounded route-decision events",
        "complete_blocked_constraint_routing_stream",
        exp.INFERENCE_SUBSTRATE,
        exp.MODULE_RELATIVE_PATH.as_posix(),
        exp.SCRIPT_RELATIVE_PATH.as_posix(),
        exp.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in exp.STANDARD_ARTIFACT_FIELDS


def test_scenario_cl_6790_preconditions_pin_exact_authority() -> None:
    """SCENARIO-CL-6790-BLOCKED checks readiness, hashes, families, and costs."""

    summary = exp.evaluate_preconditions(repo_root=REPO_ROOT)
    assert summary["all_passed"] is True
    assert [row["check"] for row in summary["checks"]] == [
        "exp6786_artifact_exists",
        "exp6786_artifact_parses",
        "constraint_group_fixture_ready",
        "source_artifact_sha256",
        "source_module_sha256",
        "source_manifest_sha256",
        "nested_source_hashes",
        "minimum_topology_families",
        "local_failure_rows",
        "cross_dependency_failure_rows",
        "bounded_route_cost_model",
    ]
    assert exp.first_failed_check(summary)["check"] == "all_preconditions"

    failed = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        overrides={
            "constraint_group_fixture_ready": False,
            "source_artifact_sha256": "sha256:changed",
            "source_module_sha256": "sha256:changed",
            "source_manifest_sha256": "sha256:changed",
            "nested_source_hashes": {},
            "minimum_topology_families": 2,
            "local_failure_rows": 0,
            "cross_dependency_failure_rows": 0,
            "bounded_route_cost_model": False,
        },
    )
    assert failed["all_passed"] is False
    assert exp.first_failed_check(failed)["check"] == "constraint_group_fixture_ready"
    assert {row["check"] for row in failed["checks"] if not row["passed"]} == {
        "constraint_group_fixture_ready",
        "source_artifact_sha256",
        "source_module_sha256",
        "source_manifest_sha256",
        "nested_source_hashes",
        "minimum_topology_families",
        "local_failure_rows",
        "cross_dependency_failure_rows",
        "bounded_route_cost_model",
    }


def test_scenario_cl_6790_events_have_bounded_route_opportunities(
    events: list[exp.JsonDict],
) -> None:
    """SCENARIO-CL-6790-ROUTES gives every event real non-exhaustive actions."""

    assert len(events) == exp.EVENT_COUNT == 240
    assert len({event["event_id"] for event in events}) == exp.EVENT_COUNT
    assert {event["event_class"] for event in events} == {
        "hard_cross_dependency_failure",
        "easy_local_failure",
        "clean_valid_case",
    }
    assert {
        family: sum(event["topology_family"] == family for event in events)
        for family in exp.TOPOLOGY_FAMILIES
    } == {family: 80 for family in exp.TOPOLOGY_FAMILIES}
    assert sum(event["poison_status"] != "none" for event in events) >= 20
    assert len({event["reusable_motif_id"] for event in events}) < len(events)

    live_routes = {
        route_id
        for route_id, definition in exp.ROUTE_DEFINITIONS.items()
        if definition["live_action"]
    }
    assert exp.EXHAUSTIVE_ROUTE_ID not in live_routes
    for event in events:
        assert set(event["available_actions"]) == live_routes
        assert len(event["available_actions"]) >= 2
        exhaustive_cost = event["exhaustive_route_cost"]
        assert exhaustive_cost == len(event["all_factor_ids"])
        assert all(
            exp.route_cost(event, route_id) == exp.LIVE_ROUTE_BUDGET < exhaustive_cost
            for route_id in event["available_actions"]
        )
        assert exp.EXHAUSTIVE_ROUTE_ID not in event["available_actions"]


def test_scenario_cl_6790_orders_preserve_holdout_and_drift(
    events: list[exp.JsonDict], orders: list[exp.JsonDict]
) -> None:
    """SCENARIO-CL-6790-HELD-FUTURE freezes five distinct chronological orders."""

    event_by_id = {event["event_id"]: event for event in events}
    assert len(orders) == exp.ORDER_COUNT == 5
    assert len({order["order_hash"] for order in orders}) == exp.ORDER_COUNT
    assert all(order["frozen_before_replay"] is True for order in orders)
    for order in orders:
        assert len(order["event_ids"]) == exp.EVENT_COUNT
        assert set(order["event_ids"]) == set(event_by_id)
        held_positions = [
            position
            for position, event_id in enumerate(order["event_ids"])
            if event_by_id[event_id]["held_future"]
        ]
        assert held_positions == list(range(exp.NON_HELD_EVENT_COUNT, exp.EVENT_COUNT))
        assert order["held_future_first_position"] == exp.NON_HELD_EVENT_COUNT
        difficulty_ranks = [
            exp.DIFFICULTY_RANK[event_by_id[event_id]["difficulty"]]
            for event_id in order["event_ids"][: exp.NON_HELD_EVENT_COUNT]
        ]
        assert sum(difficulty_ranks[-40:]) > sum(difficulty_ranks[:40])


def test_scenario_cl_6790_rows_separate_actions_and_receipts(
    events: list[exp.JsonDict],
    orders: list[exp.JsonDict],
    rows: list[exp.JsonDict],
) -> None:
    """SCENARIO-CL-6790-CHRONOLOGY hides receipts until both routes are fixed."""

    assert len(rows) == exp.EVENT_COUNT * exp.ORDER_COUNT == 1200
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert exp.audit_feature_contract(rows) == []
    assert exp.audit_row_consistency(rows, events, orders) == []

    for row in rows:
        observation = row["pre_action"]["legal_observation"]
        receipt = row["revealed_post_action_receipt"]
        actions = row["chosen_baseline_actions"]
        assert set(observation) == set(exp.FEATURE_ALLOWLIST)
        assert row["hidden_receipt_hash"] == exp.sha256_json(receipt)
        assert actions["frozen_policy"] in row["available_actions"]
        assert actions["random_route"] in row["available_actions"]
        assert exp.EXHAUSTIVE_ROUTE_ID not in actions.values()
        assert receipt["frozen_policy"]["route_id"] == actions["frozen_policy"]
        assert receipt["random_route"]["route_id"] == actions["random_route"]
        assert receipt["frozen_policy"]["route_cost"] == exp.LIVE_ROUTE_BUDGET
        assert receipt["exhaustive_diagnostic"]["live_action"] is False
        assert receipt["exhaustive_diagnostic"]["route_cost"] > exp.LIVE_ROUTE_BUDGET
        assert set(receipt["frozen_policy"]["safe_factors"]) <= set(
            receipt["frozen_policy"]["checked_factors"]
        )
        assert row["row_sha256"] == exp.row_checksum(row)


def test_scenario_cl_6790_held_future_never_leaks_into_prior_state(
    rows: list[exp.JsonDict], orders: list[exp.JsonDict]
) -> None:
    """SCENARIO-CL-6790-HELD-FUTURE isolates family stats, retrieval, and tuning."""

    held = exp.HELD_FUTURE_FAMILY
    for order in orders:
        order_rows = sorted(
            (row for row in rows if row["order_id"] == order["order_id"]),
            key=lambda row: row["chronology"]["position"],
        )
        first = next(index for index, row in enumerate(order_rows) if row["held_future"])
        assert first == exp.NON_HELD_EVENT_COUNT
        for row in order_rows[: first + 1]:
            history = row["pre_action"]["history_snapshot"]
            assert held not in history["factor_statistics"]
            assert all(item["topology_family"] != held for item in history["retrieval_memory"])
            assert all(item["topology_family"] != held for item in history["tuning_data"])
        assert held in order_rows[first + 1]["pre_action"]["history_snapshot"]["factor_statistics"]


def test_scenario_cl_6790_poison_marks_apparent_and_credited_reward(
    rows: list[exp.JsonDict],
) -> None:
    """SCENARIO-CL-6790-POISON preserves apparent reward but denies unsafe credit."""

    poison_rows = [row for row in rows if row["poison_status"] != "none"]
    rewarded_poison = [
        row
        for row in poison_rows
        if row["revealed_post_action_receipt"]["frozen_policy"]["apparent_reward"] == 1
    ]
    assert poison_rows
    assert rewarded_poison
    assert {row["poison_status"] for row in poison_rows} == {
        "provenance_conflict",
        "retention_conflict",
    }
    assert all(
        row["revealed_post_action_receipt"]["frozen_policy"]["credited_reward"] == 0
        for row in rewarded_poison
    )
    assert all(
        row["revealed_post_action_receipt"]["poison_status"] == row["poison_status"]
        for row in poison_rows
    )


def test_scenario_cl_6790_every_order_has_reusable_signal_and_headroom(
    rows: list[exp.JsonDict], orders: list[exp.JsonDict]
) -> None:
    """SCENARIO-CL-6790-HEADROOM keeps frozen performance above floor and below ceiling."""

    summary = exp.summarize_rows(rows, orders)
    for order in orders:
        order_id = order["order_id"]
        frozen = summary["frozen_policy_metrics_by_order"][order_id]
        random = summary["random_route_metrics_by_order"][order_id]
        headroom = summary["diagnostic_headroom_by_order"][order_id]
        assert summary["event_count_by_order"][order_id] == exp.EVENT_COUNT
        assert summary["held_future_counts"][order_id] == 80
        assert summary["poison_counts"][order_id] > 0
        assert summary["reusable_motif_counts"][order_id] > 0
        assert set(summary["topology_count_by_order"][order_id]) == set(exp.TOPOLOGY_FAMILIES)
        assert random["decision_accuracy"] < frozen["decision_accuracy"]
        assert frozen["decision_accuracy"] < headroom["exhaustive_decision_accuracy"]
        assert headroom["accuracy_gap"] > 0
        assert frozen["actual_alternative_count"] == exp.EVENT_COUNT


def test_scenario_cl_6790_replay_is_deterministic_and_cold(
    source: exp.JsonDict,
    events: list[exp.JsonDict],
    orders: list[exp.JsonDict],
    rows: list[exp.JsonDict],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-6790-REPLAY reproduces all row hashes in a fresh process."""

    second_events = exp.build_events(source)
    second_orders = exp.freeze_orders(second_events)
    second_rows = exp.build_rows(second_events, second_orders)
    assert second_events == events
    assert second_orders == orders
    assert second_rows == rows

    expected = exp.replay_hashes(rows, orders)
    cold = exp.run_cold_replay(events, orders, repo_root=REPO_ROOT)
    assert cold["agreement"] is True
    assert cold["fresh_process"] is True
    assert cold["aggregate_rows_sha256"] == expected["aggregate_rows_sha256"]
    assert cold["order_row_hashes"] == expected["order_row_hashes"]
    assert cold["cold_pid"] != cold["producer_pid"]

    monkeypatch.setattr(
        exp.sys, "stdin", StringIO(json.dumps({"events": events, "orders": orders}))
    )
    assert exp._cold_replay_worker() == 0
    worker = json.loads(capsys.readouterr().out)
    assert worker["aggregate_rows_sha256"] == expected["aggregate_rows_sha256"]

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=5, stderr="forced cold failure"),
    )
    with pytest.raises(RuntimeError, match="forced cold failure"):
        exp.run_cold_replay(events[:1], [exp.single_event_order(events[0])], repo_root=REPO_ROOT)


def test_req_cl_6790_artifact_is_complete_row_derived_and_atomic(tmp_path: Path) -> None:
    """REQ-CL-6790 publishes one validated artifact with all required evidence."""

    path = tmp_path / "artifact.json"
    artifact = exp.write_artifact(
        artifact_path=path,
        repo_root=REPO_ROOT,
        run_date="20260830",
        duration_s=1.25,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["source_artifact_hash"] == exp.EXPECTED_SOURCE_ARTIFACT_SHA256
    assert artifact["constraint_routing_stream_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["future_feature_violations"] == []
    assert artifact["cold_replay_hashes"]["agreement"] is True
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"][0]["pre_action"]["legal_observation"]["exact_label"] = False
    changed["future_feature_violations"] = exp.audit_feature_contract(changed["rows"])
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "ready artifact contains future feature violations" in exp.validate_artifact(changed)


def test_scenario_cl_6790_blocked_and_defensive_paths_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-6790-BLOCKED retains failed observations and rejects bad output."""

    blocked = exp.build_artifact(
        repo_root=REPO_ROOT,
        run_date="20260830",
        duration_s=0.1,
        precondition_overrides={"constraint_group_fixture_ready": False},
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "complete_blocked_constraint_routing_stream"
    assert blocked["rows"] == []
    assert blocked["constraint_routing_stream_ready"] is False
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_constraint_routing_stream:")
    assert blocked["gate_check_summary"]["first_failure"]["check"] == (
        "constraint_group_fixture_ready"
    )

    missing = exp.build_artifact(
        repo_root=REPO_ROOT,
        source_artifact_path=tmp_path / "missing.json",
        run_date="20260830",
        duration_s=0.1,
    )
    assert missing["gate_check_summary"]["first_failure"]["check"] == ("exp6786_artifact_exists")
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(repo_root=REPO_ROOT, run_date="2026-08-30")

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        exp.load_json_object(non_object)

    artifact = exp.build_artifact(repo_root=REPO_ROOT, duration_s=0.2)
    validation_cases = (
        (lambda value: value.pop("schema"), "required field set mismatch"),
        (
            lambda value: value["field_principles"].pop("schema"),
            "field principle coverage mismatch",
        ),
        (lambda value: value.__setitem__("inference_substrate", "bad"), "inference substrate"),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s"),
        (lambda value: value.__setitem__("random_seed", -1), "random seed"),
        (lambda value: value.__setitem__("verdict_class", "bad"), "verdict class"),
        (lambda value: value.__setitem__("honest_verdict", "bad"), "terminal prefix"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (lambda value: value.__setitem__("status", "bad"), "ready artifact status"),
        (lambda value: value.__setitem__("rows", []), "ready artifact row count"),
        (lambda value: value.__setitem__("order_definitions", []), "ready artifact order count"),
        (
            lambda value: value["gate_check_summary"].__setitem__("all_passed", False),
            "ready artifact has failed gates",
        ),
        (
            lambda value: value["cold_replay_hashes"].__setitem__("agreement", False),
            "ready artifact lacks cold replay agreement",
        ),
        (
            lambda value: value["diagnostic_headroom_by_order"]["order_1"].__setitem__(
                "accuracy_gap", 0.0
            ),
            "ready artifact lacks headroom",
        ),
    )
    for mutate, expected in validation_cases:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert any(expected in error for error in exp.validate_artifact(changed))

    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp.write_artifact(
            artifact_path=tmp_path / "replace-failure.json",
            repo_root=REPO_ROOT,
            duration_s=0.1,
        )
    assert not (tmp_path / "replace-failure.json.tmp").exists()

    blocked_validation_cases = (
        (lambda value: value.__setitem__("status", "bad"), "complete blocked status"),
        (lambda value: value.__setitem__("rows", [{}]), "blocked artifact must not contain rows"),
        (lambda value: value.__setitem__("verdict_class", "null"), "blocked verdict class"),
    )
    for mutate, expected in blocked_validation_cases:
        changed = deepcopy(blocked)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert any(expected in error for error in exp.validate_artifact(changed))


def test_req_cl_6790_defensive_reducers_and_build_failures_are_closed(
    source: exp.JsonDict,
    events: list[exp.JsonDict],
    orders: list[exp.JsonDict],
    rows: list[exp.JsonDict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6790 keeps malformed sources, rows, routes, and publication closed."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.evaluate_preconditions(source_artifact_path=malformed)["all_passed"] is False

    short_family = deepcopy(source)
    short_family["frozen_manifest"]["units"] = short_family["frozen_manifest"]["units"][1:]
    with pytest.raises(ValueError, match="32 units"):
        exp.build_events(short_family)

    missing_row = deepcopy(source)
    missing_row["rows"] = missing_row["rows"][1:]
    with pytest.raises(ValueError, match="missing source row"):
        exp.build_events(missing_row)

    monkeypatch.setattr(exp, "EVENT_COUNT", 241)
    with pytest.raises(ValueError, match="240 unique events"):
        exp.build_events(source)
    monkeypatch.setattr(exp, "EVENT_COUNT", 240)

    with pytest.raises(ValueError, match="unknown route"):
        exp.route_factor_ids(events[0], "unknown")

    nested_feature = deepcopy(rows[0])
    nested_feature["pre_action"]["legal_observation"]["candidate_graph"]["exact_label"] = False
    assert any(
        "candidate_graph.exact_label" in violation
        for violation in exp.audit_feature_contract([nested_feature])
    )

    duplicate_rows = [rows[0], rows[0]]
    assert "duplicate_row_keys" in exp.audit_row_consistency(
        duplicate_rows, events[:1], [exp.single_event_order(events[0])]
    )
    missing_order = exp.single_event_order(events[0])
    missing_order["event_ids"] = ["missing-event"]
    assert any(
        violation.startswith("missing:")
        for violation in exp.audit_row_consistency([], events[:1], [missing_order])
    )

    broken_row = deepcopy(rows[0])
    broken_row["chronology"]["position"] = 99
    broken_row["available_actions"] = ["local_prefix"]
    broken_row["chosen_baseline_actions"]["random_route"] = "dependency_suffix"
    broken_row["hidden_receipt_hash"] = "sha256:bad"
    broken_row["row_sha256"] = "sha256:bad"
    broken_row["route_cost"]["frozen_policy"] = 99
    broken_event = next(event for event in events if event["event_id"] == broken_row["event_id"])
    broken = exp.audit_row_consistency(
        [broken_row], [broken_event], [exp.single_event_order(broken_event)]
    )
    assert {item.split(":", 1)[0] for item in broken} >= {
        "position",
        "actions",
        "choice",
        "receipt",
        "hash",
        "cost",
    }

    real_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced validation"])
    with pytest.raises(ValueError, match="forced validation"):
        exp.build_artifact(
            repo_root=REPO_ROOT,
            duration_s=0.1,
            precondition_overrides={"constraint_group_fixture_ready": False},
        )

    monkeypatch.setattr(exp, "validate_artifact", real_validate)
    real_cold = exp.run_cold_replay
    monkeypatch.setattr(
        exp,
        "run_cold_replay",
        lambda built_events, built_orders, **_kwargs: {
            **exp.replay_hashes(exp.build_rows(built_events, built_orders), built_orders),
            "agreement": False,
            "fresh_process": True,
            "worker_exit_code": 0,
        },
    )
    internal_block = exp.build_artifact(repo_root=REPO_ROOT, duration_s=0.1)
    assert internal_block["gate_check_summary"]["first_failure"]["check"] == (
        "cold_replay_agreement"
    )

    monkeypatch.setattr(
        exp,
        "run_cold_replay",
        lambda built_events, built_orders, **_kwargs: {
            **exp.replay_hashes(exp.build_rows(built_events, built_orders), built_orders),
            "agreement": True,
            "fresh_process": True,
            "worker_exit_code": 0,
        },
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced ready validation"])
    with pytest.raises(ValueError, match="forced ready validation"):
        exp.build_artifact(repo_root=REPO_ROOT, duration_s=0.1)

    monkeypatch.setattr(exp, "run_cold_replay", real_cold)
    monkeypatch.setattr(exp, "validate_artifact", real_validate)
    terminal_block = exp.build_artifact(
        repo_root=REPO_ROOT,
        duration_s=0.1,
        precondition_overrides={"constraint_group_fixture_ready": False},
    )
    invalid_block = deepcopy(terminal_block)
    invalid_block.pop("schema")
    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: invalid_block)
    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.write_artifact(artifact_path=tmp_path / "invalid.json")

    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: terminal_block)
    written: list[Path] = []
    monkeypatch.setattr(exp, "_atomic_write", lambda path, _data: written.append(path))
    exp.write_artifact(artifact_path=Path("relative.json"), repo_root=tmp_path)
    assert written == [tmp_path / "relative.json"]

    monkeypatch.setattr(exp, "_cold_replay_worker", lambda: 0)
    assert exp.main(["--cold-replay-worker"]) == 0


def test_req_cl_6790_main_writes_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-6790 keeps the required dated command thin and deterministic."""

    output = tmp_path / "cli.json"
    assert exp.main(["--date", "20260830", "--output", str(output)]) == 0
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["constraint_routing_stream_ready"] is True
    assert capsys.readouterr().out.startswith("complete:")
