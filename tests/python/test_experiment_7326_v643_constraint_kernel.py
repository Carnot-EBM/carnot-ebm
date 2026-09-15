"""Tests for the Exp7326 acquired-constraint kernel measurement."""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7326_v643_constraint_kernel as exp


def separation(
    constraint_id: str = "sep-a-b",
    *,
    version: str = "executor-v1",
    minimum: int = 2,
) -> dict[str, object]:
    """Return one serialized separation term used by both implementations."""

    return {
        "kind": "pairwise_separation",
        "constraint_id": constraint_id,
        "version": version,
        "left": "a",
        "right": "b",
        "minimum": minimum,
    }


def capacity(
    constraint_id: str = "cap",
    *,
    version: str = "executor-v1",
    window_size: int = 2,
    maximum: int = 2,
) -> dict[str, object]:
    """Return one serialized sliding-window capacity term."""

    return {
        "kind": "sliding_window_capacity",
        "constraint_id": constraint_id,
        "version": version,
        "window_size": window_size,
        "maximum": maximum,
    }


def request(*, constraints: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Return a fully supplied finite schedule request."""

    return {
        "schema": exp.REQUEST_SCHEMA,
        "executor_version": "executor-v1",
        "slot_min": 0,
        "slot_max": 3,
        "schedule": [
            {"activity": "a", "slot": 0},
            {"activity": "b", "slot": 2},
            {"activity": "c", "slot": 1},
            {"activity": "d", "slot": 2},
        ],
        "constraints": constraints if constraints is not None else [separation(), capacity()],
    }


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-ENERGY
def test_exact_integer_energies_include_boundary_and_overlapping_windows() -> None:
    result = exp.evaluate_request(request())

    assert result == {
        "valid_input": True,
        "feasible": False,
        "complete_oracle_certificate": False,
        "total_energy": 1,
        "terms": [
            {
                "constraint_id": "sep-a-b",
                "kind": "pairwise_separation",
                "energy": 0,
                "satisfied": True,
            },
            {
                "constraint_id": "cap",
                "kind": "sliding_window_capacity",
                "energy": 1,
                "satisfied": False,
            },
        ],
        "error": None,
    }

    equal = request(constraints=[separation(), capacity(maximum=3)])
    assert exp.evaluate_request(equal)["total_energy"] == 0
    assert exp.evaluate_request(equal)["feasible"] is True

    violated = request(constraints=[separation(minimum=3)])
    assert exp.evaluate_request(violated)["terms"][0]["energy"] == 1


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-INVALID
@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row.update(schedule=[]), "empty_schedule"),
        (
            lambda row: row["schedule"].append({"activity": "a", "slot": 1}),
            "duplicate_activity:a",
        ),
        (
            lambda row: row["schedule"][0].update(slot=9),
            "slot_out_of_domain:a",
        ),
        (
            lambda row: row["constraints"][0].update(version="executor-v2"),
            "version_mismatch:sep-a-b",
        ),
        (
            lambda row: row["constraints"][0].update(left="missing"),
            "missing_activity:sep-a-b:missing",
        ),
        (
            lambda row: row["constraints"][0].update(minimum=-1),
            "negative_minimum:sep-a-b",
        ),
        (
            lambda row: row["constraints"][1].update(window_size=0),
            "nonpositive_window:cap",
        ),
        (
            lambda row: row["constraints"][1].update(window_size=5),
            "window_exceeds_domain:cap",
        ),
        (
            lambda row: row["constraints"][1].update(maximum=-1),
            "negative_maximum:cap",
        ),
        (
            lambda row: (
                row.update(slot_min=exp.I64_MAX, slot_max=exp.I64_MAX)
                or row.update(schedule=[{"activity": "a", "slot": exp.I64_MAX}])
                or row.update(constraints=[capacity(window_size=1)])
            ),
            "window_end_overflow:cap",
        ),
    ],
)
def test_invalid_schedules_fail_closed(mutate: object, expected: str) -> None:
    row = request()
    mutate(row)  # type: ignore[operator]

    result = exp.evaluate_request(row)

    assert result["valid_input"] is False
    assert result["feasible"] is False
    assert result["complete_oracle_certificate"] is False
    assert result["total_energy"] is None
    assert result["terms"] == []
    assert result["error"] == expected


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-INVALID
def test_invalid_scalar_shapes_and_energy_overflow_fail_closed() -> None:
    malformed = request()
    malformed["slot_min"] = True
    assert exp.evaluate_request(malformed)["error"] == "invalid_integer:slot_min"

    unknown = request()
    unknown["constraints"] = [{"kind": "unknown", "constraint_id": "bad"}]
    assert exp.evaluate_request(unknown)["error"] == "unknown_constraint_kind:bad"

    overflow = request(constraints=[separation(minimum=exp.I64_MAX)])
    overflow["slot_min"] = 0
    overflow["slot_max"] = 0
    overflow["schedule"] = [
        {"activity": "a", "slot": 0},
        {"activity": "b", "slot": 0},
    ]
    assert exp.evaluate_request(overflow)["error"] == "energy_overflow:sep-a-b"


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-ENERGY
def test_empty_constraint_set_is_explicitly_incomplete_not_an_oracle_claim() -> None:
    row = request(constraints=[])
    result = exp.evaluate_request(row)

    assert result["valid_input"] is True
    assert result["feasible"] is True
    assert result["total_energy"] == 0
    assert result["complete_oracle_certificate"] is False


def make_atom(
    kind: str,
    payload: dict[str, object],
    *,
    version: str = "executor-v1",
) -> dict[str, object]:
    """Build an atom with the producer's exact hash contract."""

    body: dict[str, object] = {
        "kind": kind,
        "version": version,
        "payload": payload,
        "witness": {"query_id": "q1"},
        "query_receipts": [{"query_id": "q1"}],
    }
    return {**body, "atom_id": exp.sha256_json(body)}


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PREFLIGHT
def test_acquired_atoms_are_hash_authenticated_and_translated() -> None:
    sep = make_atom("pairwise_separation", {"pair": ["a", "b"], "minimum": 2})
    cap = make_atom("capacity", {"maximum": 2})
    rows = [
        {"arm": exp.PERSISTENT_ARM, "new_atoms": [sep, cap]},
        {"arm": "control", "new_atoms": [make_atom("capacity", {"maximum": 1})]},
    ]

    records, errors = exp.extract_acquired_records(rows)

    assert errors == []
    assert [row["atom_id"] for row in records] == [sep["atom_id"], cap["atom_id"]]
    terms = exp.constraints_by_version(records)["executor-v1"]
    assert terms == [
        {
            "kind": "pairwise_separation",
            "constraint_id": sep["atom_id"],
            "version": "executor-v1",
            "left": "a",
            "right": "b",
            "minimum": 2,
        },
        {
            "kind": "sliding_window_capacity",
            "constraint_id": cap["atom_id"],
            "version": "executor-v1",
            "window_size": 1,
            "maximum": 2,
        },
    ]

    changed = deepcopy(rows)
    changed[0]["new_atoms"][0]["payload"]["minimum"] = 3
    assert exp.extract_acquired_records(changed)[1] == [f"atom_hash:{sep['atom_id']}"]


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PARITY
def test_captured_plan_recovery_authenticates_hash_and_keeps_abstention() -> None:
    plan = {
        "request_id": "held-out-00-request-00",
        "assignments": {"a": 0, "b": 2, "c": 1, "d": 3},
    }
    row = {
        "request_id": plan["request_id"],
        "request_index": 0,
        "executor_version": "held-out-00-executor-a",
        "returned": True,
        "returned_plan_hash": exp.sha256_json(plan),
    }

    recovered = exp.recover_captured_request(row, [])
    assert recovered["schedule"] == [
        {"activity": "a", "slot": 0},
        {"activity": "b", "slot": 2},
        {"activity": "c", "slot": 1},
        {"activity": "d", "slot": 3},
    ]

    abstained = {**row, "returned": False, "returned_plan_hash": None}
    assert exp.recover_captured_request(abstained, [separation()])["schedule"] == []

    missing = {**row, "returned_plan_hash": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="captured_plan_hash_not_recovered"):
        exp.recover_captured_request(missing, [])


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PARITY
def test_seeded_fixture_budget_is_frozen_and_contains_required_cases() -> None:
    fixtures = exp.seeded_fixtures(7326002, count=1000)

    assert len(fixtures) == 1000
    assert fixtures == exp.seeded_fixtures(7326002, count=1000)
    assert {row["case"] for row in fixtures} >= {
        "valid",
        "invalid_slot",
        "empty_schedule",
        "version_mismatch",
        "overlapping_windows",
        "overflow",
    }
    assert all(row["fixture_id"] == f"seeded-{index:04d}" for index, row in enumerate(fixtures))


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PARITY
def test_batch_and_service_protocol_preserve_ordered_results() -> None:
    batch = [request(), request(constraints=[separation()])]
    assert exp.evaluate_batch(batch) == [exp.evaluate_request(row) for row in batch]
    assert exp.service_response({"operation": "ping"}) == {"kind": "ready"}
    assert exp.service_response({"operation": "evaluate", "requests": batch}) == {
        "results": exp.evaluate_batch(batch)
    }
    with pytest.raises(ValueError, match="unknown_operation"):
        exp.service_response({"operation": "bad"})


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PREFLIGHT
def test_preconditions_keep_exact_failed_value(tmp_path: Path) -> None:
    raw = tmp_path / "rows.jsonl"
    raw.write_text(json.dumps({"row": 1}) + "\n", encoding="utf-8")
    upstream = {
        "status": "complete",
        "addition_promotion_score": 1,
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "source_artifact_hashes": {
            "raw_evidence": {
                "rows": {
                    "path": str(raw),
                    "row_count": 1,
                    "sha256": exp.sha256_file(raw),
                }
            }
        },
    }
    upstream_path = tmp_path / "upstream.json"
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")

    checks, hashes, resolved_raw = exp.collect_preconditions(upstream_path)
    assert exp.gate_check_summary(checks)["passed"] is True
    assert hashes["upstream_artifact"] == exp.sha256_file(upstream_path)
    assert resolved_raw == raw

    upstream["verdict_class"] = "blocked"
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
    checks, _, _ = exp.collect_preconditions(upstream_path)
    summary = exp.gate_check_summary(checks)
    assert summary["passed"] is False
    assert summary["first_failure"] == {
        "upstream": str(upstream_path),
        "check": "upstream_not_blocked",
        "field": "verdict_class",
        "expected_value": "not blocked",
        "observed_value": "blocked",
        "passed": False,
        "principle": "An ineligible external class overrides a score of one.",
    }


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-COST
def test_paired_interval_and_terminal_scores_keep_performance_separate() -> None:
    interval = exp.paired_speedup_interval([12.0, 13.0, 14.0, 15.0], seed=7)
    assert interval["n"] == 4
    assert interval["point_estimate"] == pytest.approx(13.5)
    assert interval["ci95_lower"] <= interval["point_estimate"] <= interval["ci95_upper"]

    assert exp.derive_terminal_scores(parity_complete=True, costs_complete=True, speedup=False) == (
        1,
        0,
        "null",
    )
    assert exp.derive_terminal_scores(parity_complete=True, costs_complete=True, speedup=True) == (
        1,
        1,
        "circular_positive",
    )
    assert exp.derive_terminal_scores(parity_complete=False, costs_complete=True, speedup=True) == (
        0,
        0,
        "disqualified",
    )


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-HARDWARE
def test_hardware_projection_uses_exact_records_and_stays_software_only() -> None:
    records = [
        make_atom("pairwise_separation", {"pair": ["a", "b"], "minimum": 2}),
        make_atom("capacity", {"maximum": 2}),
    ]
    fixtures = [{"request": request()}]

    projection = exp.hardware_projection(records, fixtures)

    assert projection["classification"] == "software_projection_not_fpga_or_tsu_execution"
    assert projection["separation_term_count"] == 1
    assert projection["capacity_term_count"] == 1
    assert projection["sparse_coefficient_count"] == 6
    assert projection["maximum_coupling_degree"] == 3
    assert projection["serialized_constraint_bytes"] == len(exp.canonical_bytes(records))
    assert projection["measured_board_timing"] is None


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-TERMINAL
def test_artifact_validation_and_atomic_write(tmp_path: Path) -> None:
    artifact = exp.base_artifact([], {})
    artifact.update(
        {
            "status": "complete",
            "constraint_kernel_complete_score": 1,
            "kernel_speedup_score": 0,
            "verdict_class": "null",
            "honest_verdict": "complete_null: exact parity passed but the 10x cost gate did not",
            "rows": [{"arm": "python", "censored": False}],
            "kernel_rows": {"parity": [{"matched": True}], "cost": [{"complete": True}]},
            "sample_size_budget": {
                "captured_planned": 2304,
                "captured_attempted": 2304,
                "captured_complete": 2304,
                "captured_censored": 0,
                "seeded_planned": 1000,
                "seeded_attempted": 1000,
                "seeded_complete": 1000,
                "seeded_censored": 0,
                "paired_blocks_per_size": 30,
            },
            "acceptance_gate_results": {
                "parity": {"passed": True},
                "complete_cost_rows": {"passed": True},
                "speedup_lower_ci95": {"passed": False},
            },
            "required_checks_passed": True,
            "validation_receipts": [],
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["constraint_kernel_complete_score"] = 0
    assert "complete_score" in exp.validate_artifact(changed)

    output = tmp_path / "artifact.json"
    receipt = exp.write_artifact(output, artifact)
    assert receipt["path"] == str(output)
    assert receipt["sha256"] == exp.sha256_file(output)


def test_canonical_hash_is_order_stable_and_finite_domains_recover() -> None:
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert exp.canonical_bytes(left) == exp.canonical_bytes(right)
    assert exp.sha256_json(left) == exp.sha256_json(right)

    values = list(itertools.product(range(2), repeat=2))
    assert values == [(0, 0), (0, 1), (1, 0), (1, 1)]


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-INVALID
@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda row: row.update(schema="wrong"), "invalid_schema"),
        (lambda row: row.update(executor_version=""), "empty_executor_version"),
        (lambda row: row.update(slot_max=True), "invalid_integer:slot_max"),
        (lambda row: row.update(slot_min=4, slot_max=3), "invalid_slot_domain"),
        (lambda row: row.update(schedule=["bad"]), "invalid_assignment"),
        (lambda row: row["schedule"][0].update(activity=""), "empty_activity"),
        (lambda row: row["schedule"][0].update(slot=True), "invalid_integer:schedule:a"),
        (lambda row: row.update(constraints={}), "invalid_constraints"),
        (lambda row: row.update(constraints=["bad"]), "invalid_constraint"),
        (
            lambda row: row.update(
                constraints=[{"kind": "pairwise_separation", "constraint_id": ""}]
            ),
            "empty_constraint_id",
        ),
        (lambda row: row["constraints"][0].pop("left"), "missing_text:sep-a-b:left"),
        (lambda row: row["constraints"][0].pop("right"), "missing_text:sep-a-b:right"),
        (
            lambda row: row["constraints"][0].update(right="a"),
            "identical_pair:sep-a-b",
        ),
        (
            lambda row: row["constraints"][0].pop("minimum"),
            "missing_integer:sep-a-b:minimum",
        ),
        (
            lambda row: row["constraints"][0].update(right="missing"),
            "missing_activity:sep-a-b:missing",
        ),
        (
            lambda row: row["constraints"][1].pop("window_size"),
            "missing_integer:cap:window_size",
        ),
        (
            lambda row: row["constraints"][1].pop("maximum"),
            "missing_integer:cap:maximum",
        ),
    ],
)
def test_all_validation_boundaries_have_stable_errors(change: object, expected: str) -> None:
    row = request()
    change(row)  # type: ignore[operator]
    assert exp.evaluate_request(row)["error"] == expected


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-INVALID
def test_checked_total_energy_overflow_fails_closed() -> None:
    row = request(
        constraints=[
            separation("first", minimum=4_000_000_000),
            separation("second", minimum=4_000_000_000),
        ]
    )
    row["schedule"] = [{"activity": "a", "slot": 0}, {"activity": "b", "slot": 0}]
    assert exp.evaluate_request(row)["error"] == "total_energy_overflow:second"


def test_service_shutdown_and_invalid_evaluate_request() -> None:
    assert exp.service_response({"operation": "shutdown"}) == {"kind": "shutdown"}
    with pytest.raises(ValueError, match="invalid_requests"):
        exp.service_response({"operation": "evaluate", "requests": {}})


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PREFLIGHT
def test_json_inputs_and_absent_upstream_fail_without_placeholders(tmp_path: Path) -> None:
    object_path = tmp_path / "object.json"
    object_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not_json_object"):
        exp._read_object(object_path)

    checks, hashes, raw_path = exp.collect_preconditions(tmp_path / "missing.json")
    assert exp.gate_check_summary(checks)["first_failure"]["check"] == "upstream_available"
    assert hashes == {"upstream_artifact": None}
    assert raw_path == Path()

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('{"ok":true}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_jsonl_row"):
        exp.read_jsonl(rows_path)
    rows_path.write_text('{"ok":true}\n', encoding="utf-8")
    assert exp.read_jsonl(rows_path) == [{"ok": True}]


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PREFLIGHT
def test_malformed_and_unsupported_acquired_records_are_rejected() -> None:
    records, errors = exp.extract_acquired_records(
        [
            {"arm": exp.PERSISTENT_ARM, "new_atoms": {}},
            {"arm": exp.PERSISTENT_ARM, "new_atoms": ["bad"]},
        ]
    )
    assert records == []
    assert errors == ["invalid_atom", "invalid_new_atoms"]

    unsupported = make_atom("unsupported", {"value": 1})
    with pytest.raises(ValueError, match="unsupported_acquired_kind"):
        exp.constraints_by_version([unsupported])


def test_interval_empty_and_singleton_edges() -> None:
    assert exp._percentile([4.0], 0.5) == 4.0
    assert exp.paired_speedup_interval([], seed=1) == {
        "n": 0,
        "point_estimate": None,
        "ci95_lower": None,
        "ci95_upper": None,
    }


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-TERMINAL
def test_blocked_and_positive_artifact_score_consistency(tmp_path: Path) -> None:
    failure = {
        "upstream": "missing.json",
        "check": "upstream_available",
        "field": "path",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
        "principle": "required",
    }
    blocked = exp._mark_blocked(exp.base_artifact([failure], {}))
    assert exp.validate_artifact(blocked) == []

    broken_blocked = deepcopy(blocked)
    broken_blocked.update(
        {
            "rows": [{}],
            "constraint_kernel_complete_score": 1,
            "kernel_speedup_score": 1,
            "verdict_class": "null",
            "honest_verdict": "complete_null: wrong",
        }
    )
    broken_blocked["reproducibility_checksum"] = exp.reproducibility_checksum(broken_blocked)
    assert set(exp.validate_artifact(broken_blocked)) >= {
        "blocked_rows",
        "blocked_scores",
        "blocked_verdict",
        "blocked_prefix",
    }

    positive = exp.base_artifact([], {})
    positive.update(
        {
            "status": "complete",
            "constraint_kernel_complete_score": 1,
            "kernel_speedup_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete: exact parity and speed gate passed",
            "rows": [{}],
            "sample_size_budget": {
                "captured_complete": 2304,
                "seeded_complete": 1000,
                "paired_blocks_per_size": 30,
            },
            "acceptance_gate_results": {
                "parity": {"passed": True},
                "complete_cost_rows": {"passed": True},
                "speedup_lower_ci95": {"passed": True},
            },
        }
    )
    positive["reproducibility_checksum"] = exp.reproducibility_checksum(positive)
    assert exp.validate_artifact(positive) == []

    invalid_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(invalid_path, {"status": "complete"})


def test_internal_fixture_reduction_and_atomic_jsonl(tmp_path: Path, capsys: object) -> None:
    plan = {
        "request_id": "held-out-00-request-00",
        "assignments": {"a": 0, "b": 2, "c": 1, "d": 3},
    }
    rows = [
        {
            "request_id": plan["request_id"],
            "request_index": 0,
            "executor_version": "held-out-00-executor-a",
            "returned": True,
            "returned_plan_hash": exp.sha256_json(plan),
        }
    ]
    fixtures = exp._captured_fixtures(rows, {"held-out-00-executor-a": []})
    assert fixtures[0]["fixture_id"] == "captured-0000"

    path = tmp_path / "rows.jsonl"
    exp._atomic_jsonl(path, fixtures)
    assert exp.read_jsonl(path) == fixtures

    spans: list[dict[str, object]] = []
    start = exp.time.monotonic()
    exp._phase(spans, "unit", start, start, 1)
    assert spans[0]["phase"] == "unit"
    assert "phase=unit event=end" in capsys.readouterr().out  # type: ignore[attr-defined]

    hashes = exp._source_hashes(exp.REPO_ROOT)
    assert set(exp.RUST_PATHS).issubset(Path(path) for path in hashes)


def test_gate_shape_is_exact() -> None:
    assert exp._gate(1, 1, True, "same") == {
        "expected": 1,
        "observed": 1,
        "passed": True,
        "principle": "same",
    }


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-PARITY
def test_kernel_service_reserves_stdout_for_json_protocol(
    monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.setattr(
        exp,
        "_parse_args",
        lambda _argv=None: SimpleNamespace(kernel_service=True, validate=None, date=None),
    )
    monkeypatch.setattr(exp, "_serve", lambda: 0)

    assert exp.main(["--kernel-service"]) == 0
    assert capsys.readouterr().out == ""  # type: ignore[attr-defined]


# REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-TERMINAL
def test_scoped_validation_private_parent_is_prepared(tmp_path: Path) -> None:
    private_parent = tmp_path / "private" / "exp7326"

    assert exp.prepare_scoped_basetemp(private_parent) == private_parent
    assert private_parent.is_dir()
