"""Tests for the bounded feasible-token flow canary.

Spec refs: REQ-REPORT-6596, SCENARIO-REPORT-6596-FIXTURES,
SCENARIO-REPORT-6596-CONTROLS, SCENARIO-REPORT-6596-ROWS,
SCENARIO-REPORT-6596-EXACT, SCENARIO-REPORT-6596-ROBUSTNESS,
SCENARIO-REPORT-6596-ATTACKS, SCENARIO-REPORT-6596-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6596_convergeflow_feasible_token_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _test_receipts() -> list[dict[str, object]]:
    return [
        {"command": command, "exit_code": 0, "duration_s": 0.01, "state": "passed"}
        for command in mod.VALIDATION_COMMANDS
    ]


def _checksum(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6596_spec_declares_full_contract() -> None:
    """REQ-REPORT-6596: OpenSpec owns fixtures, controls, rows, and fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6596") :]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-REPORT-6596-FIXTURES",
        "SCENARIO-REPORT-6596-CONTROLS",
        "SCENARIO-REPORT-6596-ROWS",
        "SCENARIO-REPORT-6596-EXACT",
        "SCENARIO-REPORT-6596-ROBUSTNESS",
        "SCENARIO-REPORT-6596-ATTACKS",
        "SCENARIO-REPORT-6596-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_report_6596_fixtures_are_frozen_and_nontrivial() -> None:
    """SCENARIO-REPORT-6596-FIXTURES: hashes bind all fixture inputs."""

    fixtures = mod.build_fixtures()
    assert len(fixtures) == 4
    ordinary = [fixture for fixture in fixtures if not fixture.expected_failure]
    empty = [fixture for fixture in fixtures if fixture.expected_failure]
    assert len(ordinary) == 3
    assert len(empty) == 1
    assert empty[0].feasible_token_ids == ()
    assert {fixture.dimension for fixture in ordinary} >= {2, 3}
    assert all(0 < len(fixture.feasible_token_ids) < len(fixture.token_ids) for fixture in ordinary)
    assert all(len(set(fixture.token_ids)) == len(fixture.token_ids) for fixture in fixtures)

    receipts = mod.build_embedding_and_constraint_receipts(fixtures)
    assert receipts["fixture_count"] == 4
    assert receipts["ordinary_fixture_count"] == 3
    assert receipts["expected_failure_fixture_count"] == 1
    assert receipts["all_fixture_hashes_present"] is True
    assert receipts["all_ordinary_sets_nontrivial"] is True
    assert receipts["empty_set_fails_closed"] is True
    assert len(receipts["start_rows"]) == 4 * len(mod.HELD_SEEDS)
    assert len(receipts["perturbation_rows"]) == (4 * len(mod.HELD_SEEDS) * len(mod.ERROR_LEVELS))
    assert all(row["start_hash"].startswith("sha256:") for row in receipts["start_rows"])
    assert all(
        row["perturbation_hash"].startswith("sha256:") for row in receipts["perturbation_rows"]
    )
    for row in receipts["fixture_rows"]:
        assert row["embedding_sha256"].startswith("sha256:")
        assert row["feasible_set_sha256"].startswith("sha256:")
        assert row["exact_automaton_sha256"].startswith("sha256:")
        transitions = row["exact_automaton_definition"]["transitions"]
        accepted = {
            transition["symbol"] for transition in transitions if transition["to_state"] == "accept"
        }
        assert accepted == set(row["feasible_token_ids"])

    starts = {
        (row["geometry_id"], row["seed"]): row["start_hash"] for row in receipts["start_rows"]
    }
    assert len(starts) == 4 * len(mod.HELD_SEEDS)
    for row in receipts["perturbation_rows"]:
        assert row["start_hash"] == starts[(row["geometry_id"], row["seed"])]
        expected = float(row["error_magnitude"])
        assert row["observed_error_magnitude"] == pytest.approx(expected, abs=1e-12)


def test_req_report_6596_convex_predictor_has_positive_feasible_weights() -> None:
    """REQ-REPORT-6596-ARMS: treatment stays inside the feasible hull."""

    fixture = mod.build_fixtures()[0]
    feasible = fixture.feasible_embeddings()
    state = np.asarray([0.21, -0.37], dtype=np.float64)
    raw = np.asarray([0.8, -0.5], dtype=np.float64)
    prediction, weights = mod.convex_hull_predictor(state, raw, feasible, sigma=0.4)
    assert prediction.shape == (fixture.dimension,)
    assert weights.shape == (len(fixture.feasible_token_ids),)
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    assert prediction == pytest.approx(weights @ feasible)

    with pytest.raises(mod.EmptyFeasibleSetError, match="empty feasible set"):
        mod.convex_hull_predictor(state, raw, np.empty((0, 2)), sigma=0.4)
    with pytest.raises(ValueError, match="positive"):
        mod.convex_hull_predictor(state, raw, feasible, sigma=0.0)
    with pytest.raises(ValueError, match="shape"):
        mod.convex_hull_predictor(np.ones(3), raw, feasible, sigma=0.4)
    with pytest.raises(ValueError, match="two-dimensional"):
        mod.convex_hull_predictor(state, raw, np.ones(2), sigma=0.4)


def test_scenario_report_6596_controls_emit_complete_matched_rows() -> None:
    """SCENARIO-REPORT-6596-CONTROLS: arms share inputs and charge work."""

    fixtures = mod.build_fixtures()
    rows = mod.evaluate_rows(fixtures)
    expected = len(fixtures) * len(mod.ERROR_LEVELS) * len(mod.HELD_SEEDS) * len(mod.ARMS)
    assert len(rows) == expected
    keys = {
        (
            row["geometry_id"],
            row["constraint_id"],
            row["error_id"],
            row["seed"],
            row["arm"],
        )
        for row in rows
    }
    assert len(keys) == expected
    assert all(set(mod.PER_UNIT_METRIC_FIELDS) <= set(row) for row in rows)

    by_unit: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["geometry_id"]), str(row["error_id"]), int(row["seed"]))
        by_unit.setdefault(key, []).append(row)
    assert all(len(group) == len(mod.ARMS) for group in by_unit.values())
    for group in by_unit.values():
        assert len({row["start_hash"] for row in group}) == 1
        assert len({row["perturbation_hash"] for row in group}) == 1
        assert len({row["raw_predictor_hash"] for row in group}) == 1
        assert len({row["integration_grid_hash"] for row in group}) == 1

        fixture_failure = bool(group[0]["expected_fixture_failure"])
        if fixture_failure:
            assert all(row["failure"] == "empty_feasible_set" for row in group)
            assert all(row["valid_endpoint"] is False for row in group)
            assert all(row["exact_constraint_result"]["accepted"] is False for row in group)
            continue

        arms = {str(row["arm"]): row for row in group}
        unconstrained = arms["unconstrained_flow"]
        nearest = arms["nearest_token_rounding"]
        treatment = arms["convex_hull_predictor_projection"]
        assert unconstrained["feasible_set_access_count"] == 0
        assert nearest["feasible_set_access_count"] == 0
        assert nearest["projection_calls"] == 0
        assert nearest["endpoint_snap_operations"] == 1
        assert nearest["path_length"] == pytest.approx(
            nearest["continuous_path_length"] + nearest["endpoint_snap_distance"]
        )
        assert treatment["projection_calls"] == mod.INTEGRATION_STEPS
        assert treatment["feasible_set_access_count"] == mod.INTEGRATION_STEPS
        assert treatment["endpoint_snap_operations"] == 0
        assert treatment["charged_work_units"] > nearest["charged_work_units"]
        assert all(row["steps"] == mod.INTEGRATION_STEPS for row in group)
        assert all(row["failure"] is None for row in group)


def test_scenario_report_6596_exact_checks_ignore_claimed_validity() -> None:
    """SCENARIO-REPORT-6596-EXACT: exact checks recompute endpoint validity."""

    fixtures = mod.build_fixtures()
    rows = mod.evaluate_rows(fixtures)
    checks = mod.build_exact_endpoint_check_rows(rows, fixtures)
    assert len(checks) == len(rows)
    checks_by_id = {row["row_id"]: row for row in checks}
    assert all(checks_by_id[row["row_id"]]["accepted"] == row["valid_endpoint"] for row in rows)
    assert all(check["accepted"] is False for check in checks if check["expected_fixture_failure"])

    changed = deepcopy(rows)
    changed[0]["valid_endpoint"] = not changed[0]["valid_endpoint"]
    repeated = mod.build_exact_endpoint_check_rows(changed, fixtures)
    assert repeated[0] == checks[0]

    off_token = deepcopy(rows[0])
    off_token["endpoint"] = [99.0] * len(off_token["endpoint"])
    off_token["endpoint_hash"] = mod.sha256_json(off_token["endpoint"])
    off_check = mod.build_exact_endpoint_check_rows([off_token], fixtures)[0]
    assert off_check["converged_to_token"] is False
    assert off_check["accepted"] is False


def test_scenario_report_6596_robustness_and_cost_stay_separate() -> None:
    """SCENARIO-REPORT-6596-ROBUSTNESS: validity cannot hide cost."""

    rows = mod.evaluate_rows(mod.build_fixtures())
    robustness = mod.build_robustness_summary(rows)
    cost = mod.build_distortion_and_cost_summary(rows)
    assert {row["predictor_condition"] for row in robustness} == {
        "clean",
        "held_perturbed",
    }
    assert {row["arm"] for row in robustness} == set(mod.ARMS)
    assert len(robustness) == 2 * len(mod.ARMS)
    assert len(cost) == 2 * len(mod.ARMS)
    for row in robustness:
        assert row["row_count"] > 0
        assert 0.0 <= row["validity_rate"] <= 1.0
        assert 0.0 <= row["convergence_rate"] <= 1.0
        assert row["expected_failure_count"] == len(mod.HELD_SEEDS) * (
            1 if row["predictor_condition"] == "clean" else len(mod.ERROR_LEVELS) - 1
        )
    for row in cost:
        assert row["mean_endpoint_distortion"] is not None
        assert row["mean_path_length"] is not None
        assert row["total_steps"] > 0
        assert row["total_charged_work_units"] >= row["total_steps"]
        assert row["wall_time_s"] >= 0.0


def test_req_report_6596_report_is_complete_and_circular_at_best() -> None:
    """REQ-REPORT-6596-VERDICT: exact-set success stays circular."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    assert report["status"] == "complete_fixture_evidence"
    assert report["honest_verdict"].startswith("complete:")
    assert "toy" in report["honest_verdict"]
    assert "language model" in report["honest_verdict"]
    assert report["verdict_class"] == "circular_positive"
    assert report["convergeflow_canary_ready_score"] == 1.0
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["method_source_receipt"]["source_sha256"] == mod.ARXIV_SOURCE_SHA256
    assert report["method_source_receipt"]["official_code_check"]["available"] is False
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["gate_check_summary"]["blocked"] is False
    assert report["duration_s"] == 0.5
    assert mod.validate_report(report, REPO) == []


def test_scenario_report_6596_attacks_and_validator_fail_closed() -> None:
    """SCENARIO-REPORT-6596-ATTACKS: each named shortcut is rejected."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    attacks = {row["attack_id"]: row for row in report["attack_rows"]}
    assert set(attacks) == set(mod.ATTACK_IDS)
    assert all(row["detected"] and row["failed_closed"] for row in attacks.values())

    bad = deepcopy(report)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "per_unit_rows key coverage mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    nearest = next(row for row in bad["per_unit_rows"] if row["arm"] == "nearest_token_rounding")
    nearest["projection_calls"] = 1
    assert "nearest-token control contaminated" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    nearest = next(row for row in bad["per_unit_rows"] if row["arm"] == "nearest_token_rounding")
    nearest["endpoint_snap_distance"] = 0.0
    assert "endpoint snap cost hidden" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    empty = next(row for row in bad["per_unit_rows"] if row["expected_fixture_failure"])
    empty["valid_endpoint"] = True
    assert "empty feasible set accepted" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["per_unit_rows"][0]["steps"] -= 1
    assert "post-outcome integration tuning" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["exact_endpoint_check_rows"] = []
    assert "exact_endpoint_check_rows mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["robustness_summary"] = []
    assert "robustness_summary mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["distortion_and_cost_summary"] = []
    assert "distortion_and_cost_summary mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["convergeflow_canary_ready_score"] = 0.0
    assert "ready score mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["verdict_class"] = "positive"
    assert "oracle-defined validity cannot be positive" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["inference_substrate"] = "live_llm"
    assert "inference_substrate mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["field_provenance"] = {}
    assert "field_provenance coverage mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["protected_files_unchanged"]["all_unchanged"] = False
    assert "protected files changed" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["attack_rows"] = []
    assert "attack_rows incomplete" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad, REPO)

    bad = deepcopy(report)
    del bad["status"]
    assert "missing required field: status" in mod.validate_report(bad, REPO)


def test_req_report_6596_blocked_precondition_names_observed_value() -> None:
    """REQ-REPORT-6596-PRECONDITIONS: a named failed check blocks rows."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.1,
        tests_run=_test_receipts(),
        precondition_overrides={"method_source_hash": False},
    )
    assert report["status"] == "blocked_precondition"
    assert report["verdict_class"] == "blocked"
    assert report["honest_verdict"].startswith("blocked_")
    assert report["per_unit_rows"] == []
    assert report["exact_endpoint_check_rows"] == []
    assert report["convergeflow_canary_ready_score"] == 0.0
    assert report["gate_check_summary"]["blocked"] is True
    failed = report["gate_check_summary"]["failed_checks"]
    assert failed[0]["check_id"] == "method_source_hash"
    assert failed[0]["observed_value"] is False
    assert mod.validate_report(report, REPO) == []

    bad = deepcopy(report)
    bad["gate_check_summary"]["failed_checks"] = []
    assert "blocked report lacks failed gate detail" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["per_unit_rows"] = [{"fabricated": True}]
    assert "blocked report fabricated per_unit_rows" in mod.validate_report(_checksum(bad), REPO)


def test_req_report_6596_fixture_and_arm_edges_fail_closed() -> None:
    """REQ-REPORT-6596-FIXTURES: malformed fixtures and arms do not run."""

    fixture = mod.build_fixtures()[0]
    with pytest.raises(ValueError, match="unknown arm"):
        mod.run_arm(fixture, mod.HELD_SEEDS[0], mod.ERROR_LEVELS[0], "bad_arm")

    duplicate = mod.GeometryFixture(
        geometry_id="duplicate",
        constraint_id="duplicate_set",
        dimension=2,
        token_ids=("a", "a"),
        embeddings=((0.0, 0.0), (1.0, 0.0)),
        feasible_token_ids=("a",),
        expected_failure=None,
    )
    with pytest.raises(ValueError, match="unique"):
        mod.build_embedding_and_constraint_receipts([duplicate])

    missing = mod.GeometryFixture(
        geometry_id="missing",
        constraint_id="missing_set",
        dimension=2,
        token_ids=("a", "b"),
        embeddings=((0.0, 0.0), (1.0, 0.0)),
        feasible_token_ids=("c",),
        expected_failure=None,
    )
    with pytest.raises(ValueError, match="vocabulary"):
        mod.build_embedding_and_constraint_receipts([missing])

    bad_shape = mod.GeometryFixture(
        geometry_id="bad_shape",
        constraint_id="bad_shape_set",
        dimension=3,
        token_ids=("a", "b"),
        embeddings=((0.0, 0.0), (1.0, 0.0)),
        feasible_token_ids=("a",),
        expected_failure=None,
    )
    with pytest.raises(ValueError, match="embedding shape"):
        mod.build_embedding_and_constraint_receipts([bad_shape])

    undeclared_empty = mod.GeometryFixture(
        geometry_id="undeclared_empty",
        constraint_id="undeclared_empty_set",
        dimension=2,
        token_ids=("a", "b"),
        embeddings=((0.0, 0.0), (1.0, 0.0)),
        feasible_token_ids=(),
        expected_failure=None,
    )
    with pytest.raises(ValueError, match="declare its expected failure"):
        mod.build_embedding_and_constraint_receipts([undeclared_empty])


def test_req_report_6596_readiness_rejects_each_control_mutation() -> None:
    """REQ-REPORT-6596-READINESS: incomplete or contaminated replay scores zero."""

    fixtures = mod.build_fixtures()
    rows = mod.evaluate_rows(fixtures)
    checks = mod.build_exact_endpoint_check_rows(rows, fixtures)
    assert mod.compute_ready_score(rows, checks) == 1.0
    assert mod.compute_ready_score(rows, checks[:-1]) == 0.0

    bad = deepcopy(rows)
    empty = next(row for row in bad if row["expected_fixture_failure"])
    empty["failure"] = None
    assert mod.compute_ready_score(bad, checks) == 0.0

    bad = deepcopy(rows)
    control = next(row for row in bad if row["arm"] == "unconstrained_flow")
    control["feasible_set_access_count"] = 1
    assert mod.compute_ready_score(bad, checks) == 0.0

    bad = deepcopy(rows)
    treatment = next(row for row in bad if row["arm"] == "convex_hull_predictor_projection")
    treatment["projection_calls"] -= 1
    assert mod.compute_ready_score(bad, checks) == 0.0


def test_req_report_6596_validator_rejects_enum_leakage_and_blocked_score() -> None:
    """REQ-REPORT-6596-VERDICT: enum, control, and blocked score checks bite."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    bad = deepcopy(report)
    bad["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    unconstrained = next(row for row in bad["per_unit_rows"] if row["arm"] == "unconstrained_flow")
    unconstrained["feasible_set_access_count"] = 1
    assert "feasible-set leakage into unconstrained control" in mod.validate_report(
        _checksum(bad), REPO
    )

    blocked = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.1,
        tests_run=_test_receipts(),
        precondition_overrides={"method_source_hash": False},
    )
    blocked["convergeflow_canary_ready_score"] = 1.0
    assert "blocked report has nonzero ready score" in mod.validate_report(_checksum(blocked), REPO)


def test_req_report_6596_resource_fallback_and_current_hash_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6596-PRECONDITIONS: resource and hash failures stay explicit."""

    def unavailable_sysconf(name: str) -> int:
        del name
        raise ValueError("unavailable")

    monkeypatch.setattr(mod.os, "sysconf", unavailable_sysconf)
    resources = mod.resource_receipt(REPO)
    assert resources["ram_total_bytes"] is None
    assert resources["ram_available_bytes"] is None

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    bad = deepcopy(report)
    bad["protected_files_unchanged"]["rows"][0]["after_sha256"] = "sha256:bad"
    bad["protected_files_unchanged"]["all_unchanged"] = True
    assert "protected file current hash mismatch" in mod.validate_report(_checksum(bad), REPO)


def test_scenario_report_6596_atomic_write_and_existing_receipts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6596-ATOMIC: output syncs, validates, and reloads."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    output = tmp_path / "canary.json"
    receipt = mod.atomic_write_report(output, report, repo_root=REPO)
    assert receipt["file_fsync"] is True
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["output_sha256"].startswith("sha256:")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report
    assert mod.existing_test_receipts(output) == _test_receipts()
    assert mod.existing_test_receipts(tmp_path / "missing.json") == list(mod.DEFAULT_TESTS_RUN)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.existing_test_receipts(malformed) == list(mod.DEFAULT_TESTS_RUN)
    malformed.write_text(json.dumps({"tests_run": ["not-a-row"]}), encoding="utf-8")
    assert mod.existing_test_receipts(malformed) == list(mod.DEFAULT_TESTS_RUN)

    invalid = deepcopy(report)
    del invalid["status"]
    with pytest.raises(ValueError, match="missing required field: status"):
        mod.atomic_write_report(tmp_path / "invalid.json", invalid, repo_root=REPO)
