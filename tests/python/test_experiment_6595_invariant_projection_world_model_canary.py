"""Tests for the frozen world-model invariant-projection canary.

Spec refs: REQ-REPORT-6595, SCENARIO-REPORT-6595-CALIBRATION,
SCENARIO-REPORT-6595-FROZEN, SCENARIO-REPORT-6595-CONTROLS,
SCENARIO-REPORT-6595-ROWS, SCENARIO-REPORT-6595-POSITIVE,
SCENARIO-REPORT-6595-ATTACKS, SCENARIO-REPORT-6595-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6595_invariant_projection_world_model_canary as mod
import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _test_receipts() -> list[dict[str, object]]:
    return [
        {"command": command, "exit_code": 0, "duration_s": 0.01}
        for command in mod.VALIDATION_COMMANDS
    ]


def _checksum(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6595_spec_declares_full_contract() -> None:
    """REQ-REPORT-6595: OpenSpec owns fields, controls, and attacks."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6595") :]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-REPORT-6595-CALIBRATION",
        "SCENARIO-REPORT-6595-FROZEN",
        "SCENARIO-REPORT-6595-CONTROLS",
        "SCENARIO-REPORT-6595-ROWS",
        "SCENARIO-REPORT-6595-POSITIVE",
        "SCENARIO-REPORT-6595-ATTACKS",
        "SCENARIO-REPORT-6595-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_req_report_6595_analytic_fixtures_and_split_receipts() -> None:
    """REQ-REPORT-6595-FIXTURES: matched fixtures have analytic checks."""

    fixtures = mod.build_fixtures()
    assert len(fixtures) == 4
    assert {fixture.regime for fixture in fixtures} == {"conservative", "damped"}
    assert all(mod.analytic_transition_check(fixture)["passed"] for fixture in fixtures)
    assert {fixture.matched_pair_id for fixture in fixtures} == {
        "isotropic_oscillator",
        "elliptic_oscillator",
    }

    receipts = mod.build_fixture_and_split_receipts(fixtures)
    assert receipts["fixture_count"] == 4
    assert receipts["all_analytic_checks_passed"] is True
    assert receipts["calibration_split_hash"].startswith("sha256:")
    assert receipts["held_split_hash"].startswith("sha256:")
    assert receipts["calibration_and_held_disjoint"] is True
    assert len(receipts["calibration_membership"]) >= len(mod.CALIBRATION_SEEDS)
    assert len(receipts["held_membership"]) == 4 * len(mod.HELD_SEEDS)


def test_scenario_report_6595_calibration_only_selection_is_frozen() -> None:
    """SCENARIO-REPORT-6595-CALIBRATION: held data cannot tune selection."""

    fixtures = mod.build_fixtures()
    selection = mod.select_invariants(fixtures)
    rows = selection["rows"]
    assert len(rows) == len(fixtures) * len(mod.CANDIDATE_FAMILIES) * len(mod.CALIBRATION_SEEDS)
    assert all(row["data_scope"] == "calibration_only" for row in rows)
    assert all(row["held_outcomes_used"] == 0 for row in rows)
    assert all(row["capacity"] <= mod.MAX_INVARIANT_CAPACITY for row in rows)

    selected = selection["selected_by_fixture"]
    conservative = [row for row in selected if row["regime"] == "conservative"]
    damped = [row for row in selected if row["regime"] == "damped"]
    assert all(row["selected"] for row in conservative)
    assert all(row["candidate_family"] == "quadratic_full" for row in conservative)
    assert all(not row["selected"] for row in damped)
    assert all(row["selection_reason"] == "no_comparable_conserved_candidate" for row in damped)
    assert all(row["invariant_sha256"].startswith("sha256:") for row in selected)

    changed_held = mod.build_held_inputs(fixtures, disturbance_scale=0.9)
    repeated = mod.select_invariants(fixtures, forbidden_held_inputs=changed_held)
    assert repeated["selection_hash"] == selection["selection_hash"]


def test_scenario_report_6595_controls_emit_complete_matched_rows() -> None:
    """SCENARIO-REPORT-6595-CONTROLS: all arms share held inputs."""

    fixtures = mod.build_fixtures()
    selection = mod.select_invariants(fixtures)
    rows = mod.evaluate_held_rollouts(fixtures, selection)
    expected = len(fixtures) * len(mod.HORIZONS) * len(mod.HELD_SEEDS) * len(mod.ARMS)
    assert len(rows) == expected
    assert {row["arm"] for row in rows} == set(mod.ARMS)
    assert all(set(mod.PER_UNIT_METRIC_FIELDS) <= set(row) for row in rows)

    by_unit: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["fixture_id"], row["regime"], row["horizon"], row["seed"])
        by_unit.setdefault(key, []).append(row)
    assert all(len(group) == len(mod.ARMS) for group in by_unit.values())
    for group in by_unit.values():
        assert len({row["initial_state_hash"] for row in group}) == 1
        assert len({row["disturbance_hash"] for row in group}) == 1
        random_row = next(row for row in group if row["arm"] == "norm_matched_random_projection")
        assert random_row["constraint_norm_match_error"] <= mod.RANDOM_NORM_TOLERANCE
        exact_row = next(row for row in group if row["arm"] == "exact_invariant_diagnostic")
        assert exact_row["headline_eligible"] is False
    assert all(row["failure"] is None for row in rows)


def test_scenario_report_6595_positive_recomputes_without_exact_headline() -> None:
    """SCENARIO-REPORT-6595-POSITIVE: learned held benefit clears all gates."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )
    assert report["status"] == "complete_held_comparative_evidence"
    assert report["verdict_class"] == "positive"
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert report["honest_verdict"].startswith("complete:")
    assert "random" in report["honest_verdict"]
    assert "damped" in report["honest_verdict"]

    summaries = mod.build_arm_summary_rows(report["per_unit_rows"])
    assert report["arm_summary_rows"] == summaries
    assert len(summaries) == 8
    paired = {row["comparison_id"]: row for row in report["paired_statistical_receipts"]}
    assert paired["conservative_learned_vs_no_projection"]["interval"]["lower"] > 0.0
    assert paired["conservative_learned_vs_random"]["interval"]["lower"] > 0.0
    assert paired["damped_learned_vs_no_projection"]["effect"] == pytest.approx(0.0)
    assert paired["damped_learned_vs_no_projection"]["ties"] == 30

    gates = {row["gate_id"]: row for row in report["acceptance_gate_rows"]}
    assert all(row["passed"] for row in gates.values())
    assert gates["learned_held_conservative_improvement"]["headline_arm"] == (
        "learned_invariant_projection"
    )
    assert (
        "exact_invariant_diagnostic"
        not in gates["learned_held_conservative_improvement"]["observed_source_arms"]
    )
    assert report["conservative_damped_specificity"]["passed"] is True
    assert report["frozen_model_receipts"]["all_frozen_unchanged"] is True
    assert report["method_source_receipt"]["arxiv_source_sha256"] == (
        "sha256:ab085934e654cf45efe405440a33db45792e8062ef542d5b8fddf5ba3f1d5237"
    )
    assert report["method_source_receipt"]["method_hash"].startswith("sha256:")
    assert mod.validate_report(report, REPO) == []


def test_scenario_report_6595_attacks_and_validator_fail_closed() -> None:
    """SCENARIO-REPORT-6595-ATTACKS: every shortcut has a detector."""

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
    random_row = next(
        row for row in bad["per_unit_rows"] if row["arm"] == "norm_matched_random_projection"
    )
    random_row["constraint_norm_match_error"] = 1.0
    assert "random constraint norm mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["frozen_model_receipts"]["all_frozen_unchanged"] = False
    assert "frozen model or invariant changed" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["arm_summary_rows"] = []
    assert "arm_summary_rows mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["acceptance_gate_rows"][0]["passed"] = False
    assert "acceptance_gate_rows mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["field_provenance"] = {}
    assert "field_provenance coverage mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["protected_files_unchanged"]["all_unchanged"] = False
    assert "protected files changed" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad, REPO)

    bad = deepcopy(report)
    del bad["status"]
    assert "missing required field: status" in mod.validate_report(bad, REPO)


def test_req_report_6595_blocked_precondition_names_observed_value() -> None:
    """REQ-REPORT-6595-PRECONDITIONS: numerical failure blocks by name."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.1,
        tests_run=_test_receipts(),
        precondition_overrides={"analytic_transition_checks": False},
    )
    assert report["status"] == "blocked_precondition"
    assert report["verdict_class"] == "blocked"
    assert report["per_unit_rows"] == []
    assert report["gate_check_summary"]["blocked"] is True
    assert report["gate_check_summary"]["failed_checks"][0]["check_id"] == (
        "analytic_transition_checks"
    )
    assert report["gate_check_summary"]["failed_checks"][0]["observed_value"] is False
    assert report["honest_verdict"].startswith("blocked_")
    assert mod.validate_report(report, REPO) == []


def test_req_report_6595_projection_and_statistics_edge_cases() -> None:
    """REQ-REPORT-6595-STATISTICS: numerical edges remain explicit."""

    assert mod.paired_interval([])["underpowered"] is True
    with pytest.raises(ValueError, match="resamples must be positive"):
        mod.paired_interval([1.0], resamples=0)
    with pytest.raises(ValueError, match="quadratic matrix"):
        mod.project_to_level_set(np.ones(2), np.ones((2, 3)), 1.0)

    zero = mod.project_to_level_set(np.zeros(2), np.eye(2), 1.0)
    assert zero["converged"] is False
    assert zero["failure"] == "zero_gradient"
    assert zero["iterations"] == 0

    already_level = mod.project_to_level_set(np.ones(2), np.eye(2), 2.0, max_iterations=0)
    assert already_level["converged"] is True
    maxed = mod.project_to_level_set(np.ones(2), np.eye(2), 3.0, max_iterations=0)
    assert maxed["failure"] == "max_iterations"

    with pytest.raises(ValueError, match="unknown candidate family"):
        mod._candidate_features(np.ones((2, 2)), "unknown")
    with pytest.raises(ValueError, match="unknown candidate family"):
        mod._coefficient_to_matrix("unknown", np.ones(3))


def test_req_report_6595_projection_failures_and_resource_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6595-PRECONDITIONS: failures and CPU probes remain explicit."""

    fixture = mod.build_fixtures()[0]
    selected = mod.select_invariants([fixture])["selected_by_fixture"][0]
    held_input = mod.build_held_inputs([fixture])[0]

    def failed_projection(
        state: np.ndarray, quadratic_matrix: np.ndarray, target: float
    ) -> dict[str, object]:
        del quadratic_matrix, target
        return {
            "state": state,
            "distance": 0.0,
            "iterations": 0,
            "converged": False,
            "failure": "forced_failure",
        }

    monkeypatch.setattr(mod, "project_to_level_set", failed_projection)
    row = mod._rollout_row(
        fixture,
        selected,
        held_input,
        1,
        "learned_invariant_projection",
    )
    assert row["failure"] == "forced_failure"
    assert row["failures"] == 1

    def unavailable_sysconf(name: str) -> int:
        del name
        raise ValueError("unavailable")

    monkeypatch.setattr(mod.os, "sysconf", unavailable_sysconf)
    resources = mod._resource_receipt()
    assert resources["ram_total_bytes"] is None
    assert resources["ram_available_bytes"] is None


def test_scenario_report_6595_atomic_write_and_existing_test_receipts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6595-ATOMIC: durable output validates and reloads."""

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


def test_req_report_6595_quadratic_fit_rejects_bad_trajectory_shapes() -> None:
    """REQ-REPORT-6595-CALIBRATION: malformed calibration data fails closed."""

    with pytest.raises(ValueError, match="at least two trajectories"):
        mod.fit_candidate_family([np.zeros((3, 2))], "quadratic_full")
    with pytest.raises(ValueError, match="unknown candidate family"):
        mod.fit_candidate_family([np.zeros((3, 2)), np.ones((3, 2))], "unknown")
    with pytest.raises(ValueError, match="shape"):
        mod.fit_candidate_family([np.zeros((3, 3)), np.ones((3, 3))], "quadratic_full")


def test_req_report_6595_validator_rejects_all_decision_tampering() -> None:
    """REQ-REPORT-6595-ATTACKS: each decision-bearing reducer fails closed."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
    )

    bad = deepcopy(report)
    bad["protected_files_unchanged"]["rows"][0]["after_sha256"] = "sha256:bad"
    assert "protected file current hash mismatch" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["attack_rows"] = []
    assert "attack_rows incomplete" in mod.validate_report(_checksum(bad), REPO)

    blocked = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.1,
        tests_run=_test_receipts(),
        precondition_overrides={"analytic_transition_checks": False},
    )
    bad = deepcopy(blocked)
    bad["gate_check_summary"]["failed_checks"] = []
    assert "blocked report lacks failed gate detail" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(blocked)
    bad["per_unit_rows"] = [{"fabricated": True}]
    assert "blocked report fabricated per_unit_rows" in mod.validate_report(_checksum(bad), REPO)

    bad = deepcopy(report)
    bad["paired_statistical_receipts"] = []
    assert "paired_statistical_receipts mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["conservative_damped_specificity"]["passed"] = False
    assert "conservative_damped_specificity mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["verdict_class"] = "null"
    assert "verdict_class mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["invariant_selection_rows"][0]["held_outcomes_used"] = 1
    assert "held leakage in invariant selection" in mod.validate_report(_checksum(bad), REPO)


def test_req_report_6595_substrate_has_reviewed_duration_floor() -> None:
    """REQ-REPORT-6595-VERDICT: deterministic CPU evaluation has a duration floor."""

    floor = adversarial_verify.duration_floor_for_artifact(
        {"inference_substrate": mod.INFERENCE_SUBSTRATE}
    )
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.DETERMINISTIC_VERIFIER_MIN_DURATION_S,
        "reason": "deterministic_verifier",
    }
