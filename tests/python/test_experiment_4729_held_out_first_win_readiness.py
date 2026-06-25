"""Tests for Exp 4729 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4729, SCENARIO-CAPSTONE-4729,
SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4729_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_lower: float = 0.0,
    attempts: int = 100,
    multi_level_rate: float = 0.0,
) -> JsonDict:
    solved = int(round(first_win_rate * attempts))
    deepened = int(round(multi_level_rate * attempts))
    rows = [
        {
            "attempted": True,
            "first_win": index < solved,
            "depth_reached": 2 if index < deepened else 1,
        }
        for index in range(attempts)
    ]
    return {
        "experiment": "experiment_4605_live_integration_scored_agent",
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "point": first_win_rate - mod.FIRST_WIN_BASELINE,
            "ci95": [ci_lower, max(ci_lower, first_win_rate - mod.FIRST_WIN_BASELINE)],
        },
        "multi_level_solve_rate": multi_level_rate,
        "integrated_measurement": {
            "variant_attempts_count": attempts,
            "variant_attempts": rows,
        },
    }


def _floor(*, live_count: int = 60, reproduced: bool = True) -> JsonDict:
    return {
        "package_path": "results/experiment_4679_submission_package_operator_resubmit.json",
        "package_exists": True,
        "replay_package_floor_reproduced": reproduced,
        "live_submittable_level_count": live_count,
        "offline_reproduced": reproduced,
        "ready_for_operator_submit": live_count > 33 and reproduced,
        "note": mod.REPLAY_FLOOR_NOTE,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "offline_arcade": True,
        "experiment_4605_importable": True,
    }


def _parity(passed: bool = True) -> JsonDict:
    return {
        "passed": passed,
        "command": "pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    }


def _lever_inputs() -> JsonDict:
    return {
        "a1": {
            "path": "results/experiment_4726_online_action_learning_driver_valid_test.json",
            "chosen_submitted_config": "unchanged",
        },
        "a2": {
            "path": "results/experiment_4727_active_probe_disambiguation.json",
            "chosen_submitted_config": "unchanged",
        },
    }


def test_req_capstone_4729_spec_declares_readiness_contract() -> None:
    """REQ-CAPSTONE-4729: OpenSpec declares the .435 held-out readiness gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4729",
        "SCENARIO-CAPSTONE-4729",
        "SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    assert "avoid redundant top-level numeric minimum fields" in spec


def test_scenario_capstone_4729_flat_first_win_emits_required_markers() -> None:
    """SCENARIO-CAPSTONE-4729: flat first-win is an explicit no-change."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_no_leaderboard_change"
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["first_win_baseline"] == 0.04
    assert artifact["first_win_delta_vs_baseline"] == 0.0
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["held_out_variant_attempts"] == 100
    assert "min_held_out_variant_attempts" not in artifact
    assert artifact["held_out_variant_attempt_floor"] == "B>=100"
    assert artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is True
    assert artifact["replay_count_is_not_the_score"] is True
    assert artifact["replay_floor"]["live_submittable_level_count"] == 60
    assert artifact["v435_lever_inputs"]["a1"]["chosen_submitted_config"] == "unchanged"
    assert artifact["v435_lever_inputs"]["a2"]["chosen_submitted_config"] == "unchanged"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4729_success_requires_ci_parity_and_b100() -> None:
    """SCENARIO-CAPSTONE-4729: readiness needs CI, parity, and B>=100."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(
            first_win_rate=0.12,
            ci_lower=0.02,
            attempts=125,
            multi_level_rate=0.04,
        ),
        replay_floor=_floor(live_count=0, reproduced=False),
        v435_lever_inputs=_lever_inputs(),
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success: held_out_first_win_improved_0.08"
    assert artifact["first_win_delta_vs_baseline"] == 0.08
    assert artifact["first_win_ci_lower"] == 0.02
    assert artifact["multi_level_deepen_rate_integrated"] == 0.04
    assert artifact["positive_control_passed"] is True
    assert artifact["null_delta_methodology_note"] == ""
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    undersized = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02, attempts=99),
        replay_floor=_floor(live_count=60, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert undersized["ready_for_operator_submit"] is False
    assert undersized["honest_verdict"] == "complete: held_out_first_win_measurement_below_b100"
    assert "held_out_variant_attempts_below_minimum" in mod.artifact_schema_errors(undersized)


def test_req_capstone_4729_replay_count_cannot_mask_regression() -> None:
    """REQ-CAPSTONE-4729: replay package floor never creates readiness."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.0, ci_lower=-0.04, attempts=100),
        replay_floor=_floor(live_count=99, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_below_baseline_no_leaderboard_change"
    assert artifact["positive_control_passed"] is True
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_floor"]["live_submittable_level_count"] == 99
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4729_parity_failure_invalidates_flat_exemption() -> None:
    """REQ-CAPSTONE-4729: positive control gates the flat-null exemption."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(False),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["parity_test_green"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["null_delta_methodology_note"]
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_unvalidated_no_leaderboard_change"
    assert mod.artifact_schema_errors(artifact) == []

    point_up = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(False),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert point_up["honest_verdict"] == (
        "complete: held_out_first_win_point_up_ci_supported_but_parity_failed"
    )
    assert point_up["positive_control_passed"] is False
    assert point_up["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(point_up) == []


def test_req_capstone_4729_schema_rejects_marker_and_shape_errors() -> None:
    """SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES: schema protects the artifact shape."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )
    bad = dict(artifact)
    bad["null_delta_methodology_note"] = ""
    bad["positive_control_passed"] = False
    bad["replay_count_is_not_the_score"] = False
    bad["verifier_is_oracle"] = True
    bad["submitted_to_leaderboard"] = True
    bad["field_principles"] = {}
    bad["honest_verdict"] = "not_terminal"
    bad["min_held_out_variant_attempts"] = 100
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "null_delta_methodology_note" in errors
    assert "positive_control_passed" in errors
    assert "replay_count_is_not_the_score_true" in errors
    assert "verifier_is_oracle_false" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "redundant_min_held_out_variant_attempts" in errors
    assert "reproducibility_checksum" in errors

    missing = dict(artifact)
    missing.pop("first_win_baseline")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing required field first_win_baseline" in mod.artifact_schema_errors(missing)

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), broken)


def test_scenario_capstone_4729_runner_sequences_parity_proxy_and_writes(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4729: runner writes after parity and proxy measurement."""

    calls: list[str] = []

    def parity_check(_root: Path) -> JsonDict:
        calls.append("parity")
        return _parity(True)

    def proxy_runner(_root: Path, parity_test: JsonDict) -> JsonDict:
        calls.append("proxy")
        assert parity_test["passed"] is True
        return _proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=parity_check,
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        lever_input_loader=lambda _root: _lever_inputs(),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == ["parity", "proxy"]
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["ready_for_operator_submit"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4729_blocked_precondition_writes_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION: missing resources block early."""

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "offline_arcade",
            "offline_arcade": False,
            "experiment_4605_importable": True,
        },
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity_test: _proxy(first_win_rate=0.12, ci_lower=0.02),
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        lever_input_loader=lambda _root: _lever_inputs(),
        now=lambda: 5.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_offline_arcade"
    assert artifact["parity_test_green"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4729_parity_red_and_b100_proxy_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION: parity and B>=100 failures block."""

    def proxy_must_not_run(_root: Path, _parity_test: JsonDict) -> JsonDict:
        raise AssertionError("proxy must not run after red parity")

    parity_blocked = mod.run(
        root=tmp_path / "parity",
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(False),
        proxy_runner=proxy_must_not_run,
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        lever_input_loader=lambda _root: _lever_inputs(),
        now=lambda: 8.0,
        sleep_fn=lambda _seconds: None,
    )

    assert parity_blocked["honest_verdict"] == "blocked_parity_test"
    assert parity_blocked["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(parity_blocked) == []

    undersized = mod.run(
        root=tmp_path / "b100",
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity_test: _proxy(
            first_win_rate=0.12,
            ci_lower=0.02,
            attempts=99,
        ),
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        lever_input_loader=lambda _root: _lever_inputs(),
        now=lambda: 9.0,
        sleep_fn=lambda _seconds: None,
    )

    assert undersized["honest_verdict"] == "blocked_experiment_4605_proxy_b100"
    assert undersized["preconditions_checked"]["held_out_variant_attempts"] == 99
    assert undersized["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(undersized) == []


def test_req_capstone_4729_default_proxy_forces_b100_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4729: default proxy asks Exp4605 for four color variants."""

    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    seen: dict[str, str | None] = {}

    def fake_run(*, root: Path, parity_check: Any) -> JsonDict:
        seen["deepen"] = os.environ.get(exp4605.DEEPEN_ENV)
        seen["variant_ids"] = os.environ.get(exp4605.VARIANT_IDS_ENV)
        assert parity_check(root)["passed"] is True
        return _proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100)

    monkeypatch.setattr(exp4605, "run", fake_run)
    monkeypatch.setenv(exp4605.DEEPEN_ENV, "old")
    monkeypatch.setenv(exp4605.VARIANT_IDS_ENV, "old")

    proxy = mod.run_held_out_proxy(REPO, _parity(True))

    assert proxy["first_win_rate_integrated"] == 0.04
    assert seen == {"deepen": "1", "variant_ids": "1,2,3,4"}
    assert os.environ[exp4605.DEEPEN_ENV] == "old"
    assert os.environ[exp4605.VARIANT_IDS_ENV] == "old"

    seen.clear()
    monkeypatch.delenv(exp4605.DEEPEN_ENV, raising=False)
    monkeypatch.delenv(exp4605.VARIANT_IDS_ENV, raising=False)

    proxy = mod.run_held_out_proxy(REPO, _parity(True))

    assert proxy["first_win_rate_integrated"] == 0.04
    assert seen == {"deepen": "1", "variant_ids": "1,2,3,4"}
    assert exp4605.DEEPEN_ENV not in os.environ
    assert exp4605.VARIANT_IDS_ENV not in os.environ


def test_req_capstone_4729_loaders_and_fallback_extractors(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4729: file loaders and fallback proxy shapes remain auditable."""

    for rel, payload in {
        "results/experiment_4726_online_action_learning_driver_valid_test.json": {
            "experiment": "experiment_4726_online_action_learning_driver_valid_test",
            "honest_verdict": "complete: a1",
            "chosen_submitted_config": "unchanged",
        },
        "results/experiment_4727_active_probe_disambiguation.json": {
            "experiment": "experiment_4727_active_probe_disambiguation",
            "honest_verdict": "complete: a2",
            "chosen_submitted_config": {"active_probe_controller": True},
        },
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = mod.load_v435_lever_inputs(tmp_path)
    assert set(loaded) == {"a1", "a2"}
    for row in loaded.values():
        assert row["exists"] is True
        assert row["sha256"]
        assert "error" not in row
    assert loaded["a1"]["chosen_submitted_config"] == "unchanged"
    assert loaded["a2"]["chosen_submitted_config"] == {"active_probe_controller": True}

    missing = mod.load_v435_lever_inputs(tmp_path / "missing")
    assert missing["a1"]["exists"] is False
    assert missing["a1"]["sha256"] == ""
    assert missing["a1"]["error"] == "missing_artifact"

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("not-a-number", 3.0) == 3.0
    assert mod._extract_first_win_rate({"first_win_rate_integrated": 0.12}) == 0.12
    assert mod._extract_first_win_rate({"integrated_measurement": {"first_win_rate": 0.08}}) == 0.08
    assert mod._extract_first_win_rate({}) == 0.0
    assert mod._extract_ci_lower({"first_win_ci": {"ci95": [0.03, 0.2]}}) == 0.03
    assert mod._extract_ci_lower({"first_win_ci_lower": 0.05}) == 0.05
    assert mod._extract_ci_lower({"first_win_ci": {"low": 0.01}}) == 0.01
    assert mod._extract_ci_lower({}) == 0.0
    assert mod._extract_multi_level_deepen_rate({"multi_level_solve_rate": 0.0}) == 0.0
    assert (
        mod._extract_multi_level_deepen_rate(
            {"integrated_measurement": {"variant_attempts": [{"attempted": False}]}}
        )
        == 0.0
    )
    assert mod._extract_held_out_variant_attempts({"held_out_variant_attempts": 123}) == 123
    assert (
        mod._extract_held_out_variant_attempts(
            {
                "integrated_measurement": {
                    "variant_attempts_count": "bad",
                    "variant_attempts": [{"attempted": True}],
                }
            }
        )
        == 1
    )

    fallback_proxy = {
        "integrated_measurement": {
            "first_win_rate": 0.08,
            "variant_attempts_count": 100,
            "variant_attempts": [
                {"attempted": True, "first_win": True, "depth_reached": "bad"},
                {"attempted": True, "solved": True, "depth_reached": 2},
                {"attempted": False, "first_win": True, "depth_reached": 9},
            ],
        },
        "first_win_ci": {"low": 0.0},
    }
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=fallback_proxy,
        replay_floor=_floor(live_count=0, reproduced=False),
        v435_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["first_win_rate_integrated"] == 0.08
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["multi_level_deepen_rate_integrated"] == 0.5
    assert artifact["honest_verdict"] == (
        "complete: held_out_first_win_point_up_ci_overlaps_baseline_no_leaderboard_change"
    )
    assert artifact["ready_for_operator_submit"] is False
