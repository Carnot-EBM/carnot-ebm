"""Tests for Exp 4703 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4703, SCENARIO-CAPSTONE-4703,
SCENARIO-CAPSTONE-4703-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4703-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4703_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_lower: float = 0.0,
    multi_level_rate: float = 0.0,
) -> JsonDict:
    return {
        "experiment": "experiment_4605_live_integration_scored_agent",
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "point": first_win_rate - mod.FIRST_WIN_BASELINE,
            "ci95": [ci_lower, max(ci_lower, first_win_rate - mod.FIRST_WIN_BASELINE)],
        },
        "multi_level_solve_rate": multi_level_rate,
        "integrated_measurement": {
            "variant_attempts": [
                {
                    "attempted": True,
                    "first_win": first_win_rate > 0.0,
                    "depth_reached": 2 if multi_level_rate > 0.0 else 1,
                }
            ],
        },
    }


def _floor(*, live_count: int = 61, reproduced: bool = True) -> JsonDict:
    return {
        "source_result_path": "results/experiment_4679_refresh_submission_package.json",
        "package_path": "results/experiment_4679_submission_package_operator_resubmit.json",
        "package_exists": True,
        "source_result_exists": True,
        "replay_package_floor_reproduced": reproduced,
        "live_submittable_level_count": live_count,
        "offline_reproduced": reproduced,
        "ready_for_operator_submit": live_count > 33 and reproduced,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "submitted_agent_config_importable": True,
        "experiment_4605_importable": True,
    }


def _parity(passed: bool = True) -> JsonDict:
    return {"passed": passed, "command": "pytest tests/python/test_arc_submitted_agent_parity.py -q"}


def test_req_capstone_4703_spec_declares_marker_hardened_readiness_gate() -> None:
    """REQ-CAPSTONE-4703: OpenSpec declares the null-delta marker contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4703",
        "SCENARIO-CAPSTONE-4703",
        "SCENARIO-CAPSTONE-4703-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4703-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4703_flat_first_win_emits_null_delta_markers() -> None:
    """SCENARIO-CAPSTONE-4703: a validated flat first-win is an explicit no-change."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0),
        replay_floor=_floor(live_count=61, reproduced=True),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_no_leaderboard_change"
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["first_win_baseline"] == 0.04
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["first_win_delta_vs_baseline"] == 0.0
    assert artifact["no_regression_vs_baseline"] is True
    assert artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert artifact["held_out_first_win_readiness"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is True
    assert artifact["replay_floor"]["live_submittable_level_count"] == 61
    assert artifact["replay_count_is_not_the_score"] is True
    assert "not the leaderboard score" in artifact["replay_floor"]["note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4703_success_requires_ci_and_parity() -> None:
    """SCENARIO-CAPSTONE-4703: improvement readiness needs parity and CI support."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02, multi_level_rate=0.04),
        replay_floor=_floor(live_count=0, reproduced=False),
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success: held_out_first_win_improved_ci_excludes_baseline"
    assert artifact["first_win_delta_vs_baseline"] == 0.08
    assert artifact["first_win_ci_lower"] == 0.02
    assert artifact["multi_level_deepen_rate_integrated"] == 0.04
    assert artifact["positive_control_passed"] is True
    assert artifact["null_delta_methodology_note"] == ""
    assert artifact["held_out_first_win_readiness"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is False
    assert artifact["replay_count_is_not_the_score"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4703_replay_count_cannot_mask_regression() -> None:
    """REQ-CAPSTONE-4703: replay package depth never creates readiness."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.0, ci_lower=-0.04),
        replay_floor=_floor(live_count=99, reproduced=True),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_below_baseline_no_leaderboard_change"
    assert artifact["no_regression_vs_baseline"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_floor"]["live_submittable_level_count"] == 99
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4703_parity_failure_invalidates_flat_exemption() -> None:
    """REQ-CAPSTONE-4703: unvalidated flat nulls are not excused."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(False),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0),
        replay_floor=_floor(live_count=61, reproduced=True),
        duration_s=1.0,
    )

    assert artifact["parity_test_green"] is False
    assert artifact["no_regression_vs_baseline"] is True
    assert artifact["positive_control_passed"] is False
    assert artifact["null_delta_methodology_note"]
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_unvalidated_no_leaderboard_change"
    assert mod.artifact_schema_errors(artifact) == []

    point_up = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(False),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02),
        replay_floor=_floor(live_count=61, reproduced=True),
        duration_s=1.0,
    )

    assert point_up["honest_verdict"] == (
        "complete: held_out_first_win_point_up_ci_supported_but_parity_failed"
    )
    assert point_up["positive_control_passed"] is False
    assert point_up["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(point_up) == []


def test_req_capstone_4703_schema_rejects_missing_flat_markers() -> None:
    """REQ-CAPSTONE-4703: flat first-win artifacts need note and positive control."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0),
        replay_floor=_floor(live_count=61, reproduced=True),
        duration_s=1.0,
    )
    bad = dict(artifact)
    bad["null_delta_methodology_note"] = ""
    bad["positive_control_passed"] = False
    bad["ready_for_operator_submit"] = False
    bad["replay_count_is_not_the_score"] = False
    bad["verifier_is_oracle"] = True
    bad["submitted_to_leaderboard"] = True
    bad["field_principles"] = {}
    bad["honest_verdict"] = "not_terminal"
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "held_out_first_win_readiness_gate" in errors
    assert "replay_count_is_not_the_score_true" in errors
    assert "verifier_is_oracle_false" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "null_delta_methodology_note" in errors
    assert "positive_control_passed" in errors
    assert "reproducibility_checksum" in errors

    no_regression_bad = dict(artifact)
    no_regression_bad["no_regression_vs_baseline"] = False
    no_regression_bad["reproducibility_checksum"] = mod.payload_checksum(no_regression_bad)
    assert "no_regression_vs_baseline" in mod.artifact_schema_errors(no_regression_bad)

    ready_bad = dict(artifact)
    ready_bad["ready_for_operator_submit"] = False
    ready_bad["reproducibility_checksum"] = mod.payload_checksum(ready_bad)
    assert "ready_for_operator_submit_gate" in mod.artifact_schema_errors(ready_bad)


def test_req_capstone_4703_fallback_extractors_and_schema_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4703: fallback proxy shapes remain auditable and gated."""

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("not-a-number", 3.0) == 3.0
    assert mod._extract_ci_lower({"first_win_ci_lower": 0.1234567}) == 0.123457
    assert (
        mod._extract_multi_level_deepen_rate(
            {"integrated_measurement": {"variant_attempts": [{"attempted": False}]}}
        )
        == 0.0
    )

    fallback_proxy = {
        "integrated_measurement": {
            "first_win_rate": 0.08,
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
        duration_s=1.0,
    )

    assert artifact["first_win_rate_integrated"] == 0.08
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["multi_level_deepen_rate_integrated"] == 0.5
    assert artifact["honest_verdict"] == (
        "complete: held_out_first_win_point_up_ci_overlaps_baseline_no_leaderboard_change"
    )
    assert artifact["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(artifact) == []

    stale = dict(artifact)
    stale["held_out_first_win_readiness"] = True
    stale["reproducibility_checksum"] = mod.payload_checksum(stale)
    assert "held_out_first_win_readiness_gate" in mod.artifact_schema_errors(stale)

    missing = dict(artifact)
    missing.pop("first_win_baseline")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing required field first_win_baseline" in mod.artifact_schema_errors(missing)

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, broken)


def test_scenario_capstone_4703_runner_writes_artifact_after_parity(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4703: runner writes the retargeted readiness artifact."""

    calls: list[str] = []

    def parity_check(_root: Path) -> JsonDict:
        calls.append("parity")
        return _parity(True)

    def proxy_runner(_root: Path, parity_test: JsonDict) -> JsonDict:
        calls.append("proxy")
        assert parity_test["passed"] is True
        return _proxy(first_win_rate=0.04, ci_lower=0.0)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=parity_check,
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: _floor(live_count=61, reproduced=True),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == ["parity", "proxy"]
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["ready_for_operator_submit"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4703_runner_skips_proxy_when_parity_red(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4703-BLOCKED-PRECONDITION: red parity avoids invalid proxy use."""

    def proxy_runner(_root: Path, _parity_test: JsonDict) -> JsonDict:
        raise AssertionError("proxy must not run when parity is red")

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(False),
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: _floor(live_count=61, reproduced=True),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["parity_test_green"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["first_win_rate_integrated"] == 0.0
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4703_blocked_precondition_writes_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4703-BLOCKED-PRECONDITION: missing imports block early."""

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "submitted_agent_config_import",
            "submitted_agent_config_importable": False,
            "experiment_4605_importable": True,
        },
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity_test: _proxy(first_win_rate=0.12, ci_lower=0.02),
        replay_floor_loader=lambda _root: _floor(live_count=61, reproduced=True),
        now=lambda: 5.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_submitted_agent_config_import"
    assert artifact["parity_test_green"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []
