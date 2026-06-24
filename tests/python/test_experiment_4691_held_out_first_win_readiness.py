"""Tests for Exp 4691 held-out first-win readiness retarget.

Spec refs: REQ-CAPSTONE-4691, SCENARIO-CAPSTONE-4691,
SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4691-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4691_held_out_first_win_readiness as mod


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


def _floor(*, live_count: int = 60, reproduced: bool = True) -> JsonDict:
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


def test_req_capstone_4691_spec_declares_retargeted_first_win_gate() -> None:
    """REQ-CAPSTONE-4691: OpenSpec declares the held-out first-win readiness gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4691",
        "SCENARIO-CAPSTONE-4691",
        "SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4691-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4691_flat_first_win_does_not_promote_replay_count() -> None:
    """SCENARIO-CAPSTONE-4691: replay count above 33 is only a floor, not readiness."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0),
        replay_floor=_floor(live_count=60, reproduced=True),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_no_leaderboard_change"
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["first_win_baseline"] == 0.04
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_package_floor_reproduced"] is True
    assert artifact["replay_floor"]["live_submittable_level_count"] == 60
    assert artifact["replay_count_is_not_the_score"] is True
    assert "no leaderboard-relevant change this milestone" in artifact["leaderboard_relevant_change_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4691_success_requires_ci_excluding_baseline() -> None:
    """SCENARIO-CAPSTONE-4691: first-win readiness needs parity and CI support."""

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
    assert artifact["held_out_first_win_readiness"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is False
    assert artifact["replay_count_is_not_the_score"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4691_parity_failure_invalidates_point_estimate() -> None:
    """REQ-CAPSTONE-4691: parity miss blocks readiness even with a higher point estimate."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(False),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02),
        replay_floor=_floor(live_count=60, reproduced=True),
        duration_s=1.0,
    )

    assert artifact["parity_test_green"] is False
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["honest_verdict"] == "complete: held_out_first_win_point_up_ci_supported_but_parity_failed"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4691_schema_rejects_replay_count_readiness() -> None:
    """REQ-CAPSTONE-4691: artifact validation rejects the retired count gate."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0),
        replay_floor=_floor(live_count=60, reproduced=True),
        duration_s=1.0,
    )
    bad = dict(artifact)
    bad["ready_for_operator_submit"] = True
    bad["replay_count_is_not_the_score"] = False
    bad["verifier_is_oracle"] = True
    bad["submitted_to_leaderboard"] = True
    bad["field_principles"] = {}
    bad["honest_verdict"] = "not_terminal"
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "ready_for_operator_submit_requires_held_out_first_win_readiness" in errors
    assert "replay_count_is_not_the_score_true" in errors
    assert "verifier_is_oracle_false" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "reproducibility_checksum" in errors


def test_req_capstone_4691_fallback_extractors_and_schema_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4691: fallback proxy shapes remain auditable and gated."""

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


def test_scenario_capstone_4691_runner_writes_artifact_after_parity(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4691: runner writes the retargeted readiness artifact."""

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
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == ["parity", "proxy"]
    assert written == artifact
    assert artifact["duration_s"] == 1.0
    assert artifact["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4691_runner_skips_proxy_when_parity_red(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION: red parity avoids invalid proxy use."""

    def proxy_runner(_root: Path, _parity_test: JsonDict) -> JsonDict:
        raise AssertionError("proxy must not run when parity is red")

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(False),
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["parity_test_green"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["first_win_rate_integrated"] == 0.0
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4691_blocked_precondition_writes_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION: missing imports block early."""

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
        replay_floor_loader=lambda _root: _floor(live_count=60, reproduced=True),
        now=lambda: 5.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_submitted_agent_config_import"
    assert artifact["parity_test_green"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []
