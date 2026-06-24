"""Tests for Exp 4658 value-routing CI gate and residual diagnostic.

Spec refs: REQ-LEARN-4658, SCENARIO-LEARN-4658-CIGATE,
SCENARIO-LEARN-4658-DIAGNOSTIC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
A1_PATH = REPO / "results" / "experiment_4652_value_routing_cost_fix_live.json"


def _attempt(
    signature: str,
    *,
    first_win: bool,
    reached_level: int,
    actions: int = 100,
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "attempted": True,
        "variant_signature": signature,
        "first_win": first_win,
        "solved": first_win,
        "reached_level": reached_level,
        "actions": actions,
        "actions_to_first_levelup": actions if first_win else None,
        "timed_out": timed_out,
    }


def _measurement(
    attempts: list[dict[str, Any]],
    *,
    first_win_rate: float,
    solve_rate: float,
) -> dict[str, Any]:
    return {
        "variant_attempts": attempts,
        "variant_attempts_count": len(attempts),
        "variant_signatures": [str(row["variant_signature"]) for row in attempts],
        "first_win_rate": first_win_rate,
        "solve_rate": solve_rate,
        "timed_out_attempts": sum(1 for row in attempts if row.get("timed_out") is True),
    }


def _a1_artifact() -> dict[str, Any]:
    attempts = [
        _attempt("aa00~color01", first_win=True, reached_level=1, actions=8),
        _attempt("bb00~color01", first_win=False, reached_level=0, actions=100),
    ]
    measurement = _measurement(attempts, first_win_rate=0.5, solve_rate=0.0)
    return {
        "honest_verdict": "complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration.",
        "verifier_is_oracle": False,
        "sim_timed_out": False,
        "per_node_feature_cost_ms": 0.42,
        "value_weight_set": 1e-12,
        "live_first_win_rate_value_routed": 0.5,
        "live_solve_rate_value_routed": 0.0,
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "residual_cause_hypothesis": "distribution_shift_or_calibration",
        "value_routed_measurement": measurement,
        "baseline_measurement": measurement,
        "live_baseline_value_weight_zero": {"first_win_rate": 0.5, "solve_rate": 0.0},
        "random_seed": 4652,
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "a1_artifact_present": True,
        "agentic_imports": True,
        "spec_has_req_4658": True,
        "live_llm_inference": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_learn_4658_spec_declares_cigate_and_diagnostic_contract() -> None:
    """REQ-LEARN-4658: OpenSpec declares the CI gate and diagnostic schema."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4658" in spec
    assert "SCENARIO-LEARN-4658-CIGATE" in spec
    assert "SCENARIO-LEARN-4658-DIAGNOSTIC" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_learn_4658_real_a1_artifact_passes_ci_gate() -> None:
    """SCENARIO-LEARN-4658-CIGATE: the checked-in A1 floor is an active pytest gate."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    a1 = json.loads(A1_PATH.read_text(encoding="utf-8"))
    gate = mod.evaluate_value_routing_cigate(a1)

    assert gate["passed"] is True
    assert gate["errors"] == []
    assert gate["sim_timed_out"] is False
    assert gate["value_routed_attempt_count"] == 25
    assert gate["first_win_rate"] >= gate["first_win_floor"]
    assert gate["solve_rate"] >= gate["solve_rate_floor"]


def test_scenario_learn_4658_ci_gate_fails_closed_on_timeout_or_floor_breach() -> None:
    """SCENARIO-LEARN-4658-CIGATE: timeout and metric regression are hard failures."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    ok = mod.evaluate_value_routing_cigate(_a1_artifact(), expected_attempts=2)
    assert ok["passed"] is True

    timed_out = {**_a1_artifact(), "sim_timed_out": True}
    timeout_gate = mod.evaluate_value_routing_cigate(timed_out, expected_attempts=2)
    assert timeout_gate["passed"] is False
    assert "sim_timed_out" in timeout_gate["errors"]

    attempt_timeout = _a1_artifact()
    attempt_timeout["value_routed_measurement"] = _measurement(
        [
            _attempt("aa00~color01", first_win=True, reached_level=1),
            _attempt("bb00~color01", first_win=False, reached_level=0, timed_out=True),
        ],
        first_win_rate=0.5,
        solve_rate=0.0,
    )
    attempt_gate = mod.evaluate_value_routing_cigate(attempt_timeout, expected_attempts=2)
    assert attempt_gate["passed"] is False
    assert "value_routed_attempt_timeout" in attempt_gate["errors"]

    regressed = {**_a1_artifact(), "live_first_win_rate_value_routed": 0.0}
    floor_gate = mod.evaluate_value_routing_cigate(
        regressed,
        first_win_floor=0.5,
        solve_rate_floor=0.0,
        expected_attempts=2,
    )
    assert floor_gate["passed"] is False
    assert "first_win_rate_floor" in floor_gate["errors"]

    short = _a1_artifact()
    short["value_routed_measurement"] = _measurement(
        [_attempt("aa00~color01", first_win=True, reached_level=1)],
        first_win_rate=1.0,
        solve_rate=0.0,
    )
    short_gate = mod.evaluate_value_routing_cigate(short, expected_attempts=2)
    assert short_gate["passed"] is False
    assert "value_routed_attempt_count" in short_gate["errors"]


def test_scenario_learn_4658_diagnostic_localizes_distribution_shift_or_calibration() -> None:
    """SCENARIO-LEARN-4658-DIAGNOSTIC: residual cause resolves to the dominant probe."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    shift_rows = [
        {"source": "winning_path", "value_score": 1.0},
        {"source": "winning_path", "value_score": 2.0},
        {"source": "off_path_search", "value_score": 9.0},
        {"source": "off_path_search", "value_score": 10.0},
    ]
    no_change_candidates = [
        {"candidate_id": "a", "score": 1.0, "observed_cost": 1.0, "depth": 0, "live": True},
        {"candidate_id": "b", "score": 5.0, "observed_cost": 5.0, "depth": 2, "live": True},
    ]
    shift = mod.run_residual_diagnostic(
        _a1_artifact(),
        score_rows=shift_rows,
        candidate_rows=no_change_candidates,
        distribution_shift_threshold=0.2,
    )

    assert shift["distribution_shift_score"] > 0.2
    assert shift["calibration_changes_routing"] is False
    assert shift["dominant_residual_cause"] == "distribution_shift"

    calibration_candidates = [
        {"candidate_id": "a", "score": 0.0, "observed_cost": 100.0, "depth": 0, "live": True},
        {"candidate_id": "b", "score": 10.0, "observed_cost": 10.0, "depth": 5, "live": True},
    ]
    calibration = mod.run_residual_diagnostic(
        _a1_artifact(),
        score_rows=[
            {"source": "winning_path", "value_score": 1.0},
            {"source": "off_path_search", "value_score": 1.05},
        ],
        candidate_rows=calibration_candidates,
        distribution_shift_threshold=0.2,
    )

    assert calibration["distribution_shift_score"] < 0.2
    assert calibration["calibration_changes_routing"] is True
    assert calibration["dominant_residual_cause"] == "calibration"

    lifted = {
        **_a1_artifact(),
        "first_win_rate_delta": 0.1,
        "residual_cause_hypothesis": "none",
    }
    none = mod.run_residual_diagnostic(
        lifted,
        score_rows=shift_rows,
        candidate_rows=calibration_candidates,
    )
    assert none["dominant_residual_cause"] == "none_a1_lifted"


def test_req_learn_4658_artifact_schema_and_run_paths_are_stable(tmp_path: Path) -> None:
    """REQ-LEARN-4658: artifact writing, checksum, and blocked preconditions are stable."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-LEARN-4658\nSCENARIO-LEARN-4658-CIGATE\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())

    artifact = mod.run(
        root=tmp_path,
        import_checker=lambda: {"agentic_imports": True},
        tests_added={"passed": True, "test_file": __file__, "assertions": 24},
        expected_attempts=2,
        now=lambda: 100.0,
        sleep_fn=lambda _seconds: None,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == (
        "success: value_routing_cigate_plus_diagnostic_shipped_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["cigate_added"]["passed"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    bad = {
        **artifact,
        "honest_verdict": "not terminal",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": True,
        "cigate_added": {"passed": False},
        "dominant_residual_cause": "unknown",
        "field_principles": {},
        "submitted_to_leaderboard": True,
        "reproducibility_checksum": "sha256:bad",
    }
    errors = set(mod.artifact_schema_errors(bad))
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle_false" in errors
    assert "cigate_added" in errors
    assert "dominant_residual_cause" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "reproducibility_checksum" in errors

    missing = tmp_path / "missing"
    blocked = mod.run(
        root=missing,
        import_checker=lambda: {"agentic_imports": True},
        sleep_fn=lambda _seconds: None,
    )
    assert blocked["honest_verdict"] == "blocked_agents_md_read"
    assert blocked["cigate_added"]["passed"] is False
    assert (missing / mod.RESULT_RELATIVE_PATH).exists()


def test_req_learn_4658_defensive_helpers_are_deterministic() -> None:
    """REQ-LEARN-4658: malformed inputs use deterministic conservative defaults."""

    from carnot import experiment_4658_value_routing_cigate_diagnostic as mod

    assert mod._as_float(True, 3.0) == pytest.approx(3.0)
    assert mod._as_float("bad", 4.0) == pytest.approx(4.0)
    assert mod._as_int(False, 5) == 5
    assert mod._as_int("bad", 6) == 6
    assert mod.distribution_shift_probe([])["distribution_shift_score"] == pytest.approx(0.0)
    assert mod.calibration_probe([])["calibration_changes_routing"] is False
    assert mod._floor_duration(started_at=10.0, now=lambda: 10.5, sleep_fn=lambda _: None) == (
        pytest.approx(1.0)
    )
